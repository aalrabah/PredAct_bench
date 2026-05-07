"""
Instructor agent for the agent-to-agent simulator.

Plays the role of the human instructor in the Streamlit study. Three call
sites mirror the three decision stages:

  1. initial_decision()   — Stage 1: pick flag/no-flag without chat
  2. chat_turn()          — Stage 2: ask one question to the assistant (or stop)
  3. final_decision()     — Stage 3: pick flag/no-flag with chat history

All three use OpenAI's `client.beta.chat.completions.parse()` with Pydantic
schemas, so the model output is guaranteed to match the structure we expect.
For providers that don't support strict structured outputs (some Qwen
endpoints), we fall back to plain JSON-mode and validate manually.

Per-LLM temperatures are looked up from INSTRUCTOR_TEMPS (set by user spec).
"""

from openai import OpenAI

from prompts import (
    SIM_INSTRUCTOR_SYSTEM_PROMPT,
    SIM_INSTRUCTOR_INITIAL_DECISION_TEMPLATE,
    SIM_INSTRUCTOR_CHAT_TURN_TEMPLATE,
    SIM_INSTRUCTOR_FINAL_DECISION_TEMPLATE,
)
from sim.schemas import FlagDecisions, ChatTurn
from tools import detect_grading_system


# Per-grading-system phrasing used inside the instructor prompts.
# - at_risk_definition: how the system prompt describes "what counts as D/F"
# - c_threshold:        the score that means "safe to rescue" in chat/final templates
# - d_threshold:        the score the LLM should NOT ask about (still inside D zone)
GRADING_PROMPT_FILLERS = {
    "us": {
        "at_risk_definition": "D is between 60-69. F is below 60.",
        "c_threshold": 70,
        "d_threshold": 60,
    },
    "uk": {
        "at_risk_definition": "D is between 40-49. F is below 40.",
        "c_threshold": 50,
        "d_threshold": 40,
    },
}


# Per-LLM temperatures (user spec). Unknown keys raise — no silent fallback.
INSTRUCTOR_TEMPS = {
    # Closed-source
    "gpt4o_mini":      0.7,
    "gpt5_4_mini":     0.7,
    "gpt5_5":          0.7,
    "claude_opus_4_7": 0.7,
    "claude_haiku_4_5": 0.7,
    # Open-source (placeholders for when enabled)
    "qwen_9b":            0.6,
    "qwen_35b":           0.6,
    "mistral_small_24b":  0.6,
    "ministral_3_14b":    0.6,
    "deepseek_v4_flash":  0.6,
    "deepseek_v4_pro":    0.6,
    "gemini_3_1_pro":     0.7,
    "gemini_3_flash":     0.7,
}


# -----------------------------------------------------------------------------
# Block formatters (same as before — used to render what the instructor sees)
# -----------------------------------------------------------------------------

def format_flagged_students_block(flagged):
    """Render the bulleted list the instructor sees for each flagged student.
    `flagged` is an iterable of {student_id, predicted_grade, confidence, primary_driver}."""
    lines = []
    for s in flagged:
        sid = s["student_id"]
        pred = s.get("predicted_grade", "?")
        conf = s.get("confidence")
        conf_str = f"{conf:.0f}%" if isinstance(conf, (int, float)) else "?"
        driver = s.get("primary_driver", "")
        lines.append(f"- {sid} | {pred} | {conf_str} | {driver}")
    return "\n".join(lines) if lines else "(none)"


def format_initial_decisions_block(decisions):
    """Render `{sid: 'flag'|'no_flag'}` as a bulleted list."""
    if not decisions:
        return "(none)"
    return "\n".join(f"- {sid}: {dec}" for sid, dec in decisions.items())


def format_dialogue_history(history):
    """Render the chat phase as a transcript.
    `history` is a list of {"role": "instructor"|"assistant", "content": str}."""
    if not history:
        return "(no messages yet)"
    lines = []
    for msg in history:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


# -----------------------------------------------------------------------------
# Decision dict helpers
# -----------------------------------------------------------------------------

def _decisions_to_dict(flag_decisions, expected_sids):
    """
    Convert a FlagDecisions Pydantic object to {sid: "flag"|"no_flag"},
    backfilling any missing student with "no_flag" so callers always get
    a complete dict. Unexpected sids are dropped.
    """
    out = {sid: "no_flag" for sid in expected_sids}
    if not flag_decisions:
        return out
    expected = set(expected_sids)
    for entry in flag_decisions.decisions:
        if entry.student_id in expected:
            out[entry.student_id] = entry.decision
    return out


# -----------------------------------------------------------------------------
# InstructorAgent
# -----------------------------------------------------------------------------

class InstructorAgent:
    """LLM that plays the human instructor. One instance per episode."""

    def __init__(self, llm_key, llm_configs, course_id, week, target_acc,
                 syllabus_table, flagged_students,
                 temperature=None, max_tokens=16000):
        self.llm_key = llm_key
        cfg = llm_configs[llm_key]
        self.model = cfg["model"]
        self.extra_body = cfg.get("extra_body") or {}
        # Some models (e.g. OpenAI's reasoning series) reject `temperature`.
        # Honor a per-config flag; default True (most models accept it).
        self.supports_temperature = cfg.get("supports_temperature", True)
        # Some OpenAI reasoning models reject `max_tokens` and require
        # `max_completion_tokens` instead. Default False (most models use max_tokens).
        self.uses_max_completion_tokens = cfg.get("uses_max_completion_tokens", False)
        self.client = OpenAI(base_url=cfg["api_base"], api_key=cfg["api_key"])

        self.course_id = course_id
        self.week = week
        self.target_acc = target_acc
        self.accuracy_pct = int(round(target_acc * 100)) if target_acc is not None else 0

        self.syllabus_table = syllabus_table
        self.flagged_students = list(flagged_students)
        self.flagged_sids = [s["student_id"] for s in self.flagged_students]
        self.flagged_block = format_flagged_students_block(self.flagged_students)

        # OULAD courses use UK thresholds (D=40-49, F<40); PredAct-CS uses US (D=60-69, F<60).
        self.grading_system = detect_grading_system(course_id)
        self.grading_fillers = GRADING_PROMPT_FILLERS[self.grading_system]

        # Temperature: explicit > per-LLM lookup. Unknown llm_key raises KeyError.
        if temperature is None:
            if llm_key not in INSTRUCTOR_TEMPS:
                raise KeyError(
                    f"No instructor temperature defined for llm_key={llm_key!r}. "
                    f"Add it to INSTRUCTOR_TEMPS or pass temperature= explicitly."
                )
            temperature = INSTRUCTOR_TEMPS[llm_key]
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -------------------------------------------------------------------------
    # System message (shared across all three stages)
    # -------------------------------------------------------------------------

    def _system_msg(self):
        return {
            "role": "system",
            "content": SIM_INSTRUCTOR_SYSTEM_PROMPT.format(
                course_id=self.course_id,
                week=self.week,
                accuracy_pct=self.accuracy_pct,
                at_risk_definition=self.grading_fillers["at_risk_definition"],
            ),
        }

    # -------------------------------------------------------------------------
    # Core: structured-output call with fallback
    # -------------------------------------------------------------------------

    def _parse_with_fallback(self, user_content, schema):
        """
        Call the LLM and return a parsed instance of `schema`.

        Try strict structured outputs first via client.beta.chat.completions.parse.
        If that fails (e.g. provider doesn't support strict JSON Schema), fall
        back to plain chat.completions.create with JSON-mode and manual
        validation. Returns (parsed_instance_or_None, raw_text).

        Retries transient upstream errors (429 RateLimitError, 5xx) with
        exponential backoff up to RETRY_MAX_ATTEMPTS times before giving up.
        """
        import time as _time
        RETRY_MAX_ATTEMPTS = 8
        RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt

        messages = [self._system_msg(), {"role": "user", "content": user_content}]
        kwargs = dict(
            model=self.model,
            messages=messages,
        )
        if self.uses_max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        if self.supports_temperature:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        def _is_transient(exc):
            """429 rate limits, 5xx server errors, network blips, empty responses → retry."""
            name = type(exc).__name__
            msg = str(exc)
            if "RateLimitError" in name or "rate-limited" in msg.lower():
                return True
            if "APIConnectionError" in name or "Timeout" in name:
                return True
            if " 429 " in msg or " 500 " in msg or " 502 " in msg or " 503 " in msg or " 504 " in msg:
                return True
            # Provider returned an empty/malformed response (no choices, None content)
            # — treat as transient and retry on a fresh call.
            if "NoneType" in msg and ("not iterable" in msg or "not subscriptable" in msg):
                return True
            return False

        # ---- Path A: strict structured outputs (preferred), with retries ---
        parse_err = "parse() returned no parsed object"
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                completion = self.client.beta.chat.completions.parse(
                    response_format=schema,
                    **kwargs,
                )
                msg = completion.choices[0].message
                parsed = msg.parsed
                raw_text = msg.content or ""
                if parsed is not None:
                    return parsed, raw_text
                break  # parse returned None — not transient, fall through to Path B
            except Exception as e_parse:
                parse_err = f"{type(e_parse).__name__}: {e_parse}"
                if _is_transient(e_parse) and attempt < RETRY_MAX_ATTEMPTS - 1:
                    _time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                break

        # ---- Path B: fallback — JSON-mode + manual validation --------------
        # OpenAI's JSON mode requires the word "json" in the prompt. Our
        # Pydantic-driven prompts don't mention it, so we inject a one-line
        # nudge into a fresh copy of the messages.
        e_json = None
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                json_messages = list(messages)
                json_messages.append({
                    "role": "system",
                    "content": "Reply with a single valid JSON object that matches the requested schema.",
                })
                json_kwargs = dict(kwargs)
                json_kwargs["messages"] = json_messages
                response = self.client.chat.completions.create(
                    response_format={"type": "json_object"},
                    **json_kwargs,
                )
                text = response.choices[0].message.content or ""
                parsed = schema.model_validate_json(text)
                return parsed, text
            except Exception as e:
                e_json = e
                if _is_transient(e) and attempt < RETRY_MAX_ATTEMPTS - 1:
                    _time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                break

        # ---- Path C: free-form text + regex extraction, with retries -------
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                import re
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    parsed = schema.model_validate_json(m.group(0))
                    return parsed, text
                break  # got a response but no JSON blob — not transient
            except Exception as e:
                if _is_transient(e) and attempt < RETRY_MAX_ATTEMPTS - 1:
                    _time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                break

        raw = f"[parse failed: {parse_err}; json-mode failed: {type(e_json).__name__}: {e_json}]"
        return None, raw

    # -------------------------------------------------------------------------
    # Stage 1 — initial decision (no chat yet)
    # -------------------------------------------------------------------------

    def initial_decision(self):
        """Returns ({sid: "flag"|"no_flag"}, raw_text)."""
        user_content = SIM_INSTRUCTOR_INITIAL_DECISION_TEMPLATE.format(
            course_id=self.course_id,
            accuracy_pct=self.accuracy_pct,
            syllabus_table=self.syllabus_table,
            flagged_students_block=self.flagged_block,
        )
        parsed, raw = self._parse_with_fallback(user_content, FlagDecisions)
        decisions = _decisions_to_dict(parsed, self.flagged_sids)
        return decisions, raw

    # -------------------------------------------------------------------------
    # Stage 2 — chat turn (one question OR signal terminate)
    # -------------------------------------------------------------------------

    def chat_turn(self, initial_decisions, dialogue_history, turns_used, max_turns):
        """
        Returns (action, message).
        - action="ask" → message is the question to send to the assistant
        - action="done" → message is empty; chat phase ends
        """
        # Use the highest-confidence flagged student's ID as the in-prompt example,
        # so OULAD episodes get OULAD-format ("2562034") and PredAct-CS gets PredAct-CS-format ("syn_xxx").
        example_sid = self.flagged_sids[0] if self.flagged_sids else "<student_id>"
        user_content = SIM_INSTRUCTOR_CHAT_TURN_TEMPLATE.format(
            accuracy_pct=self.accuracy_pct,
            syllabus_table=self.syllabus_table,
            flagged_students_block=self.flagged_block,
            initial_decisions_block=format_initial_decisions_block(initial_decisions),
            turns_used=turns_used,
            max_turns=max_turns,
            dialogue_history=format_dialogue_history(dialogue_history),
            c_threshold=self.grading_fillers["c_threshold"],
            d_threshold=self.grading_fillers["d_threshold"],
            example_sid=example_sid,
        )
        parsed, _raw = self._parse_with_fallback(user_content, ChatTurn)
        if parsed is None:
            # Parse failed — stop the chat to avoid a stuck loop.
            return "done", ""
        if parsed.should_terminate or not parsed.next_question.strip():
            return "done", ""
        return "ask", parsed.next_question.strip()

    # -------------------------------------------------------------------------
    # Stage 3 — final decision (after chat)
    # -------------------------------------------------------------------------

    def final_decision(self, initial_decisions, dialogue_history):
        """Returns ({sid: "flag"|"no_flag"}, raw_text)."""
        user_content = SIM_INSTRUCTOR_FINAL_DECISION_TEMPLATE.format(
            accuracy_pct=self.accuracy_pct,
            flagged_students_block=self.flagged_block,
            initial_decisions_block=format_initial_decisions_block(initial_decisions),
            dialogue_history=format_dialogue_history(dialogue_history),
            c_threshold=self.grading_fillers["c_threshold"],
        )
        parsed, raw = self._parse_with_fallback(user_content, FlagDecisions)
        decisions = _decisions_to_dict(parsed, self.flagged_sids)
        return decisions, raw
