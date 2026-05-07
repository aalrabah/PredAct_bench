"""
Assistant agent for the agent-to-agent simulator.

The LLM that has the tools and answers the instructor's questions during the
chat phase (Stage 2). Mirrors app.py's chat behavior, but:
  - The prediction tool reads from a pre-calibrated dict (so accuracy is
    exactly the cell's target, not whatever the live k-NN happens to give).
  - All other tools (grades, syllabus, counterfactuals, class stats) come
    straight from tools.py.
  - No Streamlit, no session_state — state lives on the instance.
  - System prompt is imported from prompts.py.
"""

import json

from openai import OpenAI

from prompts import SIM_ASSISTANT_SYSTEM_PROMPT
from tools import (
    detect_grading_system,
    get_student_grades,
    recalculate_grade,
    get_assignment_stats,
    minimum_score_needed,
    list_all_assignments,
    suggest_intervention_for_student,
    get_course_syllabus,
    get_class_average,
    simulate_uniform_remaining,
)


# Same tool schema as the human study (kept here so the sim is self-contained).
TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "list_students",
        "description": "List every student ID in this class. Returns student_ids and total_count.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_student_grades",
        "description": "Look up a student's full grade record (assignment scores, weights, weeks, weighted average).",
        "parameters": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "predict_final_grade_for_student",
        "description": "Predict this student's final letter grade. Returns a confidence score.",
        "parameters": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "predict_all_students",
        "description": "Predictions with confidence for ALL students, sorted by at-risk (D/F) confidence.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "suggest_intervention_for_student",
        "description": "Profile + intervention suggestions for a student.",
        "parameters": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "recalculate_grade",
        "description": "Recalculate weighted average. Supports drop/override/simulate_remaining for mixed counterfactuals.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "drop": {"type": "array", "items": {"type": "string"}},
                "override": {"type": "object", "additionalProperties": {"type": "number"}},
                "simulate_remaining": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Dict {assignment_name: simulated_score}.",
                },
            },
            "required": ["student_id"],
        },
    }},
    {"type": "function", "function": {
        "name": "simulate_uniform_remaining",
        "description": "Counterfactual: student scores `uniform_score` on EVERY remaining assignment. Returns new weighted average + letter grade.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "uniform_score": {"type": "number"},
            },
            "required": ["student_id", "uniform_score"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_assignment_stats",
        "description": "Class-wide stats for an assignment.",
        "parameters": {"type": "object", "properties": {"assignment_name": {"type": "string"}}, "required": ["assignment_name"]},
    }},
    {"type": "function", "function": {
        "name": "list_all_assignments",
        "description": "List every assignment students have attempted so far.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_course_syllabus",
        "description": "Full course syllabus (all weeks), each assignment marked graded or remaining.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_class_average",
        "description": "Overall class average weighted grade across all students.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "minimum_score_needed",
        "description": "Minimum uniform score on remaining assignments to reach a target letter grade (A/B/C/D).",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "target_grade": {"type": "string", "description": "A, B, C, or D"},
            },
            "required": ["student_id", "target_grade"],
        },
    }},
]


def _confidence_pct(conf):
    """Predict tool returns confidence in [0, 1]; the human study reports it as %."""
    if conf is None:
        return None
    if conf <= 1.0:
        return round(conf * 100, 1)
    return round(conf, 1)


def _pred_with_pct_confidence(pred):
    """Return a copy of `pred` with confidence converted to percent (0-100)."""
    if not pred or "error" in pred:
        return pred
    out = dict(pred)
    out["confidence"] = _confidence_pct(out.get("confidence"))
    return out


class AssistantAgent:
    """LLM-with-tools. One instance per episode."""

    def __init__(self, llm_key, llm_configs, course_id, week, feature_set,
                 target_acc, db, students, grades_lookup, calibrated_preds,
                 max_tool_loops=20, temperature=0.1, max_tokens=1400):
        self.llm_key = llm_key
        cfg = llm_configs[llm_key]
        self.model = cfg["model"]
        self.extra_body = cfg.get("extra_body") or {}
        self.client = OpenAI(base_url=cfg["api_base"], api_key=cfg["api_key"])

        self.course_id = course_id
        self.week = week
        self.feature_set = feature_set
        self.target_acc = target_acc
        self.db = db
        self.students = students
        self.grades_lookup = grades_lookup
        self.calibrated_preds = calibrated_preds
        self.grading_system = detect_grading_system(course_id)

        self.max_tool_loops = max_tool_loops
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.history = []  # user/assistant turns; system msg added at call time

    # -------------------------------------------------------------------------
    # System prompt (rendered from prompts.py template)
    # -------------------------------------------------------------------------

    def _system_msg(self):
        accuracy_str = (
            f"{int(self.target_acc * 100)}%" if self.target_acc is not None else "unknown"
        )
        return {
            "role": "system",
            "content": SIM_ASSISTANT_SYSTEM_PROMPT.format(
                course_id=self.course_id,
                week=self.week,
                accuracy_str=accuracy_str,
            ),
        }

    # -------------------------------------------------------------------------
    # Tool execution — predictions come from calibrated_preds; rest from tools.py
    # -------------------------------------------------------------------------

    def execute_tool(self, tool_name, args):
        try:
            if tool_name == "list_students":
                sids = sorted(self.grades_lookup.keys())
                result = {"student_ids": sids, "total_count": len(sids)}
            elif tool_name == "get_student_grades":
                result = get_student_grades(args["student_id"], self.grades_lookup)
            elif tool_name == "predict_final_grade_for_student":
                sid = args["student_id"]
                pred = self.calibrated_preds.get(sid)
                if pred is None:
                    result = {"error": f"Student {sid} not in this class"}
                else:
                    result = _pred_with_pct_confidence(pred)
            elif tool_name == "predict_all_students":
                result = self._predict_all_students()
            elif tool_name == "suggest_intervention_for_student":
                result = suggest_intervention_for_student(args["student_id"], self.grades_lookup)
            elif tool_name == "recalculate_grade":
                full_syl = get_course_syllabus(self.db, self.course_id, current_week=self.week)
                result = recalculate_grade(
                    args["student_id"], self.grades_lookup,
                    drop=args.get("drop"), override=args.get("override"),
                    simulate_remaining=args.get("simulate_remaining"),
                    full_syllabus=full_syl.get("assignments", []),
                )
            elif tool_name == "simulate_uniform_remaining":
                result = simulate_uniform_remaining(
                    self.db, self.course_id, self.week,
                    args["student_id"], self.grades_lookup, args["uniform_score"],
                )
            elif tool_name == "get_assignment_stats":
                result = get_assignment_stats(
                    args["assignment_name"], self.grades_lookup,
                    grading_system=self.grading_system,
                )
            elif tool_name == "list_all_assignments":
                result = list_all_assignments(self.grades_lookup)
            elif tool_name == "get_course_syllabus":
                result = get_course_syllabus(self.db, self.course_id, current_week=self.week)
            elif tool_name == "get_class_average":
                result = get_class_average(self.grades_lookup)
            elif tool_name == "minimum_score_needed":
                full_syl = get_course_syllabus(self.db, self.course_id, current_week=self.week)
                result = minimum_score_needed(
                    args["student_id"], args["target_grade"], self.grades_lookup,
                    full_syllabus=full_syl.get("assignments", []),
                    grading_system=self.grading_system,
                )
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _predict_all_students(self):
        """Build the same shape as predict_all_students_for_scenario, from calibrated_preds."""
        all_preds = []
        for sid, pred in self.calibrated_preds.items():
            grade = (pred.get("predicted_grade") or "").upper()
            all_preds.append({
                "student_id": sid,
                "predicted_grade": grade,
                "confidence": _confidence_pct(pred.get("confidence")),
                "failure_risk": pred.get("failure_risk"),
            })

        def sort_key(s):
            high_risk = s["predicted_grade"] in ("D", "F")
            return (0 if high_risk else 1, -(s["confidence"] or 0))

        all_preds.sort(key=sort_key)
        return {
            "total_students": len(self.calibrated_preds),
            "at_risk_students": [s for s in all_preds if s["predicted_grade"] in ("D", "F")],
            "all_predictions": all_preds,
        }

    # -------------------------------------------------------------------------
    # Chat — one user message in, one assistant reply out, tool loop inside
    # -------------------------------------------------------------------------

    def chat(self, user_message):
        """Append a user message, run the tool-calling loop, return the assistant reply."""
        self.history.append({"role": "user", "content": user_message})
        full_messages = [self._system_msg()] + list(self.history)

        for _ in range(self.max_tool_loops):
            kwargs = dict(
                model=self.model,
                messages=full_messages,
                tools=TOOL_DEFINITIONS,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body

            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                err = f"[LLM request failed: {type(e).__name__}: {e}]"
                self.history.append({"role": "assistant", "content": err})
                return err

            msg = response.choices[0].message

            if not msg.tool_calls:
                text = msg.content or "[No response]"
                self.history.append({"role": "assistant", "content": text})
                return text

            full_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_result = self.execute_tool(tc.function.name, args)
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        # Hit max_tool_loops without a final text response
        msg = "[Tool loop exhausted without final answer]"
        self.history.append({"role": "assistant", "content": msg})
        return msg

    def reset(self):
        self.history = []
