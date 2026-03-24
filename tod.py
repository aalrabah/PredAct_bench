"""
PredAct Benchmark - TOD Orchestrator
Runs the full dialogue loop between Agent 1 (user) and Agent 2 (system).

Belief state is built DETERMINISTICALLY from tools.py output.
The LLM only generates natural language — never JSON or belief state.
"""

import json
import os
import copy
from openai import OpenAI

from config import (
    VLLM_BASE_URL,
    AGENT_USER_MODEL,
    AGENT_SYSTEM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
    MAX_TURNS,
    MIN_TURNS,
    CS_DB_PATH,
    OUTPUT_DATA_PATH,
    LOGS_DIR,
    DEFAULT_INTERVENTION,
    DEFAULT_INTERVENTION_GOAL,
    DEFAULT_CONTACT_MODE,
)
from prompts import (
    AGENT1_SYSTEM_PROMPT,
    AGENT1_FIRST_TURN_TEMPLATE,
    AGENT1_FOLLOWUP_TEMPLATE,
    AGENT2_SYSTEM_PROMPT,
    AGENT2_COURSE_LOOKUP_TEMPLATE,
    AGENT2_SUMMARY_TEMPLATE,
    AGENT2_RISK_TEMPLATE,
    AGENT2_INTERVENTION_TEMPLATE,
    AGENT2_CLOSING_TEMPLATE,
    EMPTY_BELIEF_STATE,
)
from tools import load_db, lookup_course, process_students, extract_scores
from state import StateTracker


# =============================================================================
# DIALOGUE PHASES
# =============================================================================

PHASE_COURSE_LOOKUP = "course_lookup"
PHASE_SUMMARY = "summary"
PHASE_RISK = "risk"
PHASE_INTERVENTION = "intervention"
PHASE_CLOSING = "closing"

PHASE_ORDER = [
    PHASE_COURSE_LOOKUP,
    PHASE_SUMMARY,
    PHASE_RISK,
    PHASE_INTERVENTION,
    PHASE_CLOSING,
]


# =============================================================================
# LLM CLIENT
# =============================================================================

def create_client():
    """Create OpenAI-compatible client pointing to vLLM."""
    return OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed",
    )


def call_llm(client, model, system_prompt, user_message):
    """Call the LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        text = response.choices[0].message.content
        if text is None:
            return "[No response generated]"
        return text.strip()
    except Exception as e:
        print(f"    LLM ERROR: {e}")
        return "[Error generating response]"


# =============================================================================
# BELIEF STATE BUILDERS (Deterministic — no LLM involved)
# =============================================================================

def build_class_context(tool_results, course_id, current_week):
    """Build class_context from tool results. Filled at PHASE_COURSE_LOOKUP."""
    course_info = tool_results.get("course_info", {})

    # Infer department and level from course_id (e.g. "CS 225" → cs, 200)
    parts = course_id.split()
    department = parts[0].lower() if parts else "cs"
    level = ""
    if len(parts) > 1:
        try:
            num = int(parts[1])
            if num < 200:
                level = "100"
            elif num < 300:
                level = "200"
            elif num < 400:
                level = "300"
            elif num < 500:
                level = "400"
            else:
                level = "500_plus"
        except ValueError:
            level = ""

    return {
        "course_name": tool_results.get("course_name", "other"),
        "course_department": department,
        "course_level": level,
        "term": "fall",
        "week": f"week_{current_week}",
    }


def build_class_summary(tool_results):
    """Build class_summary from tool results. Filled at PHASE_SUMMARY."""
    summary = tool_results.get("class_summary", {})
    return {
        "average_gpa": summary.get("average_gpa", ""),
        "grade_trend": summary.get("grade_trend", ""),
        "common_assignment_type_issue": summary.get("common_assignment_type_issue", ""),
        "flagged_student_count": summary.get("flagged_student_count", ""),
        "summary_scope": summary.get("summary_scope", "whole_class"),
    }


def build_student_query(tool_results):
    """Build student_query from tool results. Filled at PHASE_RISK."""
    flagged = [r for r in tool_results.get("student_results", []) if r["failure_risk"] is not None]

    if not flagged:
        return {
            "student_identifier_type": "",
            "predicted_grade_filter": "unknown",
            "assignment_issue_filter": "no_issue",
        }

    # Determine the filter based on what grades are predicted for flagged students
    predicted_grades = set(r["predicted_grade"] for r in flagged)
    if "f" in predicted_grades:
        grade_filter = "below_d"
    elif "d" in predicted_grades:
        grade_filter = "below_c"
    elif "c" in predicted_grades:
        grade_filter = "below_b"
    else:
        grade_filter = "unknown"

    # Determine assignment issue filter
    reasons = [r["failure_risk_reason"] for r in flagged]
    if all(r == "missing_work" for r in reasons):
        issue_filter = "missing_multiple"
    elif all(r == "academic_underpreparedness" for r in reasons):
        issue_filter = "no_issue"
    elif any(r == "missing_work" for r in reasons):
        issue_filter = "missing_multiple"
    else:
        issue_filter = "unknown"

    return {
        "student_identifier_type": "student_id",
        "predicted_grade_filter": grade_filter,
        "assignment_issue_filter": issue_filter,
    }


def build_student_status(tool_results):
    """Build student_status from tool results. Filled at PHASE_SUMMARY."""
    risk_groups = tool_results.get("risk_groups", {})
    student_status = {}

    for risk_key, group in risk_groups.items():
        # For no_risk students, store IDs and count but no per-student dicts
        if risk_key == "no_risk":
            student_status[risk_key] = {
                "student_ids": group.get("student_ids", []),
                "count": group.get("count", 0),
                "predicted_grade": group.get("predicted_grade", ""),
                "failure_risk": None,
            }
            continue

        student_status[risk_key] = {
            "student_ids": group.get("student_ids", []),
            "count": group.get("count", 0),
            "common_grade_trend": _most_common(group.get("grade_trends", {}).values()),
            "common_assignment_type": _most_common_issue(
                tool_results.get("student_results", []),
                group.get("student_ids", []),
            ),
            "predicted_grade": group.get("predicted_grade", ""),
            "failure_risk": group.get("failure_risk"),
            "failure_risk_reasons": group.get("failure_risk_reasons", {}),
            "missing_assignments": group.get("missing_assignments", {}),
        }

    return student_status


def build_intervention(tool_results):
    """Build intervention from tool results. Filled at PHASE_INTERVENTION."""
    flagged_count = tool_results.get("class_summary", {}).get("flagged_student_count", 0)
    intervention_data = tool_results.get("intervention_data") or {}
    atrisk_week = intervention_data.get("atrisk_approx_week")
    current_week = tool_results.get("current_week", 0)

    # Case 1: Threshold not met yet
    if not tool_results.get("should_intervene"):
        result = {
            "no_intervention": {
                "reason": tool_results.get("intervention_reason", "below_threshold"),
                "flagged_student_count": flagged_count,
                "current_week": current_week,
                "atrisk_approx_week": atrisk_week,
            }
        }
        if flagged_count > 0:
            result["no_intervention"]["recommendation"] = "monitor_only"
        return result

    # Case 2: Threshold met but nobody is at risk
    intervention_plan = tool_results.get("intervention_plan", {})
    if not intervention_plan:
        return {
            "no_intervention": {
                "reason": "threshold_met_no_at_risk_students",
                "flagged_student_count": 0,
                "current_week": current_week,
                "atrisk_approx_week": atrisk_week,
            }
        }

    # Case 3: Threshold met and students need intervention
    return intervention_plan


def _most_common(values):
    """Return the most common value from an iterable."""
    from collections import Counter
    vals = list(values)
    if not vals:
        return "unknown"
    counter = Counter(vals)
    return counter.most_common(1)[0][0]


def _most_common_issue(student_results, student_ids):
    """Find the most common assignment type issue for a group of students."""
    from collections import Counter
    issues = []
    for r in student_results:
        if r["student_id"] in student_ids:
            if r["missing_assignments_count"] > 0:
                issues.append("homework")  # default; could be refined
            elif r["failure_risk_reason"] == "academic_underpreparedness":
                issues.append("homework")
    if not issues:
        return "none"
    return Counter(issues).most_common(1)[0][0]


# =============================================================================
# TEXT FORMATTERS (for Agent 2 prompts)
# =============================================================================

def format_risk_groups_text(tool_results):
    """Format risk groups into readable text for Agent 2's prompt."""
    risk_groups = tool_results.get("risk_groups", {})
    lines = []
    for risk_key, group in risk_groups.items():
        sids = group.get("student_ids", [])
        grade = group.get("predicted_grade", "?")
        risk = group.get("failure_risk", "none")
        reasons = group.get("failure_risk_reasons", {})
        missing = group.get("missing_assignments", {})
        lines.append(f"- {risk_key}: {len(sids)} students, predicted grade={grade}, risk={risk}")
        for sid in sids:
            reason = reasons.get(sid, "unknown")
            miss = missing.get(sid, 0)
            lines.append(f"    {sid}: reason={reason}, missing_assignments={miss}")
    return "\n".join(lines) if lines else "No risk groups found."


def format_risk_details_text(tool_results):
    """Format detailed risk info for the risk phase prompt."""
    risk_groups = tool_results.get("risk_groups", {})
    lines = []
    for risk_key, group in risk_groups.items():
        sids = group.get("student_ids", [])
        grade = group.get("predicted_grade", "?")
        risk = group.get("failure_risk", "none")
        reasons = group.get("failure_risk_reasons", {})
        missing = group.get("missing_assignments", {})
        trends = group.get("grade_trends", {})

        lines.append(f"{risk_key} ({len(sids)} students):")
        lines.append(f"  Predicted grade: {grade}")
        lines.append(f"  Failure risk: {risk}")
        lines.append(f"  Per-student breakdown:")
        for sid in sids:
            reason = reasons.get(sid, "unknown")
            miss = missing.get(sid, 0)
            trend = trends.get(sid, "unknown")
            lines.append(f"    - {sid}: reason={reason}, missing={miss}, trend={trend}")

    return "\n".join(lines) if lines else "No flagged students."


def format_intervention_text(tool_results):
    """Format intervention plan into readable text for Agent 2's prompt."""
    plan = tool_results.get("intervention_plan", {})
    flagged_count = tool_results.get("class_summary", {}).get("flagged_student_count", 0)
    should_intervene = tool_results.get("should_intervene", False)
    intervention_data = tool_results.get("intervention_data") or {}
    atrisk_week = intervention_data.get("atrisk_approx_week", "N/A")
    current_week = tool_results.get("current_week", 0)

    if not plan and not should_intervene:
        if flagged_count > 0:
            return (
                f"Intervention NOT YET triggered. Current week ({current_week}) "
                f"is before the at-risk threshold week ({atrisk_week}). "
                f"However, {flagged_count} students are showing risk signals. "
                f"Recommend monitoring closely and reassessing at week {atrisk_week}."
            )
        else:
            return "No intervention needed — all students are on track."

    lines = []
    for risk_key, details in plan.items():
        sids = details.get("student_ids", [])
        types = details.get("intervention_type", {})
        goals = details.get("intervention_goal", {})
        priority = details.get("priority", "unknown")
        contact = details.get("contact_mode", "unknown")
        followup = details.get("follow_up_needed", "unknown")

        lines.append(f"{risk_key} ({len(sids)} students):")
        lines.append(f"  Priority: {priority}")
        lines.append(f"  Contact mode: {contact}")
        lines.append(f"  Follow-up needed: {followup}")
        lines.append(f"  Per-student plan:")
        for sid in sids:
            itype = types.get(sid, "monitor_only")
            igoal = goals.get(sid, "reduce_failure_risk")
            lines.append(f"    - {sid}: type={itype}, goal={igoal}")

    return "\n".join(lines) if lines else "No intervention plan."


# =============================================================================
# BELIEF STATE VALIDATION
# =============================================================================

def validate_belief_state(belief_state, tracker, phase_name):
    """
    Validate belief state against ontology after each phase.
    Logs warnings for invalid values but does not block execution.
    Returns list of issues found.
    """
    issues = []
    results = tracker.validate_state(belief_state)

    for domain, slot, value, is_valid, reason in results:
        if not is_valid:
            msg = f"    VALIDATION WARNING [{phase_name}] {domain}.{slot} = {value!r} → {reason}"
            print(msg)
            issues.append({
                "phase": phase_name,
                "domain": domain,
                "slot": slot,
                "value": value,
                "reason": reason,
            })

    if not issues:
        print(f"    Validation OK [{phase_name}]")

    return issues


# =============================================================================
# DIALOGUE HISTORY FORMATTING
# =============================================================================

def format_dialogue_history(log):
    """Format dialogue log into readable string for prompts."""
    history = []
    for i, turn in enumerate(log):
        role = "Instructor" if i % 2 == 0 else "System"
        history.append(f"{role}: {turn['text']}")
    return "\n".join(history)


# =============================================================================
# LOAD DIALOGUE GOAL FROM LOG FILE
# =============================================================================

def load_goal(grades_file, db):
    """
    Load unseen student data from a grades file.
    Infer course_id, student_count, current_week from the data.
    Returns (goal_dict, unseen_students, course_id, current_week).
    """
    with open(grades_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        course_id = data.get("course_id", "unknown")
        students = data.get("students", [])
    elif isinstance(data, list):
        course_id = data[0].get("course_id", "unknown") if data else "unknown"
        students = data
    else:
        return None, None, None, None

    max_week = 0
    for student in students:
        _, _, week = extract_scores(student)
        if week > max_week:
            max_week = week

    goal = {
        "class_context": {
            "course_name": course_id,
            "course_department": "cs",
            "course_level": "",
            "week": f"week_{max_week}",
        },
        "task": "predict_and_intervene",
        "student_count": len(students),
        "message": [
            f"You are an instructor for {course_id}",
            f"You have {len(students)} students' records through week {max_week}",
            "Request a class summary, risk assessment, and intervention plan",
        ],
    }

    return goal, students, course_id, max_week


# =============================================================================
# RUN A SINGLE DIALOGUE
# =============================================================================

def run_dialogue(client, db, grades_file):
    """
    Run a complete dialogue between Agent 1 and Agent 2.
    Belief state is built deterministically from tools at each phase.
    Returns the full dialogue dict (goal + log).
    """
    # Load goal and unseen data
    goal, unseen_students, course_id, current_week = load_goal(grades_file, db)
    if goal is None:
        print(f"  ERROR: Could not load {grades_file}")
        return None

    student_count = goal["student_count"]
    grades_filename = os.path.basename(grades_file)

    print(f"  Course: {course_id}, Students: {student_count}, Week: week_{current_week}")

    # Run tools pipeline upfront (deterministic)
    tool_results = process_students(db, course_id, unseen_students)
    if "error" in tool_results:
        print(f"  ERROR: {tool_results['error']}")
        return None

    # Pre-build all belief state phases deterministically
    bs_class_context = build_class_context(tool_results, course_id, current_week)
    bs_class_summary = build_class_summary(tool_results)
    bs_student_query = build_student_query(tool_results)
    bs_student_status = build_student_status(tool_results)
    bs_intervention = build_intervention(tool_results)

    # Extract data for prompt formatting
    course_info = tool_results.get("course_info", {})
    intervention_data = tool_results.get("intervention_data") or {}
    atrisk_week = intervention_data.get("atrisk_approx_week", "N/A")

    # Initialize
    log = []
    belief_state = copy.deepcopy(EMPTY_BELIEF_STATE)
    tracker = StateTracker()
    all_validation_issues = []

    # =========================================================================
    # TURN 1: Agent 1 sends data
    # =========================================================================
    agent1_prompt = AGENT1_FIRST_TURN_TEMPLATE.format(
        course_id=course_id,
        course_name=course_id,
        term="fall",
        student_count=student_count,
        week=f"week_{current_week}",
        grades_file=grades_filename,
    )
    user_turn = call_llm(client, AGENT_USER_MODEL, AGENT1_SYSTEM_PROMPT, agent1_prompt)
    log.append({"text": user_turn, "metadata": {}})
    print(f"  [User T1] {user_turn[:100]}...")

    # =========================================================================
    # PHASE 1: Course Lookup
    # =========================================================================
    system_prompt = AGENT2_COURSE_LOOKUP_TEMPLATE.format(
        user_message=user_turn,
        course_id=course_id,
        course_department=bs_class_context["course_department"],
        course_level=bs_class_context["course_level"],
        avg_gpa=course_info.get("avg_gpa", "N/A"),
        student_count=student_count,
        current_week=current_week,
    )
    system_response = call_llm(client, AGENT_SYSTEM_MODEL, AGENT2_SYSTEM_PROMPT, system_prompt)

    # Fill belief state: class_context
    belief_state["class_context"] = copy.deepcopy(bs_class_context)
    issues = validate_belief_state(belief_state, tracker, "course_lookup")
    all_validation_issues.extend(issues)
    log.append({"text": system_response, "metadata": copy.deepcopy(belief_state)})
    print(f"  [Sys  T1] {system_response[:100]}...")

    # =========================================================================
    # PHASE 2: Summary (user asks → system responds)
    # =========================================================================
    user_turn = call_llm(
        client, AGENT_USER_MODEL, AGENT1_SYSTEM_PROMPT,
        AGENT1_FOLLOWUP_TEMPLATE.format(
            course_id=course_id,
            dialogue_history=format_dialogue_history(log),
        ),
    )
    log.append({"text": user_turn, "metadata": {}})
    print(f"  [User T2] {user_turn[:100]}...")

    system_prompt = AGENT2_SUMMARY_TEMPLATE.format(
        course_id=course_id,
        dialogue_history=format_dialogue_history(log),
        avg_gpa=bs_class_summary["average_gpa"],
        grade_trend=bs_class_summary["grade_trend"],
        common_issue=bs_class_summary["common_assignment_type_issue"],
        flagged_count=bs_class_summary["flagged_student_count"],
        total_students=student_count,
        summary_scope=bs_class_summary["summary_scope"],
        risk_groups_text=format_risk_groups_text(tool_results),
    )
    system_response = call_llm(client, AGENT_SYSTEM_MODEL, AGENT2_SYSTEM_PROMPT, system_prompt)

    # Fill belief state: class_summary + student_status
    belief_state["class_summary"] = copy.deepcopy(bs_class_summary)
    belief_state["student_status"] = copy.deepcopy(bs_student_status)
    issues = validate_belief_state(belief_state, tracker, "summary")
    all_validation_issues.extend(issues)
    log.append({"text": system_response, "metadata": copy.deepcopy(belief_state)})
    print(f"  [Sys  T2] {system_response[:100]}...")

    # =========================================================================
    # PHASE 3: Risk Details (user asks → system responds)
    # =========================================================================
    user_turn = call_llm(
        client, AGENT_USER_MODEL, AGENT1_SYSTEM_PROMPT,
        AGENT1_FOLLOWUP_TEMPLATE.format(
            course_id=course_id,
            dialogue_history=format_dialogue_history(log),
        ),
    )
    log.append({"text": user_turn, "metadata": {}})
    print(f"  [User T3] {user_turn[:100]}...")

    system_prompt = AGENT2_RISK_TEMPLATE.format(
        course_id=course_id,
        dialogue_history=format_dialogue_history(log),
        risk_details_text=format_risk_details_text(tool_results),
    )
    system_response = call_llm(client, AGENT_SYSTEM_MODEL, AGENT2_SYSTEM_PROMPT, system_prompt)

    # Fill belief state: student_query
    belief_state["student_query"] = copy.deepcopy(bs_student_query)
    issues = validate_belief_state(belief_state, tracker, "risk")
    all_validation_issues.extend(issues)
    log.append({"text": system_response, "metadata": copy.deepcopy(belief_state)})
    print(f"  [Sys  T3] {system_response[:100]}...")

    # =========================================================================
    # PHASE 4: Intervention (user asks → system responds)
    # =========================================================================
    user_turn = call_llm(
        client, AGENT_USER_MODEL, AGENT1_SYSTEM_PROMPT,
        AGENT1_FOLLOWUP_TEMPLATE.format(
            course_id=course_id,
            dialogue_history=format_dialogue_history(log),
        ),
    )
    log.append({"text": user_turn, "metadata": {}})
    print(f"  [User T4] {user_turn[:100]}...")

    system_prompt = AGENT2_INTERVENTION_TEMPLATE.format(
        course_id=course_id,
        dialogue_history=format_dialogue_history(log),
        should_intervene="Yes" if tool_results.get("should_intervene") else "No",
        intervention_reason=tool_results.get("intervention_reason", "N/A"),
        current_week=current_week,
        atrisk_week=atrisk_week,
        intervention_text=format_intervention_text(tool_results),
    )
    system_response = call_llm(client, AGENT_SYSTEM_MODEL, AGENT2_SYSTEM_PROMPT, system_prompt)

    # Fill belief state: intervention
    belief_state["intervention"] = copy.deepcopy(bs_intervention)
    issues = validate_belief_state(belief_state, tracker, "intervention")
    all_validation_issues.extend(issues)
    log.append({"text": system_response, "metadata": copy.deepcopy(belief_state)})
    print(f"  [Sys  T4] {system_response[:100]}...")

    # =========================================================================
    # PHASE 5: Closing (user thanks → system closes)
    # =========================================================================
    user_turn = call_llm(
        client, AGENT_USER_MODEL, AGENT1_SYSTEM_PROMPT,
        AGENT1_FOLLOWUP_TEMPLATE.format(
            course_id=course_id,
            dialogue_history=format_dialogue_history(log),
        ),
    )
    log.append({"text": user_turn, "metadata": {}})
    print(f"  [User T5] {user_turn[:100]}...")

    system_prompt = AGENT2_CLOSING_TEMPLATE.format(
        course_id=course_id,
        dialogue_history=format_dialogue_history(log),
        user_message=user_turn,
    )
    system_response = call_llm(client, AGENT_SYSTEM_MODEL, AGENT2_SYSTEM_PROMPT, system_prompt)

    # Final belief state stays the same — no new info
    log.append({"text": system_response, "metadata": copy.deepcopy(belief_state)})
    print(f"  [Sys  T5] {system_response[:100]}...")

    # Report validation summary
    if all_validation_issues:
        print(f"  VALIDATION: {len(all_validation_issues)} issues found")
    else:
        print(f"  VALIDATION: All checks passed")

    return {
        "goal": goal,
        "log": log,
        "validation_issues": all_validation_issues,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run N dialogues")
    args = parser.parse_args()

    print("=" * 60)
    print("PredAct Benchmark - TOD Dialogue Generation")
    print("=" * 60)

    # Load database
    print(f"\nLoading database from {CS_DB_PATH}...")
    db = load_db()
    print(f"  Loaded {len(db)} courses")

    # Create LLM client
    client = create_client()

    # Find all grade log files
    log_files = sorted([
        os.path.join(LOGS_DIR, f)
        for f in os.listdir(LOGS_DIR)
        if f.endswith("_grades.json")
    ])
    print(f"  Found {len(log_files)} dialogue log files in {LOGS_DIR}")

    if args.limit:
        log_files = log_files[:args.limit]
        print(f"  Limited to {len(log_files)} dialogues")

    # Run dialogues
    all_dialogues = {}
    total_validation_issues = 0
    for i, grades_file in enumerate(log_files):
        dlg_id = os.path.basename(grades_file).replace("_grades.json", ".json")
        print(f"\n[{i+1}/{len(log_files)}] Running dialogue {dlg_id}...")

        try:
            result = run_dialogue(client, db, grades_file)
            if result:
                total_validation_issues += len(result.get("validation_issues", []))
                # Store dialogue without validation_issues (keep data.json clean)
                all_dialogues[dlg_id] = {
                    "goal": result["goal"],
                    "log": result["log"],
                }
                print(f"  Completed: {len(result['log'])} turns")
            else:
                print(f"  SKIPPED due to errors")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # Write output
    print(f"\nWriting {len(all_dialogues)} dialogues to {OUTPUT_DATA_PATH}...")
    with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(all_dialogues)} dialogues generated.")
    if total_validation_issues > 0:
        print(f"WARNING: {total_validation_issues} total validation issues across all dialogues")
    else:
        print(f"All belief states passed ontology validation.")


if __name__ == "__main__":
    main()