"""
Episode runner for the agent-to-agent simulator.

Runs one full episode (one scenario, one cell) end-to-end:

  Stage 0  Setup                    — calibrate predictions, identify flagged
  Stage 1  Initial decision          — InstructorAgent.initial_decision()
  Stage 2  Chat phase (max N turns)  — InstructorAgent ↔ AssistantAgent
  Stage 3  Final decision            — InstructorAgent.final_decision()
  Out      Log in human-study schema — feedable to evaluate_human_study.py
"""

import json
import os
import random
import time

from tools import (
    build_grades_lookup,
    detect_grading_system,
    get_course_syllabus,
    load_db,
    GRADE_TARGETS_BY_SYSTEM,
)
from sim.accuracy_injector import get_calibrated_predictions
from sim.assistant_agent import AssistantAgent, _confidence_pct
from sim.instructor_agent import InstructorAgent


# -----------------------------------------------------------------------------
# Stratified sampling — mirrors app.py:283-298 exactly
# -----------------------------------------------------------------------------

def stratified_sample(all_students, gt_grades, sample_size, n_at_risk, seed):
    """
    Pick `sample_size` students with exactly `n_at_risk` truly at-risk (D/F)
    and the rest not-at-risk (A/B/C). Deterministic given `seed`.

    Same logic as app.py's get_students_and_lookup so sim and human study
    sample comparably.

    Args:
        all_students: list of student records loaded from a test set
        gt_grades: dict {sid: letter} ground truth for this course
        sample_size: total students wanted
        n_at_risk: how many of those should be truly at-risk
        seed: int for deterministic sampling

    Returns:
        list of student records (length sample_size). Final order shuffled.
    """
    at_risk_pool = [s for s in all_students
                    if (gt_grades.get(s["student_id"], "") or "").lower() in ("d", "f")]
    not_at_risk_pool = [s for s in all_students
                        if (gt_grades.get(s["student_id"], "") or "").lower() in ("a", "b", "c")]

    rng = random.Random(seed)

    at_risk_sorted = sorted(at_risk_pool, key=lambda s: s["student_id"])
    rng.shuffle(at_risk_sorted)
    sampled_at_risk = at_risk_sorted[:n_at_risk]

    n_not_at_risk = sample_size - n_at_risk
    not_at_risk_sorted = sorted(not_at_risk_pool, key=lambda s: s["student_id"])
    rng.shuffle(not_at_risk_sorted)
    sampled_not_at_risk = not_at_risk_sorted[:n_not_at_risk]

    sampled = sampled_at_risk + sampled_not_at_risk
    rng.shuffle(sampled)
    return sampled


# -----------------------------------------------------------------------------
# Scenario helpers (copied here so the sim never imports app.py / streamlit)
# -----------------------------------------------------------------------------

def _compute_primary_driver(student_id, grades_lookup, course_id=None):
    """
    Pick the single most important reason a student looks at-risk.
    `course_id` controls UK/US thresholds for the "Low on..." vs neutral label.
    """
    if student_id not in grades_lookup:
        return "unknown"
    scores = grades_lookup[student_id]

    missing = [(name, info) for name, info in scores.items()
               if info.get("score") is None and info.get("weight", 0) > 0]
    if missing:
        missing.sort(key=lambda x: -x[1]["weight"])
        worst_name, worst_info = missing[0]
        return f"Missing {worst_name} (weight {worst_info['weight']:.0%})"

    damaged = []
    for name, info in scores.items():
        score = info.get("score")
        weight = info.get("weight", 0)
        if score is None or weight <= 0:
            continue
        damage = weight * (100 - score) / 100
        damaged.append((name, score, weight, damage))

    if damaged:
        damaged.sort(key=lambda x: -x[3])
        worst_name, worst_score, worst_weight, _ = damaged[0]
        # "Low on" is only honest when the score is actually below the C threshold
        # for this course's grading system. Otherwise label it neutrally so the LLM
        # doesn't assume a B-grade student is failing.
        targets = GRADE_TARGETS_BY_SYSTEM[detect_grading_system(course_id)]
        prefix = "Low on" if worst_score < targets["c"] else "Worst weighted-loss on"
        return f"{prefix} {worst_name} ({worst_weight:.0%} weight, score {worst_score:.0f})"

    return "insufficient data"


def _build_syllabus_table(db, course_id):
    """Markdown table the instructor sees. Mirrors app.py's render_syllabus_inline."""
    syl_result = get_course_syllabus(db, course_id)
    syllabus = syl_result.get("assignments", [])
    if not syllabus:
        return "(no syllabus available)"
    rows = [
        f"| {a['name']} | W{a['week']} | {a['type']} | {a['weight']:.2%} |"
        for a in syllabus
    ]
    return "| Assignment | Week | Type | Weight |\n|---|---|---|---|\n" + "\n".join(rows)


def _build_flagged_students_for_instructor(calibrated_preds, grades_lookup, course_id=None):
    """
    Identify D/F-predicted students and build the per-student summary the
    instructor sees on the initial-decision screen.
    """
    flagged = []
    for sid, pred in calibrated_preds.items():
        grade = (pred.get("predicted_grade") or "").upper()
        if grade not in ("D", "F"):
            continue
        flagged.append({
            "student_id": sid,
            "predicted_grade": grade,
            "confidence": _confidence_pct(pred.get("confidence")),
            "primary_driver": _compute_primary_driver(sid, grades_lookup, course_id=course_id),
        })
    # Stable order: by descending confidence
    flagged.sort(key=lambda s: -(s["confidence"] or 0))
    return flagged


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_episode(
    *,
    assistant_llm_key,
    instructor_llm_key,
    llm_configs,
    db,
    students,
    grades_lookup,
    gt_grades,
    course_id,
    course_file,        # for the log only
    week,
    feature_set,
    target_acc,
    seed,
    max_chat_turns=10,
    condition_id=None,  # for the log only; defaults to "{assistant_llm_key}_{int(target_acc*100)}"
):
    """
    Run one episode and return a log dict in the human-study schema.

    The log can be appended to a participant-style JSON file and read by
    evaluate_human_study.py without any code changes.
    """
    if condition_id is None:
        condition_id = f"{assistant_llm_key}_{int(round(target_acc*100))}"

    t_start = time.time()

    # ---- Stage 0: setup ------------------------------------------------------
    calibrated_preds, calibration_stats = get_calibrated_predictions(
        db, students, course_id, week, feature_set,
        gt_grades, target_acc=target_acc, seed=seed,
    )
    flagged_students = _build_flagged_students_for_instructor(
        calibrated_preds, grades_lookup, course_id=course_id,
    )
    flagged_sids = [s["student_id"] for s in flagged_students]

    syllabus_table = _build_syllabus_table(db, course_id)

    instructor = InstructorAgent(
        llm_key=instructor_llm_key,
        llm_configs=llm_configs,
        course_id=course_id,
        week=week,
        target_acc=target_acc,
        syllabus_table=syllabus_table,
        flagged_students=flagged_students,
    )
    assistant = AssistantAgent(
        llm_key=assistant_llm_key,
        llm_configs=llm_configs,
        course_id=course_id,
        week=week,
        feature_set=feature_set,
        target_acc=target_acc,
        db=db,
        students=students,
        grades_lookup=grades_lookup,
        calibrated_preds=calibrated_preds,
    )

    # If no students were flagged, skip stages 1-3 (mirrors human study UI behavior).
    if not flagged_sids:
        initial_decisions = {}
        final_decisions = {}
        dialogue_history = []
        initial_raw = ""
        final_raw = ""
    else:
        # ---- Stage 1: initial decision --------------------------------------
        initial_decisions, initial_raw = instructor.initial_decision()

        # ---- Stage 2: chat phase --------------------------------------------
        dialogue_history = []
        for turn_idx in range(max_chat_turns):
            action, message = instructor.chat_turn(
                initial_decisions=initial_decisions,
                dialogue_history=dialogue_history,
                turns_used=turn_idx,
                max_turns=max_chat_turns,
            )
            if action == "done":
                break
            dialogue_history.append({"role": "instructor", "content": message})
            reply = assistant.chat(message)
            dialogue_history.append({"role": "assistant", "content": reply})

        # ---- Stage 3: final decision ----------------------------------------
        final_decisions, final_raw = instructor.final_decision(
            initial_decisions=initial_decisions,
            dialogue_history=dialogue_history,
        )

    # ---- Build per-student record (human-study schema) -----------------------
    per_student = {}
    for s in students:
        sid = s["student_id"]
        truth = (gt_grades.get(sid, "") or "").lower()
        is_at_risk = truth in ("d", "f")
        agent_flagged = sid in flagged_sids
        initial_dec = initial_decisions.get(sid)
        final_dec = final_decisions.get(sid)
        per_student[sid] = {
            "is_at_risk": is_at_risk,
            "agent_flagged": agent_flagged,
            "initial_decision": (
                "accept" if initial_dec == "flag"
                else ("reject" if initial_dec == "no_flag" else None)
            ),
            "final_decision": (
                "accept" if final_dec == "flag"
                else ("reject" if final_dec == "no_flag" else None)
            ),
        }

    duration = round(time.time() - t_start)

    return {
        "condition_id": condition_id,
        "course_id": course_id,
        "course_file": course_file,
        "week": week,
        "feature_set": feature_set,
        "has_agent": True,
        "target_accuracy": target_acc,
        "assistant_llm": assistant_llm_key,
        "instructor_llm": instructor_llm_key,
        "seed": seed,
        "duration_seconds": duration,
        "n_students_in_scenario": len(students),
        "agent_flagged_sids": flagged_sids,
        "calibration_stats": calibration_stats,  # auditability: target vs achieved accuracy
        "initial_decisions_raw": initial_raw,
        "final_decisions_raw": final_raw,
        "dialogue_history": dialogue_history,
        "per_student": per_student,
    }


# -----------------------------------------------------------------------------
# Convenience loader (mirrors app.py's data setup)
# -----------------------------------------------------------------------------

def load_scenario_data(course_file, project_root=None):
    """
    Load db, students, grades_lookup, gt_grades for one course file.

    Returns: (db, students, grades_lookup, gt_grades).
    """
    project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_set_path = os.path.join(project_root, "results", "predact_cs", "test_sets", course_file)
    gt_path = os.path.join(project_root, "results", "predact_cs", "ground_truth_for_cutoff_data.json")

    db = load_db()
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    students = test_set["students"]
    grades_lookup = build_grades_lookup(students)
    gt_grades = gt.get(course_file, {}).get("student_grades", {})
    return db, students, grades_lookup, gt_grades
