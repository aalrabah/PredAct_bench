"""
PredAct Benchmark - Evaluation
Measures prediction accuracy, dialogue state tracking, and intervention quality.

Key fixes:
- Extracts predicted grades from ALL risk groups including no_risk
- Intervention trigger: "no_intervention" key means NOT triggered
- Joint goal accuracy: checks per-turn belief state correctness
- Intervention F1 computed for all cases where at least one side triggered
- Unknown predictions penalized in distance metric
- Slot F1: per-slot precision/recall/F1 across all belief state slots
- Temporal breakdown: results sliced by early/mid/late semester
"""

import json
import argparse
from collections import defaultdict

from config import (
    OUTPUT_DATA_PATH,
    EVAL_GPA_TOLERANCE,
    EVAL_COUNT_TOLERANCE,
    EVAL_MAX_GRADE_DISTANCE,
    RISK_MAPPING,
    GRADE_TO_GPA,
)
from state import load_ontology, parse_ontology


# =============================================================================
# TEMPORAL BREAKDOWN BINS
# =============================================================================

def get_temporal_bin(cutoff_week):
    """
    Assign a cutoff week to early/mid/late bin.
    Early: weeks 1-5, Mid: weeks 6-8, Late: weeks 9+
    """
    if cutoff_week <= 5:
        return "early"
    elif cutoff_week <= 8:
        return "mid"
    else:
        return "late"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_dialogues(path=None):
    """Load generated dialogues from data.json."""
    path = path or OUTPUT_DATA_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(path):
    """
    Load ground truth file.
    Expected format from split_data.py:
    {
        "DLG_0001.json": {
            "course_id": "Course_01",
            "cutoff_week": 5,
            "student_grades": {"syn_001": "A", "syn_002": "D", ...},
            "full_student_records": {...},
            "intervention_triggered": true/false
        }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# EXTRACT PREDICTIONS FROM BELIEF STATE
# =============================================================================

def extract_predictions_from_state(final_state):
    predictions = {}
    student_status = final_state.get("student_status", {})

    for risk_key, group in student_status.items():
        if not isinstance(group, dict):
            continue

        per_student = group.get("per_student_grades", {})
        if per_student:
            for sid, grade in per_student.items():
                predictions[sid] = grade.lower()
        else:
            predicted_grade = group.get("predicted_grade", "unknown")
            for sid in group.get("student_ids", []):
                predictions[sid] = predicted_grade.lower()

    return predictions


def extract_intervention_triggered(final_state):
    """
    Determine if the system triggered intervention.
    "no_intervention" key means NOT triggered.
    Any other key (like "high_risk", "medium_risk") means triggered.
    """
    intervention = final_state.get("intervention", {})

    if not intervention:
        return False

    # If the only key is "no_intervention", intervention was NOT triggered
    keys = set(intervention.keys())
    if keys == {"no_intervention"}:
        return False

    # Any other key means intervention was triggered
    return True


# =============================================================================
# 1. GRADE PREDICTION ACCURACY
# =============================================================================

def evaluate_predictions(final_state, ground_truth_grades):
    """
    Compare predicted grades against actual final_grade ground truth.
    ground_truth_grades: {"student_id": "actual_grade", ...}
    """
    predicted_grades = extract_predictions_from_state(final_state)

    results = []
    for sid, actual in ground_truth_grades.items():
        pred = predicted_grades.get(sid, "unknown")
        actual_lower = actual.lower()
        results.append({
            "student_id": sid,
            "predicted": pred,
            "actual": actual_lower,
            "correct": pred == actual_lower,
        })

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_student": results,
    }


# =============================================================================
# 2. RISK MAPPING CONSISTENCY
# =============================================================================

def evaluate_risk_mapping(final_state):
    """
    Check if the risk level assigned to each group is consistent
    with the predicted-grade-to-risk mapping from config.
    """
    results = []
    student_status = final_state.get("student_status", {})

    for risk_key, group in student_status.items():
        if not isinstance(group, dict):
            continue

        assigned_risk = group.get("failure_risk")
        predicted_grade = group.get("predicted_grade", "")
        expected_risk = RISK_MAPPING.get(predicted_grade.lower(), None)

        student_ids = group.get("student_ids", [])
        for sid in student_ids:
            results.append({
                "student_id": sid,
                "predicted_grade": predicted_grade,
                "assigned_risk": assigned_risk,
                "expected_risk": expected_risk,
                "consistent": assigned_risk == expected_risk,
            })

    total = len(results)
    consistent = sum(1 for r in results if r["consistent"])
    accuracy = consistent / total if total > 0 else 0.0

    return {
        "consistency": round(accuracy, 4),
        "consistent_count": consistent,
        "total": total,
    }


# =============================================================================
# 3. INTERVENTION EVALUATION
# =============================================================================

def evaluate_intervention(final_state, gt_entry):
    """
    Evaluate intervention decisions:
    - Did the system correctly decide to intervene (or not)?
    - Compute targeting F1 for all cases where at least one side triggered,
      not just when both agree — avoids inflating F1 by skipping hard cases.
    """
    pred_triggered = extract_intervention_triggered(final_state)
    gold_triggered = gt_entry.get("intervention_triggered", False)

    trigger_correct = pred_triggered == gold_triggered

    # Compute targeting F1 whenever at least one side triggered
    targeting = {}
    if pred_triggered or gold_triggered:
        intervention = final_state.get("intervention", {})
        gold_grades = gt_entry.get("student_grades", {})

        # Gold flagged students: those whose actual grade maps to a risk level
        gold_flagged = set()
        for sid, grade in gold_grades.items():
            risk = RISK_MAPPING.get(grade.lower(), None)
            if risk is not None:
                gold_flagged.add(sid)

        # Predicted flagged students: from intervention plan
        pred_flagged = set()
        for risk_key, details in intervention.items():
            if risk_key == "no_intervention":
                continue
            if isinstance(details, dict):
                pred_flagged.update(details.get("student_ids", []))

        # Compute precision, recall, F1
        if pred_flagged and gold_flagged:
            precision = len(pred_flagged & gold_flagged) / len(pred_flagged)
            recall = len(pred_flagged & gold_flagged) / len(gold_flagged)
        elif not pred_flagged and not gold_flagged:
            precision = 1.0
            recall = 1.0
        elif not pred_flagged:
            precision = 0.0
            recall = 0.0
        else:
            precision = 0.0
            recall = 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        targeting = {
            "predicted_flagged": sorted(list(pred_flagged)),
            "gold_flagged": sorted(list(gold_flagged)),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "trigger_correct": trigger_correct,
        "predicted_triggered": pred_triggered,
        "gold_triggered": gold_triggered,
        "targeting": targeting,
    }


# =============================================================================
# 4. NUMERIC SLOT EVALUATION
# =============================================================================

def evaluate_numeric_slots(final_state, gt_entry):
    """
    Evaluate open_numeric slots.
    Since ground_truth doesn't have a gold belief state, we compute
    expected values from the actual student grades.

    NOTE: average_gpa MAE measures how far the system's predicted GPA
    (based on nearest-neighbor grade predictions) is from the true GPA
    (based on actual final grades). This captures prediction error, not
    just slot-filling error.
    """
    results = {}

    # Compute expected GPA from actual grades
    gold_grades = gt_entry.get("student_grades", {})
    if gold_grades:
        gold_gpas = [GRADE_TO_GPA.get(g.lower(), 0.0) for g in gold_grades.values()]
        gold_avg_gpa = round(sum(gold_gpas) / len(gold_gpas), 2)

        pred_avg_gpa = final_state.get("class_summary", {}).get("average_gpa", "")
        if pred_avg_gpa != "":
            try:
                pred_val = float(pred_avg_gpa)
                error = abs(pred_val - gold_avg_gpa)
                results["average_gpa"] = {
                    "predicted": pred_val,
                    "gold": gold_avg_gpa,
                    "absolute_error": round(error, 4),
                    "within_tolerance": error <= EVAL_GPA_TOLERANCE,
                }
            except (ValueError, TypeError):
                results["average_gpa"] = {"error": "parse_failure"}

    # Compute expected flagged count from actual grades
    if gold_grades:
        gold_flagged = sum(1 for g in gold_grades.values() if RISK_MAPPING.get(g.lower()) is not None)
        pred_flagged = final_state.get("class_summary", {}).get("flagged_student_count", "")
        if pred_flagged != "":
            try:
                pred_val = int(pred_flagged)
                error = abs(pred_val - gold_flagged)
                results["flagged_student_count"] = {
                    "predicted": pred_val,
                    "gold": gold_flagged,
                    "absolute_error": error,
                    "within_tolerance": error <= EVAL_COUNT_TOLERANCE,
                }
            except (ValueError, TypeError):
                results["flagged_student_count"] = {"error": "parse_failure"}

    return results


# =============================================================================
# 5. PREDICTION ANALYSIS (beyond exact match)
# =============================================================================

def evaluate_prediction_distance(final_state, ground_truth_grades):
    """
    Measure how far off predictions are even when not exact matches.
    A predicted B when actual is A (off by 1) is better than predicted F (off by 4).

    "unknown" predictions receive the maximum penalty distance (4) rather
    than being silently excluded, so failed predictions don't artificially
    improve the distance metrics.
    """
    grade_order = {"a": 4, "b": 3, "c": 2, "d": 1, "f": 0}

    predicted_grades = extract_predictions_from_state(final_state)
    distances = []
    within_one = 0
    total = 0
    unknown_count = 0

    for sid, actual in ground_truth_grades.items():
        pred = predicted_grades.get(sid, "unknown")
        actual_lower = actual.lower()
        actual_rank = grade_order.get(actual_lower, -1)

        # Skip if actual grade is itself unrecognized
        if actual_rank < 0:
            continue

        total += 1
        pred_rank = grade_order.get(pred, -1)

        if pred_rank >= 0:
            dist = abs(pred_rank - actual_rank)
        else:
            # "unknown" or unrecognized prediction → max penalty
            dist = EVAL_MAX_GRADE_DISTANCE
            unknown_count += 1

        distances.append(dist)
        if dist <= 1:
            within_one += 1

    avg_distance = sum(distances) / len(distances) if distances else 0.0
    within_one_pct = within_one / total if total > 0 else 0.0

    return {
        "avg_grade_distance": round(avg_distance, 4),
        "within_one_grade": round(within_one_pct, 4),
        "total_evaluated": total,
        "unknown_predictions": unknown_count,
    }


# =============================================================================
# 6. JOINT GOAL ACCURACY (per-turn belief state correctness)
# =============================================================================

def build_gold_belief_state(gt_entry):
    """
    Build a gold belief state from ground truth for verifiable slots.
    Returns a dict of {slot_path: gold_value} for slots we can verify.
    """
    gold = {}
    gold_grades = gt_entry.get("student_grades", {})

    if not gold_grades:
        return gold

    # Gold average GPA
    gold_gpas = [GRADE_TO_GPA.get(g.lower(), 0.0) for g in gold_grades.values()]
    gold["class_summary.average_gpa"] = round(sum(gold_gpas) / len(gold_gpas), 2)

    # Gold flagged count
    gold["class_summary.flagged_student_count"] = sum(
        1 for g in gold_grades.values() if RISK_MAPPING.get(g.lower()) is not None
    )

    # Gold per-student grades
    for sid, grade in gold_grades.items():
        gold[f"student_grade.{sid}"] = grade.lower()

    # Gold intervention triggered
    gold["intervention_triggered"] = gt_entry.get("intervention_triggered", False)

    return gold


def check_turn_correctness(belief_state, gold, gt_entry):
    """
    Check if all verifiable slots filled so far in the belief state
    match the gold values. Returns (is_correct, checked_count, correct_count).
    """
    predicted_grades = extract_predictions_from_state(belief_state)
    pred_triggered = extract_intervention_triggered(belief_state)

    checked = 0
    correct = 0

    # Check average_gpa if filled
    pred_gpa = belief_state.get("class_summary", {}).get("average_gpa", "")
    if pred_gpa != "" and "class_summary.average_gpa" in gold:
        checked += 1
        try:
            if abs(float(pred_gpa) - gold["class_summary.average_gpa"]) <= EVAL_GPA_TOLERANCE:
                correct += 1
        except (ValueError, TypeError):
            pass

    # Check flagged_student_count if filled
    pred_flagged = belief_state.get("class_summary", {}).get("flagged_student_count", "")
    if pred_flagged != "" and "class_summary.flagged_student_count" in gold:
        checked += 1
        try:
            if abs(int(pred_flagged) - gold["class_summary.flagged_student_count"]) <= EVAL_COUNT_TOLERANCE:
                correct += 1
        except (ValueError, TypeError):
            pass

    # Check per-student predicted grades if filled
    for sid, pred_grade in predicted_grades.items():
        gold_key = f"student_grade.{sid}"
        if gold_key in gold:
            checked += 1
            if pred_grade == gold[gold_key]:
                correct += 1

    # Check intervention triggered if filled
    intervention = belief_state.get("intervention", {})
    if intervention and "intervention_triggered" in gold:
        checked += 1
        if pred_triggered == gold["intervention_triggered"]:
            correct += 1

    is_correct = (checked > 0 and checked == correct)
    soft_accuracy = correct / checked if checked > 0 else 0.0
    return is_correct, checked, correct, soft_accuracy


def evaluate_jga(dialogue_log, gt_entry):
    """
    Joint Goal Accuracy: for each system turn with a belief state,
    check if all verifiable slots filled so far are correct.

    Reports two metrics:
    - jga_strict: fraction of turns where ALL slots are correct (standard JGA)
    - jga_soft: average fraction of correct slots per turn (more informative
      when many slots exist, e.g. 148 student grades)
    """
    gold = build_gold_belief_state(gt_entry)

    if not gold:
        return {"jga_strict": 0.0, "jga_soft": 0.0, "turns_evaluated": 0, "per_turn": []}

    turn_results = []
    turn_idx = 0

    for entry in dialogue_log:
        metadata = entry.get("metadata", {})
        if not metadata:
            continue

        turn_idx += 1
        is_correct, checked, correct_count, soft_accuracy = check_turn_correctness(
            metadata, gold, gt_entry
        )

        turn_results.append({
            "turn": turn_idx,
            "is_correct": is_correct,
            "checked": checked,
            "correct": correct_count,
            "soft_accuracy": round(soft_accuracy, 4),
        })

    total_turns = len(turn_results)
    correct_turns = sum(1 for t in turn_results if t["is_correct"])
    jga_strict = correct_turns / total_turns if total_turns > 0 else 0.0
    jga_soft = sum(t["soft_accuracy"] for t in turn_results) / total_turns if total_turns > 0 else 0.0

    return {
        "jga_strict": round(jga_strict, 4),
        "jga_soft": round(jga_soft, 4),
        "correct_turns": correct_turns,
        "turns_evaluated": total_turns,
        "per_turn": turn_results,
    }


# =============================================================================
# 7. SLOT F1 (per-slot precision / recall / F1)
# =============================================================================

def evaluate_slot_f1(final_state, gt_entry):
    """
    Compute per-slot precision, recall, and F1 across all belief state slots.

    Compares predicted slot values against gold values built from ground truth.
    A slot is "correct" if the predicted value matches the gold value
    (with tolerance for numeric slots).

    Returns:
    - slot_precision: fraction of predicted slots that are correct
    - slot_recall: fraction of gold slots that were correctly predicted
    - slot_f1: harmonic mean of precision and recall
    - per_slot: detailed per-slot results
    """
    gold = build_gold_belief_state(gt_entry)
    if not gold:
        return {"slot_precision": 0.0, "slot_recall": 0.0, "slot_f1": 0.0, "per_slot": {}}

    predicted_grades = extract_predictions_from_state(final_state)
    pred_triggered = extract_intervention_triggered(final_state)

    # Build predicted slot dict in same format as gold
    pred_slots = {}

    # Average GPA
    pred_gpa = final_state.get("class_summary", {}).get("average_gpa", "")
    if pred_gpa != "":
        try:
            pred_slots["class_summary.average_gpa"] = float(pred_gpa)
        except (ValueError, TypeError):
            pass

    # Flagged count
    pred_flagged = final_state.get("class_summary", {}).get("flagged_student_count", "")
    if pred_flagged != "":
        try:
            pred_slots["class_summary.flagged_student_count"] = int(pred_flagged)
        except (ValueError, TypeError):
            pass

    # Per-student grades
    for sid, grade in predicted_grades.items():
        pred_slots[f"student_grade.{sid}"] = grade

    # Intervention triggered
    intervention = final_state.get("intervention", {})
    if intervention:
        pred_slots["intervention_triggered"] = pred_triggered

    # Compare predicted vs gold
    all_slots = set(gold.keys()) | set(pred_slots.keys())
    per_slot = {}
    correct_count = 0
    pred_count = len(pred_slots)
    gold_count = len(gold)

    for slot in all_slots:
        gold_val = gold.get(slot)
        pred_val = pred_slots.get(slot)
        in_gold = slot in gold
        in_pred = slot in pred_slots

        # Determine correctness
        if in_gold and in_pred:
            if slot == "class_summary.average_gpa":
                try:
                    is_correct = abs(float(pred_val) - float(gold_val)) <= EVAL_GPA_TOLERANCE
                except (ValueError, TypeError):
                    is_correct = False
            elif slot == "class_summary.flagged_student_count":
                try:
                    is_correct = abs(int(pred_val) - int(gold_val)) <= EVAL_COUNT_TOLERANCE
                except (ValueError, TypeError):
                    is_correct = False
            else:
                is_correct = pred_val == gold_val
        else:
            is_correct = False

        if is_correct:
            correct_count += 1

        per_slot[slot] = {
            "gold": gold_val,
            "predicted": pred_val,
            "in_gold": in_gold,
            "in_pred": in_pred,
            "correct": is_correct,
        }

    precision = correct_count / pred_count if pred_count > 0 else 0.0
    recall = correct_count / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slot_f1": round(f1, 4),
        "correct": correct_count,
        "predicted_slots": pred_count,
        "gold_slots": gold_count,
        "per_slot": per_slot,
    }


# =============================================================================
# AGGREGATE EVALUATION
# =============================================================================

def evaluate_all(dialogues_path, ground_truth_path):
    """Run all evaluations across all dialogues."""
    dialogues = load_dialogues(dialogues_path)
    ground_truth = load_ground_truth(ground_truth_path)

    all_results = {}
    aggregate = {
        "prediction_accuracy": [],
        "prediction_within_one": [],
        "avg_grade_distance": [],
        "risk_consistency": [],
        "intervention_trigger_accuracy": [],
        "intervention_f1": [],
        "gpa_error": [],
        "flagged_count_error": [],
        "jga_strict": [],
        "jga_soft": [],
        "slot_precision": [],
        "slot_recall": [],
        "slot_f1": [],
    }

    # Temporal breakdown accumulators
    temporal = {
        "early": defaultdict(list),
        "mid": defaultdict(list),
        "late": defaultdict(list),
    }

    for dlg_id, dialogue in dialogues.items():
        if dlg_id not in ground_truth:
            print(f"  WARNING: No ground truth for {dlg_id}, skipping")
            continue

        gt = ground_truth[dlg_id]
        log = dialogue.get("log", [])
        cutoff_week = gt.get("cutoff_week", 0)
        time_bin = get_temporal_bin(cutoff_week)

        # Get final belief state (last system turn's metadata)
        final_state = {}
        for turn in reversed(log):
            if turn.get("metadata") and turn["metadata"] != {}:
                final_state = turn["metadata"]
                break

        student_grades = gt.get("student_grades", {})

        # 1. Grade prediction accuracy (exact match)
        predictions = evaluate_predictions(final_state, student_grades)
        aggregate["prediction_accuracy"].append(predictions["accuracy"])

        # 2. Grade prediction distance (how far off)
        distance = evaluate_prediction_distance(final_state, student_grades)
        aggregate["prediction_within_one"].append(distance["within_one_grade"])
        aggregate["avg_grade_distance"].append(distance["avg_grade_distance"])

        # 3. Risk consistency
        risk = evaluate_risk_mapping(final_state)
        aggregate["risk_consistency"].append(risk["consistency"])

        # 4. Intervention
        intervention = evaluate_intervention(final_state, gt)
        aggregate["intervention_trigger_accuracy"].append(
            1.0 if intervention["trigger_correct"] else 0.0
        )
        if intervention.get("targeting", {}).get("f1") is not None:
            aggregate["intervention_f1"].append(intervention["targeting"]["f1"])

        # 5. Numeric slots
        numeric = evaluate_numeric_slots(final_state, gt)
        if "average_gpa" in numeric and "absolute_error" in numeric["average_gpa"]:
            aggregate["gpa_error"].append(numeric["average_gpa"]["absolute_error"])
        if "flagged_student_count" in numeric and "absolute_error" in numeric["flagged_student_count"]:
            aggregate["flagged_count_error"].append(numeric["flagged_student_count"]["absolute_error"])

        # 6. Joint Goal Accuracy
        jga_result = evaluate_jga(log, gt)
        aggregate["jga_strict"].append(jga_result["jga_strict"])
        aggregate["jga_soft"].append(jga_result["jga_soft"])

        # 7. Slot F1
        slot_f1_result = evaluate_slot_f1(final_state, gt)
        aggregate["slot_precision"].append(slot_f1_result["slot_precision"])
        aggregate["slot_recall"].append(slot_f1_result["slot_recall"])
        aggregate["slot_f1"].append(slot_f1_result["slot_f1"])

        # Accumulate temporal breakdown
        temporal[time_bin]["prediction_accuracy"].append(predictions["accuracy"])
        temporal[time_bin]["prediction_within_one"].append(distance["within_one_grade"])
        temporal[time_bin]["avg_grade_distance"].append(distance["avg_grade_distance"])
        temporal[time_bin]["unknown_predictions"].append(distance["unknown_predictions"])
        temporal[time_bin]["intervention_trigger_accuracy"].append(
            1.0 if intervention["trigger_correct"] else 0.0
        )
        temporal[time_bin]["flagged_count"].append(
            final_state.get("class_summary", {}).get("flagged_student_count", 0)
        )
        temporal[time_bin]["slot_f1"].append(slot_f1_result["slot_f1"])

        # Count intervention triggered
        temporal[time_bin]["intervention_triggered"].append(
            1.0 if extract_intervention_triggered(final_state) else 0.0
        )

        all_results[dlg_id] = {
            "course_id": gt.get("course_id", "unknown"),
            "cutoff_week": cutoff_week,
            "temporal_bin": time_bin,
            "grade_prediction": predictions,
            "grade_distance": distance,
            "risk_mapping": risk,
            "intervention": intervention,
            "numeric_slots": numeric,
            "joint_goal_accuracy": jga_result,
            "slot_f1": slot_f1_result,
        }

    # Compute aggregate metrics
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    def safe_sum(lst):
        return sum(lst) if lst else 0

    summary = {
        "num_dialogues": len(all_results),
        "prediction_accuracy_exact": safe_avg(aggregate["prediction_accuracy"]),
        "prediction_within_one_grade": safe_avg(aggregate["prediction_within_one"]),
        "avg_grade_distance": safe_avg(aggregate["avg_grade_distance"]),
        "risk_mapping_consistency": safe_avg(aggregate["risk_consistency"]),
        "intervention_trigger_accuracy": safe_avg(aggregate["intervention_trigger_accuracy"]),
        "intervention_targeting_f1": safe_avg(aggregate["intervention_f1"]),
        "gpa_mae": safe_avg(aggregate["gpa_error"]),
        "flagged_count_mae": safe_avg(aggregate["flagged_count_error"]),
        "joint_goal_accuracy_strict": safe_avg(aggregate["jga_strict"]),
        "joint_goal_accuracy_soft": safe_avg(aggregate["jga_soft"]),
        "slot_precision": safe_avg(aggregate["slot_precision"]),
        "slot_recall": safe_avg(aggregate["slot_recall"]),
        "slot_f1": safe_avg(aggregate["slot_f1"]),
    }

    # Compute temporal breakdown summary
    temporal_summary = {}
    for time_bin in ["early", "mid", "late"]:
        bin_data = temporal[time_bin]
        num_dialogues = len(bin_data.get("prediction_accuracy", []))
        if num_dialogues == 0:
            temporal_summary[time_bin] = {"num_dialogues": 0}
            continue

        # Compute average flagged count — handle mixed types
        flagged_vals = []
        for v in bin_data.get("flagged_count", []):
            try:
                flagged_vals.append(int(v))
            except (ValueError, TypeError):
                flagged_vals.append(0)

        temporal_summary[time_bin] = {
            "num_dialogues": num_dialogues,
            "prediction_accuracy_exact": safe_avg(bin_data["prediction_accuracy"]),
            "prediction_within_one_grade": safe_avg(bin_data["prediction_within_one"]),
            "avg_grade_distance": safe_avg(bin_data["avg_grade_distance"]),
            "unknown_prediction_count": safe_sum(bin_data["unknown_predictions"]),
            "unknown_prediction_rate": round(
                safe_sum(bin_data["unknown_predictions"]) /
                max(1, num_dialogues), 2
            ),
            "intervention_trigger_accuracy": safe_avg(bin_data["intervention_trigger_accuracy"]),
            "intervention_triggered_pct": round(safe_avg(bin_data["intervention_triggered"]) * 100, 1),
            "avg_flagged_count": round(
                sum(flagged_vals) / len(flagged_vals), 1
            ) if flagged_vals else 0.0,
            "slot_f1": safe_avg(bin_data["slot_f1"]),
        }

    return {
        "summary": summary,
        "temporal_breakdown": temporal_summary,
        "per_dialogue": all_results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PredAct Benchmark Evaluation")
    parser.add_argument("--dialogues", default=OUTPUT_DATA_PATH, help="Path to generated data.json")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", default="eval_results.json", help="Output evaluation results")
    parser.add_argument("--verbose", action="store_true", help="Print per-dialogue results")
    args = parser.parse_args()

    print("=" * 60)
    print("PredAct Benchmark - Evaluation")
    print("=" * 60)

    results = evaluate_all(args.dialogues, args.ground_truth)

    # Print summary
    print("\n--- SUMMARY ---")
    for metric, value in results["summary"].items():
        print(f"  {metric}: {value}")

    # Print temporal breakdown
    print("\n--- TEMPORAL BREAKDOWN ---")
    for time_bin in ["early", "mid", "late"]:
        bin_data = results["temporal_breakdown"].get(time_bin, {})
        num = bin_data.get("num_dialogues", 0)
        if num == 0:
            print(f"\n  {time_bin.upper()} (0 dialogues)")
            continue
        print(f"\n  {time_bin.upper()} ({num} dialogues):")
        for metric, value in bin_data.items():
            if metric != "num_dialogues":
                print(f"    {metric}: {value}")

    if args.verbose:
        print("\n--- PER DIALOGUE ---")
        for dlg_id, dlg_results in results["per_dialogue"].items():
            pred = dlg_results.get("grade_prediction", {})
            dist = dlg_results.get("grade_distance", {})
            intv = dlg_results.get("intervention", {})
            jga = dlg_results.get("joint_goal_accuracy", {})
            sf1 = dlg_results.get("slot_f1", {})
            print(f"\n  {dlg_id} (week {dlg_results.get('cutoff_week', '?')}, {dlg_results.get('temporal_bin', '?')}):")
            print(f"    Prediction exact: {pred.get('accuracy', 'N/A')} ({pred.get('correct', 0)}/{pred.get('total', 0)})")
            print(f"    Within 1 grade:   {dist.get('within_one_grade', 'N/A')}")
            print(f"    Avg grade dist:   {dist.get('avg_grade_distance', 'N/A')}")
            print(f"    Unknown preds:    {dist.get('unknown_predictions', 0)}")
            print(f"    Slot F1:          {sf1.get('slot_f1', 'N/A')}")
            print(f"    Intervention:     trigger={'correct' if intv.get('trigger_correct') else 'WRONG'} "
                  f"(pred={intv.get('predicted_triggered')}, gold={intv.get('gold_triggered')})")
            if intv.get("targeting"):
                print(f"    Targeting F1:     {intv['targeting'].get('f1', 'N/A')}")
            print(f"    JGA strict:       {jga.get('jga_strict', 'N/A')} ({jga.get('correct_turns', 0)}/{jga.get('turns_evaluated', 0)} turns)")
            print(f"    JGA soft:         {jga.get('jga_soft', 'N/A')}")

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()