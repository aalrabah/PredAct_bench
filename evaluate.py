"""
PredAct Benchmark - Evaluation
Measures prediction accuracy, dialogue state tracking, and intervention quality.

Key fixes:
- Extracts predicted grades from ALL risk groups including no_risk
- Intervention trigger: "no_intervention" key means NOT triggered
- Joint goal accuracy: builds gold belief state from ground_truth.json when
  final_belief_state is not explicitly provided
"""

import json
import argparse
from collections import defaultdict

from config import (
    OUTPUT_DATA_PATH,
    EVAL_GPA_TOLERANCE,
    EVAL_COUNT_TOLERANCE,
    RISK_MAPPING,
    GRADE_TO_GPA,
)
from state import load_ontology, parse_ontology


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
            "course_id": "CS 100",
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
    """
    Extract per-student predicted grades from the belief state.
    Handles all risk groups including no_risk.
    Returns dict: {student_id: predicted_grade}
    """
    predictions = {}
    student_status = final_state.get("student_status", {})

    for risk_key, group in student_status.items():
        if not isinstance(group, dict):
            continue

        predicted_grade = group.get("predicted_grade", "unknown")
        student_ids = group.get("student_ids", [])

        for sid in student_ids:
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
    """
    pred_triggered = extract_intervention_triggered(final_state)
    gold_triggered = gt_entry.get("intervention_triggered", False)

    trigger_correct = pred_triggered == gold_triggered

    # If both triggered, evaluate which students were targeted
    targeting = {}
    if pred_triggered and gold_triggered:
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

        if gold_flagged:
            precision = len(pred_flagged & gold_flagged) / len(pred_flagged) if pred_flagged else 0.0
            recall = len(pred_flagged & gold_flagged) / len(gold_flagged) if gold_flagged else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0

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
    """
    grade_order = {"a": 4, "b": 3, "c": 2, "d": 1, "f": 0, "unknown": -1}

    predicted_grades = extract_predictions_from_state(final_state)
    distances = []
    within_one = 0
    total = 0

    for sid, actual in ground_truth_grades.items():
        pred = predicted_grades.get(sid, "unknown")
        actual_lower = actual.lower()

        pred_rank = grade_order.get(pred, -1)
        actual_rank = grade_order.get(actual_lower, -1)

        if pred_rank >= 0 and actual_rank >= 0:
            dist = abs(pred_rank - actual_rank)
            distances.append(dist)
            total += 1
            if dist <= 1:
                within_one += 1

    avg_distance = sum(distances) / len(distances) if distances else 0.0
    within_one_pct = within_one / total if total > 0 else 0.0

    return {
        "avg_grade_distance": round(avg_distance, 4),
        "within_one_grade": round(within_one_pct, 4),
        "total_evaluated": total,
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
    }

    for dlg_id, dialogue in dialogues.items():
        if dlg_id not in ground_truth:
            print(f"  WARNING: No ground truth for {dlg_id}, skipping")
            continue

        gt = ground_truth[dlg_id]
        log = dialogue.get("log", [])

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

        all_results[dlg_id] = {
            "grade_prediction": predictions,
            "grade_distance": distance,
            "risk_mapping": risk,
            "intervention": intervention,
            "numeric_slots": numeric,
        }

    # Compute aggregate metrics
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

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
    }

    return {"summary": summary, "per_dialogue": all_results}


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

    if args.verbose:
        print("\n--- PER DIALOGUE ---")
        for dlg_id, dlg_results in results["per_dialogue"].items():
            pred = dlg_results.get("grade_prediction", {})
            dist = dlg_results.get("grade_distance", {})
            intv = dlg_results.get("intervention", {})
            print(f"\n  {dlg_id}:")
            print(f"    Prediction exact: {pred.get('accuracy', 'N/A')} ({pred.get('correct', 0)}/{pred.get('total', 0)})")
            print(f"    Within 1 grade:   {dist.get('within_one_grade', 'N/A')}")
            print(f"    Avg grade dist:   {dist.get('avg_grade_distance', 'N/A')}")
            print(f"    Intervention:     trigger={'correct' if intv.get('trigger_correct') else 'WRONG'} "
                  f"(pred={intv.get('predicted_triggered')}, gold={intv.get('gold_triggered')})")

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()