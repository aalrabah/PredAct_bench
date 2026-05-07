"""
Accuracy injector for the agent-to-agent simulator.

Mirrors the calibration logic in app.py's predict_all_students_for_scenario,
but stays fully isolated from app.py / tools.py so the human study is not
touched.

Two entry points:

1. calibrate_to_accuracy(raw_preds, target_acc, seed)
   Pure mutator: given a list of (sid, pred_dict, truth, is_correct) tuples,
   flip predictions in place so that exactly target_acc fraction match truth.

2. get_calibrated_predictions(db, students, course_id, week, feature_set,
                              gt_grades, target_acc, seed)
   Higher-level convenience: runs the real k-NN, calibrates, returns
   {sid: pred_dict} ready for the assistant agent's tool stub to read from.

Seed contract: caller provides a deterministic seed per
(llm, accuracy, scenario, run_idx).
"""

import random

from tools import predict_final_grade_for_student


_ALL_GRADES = ["a", "b", "c", "d", "f"]


def calibrate_to_accuracy(raw_preds, target_acc, seed):
    """
    In-place: flip predictions so that exactly `target_acc` fraction of
    `raw_preds` match ground truth. Deterministic given `seed`.

    Args:
        raw_preds: list of (sid, pred_dict, truth, is_correct) tuples.
                   pred_dict is mutated (predicted_grade, failure_risk).
                   `truth` is the ground-truth letter grade lowercased.
        target_acc: float in [0, 1], fraction that should match truth.
                    None disables calibration (raw predictions kept as-is).
        seed: int, for reproducibility.

    Returns:
        dict {natural_correct, target_correct, achieved_correct, total,
              natural_accuracy, target_accuracy, achieved_accuracy} —
        a stats record proving the calibration hit the target.
    """
    total = len(raw_preds)
    natural_correct = sum(1 for _, _, _, c in raw_preds if c)

    if not raw_preds:
        return {"total": 0, "natural_correct": 0, "target_correct": 0,
                "achieved_correct": 0, "natural_accuracy": None,
                "target_accuracy": target_acc, "achieved_accuracy": None}

    if target_acc is None:
        # No calibration — achieved == natural
        return {"total": total, "natural_correct": natural_correct,
                "target_correct": natural_correct, "achieved_correct": natural_correct,
                "natural_accuracy": natural_correct / total,
                "target_accuracy": None,
                "achieved_accuracy": natural_correct / total}

    rng = random.Random(seed & 0xFFFFFFFF)
    target_correct = round(target_acc * total)

    if natural_correct > target_correct:
        # Too many correct → flip some correct ones to wrong
        correct_preds = [p for p in raw_preds if p[3]]
        rng.shuffle(correct_preds)
        n_to_flip = natural_correct - target_correct
        for _sid, pred, truth, _ in correct_preds[:n_to_flip]:
            wrong = [g for g in _ALL_GRADES if g != truth]
            pred["predicted_grade"] = rng.choice(wrong)
            pred["failure_risk"] = "high" if pred["predicted_grade"] in ("d", "f") else None
    elif natural_correct < target_correct:
        # Too few correct → flip some wrong ones to correct
        wrong_preds = [p for p in raw_preds if not p[3]]
        rng.shuffle(wrong_preds)
        n_to_flip = target_correct - natural_correct
        for _sid, pred, truth, _ in wrong_preds[:n_to_flip]:
            if truth:
                pred["predicted_grade"] = truth
                pred["failure_risk"] = "high" if truth in ("d", "f") else None

    # Recount AFTER the flips to verify (can't rely on is_correct flag now).
    achieved_correct = sum(
        1 for _sid, pred, truth, _ in raw_preds
        if pred["predicted_grade"].lower() == truth
    )

    return {
        "total": total,
        "natural_correct": natural_correct,
        "target_correct": target_correct,
        "achieved_correct": achieved_correct,
        "natural_accuracy":  natural_correct / total,
        "target_accuracy":   target_acc,
        "achieved_accuracy": achieved_correct / total,
    }


def get_calibrated_predictions(db, students, course_id, week, feature_set,
                               gt_grades, target_acc, seed):
    """
    Run the real k-NN over `students`, then calibrate to `target_acc`.

    Returns:
        (preds_dict, stats):
            preds_dict — {sid: pred_dict} after calibration. Each pred_dict
                         is a copy, safe to mutate downstream.
            stats      — dict from calibrate_to_accuracy() proving the
                         achieved accuracy. Also includes n_predicted (how
                         many of `students` got a prediction at all — others
                         errored and were dropped).
    """
    raw_preds = []
    n_errors = 0
    for s in students:
        sid = s["student_id"]
        pred = predict_final_grade_for_student(
            db, course_id, s,
            up_to_week=week, feature_set=feature_set,
        )
        if "error" in pred:
            n_errors += 1
            continue
        pred = dict(pred)  # defensive copy
        truth = (gt_grades.get(sid, "") or "").lower()
        is_correct = pred["predicted_grade"].lower() == truth
        raw_preds.append((sid, pred, truth, is_correct))

    stats = calibrate_to_accuracy(raw_preds, target_acc, seed)
    stats["n_students_input"] = len(students)
    stats["n_predicted"] = len(raw_preds)
    stats["n_errors"]   = n_errors

    preds_dict = {sid: pred for sid, pred, _truth, _ok in raw_preds}
    return preds_dict, stats
