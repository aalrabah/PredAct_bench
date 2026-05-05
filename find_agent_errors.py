"""
Diagnostic: find scenarios where the k-NN actually makes errors on D/F students.

For each (course, week, feature_set) combination, run predict_final_grade_for_student
on every student in the test set and compare against ground truth.

Output: per-scenario summary of TP / FP / FN / TN + sample errors.

Use this to pick scenarios with real errors for the human study.

Usage:
    python find_agent_errors.py
"""

import os
import json
from collections import defaultdict

from tools import load_db, predict_final_grade_for_student


TRAIN_DB_PATH = "results/dataset/cs_db_train.json"
TEST_SETS_DIR = "results/dataset/test_sets"
GROUND_TRUTH_PATH = "results/dataset/ground_truth_for_cutoff_data.json"


def is_at_risk(grade):
    return grade.lower() in ("d", "f")


def analyze_scenario(db, course_id, course_file, week, feature_set, ground_truth):
    path = os.path.join(TEST_SETS_DIR, course_file)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    students = test_set["students"]

    gt_grades = ground_truth.get(course_file, {}).get("student_grades", {})

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    fp_samples = []
    fn_samples = []

    for s in students:
        sid = s["student_id"]
        truth = gt_grades.get(sid, "").lower()
        if not truth:
            continue
        truly_atrisk = is_at_risk(truth)

        try:
            pred = predict_final_grade_for_student(
                db, course_id, s,
                up_to_week=week, feature_set=feature_set,
            )
        except Exception:
            continue
        if "error" in pred:
            continue

        predicted = pred.get("predicted_grade", "").lower()
        conf = pred.get("confidence", 0)
        agent_flagged = predicted in ("d", "f")

        if agent_flagged and truly_atrisk:
            tp += 1
        elif agent_flagged and not truly_atrisk:
            fp += 1
            fp_samples.append((sid, truth, predicted, conf))
        elif not agent_flagged and truly_atrisk:
            fn += 1
            fn_samples.append((sid, truth, predicted, conf))
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "course_id": course_id,
        "course_file": course_file,
        "week": week,
        "feature_set": feature_set,
        "total": total,
        "true_atrisk": tp + fn,
        "agent_flagged": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "fp_samples": fp_samples[:5],
        "fn_samples": fn_samples[:5],
    }


def main():
    print("Loading training DB...", flush=True)
    db = load_db(TRAIN_DB_PATH)
    print(f"  Loaded {len(db)} courses", flush=True)

    print("Loading ground truth...", flush=True)
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    print(f"  Loaded {len(ground_truth)} test files", flush=True)

    all_files = sorted(f for f in os.listdir(TEST_SETS_DIR) if f.endswith(".json"))
    print(f"Found {len(all_files)} test files in {TEST_SETS_DIR}", flush=True)

    # Group test files by course + weeks
    by_course = defaultdict(list)
    for fname in all_files:
        base = fname.replace(".json", "")
        if "_week" not in base:
            continue
        parts = base.split("_week")
        raw_course = parts[0]
        course = raw_course
        try:
            week = int(parts[1])
        except ValueError:
            continue
        by_course[course].append((fname, week))

    # Build candidate list: one early-week + one late-week per course
    scenarios_to_try = []
    for course_id, week_files in sorted(by_course.items()):
        weeks = sorted({w for _, w in week_files})
        if not weeks:
            continue
        early_candidates = [w for w in weeks if 3 <= w <= 5]
        late_candidates = [w for w in weeks if 8 <= w <= 10]
        for wk in early_candidates[:1]:
            fname = f"{course_id.replace(' ', '')}_week{wk}.json"
            scenarios_to_try.append((course_id, fname, wk, "minimal"))
        for wk in late_candidates[:1]:
            fname = f"{course_id.replace(' ', '')}_week{wk}.json"
            scenarios_to_try.append((course_id, fname, wk, "full"))

    print(f"Testing {len(scenarios_to_try)} (course, week, feature_set) scenarios", flush=True)
    print()

    results = []
    for i, (course_id, fname, wk, fs) in enumerate(scenarios_to_try, 1):
        print(f"[{i}/{len(scenarios_to_try)}] {course_id} week {wk} ({fs})...", flush=True)
        result = analyze_scenario(db, course_id, fname, wk, fs, ground_truth)
        if result is None:
            print(f"    SKIP: file {fname} not found", flush=True)
            continue
        if result["true_atrisk"] < 3:
            print(f"    SKIP: only {result['true_atrisk']} at-risk students (need >= 3)", flush=True)
            continue
        print(f"    n={result['total']}, atrisk={result['true_atrisk']}, "
              f"TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']} "
              f"(acc={result['accuracy']:.2f})", flush=True)
        results.append(result)

    print()
    print(f"Analyzed {len(results)} scenarios with enough at-risk students", flush=True)

    if not results:
        print("No scenarios met the criteria.", flush=True)
        return

    # Sort by scenarios with most errors (FP + FN)
    results.sort(key=lambda r: (r["fp"] + r["fn"]), reverse=True)

    print()
    print("=" * 120)
    print("SCENARIOS RANKED BY TOTAL ERRORS (FP + FN)")
    print("=" * 120)
    print(f"{'Course':<10}{'Week':<6}{'Feat':<10}{'N':<5}"
          f"{'TrueD/F':<9}{'AgentFlg':<10}{'TP':<4}{'FP':<4}{'FN':<4}{'TN':<5}"
          f"{'Acc':<7}{'Prec':<7}{'Rec':<7}")
    print("-" * 120)
    for r in results[:30]:
        print(f"{r['course_id']:<10}{r['week']:<6}{r['feature_set']:<10}"
              f"{r['total']:<5}{r['true_atrisk']:<9}{r['agent_flagged']:<10}"
              f"{r['tp']:<4}{r['fp']:<4}{r['fn']:<4}{r['tn']:<5}"
              f"{r['accuracy']:<7}{r['precision']:<7}{r['recall']:<7}")

    print()
    print("=" * 120)
    print("TOP 5 CANDIDATES WITH REAL ERRORS")
    print("=" * 120)
    for r in results[:5]:
        print()
        print(f"{r['course_id']} week {r['week']} ({r['feature_set']})")
        print(f"  N={r['total']}, true D/F={r['true_atrisk']}, "
              f"agent flagged={r['agent_flagged']} "
              f"(TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")
        if r["fp_samples"]:
            print("  False positives (agent flagged, but NOT D/F):")
            for sid, truth, pred, conf in r["fp_samples"]:
                print(f"    {sid}: true={truth.upper()}, predicted={pred.upper()} "
                      f"(conf {conf*100:.0f}%)")
        if r["fn_samples"]:
            print("  False negatives (agent missed, actually D/F):")
            for sid, truth, pred, conf in r["fn_samples"]:
                print(f"    {sid}: true={truth.upper()}, predicted={pred.upper()} "
                      f"(conf {conf*100:.0f}%)")

    # Write full CSV
    out = "agent_error_scan.csv"
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["course_id", "week", "feature_set", "total",
                         "true_atrisk", "agent_flagged",
                         "tp", "fp", "fn", "tn",
                         "accuracy", "precision", "recall"])
        for r in results:
            writer.writerow([r["course_id"], r["week"], r["feature_set"], r["total"],
                             r["true_atrisk"], r["agent_flagged"],
                             r["tp"], r["fp"], r["fn"], r["tn"],
                             r["accuracy"], r["precision"], r["recall"]])
    print()
    print(f"Full results saved to {out}", flush=True)


if __name__ == "__main__":
    main()