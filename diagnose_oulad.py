"""
PredAct Diagnosis — UK 5-Grade System
Reads ground_truth.json + grade logs directly. No tod.py needed.
Runs V6 predictions and UK threshold sweep against A/B/C/D/F ground truth.

Usage:
    python diagnose_oulad.py --gt split_uk/ground_truth.json --db split_uk/cs_db.json --logs split_uk/logs/ --limit 20
"""

import json
import os
import argparse
from collections import defaultdict, Counter

from tools import (
    load_db, lookup_course, process_students, extract_scores,
    extract_graded_scores, extract_engagement, detect_vle_data,
    build_engagement_distribution, normalize_engagement,
    get_syllabus, compute_weighted_score, compute_raw_average_score,
    learn_grade_thresholds, compute_adaptive_tolerance,
    score_to_grade_adaptive, class_prior_predict,
    match_students, predict_grade, _is_engagement_activity,
)

GRADE_ORDER = ["a", "b", "c", "d", "f"]


def uk_threshold_predict(score, a_t, b_t, c_t, d_t):
    if score is None:
        return None
    if score >= a_t:
        return "a"
    elif score >= b_t:
        return "b"
    elif score >= c_t:
        return "c"
    elif score >= d_t:
        return "d"
    else:
        return "f"


def load_grades_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("course_id", "unknown"), data.get("students", [])
    elif isinstance(data, list):
        return (data[0].get("course_id", "unknown") if data else "unknown"), data
    return None, None


def print_confusion(pairs, grades_present, label=""):
    if label:
        print(f"\n  {label}")
    col_label = "pred\\actual"
    header = f"  {col_label:>12}"
    for g in grades_present:
        header += f" {g.upper():>5}"
    header += f" {'TOT':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    confusion = defaultdict(lambda: defaultdict(int))
    for p, a in pairs:
        confusion[p][a] += 1

    for pg in grades_present:
        row = f"  {pg.upper():>12}"
        rt = 0
        for ag in grades_present:
            c = confusion[pg][ag]
            rt += c
            row += f" {c:>5}"
        row += f" {rt:>5}"
        print(row)


def print_per_grade_acc(pairs, grades_present):
    for g in grades_present:
        students = [(p, a) for p, a in pairs if a == g]
        if not students:
            continue
        correct = sum(1 for p, a in students if p == a)
        acc = correct / len(students) * 100
        pred_dist = Counter(p for p, _ in students)
        dist_str = ", ".join(f"{k.upper()}:{v}" for k, v in sorted(pred_dist.items(), key=lambda x: -x[1]))
        print(f"    {g.upper()}: {correct}/{len(students)} = {acc:.1f}%  [{dist_str}]")


def run(gt_path, db_path, logs_dir, limit=None):
    print("=" * 60)
    print("PredAct Diagnosis — UK 5-Grade System")
    print("=" * 60)

    # Load ground truth
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    dlg_ids = sorted(ground_truth.keys())
    if limit:
        dlg_ids = dlg_ids[:limit]
        print(f"  Limited to {limit} dialogues")

    # Build actual lookup: dlg_id → {student_id: grade}
    actual_by_dlg = {}
    for dlg_id in dlg_ids:
        actual_by_dlg[dlg_id] = {
            sid: g.lower()
            for sid, g in ground_truth[dlg_id].get("student_grades", {}).items()
        }

    # Load DB
    db = load_db(db_path)

    # Map dlg_id to grades file
    log_files = {}
    for f in os.listdir(logs_dir):
        if f.endswith("_grades.json"):
            dlg_id = f.replace("_grades.json", ".json")
            log_files[dlg_id] = os.path.join(logs_dir, f)

    # =========================================================================
    # RUN V6 PREDICTIONS + PRECOMPUTE FOR THRESHOLD SWEEP
    # =========================================================================
    print(f"\n  Running V6 predictions on {len(dlg_ids)} dialogues...\n")

    v6_pairs = []  # (predicted, actual)
    dialogue_data = []  # for threshold sweep

    for idx, dlg_id in enumerate(dlg_ids):
        if dlg_id not in log_files:
            continue

        course_id, students = load_grades_file(log_files[dlg_id])
        if course_id is None:
            continue

        actuals = actual_by_dlg.get(dlg_id, {})
        cutoff = ground_truth[dlg_id].get("cutoff_week", "?")

        print(f"  [{idx+1}/{len(dlg_ids)}] {dlg_id} | {course_id} | "
              f"{len(students)} students | week {cutoff}", end="")

        # Run V6
        try:
            results = process_students(db, course_id, students)
        except Exception as e:
            print(f" — ERROR: {e}")
            continue
        if "error" in results:
            print(f" — ERROR: {results['error']}")
            continue

        v6_preds = {r["student_id"]: r["predicted_grade"].lower()
                    for r in results["student_results"]}

        # Precompute for threshold sweep
        course_info, intervention_data, historical_students = lookup_course(db, course_id)
        all_max_weeks = [extract_scores(s)[2] for s in students]
        current_week = max(all_max_weeks) if all_max_weeks else 0
        syllabus = get_syllabus(historical_students, up_to_week=current_week)

        student_ws_list = []
        dlg_correct = 0
        dlg_total = 0

        for student in students:
            sid = student.get("student_id", "unknown")
            actual = actuals.get(sid, "unknown")
            if actual == "unknown":
                continue

            v6_pred = v6_preds.get(sid, "unknown")
            v6_pairs.append((v6_pred, actual))

            scores, _, _ = extract_scores(student)
            ws, _ = compute_weighted_score(scores, syllabus)
            student_ws_list.append((sid, actual, ws))

            dlg_total += 1
            if v6_pred == actual:
                dlg_correct += 1

        if dlg_total > 0:
            print(f" | V6={dlg_correct/dlg_total*100:.1f}%")
        else:
            print(f" | no students")

        dialogue_data.append({
            "dlg_id": dlg_id,
            "students": student_ws_list,
        })

    # =========================================================================
    # V6 BASELINE RESULTS
    # =========================================================================
    v6_correct = sum(1 for p, a in v6_pairs if p == a)
    v6_total = len(v6_pairs)
    grades_present = [g for g in GRADE_ORDER if any(a == g for _, a in v6_pairs)]

    print(f"\n{'='*60}")
    print(f"V6 BASELINE: {v6_correct}/{v6_total} = {v6_correct/v6_total*100:.1f}%")
    print(f"{'='*60}")

    print_confusion(v6_pairs, grades_present, "Confusion matrix:")
    print(f"\n  Per-grade accuracy:")
    print_per_grade_acc(v6_pairs, grades_present)

    # Actual distribution
    actual_dist = Counter(a for _, a in v6_pairs)
    print(f"\n  Actual distribution:")
    for g in grades_present:
        c = actual_dist.get(g, 0)
        pct = c / v6_total * 100
        print(f"    {g.upper()}: {c} ({pct:.1f}%)")
    majority = actual_dist.most_common(1)[0]
    print(f"  Majority baseline (all {majority[0].upper()}): {majority[1]/v6_total*100:.1f}%")

    # =========================================================================
    # SCORE DISTRIBUTION BY GRADE
    # =========================================================================
    print(f"\n{'='*60}")
    print("SCORE DISTRIBUTION BY ACTUAL GRADE")
    print(f"{'='*60}")

    grade_scores = defaultdict(list)
    no_score = 0
    for ddata in dialogue_data:
        for sid, actual, ws in ddata["students"]:
            if ws is not None:
                grade_scores[actual].append(ws)
            else:
                no_score += 1

    for g in grades_present:
        scores = grade_scores.get(g, [])
        if not scores:
            print(f"  {g.upper()}: no scores")
            continue
        scores.sort()
        n = len(scores)
        print(f"  {g.upper()} (n={n:>4}): min={min(scores):>5.1f}  "
              f"p25={scores[n//4]:>5.1f}  med={scores[n//2]:>5.1f}  "
              f"avg={sum(scores)/n:>5.1f}  p75={scores[3*n//4]:>5.1f}  "
              f"max={max(scores):>5.1f}")

    print(f"  No score (no graded work yet): {no_score}")

    # =========================================================================
    # THRESHOLD SWEEP
    # =========================================================================
    configs = []
    configs.append(("V6 baseline", None, None, None, None))

    # Standard UK
    configs.append(("UK 70/60/50/40", 70, 60, 50, 40))

    # Shift A threshold
    for a_t in [65, 68, 72, 75, 80]:
        configs.append((f"a={a_t}/60/50/40", a_t, 60, 50, 40))

    # Shift B threshold
    for b_t in [55, 58, 62, 65]:
        configs.append((f"70/{b_t}/50/40", 70, b_t, 50, 40))

    # Shift D/F boundary
    for d_t in [35, 38, 42, 45]:
        configs.append((f"70/60/50/{d_t}", 70, 60, 50, d_t))

    print(f"\n{'='*100}")
    print(f"SWEEPING {len(configs)} THRESHOLD CONFIGURATIONS")
    print(f"{'='*100}")
    print(f"{'Config':<20} {'Overall':>8} {'A':>7} {'B':>7} {'C':>7} {'D':>7} {'F':>7} "
          f"{'A#':>5} {'B#':>5} {'C#':>5} {'D#':>5} {'F#':>5} {'noSc':>5}")
    print(f"{'-'*100}")

    best_acc = 0
    best_name = ""

    for config_name, a_t, b_t, c_t, d_t in configs:
        if a_t is None:
            # V6 baseline
            pairs = v6_pairs
            no_sc = "  N/A"
        else:
            pairs = []
            no_sc_count = 0
            for ddata in dialogue_data:
                for sid, actual, ws in ddata["students"]:
                    pred = uk_threshold_predict(ws, a_t, b_t, c_t, d_t)
                    if pred is None:
                        pred = "a"  # majority fallback
                        no_sc_count += 1
                    pairs.append((pred, actual))
            no_sc = f"{no_sc_count:>5}"

        correct = sum(1 for p, a in pairs if p == a)
        total = len(pairs)
        acc = correct / total * 100 if total > 0 else 0

        gc = defaultdict(int)
        gt_counts = defaultdict(int)
        pc = Counter()
        for p, a in pairs:
            gt_counts[a] += 1
            pc[p] += 1
            if p == a:
                gc[a] += 1

        per_g = {}
        for g in GRADE_ORDER:
            if gt_counts.get(g, 0) > 0:
                per_g[g] = gc[g] / gt_counts[g] * 100
            else:
                per_g[g] = 0.0

        if acc > best_acc:
            best_acc = acc
            best_name = config_name

        print(f"{config_name:<20} {acc:>7.1f}% "
              f"{per_g['a']:>6.1f}% {per_g['b']:>6.1f}% {per_g['c']:>6.1f}% "
              f"{per_g['d']:>6.1f}% {per_g['f']:>6.1f}% "
              f"{pc.get('a', 0):>5} {pc.get('b', 0):>5} {pc.get('c', 0):>5} "
              f"{pc.get('d', 0):>5} {pc.get('f', 0):>5} {no_sc}")

    print(f"{'='*100}")
    print(f"\n  BEST: {best_name} at {best_acc:.1f}%")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="PredAct UK 5-Grade Diagnosis")
    parser.add_argument("--gt", required=True, help="Path to ground_truth.json")
    parser.add_argument("--db", required=True, help="Path to cs_db.json")
    parser.add_argument("--logs", required=True, help="Path to logs directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit dialogues")
    args = parser.parse_args()
    run(args.gt, args.db, args.logs, limit=args.limit)


if __name__ == "__main__":
    main()