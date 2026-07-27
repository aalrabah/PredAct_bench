"""
Exp 1 — Tool Characterization

Measures k-NN prediction accuracy across the semester on PredAct-CS and OULAD.

For each (dataset, course, cutoff_week, feature_set, student):
  - run predict_final_grade_for_student
  - compare against ground truth
  - log predicted_grade, confidence, correct flag, pct_complete

Outputs:
  - results/exp1/exp1_raw.csv       (one row per prediction)
  - results/exp1/exp1_summary.csv   (aggregated by pct_bin)
  - results/exp1/exp1_plot.png      (accuracy + ECE vs pct_complete)
  - Terminal summary tables

Usage:
    python exp1_tool_characterization.py
    python exp1_tool_characterization.py --limit-courses 5
    python exp1_tool_characterization.py --feature-sets full
"""

import os
import re
import json
import csv
import argparse
from collections import defaultdict
from tools import load_db, predict_final_grade_for_student


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

DATASETS = [
    {
        "name": "predact_cs",
        "train_db": os.path.join(RESULTS_ROOT, "predact_cs", "cs_db_train.json"),
        "test_sets_dir": os.path.join(RESULTS_ROOT, "predact_cs", "test_sets"),
        "ground_truth": os.path.join(RESULTS_ROOT, "predact_cs", "ground_truth_for_cutoff_data.json"),
    },
    {
        "name": "oulad",
        "train_db": os.path.join(RESULTS_ROOT, "oulad", "oulad_db_train.json"),
        "test_sets_dir": os.path.join(RESULTS_ROOT, "oulad", "test_sets"),
        "ground_truth": os.path.join(RESULTS_ROOT, "oulad", "ground_truth_for_cutoff_data.json"),
    },
]

AT_RISK_GRADES = {"d", "f"}
WEEK_REGEX = re.compile(r"_week(\d+)\.json$")

# Bin edges for pct_complete (percent of semester finished)
PCT_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# =============================================================================
# HELPERS
# =============================================================================

def parse_week_from_filename(fname):
    m = WEEK_REGEX.search(fname)
    return int(m.group(1)) if m else None


def pct_to_bin(pct):
    """Return string label like '0-10%' for a percent 0-100."""
    for i in range(len(PCT_BIN_EDGES) - 1):
        lo, hi = PCT_BIN_EDGES[i], PCT_BIN_EDGES[i + 1]
        if pct >= lo and (pct < hi or (i == len(PCT_BIN_EDGES) - 2 and pct <= hi)):
            return f"{lo}-{hi}%"
    return f"{PCT_BIN_EDGES[-2]}-{PCT_BIN_EDGES[-1]}%"


def pct_bin_midpoint(bin_label):
    """'0-10%' -> 5.0 (for plotting x positions)."""
    m = re.match(r"(\d+)-(\d+)%", bin_label)
    if not m:
        return 0.0
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo + hi) / 2.0


def grade_correct(pred, truth):
    if pred is None or truth is None:
        return None
    return pred.lower() == truth.lower()


def at_risk_prob(predicted_grade, confidence):
    """Approximate P(at-risk) from predicted class + confidence."""
    if predicted_grade is None:
        return 0.5
    if predicted_grade.lower() in AT_RISK_GRADES:
        return confidence
    return 1.0 - confidence


# =============================================================================
# METRICS
# =============================================================================

def compute_accuracy(rows):
    if not rows:
        return None
    return sum(1 for r in rows if r["correct"]) / len(rows)


def compute_ece(rows, n_bins=10):
    """Expected Calibration Error for binary at-risk vs not."""
    if not rows:
        return None
    bins = defaultdict(list)
    for r in rows:
        p = r["at_risk_prob"]
        outcome = 1 if r["truth_at_risk"] else 0
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, outcome))

    n = len(rows)
    ece = 0.0
    for pairs in bins.values():
        bin_size = len(pairs)
        bin_avg_p = sum(p for p, _ in pairs) / bin_size
        bin_avg_outcome = sum(o for _, o in pairs) / bin_size
        ece += (bin_size / n) * abs(bin_avg_outcome - bin_avg_p)
    return ece


def compute_brier(rows):
    """Binary Brier score on at-risk vs not."""
    if not rows:
        return None
    total = 0.0
    for r in rows:
        p = r["at_risk_prob"]
        outcome = 1 if r["truth_at_risk"] else 0
        total += (p - outcome) ** 2
    return total / len(rows)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_predictions(dataset_cfg, feature_sets, limit_courses=None, verbose=True):
    name = dataset_cfg["name"]
    db = load_db(dataset_cfg["train_db"])

    course_ids_available = {c["course_id"] for c in db}

    with open(dataset_cfg["ground_truth"], "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    test_files = sorted(os.listdir(dataset_cfg["test_sets_dir"]))

    if limit_courses:
        selected_courses = set()
        filtered = []
        for fname in test_files:
            base = WEEK_REGEX.sub("", fname)
            if base not in selected_courses and len(selected_courses) >= limit_courses:
                continue
            selected_courses.add(base)
            filtered.append(fname)
        test_files = filtered

    # Max week per course (for pct_complete)
    course_max_week = defaultdict(int)
    for fname in test_files:
        week = parse_week_from_filename(fname)
        if week is None:
            continue
        course_key = WEEK_REGEX.sub("", fname)
        if week > course_max_week[course_key]:
            course_max_week[course_key] = week

    rows = []
    total_files = len(test_files)

    for i, fname in enumerate(test_files, 1):
        week = parse_week_from_filename(fname)
        if week is None:
            continue

        test_path = os.path.join(dataset_cfg["test_sets_dir"], fname)
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        course_id = test_data.get("course_id")
        students = test_data.get("students", [])

        if course_id not in course_ids_available:
            continue

        gt_entry = ground_truth.get(fname, {})
        gt_grades = gt_entry.get("student_grades", {})

        course_key = WEEK_REGEX.sub("", fname)
        max_week = course_max_week[course_key] or 1
        pct_complete = round(100.0 * week / max_week, 2)
        pct_bin = pct_to_bin(pct_complete)

        if verbose:
            print(f"  [{name}] {i}/{total_files}  {fname}  "
                  f"({len(students)} students, pct={pct_complete:.1f}%)",
                  flush=True)

        for student in students:
            sid = student["student_id"]
            truth = gt_grades.get(sid)
            if truth is None:
                continue
            truth_at_risk = truth.lower() in AT_RISK_GRADES

            for fs in feature_sets:
                try:
                    pred = predict_final_grade_for_student(
                        db, course_id, student,
                        up_to_week=week, feature_set=fs,
                    )
                except Exception:
                    continue

                if "error" in pred:
                    continue

                predicted_grade = pred.get("predicted_grade", "unknown")
                confidence = pred.get("confidence", 0.0)

                rows.append({
                    "dataset": name,
                    "course_id": course_id,
                    "week": week,
                    "pct_complete": pct_complete,
                    "pct_bin": pct_bin,
                    "feature_set": fs,
                    "student_id": sid,
                    "predicted_grade": predicted_grade,
                    "confidence": confidence,
                    "truth_grade": truth.lower(),
                    "correct": grade_correct(predicted_grade, truth),
                    "truth_at_risk": truth_at_risk,
                    "at_risk_prob": at_risk_prob(predicted_grade, confidence),
                })

    return rows


def aggregate(rows, group_keys):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups[key].append(r)

    summary = []
    for key, group_rows in sorted(groups.items()):
        entry = {k: v for k, v in zip(group_keys, key)}
        entry["n"] = len(group_rows)
        entry["accuracy"] = compute_accuracy(group_rows)
        entry["ece"] = compute_ece(group_rows)
        entry["brier"] = compute_brier(group_rows)

        preds_at_risk = sum(1 for r in group_rows if r["predicted_grade"] in AT_RISK_GRADES)
        tp = sum(1 for r in group_rows
                 if r["predicted_grade"] in AT_RISK_GRADES and r["truth_at_risk"])
        fn = sum(1 for r in group_rows
                 if r["predicted_grade"] not in AT_RISK_GRADES and r["truth_at_risk"])
        entry["atrisk_precision"] = (tp / preds_at_risk) if preds_at_risk > 0 else None
        entry["atrisk_recall"] = (tp / (tp + fn)) if (tp + fn) > 0 else None
        summary.append(entry)

    return summary


def print_table(summary, group_keys):
    if not summary:
        print("  (no data)")
        return

    header_keys = group_keys + ["n", "accuracy", "ece", "brier", "atrisk_precision", "atrisk_recall"]
    col_widths = {k: max(len(k), 10) for k in header_keys}
    for row in summary:
        for k in header_keys:
            v = row.get(k)
            s = f"{v:.3f}" if isinstance(v, float) else str(v)
            col_widths[k] = max(col_widths[k], len(s))

    header = "  ".join(k.ljust(col_widths[k]) for k in header_keys)
    print(header)
    print("  ".join("-" * col_widths[k] for k in header_keys))
    for row in summary:
        parts = []
        for k in header_keys:
            v = row.get(k)
            if v is None:
                s = "-"
            elif isinstance(v, float):
                s = f"{v:.3f}"
            else:
                s = str(v)
            parts.append(s.ljust(col_widths[k]))
        print("  ".join(parts))


def save_csv(rows, path, fieldnames):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# PLOT
# =============================================================================

def plot_accuracy_and_ece(summary, output_path):
    """
    summary: list of dicts with keys dataset, feature_set, pct_bin, n, accuracy, ece
    Produces a 2-panel stacked PNG: accuracy on top, ECE on bottom.
    """
    import matplotlib.pyplot as plt

    # Group by (dataset, feature_set) -> list of (pct_midpoint, acc, ece)
    lines = defaultdict(list)
    for row in summary:
        mid = pct_bin_midpoint(row["pct_bin"])
        lines[(row["dataset"], row["feature_set"])].append(
            (mid, row["accuracy"], row["ece"])
        )

    for key in lines:
        lines[key].sort(key=lambda x: x[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    colors = {"predact_cs": "#1f77b4", "oulad": "#d62728"}
    linestyles = {"full": "-", "minimal": "--"}

    for (dataset, feature_set), points in lines.items():
        xs = [p[0] for p in points]
        accs = [p[1] for p in points if p[1] is not None]
        eces = [p[2] for p in points if p[2] is not None]
        xs_acc = [p[0] for p in points if p[1] is not None]
        xs_ece = [p[0] for p in points if p[2] is not None]

        label = f"{dataset.upper()} ({feature_set})"
        color = colors.get(dataset, "black")
        style = linestyles.get(feature_set, "-")

        ax1.plot(xs_acc, accs, marker="o", linestyle=style, color=color, label=label)
        ax2.plot(xs_ece, eces, marker="o", linestyle=style, color=color, label=label)

    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title("k-NN Tool Characterization across the Semester")

    ax2.set_ylabel("ECE (lower is better)")
    ax2.set_xlabel("Percent of semester completed")
    ax2.grid(True, alpha=0.3)

    ax2.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-courses", type=int, default=None)
    parser.add_argument("--feature-sets", nargs="+", default=["full"],
                        choices=["minimal", "full"])
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_ROOT, "exp1"))
    parser.add_argument("--datasets", nargs="+", default=["predact_cs", "oulad"],
                        choices=["predact_cs", "oulad"])
    args = parser.parse_args()

    all_rows = []
    for dataset_cfg in DATASETS:
        if dataset_cfg["name"] not in args.datasets:
            continue
        print(f"\n{'=' * 70}")
        print(f"Running Exp 1 on: {dataset_cfg['name'].upper()}")
        print(f"{'=' * 70}")
        rows = run_predictions(
            dataset_cfg,
            feature_sets=args.feature_sets,
            limit_courses=args.limit_courses,
            verbose=True,
        )
        print(f"  Total predictions: {len(rows)}")
        all_rows.extend(rows)

    # Save raw CSV
    raw_fields = [
        "dataset", "course_id", "week", "pct_complete", "pct_bin", "feature_set",
        "student_id", "predicted_grade", "confidence", "truth_grade",
        "correct", "truth_at_risk", "at_risk_prob",
    ]
    raw_path = os.path.join(args.output_dir, "exp1_raw.csv")
    save_csv(all_rows, raw_path, raw_fields)
    print(f"\nRaw predictions saved to: {raw_path}")

    # Summary by dataset × feature_set (overall)
    print(f"\n{'=' * 70}")
    print("SUMMARY: Dataset × Feature Set (overall)")
    print("=" * 70)
    summary_overall = aggregate(all_rows, ["dataset", "feature_set"])
    print_table(summary_overall, ["dataset", "feature_set"])

    # Summary by dataset × feature_set × pct_bin
    print(f"\n{'=' * 70}")
    print("SUMMARY: Dataset × Feature Set × Percent Complete")
    print("=" * 70)
    summary_by_pct = aggregate(all_rows, ["dataset", "feature_set", "pct_bin"])

    # Sort pct_bin numerically for display
    summary_by_pct.sort(key=lambda r: (r["dataset"], r["feature_set"], pct_bin_midpoint(r["pct_bin"])))
    print_table(summary_by_pct, ["dataset", "feature_set", "pct_bin"])

    # Save summary CSV
    sum_fields = ["dataset", "feature_set", "pct_bin", "n",
                  "accuracy", "ece", "brier", "atrisk_precision", "atrisk_recall"]
    sum_path = os.path.join(args.output_dir, "exp1_summary.csv")

    summary_combined = []
    for row in summary_overall:
        r = dict(row)
        r["pct_bin"] = "all"
        summary_combined.append(r)
    summary_combined.extend(summary_by_pct)
    save_csv(summary_combined, sum_path, sum_fields)
    print(f"\nSummary saved to: {sum_path}")

    # Plot
    plot_path = os.path.join(args.output_dir, "exp1_plot.png")
    plot_accuracy_and_ece(summary_by_pct, plot_path)

    print(f"\nDone.")


if __name__ == "__main__":
    main()