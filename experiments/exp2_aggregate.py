"""
Exp 2 — aggregate per-cell metrics across all runs.

Walks SIM_LOGS_ROOT, reads every episode log, computes F1/RAIR/RSR per log
(reusing sim/evaluate_episode.py's evaluate_one), then groups by
(instructor_llm, dataset, target_accuracy) and reports mean ± std.

Usage:
    python -m experiments.exp2_aggregate
    python -m experiments.exp2_aggregate --csv results/exp2/exp2_per_cell.csv
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

from experiments.exp2_config import AGGREGATE_CSV, SIM_LOGS_ROOT
from sim.evaluate_episode import evaluate_one


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def collect_logs(logs_root):
    """Yield (path, log) for every .json under logs_root."""
    for path in sorted(glob.glob(os.path.join(logs_root, "**", "*.json"), recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield path, json.load(f)
        except Exception as e:
            print(f"  SKIP {path}: {e}")


def aggregate(logs_root):
    """Group reports by (instructor, dataset, accuracy) and compute summary stats."""
    buckets = defaultdict(list)  # key -> list of evaluate_one(report)

    n_total = 0
    for path, log in collect_logs(logs_root):
        n_total += 1
        report = evaluate_one(log)
        key = (
            log.get("instructor_llm"),
            log.get("dataset"),
            log.get("target_accuracy"),
        )
        buckets[key].append(report)

    rows = []
    for (instructor, dataset, acc), reports in sorted(buckets.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2] or 0)):
        n = len(reports)

        f1_initial = [r["initial"]["f1"] if r["initial"] else None for r in reports]
        f1_final   = [r["final"]["f1"] for r in reports]
        prec_final = [r["final"]["precision"] for r in reports]
        rec_final  = [r["final"]["recall"] for r in reports]
        rair       = [r["trajectory"]["trajectory_rair"] for r in reports]
        rsr        = [r["trajectory"]["trajectory_rsr"]  for r in reports]

        f1_init_mean, f1_init_std = _mean_std(f1_initial)
        f1_fin_mean, f1_fin_std   = _mean_std(f1_final)
        prec_mean, prec_std       = _mean_std(prec_final)
        rec_mean, rec_std         = _mean_std(rec_final)
        rair_mean, rair_std       = _mean_std(rair)
        rsr_mean, rsr_std         = _mean_std(rsr)

        rows.append({
            "instructor_llm": instructor,
            "dataset": dataset,
            "target_accuracy": acc,
            "n_runs": n,
            "f1_initial_mean": f1_init_mean,
            "f1_initial_std":  f1_init_std,
            "f1_final_mean":   f1_fin_mean,
            "f1_final_std":    f1_fin_std,
            "precision_final_mean": prec_mean,
            "precision_final_std":  prec_std,
            "recall_final_mean":    rec_mean,
            "recall_final_std":     rec_std,
            "rair_mean": rair_mean,
            "rair_std":  rair_std,
            "rsr_mean":  rsr_mean,
            "rsr_std":   rsr_std,
        })

    return rows, n_total


def print_table(rows):
    if not rows:
        print("No rows.")
        return
    print()
    print(f"{'instructor':<20}{'dataset':<8}{'acc':<6}{'n':<5}"
          f"{'F1(init)':<12}{'F1(final)':<13}{'Prec':<10}{'Rec':<10}"
          f"{'RAIR':<10}{'RSR':<10}")
    print("-" * 110)
    for r in rows:
        def fmt(mean, std):
            if mean is None: return "-".ljust(11)
            return f"{mean:.3f}±{std:.2f}"
        print(
            f"{r['instructor_llm'] or '?':<20}"
            f"{r['dataset'] or '?':<8}"
            f"{r['target_accuracy'] or 0:<6.2f}"
            f"{r['n_runs']:<5}"
            f"{fmt(r['f1_initial_mean'], r['f1_initial_std']):<12}"
            f"{fmt(r['f1_final_mean'], r['f1_final_std']):<13}"
            f"{fmt(r['precision_final_mean'], r['precision_final_std']):<10}"
            f"{fmt(r['recall_final_mean'], r['recall_final_std']):<10}"
            f"{fmt(r['rair_mean'], r['rair_std']):<10}"
            f"{fmt(r['rsr_mean'], r['rsr_std']):<10}"
        )


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=SIM_LOGS_ROOT)
    parser.add_argument("--csv", default=AGGREGATE_CSV)
    args = parser.parse_args()

    if not os.path.isdir(args.logs):
        print(f"No logs directory at {args.logs}")
        return

    print(f"Reading logs from {args.logs}")
    rows, n_total = aggregate(args.logs)
    print(f"Read {n_total} log files; aggregated into {len(rows)} cells.")
    print_table(rows)
    write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
