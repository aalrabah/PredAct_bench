"""
Generate human_study_per_cell.csv — per (llm, target_accuracy) and overall rows
with mean and std of final_f1 across participants.

Uses population std (np.std, divide by n) to match evaluate_human_study_v2.py line 264.

Output: results/human_study_per_cell.csv
"""
import os
import csv
import numpy as np

IN_CSV  = "results_per_participant.csv"
OUT_CSV = "results/human_study_per_cell.csv"


def mean_std(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))  # population std, matches existing code


def main():
    rows = []
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by (llm, target_accuracy)
    # no_agent rows have llm="" and target_accuracy=""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        llm = r["llm"] if r["llm"] else "no_agent"
        acc = r["target_accuracy"] if r["target_accuracy"] else None
        try:
            f1 = float(r["final_f1"])
        except (ValueError, TypeError):
            f1 = None
        buckets[(llm, acc)].append(f1)

    # Also build per-llm overall buckets (all participants across all accuracies)
    llm_all = defaultdict(list)
    for (llm, acc), vals in buckets.items():
        llm_all[llm].extend(vals)

    out_rows = []
    fieldnames = ["llm", "target_accuracy", "n", "f1_mean", "f1_std"]

    # Per-cell rows
    for (llm, acc), vals in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1] or "")):
        m, s = mean_std(vals)
        out_rows.append({
            "llm": llm,
            "target_accuracy": acc if acc else "all",
            "n": len([v for v in vals if v is not None]),
            "f1_mean": round(m * 100, 2) if m is not None else None,
            "f1_std":  round(s * 100, 2) if s is not None else None,
        })

    # Overall rows per llm
    for llm, vals in sorted(llm_all.items()):
        m, s = mean_std(vals)
        out_rows.append({
            "llm": llm,
            "target_accuracy": "overall",
            "n": len([v for v in vals if v is not None]),
            "f1_mean": round(m * 100, 2) if m is not None else None,
            "f1_std":  round(s * 100, 2) if s is not None else None,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Written: {OUT_CSV}  ({len(out_rows)} rows)")
    print()
    # Pretty print
    print(f"{'llm':<20} {'acc':<10} {'n':<4} {'f1_mean':>8} {'f1_std':>8}")
    print("-" * 55)
    for r in sorted(out_rows, key=lambda x: (x["llm"], x["target_accuracy"])):
        m = f"{r['f1_mean']:.1f}" if r['f1_mean'] is not None else "---"
        s = f"{r['f1_std']:.1f}"  if r['f1_std']  is not None else "---"
        print(f"{r['llm']:<20} {str(r['target_accuracy']):<10} {r['n']:<4} {m:>8} {s:>8}")


if __name__ == "__main__":
    main()
