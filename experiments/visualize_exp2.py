"""
Exp 2 — visualization.

Reads the per-cell CSV from exp2_aggregate.py and plots three panels
(F1-final, Trajectory-RAIR, Trajectory-RSR) vs target accuracy on the
x-axis, one curve per instructor LLM, faceted by dataset (PredAct-CS | OULAD).

Usage:
    python -m experiments.visualize_exp2
    python -m experiments.visualize_exp2 --csv results/exp2/exp2_per_cell.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from experiments.exp2_config import AGGREGATE_CSV, EXP2_RESULTS_ROOT


METRICS = [
    ("f1_final_mean",   "f1_final_std",   "F1 (final decisions)"),
    ("rair_mean",       "rair_std",       "Trajectory-RAIR"),
    ("rsr_mean",        "rsr_std",        "Trajectory-RSR"),
]

DATASETS = ["predact_cs", "oulad"]


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if v in ("", None):
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(r)
    return rows


def plot(rows, out_path):
    if not rows:
        print("No data to plot.")
        return

    # Group by (dataset, instructor) → list of (acc, mean, std) for each metric
    fig, axes = plt.subplots(len(METRICS), len(DATASETS),
                              figsize=(6 * len(DATASETS), 4 * len(METRICS)),
                              squeeze=False)

    for col, dataset in enumerate(DATASETS):
        ds_rows = [r for r in rows if r.get("dataset") == dataset]
        instructors = sorted({r["instructor_llm"] for r in ds_rows if r.get("instructor_llm")})

        for row_i, (mean_key, std_key, title) in enumerate(METRICS):
            ax = axes[row_i][col]
            for inst in instructors:
                inst_rows = sorted(
                    [r for r in ds_rows if r["instructor_llm"] == inst],
                    key=lambda r: r["target_accuracy"] or 0,
                )
                xs   = [r["target_accuracy"] for r in inst_rows]
                ys   = [r.get(mean_key) for r in inst_rows]
                errs = [r.get(std_key)  for r in inst_rows]
                # Drop None pairs
                xys = [(x, y, e) for x, y, e in zip(xs, ys, errs) if x is not None and y is not None]
                if not xys:
                    continue
                xs2 = [p[0] for p in xys]
                ys2 = [p[1] for p in xys]
                es2 = [p[2] if p[2] is not None else 0 for p in xys]
                ax.errorbar(xs2, ys2, yerr=es2, marker="o", capsize=3, label=inst)

            ax.set_title(f"{dataset.upper()} — {title}")
            ax.set_xlabel("Target tool accuracy")
            ax.set_ylabel(title.split(" ")[0])
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            if row_i == 0 and col == 0:
                ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=AGGREGATE_CSV)
    parser.add_argument("--out", default=os.path.join(EXP2_RESULTS_ROOT, "exp2_curves.png"))
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"No aggregate CSV at {args.csv}. Run exp2_aggregate.py first.")
        return

    rows = load_rows(args.csv)
    print(f"Loaded {len(rows)} cells from {args.csv}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
