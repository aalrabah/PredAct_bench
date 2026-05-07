"""
Generate exp2_with_overall.csv — same as exp2_per_cell.csv but with an
extra "overall" row per model (dataset="overall", target_accuracy=NaN)
where each metric's mean and std are computed across the per-cell means.
"""
import math
import os
import pandas as pd

IN_CSV  = "results/exp2/exp2_per_cell.csv"
OUT_CSV = "results/exp2/exp2_with_overall.csv"

MEAN_COLS = [
    "f1_initial_mean", "f1_final_mean", "precision_final_mean",
    "recall_final_mean", "rair_mean", "rsr_mean",
]
STD_COLS = [
    "f1_initial_std", "f1_final_std", "precision_final_std",
    "recall_final_std", "rair_std", "rsr_std",
]


def mean_std(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def main():
    df = pd.read_csv(IN_CSV)
    overall_rows = []

    for model, grp in df.groupby("instructor_llm"):
        row = {"instructor_llm": model, "dataset": "overall",
               "target_accuracy": float("nan"), "n_runs": len(grp)}
        for mc, sc in zip(MEAN_COLS, STD_COLS):
            m, s = mean_std(grp[mc].tolist())
            row[mc] = m
            row[sc] = s
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows, columns=df.columns)
    out = pd.concat([df, overall_df], ignore_index=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Written: {OUT_CSV}  ({len(out)} rows = {len(df)} cell + {len(overall_rows)} overall)")


if __name__ == "__main__":
    main()
