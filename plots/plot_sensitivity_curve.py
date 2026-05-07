"""plot_sensitivity_curve.py — class-separation density figure for PredAct paper.

Shows how well the AI tool's predicted at-risk probability separates students
who actually passed (yellow) vs. those who actually failed (blue), at the
~60%-accuracy operating point used by Exp 2.

Cells (matching Exp 2 60%-target-accuracy anchors):
  PredAct-CS  →  Course_B, week 8
  OULAD       →  AAA_2013J, week 20

Data source : results/exp1/exp1_raw.csv  (columns: dataset, course_id, week,
              at_risk_prob, truth_at_risk, ...)
Outputs     : figures/sensitivity_curve.{pdf,png}
              figures/sensitivity_curve_note.md   (AUC values for paper text)
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

CSV_PATH = os.path.join(PROJECT_ROOT, "results/exp1/exp1_raw.csv")
OUT_DIR  = os.path.join(PROJECT_ROOT, "figures")

CELLS = [
    {"label": "PredAct-CS  (Course_B, week 8)",
     "dataset": "PredAct-CS", "course_id": "Course_B", "week": 8},
    {"label": "OULAD  (AAA_2013J, week 20)",
     "dataset": "oulad",      "course_id": "AAA_2013J", "week": 20},
]
COLOR_PASS = "#F4C430"   # soft yellow — not at-risk
COLOR_FAIL = "#5B9BD5"   # soft blue   — at-risk

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_cell(df, cell):
    sub = df[(df["dataset"] == cell["dataset"])
             & (df["course_id"] == cell["course_id"])
             & (df["week"] == cell["week"])].dropna(subset=["at_risk_prob",
                                                            "truth_at_risk"])
    return sub["at_risk_prob"].values, sub["truth_at_risk"].astype(int).values


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thr[best]), float(tpr[best]), float(fpr[best])


def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True,
                             sharex=True, gridspec_kw={"wspace": 0.10})

    note_lines = ["# Sensitivity curve — AUC values\n"]
    for ax, cell in zip(axes, CELLS):
        scores, y = load_cell(df, cell)
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        auc = roc_auc_score(y, scores)
        thr, tpr, fpr = youden_threshold(y, scores)

        sns.kdeplot(x=scores[y == 0], ax=ax, fill=True, alpha=0.6,
                    color=COLOR_PASS, linewidth=1.4, clip=(0, 1),
                    label=f"Not at-risk  (n={n_neg})")
        sns.kdeplot(x=scores[y == 1], ax=ax, fill=True, alpha=0.6,
                    color=COLOR_FAIL, linewidth=1.4, clip=(0, 1),
                    label=f"At-risk  (n={n_pos})")

        ax.axvline(thr, color="#444", linestyle="--", linewidth=1.2)
        ax.text(thr + 0.012, ax.get_ylim()[1] * 0.92,
                f"Youden threshold\n= {thr:.2f}",
                fontsize=8.5, color="#333", va="top")
        ax.text(0.97, 0.97, f"AUC = {auc:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#888", linewidth=0.6))
        ax.text(0.50, -0.18,
                "<-- favor recall  |  favor precision -->",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#666", style="italic")

        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted at-risk probability", fontsize=11)
        ax.set_title(cell["label"], fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper center", frameon=False, fontsize=10)

        note_lines.append(
            f"- **{cell['label']}**: AUC = {auc:.3f}, "
            f"Youden threshold = {thr:.2f} "
            f"(TPR={tpr:.2f}, FPR={fpr:.2f}, n={len(y)} students)"
        )

    axes[0].set_ylabel("Density", fontsize=11)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, "sensitivity_curve.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()

    note_path = os.path.join(OUT_DIR, "sensitivity_curve_note.md")
    with open(note_path, "w") as f:
        f.write("\n".join(note_lines) + "\n")

    print(f"Saved → {out_pdf}")
    print(f"Saved → {note_path}")
    for line in note_lines[1:]:
        print(line)


if __name__ == "__main__":
    main()
