"""
Redesigned Likert forest plot — wide landscape, publication-ready.
Reads from results/likert_summary.csv, saves figures/likert_redesign.png/.pdf
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import csv

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        10,
    "axes.linewidth":   0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "likert_summary.csv")
OUT_DIR  = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
data = {}   # data[(metric, condition)] = (mean, sd)
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        data[(row["metric"], row["condition"])] = (float(row["mean"]), float(row["sd"]))

# ── layout config ──────────────────────────────────────────────────────────────
QUESTIONS = [
    ("Decision confidence",       "Decision\nConfidence"),
    ("Useful collaborator",       "Useful\nCollaborator"),
    ("Could tell when wrong",     "Could Tell\nWhen Wrong"),
    ("Trust scaled with accuracy","Trust Scaled\nw/ Accuracy"),
    ("Would deploy in own courses","Would\nDeploy"),
]

CONDITIONS = [
    ("No Agent",         "No Agent",       "#94A3B8", "s", 6.5, 0.8),   # gray-blue square
    ("Agent 1",          "GPT-4o Mini",    "#3B82F6", "o", 7.0, 0.85),  # blue circle
    ("Agent 2",          "Qwen-9B",        "#F59E0B", "o", 7.0, 0.85),  # amber circle
    ("Agent 3",          "Qwen-35B",       "#10B981", "o", 7.0, 0.85),  # emerald circle
    ("Overall",          "Overall (agents)","#1E293B","D", 8.5, 1.0),   # dark diamond
    ("Overall (agents)", "Overall (agents)","#1E293B","D", 8.5, 1.0),
]
# deduplicate by condition key
seen = set()
COND_LIST = []
for ckey, label, color, marker, ms, alpha in CONDITIONS:
    if ckey not in seen:
        seen.add(ckey)
        COND_LIST.append((ckey, label, color, marker, ms, alpha))

# y-offsets within each question group so points don't overlap
N_CONDS = len(COND_LIST)
offsets = np.linspace(0.28, -0.28, N_CONDS)

# ── figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

n_q = len(QUESTIONS)
x_positions = np.arange(n_q)

# subtle column shading for every other question
for i in range(n_q):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#F8FAFC", zorder=0)

# neutral reference line
ax.axhline(3, color="#CBD5E1", linewidth=0.9,
           linestyle=(0, (6, 3)), zorder=1, label="_nolegend_")

# plot each condition for each question
for qi, (metric_key, _) in enumerate(QUESTIONS):
    for ci, (ckey, label, color, marker, ms, alpha) in enumerate(COND_LIST):
        key = (metric_key, ckey)
        if key not in data:
            continue
        mu, sd = data[key]
        y  = mu
        x  = qi + offsets[ci]

        is_overall = ckey in ("Overall", "Overall (agents)")
        lw   = 1.8 if is_overall else 1.0
        ealpha = 0.55 if is_overall else 0.40
        zord = 5 if is_overall else 4

        ax.errorbar(
            x, y, yerr=sd,
            fmt=marker,
            color=color,
            markersize=ms,
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=lw,
            capsize=3,
            capthick=lw,
            alpha=alpha,
            zorder=zord,
        )

# ── axes ───────────────────────────────────────────────────────────────────────
ax.set_xticks(x_positions)
ax.set_xticklabels(
    [label for _, label in QUESTIONS],
    fontsize=10.5, fontweight="500",
)
ax.set_xlim(-0.6, n_q - 0.4)

ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels([
    "1\nStr. Disagree", "2", "3\nNeutral", "4", "5\nStr. Agree"
], fontsize=8.5)
ax.set_ylim(0.5, 5.7)
ax.set_ylabel("Likert Response", fontsize=11, labelpad=8)

ax.tick_params(axis="x", length=0, pad=8)
ax.tick_params(axis="y", length=3, color="#94A3B8")

ax.grid(axis="y", color="#E2E8F0", linewidth=0.6, zorder=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#CBD5E1")
ax.spines["bottom"].set_linewidth(0.7)

ax.set_title(
    "Participant Survey Responses by Condition  (mean ± SD,  n = 13)",
    fontsize=12.5, fontweight="600", pad=14, loc="left", color="#1E293B",
)

# neutral label
ax.text(
    n_q - 0.38, 3.06, "Neutral (3)",
    fontsize=7.5, color="#94A3B8", va="bottom", ha="right",
    style="italic",
)

# ── legend ─────────────────────────────────────────────────────────────────────
legend_specs = [
    ("No Agent",          "#94A3B8", "s"),
    ("GPT-4o Mini",       "#3B82F6", "o"),
    ("Qwen-9B",           "#F59E0B", "o"),
    ("Qwen-35B",          "#10B981", "o"),
    ("Overall (agents)",  "#1E293B", "D"),
]
handles = [
    mlines.Line2D([], [],
                  marker=mk, color=col, linestyle="none",
                  markersize=7 if mk != "D" else 8,
                  markeredgecolor="white", markeredgewidth=0.8,
                  label=lbl,
                  alpha=0.9 if lbl != "Overall (agents)" else 1.0)
    for lbl, col, mk in legend_specs
]
ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=9,
    frameon=True,
    edgecolor="#E2E8F0",
    framealpha=0.95,
    handletextpad=0.6,
    labelspacing=0.45,
    borderpad=0.7,
    ncol=1,
)

plt.tight_layout(pad=1.2)

out_png = os.path.join(OUT_DIR, "likert_redesign.png")
out_pdf = os.path.join(OUT_DIR, "likert_redesign.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
