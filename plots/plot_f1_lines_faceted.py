"""
Two-panel faceted line plot: F1 vs Target Tool Accuracy
Left panel = PredAct-CS, Right panel = OULAD
One line per model (13 models), colored by model family.
Data: results/exp2/exp2_per_cell.csv
"""
import csv
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

CSV_PATH = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
OUT_PNG  = os.path.join(PROJECT_ROOT, "figures/f1_lines_faceted.png")
OUT_PDF  = os.path.join(PROJECT_ROOT, "figures/f1_lines_faceted.pdf")

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ── model display names ────────────────────────────────────────────────────────
DISPLAY = {
    "gpt5_5":            "GPT-5.5",
    "gpt5_4_mini":       "GPT-5.4 Mini",
    "gpt4o_mini":        "GPT-4o Mini",
    "claude_opus_4_7":   "Claude Opus 4.7",
    "claude_haiku_4_5":  "Claude Haiku 4.5",
    "gemini_3_1_pro":    "Gemini 3.1 Pro",
    "gemini_3_flash":    "Gemini 3 Flash",
    "deepseek_v4_pro":   "DeepSeek V4 Pro",
    "deepseek_v4_flash": "DeepSeek V4 Flash",
    "mistral_small_24b": "Mistral Small 24B",
    "ministral_3_14b":   "Ministral 3 14B",
    "qwen_35b":          "Qwen 35B",
    "qwen_9b":           "Qwen 9B",
}

# ── family colors + markers ────────────────────────────────────────────────────
FAMILIES = {
    "gpt5_5":            ("#C0392B", "o"),   # GPT  — red shades
    "gpt5_4_mini":       ("#E74C3C", "o"),
    "gpt4o_mini":        ("#F1948A", "o"),
    "claude_opus_4_7":   ("#E67E22", "s"),   # Claude — orange shades
    "claude_haiku_4_5":  ("#F0B27A", "s"),
    "gemini_3_1_pro":    ("#2980B9", "^"),   # Gemini — blue shades
    "gemini_3_flash":    ("#85C1E9", "^"),
    "deepseek_v4_pro":   ("#27AE60", "D"),   # DeepSeek — green shades
    "deepseek_v4_flash": ("#82E0AA", "D"),
    "mistral_small_24b": ("#8E44AD", "P"),   # Mistral — purple shades
    "ministral_3_14b":   ("#C39BD3", "P"),
    "qwen_35b":          ("#7F6000", "X"),   # Qwen — brown shades
    "qwen_9b":           ("#C9AC6A", "X"),
}

# ── load CSV ───────────────────────────────────────────────────────────────────
data = {}   # data[dataset][model][target_acc] = (f1_mean, f1_std)
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        ds  = row["dataset"]
        m   = row["instructor_llm"]
        t   = float(row["target_accuracy"])
        mu  = float(row["f1_final_mean"])
        std = float(row["f1_final_std"])
        data.setdefault(ds, {}).setdefault(m, {})[t] = (mu, std)

DATASET_LABEL = {"predact_cs": "PredAct-CS", "oulad": "OULAD"}
datasets = ["predact_cs", "oulad"]   # left → right
models   = list(DISPLAY.keys())
targets  = sorted({t for ds in data.values() for m in ds.values() for t in m})
x_ticks  = [f"{int(t*100)}%" for t in targets]

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.subplots_adjust(wspace=0.06)

for ax, ds in zip(axes, datasets):
    ds_data = data.get(ds, {})
    for m in models:
        m_data = ds_data.get(m, {})
        if not m_data:
            continue
        color, marker = FAMILIES[m]
        ys   = [m_data[t][0] if t in m_data else np.nan for t in targets]
        errs = [m_data[t][1] if t in m_data else 0      for t in targets]
        ax.plot(targets, ys, color=color, marker=marker,
                markersize=6, linewidth=1.8, zorder=3)
        ax.fill_between(targets,
                        [y - e for y, e in zip(ys, errs)],
                        [y + e for y, e in zip(ys, errs)],
                        color=color, alpha=0.10, zorder=2)

    ax.set_xticks(targets)
    ax.set_xticklabels(x_ticks, fontsize=11)
    ax.set_xlabel("Target Tool Accuracy", fontsize=12)
    ax.set_title(DATASET_LABEL[ds], fontsize=13, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_tick_params(labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.grid(axis="x", linestyle=":",  linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("F1", fontsize=12)

# ── single shared legend below both panels ─────────────────────────────────────
legend_handles = [
    mlines.Line2D([], [], color=FAMILIES[m][0], marker=FAMILIES[m][1],
                  markersize=6, linewidth=1.8, label=DISPLAY[m])
    for m in models
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=7,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.12),
    columnspacing=1.2,
    handletextpad=0.5,
)

plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()
print(f"Saved → {OUT_PNG}")
print(f"Saved → {OUT_PDF}")
