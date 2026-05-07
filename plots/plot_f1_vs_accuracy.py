"""
F1 vs Target Tool Accuracy — clean faceted line plot
Two panels: PredAct-CS (left), OULAD (right)
Data: results/exp2/exp2_per_cell.csv
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

CSV_PATH = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
OUT_DIR  = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── confirm columns ────────────────────────────────────────────────────────────
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("columns:", list(rows[0].keys()))
print("sample rows:")
for r in rows[:3]:
    print(" ", {k: r[k] for k in ["instructor_llm", "dataset", "target_accuracy", "f1_final_mean"]})

# ── style map ──────────────────────────────────────────────────────────────────
# Each entry: display label, color, linestyle, marker
MODEL_STYLE = {
    # OpenAI — deep red
    "gpt5_5":            ("GPT-5.5",           "#b03030", "-",  "o"),
    "gpt5_4_mini":       ("GPT-5.4 Mini",       "#b03030", "--", "^"),
    "gpt4o_mini":        ("GPT-4o Mini",         "#b03030", ":",  "s"),
    # Anthropic — muted orange
    "claude_opus_4_7":   ("Claude Opus 4.7",    "#c4795b", "-",  "o"),
    "claude_haiku_4_5":  ("Claude Haiku 4.5",   "#c4795b", ":",  "s"),
    # Google — slate blue
    "gemini_3_1_pro":    ("Gemini 3.1 Pro",     "#5a7fa3", "-",  "o"),
    "gemini_3_flash":    ("Gemini 3 Flash",      "#5a7fa3", ":",  "s"),
    # DeepSeek — forest green
    "deepseek_v4_pro":   ("DeepSeek V4 Pro",    "#4a7340", "-",  "o"),
    "deepseek_v4_flash": ("DeepSeek V4 Flash",  "#4a7340", ":",  "s"),
    # Mistral — dusty purple
    "mistral_small_24b": ("Mistral Small 24B",  "#7d6b9e", "-",  "o"),
    "ministral_3_14b":   ("Ministral 3 14B",    "#7d6b9e", ":",  "s"),
    # Qwen — warm brown
    "qwen_35b":          ("Qwen 35B",           "#8e6f4a", "-",  "o"),
    "qwen_9b":           ("Qwen 9B",            "#8e6f4a", ":",  "s"),
}

# ── load data ──────────────────────────────────────────────────────────────────
data = {}  # data[dataset][model][target] = f1_mean
for r in rows:
    ds  = r["dataset"]
    m   = r["instructor_llm"]
    t   = float(r["target_accuracy"])
    f1  = float(r["f1_final_mean"])
    data.setdefault(ds, {}).setdefault(m, {})[t] = f1

DATASETS    = [("predact_cs", "PredAct-CS"), ("oulad", "OULAD")]
targets     = sorted({t for ds in data.values() for m in ds.values() for t in m})
x_labels    = [f"{int(t*100)}%" for t in targets]

# ── figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif"})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.subplots_adjust(wspace=0.06)

for ax, (ds_key, ds_title) in zip(axes, DATASETS):
    ds_data = data.get(ds_key, {})

    for m_key, (label, color, ls, marker) in MODEL_STYLE.items():
        m_data = ds_data.get(m_key, {})
        if not m_data:
            continue
        ys = [m_data.get(t, np.nan) for t in targets]
        ax.plot(
            targets, ys,
            color=color, linestyle=ls, marker=marker,
            linewidth=1.8, markersize=6,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=3,
        )

    # axes styling
    ax.set_xticks(targets)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Target Tool Accuracy", fontsize=11)
    ax.set_title(ds_title, fontsize=12, fontweight="bold", pad=6)

    ax.yaxis.grid(True, color="#dddddd", alpha=0.4, linewidth=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#555555")

axes[0].set_ylabel("F1", fontsize=11)

# ── shared legend ──────────────────────────────────────────────────────────────
legend_handles = [
    mlines.Line2D([], [],
                  color=color, linestyle=ls, marker=marker,
                  linewidth=1.8, markersize=6,
                  markeredgecolor="white", markeredgewidth=0.7,
                  label=label)
    for _, (label, color, ls, marker) in MODEL_STYLE.items()
]

plt.tight_layout()
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.18),
    columnspacing=1.4,
    handletextpad=0.6,
    handlelength=2.2,
)
fig.subplots_adjust(bottom=0.28)

out_png = os.path.join(OUT_DIR, "f1_vs_accuracy.png")
out_pdf = os.path.join(OUT_DIR, "f1_vs_accuracy.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_png}")
print(f"Saved → {out_pdf}")
