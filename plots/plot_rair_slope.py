"""plot_rair_slope.py — RAIR-by-cutoff slope figure for the PredAct paper."""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

CSV_PATH = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
OUT_DIR  = os.path.join(PROJECT_ROOT, "figures")
RSR_GATE = 0.95
CUTOFFS  = [0.4, 0.5, 0.6, 0.7, 0.8]

MODELS = {  # llm_key: (color, linestyle)
    "gpt5_5":            ("#534AB7", "-"),
    "gpt5_4_mini":       ("#5F5E5A", "--"),
    "gpt4o_mini":        ("#888780", "-"),
    "claude_opus_4_7":   ("#185FA5", "-"),
    "claude_haiku_4_5":  ("#378ADD", "--"),
    "gemini_3_1_pro":    ("#0F6E56", "--"),
    "gemini_3_flash":    ("#1D9E75", "-"),
    "deepseek_v4_pro":   ("#D85A30", "-"),
    "deepseek_v4_flash": ("#F0997B", "--"),
    "qwen_35b":          ("#D4537E", "-"),
    "qwen_9b":           ("#ED93B1", "--"),
    "ministral_3_14b":   ("#BA7517", "-"),
    "mistral_small_24b": ("#854F0B", "--"),
}
DISPLAY = {
    "gpt5_5": "GPT-5.5", "gpt5_4_mini": "GPT-5.4 Mini", "gpt4o_mini": "GPT-4o Mini",
    "claude_opus_4_7": "Claude Opus 4.7", "claude_haiku_4_5": "Claude Haiku 4.5",
    "gemini_3_1_pro": "Gemini 3.1 Pro", "gemini_3_flash": "Gemini 3 Flash",
    "deepseek_v4_pro": "DeepSeek V4 Pro", "deepseek_v4_flash": "DeepSeek V4 Flash",
    "qwen_35b": "Qwen 35B", "qwen_9b": "Qwen 9B",
    "ministral_3_14b": "Ministral 3 14B", "mistral_small_24b": "Mistral Small 24B",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "lines.antialiased": True, "path.simplify": True,
})

LW = 1.5
LINE_KW = dict(lw=LW, solid_capstyle="round", solid_joinstyle="round",
               dash_capstyle="round")

df = pd.read_csv(CSV_PATH)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True,
                         gridspec_kw={"wspace": 0.12})

for (ds, label), ax in zip([("predact_cs", "PredAct - CS"), ("oulad", "OULAD")], axes):
    sub = df[df["dataset"] == ds]
    for m, (color, ls) in MODELS.items():
        ms = sub[sub["instructor_llm"] == m].set_index("target_accuracy")
        rair = np.array([ms.loc[c, "rair_mean"] if c in ms.index else np.nan
                         for c in CUTOFFS])
        rsr  = np.array([ms.loc[c, "rsr_mean"] if c in ms.index else np.nan
                         for c in CUTOFFS])
        # HTML-style smooth: PCHIP through the valid points (monotone cubic, no overshoot).
        valid = ~np.isnan(rair)
        if valid.sum() >= 2:
            xs_v = np.array(CUTOFFS)[valid]
            ys_v = rair[valid]
            xs_dense = np.linspace(xs_v.min(), xs_v.max(), 200)
            ys_dense = PchipInterpolator(xs_v, ys_v)(xs_dense)
            ax.plot(xs_dense, ys_dense, color=color, linestyle=ls, **LINE_KW)
        else:
            ax.plot(CUTOFFS, rair, color=color, linestyle=ls, **LINE_KW)
        gate_fail = (~np.isnan(rsr)) & (~np.isnan(rair)) & (rsr < RSR_GATE)
        if gate_fail.any():
            ax.plot(np.array(CUTOFFS)[gate_fail], rair[gate_fail],
                    marker="o", markersize=5, linestyle="None", zorder=3,
                    markerfacecolor="#E24B4A", markeredgecolor="#791F1F",
                    clip_on=False)
    ax.set_ylim(0, 1.02); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(CUTOFFS[0], CUTOFFS[-1])
    ax.set_xticks(CUTOFFS); ax.set_xlabel("Target Accuracy Cutoff")
    ax.text(0.5, 0.96, label, transform=ax.transAxes, fontsize=13,
            ha="center", va="top")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axes[0].set_ylabel("RAIR (mean)")

handles = [Line2D([0], [0], color=c, linestyle=ls, lw=LW,
                  solid_capstyle="round", dash_capstyle="round",
                  label=DISPLAY[m])
           for m, (c, ls) in MODELS.items()]
handles.append(Line2D([0], [0], marker="o", markersize=5, linestyle="None",
                      markerfacecolor="#E24B4A", markeredgecolor="#791F1F",
                      label="RSR < 0.95 (gate failure)"))
fig.subplots_adjust(bottom=0.26, top=0.90, left=0.07, right=0.98)
fig.text(0.5, 0.96,
         "Red dot = RSR < 0.95  (chat dropped a previously-correct call)",
         ha="center", va="top", fontsize=10, color="#791F1F")
# Constrain legend to span only the width of the two panels (no wider).
left_edge  = axes[0].get_position().x0
right_edge = axes[1].get_position().x1
legend = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                    fontsize=10, mode="expand",
                    bbox_to_anchor=(left_edge, 0.02, right_edge - left_edge, 0.18))

os.makedirs(OUT_DIR, exist_ok=True)
save_kw = dict(dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))
plt.savefig(os.path.join(OUT_DIR, "rair_slope.pdf"), **save_kw)
plt.savefig(os.path.join(OUT_DIR, "rair_slope.png"), **save_kw)
plt.close()

fails = df[df["rsr_mean"] < RSR_GATE][["dataset", "instructor_llm",
                                        "target_accuracy", "rsr_mean"]]
fails = fails.sort_values(["dataset", "instructor_llm", "target_accuracy"])
print(f"RSR gate failures (rsr_mean < {RSR_GATE}):  n={len(fails)}")
for _, r in fails.iterrows():
    print(f"  {r['dataset']:6s}  {r['instructor_llm']:20s}  "
          f"cutoff={r['target_accuracy']:.1f}  rsr={r['rsr_mean']:.3f}")
