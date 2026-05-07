"""
F1 vs Target Tool Accuracy — 2×3 family grid
One panel per model family; both datasets overlaid per panel.
PredAct-CS = solid + filled marker
OULAD      = dashed + open marker
Per-panel inset legend for variants; global bottom legend for datasets.
"""
import os, csv, colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mc
import matplotlib.patches as mpatches

CSV_PATH = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
OUT_DIR  = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── color helpers ──────────────────────────────────────────────────────────────
def lighten(hex_color, factor):
    """Move hex_color factor% toward white (0=unchanged, 1=white)."""
    r, g, b = mc.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, l + factor * (1.0 - l))
    return colorsys.hls_to_rgb(h, l, s)

# ── family definitions ─────────────────────────────────────────────────────────
# (display_label, model_key, variant_index)
# variant 0 = flagship (full color, 'o')
# variant 1 = smaller  (lighten 25%, 's')
# variant 2 = third    (lighten 45%, '^')  — OpenAI only
FAMILIES = [
    ("OpenAI",    "#b03030", [
        ("GPT-5.5",           "gpt5_5",            0),
        ("GPT-5.4 Mini",      "gpt5_4_mini",        2),
        ("GPT-4o Mini",       "gpt4o_mini",          1),
    ]),
    ("Anthropic", "#c4795b", [
        ("Claude Opus 4.7",   "claude_opus_4_7",    0),
        ("Claude Haiku 4.5",  "claude_haiku_4_5",    1),
    ]),
    ("Google",    "#5a7fa3", [
        ("Gemini 3.1 Pro",    "gemini_3_1_pro",     0),
        ("Gemini 3 Flash",    "gemini_3_flash",      1),
    ]),
    ("DeepSeek",  "#4a7340", [
        ("DeepSeek V4 Pro",   "deepseek_v4_pro",    0),
        ("DeepSeek V4 Flash", "deepseek_v4_flash",   1),
    ]),
    ("Mistral",   "#7d6b9e", [
        ("Mistral Small 24B", "mistral_small_24b",   0),
        ("Ministral 3 14B",   "ministral_3_14b",     1),
    ]),
    ("Qwen",      "#8e6f4a", [
        ("Qwen 35B",          "qwen_35b",            0),
        ("Qwen 9B",           "qwen_9b",             1),
    ]),
]

VARIANT_MARKER  = {0: "o", 1: "s", 2: "^"}
VARIANT_LIGHTEN = {0: 0.0, 1: 0.25, 2: 0.45}

DATASET_STYLE = {
    "predact_cs":  dict(ls="-",              filled=True,  lw=1.8),
    "oulad": dict(ls=(0, (4, 1.5)),    filled=False, lw=1.6),
}

SHORT = {
    "GPT-5.5":           "GPT-5.5",
    "GPT-5.4 Mini":      "GPT-5.4 Mini",
    "GPT-4o Mini":       "GPT-4o Mini",
    "Claude Opus 4.7":   "Opus 4.7",
    "Claude Haiku 4.5":  "Haiku 4.5",
    "Gemini 3.1 Pro":    "3.1 Pro",
    "Gemini 3 Flash":    "3 Flash",
    "DeepSeek V4 Pro":   "V4 Pro",
    "DeepSeek V4 Flash": "V4 Flash",
    "Mistral Small 24B": "Small 24B",
    "Ministral 3 14B":   "3 14B",
    "Qwen 35B":          "Qwen 35B",
    "Qwen 9B":           "Qwen 9B",
}

# ── load data ──────────────────────────────────────────────────────────────────
data = {}
with open(CSV_PATH, newline="") as f:
    for r in csv.DictReader(f):
        ds = r["dataset"]
        m  = r["instructor_llm"]
        t  = float(r["target_accuracy"])
        f1 = float(r["f1_final_mean"])
        data.setdefault(ds, {}).setdefault(m, {})[t] = f1

targets  = sorted({t for ds in data.values() for m in ds.values() for t in m})
x_labels = [f"{int(t*100)}%" for t in targets]
x_mid    = np.median(targets)   # 0.6

# ── emptiest-corner detection ──────────────────────────────────────────────────
def emptiest_corner(ax_data_by_variant, targets):
    """
    ax_data_by_variant: list of y-arrays (one per variant, averaged across datasets)
    Returns a matplotlib legend loc string for the emptiest quadrant.
    """
    corners = {
        "upper left":  (lambda t, y: t <= x_mid and y >= 0.5),
        "upper right": (lambda t, y: t >  x_mid and y >= 0.5),
        "lower left":  (lambda t, y: t <= x_mid and y <  0.5),
        "lower right": (lambda t, y: t >  x_mid and y <  0.5),
    }
    density = {loc: 0.0 for loc in corners}
    for ys in ax_data_by_variant:
        for t, y in zip(targets, ys):
            if np.isnan(y):
                continue
            for loc, test in corners.items():
                if test(t, y):
                    density[loc] += 1
    return min(density, key=density.get)

# ── figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif"})
fig, axes = plt.subplots(2, 3, figsize=(12, 6),
                         sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.38, wspace=0.08)

for ax, (fam_name, base_color, variants) in zip(axes.flat, FAMILIES):

    # per-variant y values averaged across datasets (for corner detection)
    variant_ys_avg = []
    inset_handles  = []

    for label, m_key, v_idx in variants:
        color  = lighten(base_color, VARIANT_LIGHTEN[v_idx])
        marker = VARIANT_MARKER[v_idx]
        short  = SHORT.get(label, label)

        ys_across_ds = []
        for ds_key, dstyle in DATASET_STYLE.items():
            m_data = data.get(ds_key, {}).get(m_key, {})
            if not m_data:
                continue
            ys = [m_data.get(t, np.nan) for t in targets]
            mfc = color if dstyle["filled"] else "white"
            mec = "white" if dstyle["filled"] else color
            mew = 1.0     if dstyle["filled"] else 1.4
            ax.plot(
                targets, ys,
                color=color,
                linestyle=dstyle["ls"],
                marker=marker,
                linewidth=dstyle["lw"],
                markersize=7,
                markerfacecolor=mfc,
                markeredgecolor=mec,
                markeredgewidth=mew,
                zorder=3,
            )
            ys_across_ds.append(ys)

        if ys_across_ds:
            avg_ys = np.nanmean(ys_across_ds, axis=0)
            variant_ys_avg.append(avg_ys)

        # one handle per variant (filled, family color)
        inset_handles.append(
            mlines.Line2D([], [],
                          color=color,
                          linestyle="none",
                          marker=marker,
                          markersize=6,
                          markerfacecolor=color,
                          markeredgecolor="white",
                          markeredgewidth=0.8,
                          label=short)
        )

    # ── inset legend ───────────────────────────────────────────────────────────
    leg = ax.legend(
        handles=inset_handles,
        loc="best",
        fontsize=7.5,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#cccccc",
        fancybox=False,
        borderpad=0.5,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    leg.get_frame().set_linewidth(0.6)

    # ── panel styling ──────────────────────────────────────────────────────────
    ax.set_title(fam_name, fontsize=11, fontweight="bold",
                 color=base_color, pad=5)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(targets[0] - 0.02, targets[-1] + 0.02)
    ax.set_xticks(targets)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_tick_params(labelsize=9)

    ax.yaxis.grid(True, color="#dddddd", alpha=0.5, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(base_color)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("#aaaaaa")

# ── shared axis labels ─────────────────────────────────────────────────────────
for ax in axes[:, 0]:
    ax.set_ylabel("F1", fontsize=10)
for ax in axes[1, :]:
    ax.set_xlabel("Target Tool Accuracy", fontsize=10, labelpad=10)

# ── global dataset legend at bottom ───────────────────────────────────────────
global_handles = [
    mlines.Line2D([], [], color="#555555", linestyle="-",           marker="o",
                  markersize=6, markerfacecolor="#555555", markeredgecolor="white",
                  markeredgewidth=0.8, linewidth=1.8,
                  label="PredAct-CS  (solid, filled marker)"),
    mlines.Line2D([], [], color="#555555", linestyle=(0, (4, 1.5)), marker="o",
                  markersize=6, markerfacecolor="white",   markeredgecolor="#555555",
                  markeredgewidth=0.8, linewidth=1.6,
                  label="OULAD  (dashed, open marker)"),
]
fig.legend(
    handles=global_handles,
    loc="lower center",
    ncol=2,
    fontsize=9.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.09),
    handlelength=2.8,
    handletextpad=0.7,
    columnspacing=2.5,
)

out_png = os.path.join(OUT_DIR, "f1_family_grid.png")
out_pdf = os.path.join(OUT_DIR, "f1_family_grid.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
