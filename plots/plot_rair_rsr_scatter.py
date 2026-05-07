"""
RAIR × RSR scatter plot — human vs. 13 LLM agents.
Option C: cluster labeled as a region; only outliers labeled individually.
"""
import os, csv, colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mc
import matplotlib.patches as mpatches

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLED  = os.path.join(PROJECT, "results_pooled.csv")
EXP2    = os.path.join(PROJECT, "results/exp2/exp2_per_cell.csv")
OUT_DIR = os.path.join(PROJECT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────
def lighten(hex_color, factor):
    r, g, b = mc.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(1.0, l + factor * (1.0 - l)), s)

FAMILIES = {
    "OpenAI":    "#b03030",
    "Anthropic": "#c4795b",
    "Google":    "#5a7fa3",
    "DeepSeek":  "#4a7340",
    "Mistral":   "#7d6b9e",
    "Qwen":      "#8e6f4a",
}

AGENTS = [
    ("gpt5_5",            "OpenAI",    0.00, "GPT-5.5"),
    ("gpt5_4_mini",       "OpenAI",    0.35, "GPT-5.4 Mini"),
    ("gpt4o_mini",        "OpenAI",    0.20, "GPT-4o Mini"),
    ("claude_opus_4_7",   "Anthropic", 0.00, "Opus 4.7"),
    ("claude_haiku_4_5",  "Anthropic", 0.20, "Haiku 4.5"),
    ("gemini_3_1_pro",    "Google",    0.00, "Gemini 3.1 Pro"),
    ("gemini_3_flash",    "Google",    0.20, "Gemini 3 Flash"),
    ("deepseek_v4_pro",   "DeepSeek",  0.00, "V4 Pro"),
    ("deepseek_v4_flash", "DeepSeek",  0.20, "V4 Flash"),
    ("mistral_small_24b", "Mistral",   0.00, "Small 24B"),
    ("ministral_3_14b",   "Mistral",   0.20, "3 14B"),
    ("qwen_35b",          "Qwen",      0.00, "Qwen 35B"),
    ("qwen_9b",           "Qwen",      0.20, "Qwen 9B"),
]

# ── load human ─────────────────────────────────────────────────────────────────
cf_h = bf_h = co_h = bo_h = 0
with open(POOLED, newline="") as f:
    for r in csv.DictReader(f):
        if not r["llm"] or r["llm"] == "no_agent":
            continue
        cf_h += int(r["correct_follow"])
        bf_h += int(r["bad_follow"])
        co_h += int(r["correct_override"])
        bo_h += int(r["bad_override"])

human_rair = co_h / (co_h + bf_h) if (co_h + bf_h) > 0 else 0.0
human_rsr  = cf_h / (cf_h + bo_h) if (cf_h + bo_h) > 0 else 0.0

# ── load agents ────────────────────────────────────────────────────────────────
_acc = {}
with open(EXP2, newline="") as f:
    for r in csv.DictReader(f):
        m = r["instructor_llm"]
        try:
            rv, sv = float(r["rair_mean"]), float(r["rsr_mean"])
        except ValueError:
            continue
        _acc.setdefault(m, []).append((rv, sv))

agent_pts = {m: (float(np.mean([v[0] for v in vals])),
                 float(np.mean([v[1] for v in vals])))
             for m, vals in _acc.items()}

# ── classify outliers vs cluster ───────────────────────────────────────────────
# Outliers: RAIR >= 0.19 (captures Gemini Flash at 0.192, all others below 0.16)
OUTLIER_THRESHOLD = 0.19

outliers = []
cluster  = []
for m_key, family, lf, short in AGENTS:
    if m_key not in agent_pts:
        continue
    rair, rsr = agent_pts[m_key]
    if rair >= OUTLIER_THRESHOLD:
        outliers.append((short, family, lf, rair, rsr))
    else:
        cluster.append((short, family, lf, rair, rsr))

print(f"\nHuman:  RAIR={human_rair:.3f}  RSR={human_rsr:.3f}")
print(f"\nOUTLIERS (RAIR >= {OUTLIER_THRESHOLD}) — will be individually labeled:")
print(f"  {'Model':<18} {'RAIR':>6} {'RSR':>6}")
for name, _, _, rair, rsr in sorted(outliers, key=lambda x: -x[3]):
    print(f"  {name:<18} {rair:>6.3f} {rsr:>6.3f}")

print(f"\nCLUSTER (RAIR < {OUTLIER_THRESHOLD}) — labeled as region (n={len(cluster)}):")
print(f"  {'Model':<18} {'RAIR':>6} {'RSR':>6}")
for name, _, _, rair, rsr in sorted(cluster, key=lambda x: -x[3]):
    print(f"  {name:<18} {rair:>6.3f} {rsr:>6.3f}")

cluster_rairs = [x[3] for x in cluster]
cluster_rsrs  = [x[4] for x in cluster]
print(f"\n  Cluster bounds: RAIR [{min(cluster_rairs):.3f}, {max(cluster_rairs):.3f}]"
      f"  RSR [{min(cluster_rsrs):.3f}, {max(cluster_rsrs):.3f}]")

# cluster bounding box with padding
CX0 = -0.02
CX1 = max(cluster_rairs) + 0.06   # right edge with padding
CY0 = min(cluster_rsrs)  - 0.025  # bottom edge
CY1 = 1.02                        # top of plot

# ── figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=300)

# cluster shaded region
cluster_patch = mpatches.FancyBboxPatch(
    (CX0, CY0), CX1 - CX0, CY1 - CY0,
    boxstyle="round,pad=0.01",
    facecolor="#888888", alpha=0.10,
    edgecolor="#888888", linestyle=(0, (4, 1.5)),
    linewidth=1.0, zorder=1,
)
ax.add_patch(cluster_patch)

# cluster annotation — anchor at centroid of the 10 cluster dots
cluster_cx = float(np.mean([x[3] for x in cluster]))
cluster_cy = float(np.mean([x[4] for x in cluster]))
ax.annotate(
    f"Over-reliance cluster (n={len(cluster)})\nRAIR < 0.2,  RSR > 0.97",
    xy=(cluster_cx, cluster_cy),
    xytext=(0.01, 0.78),
    fontsize=8, color="#555555", style="italic",
    ha="left", va="top", zorder=4,
    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6,
                    connectionstyle="arc3,rad=0.2"),
)

# quadrant reference lines
ref_kw = dict(color="#cccccc", linewidth=0.8, linestyle=(0, (4, 1.5)), zorder=2)
ax.axvline(0.5, **ref_kw)
ax.axhline(0.5, **ref_kw)

# quadrant corner labels — top-left removed (cluster annotation is more informative)
quad = [
    (0.97, 0.97, "CALIBRATED",          "right", "top",    "#15803d"),
    (0.97, 0.09, "over-trusts agent",   "right", "bottom", "#aaaaaa"),
    (0.03, 0.09, "random / both wrong", "left",  "bottom", "#aaaaaa"),
]
for qx, qy, qlbl, ha, va, col in quad:
    ax.text(qx, qy, qlbl, fontsize=7.5, style="italic", color=col,
            ha=ha, va=va, transform=ax.transAxes, zorder=3)

# ── plot all dots (cluster + outliers) ────────────────────────────────────────
for m_key, family, lf, short in AGENTS:
    if m_key not in agent_pts:
        continue
    rair, rsr = agent_pts[m_key]
    color = lighten(FAMILIES[family], lf)
    ax.scatter(rair, rsr, s=90, color=color,
               edgecolors="white", linewidths=1.2, zorder=5)

# ── human dot ─────────────────────────────────────────────────────────────────
ax.scatter(human_rair, human_rsr, s=180, marker="D",
           color="#f59e0b", edgecolors="#1e1e1e", linewidths=1.5, zorder=10)

# ── annotate outliers only ─────────────────────────────────────────────────────
# offsets tuned to actual point positions so labels sit clearly off the dot
outlier_offsets = {
    "HUMAN":         (0.05,  -0.04,  "left"),
    "GPT-5.5":       (0.045,  0.035, "left"),
    "Gemini 3.1 Pro":(0.025,  0.030, "left"),
    "Gemini 3 Flash":(0.045,  0.045, "left"),
}
# Also annotate Human (handled separately since it's not in AGENTS list)
labeled_outliers = {name: (rair, rsr) for name, _, _, rair, rsr in outliers}
labeled_outliers["HUMAN"] = (human_rair, human_rsr)

annot_style = dict(
    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.7, shrinkA=4, shrinkB=4),
    zorder=8,
)
for name, (rair, rsr) in labeled_outliers.items():
    dx, dy, ha = outlier_offsets.get(name, (0.03, 0.03, "left"))
    is_human = (name == "HUMAN")
    fam_color = "#1e1e1e" if is_human else FAMILIES[
        next(f for m, f, *_ in AGENTS if
             next((s for _, _, _, s in [a for a in AGENTS if a[0] == m]), "") == name
             or (m in agent_pts and name in [s for _, _, _, s in AGENTS if _ == f]))
    ]
    # simpler color lookup
    color = "#1e1e1e" if is_human else next(
        (lighten(FAMILIES[fam], lf)
         for m, fam, lf, short in AGENTS if short == name),
        "#555555"
    )
    ax.annotate(
        name,
        xy=(rair, rsr),
        xytext=(rair + dx, rsr + dy),
        fontsize=8 if is_human else 7.5,
        fontweight="bold" if is_human else "normal",
        color=color,
        ha=ha, va="center",
        **annot_style,
    )

# ── legend (family colors + Human) ────────────────────────────────────────────
legend_handles = [
    mlines.Line2D([], [], marker="o", linestyle="none", markersize=8,
                  color=col, markeredgecolor="white", markeredgewidth=1.0,
                  label=fam)
    for fam, col in FAMILIES.items()
]
legend_handles.append(
    mlines.Line2D([], [], marker="D", linestyle="none", markersize=8,
                  color="#f59e0b", markeredgecolor="#1e1e1e", markeredgewidth=1.2,
                  label="HUMAN")
)
ax.legend(handles=legend_handles, loc="center right", fontsize=7.5,
          frameon=True, edgecolor="#cccccc", framealpha=0.95,
          handletextpad=0.5, labelspacing=0.45,
          bbox_to_anchor=(1.0, 0.38))

# ── axes styling ───────────────────────────────────────────────────────────────
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.48, 1.02)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0.5, 0.75, 1.0])
ax.tick_params(labelsize=8)
ax.set_xlabel("RAIR  (correctly dismiss wrong agent →)", fontsize=9, labelpad=6)
ax.set_ylabel("RSR  (correctly trust right agent →)",   fontsize=9, labelpad=6)
ax.set_title("Decision Calibration: Human vs. 13 LLM Instructors",
             fontsize=10, pad=22)
ax.grid(False)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#555555")
    ax.spines[sp].set_linewidth(0.8)

plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, "rair_rsr_scatter.pdf")
out_png = os.path.join(OUT_DIR, "rair_rsr_scatter.png")
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_png}")
print(f"Saved → {out_pdf}")
