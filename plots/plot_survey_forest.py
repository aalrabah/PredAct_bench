"""
Forest plot: mean ± SD for 5 Likert metrics across conditions.
All rendering fixes applied (clipped whiskers, legend above title,
fixed condition order, constant row height, Okabe-Ito palette, etc.)
Saves figures/survey_forest.png and figures/survey_forest.pdf
"""
import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

LOGS_DIR = os.path.join(PROJECT_ROOT, "study_logs")
OUT_DIR  = os.path.join(PROJECT_ROOT, "figures")
CSV_OUT  = os.path.join(PROJECT_ROOT, "results", "likert_summary.csv")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)

# Okabe-Ito colorblind-safe palette
COLORS = {
    "No Agent":          "#999999",
    "Agent 1":           "#D55E00",
    "Agent 2":           "#E69F00",
    "Agent 3":           "#009E73",
    "Overall":           "#0072B2",
    "Overall (agents)":  "#0072B2",
}

# ── load participants ──────────────────────────────────────────────────────────
participants = []
for fname in sorted(os.listdir(LOGS_DIR)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(LOGS_DIR, fname)) as f:
        d = json.load(f)
    br = d.get("block_responses", {})
    participants.append({int(k): v for k, v in br.items()})

N = len(participants)
print(f"Loaded {N} participants")

# ── metric extraction ──────────────────────────────────────────────────────────
def confidence(p, block):
    br = p.get(block)
    if br is None:
        return None
    if block == 1:
        v = br.get("q1")
        return float(v) if v is not None else np.nan
    vals = [br.get(f"q{i}") for i in range(1, 4)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan

def single_q(p, block, q):
    br = p.get(block)
    if br is None:
        return None
    v = br.get(q)
    return float(v) if v is not None else np.nan

def get_vals(metric, block, plist):
    if metric == "Decision confidence":
        raw = [confidence(p, block) for p in plist]
    elif metric == "Useful collaborator":
        if block == 1: return None
        raw = [single_q(p, block, "q4") for p in plist]
    elif metric == "Could tell when wrong":
        if block == 1: return None
        raw = [single_q(p, block, "q5") for p in plist]
    elif metric == "Trust scaled with accuracy":
        if block == 1: return None
        raw = [single_q(p, block, "q6") for p in plist]
    elif metric == "Would deploy in own courses":
        if block != 4: return None
        raw = [single_q(p, 4, "q8") for p in plist]
    else:
        return None
    clean = [v for v in raw if v is not None and not np.isnan(v)]
    return clean if clean else None

def overall_vals(metric, plist):
    if metric == "Would deploy in own courses":
        return get_vals(metric, 4, plist)
    per_p = []
    for p in plist:
        vals = []
        for b in [2, 3, 4]:
            v = get_vals(metric, b, [p])
            if v:
                vals.extend(v)
        if vals:
            per_p.append(float(np.mean(vals)))
    return per_p if per_p else None

def stats(vals):
    a = np.array(vals)
    return len(a), float(np.mean(a)), float(np.std(a, ddof=1)), float(np.median(a))

# ── build data rows ────────────────────────────────────────────────────────────
METRICS = [
    "Decision confidence",
    "Useful collaborator",
    "Could tell when wrong",
    "Trust scaled with accuracy",
    "Would deploy in own courses",
]

rows = []
for metric in METRICS:
    for label, block in [("No Agent", 1), ("Agent 1", 2), ("Agent 2", 3), ("Agent 3", 4)]:
        if metric == "Would deploy in own courses":
            continue  # only Overall shown for this metric
        vals = get_vals(metric, block, participants)
        if vals is None:
            continue
        n, mu, sd, med = stats(vals)
        rows.append((metric, label, n, mu, sd, med, COLORS[label]))
    vals = overall_vals(metric, participants)
    if vals:
        n, mu, sd, med = stats(vals)
        key = "Overall (agents)" if metric == "Decision confidence" else "Overall"
        rows.append((metric, key, n, mu, sd, med, COLORS["Overall"]))

# ── save CSV ───────────────────────────────────────────────────────────────────
with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "condition", "n", "mean", "sd", "median"])
    for metric, label, n, mu, sd, med, _ in rows:
        w.writerow([metric, label, n, f"{mu:.3f}", f"{sd:.3f}", f"{med:.3f}"])
print(f"Saved CSV → {CSV_OUT}")

# ── y-position layout ──────────────────────────────────────────────────────────
# Fixed order bottom-to-top within each group so it reads top-to-bottom:
#   top = Overall/Overall(agents)  [blue diamond]
#   ↓     Agent 3                  [green]
#   ↓     Agent 2                  [amber]
#   ↓     Agent 1                  [vermilion]
#   bot = No Agent                 [gray]   — only when applicable
COND_SORT_BOTTOM_UP = [
    "No Agent", "Agent 1", "Agent 2", "Agent 3", "Overall", "Overall (agents)"
]

ROW_STEP  = 1.0   # data units between condition rows within a group
GROUP_GAP = 1.6   # extra data units between metric groups

y_positions    = {}  # (metric, condition) -> y
y_metric_label = {}  # metric -> y for tick label
separator_ys   = []  # horizontal separator lines between groups

current_y = 0.0
# Build bottom-up; METRICS[0] ends up at top
for metric in reversed(METRICS):
    metric_rows = [(m, c, n, mu, sd, med, col)
                   for m, c, n, mu, sd, med, col in rows if m == metric]
    metric_rows.sort(
        key=lambda r: COND_SORT_BOTTOM_UP.index(r[1])
        if r[1] in COND_SORT_BOTTOM_UP else 99
    )
    if not metric_rows:
        continue
    ys = []
    for i, (m, c, n, mu, sd, med, col) in enumerate(metric_rows):
        y = current_y + i * ROW_STEP
        y_positions[(metric, c)] = y
        ys.append(y)
    y_metric_label[metric] = float(np.mean(ys))
    top = max(ys)
    separator_ys.append(top + GROUP_GAP * 0.45)
    current_y = top + GROUP_GAP

if separator_ys:
    separator_ys.pop()  # no line above the topmost group

y_lo = min(y_positions.values()) - ROW_STEP * 0.65
y_hi = max(y_positions.values()) + ROW_STEP * 0.65

# ── figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)

# x=3 neutral reference line (behind everything)
ax.axvline(3, color="#cccccc", linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)

# subtle separator lines between metric groups
for sy in separator_ys:
    ax.axhline(sy, color="#dddddd", linewidth=0.5, zorder=2)

# ── dots + clipped whiskers ────────────────────────────────────────────────────
for metric, condition, n, mu, sd, med, color in rows:
    y = y_positions.get((metric, condition))
    if y is None:
        continue
    xlo = max(1.0, mu - sd)
    xhi = min(5.0, mu + sd)
    xerr_lo = mu - xlo
    xerr_hi = xhi - mu
    is_overall = condition in ("Overall", "Overall (agents)")
    ax.errorbar(
        mu, y,
        xerr=[[xerr_lo], [xerr_hi]],
        fmt="D" if is_overall else "o",
        color=color,
        markersize=9 if is_overall else 7,
        markeredgecolor="white",
        markeredgewidth=1.0,
        ecolor=color,
        elinewidth=2.0,
        capsize=4,
        capthick=2.0,
        zorder=4,
    )

# ── y-axis ─────────────────────────────────────────────────────────────────────
# Preserve METRICS order for tick labels (top metric = largest y)
ordered_metrics = [m for m in METRICS if m in y_metric_label]
ax.set_yticks([y_metric_label[m] for m in ordered_metrics])
ax.set_yticklabels(ordered_metrics, fontsize=10, fontweight="bold", ha="right")
ax.yaxis.set_tick_params(length=0, pad=10)
ax.set_ylim(y_lo, y_hi)

# ── x-axis ─────────────────────────────────────────────────────────────────────
ax.set_xlim(0.6, 5.4)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=10)

# "Strongly disagree" / "Strongly agree" in small italic gray below x=1, x=5
trans = ax.get_xaxis_transform()
ax.text(1, -0.07, "Strongly disagree", transform=trans,
        ha="center", va="top", fontsize=8, style="italic", color="#888888")
ax.text(5, -0.07, "Strongly agree", transform=trans,
        ha="center", va="top", fontsize=8, style="italic", color="#888888")

# ── spine / grid cleanup ───────────────────────────────────────────────────────
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#aaaaaa")
ax.spines["bottom"].set_linewidth(0.7)
ax.tick_params(axis="x", colors="#555555")
ax.tick_params(axis="y", colors="#333333")
ax.grid(axis="x", color="#eeeeee", linewidth=0.5, zorder=0)
ax.grid(axis="y", visible=False)

# ── title + subtitle ──────────────────────────────────────────────────────────
ax.text(0, 1.17, "Participant survey responses by condition",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        ha="left", va="bottom")
ax.text(0, 1.10, "Mean ± SD on a 5-point Likert scale  ·  whiskers clipped to scale bounds",
        transform=ax.transAxes, fontsize=8.5, color="#777777",
        ha="left", va="bottom")

# ── legend above title, one row, no frame ─────────────────────────────────────
legend_items = [
    ("No Agent",         "#999999", "o"),
    ("Agent 1",          "#D55E00", "o"),
    ("Agent 2",          "#E69F00", "o"),
    ("Agent 3",          "#009E73", "o"),
    ("Overall (agents)", "#0072B2", "D"),
]
handles = [
    mlines.Line2D([], [], marker=mk, color=col, linestyle="none",
                  markersize=9 if mk == "D" else 7,
                  markeredgecolor="white", markeredgewidth=1.0,
                  label=lbl)
    for lbl, col, mk in legend_items
]
ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.26),
    ncol=5,
    frameon=False,
    fontsize=9,
    handletextpad=0.5,
    columnspacing=1.4,
)

# ── save ──────────────────────────────────────────────────────────────────────
out_png = os.path.join(OUT_DIR, "survey_forest.png")
out_pdf = os.path.join(OUT_DIR, "survey_forest.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")

# ── verify whisker clipping ────────────────────────────────────────────────────
print("\nWhisker clip verification:")
any_clipped = False
for metric, condition, n, mu, sd, med, _ in rows:
    if mu - sd < 1.0 - 1e-9:
        print(f"  CLIPPED LEFT:  {condition:<22} {metric}  "
              f"raw=[{mu-sd:.3f}, {mu+sd:.3f}] → [{max(1.0, mu-sd):.3f}, ...]")
        any_clipped = True
    if mu + sd > 5.0 + 1e-9:
        print(f"  CLIPPED RIGHT: {condition:<22} {metric}  "
              f"raw=[{mu-sd:.3f}, {mu+sd:.3f}] → [..., {min(5.0, mu+sd):.3f}]")
        any_clipped = True
if not any_clipped:
    print("  All whiskers within [1, 5] — no clipping needed for this dataset.")
