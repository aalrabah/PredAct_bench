"""
Forest plot: mean ± SD for 5 Likert metrics across conditions.
Reads study_logs/*.json, saves figures/likert_forest.png and results/likert_summary.csv
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

# ── load all participants ──────────────────────────────────────────────────────
participants = []
for fname in sorted(os.listdir(LOGS_DIR)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(LOGS_DIR, fname)) as f:
        d = json.load(f)
    br = d.get("block_responses", {})
    # normalise keys to int
    participants.append({int(k): v for k, v in br.items()})

N = len(participants)
print(f"Loaded {N} participants")

# ── metric extraction per participant per condition ───────────────────────────
# Returns list of per-participant values (float) for a given condition.
# Returns None for n/a conditions.

def confidence(p, block):
    br = p.get(block)
    if br is None:
        return None
    if block == 1:
        return float(br.get("q1", np.nan))
    # blocks 2-4: average q1, q2, q3
    vals = [br.get("q1"), br.get("q2"), br.get("q3")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan

def single_q(p, block, q):
    br = p.get(block)
    if br is None:
        return None
    v = br.get(q)
    return float(v) if v is not None else np.nan

# conditions: (label, block, color)
CONDITIONS = [
    ("No Agent",    1, "#888888"),
    ("GPT-4o Mini", 2, "#c0392b"),
    ("Qwen-9B",     3, "#e67e22"),
    ("Qwen-35B",    4, "#27ae60"),
]
OVERALL_COLOR = "#2980b9"

METRICS = [
    "Decision confidence",
    "Useful collaborator",
    "Could tell when wrong",
    "Trust scaled with accuracy",
    "Would deploy in own courses",
]

def get_vals(metric, block, participants):
    """Return list of per-participant floats for a metric+block. None if n/a."""
    if metric == "Decision confidence":
        raw = [confidence(p, block) for p in participants]
    elif metric == "Useful collaborator":
        if block == 1: return None
        raw = [single_q(p, block, "q4") for p in participants]
    elif metric == "Could tell when wrong":
        if block == 1: return None
        raw = [single_q(p, block, "q5") for p in participants]
    elif metric == "Trust scaled with accuracy":
        if block == 1: return None
        raw = [single_q(p, block, "q6") for p in participants]
    elif metric == "Would deploy in own courses":
        # only Overall marker; skip individual agent conditions here
        if block != 4: return None
        raw = [single_q(p, 4, "q8") for p in participants]
    else:
        return None
    clean = [v for v in raw if v is not None and not np.isnan(v)]
    return clean if clean else None

def overall_vals(metric, participants):
    """Pool across agent blocks 2, 3, 4 — per-participant mean then collect."""
    if metric == "Would deploy in own courses":
        # single item from block 4 only
        return get_vals(metric, 4, participants)
    agent_blocks = [2, 3, 4]
    per_p = []
    for p in participants:
        vals = []
        for b in agent_blocks:
            v = get_vals(metric, b, [p])
            if v:
                vals.extend(v)
        if vals:
            per_p.append(float(np.mean(vals)))
    return per_p if per_p else None

def stats(vals):
    a = np.array(vals)
    return len(a), float(np.mean(a)), float(np.std(a, ddof=1)), float(np.median(a))

# ── build data table ───────────────────────────────────────────────────────────
rows = []  # (metric, condition_label, n, mean, sd, median, color, y_offset)

for metric in METRICS:
    for label, block, color in CONDITIONS:
        if metric == "Would deploy in own courses" and label != "No Agent":
            # skip individual agent conditions for this metric (only Overall shown)
            continue
        vals = get_vals(metric, block, participants)
        if vals is None:
            continue
        n, mu, sd, med = stats(vals)
        rows.append((metric, label, n, mu, sd, med, color))
    # Overall (agent blocks pooled)
    if metric != "Decision confidence" or True:  # always add overall
        vals = overall_vals(metric, participants)
        if vals and metric != "Decision confidence":
            # skip overall for "Decision confidence" — No Agent is the reference
            n, mu, sd, med = stats(vals)
            rows.append((metric, "Overall", n, mu, sd, med, OVERALL_COLOR))
        elif metric == "Decision confidence":
            # include overall for confidence too (agent overall)
            vals = overall_vals(metric, participants)
            if vals:
                n, mu, sd, med = stats(vals)
                rows.append((metric, "Overall (agents)", n, mu, sd, med, OVERALL_COLOR))

# ── save CSV ───────────────────────────────────────────────────────────────────
with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "condition", "n", "mean", "sd", "median"])
    for metric, label, n, mu, sd, med, _ in rows:
        w.writerow([metric, label, n, f"{mu:.3f}", f"{sd:.3f}", f"{med:.3f}"])
print(f"Saved CSV → {CSV_OUT}")

# ── forest plot ────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif"})

# layout: one row per metric, with sub-rows for each condition
# collect the y positions
metric_order = METRICS
y_gap_between_metrics = 0.7   # extra gap between metric groups
y_step = 0.38                 # spacing between conditions within a metric

# assign y positions
y_positions = {}   # (metric, condition) -> y
y_metric_centers = {}
current_y = 0

condition_order = ["No Agent", "Agent 1", "Agent 2", "Agent 3", "Overall", "Overall (agents)"]

for metric in reversed(metric_order):  # reversed so top metric is at top of plot
    metric_rows = [(m, c, n, mu, sd, med, col) for m, c, n, mu, sd, med, col in rows if m == metric]
    # sort by condition_order
    metric_rows.sort(key=lambda r: condition_order.index(r[1]) if r[1] in condition_order else 99)

    ys = []
    for i, (m, c, n, mu, sd, med, col) in enumerate(metric_rows):
        y = current_y + i * y_step
        y_positions[(metric, c)] = y
        ys.append(y)
    y_metric_centers[metric] = np.mean(ys) if ys else current_y
    current_y = (max(ys) if ys else current_y) + y_gap_between_metrics

fig, ax = plt.subplots(figsize=(9, 9), dpi=300)

for metric, condition, n, mu, sd, med, color in rows:
    y = y_positions.get((metric, condition))
    if y is None:
        continue
    marker = "D" if condition in ("Overall", "Overall (agents)") else "o"
    ms = 8 if condition in ("Overall", "Overall (agents)") else 7
    ax.errorbar(mu, y, xerr=sd,
                fmt=marker, color=color,
                markersize=ms, markeredgecolor="white", markeredgewidth=0.8,
                ecolor=color, elinewidth=1.4, capsize=3, capthick=1.4,
                zorder=4)

# metric group labels on y axis
ax.set_yticks(list(y_metric_centers.values()))
ax.set_yticklabels(list(y_metric_centers.keys()), fontsize=11)

# light horizontal band per metric group
for i, metric in enumerate(reversed(metric_order)):
    metric_rows = [y_positions[(m, c)] for (m, c) in y_positions if m == metric]
    if not metric_rows:
        continue
    y_lo = min(metric_rows) - y_step * 0.45
    y_hi = max(metric_rows) + y_step * 0.45
    if i % 2 == 0:
        ax.axhspan(y_lo, y_hi, color="#f5f5f5", zorder=0)

ax.axvline(3, color="#cccccc", linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
ax.set_xlim(0.8, 5.2)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1\nStrongly\nDisagree", "2", "3\nNeutral", "4", "5\nStrongly\nAgree"],
                   fontsize=9)
ax.set_xlabel("Likert response (1 = strongly disagree → 5 = strongly agree)",
              fontsize=10, labelpad=10)
ax.set_title("Participant survey responses by condition  (mean ± SD)",
             fontsize=12, pad=14)

# legend
legend_items = [
    ("No Agent",           "#888888", "o"),
    ("GPT-4o Mini",        "#c0392b", "o"),
    ("Qwen-9B",            "#e67e22", "o"),
    ("Qwen-35B",           "#27ae60", "o"),
    ("Overall (agents)",   "#2980b9", "D"),
]
handles = [
    mlines.Line2D([], [], marker=mk, color=col, linestyle="none",
                  markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                  label=lbl)
    for lbl, col, mk in legend_items
]
ax.legend(handles=handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.11),
          ncol=5, fontsize=9,
          frameon=True, edgecolor="#cccccc", framealpha=0.95,
          handletextpad=0.5, labelspacing=0.4)

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#888888")
    ax.spines[sp].set_linewidth(0.7)

ax.yaxis.set_tick_params(length=0)
ax.grid(axis="x", color="#eeeeee", linewidth=0.6, zorder=0)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "likert_forest.png")
out_pdf = os.path.join(OUT_DIR, "likert_forest.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
