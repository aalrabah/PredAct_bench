"""plot_human_agent_agreement.py — per-student agreement scatter for PredAct-CS.

Compares humans (n=6) vs LLM agents (13 instructors × ~10 episodes per cell)
on the SAME flagged students at the SAME cells the human study used.

Matched cells (PredAct-CS, week 8 each):
  Course_A  →  40% target accuracy
  Course_B  →  60% target accuracy
  Course_C  →  80% target accuracy

Output:
  figures/human_agent_agreement.{pdf,png}
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

HUMAN_DIR = os.path.join(PROJECT_ROOT, "study_logs")
EXP2_DIR  = os.path.join(PROJECT_ROOT, "results/exp2/sim_logs")
OUT_DIR   = os.path.join(PROJECT_ROOT, "figures")

# (course, week) → accuracy bucket (matches human study + exp2_config.py)
MATCHED = {
    ("Course_A", 8): 0.4,
    ("Course_B", 8): 0.6,
    ("Course_C", 8): 0.8,
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def derive_flag(per_student_entry):
    """Did the evaluator (human or instructor LLM) ultimately mark this
    student as at-risk? Returns None if the evaluator never decided on them."""
    af = per_student_entry.get("agent_flagged")
    fd = per_student_entry.get("final_decision")
    if fd is None:
        return None
    if af:
        return fd == "accept"
    return fd == "reject"


# --- Collect human-side per-student decisions -------------------------------
human_flags = {}    # sid -> list[bool]
truth       = {}    # sid -> bool (is_at_risk)
for path in sorted(glob.glob(os.path.join(HUMAN_DIR, "*.json"))):
    d = json.load(open(path))
    for cond in d.get("conditions", []):
        if not cond.get("has_agent"):
            continue
        key = (cond.get("course_id"), cond.get("week"))
        if key not in MATCHED:
            continue
        for sid, ps in cond.get("per_student", {}).items():
            flag = derive_flag(ps)
            if flag is None:
                continue
            human_flags.setdefault(sid, []).append(flag)
            truth[sid] = bool(ps.get("is_at_risk"))

# --- Collect agent-side per-student decisions -------------------------------
agent_flags = {}    # sid -> list[bool]
for path in sorted(glob.glob(os.path.join(EXP2_DIR, "*", "predact_cs_*", "run_*.json"))):
    d = json.load(open(path))
    key = (d.get("course_id"), d.get("week"))
    if key not in MATCHED:
        continue
    for sid, ps in d.get("per_student", {}).items():
        flag = derive_flag(ps)
        if flag is None:
            continue
        agent_flags.setdefault(sid, []).append(flag)
        truth.setdefault(sid, bool(ps.get("is_at_risk")))

# --- Intersect on students reviewed by BOTH sides ---------------------------
common = sorted(set(human_flags) & set(agent_flags))
xs, ys, labels = [], [], []
for sid in common:
    if sid not in truth:
        continue
    h = sum(human_flags[sid]) / len(human_flags[sid])
    a = sum(agent_flags[sid]) / len(agent_flags[sid])
    xs.append(h); ys.append(a); labels.append(truth[sid])
xs, ys, labels = np.array(xs), np.array(ys), np.array(labels, dtype=bool)
n = len(xs)

# --- Agreement statistics ---------------------------------------------------
pr, _ = pearsonr(xs, ys) if n >= 2 else (np.nan, None)
sr, _ = spearmanr(xs, ys) if n >= 2 else (np.nan, None)
kappa = cohen_kappa_score(xs >= 0.5, ys >= 0.5) if n >= 2 else np.nan

# --- Scatter ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], linestyle="--", color="#888", lw=1.0, zorder=1)
# tiny jitter so dots at exact 0/1 don't overlap into one blob
rng = np.random.default_rng(0)
jx = xs + rng.uniform(-0.015, 0.015, size=n)
jy = ys + rng.uniform(-0.015, 0.015, size=n)
for is_fail, color, lbl in [
    (False, "#F4C430", "Actually passed (not at-risk)"),
    (True,  "#5B9BD5", "Actually failed (at-risk)"),
]:
    mask = labels == is_fail
    ax.scatter(jx[mask], jy[mask], s=44, color=color,
               edgecolor="black", linewidth=0.4, alpha=0.78,
               label=lbl, zorder=3)

ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
ax.set_aspect("equal")
ax.set_xlabel("Human flag rate", fontsize=11)
ax.set_ylabel("Agent flag rate",  fontsize=11)
ax.grid(True, color="#D3D1C7", linestyle=":", alpha=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

stats_txt = (f"Pearson r = {pr:.3f}\n"
             f"Spearman $\\rho$ = {sr:.3f}\n"
             f"Cohen $\\kappa$ @0.5 = {kappa:.3f}\n"
             f"N = {n} students")
ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes,
        ha="left", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#888", lw=0.6))
ax.legend(loc="lower right", frameon=False, fontsize=9)

os.makedirs(OUT_DIR, exist_ok=True)
out_pdf = os.path.join(OUT_DIR, "human_agent_agreement.pdf")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved → {out_pdf}")
print(f"  N={n} students reviewed by both sides")
print(f"  Pearson r        = {pr:.3f}")
print(f"  Spearman rho     = {sr:.3f}")
print(f"  Cohen kappa @0.5 = {kappa:.3f}")
print(f"  Per-cell student counts (humans-reviewed):")
print(f"    humans saw: {len(human_flags)} unique sids across all matched cells")
print(f"    agents saw: {len(agent_flags)} unique sids across all matched cells")
