"""plot_human_validation.py — three figures validating the agent-to-agent
simulator against the human study (PredAct-CS only).

Outputs (figures/human_validation/):
  per_condition_agreement.{pdf,png}   human-with-agent F1 vs agent-to-agent F1
  initial_to_final_shifts.{pdf,png}   stacked bar of dialogue-induced flips
  override_correctness.{pdf,png}      2x2 of human overrides vs ground truth

Console: per-condition N, agreement values, conditions where human-vs-agent F1
gap > 15 points.
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HUMAN_DIR = os.path.join(PROJECT_ROOT, "study_logs")
RAW_CSV   = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
OUT_DIR   = os.path.join(PROJECT_ROOT, "figures/human_validation")

# condition_id -> (tool LLM key as used in exp2_per_cell.csv, target accuracy)
COND_INFO = {
    "gpt_40": ("gpt4o_mini", 0.4), "gpt_60": ("gpt4o_mini", 0.6), "gpt_80": ("gpt4o_mini", 0.8),
    "q9b_40": ("qwen_9b",   0.4), "q9b_60": ("qwen_9b",   0.6), "q9b_80": ("qwen_9b",   0.8),
    "q35_40": ("qwen_35b",  0.4), "q35_60": ("qwen_35b",  0.6), "q35_80": ("qwen_35b",  0.8),
}
COND_ORDER = list(COND_INFO.keys())   # left-to-right order on x-axis
DISPLAY = {"gpt4o_mini": "GPT-4o Mini", "qwen_9b": "Qwen 9B", "qwen_35b": "Qwen 35B"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def derive_flag(ps):
    af, fd = ps.get("agent_flagged"), ps.get("final_decision")
    if fd is None:
        return None
    return (af and fd == "accept") or ((not af) and fd == "reject")


# -- Load human-study per-student decisions ---------------------------------
records = []  # list of dicts: condition, participant, sid, init, final, af, human_flag, truth
for path in sorted(glob.glob(os.path.join(HUMAN_DIR, "*.json"))):
    d = json.load(open(path))
    name = d["participant"]["name"]
    for cond in d.get("conditions", []):
        cid = cond.get("condition_id")
        if cid not in COND_INFO or not cond.get("has_agent"):
            continue
        for sid, ps in cond.get("per_student", {}).items():
            flag = derive_flag(ps)
            if flag is None:
                continue
            records.append({
                "condition":     cid,
                "participant":   name,
                "sid":           sid,
                "init":          ps.get("initial_decision"),
                "final":         ps.get("final_decision"),
                "agent_flagged": ps.get("agent_flagged"),
                "human_flag":    flag,
                "truth":         bool(ps.get("is_at_risk")),
            })

# -- Per-condition human F1 (mean ± SE across participants) -----------------
groups = defaultdict(list)
for r in records:
    groups[(r["condition"], r["participant"])].append(r)

human_f1 = defaultdict(list)
for (cid, _), decs in groups.items():
    tp = sum(1 for r in decs if r["human_flag"] and r["truth"])
    fp = sum(1 for r in decs if r["human_flag"] and not r["truth"])
    fn = sum(1 for r in decs if not r["human_flag"] and r["truth"])
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    human_f1[cid].append(f1)

# -- Agent-to-agent F1 from exp2_per_cell.csv -------------------------------
df = pd.read_csv(RAW_CSV)
df = df[df["dataset"] == "predact_cs"]
agent_f1 = {}
for cid, (llm, acc) in COND_INFO.items():
    row = df[(df["instructor_llm"] == llm) & (df["target_accuracy"] == acc)]
    agent_f1[cid] = float(row["f1_final_mean"].iloc[0]) if not row.empty else np.nan


# ============================================================================
# Figure 1 — per-condition agreement bar chart
# ============================================================================
def fig_agreement():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(COND_ORDER))
    bw = 0.4
    h_means = [np.mean(human_f1[c]) if human_f1[c] else np.nan for c in COND_ORDER]
    h_ses   = [np.std(human_f1[c], ddof=1) / np.sqrt(len(human_f1[c]))
               if len(human_f1[c]) > 1 else 0 for c in COND_ORDER]
    a_vals  = [agent_f1.get(c, np.nan) for c in COND_ORDER]

    ax.bar(x - bw/2, h_means, bw, yerr=h_ses, color="#5B9BD5",
           edgecolor="black", linewidth=0.5, capsize=3,
           label="Human-with-agent F1 (mean ± SE across 6 participants)")
    ax.bar(x + bw/2, a_vals, bw, color="#F4A261",
           edgecolor="black", linewidth=0.5,
           label="Agent-to-agent F1 (mean across 10 episodes)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{DISPLAY[COND_INFO[c][0]]}\n{int(COND_INFO[c][1]*100)}%"
         for c in COND_ORDER], fontsize=9)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, fontsize=9, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "per_condition_agreement.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")
    return h_means, a_vals


# ============================================================================
# Figure 2 — initial->final shift breakdown (stacked bars)
# ============================================================================
def fig_shifts():
    cats = ["accept->accept", "accept->reject", "reject->accept", "reject->reject", "(none->reject)"]
    cat_colors = ["#A8D5BA", "#E8849A", "#F4D35E", "#9DB4C0", "#C9B6E0"]
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        init, final = r["init"], r["final"]
        if init is None:
            key = "(none->reject)" if final == "reject" else None
        else:
            key = f"{init}->{final}"
        if key in cats:
            counts[r["condition"]][key] += 1

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(COND_ORDER))
    bottom = np.zeros(len(COND_ORDER))
    for cat, color in zip(cats, cat_colors):
        vals = [counts[c].get(cat, 0) for c in COND_ORDER]
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="black",
               linewidth=0.4, label=cat)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{DISPLAY[COND_INFO[c][0]]}\n{int(COND_INFO[c][1]*100)}%"
         for c in COND_ORDER], fontsize=9)
    ax.set_ylabel("Decisions (pooled across 6 participants)", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, fontsize=8.5, ncol=5,
              title="initial -> final", title_fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "initial_to_final_shifts.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")
    return counts


# ============================================================================
# Figure 3 — override correctness (2x2)
# ============================================================================
def fig_overrides():
    # Override = human_flag != agent_flagged
    # 2x2: rows = override type (reject flag / add flag), cols = right / wrong
    mat = np.zeros((2, 2), dtype=int)
    row_labels = ["Reject the agent's flag\n(agent flagged -> human said no)",
                  "Add a new flag\n(agent didn't flag -> human said yes)"]
    col_labels = ["Human was right", "Human was wrong"]
    for r in records:
        if r["human_flag"] == r["agent_flagged"]:
            continue   # not an override
        is_reject = r["agent_flagged"]   # agent flagged but human said no
        right = (r["human_flag"] == r["truth"])
        mat[0 if is_reject else 1, 0 if right else 1] += 1

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0,
                   vmax=max(mat.max(), 1))
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            row_total = mat[i].sum()
            pct = (100 * v / row_total) if row_total else 0
            color = "white" if v > 0.5 * mat.max() else "black"
            ax.text(j, i, f"{v}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(f"Human override correctness  (n={mat.sum()} overrides pooled)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "override_correctness.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")
    return mat


# -- Run ----------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
h_means, a_vals = fig_agreement()
fig_shifts()
mat = fig_overrides()

# -- Console report -----------------------------------------------------------
print("\n=== Per-condition summary (human vs agent) ===")
print(f"{'Condition':<10} {'tool':<14} {'acc':<5} {'n_part':>7} {'n_dec':>7} "
      f"{'human F1':>10} {'agent F1':>10} {'gap':>7}")
for cid, h_mean, a_val in zip(COND_ORDER, h_means, a_vals):
    n_part = len(human_f1[cid])
    n_dec  = sum(1 for r in records if r["condition"] == cid)
    gap = (h_mean - a_val) if not (np.isnan(h_mean) or np.isnan(a_val)) else np.nan
    flag = "  *N<10" if n_dec < 10 else ""
    big_gap = "  >15pt!" if (not np.isnan(gap)) and abs(gap) > 0.15 else ""
    print(f"{cid:<10} {DISPLAY[COND_INFO[cid][0]]:<14} "
          f"{int(COND_INFO[cid][1]*100):>3}%  "
          f"{n_part:>7} {n_dec:>7}  "
          f"{h_mean:>10.3f} {a_val:>10.3f}  {gap:>+6.3f}{flag}{big_gap}")

print("\n=== Override correctness ===")
print(f"  Reject the agent's flag: right={mat[0,0]}, wrong={mat[0,1]}  "
      f"(accuracy={mat[0,0]/max(mat[0].sum(),1):.2f})")
print(f"  Add a new flag         : right={mat[1,0]}, wrong={mat[1,1]}  "
      f"(accuracy={mat[1,0]/max(mat[1].sum(),1):.2f})")
print(f"  Total overrides        : {mat.sum()}")
