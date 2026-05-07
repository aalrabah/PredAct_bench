"""plot_rair_rsr_human_vs_llm.py — RAIR/RSR box plots, humans + 13 LLM agents."""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

HUMAN_DIR = os.path.join(PROJECT_ROOT, "study_logs")
EXP2_LOGS = os.path.join(PROJECT_ROOT, "results/exp2/sim_logs")
OUT_PDF   = os.path.join(PROJECT_ROOT, "figures/rair_rsr_human_vs_llm.pdf")

CUTOFFS = {0.4, 0.6, 0.8}
HUMAN_COND = {"gpt_40": 0.4, "gpt_60": 0.6, "gpt_80": 0.8}

MODEL_COLOR = {
    "gpt5_5":            "#D85F7C",
    "gpt5_4_mini":       "#E8849A",
    "gpt4o_mini":        "#F8C8D2",
    "claude_opus_4_7":   "#F4B07A",
    "claude_haiku_4_5":  "#FCE0C2",
    "gemini_3_1_pro":    "#7AB8D6",
    "gemini_3_flash":    "#CDE7F0",
    "deepseek_v4_pro":   "#7CCCAB",
    "deepseek_v4_flash": "#D4F0E0",
    "mistral_small_24b": "#A38FCE",
    "ministral_3_14b":   "#E0D5EE",
    "qwen_35b":          "#C9A98B",
    "qwen_9b":           "#F0E0D0",
}
DISPLAY = {
    "gpt5_5": "GPT-5.5", "gpt5_4_mini": "GPT-5.4 Mini", "gpt4o_mini": "GPT-4o Mini",
    "claude_opus_4_7": "Claude Opus 4.7", "claude_haiku_4_5": "Claude Haiku 4.5",
    "gemini_3_1_pro": "Gemini 3.1 Pro", "gemini_3_flash": "Gemini 3 Flash",
    "deepseek_v4_pro": "DeepSeek V4 Pro", "deepseek_v4_flash": "DeepSeek V4 Flash",
    "mistral_small_24b": "Mistral Small 24B", "ministral_3_14b": "Ministral 3 14B",
    "qwen_35b": "Qwen 35B", "qwen_9b": "Qwen 9B",
}
HUMAN_COLOR = "#4A4A4A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Trajectory metric: per-episode/condition RAIR + RSR (Schemmer 2023).
# ---------------------------------------------------------------------------
def trajectory(per_student):
    rair_n = rair_d = rsr_n = rsr_d = 0
    for _, ps in per_student.items():
        if not ps.get("agent_flagged"):
            continue
        init = ps.get("initial_decision")
        fin  = ps.get("final_decision")
        truth = ps.get("is_at_risk")
        if init is None or fin is None or truth is None:
            continue
        init_correct  = (init == "accept") == truth
        final_correct = (fin == "accept")  == truth
        agent_correct = truth
        if (not init_correct) and agent_correct:
            rair_d += 1
            if final_correct: rair_n += 1
        if init_correct and (not agent_correct):
            rsr_d += 1
            if final_correct: rsr_n += 1
    rair = rair_n / rair_d if rair_d else None
    rsr  = rsr_n  / rsr_d  if rsr_d  else None
    return rair, rsr


# ---------------------------------------------------------------------------
# Collect human (participant × cutoff) values
# ---------------------------------------------------------------------------
def collect_human():
    rair_vals, rsr_vals = [], []
    for path in glob.glob(os.path.join(HUMAN_DIR, "*.json")):
        d = json.load(open(path))
        for cond in d.get("conditions", []):
            cid = cond.get("condition_id")
            if cid not in HUMAN_COND:
                continue
            ra, rs = trajectory(cond.get("per_student", {}) or {})
            if ra is not None: rair_vals.append(ra)
            if rs is not None: rsr_vals.append(rs)
    return rair_vals, rsr_vals


# ---------------------------------------------------------------------------
# Collect per-episode LLM values
# ---------------------------------------------------------------------------
def collect_llm():
    rair = defaultdict(list)
    rsr  = defaultdict(list)
    for path in glob.glob(os.path.join(EXP2_LOGS, "*", "predact_cs_*", "run_*.json")):
        d = json.load(open(path))
        if d.get("target_accuracy") not in CUTOFFS:
            continue
        llm = d.get("instructor_llm")
        if llm not in MODEL_COLOR:
            continue
        ra, rs = trajectory(d.get("per_student", {}) or {})
        if ra is not None: rair[llm].append(ra)
        if rs is not None: rsr[llm].append(rs)
    return rair, rsr


def median_iqr(vals):
    if not vals:
        return None, None, None
    a = np.array(vals)
    return float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))


def main():
    h_rair, h_rsr = collect_human()
    l_rair, l_rsr = collect_llm()

    # Sort LLMs by median RAIR descending; humans pinned to top.
    medians = {llm: (median_iqr(l_rair[llm])[0] or -1) for llm in MODEL_COLOR}
    sorted_llms = sorted(MODEL_COLOR.keys(), key=lambda m: -medians[m])

    # ---- Console ----
    def fmt(v): return "—" if v is None else f"{v:.3f}"
    print(f"{'Row':<22} {'RAIR med [Q1, Q3]':<30} {'RSR med [Q1, Q3]':<30}  N_RAIR  N_RSR")
    hr_m, hr_lo, hr_hi = median_iqr(h_rair)
    rs_m, rs_lo, rs_hi = median_iqr(h_rsr)
    print(f"{'Human (pooled)':<22} "
          f"{fmt(hr_m)} [{fmt(hr_lo)}, {fmt(hr_hi)}]   "
          f"{fmt(rs_m)} [{fmt(rs_lo)}, {fmt(rs_hi)}]    "
          f"{len(h_rair):>3}    {len(h_rsr):>3}")
    for llm in sorted_llms:
        ra_m, ra_lo, ra_hi = median_iqr(l_rair[llm])
        rs_m_, rs_lo_, rs_hi_ = median_iqr(l_rsr[llm])
        print(f"{DISPLAY[llm]:<22} "
              f"{fmt(ra_m)} [{fmt(ra_lo)}, {fmt(ra_hi)}]   "
              f"{fmt(rs_m_)} [{fmt(rs_lo_)}, {fmt(rs_hi_)}]    "
              f"{len(l_rair[llm]):>3}    {len(l_rsr[llm]):>3}")

    # ---- Plot ----
    rows = ["__human__"] + sorted_llms
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * n + 1.6), sharey=True)

    def draw(ax, datasets, title):
        positions = list(range(n, 0, -1))   # human at top
        bp = ax.boxplot(
            datasets, positions=positions, vert=False, widths=0.55,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="#666", linewidth=1.0),
            capprops=dict(color="#666", linewidth=1.0),
            boxprops=dict(linewidth=0.6),
        )
        for patch, row in zip(bp["boxes"], rows):
            color = HUMAN_COLOR if row == "__human__" else MODEL_COLOR[row]
            patch.set_facecolor(color)
            patch.set_alpha(0.92)
            patch.set_edgecolor("black")
        ax.set_yticks(positions)
        ax.set_yticklabels(["Human" if r == "__human__" else DISPLAY[r] for r in rows],
                           fontsize=10)
        for tick, r in zip(ax.get_yticklabels(), rows):
            if r == "__human__":
                tick.set_fontweight("bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("higher = better", fontsize=10, color="#666", style="italic")
        ax.set_title(title, fontsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", linestyle=":", color="#CCC", alpha=0.7)
        ax.set_axisbelow(True)

    rair_data = [h_rair] + [l_rair[m] for m in sorted_llms]
    rsr_data  = [h_rsr]  + [l_rsr[m]  for m in sorted_llms]
    draw(axes[0], rair_data, "RAIR — fix wrong gut answer")
    draw(axes[1], rsr_data,  "RSR — resist bad agent advice")

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
