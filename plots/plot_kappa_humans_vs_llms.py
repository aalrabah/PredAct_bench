"""plot_kappa_humans_vs_llms.py — Cohen's kappa between aggregated humans
and each of 13 instructor LLMs on PredAct-CS, pooled across cutoffs."""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import cohen_kappa_score

HUMAN_DIR = os.path.join(PROJECT_ROOT, "study_logs")
EXP2_LOGS = os.path.join(PROJECT_ROOT, "results/exp2/sim_logs")
OUT_PDF   = os.path.join(PROJECT_ROOT, "figures/kappa_humans_vs_llms.pdf")

# (course, week) → cutoff (matches human study + exp2_config.py)
CELLS = {
    ("Course_A", 8): 0.4,
    ("Course_B", 8): 0.6,
    ("Course_C", 8): 0.8,
}
HUMAN_COND = {"gpt_40", "gpt_60", "gpt_80"}
HUMAN_MAJORITY_THRESHOLD = 3   # >=3 of 6 = "accept"

# Same family palette as Figure 1 (visualize_exp2_2.py)
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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def derive(ps):
    af, fd = ps.get("agent_flagged"), ps.get("final_decision")
    if fd is None:
        return None
    return (af and fd == "accept") or ((not af) and fd == "reject")


# ---------------------------------------------------------------------------
# 1. Aggregate human flags by majority vote per (cutoff, student).
# ---------------------------------------------------------------------------
def collect_human_majority():
    votes = defaultdict(lambda: defaultdict(list))   # cutoff -> sid -> list[bool]
    truth = {}
    for path in sorted(glob.glob(os.path.join(HUMAN_DIR, "*.json"))):
        d = json.load(open(path))
        for cond in d.get("conditions", []):
            if cond.get("condition_id") not in HUMAN_COND:
                continue
            key = (cond.get("course_id"), cond.get("week"))
            if key not in CELLS:
                continue
            cutoff = CELLS[key]
            for sid, ps in cond.get("per_student", {}).items():
                v = derive(ps)
                if v is None:
                    continue
                votes[cutoff][sid].append(v)
                truth[sid] = bool(ps.get("is_at_risk"))
    aggregated = {}    # (cutoff, sid) -> True/False
    for cutoff, sids in votes.items():
        for sid, vs in sids.items():
            aggregated[(cutoff, sid)] = sum(vs) >= HUMAN_MAJORITY_THRESHOLD
    return aggregated, truth


# ---------------------------------------------------------------------------
# 2. Per-LLM flag for each (cutoff, sid): True if any episode at that cutoff
#    had the LLM-instructor's final decision = flagged.
# ---------------------------------------------------------------------------
def collect_llm_flags():
    """For each (llm, cutoff): set of sids the LLM flagged in ANY of its 10
    PredAct-CS episodes. Explicitly restricted to the human-study cutoffs."""
    allowed_cutoffs = set(CELLS.values())     # {0.4, 0.6, 0.8}
    out = defaultdict(lambda: defaultdict(set))
    out_seen = defaultdict(lambda: defaultdict(set))   # sids the LLM saw at all
    for path in glob.glob(os.path.join(EXP2_LOGS, "*", "predact_cs_*", "run_*.json")):
        d = json.load(open(path))
        cutoff = d.get("target_accuracy")
        if cutoff not in allowed_cutoffs:
            continue
        llm = d.get("instructor_llm")
        for sid, ps in d.get("per_student", {}).items():
            out_seen[llm][cutoff].add(sid)
            v = derive(ps)
            if v:
                out[llm][cutoff].add(sid)
    return out, out_seen


# ---------------------------------------------------------------------------
# 3. Compute Cohen's kappa.
# ---------------------------------------------------------------------------
def main():
    human_agg, _ = collect_human_majority()
    llm_flags, llm_seen = collect_llm_flags()

    keys = sorted(human_agg.keys())
    human_vec = np.array([human_agg[k] for k in keys], dtype=int)

    print("=" * 80)
    print("DATA INVENTORY")
    print("=" * 80)
    print(f"Human side:")
    print(f"  Conditions used: gpt_40, gpt_60, gpt_80 across 6 participants")
    print(f"  Cells: Course_A w8 (40%), Course_B w8 (60%), Course_C w8 (80%)")
    print(f"  Aggregation: majority vote, threshold >= {HUMAN_MAJORITY_THRESHOLD} of 6")
    print(f"  Items (cutoff, student): {len(keys)} total")
    by_cutoff = defaultdict(int)
    for c, _ in keys:
        by_cutoff[c] += 1
    for c in sorted(by_cutoff):
        n_pos = sum(1 for k in keys if k[0] == c and human_agg[k])
        print(f"    {int(c*100)}%: {by_cutoff[c]} students, {n_pos} flagged by majority")
    print(f"\nAgent side:")
    print(f"  Filter: dataset == 'predact_cs' AND target_accuracy in [0.4, 0.6, 0.8]")
    print(f"  Each LLM × cutoff: 10 PredAct-CS episodes; 'flag' = LLM flagged sid in any of those 10")

    print("\n" + "=" * 80)
    print("PER-MODEL DETAIL")
    print("=" * 80)
    print(f"{'Model':<20} {'kappa':>8}   {'N':>3}   {'agree':>5}   {'matched_seen':>13}")

    results = []
    for llm in MODEL_COLOR:
        agent_vec = np.array([(k[1] in llm_flags.get(llm, {}).get(k[0], set()))
                              for k in keys], dtype=int)
        seen_count = sum(1 for k in keys if k[1] in llm_seen.get(llm, {}).get(k[0], set()))
        try:
            kp = cohen_kappa_score(human_vec, agent_vec)
        except Exception:
            kp = float("nan")
        agree = int(np.sum(human_vec == agent_vec))
        results.append((llm, kp, len(keys), agree, seen_count))

    results.sort(key=lambda r: -r[1] if not np.isnan(r[1]) else 1)
    for llm, kp, n, agree, seen in results:
        print(f"  {DISPLAY[llm]:<18} {kp:>8.3f}   {n:>3}   {agree}/{n:>2}   "
              f"{seen}/{n} (LLM episodes saw this sid)")
    n = len(results)

    # ---- plot ----
    n = len(results)
    fig, ax = plt.subplots(figsize=(8, 0.45 * n + 1.5))
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.5, n - 0.5)

    # Highlight band for GPT-4o Mini row
    for i, (llm, k, *_rest) in enumerate(results):
        if llm == "gpt4o_mini":
            ax.axhspan(i - 0.45, i + 0.45, color="#FBE4EC", alpha=0.6, zorder=0)

    bar_height = 0.55
    for i, (llm, k, *_rest) in enumerate(results):
        color = MODEL_COLOR[llm]
        # Rounded bar via FancyBboxPatch
        bar = FancyBboxPatch(
            (0, i - bar_height / 2),
            max(k, 0), bar_height,
            boxstyle="round,pad=0,rounding_size=0.04",
            linewidth=0.6, edgecolor="black",
            facecolor=color, alpha=0.92, zorder=2,
        )
        ax.add_patch(bar)
        ax.text(max(k, 0) + 0.012, i, f"{k:.2f}",
                va="center", ha="left", fontsize=10, color="#333")

    # Y-tick labels
    labels = []
    for llm, *_ in results:
        if llm == "gpt4o_mini":
            labels.append(f"GPT-4o Mini (apples-to-apples)")
        else:
            labels.append(DISPLAY[llm])
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    # Bold the GPT-4o Mini tick label
    for tick, (llm, *_) in zip(ax.get_yticklabels(), results):
        if llm == "gpt4o_mini":
            tick.set_fontweight("bold")

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Cohen's kappa  (humans vs LLM, majority vote)", fontsize=11)

    # Dashed vertical gridlines at 0.2 intervals
    for x in np.arange(0, 0.81, 0.2):
        ax.axvline(x, color="#CCC", linestyle="--", linewidth=0.5, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.suptitle("PredAct-CS, pooled across cutoffs (40%, 60%, 80%)",
                 fontsize=10, color="#555", y=0.06)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
