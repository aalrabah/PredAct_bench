"""
PredAct - Override Behavior Visualization

Reads results_pooled.csv produced by evaluate_study.py and plots the
4-bucket override behavior as a stacked bar chart per (LLM x target_accuracy) cell.

Usage:
    python plot_override.py
    python plot_override.py --pooled results_pooled.csv --out figures/override.pdf
"""

import os
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLM_DISPLAY = {
    "gpt4o_mini": "GPT-4o Mini",
    "qwen_9b": "Qwen 9B",
    "qwen_35b": "Qwen 35B",
}

# Soft, paper-friendly palette: muted greens for "good", muted warm tones for "bad".
COLORS = {
    "correct_follow":   "#6BA368",  # muted sage green — agent right, human kept
    "correct_override": "#B7D7B0",  # pale mint        — agent wrong, human dismissed
    "bad_follow":       "#E8A87C",  # soft peach       — agent wrong, human kept (FP)
    "bad_override":     "#C97064",  # dusty rose       — agent right, human dismissed (MISS)
}

LABELS = {
    "correct_follow":   "Correct follow (agent right, kept)",
    "correct_override": "Correct override (agent wrong, dismissed)",
    "bad_follow":       "Bad follow (agent wrong, kept = FP)",
    "bad_override":     "Bad override (agent right, dismissed = MISS)",
}


def load_pooled(path):
    rows = []
    no_agent_f1 = None
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["llm"] in ("no_agent", "", None):
                # Stash the no-agent F1 to use as a baseline reference line.
                try:
                    no_agent_f1 = float(r.get("pooled_f1") or "")
                except ValueError:
                    pass
                continue
            rows.append({
                "llm": r["llm"],
                "target_accuracy": float(r["target_accuracy"]),
                "total": int(r["total_decisions"]),
                "correct_follow": int(r["correct_follow"]),
                "bad_follow": int(r["bad_follow"]),
                "correct_override": int(r["correct_override"]),
                "bad_override": int(r["bad_override"]),
            })
    return rows, no_agent_f1


def plot_stacked_bars(rows, out_path, normalize=True, no_agent_f1=None):
    # Order bars: by LLM, then by target accuracy
    rows = sorted(rows, key=lambda r: (r["llm"], r["target_accuracy"]))

    # Per-bar label: just the accuracy. Model name placed once per group below.
    labels = [f"{int(r['target_accuracy']*100)}%" for r in rows]
    n = len(rows)
    x = np.arange(n)

    keys = ["correct_follow", "correct_override", "bad_follow", "bad_override"]
    if normalize:
        data = {k: np.array([r[k] / r["total"] * 100 if r["total"] else 0 for r in rows])
                for k in keys}
        ylabel = "Share of agent-flagged decisions (%)"
        ymax = 100
    else:
        data = {k: np.array([r[k] for r in rows]) for k in keys}
        ylabel = "Decisions (count)"
        ymax = max(r["total"] for r in rows) * 1.1

    # Add a small extra gap between model groups (every 3 bars in our data).
    llms_seq = [r["llm"] for r in rows]
    x_pos = []
    cursor = 0.0
    for i, llm in enumerate(llms_seq):
        if i > 0 and llm != llms_seq[i - 1]:
            cursor += 0.5    # extra space at model boundaries
        x_pos.append(cursor)
        cursor += 1.0
    x = np.array(x_pos)

    fig, ax = plt.subplots(figsize=(11, 6.0))
    bottom = np.zeros(n)
    # Slightly thinner white seams + softer alpha for a friendlier look.
    for k in keys:
        ax.bar(x, data[k], bottom=bottom, color=COLORS[k], label=LABELS[k],
               edgecolor="white", linewidth=0.8, width=0.55, alpha=0.92)
        bottom += data[k]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Group label: one model name centered under each model's set of bars.
    llms = [r["llm"] for r in rows]
    group_starts = {}     # llm -> list of bar indices
    for i, llm in enumerate(llms):
        group_starts.setdefault(llm, []).append(i)
    for llm, idxs in group_starts.items():
        center = sum(x[i] for i in idxs) / len(idxs)
        ax.text(center, -10.5, LLM_DISPLAY.get(llm, llm),
                ha="center", va="top", fontsize=12,
                transform=ax.transData)

    # No-agent baseline reference line on F1 scale (×100).
    baseline_handle = None
    if no_agent_f1 is not None and normalize:
        y_ref = no_agent_f1 * 100
        baseline_handle = ax.axhline(y_ref, color="#444", linestyle="--",
                                     linewidth=1.2, alpha=0.8,
                                     label=f"no-agent baseline F1 = {no_agent_f1:.2f}")

    # Trim x-axis so there's no trailing whitespace on the right.
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=2, fontsize=12, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.replace('.pdf', '.png')}")


# =============================================================================
# Extension: Human + 13 LLM agents — pooled and per-cutoff override figures.
# =============================================================================
import json
import glob
from collections import defaultdict

HUMAN_DIR  = os.path.join(PROJECT_ROOT, "study_logs")
EXP2_LOGS  = os.path.join(PROJECT_ROOT, "results/exp2/sim_logs")
HUMAN_COND = {"gpt_40": 0.4, "gpt_60": 0.6, "gpt_80": 0.8}
EXTENDED_CUTOFFS = (0.4, 0.6, 0.8)

# Match Figure 1 family palette.
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
MODEL_DISPLAY = {
    "gpt5_5": "GPT-5.5", "gpt5_4_mini": "GPT-5.4 Mini", "gpt4o_mini": "GPT-4o Mini",
    "claude_opus_4_7": "Claude Opus 4.7", "claude_haiku_4_5": "Claude Haiku 4.5",
    "gemini_3_1_pro": "Gemini 3.1 Pro", "gemini_3_flash": "Gemini 3 Flash",
    "deepseek_v4_pro": "DeepSeek V4 Pro", "deepseek_v4_flash": "DeepSeek V4 Flash",
    "mistral_small_24b": "Mistral Small 24B", "ministral_3_14b": "Ministral 3 14B",
    "qwen_35b": "Qwen 35B", "qwen_9b": "Qwen 9B",
}
HUMAN_KEY = "__human__"


def _buckets_from_per_student(per_student):
    """Returns (correct_follow, bad_follow, correct_override, bad_override)
    counts across all agent_flagged students with a final decision."""
    cf = bf = co = bo = 0
    for _, ps in (per_student or {}).items():
        if not ps.get("agent_flagged"):
            continue
        fd = ps.get("final_decision")
        truth = ps.get("is_at_risk")
        if fd is None or truth is None:
            continue
        kept = (fd == "accept")
        if kept and truth: cf += 1
        elif kept and not truth: bf += 1
        elif (not kept) and (not truth): co += 1
        elif (not kept) and truth: bo += 1
    return cf, bf, co, bo


def collect_extended_buckets():
    """Returns dict[rater_key][cutoff] = [cf, bf, co, bo].
    Rater keys: HUMAN_KEY for the pooled human row, otherwise an LLM key."""
    out = defaultdict(lambda: {t: [0, 0, 0, 0] for t in EXTENDED_CUTOFFS})

    # Humans (gpt_40 / gpt_60 / gpt_80, pooled across 6 participants)
    for path in sorted(glob.glob(os.path.join(HUMAN_DIR, "*.json"))):
        d = json.load(open(path))
        for cond in d.get("conditions", []):
            cid = cond.get("condition_id")
            if cid not in HUMAN_COND:
                continue
            t = HUMAN_COND[cid]
            cf, bf, co, bo = _buckets_from_per_student(cond.get("per_student", {}))
            for i, v in enumerate((cf, bf, co, bo)):
                out[HUMAN_KEY][t][i] += v

    # LLMs (filter to PredAct-CS at the matching cutoffs)
    for path in glob.glob(os.path.join(EXP2_LOGS, "*", "predact_cs_*", "run_*.json")):
        d = json.load(open(path))
        t = d.get("target_accuracy")
        if t not in EXTENDED_CUTOFFS:
            continue
        llm = d.get("instructor_llm")
        if llm not in MODEL_COLOR:
            continue
        cf, bf, co, bo = _buckets_from_per_student(d.get("per_student", {}))
        for i, v in enumerate((cf, bf, co, bo)):
            out[llm][t][i] += v
    return dict(out)


def _row_label(rk):
    return "Human" if rk == HUMAN_KEY else MODEL_DISPLAY.get(rk, rk)


def _sorted_raters(buckets):
    """Human pinned first; LLMs grouped by family (same family next to each other).
    Family order matches Figure 1; within a family, larger / more-capable variant first."""
    family_order = [
        ["gpt5_5", "gpt5_4_mini", "gpt4o_mini"],          # GPT
        ["claude_opus_4_7", "claude_haiku_4_5"],          # Claude
        ["gemini_3_1_pro", "gemini_3_flash"],             # Gemini
        ["deepseek_v4_pro", "deepseek_v4_flash"],         # DeepSeek
        ["mistral_small_24b", "ministral_3_14b"],         # Mistral
        ["qwen_35b", "qwen_9b"],                          # Qwen
    ]
    llms = [m for fam in family_order for m in fam if m in buckets]
    return [HUMAN_KEY] + llms


def _stack_one_bar(ax, x_center, width, counts):
    cf, bf, co, bo = counts
    total = cf + bf + co + bo
    if total == 0:
        return  # nothing to draw
    pct = [100 * v / total for v in (cf, co, bf, bo)]
    keys = ["correct_follow", "correct_override", "bad_follow", "bad_override"]
    bottom = 0.0
    for k, p in zip(keys, pct):
        ax.bar(x_center, p, width, bottom=bottom,
               color=COLORS[k], edgecolor="black", linewidth=0.5,
               alpha=0.92)
        bottom += p


def plot_pooled(buckets, out_path):
    raters = _sorted_raters(buckets)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(raters))
    for i, rk in enumerate(raters):
        # Sum counts across cutoffs.
        pooled = [0, 0, 0, 0]
        for t in EXTENDED_CUTOFFS:
            for j, v in enumerate(buckets[rk][t]):
                pooled[j] += v
        _stack_one_bar(ax, x[i], 0.65, pooled)
    ax.set_xticks(x)
    ax.set_xticklabels([_row_label(rk) for rk in raters], fontsize=10,
                       rotation=35, ha="right")
    ax.set_ylabel("Share of agent-flagged decisions (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in
               ("correct_follow", "correct_override", "bad_follow", "bad_override")]
    labels = [LABELS[k] for k in
              ("correct_follow", "correct_override", "bad_follow", "bad_override")]
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=10)
    ax.set_title("Pooled across 40 / 60 / 80 % cutoffs", fontsize=11, color="#555",
                 pad=28)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_by_cutoff(buckets, out_path):
    raters = _sorted_raters(buckets)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    n_raters = len(raters)
    n_cuts   = len(EXTENDED_CUTOFFS)
    bar_w    = 0.22
    group_gap = 1.4   # x-spacing between rater groups
    centers = []
    for i, rk in enumerate(raters):
        base = i * group_gap
        for j, t in enumerate(EXTENDED_CUTOFFS):
            xc = base + (j - 1) * bar_w
            _stack_one_bar(ax, xc, bar_w * 0.95, buckets[rk][t])
            ax.text(xc, -3, f"{int(t*100)}", ha="center", va="top",
                    fontsize=7, color="#666")
        centers.append(base)
        if i < n_raters - 1:
            ax.axvline(base + group_gap / 2, linestyle=":", color="#CCC",
                       linewidth=0.6, zorder=0)
    ax.set_xticks(centers)
    ax.set_xticklabels([_row_label(rk) for rk in raters], fontsize=10,
                       rotation=35, ha="right")
    ax.set_ylabel("Share of agent-flagged decisions (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in
               ("correct_follow", "correct_override", "bad_follow", "bad_override")]
    labels = [LABELS[k] for k in
              ("correct_follow", "correct_override", "bad_follow", "bad_override")]
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=10)
    ax.set_title("Three thin bars per rater = 40 % / 60 % / 80 %",
                 fontsize=11, color="#555")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_extended_table(buckets):
    raters = _sorted_raters(buckets)
    print(f"\n{'Rater':<22} {'cutoff':<7} "
          f"{'CorrFol':>8} {'BadFol':>8} {'CorrOvr':>8} {'BadOvr':>8} "
          f"{'(% pct)':<24}")
    print("-" * 95)
    for rk in raters:
        for t in EXTENDED_CUTOFFS:
            cf, bf, co, bo = buckets[rk][t]
            tot = cf + bf + co + bo
            pct = (lambda v: f"{100*v/tot:.0f}%" if tot else "—")
            print(f"{_row_label(rk):<22} {int(t*100):>3}%   "
                  f"{cf:>8} {bf:>8} {co:>8} {bo:>8}   "
                  f"{pct(cf)}/{pct(co)}/{pct(bf)}/{pct(bo)}")
        # pooled row
        pooled = [sum(buckets[rk][t][i] for t in EXTENDED_CUTOFFS) for i in range(4)]
        cf, bf, co, bo = pooled
        tot = cf + bf + co + bo
        pct = (lambda v: f"{100*v/tot:.0f}%" if tot else "—")
        print(f"{_row_label(rk):<22} {'pool':>4}   "
              f"{cf:>8} {bf:>8} {co:>8} {bo:>8}   "
              f"{pct(cf)}/{pct(co)}/{pct(bf)}/{pct(bo)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled", default=os.path.join(PROJECT_ROOT, "results_pooled.csv"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "figures", "override_behavior.pdf"))
    parser.add_argument("--counts", action="store_true", help="Plot raw counts instead of percentages")
    args = parser.parse_args()

    # Existing figure (humans only, 9 LLM-tool×accuracy bars).
    if os.path.exists(args.pooled):
        rows, no_agent_f1 = load_pooled(args.pooled)
        if rows:
            plot_stacked_bars(rows, args.out, normalize=not args.counts,
                              no_agent_f1=no_agent_f1)

    # Extended figures: humans + 13 LLM agents.
    print("\nLoading extended (human + 13 LLM) override data...")
    buckets = collect_extended_buckets()
    print_extended_table(buckets)
    plot_pooled(buckets, os.path.join(PROJECT_ROOT, "figures", "override_pooled.pdf"))
    plot_by_cutoff(buckets, os.path.join(PROJECT_ROOT, "figures", "override_by_cutoff.pdf"))


if __name__ == "__main__":
    main()
