"""plot_human_vs_gpt4o.py — humans-with-GPT-4o-Mini vs all 13 LLM agents.

Generates 4 alternative visualizations of the same comparison:
  Option A — strip plot, 3 panels, LLM dots + human marker
  Option B — caterpillar / forest plot, 14 rows per panel
  Option C — slope chart (LLM lines + human line)
  Option D — distribution strip plot:
              gray = all PredAct-CS agent episodes at cutoff
              blue = the 30 gpt4o_mini agent episodes (10 per cutoff)
              orange = each human participant + pooled mean ± SE

Output: figures/human_vs_gpt4o/{A,B,C,D}_*.{pdf,png}
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP2_CSV   = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
EXP2_LOGS  = os.path.join(PROJECT_ROOT, "results/exp2/sim_logs")
HUMAN_CSV  = os.path.join(PROJECT_ROOT, "results_per_participant.csv")   # produced by evaluate_human_study_v2.py
OUT_DIR    = os.path.join(PROJECT_ROOT, "figures/human_vs_gpt4o")

CUTOFFS = [0.4, 0.6, 0.8]            # human study only ran these
ALL_CUT = [0.4, 0.5, 0.6, 0.7, 0.8]  # full LLM cutoffs

DISPLAY = {
    "claude_haiku_4_5":  "Claude Haiku 4.5",
    "claude_opus_4_7":   "Claude Opus 4.7",
    "deepseek_v4_flash": "DeepSeek V4 Flash",
    "deepseek_v4_pro":   "DeepSeek V4 Pro",
    "gemini_3_1_pro":    "Gemini 3.1 Pro",
    "gemini_3_flash":    "Gemini 3 Flash",
    "gpt4o_mini":        "GPT-4o Mini",
    "gpt5_4_mini":       "GPT-5.4 Mini",
    "gpt5_5":            "GPT-5.5",
    "ministral_3_14b":   "Ministral 3 14B",
    "mistral_small_24b": "Mistral Small 24B",
    "qwen_9b":           "Qwen 9B",
    "qwen_35b":          "Qwen 35B",
}
HUMAN_COND = {"gpt_40": 0.4, "gpt_60": 0.6, "gpt_80": 0.8}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Per-episode F1 for the agent side (computed from per_student dumps)
# ---------------------------------------------------------------------------
def _derived_flag(ps):
    af, fd = ps.get("agent_flagged"), ps.get("final_decision")
    if fd is None:
        return False           # treat as not flagged for F1 computation
    return (af and fd == "accept") or ((not af) and fd == "reject")


def episode_f1(per_student):
    tp = fp = fn = 0
    for sid, ps in per_student.items():
        h = _derived_flag(ps)
        truth = bool(ps.get("is_at_risk"))
        if h and truth:    tp += 1
        elif h:            fp += 1
        elif truth:        fn += 1
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return 2 * p * r / (p + r) if (p + r) else 0


def load_agent_episodes():
    """Return DataFrame: instructor_llm, dataset, target_accuracy, run_idx, f1."""
    rows = []
    for f in glob.glob(os.path.join(EXP2_LOGS, "*", "predact_cs_*", "run_*.json")):
        d = json.load(open(f))
        rows.append({
            "instructor_llm":  d.get("instructor_llm"),
            "dataset":         d.get("dataset"),
            "target_accuracy": d.get("target_accuracy"),
            "run_idx":         d.get("run_idx"),
            "f1":              episode_f1(d.get("per_student", {})),
        })
    return pd.DataFrame(rows)


def load_human_per_participant():
    df = pd.read_csv(HUMAN_CSV)
    df = df[df["condition_id"].isin(HUMAN_COND)].copy()
    df["target_accuracy"] = df["condition_id"].map(HUMAN_COND)
    return df[["participant_name", "condition_id", "target_accuracy",
               "final_f1"]]


# ---------------------------------------------------------------------------
# OPTION A — strip plot per cutoff (LLM dots + human marker)
# ---------------------------------------------------------------------------
def plot_A(agent_df, human_df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, t in zip(axes, CUTOFFS):
        sub_a = (agent_df[agent_df["target_accuracy"] == t]
                 .groupby("instructor_llm")["f1"].mean().reset_index())
        x_jit = rng.uniform(-0.18, 0.18, size=len(sub_a))
        ax.scatter(x_jit, sub_a["f1"], s=70, color="#5B9BD5",
                   edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3,
                   label="LLM agent (mean F1, one per model)")
        # Human pooled mean ± SE
        h_vals = human_df[human_df["target_accuracy"] == t]["final_f1"].values
        h_mean = h_vals.mean() if len(h_vals) else float("nan")
        h_se   = h_vals.std(ddof=1) / np.sqrt(len(h_vals)) if len(h_vals) > 1 else 0.0
        ax.errorbar(0, h_mean, yerr=h_se, fmt="D", markersize=12,
                    color="#C97064", markerfacecolor="#E8A87C",
                    markeredgecolor="#791F1F", capsize=5, lw=1.6, zorder=4,
                    label=f"Human + GPT-4o Mini (mean ± SE, n={len(h_vals)})")
        ax.axhline(h_mean, color="#C97064", linestyle=":", lw=0.8, alpha=0.6)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(f"Tool accuracy = {int(t*100)}%", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("F1", fontsize=12)
    axes[0].set_ylim(0, 1.0)
    axes[-1].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Option A — humans-with-GPT-4o-Mini vs 13 LLM agents", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "A_strip_panels.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# OPTION B — caterpillar / forest plot, 14 rows per panel
# ---------------------------------------------------------------------------
def plot_B(agent_df, human_df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharex=True)
    for ax, t in zip(axes, CUTOFFS):
        agg = (agent_df[agent_df["target_accuracy"] == t]
               .groupby("instructor_llm")["f1"].agg(["mean", "std"]).reset_index())
        agg["model"] = agg["instructor_llm"].map(DISPLAY)

        h_vals = human_df[human_df["target_accuracy"] == t]["final_f1"].values
        h_mean = h_vals.mean() if len(h_vals) else float("nan")
        h_se   = h_vals.std(ddof=1) / np.sqrt(len(h_vals)) if len(h_vals) > 1 else 0.0
        rows = [{"model": "Human + GPT-4o Mini", "mean": h_mean, "std": h_se,
                 "is_human": True}]
        for _, r in agg.iterrows():
            rows.append({"model": r["model"], "mean": r["mean"], "std": r["std"],
                         "is_human": False})
        rows.sort(key=lambda r: r["mean"])

        y = np.arange(len(rows))
        for i, r in enumerate(rows):
            color = "#C97064" if r["is_human"] else "#5B9BD5"
            ax.errorbar(r["mean"], i, xerr=r["std"], fmt="o", markersize=7,
                        color=color, markerfacecolor=color,
                        markeredgecolor="black", capsize=3,
                        lw=1.2, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([r["model"] for r in rows], fontsize=8)
        ax.set_title(f"Tool accuracy = {int(t*100)}%", fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_xlabel(""); axes[1].set_xlabel("F1", fontsize=11); axes[2].set_xlabel("")
    fig.suptitle("Option B — caterpillar plot: each model + human, sorted by F1",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "B_caterpillar.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# OPTION C — slope chart (13 LLM lines + 1 human line)
# ---------------------------------------------------------------------------
def plot_C(agent_df, human_df):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for m in DISPLAY:
        ys = []
        for t in ALL_CUT:
            v = (agent_df[(agent_df["instructor_llm"] == m)
                          & (agent_df["target_accuracy"] == t)]["f1"].mean())
            ys.append(v)
        ax.plot(ALL_CUT, ys, color="#9DB4C0", lw=1.0, alpha=0.6)

    # Human line
    h_xs, h_ys, h_ses = [], [], []
    for t in CUTOFFS:
        v = human_df[human_df["target_accuracy"] == t]["final_f1"].values
        if len(v):
            h_xs.append(t); h_ys.append(v.mean())
            h_ses.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    ax.errorbar(h_xs, h_ys, yerr=h_ses, fmt="-D", color="#C97064",
                markersize=9, lw=2.2, capsize=5,
                markerfacecolor="#E8A87C", markeredgecolor="#791F1F",
                zorder=4, label="Human + GPT-4o Mini (mean ± SE)")
    ax.set_xticks(ALL_CUT)
    ax.set_xticklabels([f"{int(t*100)}%" for t in ALL_CUT])
    ax.set_xlabel("Target tool accuracy", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [Line2D([0], [0], color="#9DB4C0", lw=1.0, label="Each LLM agent (13 lines)")]
    hh, ll = ax.get_legend_handles_labels()
    ax.legend(handles + hh, ["Each LLM agent (13 lines)"] + ll,
              loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Option C — slope chart, all 13 LLMs and human", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "C_slope.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# OPTION D — distribution strip plot (the user's pick)
#   gray  = all PredAct-CS agent episodes at cutoff (full distribution)
#   blue  = the 30 gpt4o_mini agent episodes (10 per cutoff)
#   orange= each human participant's F1 + pooled mean ± SE
# ---------------------------------------------------------------------------
def plot_D(agent_df, human_df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rng = np.random.default_rng(7)
    panel_x = {0.4: 0, 0.6: 1, 0.8: 2}
    legend_handles = []

    for t, x_center in panel_x.items():
        # All agent episodes (gray background)
        all_eps = agent_df[agent_df["target_accuracy"] == t]["f1"].values
        x_all = x_center + rng.uniform(-0.30, 0.30, size=len(all_eps))
        h_gray = ax.scatter(x_all, all_eps, s=14, color="#BBB", alpha=0.55,
                            edgecolor="none", zorder=2)

        # gpt4o_mini agent episodes (blue overlay)
        gpt_eps = agent_df[(agent_df["target_accuracy"] == t)
                           & (agent_df["instructor_llm"] == "gpt4o_mini")]["f1"].values
        x_gpt = x_center + rng.uniform(-0.18, 0.18, size=len(gpt_eps))
        h_blue = ax.scatter(x_gpt, gpt_eps, s=42, color="#5B9BD5",
                            edgecolor="black", linewidth=0.4, alpha=0.92, zorder=3)

        # Each human participant (orange dot)
        hh = human_df[human_df["target_accuracy"] == t]
        x_h = x_center + rng.uniform(-0.10, 0.10, size=len(hh))
        h_orange = ax.scatter(x_h, hh["final_f1"].values, s=64,
                              color="#E8A87C", edgecolor="#791F1F",
                              linewidth=0.6, alpha=0.95, zorder=4, marker="o")

        # Pooled human mean ± SE (diamond + error bar)
        if len(hh):
            m = hh["final_f1"].mean()
            se = hh["final_f1"].std(ddof=1) / np.sqrt(len(hh)) if len(hh) > 1 else 0.0
            h_diamond = ax.errorbar(x_center, m, yerr=se, fmt="D", markersize=14,
                                    color="#C97064", markerfacecolor="#C97064",
                                    markeredgecolor="black", capsize=6, lw=2,
                                    zorder=5)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBB",
               markeredgecolor="none", markersize=6,
               label=f"All agent episodes (n={len(agent_df)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#5B9BD5",
               markeredgecolor="black", markersize=8,
               label="GPT-4o Mini agent episodes (10/cutoff)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8A87C",
               markeredgecolor="#791F1F", markersize=8,
               label="Each human participant"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#C97064",
               markeredgecolor="black", markersize=10,
               label="Human pooled mean ± SE"),
    ]
    ax.set_xticks(list(panel_x.values()))
    ax.set_xticklabels([f"{int(t*100)}%" for t in panel_x])
    ax.set_xlabel("Target tool accuracy", fontsize=12)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.6, 2.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10)
    ax.set_title("Option D — distribution strip: full agent cloud, "
                 "GPT-4o Mini overlay, and humans",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "D_distribution_strip.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading agent episodes...")
    agent_df = load_agent_episodes()
    print(f"  {len(agent_df)} PredAct-CS episodes loaded")
    print("Loading human study results...")
    human_df = load_human_per_participant()
    print(f"  {len(human_df)} human×condition rows (3 cutoffs × 6 participants)")
    print()
    plot_A(agent_df, human_df)
    plot_B(agent_df, human_df)
    plot_C(agent_df, human_df)
    plot_D(agent_df, human_df)
    print("\nAll 4 figures generated in figures/human_vs_gpt4o/")


if __name__ == "__main__":
    main()
