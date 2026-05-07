"""plot_human_vs_agent_paired.py — paired bar chart, GPT-4o Mini agent vs human."""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXP2_CSV  = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
HUMAN_DIR = os.path.join(PROJECT_ROOT, "study_logs")
OUT_PDF   = os.path.join(PROJECT_ROOT, "figures/human_vs_agent_paired.pdf")

CUTOFFS  = [0.4, 0.6, 0.8]
HUMAN_COND = {"gpt_40": 0.4, "gpt_60": 0.6, "gpt_80": 0.8}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Per-decision human data: list of (predicted, truth) for each cutoff.
# ---------------------------------------------------------------------------
def derive(ps):
    af, fd = ps.get("agent_flagged"), ps.get("final_decision")
    if fd is None:
        return None
    return (af and fd == "accept") or ((not af) and fd == "reject")


def collect_human_decisions():
    decisions = {t: [] for t in CUTOFFS}
    for path in glob.glob(os.path.join(HUMAN_DIR, "*.json")):
        d = json.load(open(path))
        for cond in d.get("conditions", []):
            cid = cond.get("condition_id")
            if cid not in HUMAN_COND:
                continue
            t = HUMAN_COND[cid]
            for sid, ps in cond.get("per_student", {}).items():
                pred = derive(ps)
                if pred is None:
                    continue
                truth = bool(ps.get("is_at_risk"))
                decisions[t].append((pred, truth))
    return decisions


def f1_from_pairs(pairs):
    tp = sum(1 for p, t in pairs if p and t)
    fp = sum(1 for p, t in pairs if p and not t)
    fn = sum(1 for p, t in pairs if (not p) and t)
    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0


def bootstrap_ci(pairs, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.arange(n)
    samples = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        samples.append(f1_from_pairs([pairs[i] for i in s]))
    samples = np.array(samples)
    return f1_from_pairs(pairs), float(np.quantile(samples, alpha/2)), float(np.quantile(samples, 1 - alpha/2))


def main():
    # Agent side: aggregated GPT-4o Mini PredAct-CS rows
    df = pd.read_csv(EXP2_CSV)
    df = df[(df["instructor_llm"] == "gpt4o_mini") & (df["dataset"] == "predact_cs")]
    agent = {float(r.target_accuracy): (float(r.f1_final_mean), float(r.f1_final_std))
             for _, r in df.iterrows()}

    # Human side: per-decision pairs, then bootstrap
    decisions = collect_human_decisions()
    human = {t: bootstrap_ci(decisions[t]) for t in CUTOFFS}

    # ---- Console report ----
    print(f"{'cutoff':<8}{'agent F1 ± std':<22}{'human F1 [95% CI]':<28}{'higher':<8}")
    for t in CUTOFFS:
        am, asd = agent[t]
        hm, hlo, hhi = human[t]
        winner = "agent" if am > hm else "human" if hm > am else "tie"
        print(f"{int(t*100):>4}%   "
              f"{am:.3f} ± {asd:.3f}        "
              f"{hm:.3f}  [{hlo:.3f}, {hhi:.3f}]    {winner}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(CUTOFFS))
    w = 0.35
    a_means = [agent[t][0] for t in CUTOFFS]
    a_stds  = [agent[t][1] for t in CUTOFFS]
    h_means = [human[t][0] for t in CUTOFFS]
    h_lo    = [human[t][0] - human[t][1] for t in CUTOFFS]
    h_hi    = [human[t][2] - human[t][0] for t in CUTOFFS]

    b1 = ax.bar(x - w/2, a_means, w, yerr=a_stds, capsize=3,
                color="#9CC4E8", edgecolor="#3A7AB0", linewidth=0.6,
                label="GPT-4o Mini agent (mean ± std, 10 episodes)")
    b2 = ax.bar(x + w/2, h_means, w, yerr=[h_lo, h_hi], capsize=3,
                color="#F4C99B", edgecolor="#A06A2C", linewidth=0.6,
                label="Human + GPT-4o Mini (mean, 95% bootstrap CI)")

    for bar, val in zip(b1, a_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, color="#234A6B")
    for bar, val in zip(b2, h_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.025, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, color="#7A4F1E")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}%" for t in CUTOFFS])
    ax.set_xlabel("Target tool accuracy", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {OUT_PDF}")
    print("Caption note: N=6 humans, 5 flagged decisions per cell (30 decisions per cutoff).")


if __name__ == "__main__":
    main()
