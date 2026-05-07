"""plot_human_vs_agent.py — Human-vs-Agent sensitivity comparison for PredAct-CS.

Sources:
  - results_summary.csv   (per-participant per-condition human metrics from
                           evaluate_human_study.py)
  - results/exp2/exp2_per_cell.csv  (per-cell LLM-instructor metrics from Exp 2)

Outputs (in figures/):
  human_vs_agent_f1.{pdf,png}     main paper figure (F1)
  human_vs_agent_rair.{pdf,png}   appendix (RAIR)
  human_vs_agent_rsr.{pdf,png}    appendix (RSR)
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP2_CSV  = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
HUMAN_CSV = os.path.join(PROJECT_ROOT, "results_summary.csv")
OUT_DIR   = os.path.join(PROJECT_ROOT, "figures")
LLM_CUTOFFS   = [0.4, 0.5, 0.6, 0.7, 0.8]
HUMAN_CUTOFFS = [0.4, 0.6, 0.8]

# Map human-study condition_id → target accuracy. Tool identity (gpt/q9b/q35)
# is collapsed since we want one human point per accuracy aggregating across tools.
COND_ACC = {
    "gpt_40": 0.4, "gpt_60": 0.6, "gpt_80": 0.8,
    "q9b_40": 0.4, "q9b_60": 0.6, "q9b_80": 0.8,
    "q35_40": 0.4, "q35_60": 0.6, "q35_80": 0.8,
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "lines.antialiased": True, "path.simplify": True,
})


def load_data():
    llm = pd.read_csv(EXP2_CSV)
    llm = llm[llm["dataset"] == "predact_cs"].copy()    # PredAct-CS only
    hum = pd.read_csv(HUMAN_CSV)
    hum = hum[hum["has_agent"] == True].copy()
    hum["target_accuracy"] = hum["condition_id"].map(COND_ACC)
    hum = hum.dropna(subset=["target_accuracy"])
    return llm, hum


def llm_band(llm, metric):
    """For each cutoff: return (min, mean, max) across the 13 instructors."""
    out = {"x": [], "lo": [], "mid": [], "hi": []}
    for t in LLM_CUTOFFS:
        vals = llm[llm["target_accuracy"] == t][metric].dropna().values
        if len(vals) == 0:
            continue
        out["x"].append(t)
        out["lo"].append(vals.min())
        out["mid"].append(vals.mean())
        out["hi"].append(vals.max())
    return out


def human_points(hum, metric):
    """For each cutoff: mean and SE across participants × tools."""
    rows = []
    for t in HUMAN_CUTOFFS:
        sub = hum[hum["target_accuracy"] == t][metric].dropna().values
        if len(sub) == 0:
            continue
        mean = sub.mean()
        se   = sub.std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
        rows.append((t, mean, se, len(sub)))
    return rows


def make_figure(metric_llm, metric_hum, ylabel, out_name, ylim=(0, 1.02)):
    llm, hum = load_data()
    band = llm_band(llm, metric_llm)
    hpts = human_points(hum, metric_hum)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    # LLM band (min-max shaded) + mean line
    ax.fill_between(band["x"], band["lo"], band["hi"],
                    color="#5B95C9", alpha=0.22,
                    label="LLM range (min–max across 13 instructors)")
    ax.plot(band["x"], band["mid"], color="#1F5A9E", lw=2.0, marker="o",
            markersize=6, markerfacecolor="#1F5A9E", markeredgecolor="white",
            label="LLM mean")

    # Human points with SE error bars
    if hpts:
        xs = [r[0] for r in hpts]
        ms = [r[1] for r in hpts]
        es = [r[2] for r in hpts]
        ax.errorbar(xs, ms, yerr=es, fmt="D", markersize=9,
                    color="#C0392B", ecolor="#C0392B",
                    markerfacecolor="#E24B4A", markeredgecolor="#791F1F",
                    capsize=4, lw=1.4, zorder=4,
                    label=f"Human mean ± SE (n={hpts[0][3]} per cutoff)")

    ax.set_xlim(LLM_CUTOFFS[0], LLM_CUTOFFS[-1])
    ax.set_xticks(LLM_CUTOFFS)
    ax.set_xticklabels([f"{int(t*100)}%" for t in LLM_CUTOFFS])
    ax.set_xlabel("Target Tool Accuracy", fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("PredAct-CS  —  Human vs. LLM-instructor", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUT_DIR, f"{out_name}.pdf")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_pdf}")


def print_summary():
    llm, hum = load_data()
    print("\n=== Human-vs-Agent comparison summary (PredAct-CS only) ===")
    for metric_llm, metric_hum, label in [
        ("f1_final_mean", "final_f1",        "F1"),
        ("rair_mean",     "trajectory_rair", "RAIR"),
        ("rsr_mean",      "trajectory_rsr",  "RSR"),
    ]:
        print(f"\n{label}")
        for t in LLM_CUTOFFS:
            llm_vals = llm[llm["target_accuracy"] == t][metric_llm].dropna().values
            llm_str = (f"LLM range [{llm_vals.min():.2f}, {llm_vals.max():.2f}] "
                       f"mean={llm_vals.mean():.2f} (n={len(llm_vals)})"
                       if len(llm_vals) else "(no LLM data)")
            hum_vals = hum[hum["target_accuracy"] == t][metric_hum].dropna().values
            hum_str = (f"Human mean={hum_vals.mean():.2f} ± "
                       f"{hum_vals.std(ddof=1)/np.sqrt(len(hum_vals)):.2f} (n={len(hum_vals)})"
                       if len(hum_vals) else "—")
            print(f"  {int(t*100):>3d}%  |  {llm_str:<55s} |  {hum_str}")


if __name__ == "__main__":
    make_figure("f1_final_mean", "final_f1",        "F1",   "human_vs_agent_f1")
    make_figure("rair_mean",     "trajectory_rair", "RAIR", "human_vs_agent_rair")
    make_figure("rsr_mean",      "trajectory_rsr",  "RSR",  "human_vs_agent_rsr")
    print_summary()
