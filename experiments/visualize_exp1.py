"""
Exp 1 — Visualization

Reads exp1_raw.csv and produces two PNGs:
  - exp1_reliability_clean.png   Guo-style reliability diagram, one panel per
                                 dataset. Blue bars = empirical at-risk rate,
                                 red hatched bars = calibration gap, dashed
                                 diagonal = perfect calibration.
  - exp1_over_time.png           Accuracy and ECE across the semester
                                 (two-panel line plot, one line per dataset).

X-axis on companion plot: percent of semester completed (10-100%).

Usage:
    python visualize_exp1.py
    python visualize_exp1.py --input /path/to/exp1_raw.csv --output-dir /path/to/out
"""

import os
import csv
import argparse
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "results", "exp1", "exp1_raw.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "exp1")

PCT_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
AT_RISK_GRADES = {"d", "f"}

# Canonical display names — anything else gets mapped to one of these.
DISPLAY_NAME = {
    "PredAct-CS": "PredAct",
    "predact-cs": "PredAct",
    "predact": "PredAct",
    "uiuc":    "PredAct",
    "UIUC":    "PredAct",
    "oulad":   "OULAD",
    "OULAD":   "OULAD",
}

COLORS = {
    "PredAct": "#1f77b4",
    "OULAD":   "#d62728",
}


# =============================================================================
# HELPERS
# =============================================================================

def canon_dataset(name):
    return DISPLAY_NAME.get(name, name)


def pct_to_bin(pct):
    for i in range(len(PCT_BIN_EDGES) - 1):
        lo, hi = PCT_BIN_EDGES[i], PCT_BIN_EDGES[i + 1]
        if pct >= lo and (pct < hi or (i == len(PCT_BIN_EDGES) - 2 and pct <= hi)):
            return f"{lo}-{hi}%"
    return f"{PCT_BIN_EDGES[-2]}-{PCT_BIN_EDGES[-1]}%"


def pct_bin_midpoint(bin_label):
    m = re.match(r"(\d+)-(\d+)%", bin_label)
    if not m:
        return 0.0
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo + hi) / 2.0


def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["pct_complete"] = float(r.get("pct_complete", 0) or 0)
        r["confidence"] = float(r.get("confidence", 0) or 0)
        r["at_risk_prob"] = float(r.get("at_risk_prob", 0) or 0)
        r["correct"] = r.get("correct", "").strip().lower() in ("true", "1", "yes")
        r["truth_at_risk"] = r.get("truth_at_risk", "").strip().lower() in ("true", "1", "yes")
        if not r.get("pct_bin"):
            r["pct_bin"] = pct_to_bin(r["pct_complete"])
        r["dataset"] = canon_dataset(r.get("dataset", ""))
    return rows


# =============================================================================
# METRICS
# =============================================================================

def compute_accuracy(rows):
    if not rows:
        return None
    return sum(1 for r in rows if r["correct"]) / len(rows)


def compute_ece(rows, n_bins=10):
    if not rows:
        return None
    bins = defaultdict(list)
    for r in rows:
        p = r["at_risk_prob"]
        outcome = 1 if r["truth_at_risk"] else 0
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, outcome))
    n = len(rows)
    ece = 0.0
    for pairs in bins.values():
        sz = len(pairs)
        avg_p = sum(p for p, _ in pairs) / sz
        avg_o = sum(o for _, o in pairs) / sz
        ece += (sz / n) * abs(avg_o - avg_p)
    return ece


def compute_brier(rows):
    """Brier score for binary at-risk classification: mean((prob - truth)^2)."""
    if not rows:
        return None
    total = 0.0
    for r in rows:
        outcome = 1 if r["truth_at_risk"] else 0
        total += (r["at_risk_prob"] - outcome) ** 2
    return total / len(rows)


def reliability_curve(rows, n_bins=10):
    """Return (bin_centers, empirical_rates, bin_counts) for a reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, rates, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            subset = [r for r in rows if lo <= r["at_risk_prob"] <= hi]
        else:
            subset = [r for r in rows if lo <= r["at_risk_prob"] < hi]
        if not subset:
            continue
        centers.append(sum(r["at_risk_prob"] for r in subset) / len(subset))
        rates.append(sum(1 for r in subset if r["truth_at_risk"]) / len(subset))
        counts.append(len(subset))
    return centers, rates, counts


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_by_pct(rows):
    """
    Group rows by (dataset, pct_bin).
    Return dict: {dataset: [(pct_midpoint, accuracy, ece, brier), ...] sorted by pct}
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["pct_bin"])
        groups[key].append(r)

    lines = defaultdict(list)
    for (dataset, pct_bin), rs in groups.items():
        mid = pct_bin_midpoint(pct_bin)
        acc = compute_accuracy(rs)
        ece = compute_ece(rs)
        brier = compute_brier(rs)
        lines[dataset].append((mid, acc, ece, brier))

    for key in lines:
        lines[key].sort(key=lambda x: x[0])
    return lines


def group_for_reliability(rows):
    """Group rows by (dataset, pct_bin) for the faceted reliability diagram."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["pct_bin"])].append(r)
    return groups


# =============================================================================
# PLOTTING — RELIABILITY DIAGRAM (Guo-style clean)
# =============================================================================

def _reliability_title(empirical_rates, confidences):
    """Label the panel based on overall direction of miscalibration."""
    if not empirical_rates:
        return ""
    gaps = [r - c for r, c in zip(empirical_rates, confidences)]
    mean_gap = sum(gaps) / len(gaps)
    if abs(mean_gap) < 0.02:
        return "Well calibrated"
    return "Underconfident" if mean_gap > 0 else "Overconfident"


def plot_reliability_clean(rows, output_path, n_bins=10):
    """
    Clean Guo-style reliability diagram with per-panel histogram and ECE.
    One column per dataset; each column has:
      - top:    histogram of predicted at-risk probability (sample support)
      - bottom: reliability bars (blue = empirical rate, red hatched = gap)
    All rows pooled across the semester.
    """
    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)
    datasets = sorted(by_dataset.keys())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    bar_w = (1.0 / n_bins) * 0.92

    # Figure with 2 rows (hist on top, reliability below) × N datasets.
    fig = plt.figure(figsize=(5.2 * len(datasets), 6.0))
    gs = fig.add_gridspec(2, len(datasets),
                          height_ratios=[1, 4], hspace=0.08, wspace=0.12)

    blue = "#2E6FB7"
    red_edge = "#C0392B"
    red_face = "#F5C6C4"
    hist_color = "#7F7F7F"

    rel_axes = []
    for col, dataset in enumerate(datasets):
        subset = by_dataset[dataset]
        ax_hist = fig.add_subplot(gs[0, col])
        ax_rel = fig.add_subplot(gs[1, col], sharex=ax_hist)
        rel_axes.append(ax_rel)

        # ---- per-bin stats ----
        bin_counts = [0] * n_bins
        empirical = [None] * n_bins
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                bucket = [r for r in subset if lo <= r["at_risk_prob"] <= hi]
            else:
                bucket = [r for r in subset if lo <= r["at_risk_prob"] < hi]
            bin_counts[i] = len(bucket)
            if bucket:
                empirical[i] = sum(1 for r in bucket if r["truth_at_risk"]) / len(bucket)

        # ---- ECE + Brier over the full dataset ----
        ece = compute_ece(subset, n_bins=n_bins) or 0.0
        brier = compute_brier(subset) or 0.0

        # ---- histogram (top) ----
        total = sum(bin_counts)
        frac = [c / total if total else 0 for c in bin_counts]
        ax_hist.bar(mids, frac, width=bar_w,
                    color=hist_color, edgecolor="black", linewidth=0.6, zorder=2)
        ax_hist.set_xlim(-0.02, 1.02)
        ax_hist.set_ylim(0, max(frac) * 1.15 if max(frac) > 0 else 1.0)
        ax_hist.set_yticks([])
        ax_hist.set_ylabel("Samples", fontsize=10)
        ax_hist.tick_params(axis="x", labelbottom=False)
        for spine in ("top", "right", "left"):
            ax_hist.spines[spine].set_visible(False)

        title_dir = _reliability_title(
            [e for e in empirical if e is not None],
            [m for m, e in zip(mids, empirical) if e is not None],
        )
        header = f"{dataset} — {title_dir}" if title_dir else dataset
        header += f"\nECE = {ece:.3f}   Brier = {brier:.3f}"
        ax_hist.set_title(header, fontsize=12, pad=8)

        # ---- reliability bars (bottom) ----
        bar_heights = [e if e is not None else 0 for e in empirical]
        ax_rel.bar(mids, bar_heights, width=bar_w,
                   color=blue, edgecolor="black", linewidth=0.8,
                   zorder=2, label="Outputs")

        gap_bottoms, gap_heights = [], []
        for m, e in zip(mids, empirical):
            if e is None:
                gap_bottoms.append(0); gap_heights.append(0); continue
            gap_bottoms.append(min(e, m))
            gap_heights.append(abs(m - e))
        ax_rel.bar(mids, gap_heights, width=bar_w, bottom=gap_bottoms,
                   facecolor=red_face, edgecolor=red_edge, linewidth=0.8,
                   hatch="//", zorder=3, label="Calibration gap")

        ax_rel.plot([0, 1], [0, 1], color="black", linestyle="--",
                    linewidth=1.2, zorder=4)
        ax_rel.set_xlabel("Confidence", fontsize=12)
        ax_rel.set_xlim(-0.02, 1.02)
        ax_rel.set_ylim(0.0, 1.0)
        ax_rel.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_rel.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_rel.tick_params(labelsize=11)

        if col == 0:
            ax_rel.set_ylabel("Accuracy", fontsize=12)
            ax_rel.legend(loc="upper left", fontsize=11, frameon=True)
        else:
            ax_rel.tick_params(axis="y", labelleft=False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram -> {output_path}")


# =============================================================================
# PLOTTING — ACCURACY & ECE OVER TIME (companion)
# =============================================================================

def plot_over_time(lines, output_path):
    """Three-panel line plot: accuracy, ECE, and Brier vs. percent of semester."""
    fig, (ax_acc, ax_ece, ax_brier) = plt.subplots(1, 3, figsize=(16, 4.5))

    for dataset, points in lines.items():
        xs = [p[0] for p in points if p[1] is not None]
        ys = [p[1] for p in points if p[1] is not None]
        ax_acc.plot(xs, ys, marker="o", linewidth=2,
                    color=COLORS.get(dataset, "black"), label=dataset)

    ax_acc.set_xlabel("Percent of semester completed")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax_acc.legend(loc="lower right", fontsize=10)
    ax_acc.set_title("Final-grade prediction accuracy")

    for dataset, points in lines.items():
        xs = [p[0] for p in points if p[2] is not None]
        ys = [p[2] for p in points if p[2] is not None]
        ax_ece.plot(xs, ys, marker="o", linewidth=2,
                    color=COLORS.get(dataset, "black"), label=dataset)

    ax_ece.set_xlabel("Percent of semester completed")
    ax_ece.set_ylabel("ECE (lower is better)")
    ax_ece.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax_ece.legend(loc="upper right", fontsize=10)
    ax_ece.set_title("At-risk calibration error")

    for dataset, points in lines.items():
        xs = [p[0] for p in points if len(p) > 3 and p[3] is not None]
        ys = [p[3] for p in points if len(p) > 3 and p[3] is not None]
        ax_brier.plot(xs, ys, marker="o", linewidth=2,
                      color=COLORS.get(dataset, "black"), label=dataset)

    ax_brier.set_xlabel("Percent of semester completed")
    ax_brier.set_ylabel("Brier score (lower is better)")
    ax_brier.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax_brier.legend(loc="upper right", fontsize=10)
    ax_brier.set_title("At-risk Brier score")

    fig.suptitle("Prediction quality across the semester", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Over-time plot -> {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Path to exp1_raw.csv (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output dir for PNGs (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    rows = load_raw(args.input)
    print(f"  {len(rows)} predictions loaded")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_reliability_clean(rows, os.path.join(args.output_dir, "exp1_reliability_clean.png"))

    lines = aggregate_by_pct(rows)
    print(f"  {len(lines)} dataset line(s) for the companion plot")
    plot_over_time(lines, os.path.join(args.output_dir, "exp1_over_time.png"))

    print("Done.")


if __name__ == "__main__":
    main()