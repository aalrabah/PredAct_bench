"""
Figure 1 candidates for PredAct.
Generates 3 visualization styles so you can pick which one to use.
Datasets: OULAD, PredAct-CS.
Metric on y-axis: f1_final_mean (change METRIC below to swap).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Config ---
CSV_PATH = "results/exp2/exp2_per_cell.csv"
METRIC = "f1_final_mean"          # y-axis metric
METRIC_STD = "f1_final_std"       # matching std column
METRIC_LABEL = "F1"       # axis label # f1 final
OUTPUT_PREFIX = "figure1"

# Display names
DISPLAY_NAMES = {
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
DATASET_NAMES = {"oulad": "OULAD", "predact_cs": "PredAct-CS"}

# Target accuracy → color (sequential)
TARGET_COLORS = {0.4: "#A8C8E8", 0.5: "#5B95C9", 0.6: "#1F5A9E",
                 0.7: "#FFA500", 0.8: "#D32F2F"}

# --- Load ---
df = pd.read_csv(CSV_PATH)
df["model_display"] = df["instructor_llm"].map(lambda m: DISPLAY_NAMES.get(m, m))
df["dataset_display"] = df["dataset"].map(lambda d: DATASET_NAMES.get(d, d.upper()))

datasets = sorted(df["dataset"].unique())
targets = sorted(df["target_accuracy"].unique())

# Sort models once by overall mean across all targets+datasets (ascending)
model_order = (df.groupby("instructor_llm")[METRIC].mean()
                 .sort_values().index.tolist())
model_labels = [DISPLAY_NAMES.get(m, m) for m in model_order]


# =============================================================
# OPTION A — Grouped bars: 2 panels (one per dataset),
# 3 bars per model (one per target accuracy).
# =============================================================
def plot_grouped_bars():
    fig, axes = plt.subplots(1, len(datasets), figsize=(9 * len(datasets), 5),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    bar_w = 0.25
    x = np.arange(len(model_order))

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for i, t in enumerate(targets):
            row = sub[sub["target_accuracy"] == t].set_index("instructor_llm")
            means = [row.loc[m, METRIC] if m in row.index else 0 for m in model_order]
            stds  = [row.loc[m, METRIC_STD] if m in row.index else 0 for m in model_order]
            ax.bar(x + (i - 1) * bar_w, means, bar_w,
                   color=TARGET_COLORS[t], edgecolor="black", linewidth=0.6,
                   label=f"target = {t}")
            ax.errorbar(x + (i - 1) * bar_w, means, yerr=stds,
                        fmt="none", ecolor="black", elinewidth=0.7, capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=10)
        ax.set_title(DATASET_NAMES.get(ds, ds.upper()), fontsize=12)
        ax.set_ylabel(METRIC_LABEL)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_A_grouped.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================
# OPTION B — Faceted small multiples: dataset × target_acc grid.
# Rows = datasets, Cols = target accuracies. One bar per model per panel.
# =============================================================
def plot_faceted():
    fig, axes = plt.subplots(len(datasets), len(targets),
                             figsize=(5 * len(targets), 3.5 * len(datasets)),
                             sharey=True)
    if len(datasets) == 1:
        axes = np.array([axes])
    if len(targets) == 1:
        axes = axes.reshape(-1, 1)

    x = np.arange(len(model_order))
    for r, ds in enumerate(datasets):
        for c, t in enumerate(targets):
            ax = axes[r, c]
            row = df[(df["dataset"] == ds) & (df["target_accuracy"] == t)] \
                    .set_index("instructor_llm")
            means = [row.loc[m, METRIC] if m in row.index else 0 for m in model_order]
            stds  = [row.loc[m, METRIC_STD] if m in row.index else 0 for m in model_order]
            ax.bar(x, means, color=TARGET_COLORS[t],
                   edgecolor="black", linewidth=0.6)
            ax.errorbar(x, means, yerr=stds,
                        fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(f"target = {t}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{DATASET_NAMES.get(ds, ds.upper())}\n{METRIC_LABEL}",
                              fontsize=10)

    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_B_faceted.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================
# OPTION C — Line plot: 2 panels (one per dataset).
# X = models, Y = metric, one line per target_acc.
# Best when you want to see how each model degrades across targets.
# =============================================================
def plot_lines():
    fig, axes = plt.subplots(1, len(datasets), figsize=(9 * len(datasets), 5),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    x = np.arange(len(model_order))
    markers = {0.4: "o", 0.5: "s", 0.6: "^", 0.7: "D", 0.8: "v"}

    for ax, ds in zip(axes, datasets):
        for t in targets:
            row = df[(df["dataset"] == ds) & (df["target_accuracy"] == t)] \
                    .set_index("instructor_llm")
            means = [row.loc[m, METRIC] if m in row.index else np.nan
                     for m in model_order]
            stds  = [row.loc[m, METRIC_STD] if m in row.index else 0
                     for m in model_order]
            ax.errorbar(x, means, yerr=stds,
                        marker=markers[t], color=TARGET_COLORS[t],
                        linewidth=1.8, markersize=6, capsize=3,
                        label=f"target = {t}")

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=10)
        ax.set_title(DATASET_NAMES.get(ds, ds.upper()), fontsize=12)
        ax.set_ylabel(METRIC_LABEL)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_C_lines.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================
# OPTION D — SINGLE PANEL: both datasets together.
# X = models. Lines = one per (dataset, target_acc). Solid = PredAct-CS,
# dashed = OULAD. Color encodes target accuracy.
# =============================================================
def plot_combined_single_panel():
    """Single-panel grouped bar chart. For each model, two bars side-by-side
    (one per dataset), averaged across all target accuracies. Color = dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_order))
    bar_w = 0.38
    DS_COLORS = {"predact_cs": "#2E86AB", "oulad": "#E63946"}

    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        means, stds = [], []
        for m in model_order:
            vals = sub[sub["instructor_llm"] == m][METRIC]
            means.append(vals.mean() if not vals.empty else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)
        offset = (i - (len(datasets) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means, bar_w,
                      color=DS_COLORS.get(ds, "#666"),
                      edgecolor="black", linewidth=0.6,
                      label=DATASET_NAMES.get(ds, ds.upper()))
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title("F1 by instructor Agent — both datasets across 5 target accuracies",
                 fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(0.95, ax.get_ylim()[1]))
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_D_combined.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================
# OPTION E — FIGURE 1 candidate.
# 2 panels (one per dataset). X-axis = target accuracy (5 values).
# For each accuracy, 12 grouped bars (one per instructor model).
# Top 3 instructors get bold colors; the rest are muted grays.
# Reader sees: F1 climbs with target accuracy, top 3 lead consistently,
# and any over-trust dip on OULAD 80% becomes visually obvious.
# =============================================================
def plot_figure1():
    # Rank models by overall mean (descending)
    model_rank = (df.groupby("instructor_llm")[METRIC].mean()
                    .sort_values(ascending=False).index.tolist())
    model_labels_ranked = [DISPLAY_NAMES.get(m, m) for m in model_rank]

    # Top 3 vibrant; remaining 9 muted grayscale (lighter for lower-ranked)
    TOP_COLORS = ["#1F5A9E", "#E63946", "#F4A261"]  # blue, red, orange
    other_grays = [str(0.55 + 0.04 * i) for i in range(len(model_rank) - 3)]
    color_map = dict(zip(model_rank[:3], TOP_COLORS))
    color_map.update(dict(zip(model_rank[3:], other_grays)))

    n_models = len(model_rank)
    n_targets = len(targets)
    bar_w = 0.95 / n_models
    x = np.arange(n_targets)

    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5),
                             sharey=True)
    if len(datasets) == 1: axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for j, m in enumerate(model_rank):
            offset = (j - (n_models - 1) / 2) * bar_w
            row = sub[sub["instructor_llm"] == m].set_index("target_accuracy")
            means = [row.loc[t, METRIC] if t in row.index else 0 for t in targets]
            stds  = [row.loc[t, METRIC_STD] if t in row.index else 0 for t in targets]
            label = DISPLAY_NAMES.get(m, m) if j < 3 else None  # only top 3 in legend
            zorder = 3 if j < 3 else 2  # top 3 on top
            ax.bar(x + offset, means, bar_w,
                   color=color_map[m], edgecolor="black", linewidth=0.4,
                   label=label, zorder=zorder)
            ax.errorbar(x + offset, means, yerr=stds,
                        fmt="none", ecolor="black", elinewidth=0.5, capsize=1.2,
                        zorder=zorder + 1)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(t*100)}%" for t in targets])
        ax.set_xlabel("Target Tool Accuracy")
        ax.set_ylabel(METRIC_LABEL)
        ax.set_title(DATASET_NAMES.get(ds, ds.upper()), fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.05)

    # Legend: only top 3 named; add a "others (9 instructors)" gray entry
    handles, lbls = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="0.7", edgecolor="black", label="Other 9 instructors"))
    lbls.append("Other 9 instructors")
    axes[0].legend(handles, lbls, frameon=False, loc="upper left", fontsize=9)

    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_E_main.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}  (top 3: {', '.join(model_labels_ranked[:3])})")


# =============================================================
# OPTION F — Combined-datasets, family-colored.
# Single panel. X-axis = target accuracy (5 values).
# 12 grouped bars per accuracy, colored by MODEL FAMILY.
# Each family has 2 variants distinguished by darker/lighter shade.
# F1 averaged across both datasets (PredAct-CS + OULAD).
# =============================================================
def plot_figure1_combined_families():
    # Model family groupings + colors
    FAMILIES = {
        "GPT":      {"color": "#F4A6B5", "models": ["gpt5_5", "gpt5_4_mini", "gpt4o_mini"]},  # soft pink
        "Claude":   {"color": "#FFD3A5", "models": ["claude_opus_4_7", "claude_haiku_4_5"]},  # peach
        "Gemini":   {"color": "#A8D8EA", "models": ["gemini_3_1_pro", "gemini_3_flash"]},     # baby blue
        "DeepSeek": {"color": "#B4E7CE", "models": ["deepseek_v4_pro", "deepseek_v4_flash"]}, # mint
        "Mistral":  {"color": "#C9B6E0", "models": ["mistral_small_24b", "ministral_3_14b"]}, # lavender
        "Qwen":     {"color": "#E8D5C4", "models": ["qwen_35b", "qwen_9b"]},                  # warm tan
    }
    # Flatten in the order families are declared (within family: bigger model first)
    flat_models = []
    for fam, info in FAMILIES.items():
        for i, m in enumerate(info["models"]):
            flat_models.append((fam, m, info["color"], i))   # i = 0 (darker) or 1 (lighter)

    n_models = len(flat_models)   # 12
    n_targets = len(targets)      # 5
    bar_w = 0.95 / n_models
    x = np.arange(n_targets)

    fig, ax = plt.subplots(figsize=(13, 6))

    for j, (fam, m, base_color, variant_idx) in enumerate(flat_models):
        offset = (j - (n_models - 1) / 2) * bar_w
        # Average across both datasets per target accuracy
        means, stds = [], []
        for t in targets:
            sub = df[(df["instructor_llm"] == m) & (df["target_accuracy"] == t)]
            vals = sub[METRIC]
            means.append(vals.mean() if not vals.empty else 0)
            # std across the two datasets
            stds.append(vals.std() if len(vals) > 1 else 0)
        # variant_idx 0 = solid base color (bigger model), 1 = lighter shade (smaller)
        if variant_idx == 0:
            color = base_color
            hatch = ""
            edge = "black"
        else:
            color = base_color
            hatch = "//"
            edge = "black"
        ax.bar(x + offset, means, bar_w,
               color=color, edgecolor=edge, linewidth=0.5, hatch=hatch,
               label=DISPLAY_NAMES.get(m, m))
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", ecolor="#999999", elinewidth=0.5, capsize=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets], fontsize=12)
    ax.set_xlabel("Target Tool Accuracy", fontsize=14)
    ax.set_ylabel(f"{METRIC_LABEL} (mean across both datasets)", fontsize=11)
    ax.set_title("Instructor F1 vs target accuracy — both datasets combined, grouped by model family",
                 fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)

    # Legend: 2 columns, ordered by family
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, frameon=False, loc="upper left",
              fontsize=8, ncol=3, columnspacing=0.8)

    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_F_combined_families.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================
# OPTION G — Both datasets in ONE panel, dataset distinguishable.
# Same X-axis (target accuracy). For each model: 2 bars side-by-side,
# left = PredAct-CS (solid), right = OULAD (hatched). Family color.
# =============================================================
def plot_figure1_two_datasets_one_panel():
    # Each family has two shades: darker = bigger/Pro model, lighter = smaller/Flash
    FAMILIES = {
        "GPT":      {"colors": ["#D85F7C", "#E8849A", "#F8C8D2"],  # pink: darkest / mid / lightest
                     "models": ["gpt5_5", "gpt5_4_mini", "gpt4o_mini"]},
        "Claude":   {"colors": ["#F4B07A", "#FCE0C2"],  # peach
                     "models": ["claude_opus_4_7", "claude_haiku_4_5"]},
        "Gemini":   {"colors": ["#7AB8D6", "#CDE7F0"],  # baby blue
                     "models": ["gemini_3_1_pro", "gemini_3_flash"]},
        "DeepSeek": {"colors": ["#7CCCAB", "#D4F0E0"],  # mint
                     "models": ["deepseek_v4_pro", "deepseek_v4_flash"]},
        "Mistral":  {"colors": ["#A38FCE", "#E0D5EE"],  # lavender
                     "models": ["mistral_small_24b", "ministral_3_14b"]},
        "Qwen":     {"colors": ["#C9A98B", "#F0E0D0"],  # warm tan
                     "models": ["qwen_35b", "qwen_9b"]},
    }
    flat_models = []
    for fam, info in FAMILIES.items():
        for color, m in zip(info["colors"], info["models"]):
            flat_models.append((fam, m, color))

    n_models = len(flat_models)   # 12
    n_targets = len(targets)
    # 24 bars per accuracy group (12 models × 2 datasets).
    # group_spread sets total slot per accuracy; fill_ratio decides
    # how much is bars vs. white space between groups.
    group_spread = 5.0
    fill_ratio = 0.80               # 80% bars, 20% whitespace between groups
    bar_w = (group_spread * fill_ratio) / (n_models * 2)
    x = np.arange(n_targets) * group_spread

    fig, ax = plt.subplots(figsize=(24, 7))

    for j, (fam, m, base_color) in enumerate(flat_models):
        # PredAct-CS bar (solid)
        offset_u = (2 * j - (n_models * 2 - 1) / 2) * bar_w
        u_means = [df[(df["instructor_llm"]==m) & (df["dataset"]=="predact_cs") &
                       (df["target_accuracy"]==t)][METRIC].iloc[0]
                   if not df[(df["instructor_llm"]==m) & (df["dataset"]=="predact_cs") &
                             (df["target_accuracy"]==t)].empty else 0
                   for t in targets]
        # OULAD bar (hatched)
        offset_o = offset_u + bar_w
        o_means = [df[(df["instructor_llm"]==m) & (df["dataset"]=="oulad") &
                       (df["target_accuracy"]==t)][METRIC].iloc[0]
                   if not df[(df["instructor_llm"]==m) & (df["dataset"]=="oulad") &
                             (df["target_accuracy"]==t)].empty else 0
                   for t in targets]
        ax.bar(x + offset_u, u_means, bar_w,
               color=base_color, edgecolor="black", linewidth=0.4,
               label=DISPLAY_NAMES.get(m, m))
        ax.bar(x + offset_o, o_means, bar_w,
               color=base_color, edgecolor="#555555", linewidth=0.4, hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets], fontsize=11)
    ax.set_xlabel("Target Tool Accuracy", fontsize=14)
    ax.set_ylabel(METRIC_LABEL, fontsize=11)
    ax.set_title("F1 by Instructor Model and Accuracy — PredAct-CS vs OULAD",
                 fontsize=12)
    ax.set_ylim(0, 1.05)

    # Legend: 1) one entry per model (color), 2) two entries for dataset patterns
    model_handles, model_lbls = ax.get_legend_handles_labels()
    dataset_handles = [
        Patch(facecolor="0.85", edgecolor="black", label="PredAct-CS"),
        Patch(facecolor="0.85", edgecolor="black", hatch="///", label="OULAD"),
    ]
    # Pad each label with TRAILING spaces — color swatch is tight to label,
    # white space goes AFTER the model name (before the next column's color box),
    # which is where the logo gets dropped.
    padded_lbls = [lbl + "      " for lbl in model_lbls]
    leg1 = ax.legend(model_handles, padded_lbls, frameon=False,
                     loc="upper left", fontsize=8, ncol=3,
                     columnspacing=3.5,        # big gap between columns (= logo space)
                     handletextpad=0.5,        # tight gap between swatch and label
                     labelspacing=1.2,         # vertical breathing room
                     title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=dataset_handles, frameon=False,
              loc="upper right", fontsize=9, title="Dataset",
              labelspacing=1.0, handletextpad=1.5)

    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_G_two_datasets_one_panel.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================
# OPTION H — One chart per dataset (recommended for paper).
# Same style as G but only 1 dataset per figure → bars are wider,
# more readable. Generates two files: predact-cs.pdf/png and oulad.pdf/png.
# =============================================================
def plot_per_dataset_separate():
    FAMILIES = {
        "GPT":      {"colors": ["#D85F7C", "#E8849A", "#F8C8D2"],
                     "models": ["gpt5_5", "gpt5_4_mini", "gpt4o_mini"]},
        "Claude":   {"colors": ["#F4B07A", "#FCE0C2"],
                     "models": ["claude_opus_4_7", "claude_haiku_4_5"]},
        "Gemini":   {"colors": ["#7AB8D6", "#CDE7F0"],
                     "models": ["gemini_3_1_pro", "gemini_3_flash"]},
        "DeepSeek": {"colors": ["#7CCCAB", "#D4F0E0"],
                     "models": ["deepseek_v4_pro", "deepseek_v4_flash"]},
        "Mistral":  {"colors": ["#A38FCE", "#E0D5EE"],
                     "models": ["mistral_small_24b", "ministral_3_14b"]},
        "Qwen":     {"colors": ["#C9A98B", "#F0E0D0"],
                     "models": ["qwen_35b", "qwen_9b"]},
    }
    flat_models = []
    for fam, info in FAMILIES.items():
        for color, m in zip(info["colors"], info["models"]):
            flat_models.append((fam, m, color))

    n_models = len(flat_models)   # 12
    n_targets = len(targets)
    group_spread = 4.0
    fill_ratio = 0.85
    bar_w = (group_spread * fill_ratio) / n_models
    x = np.arange(n_targets) * group_spread

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(14, 6))

        for j, (fam, m, color) in enumerate(flat_models):
            offset = (j - (n_models - 1) / 2) * bar_w
            means = [df[(df["instructor_llm"] == m) & (df["dataset"] == ds) &
                         (df["target_accuracy"] == t)][METRIC].iloc[0]
                     if not df[(df["instructor_llm"] == m) & (df["dataset"] == ds) &
                               (df["target_accuracy"] == t)].empty else 0
                     for t in targets]
            ax.bar(x + offset, means, bar_w,
                   color=color, edgecolor="black", linewidth=0.4,
                   label=DISPLAY_NAMES.get(m, m))

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(t*100)}%" for t in targets], fontsize=11)
        ax.set_xlabel("Target tool accuracy", fontsize=14)
        ax.set_ylabel(METRIC_LABEL, fontsize=11)
        ax.set_title(f"F1 by instructor — {DATASET_NAMES.get(ds, ds.upper())}",
                     fontsize=12)
        ax.set_ylim(0, 1.05)

        # Legend with whitespace for manual logos (after model name)
        handles, lbls = ax.get_legend_handles_labels()
        padded_lbls = [lbl + "      " for lbl in lbls]
        ax.legend(handles, padded_lbls, frameon=False,
                  loc="upper left", fontsize=8, ncol=3,
                  columnspacing=3.5, handletextpad=0.5,
                  labelspacing=1.2, title="Model")

        plt.tight_layout()
        ds_slug = "predact-cs" if ds == "predact_cs" else ds
        out = f"{OUTPUT_PREFIX}_H_{ds_slug}.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved → {out}")


# =============================================================
# OPTION I — One bar per (model, accuracy), with both datasets overlaid.
# Bar's solid fill = PredAct-CS (primary).
# Hatched overlay (drawn on top of the same bar) = OULAD.
# Bar's TOTAL height = max of the two; below max, you see one or both.
# =============================================================
def plot_overlaid_datasets():
    FAMILIES = {
        "GPT":      {"colors": ["#D85F7C", "#E8849A", "#F8C8D2"],
                     "models": ["gpt5_5", "gpt5_4_mini", "gpt4o_mini"]},
        "Claude":   {"colors": ["#F4B07A", "#FCE0C2"],
                     "models": ["claude_opus_4_7", "claude_haiku_4_5"]},
        "Gemini":   {"colors": ["#7AB8D6", "#CDE7F0"],
                     "models": ["gemini_3_1_pro", "gemini_3_flash"]},
        "DeepSeek": {"colors": ["#7CCCAB", "#D4F0E0"],
                     "models": ["deepseek_v4_pro", "deepseek_v4_flash"]},
        "Mistral":  {"colors": ["#A38FCE", "#E0D5EE"],
                     "models": ["mistral_small_24b", "ministral_3_14b"]},
        "Qwen":     {"colors": ["#C9A98B", "#F0E0D0"],
                     "models": ["qwen_35b", "qwen_9b"]},
    }
    flat_models = []
    for fam, info in FAMILIES.items():
        for color, m in zip(info["colors"], info["models"]):
            flat_models.append((fam, m, color))

    n_models = len(flat_models)
    n_targets = len(targets)
    group_spread = 4.0
    fill_ratio = 0.85
    bar_w = (group_spread * fill_ratio) / n_models
    x = np.arange(n_targets) * group_spread

    fig, ax = plt.subplots(figsize=(18, 6.5))

    for j, (fam, m, color) in enumerate(flat_models):
        offset = (j - (n_models - 1) / 2) * bar_w
        u_means, o_means = [], []
        for t in targets:
            u_row = df[(df["instructor_llm"] == m) & (df["dataset"] == "predact_cs") &
                        (df["target_accuracy"] == t)][METRIC]
            o_row = df[(df["instructor_llm"] == m) & (df["dataset"] == "oulad") &
                        (df["target_accuracy"] == t)][METRIC]
            u_means.append(u_row.iloc[0] if not u_row.empty else 0)
            o_means.append(o_row.iloc[0] if not o_row.empty else 0)

        # Solid bar = PredAct-CS (primary)
        ax.bar(x + offset, u_means, bar_w,
               color=color, edgecolor="black", linewidth=0.4,
               label=DISPLAY_NAMES.get(m, m), zorder=2)
        # Hatched overlay = OULAD. Family-color fill at moderate opacity +
        # medium-gray hatch lines — visible but not visually dominant.
        ax.bar(x + offset, o_means, bar_w,
               color=color, alpha=0.50, edgecolor="#555555", linewidth=0.5,
               hatch="//", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets], fontsize=12)
    ax.set_xlabel("Target Tool Accuracy", fontsize=14)
    ax.set_ylabel(METRIC_LABEL, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("F1 by Instructor Model for PredAct-CS and OULAD",
                 fontsize=14)
    ax.set_ylim(0, 1.05)

    # Two legends: model + dataset key
    handles, lbls = ax.get_legend_handles_labels()
    padded_lbls = [lbl + "      " for lbl in lbls]
    leg1 = ax.legend(handles, padded_lbls, frameon=False,
                     loc="upper left", fontsize=12, ncol=3,
                     columnspacing=3.5, handletextpad=1.6,
                     labelspacing=1.2)
    ax.add_artist(leg1)
    dataset_handles = [
        Patch(facecolor="0.85", edgecolor="black", label="PredAct-CS (solid)"),
        Patch(facecolor="0.85", alpha=0.50, edgecolor="#555555", hatch="//",
              label="OULAD (hatched)"),
    ]
    ax.legend(handles=dataset_handles, frameon=False,
              loc="upper right", fontsize=12, title="Dataset", title_fontsize=12)

    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_I_overlaid.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================
# OPTION J — Dumbbell plot.
# Per (model, accuracy): two dots (one per dataset) connected
# by a thin line. The line IS the cross-dataset gap. Less ink,
# very clean, common in benchmark papers.
# =============================================================
def plot_dumbbell():
    FAMILIES = {
        "GPT":      {"colors": ["#D85F7C", "#E8849A", "#F8C8D2"],
                     "models": ["gpt5_5", "gpt5_4_mini", "gpt4o_mini"]},
        "Claude":   {"colors": ["#F4B07A", "#FCE0C2"],
                     "models": ["claude_opus_4_7", "claude_haiku_4_5"]},
        "Gemini":   {"colors": ["#7AB8D6", "#CDE7F0"],
                     "models": ["gemini_3_1_pro", "gemini_3_flash"]},
        "DeepSeek": {"colors": ["#7CCCAB", "#D4F0E0"],
                     "models": ["deepseek_v4_pro", "deepseek_v4_flash"]},
        "Mistral":  {"colors": ["#A38FCE", "#E0D5EE"],
                     "models": ["mistral_small_24b", "ministral_3_14b"]},
        "Qwen":     {"colors": ["#C9A98B", "#F0E0D0"],
                     "models": ["qwen_35b", "qwen_9b"]},
    }
    flat_models = []
    for fam, info in FAMILIES.items():
        for color, m in zip(info["colors"], info["models"]):
            flat_models.append((fam, m, color))

    n_models = len(flat_models)
    n_targets = len(targets)
    group_spread = 4.0
    fill_ratio = 0.85
    dot_dx = (group_spread * fill_ratio) / n_models   # horizontal gap per model
    x = np.arange(n_targets) * group_spread

    fig, ax = plt.subplots(figsize=(18, 6.5))

    for j, (fam, m, color) in enumerate(flat_models):
        offset = (j - (n_models - 1) / 2) * dot_dx
        for ti, t in enumerate(targets):
            u_row = df[(df["instructor_llm"] == m) & (df["dataset"] == "predact_cs") &
                        (df["target_accuracy"] == t)][METRIC]
            o_row = df[(df["instructor_llm"] == m) & (df["dataset"] == "oulad") &
                        (df["target_accuracy"] == t)][METRIC]
            u = u_row.iloc[0] if not u_row.empty else 0
            o = o_row.iloc[0] if not o_row.empty else 0
            xpos = x[ti] + offset
            # Connector line (the "gap")
            ax.plot([xpos, xpos], [u, o], color=color, linewidth=1.4,
                    alpha=0.7, zorder=2)
            # Dots
            label = DISPLAY_NAMES.get(m, m) if ti == 0 else None
            ax.scatter([xpos], [u], color=color, edgecolor="black",
                       linewidth=0.4, s=42, marker="o", zorder=3, label=label)
            ax.scatter([xpos], [o], color=color, edgecolor="black",
                       linewidth=0.4, s=42, marker="s", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets], fontsize=11)
    ax.set_xlabel("Target tool accuracy", fontsize=11)
    ax.set_ylabel(METRIC_LABEL, fontsize=11)
    ax.set_title("F1 by instructor — circle = PredAct-CS, square = OULAD; line = cross-dataset gap",
                 fontsize=12)
    ax.set_ylim(0, 1.05)

    # Two legends: model + dataset key
    handles, lbls = ax.get_legend_handles_labels()
    padded_lbls = [lbl + "      " for lbl in lbls]
    leg1 = ax.legend(handles, padded_lbls, frameon=False,
                     loc="upper left", fontsize=8, ncol=3,
                     columnspacing=3.5, handletextpad=0.5,
                     labelspacing=1.2, title="Model")
    ax.add_artist(leg1)
    from matplotlib.lines import Line2D
    dataset_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
               markeredgecolor="black", markersize=8, label="PredAct-CS"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.5",
               markeredgecolor="black", markersize=8, label="OULAD"),
    ]
    ax.legend(handles=dataset_handles, frameon=False,
              loc="upper right", fontsize=9, title="Dataset")

    plt.tight_layout()
    out = f"{OUTPUT_PREFIX}_J_dumbbell.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    plot_grouped_bars()
    plot_faceted()
    plot_lines()
    plot_combined_single_panel()
    plot_figure1()
    plot_figure1_combined_families()
    plot_figure1_two_datasets_one_panel()
    plot_per_dataset_separate()
    plot_overlaid_datasets()
    plot_dumbbell()