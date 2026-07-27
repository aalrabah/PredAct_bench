"""
Figure 2 candidates: RAIR / RSR visualizations for Exp 2.

Generates:
  figure2_A_scatter.{pdf,png}        RAIR vs RSR scatter, both datasets in one panel
  figure2_A_scatter_panels.{pdf,png} same but split into 2 panels (per-dataset)
  figure2_B_rsr_lines.{pdf,png}      RSR vs target accuracy, one line per model
  figure2_B_rair_lines.{pdf,png}     RAIR vs target accuracy, one line per model
  figure2_combined.{pdf,png}         1x2: scatter + RSR collapse lines

RAIR = chat-fixes-wrong rate (initial wrong → final right)
RSR  = chat-keeps-right rate (initial right → final right)
Reference: Schemmer et al. (2023) on AI-human complementarity metrics.

Family colors mirror visualize_exp2_2.py so Figure 1 and Figure 2 share a palette.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATH = "results/exp2/exp2_per_cell.csv"
OUT_DIR  = "."
PREFIX   = os.path.join(OUT_DIR, "figure2")

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

# Family palette: darkest = strongest variant, lightest = smallest.
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
MODEL_COLOR = {}
for fam, info in FAMILIES.items():
    for c, m in zip(info["colors"], info["models"]):
        MODEL_COLOR[m] = c

DATASET_DISPLAY = {"oulad": "OULAD", "predact_cs": "PredAct-CS"}
DATASET_MARKER  = {"predact_cs": "o", "oulad": "s"}   # circle / square

# Order models for the legend (same blocking as the paper table)
CLOSED = ["gpt5_5", "gpt5_4_mini", "gpt4o_mini",
          "claude_opus_4_7", "claude_haiku_4_5",
          "gemini_3_1_pro", "gemini_3_flash"]
OPEN   = ["deepseek_v4_pro", "deepseek_v4_flash",
          "mistral_small_24b", "ministral_3_14b",
          "qwen_35b", "qwen_9b"]
MODEL_ORDER = CLOSED + OPEN


# -----------------------------------------------------------------------------
# Data load & per-model aggregation
# -----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
targets = sorted(df["target_accuracy"].unique())


def model_means(metric):
    """Returns dict[(model, dataset)] -> mean metric across the 5 cutoffs."""
    out = {}
    for m in MODEL_ORDER:
        for ds in ("predact_cs", "oulad"):
            sub = df[(df["instructor_llm"] == m) & (df["dataset"] == ds)]
            vals = sub[metric].dropna().tolist()
            out[(m, ds)] = sum(vals) / len(vals) if vals else float("nan")
    return out


def per_acc_means(metric):
    """Returns dict[(model, dataset, t)] -> mean metric for that cell."""
    out = {}
    for m in MODEL_ORDER:
        for ds in ("predact_cs", "oulad"):
            for t in targets:
                row = df[(df["instructor_llm"] == m)
                        & (df["dataset"] == ds)
                        & (df["target_accuracy"] == t)]
                v = row[metric].iloc[0] if not row.empty else float("nan")
                out[(m, ds, t)] = v
    return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _model_legend_handles():
    """Family-color swatch + model name, shared across figures."""
    handles = []
    for m in MODEL_ORDER:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=MODEL_COLOR[m],
                              markeredgecolor="black", markersize=10,
                              label=DISPLAY[m]))
    return handles


def _draw_quadrants(ax, x_thresh=0.5, y_thresh=0.5):
    ax.axhline(y_thresh, color="0.7", linestyle=":", linewidth=0.8, zorder=1)
    ax.axvline(x_thresh, color="0.7", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(0.97, 0.97, "Complementary",   transform=ax.transAxes,
            ha="right", va="top",     fontsize=9, color="0.45")
    ax.text(0.03, 0.97, "Too passive",     transform=ax.transAxes,
            ha="left",  va="top",     fontsize=9, color="0.45")
    ax.text(0.97, 0.03, "Over-active",     transform=ax.transAxes,
            ha="right", va="bottom",  fontsize=9, color="0.45")
    ax.text(0.03, 0.03, "Chat hurts",      transform=ax.transAxes,
            ha="left",  va="bottom",  fontsize=9, color="0.45")


# =============================================================================
# Figure 2A — RAIR vs RSR scatter (single panel, both datasets overlaid)
# =============================================================================
def fig_scatter_overlaid():
    rair = model_means("rair_mean")
    rsr  = model_means("rsr_mean")

    fig, ax = plt.subplots(figsize=(8.5, 7))

    for m in MODEL_ORDER:
        for ds in ("predact_cs", "oulad"):
            x = rair[(m, ds)]
            y = rsr[(m, ds)]
            if math.isnan(x) or math.isnan(y):
                continue
            ax.scatter(x, y, s=170, marker=DATASET_MARKER[ds],
                       color=MODEL_COLOR[m], edgecolor="black",
                       linewidth=0.6, alpha=0.9, zorder=3)

    _draw_quadrants(ax)
    ax.set_xlabel("RAIR — chat fixes wrong calls", fontsize=13)
    ax.set_ylabel("RSR — chat keeps right calls",  fontsize=13)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Instructor chat behavior (averaged across 5 tool accuracies)",
                 fontsize=13)

    # Two legends: model (color) + dataset (marker shape)
    leg1 = ax.legend(handles=_model_legend_handles(),
                     loc="lower left", bbox_to_anchor=(1.02, 0.0),
                     frameon=False, fontsize=10, title="Model")
    ax.add_artist(leg1)
    ds_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
               markeredgecolor="black", markersize=10, label="PredAct-CS"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.5",
               markeredgecolor="black", markersize=10, label="OULAD"),
    ]
    ax.legend(handles=ds_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=11, title="Dataset")
    plt.tight_layout()
    out = f"{PREFIX}_A_scatter.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================================
# Figure 2A (panels) — RAIR vs RSR scatter, one panel per dataset
# =============================================================================
def fig_scatter_panels():
    rair = model_means("rair_mean")
    rsr  = model_means("rsr_mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, ds in zip(axes, ("predact_cs", "oulad")):
        for m in MODEL_ORDER:
            x = rair[(m, ds)]; y = rsr[(m, ds)]
            if math.isnan(x) or math.isnan(y):
                continue
            ax.scatter(x, y, s=180, color=MODEL_COLOR[m],
                       edgecolor="black", linewidth=0.6, alpha=0.92, zorder=3)
        _draw_quadrants(ax)
        ax.set_title(DATASET_DISPLAY[ds], fontsize=13)
        ax.set_xlabel("RAIR — chat fixes wrong calls", fontsize=12)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("RSR — chat keeps right calls", fontsize=12)
    axes[1].legend(handles=_model_legend_handles(),
                   loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   frameon=False, fontsize=10, title="Model")
    plt.tight_layout()
    out = f"{PREFIX}_A_scatter_panels.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================================
# Figure 2B — RSR vs target accuracy (one line per model, two panels per dataset)
# =============================================================================
def fig_rsr_lines():
    cells = per_acc_means("rsr_mean")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, ds in zip(axes, ("predact_cs", "oulad")):
        for m in MODEL_ORDER:
            ys = [cells[(m, ds, t)] for t in targets]
            ax.plot(targets, ys, marker="o", linewidth=1.6, markersize=6,
                    color=MODEL_COLOR[m], label=DISPLAY[m])
        ax.set_title(DATASET_DISPLAY[ds], fontsize=13)
        ax.set_xlabel("Target tool accuracy", fontsize=12)
        ax.set_xticks(targets)
        ax.set_xticklabels([f"{int(t*100)}%" for t in targets])
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RSR (chat keeps right calls)", fontsize=12)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   frameon=False, fontsize=9, title="Model")
    fig.suptitle("RSR collapse with rising tool accuracy = over-trust",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{PREFIX}_B_rsr_lines.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================================
# Figure 2B (variant) — RAIR vs target accuracy
# =============================================================================
def fig_rair_lines():
    cells = per_acc_means("rair_mean")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, ds in zip(axes, ("predact_cs", "oulad")):
        for m in MODEL_ORDER:
            ys = [cells[(m, ds, t)] for t in targets]
            ax.plot(targets, ys, marker="o", linewidth=1.6, markersize=6,
                    color=MODEL_COLOR[m], label=DISPLAY[m])
        ax.set_title(DATASET_DISPLAY[ds], fontsize=13)
        ax.set_xlabel("Target tool accuracy", fontsize=12)
        ax.set_xticks(targets)
        ax.set_xticklabels([f"{int(t*100)}%" for t in targets])
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RAIR (chat fixes wrong calls)", fontsize=12)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   frameon=False, fontsize=9, title="Model")
    fig.suptitle("RAIR by tool accuracy — does chat fix more or fewer wrong calls?",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = f"{PREFIX}_B_rair_lines.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================================
# Combined Figure 2 — 1x2 grid: scatter (left) + RSR collapse lines (right)
# =============================================================================
def fig_combined():
    rair = model_means("rair_mean")
    rsr  = model_means("rsr_mean")
    cells = per_acc_means("rsr_mean")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: scatter ---
    ax = axes[0]
    for m in MODEL_ORDER:
        for ds in ("predact_cs", "oulad"):
            x = rair[(m, ds)]; y = rsr[(m, ds)]
            if math.isnan(x) or math.isnan(y):
                continue
            ax.scatter(x, y, s=170, marker=DATASET_MARKER[ds],
                       color=MODEL_COLOR[m], edgecolor="black",
                       linewidth=0.6, alpha=0.9, zorder=3)
    _draw_quadrants(ax)
    ax.set_xlabel("RAIR — chat fixes wrong calls", fontsize=12)
    ax.set_ylabel("RSR — chat keeps right calls",  fontsize=12)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a) Chat behavior: complementarity space", fontsize=13)
    ds_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
               markeredgecolor="black", markersize=9, label="PredAct-CS"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.5",
               markeredgecolor="black", markersize=9, label="OULAD"),
    ]
    ax.legend(handles=ds_handles, loc="lower left", frameon=False,
              fontsize=10, title="Dataset")

    # --- Right: RSR vs target accuracy (averaged across both datasets) ---
    ax = axes[1]
    for m in MODEL_ORDER:
        ys = []
        for t in targets:
            vals = [cells[(m, ds, t)] for ds in ("predact_cs", "oulad")
                    if not math.isnan(cells[(m, ds, t)])]
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
        ax.plot(targets, ys, marker="o", linewidth=1.7, markersize=6,
                color=MODEL_COLOR[m], label=DISPLAY[m])
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets])
    ax.set_xlabel("Target tool accuracy", fontsize=12)
    ax.set_ylabel("RSR (mean across both datasets)", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("(b) RSR vs tool accuracy — over-trust check", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=9, title="Model")

    plt.tight_layout()
    out = f"{PREFIX}_combined.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


# =============================================================================
# Heatmap (the recommended one) — models × cutoffs, color = metric mean.
# Two side-by-side panels per dataset. One figure per metric (RAIR, RSR).
# Diverging RdYlGn centered at 0.5 (neutral). Cells annotated with values.
# Models grouped Closed-source on top, Open-source below, with a thin gap row.
# =============================================================================
def _heatmap_grid(metric_key, title_fragment, out_suffix,
                  cmap="RdYlGn", vmin=0.0, vmax=1.0, vcenter=0.5):
    """Render: rows = models (closed block, gap, open block), cols = 5 cutoffs,
    cells = metric mean. One panel per dataset, side-by-side."""
    cells = per_acc_means(metric_key)

    # Sort each block by the model's overall metric mean (descending), so the
    # best-performing models are on top of each block.
    def block_sorted(keys):
        scored = []
        for m in keys:
            vals = []
            for ds in ("predact_cs", "oulad"):
                for t in targets:
                    v = cells[(m, ds, t)]
                    if not math.isnan(v):
                        vals.append(v)
            scored.append((m, sum(vals) / len(vals) if vals else float("nan")))
        scored.sort(key=lambda kv: -kv[1] if not math.isnan(kv[1]) else 1)
        return [m for m, _ in scored]

    closed_sorted = block_sorted(CLOSED)
    open_sorted   = block_sorted(OPEN)
    # Use a sentinel " " row as a 1-row gap between blocks.
    row_keys  = closed_sorted + [None] + open_sorted
    row_label = ([DISPLAY[m] for m in closed_sorted] + [""]
                 + [DISPLAY[m] for m in open_sorted])

    fig, axes = plt.subplots(1, 2, figsize=(13, 7),
                             gridspec_kw={"width_ratios": [1, 1.04]})
    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    for ax, ds in zip(axes, ("predact_cs", "oulad")):
        # Build the matrix: rows × cutoffs.
        mat = np.full((len(row_keys), len(targets)), np.nan)
        for i, m in enumerate(row_keys):
            if m is None:
                continue
            for j, t in enumerate(targets):
                mat[i, j] = cells[(m, ds, t)]
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")
        # Annotate each cell with its value (×100, one decimal).
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if math.isnan(v):
                    continue
                # Pick text color for contrast: dark on light, light on dark.
                rgba = im.cmap(im.norm(v))
                lum  = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "black" if lum > 0.55 else "white"
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=9, color=txt_color)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([f"{int(t*100)}%" for t in targets], fontsize=11)
        ax.set_yticks(range(len(row_keys)))
        ax.set_yticklabels(row_label, fontsize=10)
        ax.set_title(DATASET_DISPLAY[ds], fontsize=13)
        ax.set_xlabel("Target tool accuracy", fontsize=12)
        # Bracket the closed/open blocks visually.
        ax.axhline(len(closed_sorted) - 0.5, color="black", linewidth=1.2)
        ax.axhline(len(closed_sorted) + 0.5, color="black", linewidth=1.2)
    axes[0].set_ylabel("Model (Closed-source above; Open-source below)",
                       fontsize=11)
    # Hide y-tick labels on the right panel; they're redundant.
    axes[1].set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.85,
                        pad=0.02, fraction=0.04)
    cbar.set_label(f"{title_fragment} (×100)", fontsize=11)

    fig.suptitle(f"{title_fragment} by instructor model × tool accuracy",
                 fontsize=14, y=1.00)
    out = f"{PREFIX}_heatmap_{out_suffix}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


def fig_heatmap_rair():
    _heatmap_grid("rair_mean",
                  title_fragment="RAIR — chat fixes wrong calls",
                  out_suffix="rair")


def fig_heatmap_rsr():
    _heatmap_grid("rsr_mean",
                  title_fragment="RSR — chat keeps right calls",
                  out_suffix="rsr")


def fig_heatmap_rair_rsr_stacked():
    """Two heatmaps stacked vertically (RAIR top, RSR bottom), each with
    PredAct-CS | OULAD side-by-side. One figure for the paper."""
    fig, all_axes = plt.subplots(2, 2, figsize=(13, 13),
                                 gridspec_kw={"width_ratios": [1, 1.04]})
    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    def block_sorted(keys, metric_key):
        cells = per_acc_means(metric_key)
        scored = []
        for m in keys:
            vals = [cells[(m, ds, t)] for ds in ("predact_cs", "oulad") for t in targets
                    if not math.isnan(cells[(m, ds, t)])]
            scored.append((m, sum(vals) / len(vals) if vals else float("nan")))
        scored.sort(key=lambda kv: -kv[1] if not math.isnan(kv[1]) else 1)
        return [m for m, _ in scored]

    for row_idx, (metric_key, label) in enumerate([
        ("rair_mean", "RAIR — chat fixes wrong"),
        ("rsr_mean",  "RSR — chat keeps right"),
    ]):
        cells = per_acc_means(metric_key)
        closed_sorted = block_sorted(CLOSED, metric_key)
        open_sorted   = block_sorted(OPEN,   metric_key)
        row_keys  = closed_sorted + [None] + open_sorted
        row_label = ([DISPLAY[m] for m in closed_sorted] + [""]
                     + [DISPLAY[m] for m in open_sorted])

        for col_idx, ds in enumerate(("predact_cs", "oulad")):
            ax = all_axes[row_idx, col_idx]
            mat = np.full((len(row_keys), len(targets)), np.nan)
            for i, m in enumerate(row_keys):
                if m is None:
                    continue
                for j, t in enumerate(targets):
                    mat[i, j] = cells[(m, ds, t)]
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm,
                           interpolation="nearest")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if math.isnan(v):
                        continue
                    rgba = im.cmap(im.norm(v))
                    lum  = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    txt_color = "black" if lum > 0.55 else "white"
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=9, color=txt_color)
            ax.set_xticks(range(len(targets)))
            ax.set_xticklabels([f"{int(t*100)}%" for t in targets], fontsize=10)
            ax.set_yticks(range(len(row_keys)))
            ax.set_yticklabels(row_label if col_idx == 0 else [""] * len(row_keys),
                               fontsize=9)
            ax.axhline(len(closed_sorted) - 0.5, color="black", linewidth=1.0)
            ax.axhline(len(closed_sorted) + 0.5, color="black", linewidth=1.0)
            if row_idx == 0:
                ax.set_title(DATASET_DISPLAY[ds], fontsize=12)
            if row_idx == 1:
                ax.set_xlabel("Target tool accuracy", fontsize=11)
        all_axes[row_idx, 0].set_ylabel(label, fontsize=12)

    cbar = fig.colorbar(im, ax=all_axes, orientation="vertical", shrink=0.7,
                        pad=0.02, fraction=0.03)
    cbar.set_label("Rate (×100)", fontsize=11)
    fig.suptitle("Chat behavior heatmaps — RAIR (top) and RSR (bottom) "
                 "per (model, tool accuracy, dataset)", fontsize=13, y=0.995)
    out = f"{PREFIX}_heatmap_rair_rsr.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    # Old representations (kept for reference; not the recommended ones)
    # fig_scatter_overlaid()
    # fig_scatter_panels()
    # fig_rsr_lines()
    # fig_rair_lines()
    # fig_combined()

    # Heatmaps — recommended (Option A)
    fig_heatmap_rair()
    fig_heatmap_rsr()
    fig_heatmap_rair_rsr_stacked()
