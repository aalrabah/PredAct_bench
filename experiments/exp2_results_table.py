"""
ONE comprehensive Exp 2 LaTeX table.
Each metric split into mean / std columns. Banded by model with Closed-source
and Open-source blocks. Within each block models sorted by combined Overall F1.

Output: results/exp2/exp2_table_paper.tex
"""
import os
import math
import pandas as pd

CSV_PATH = "/Users/abdulrahmanalrabah/PredAct/results/exp2/exp2_per_cell.csv"
OUT_DIR  = "/Users/abdulrahmanalrabah/PredAct/results/exp2"
OUT_TEX  = os.path.join(OUT_DIR, "exp2_table_paper.tex")

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
CLOSED_SOURCE = {
    "gpt5_5", "gpt5_4_mini", "gpt4o_mini",
    "claude_opus_4_7", "claude_haiku_4_5",
    "gemini_3_1_pro", "gemini_3_flash",
}
OPEN_SOURCE = {
    "deepseek_v4_pro", "deepseek_v4_flash",
    "mistral_small_24b", "ministral_3_14b",
    "qwen_35b", "qwen_9b",
}
DATASETS_ORDER = [("uiuc", "PredAct-CS"), ("oulad", "OULAD")]
ACCURACIES = [0.4, 0.5, 0.6, 0.7, 0.8]

# (CSV mean column, CSV std column, header label)
METRICS = [
    ("f1_initial_mean",      "f1_initial_std",      "F1\\textsubscript{init}"),
    ("f1_final_mean",        "f1_final_std",        "F1\\textsubscript{final}"),
    ("precision_final_mean", "precision_final_std", "Precision"),
    ("recall_final_mean",    "recall_final_std",    "Recall"),
    ("rair_mean",            "rair_std",            "RAIR"),
    ("rsr_mean",             "rsr_std",             "RSR"),
]

N_DATA_COLS = 2 + 2 * len(METRICS)   # Dataset, Acc, then mean/std × 6 metrics


def fmt_val(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"{x * 100:.1f}"


def combined_overall_f1(df, model_key):
    rows = df[df["instructor_llm"] == model_key]
    vals = rows["f1_final_mean"].dropna().tolist()
    return sum(vals) / len(vals) if vals else float("nan")


def emit_block_rows(df, model_keys):
    sortable = [(m, combined_overall_f1(df, m)) for m in model_keys]
    sortable.sort(key=lambda t: -t[1] if not math.isnan(t[1]) else 1)

    out = []
    for idx, (m, _) in enumerate(sortable):
        if idx > 0:
            out.append("\\midrule")
        out.append(f"\\multicolumn{{{N_DATA_COLS}}}{{l}}{{\\textbf{{{DISPLAY[m]}}}}} \\\\")
        out.append("\\addlinespace[2pt]")
        for ds_key, ds_label in DATASETS_ORDER:
            for t in ACCURACIES:
                row = df[(df["instructor_llm"] == m)
                        & (df["dataset"] == ds_key)
                        & (df["target_accuracy"] == t)]
                if row.empty:
                    continue
                r = row.iloc[0]
                cells = [ds_label, f"{int(round(t*100))}\\%"]
                for mean_col, std_col, _ in METRICS:
                    cells.append(fmt_val(r[mean_col]))
                    cells.append(fmt_val(r[std_col]))
                out.append(" & ".join(cells) + " \\\\")
    return "\n".join(out)


def main():
    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    closed_keys = [m for m in DISPLAY if m in CLOSED_SOURCE]
    open_keys   = [m for m in DISPLAY if m in OPEN_SOURCE]
    closed_body = emit_block_rows(df, closed_keys)
    open_body   = emit_block_rows(df, open_keys)

    # Column spec: l (Dataset) | c (Acc) | then for each metric: c c with a |
    # between metrics so it's easier to scan.
    metric_block = "cc"
    col_spec = "l c " + ("|" + metric_block) * len(METRICS)

    # Top header: metric names spanning 2 cols each
    top_cells = ["", ""]
    cmidrules = []
    col_start = 3
    for _, _, label in METRICS:
        top_cells.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_start+1}}}")
        col_start += 2
    top_header  = " & ".join(top_cells) + " \\\\"
    cmid_line   = " ".join(cmidrules)

    sub_header = ["\\textbf{Dataset}", "\\textbf{Acc}"]
    for _ in METRICS:
        sub_header += ["mean", "std"]
    sub_line = " & ".join(sub_header) + " \\\\"

    body = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\caption{Full Exp\\,2 results: 13 instructor LLMs $\\times$ 2 datasets "
        "(PredAct-CS, OULAD) $\\times$ 5 target tool accuracies (40\\%--80\\%). "
        "All values reported as separate mean and std columns ($\\times 100$), "
        "computed over 10 episodes (30 students, 5 forced at-risk) per cell. "
        "RAIR = chat-fixes-wrong rate; RSR = chat-keeps-right rate "
        "(Schemmer et al., 2023). Rows banded by instructor model; closed-source "
        "block first, open-source second; within each block models sorted by "
        "combined Overall F1\\textsubscript{final} across both datasets. "
        "\\enquote{---} indicates the metric was undefined for that cell.}\n"
        "\\label{tab:exp2_full}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{top_header}\n"
        f"{cmid_line}\n"
        f"{sub_line}\n"
        "\\midrule\n"
        f"\\multicolumn{{{N_DATA_COLS}}}{{c}}{{\\textit{{Closed-source Models}}}} \\\\\n"
        "\\midrule\n"
        f"{closed_body}\n"
        "\\midrule\n"
        f"\\multicolumn{{{N_DATA_COLS}}}{{c}}{{\\textit{{Open-source Models}}}} \\\\\n"
        "\\midrule\n"
        f"{open_body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    open(OUT_TEX, "w").write(body)
    print(f"wrote {OUT_TEX}")
    print(f"columns: Dataset, Acc, plus mean/std for {len(METRICS)} metrics "
          f"= {N_DATA_COLS} total")


if __name__ == "__main__":
    main()
