"""plot_cost_latency_table.py — cost/latency/quality table for 13 LLM
instructors. Outputs Markdown to stdout and LaTeX to figures/.

Cost is ESTIMATED from dialogue text length × list prices in PRICES.
Token estimate: chars / 4 for dialogue, plus a fixed system-prompt overhead per call.
"""
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import glob
from collections import defaultdict

import pandas as pd

EXP2_LOGS = os.path.join(PROJECT_ROOT, "results/exp2/sim_logs")
EXP2_CSV  = os.path.join(PROJECT_ROOT, "results/exp2/exp2_per_cell.csv")
OUT_TEX   = os.path.join(PROJECT_ROOT, "figures/cost_latency_table.tex")

DISPLAY = {
    "gpt5_5": "GPT-5.5", "gpt5_4_mini": "GPT-5.4 Mini", "gpt4o_mini": "GPT-4o Mini",
    "claude_opus_4_7": "Claude Opus 4.7", "claude_haiku_4_5": "Claude Haiku 4.5",
    "gemini_3_1_pro": "Gemini 3.1 Pro", "gemini_3_flash": "Gemini 3 Flash",
    "deepseek_v4_pro": "DeepSeek V4 Pro", "deepseek_v4_flash": "DeepSeek V4 Flash",
    "mistral_small_24b": "Mistral Small 24B", "ministral_3_14b": "Ministral 3 14B",
    "qwen_35b": "Qwen 35B", "qwen_9b": "Qwen 9B",
}

# (input $/M tokens, output $/M tokens). Sources: OpenAI direct + OpenRouter.
PRICES = {
    "gpt5_5":            (5.00,   30.00),
    "gpt5_4_mini":       (0.75,    4.50),
    "gpt4o_mini":        (0.15,    0.60),
    "claude_opus_4_7":   (5.00,   25.00),
    "claude_haiku_4_5":  (1.00,    5.00),
    "gemini_3_1_pro":    (2.00,   12.00),
    "gemini_3_flash":    (0.50,    3.00),
    "deepseek_v4_pro":   (0.435,   0.87),
    "deepseek_v4_flash": (0.14,    0.28),
    "mistral_small_24b": (0.15,    0.60),
    "ministral_3_14b":   (0.20,    0.20),
    "qwen_35b":          (0.1612,  0.9653),
    "qwen_9b":           (0.10,    0.15),
}

INSTR_SYSTEM_OVERHEAD  = 2000   # tokens of system + tool schema, per API call
ASSIST_SYSTEM_OVERHEAD = 1500
DECISION_CALL_OUTPUT   = 200    # JSON output for initial / final decision calls
CHARS_PER_TOKEN        = 4


def estimate_episode(d):
    llm = d.get("instructor_llm")
    if llm not in PRICES:
        return None
    in_M_instr,  out_M_instr  = PRICES[llm]
    in_M_assist, out_M_assist = PRICES["gpt4o_mini"]  # assistant always gpt4o_mini

    dialogue = d.get("dialogue_history") or []
    n_turns  = len(dialogue)
    n_calls  = max(1, (n_turns + 1) // 2)   # instructor turns drive API calls

    instr_chars  = sum(len(m["content"]) for m in dialogue if m["role"] == "instructor")
    assist_chars = sum(len(m["content"]) for m in dialogue if m["role"] == "assistant")
    instr_t  = instr_chars  // CHARS_PER_TOKEN
    assist_t = assist_chars // CHARS_PER_TOKEN

    # Instructor LLM: per-call system overhead + cumulative assistant context;
    # output = its own messages + 2 decision calls
    instr_in_tok  = INSTR_SYSTEM_OVERHEAD * (n_calls + 2) + assist_t
    instr_out_tok = instr_t + 2 * DECISION_CALL_OUTPUT

    # Assistant LLM (gpt4o_mini): per-call system + each instructor query;
    # output = its responses
    assist_in_tok  = ASSIST_SYSTEM_OVERHEAD * n_calls + instr_t
    assist_out_tok = assist_t

    instr_cost  = (instr_in_tok  * in_M_instr  + instr_out_tok  * out_M_instr)  / 1_000_000
    assist_cost = (assist_in_tok * in_M_assist + assist_out_tok * out_M_assist) / 1_000_000

    return {
        "instr_cost":  instr_cost,
        "assist_cost": assist_cost,
        "duration":    d.get("duration_seconds", 0),
        "turns":       n_turns,
    }


def main():
    f1_df = pd.read_csv(EXP2_CSV)
    mean_f1 = f1_df.groupby("instructor_llm")["f1_final_mean"].mean().to_dict()

    per_model = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(EXP2_LOGS, "*", "*", "run_*.json")):
        d = json.load(open(path))
        rec = estimate_episode(d)
        if rec is None:
            continue
        for k, v in rec.items():
            per_model[d["instructor_llm"]][k].append(v)

    rows = []
    for llm, b in per_model.items():
        n = len(b["instr_cost"])
        if n == 0:
            continue
        rows.append({
            "llm":           llm,
            "model":         DISPLAY.get(llm, llm),
            "n_episodes":    n,
            "mean_f1":       mean_f1.get(llm, float("nan")),
            "median_lat":    pd.Series(b["duration"]).median(),
            "mean_turns":    sum(b["turns"]) / n,
            "instr_per_ep":  sum(b["instr_cost"])  / n,
            "assist_per_ep": sum(b["assist_cost"]) / n,
        })
    rows.sort(key=lambda r: -r["mean_f1"])

    avg_assist = sum(r["assist_per_ep"] for r in rows) / len(rows)

    # --- Markdown print -----------------------------------------------------
    print(f"{'Model':<22} {'F1':>6}   {'Lat (s)':>8}   {'Turns':>6}   "
          f"{'$ Instr/ep':>11}   {'$ Total/ep':>11}   N")
    print("-" * 92)
    for r in rows:
        total = r["instr_per_ep"] + r["assist_per_ep"]
        print(f"{r['model']:<22} {r['mean_f1']:>6.3f}   "
              f"{r['median_lat']:>8.1f}   {r['mean_turns']:>6.1f}   "
              f"${r['instr_per_ep']:>9.4f}   ${total:>9.4f}   {r['n_episodes']}")
    print()
    print(f"Note: each $ Total/ep includes a fixed assistant overhead of "
          f"~${avg_assist:.4f} (GPT-4o Mini, every episode).")

    # --- LaTeX --------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Cost / latency / quality trade-off across 13 instructor "
                "LLMs. Costs are estimated from dialogue text length "
                "($\\approx$\\,chars\\,/\\,4 tokens) plus a fixed system-prompt "
                "overhead per API call, multiplied by published list prices "
                "(USD per million tokens, sources: OpenAI direct + OpenRouter, "
                "as of 2026-05). Latency is median wall-clock per episode. "
                "Each row pools all episodes per model (PredAct-CS + OULAD, all 5 "
                "cutoffs). Each \\$\\,Total/ep includes a fixed assistant overhead of "
                f"$\\sim$\\${avg_assist:.4f} for the GPT-4o Mini assistant role, "
                "incurred every episode regardless of instructor.}\n")
        f.write("\\label{tab:cost_latency}\n")
        f.write("\\begin{tabular}{l c c c c c}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Mean F1} & \\textbf{Median lat. (s)} & "
                "\\textbf{Mean turns} & \\textbf{\\$\\,Instr/ep} & "
                "\\textbf{\\$\\,Total/ep} \\\\\n\\midrule\n")
        for r in rows:
            total = r["instr_per_ep"] + r["assist_per_ep"]
            f.write(f"{r['model']} & {r['mean_f1']:.2f} & {r['median_lat']:.1f} & "
                    f"{r['mean_turns']:.1f} & "
                    f"\\${r['instr_per_ep']:.4f} & \\${total:.4f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Assistant overhead & --- & --- & --- & --- & "
                f"\\${avg_assist:.4f} \\\\\n")
        f.write("\\multicolumn{6}{l}{\\footnotesize\\textit{(always GPT-4o Mini, "
                "charged every episode regardless of instructor)}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\nLaTeX -> {OUT_TEX}")


if __name__ == "__main__":
    main()
