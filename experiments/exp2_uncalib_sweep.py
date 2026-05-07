"""Uncalibrated tools sweep: 5 instructors x 5 OULAD courses x 2 runs = 50 episodes,
all with target_accuracy=None (raw k-NN output, no noise injection).

The instructor is told the tool's accuracy is UNKNOWN (rather than a specific %).
This is achieved by swapping the {accuracy_pct}% phrasing in the
instructor prompts in-memory at startup, restored on exit. No file edits.

One command:
    python -m experiments.exp2_uncalib_sweep
"""
import os
import sys
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompts as _prompts_module
from experiments.exp2_uncalib_config import (
    UNCALIB_CELLS, UNCALIB_INSTRUCTORS, UNCALIB_RUNS_PER_CELL,
    uncalib_log_path, UNCALIB_LOGS_ROOT,
)
from experiments.exp2_config import (
    ASSISTANT_LLM_KEY, SAMPLE_SIZE, N_AT_RISK, MAX_CHAT_TURNS,
    MAX_CONCURRENT_EPISODES, RESUME_SKIP_EXISTING, resolved_llm_configs,
)
from experiments.exp2_sim_sweep import _get_scenario_data, stratified_sample
from sim.episode import run_episode
from tools import build_grades_lookup


def _swap_phrase(s, target_phrases):
    out = s
    for old, new in target_phrases:
        out = out.replace(old, new)
    return out


# ----- Patch instructor prompts so {accuracy_pct} disappears and the model
# is told "the tool's accuracy is unknown" instead of "about X% accurate".
_PHRASES = [
    # Long-form replacements first (most specific) to avoid double-substitution.
    ("The agent is about {accuracy_pct}% accurate in this course.",
     "The agent's accuracy in this course is UNKNOWN."),
    ("Predictions in this course are about {accuracy_str} accurate.",
     "The accuracy of predictions in this course is UNKNOWN."),
    ("about {accuracy_pct}% accurate in this course",
     "of UNKNOWN accuracy in this course"),
    ("only {accuracy_pct}% accurate", "of UNKNOWN accuracy"),
    ("about {accuracy_pct}% accurate", "of UNKNOWN accuracy"),
]

_ORIGINALS = {}
def patch_prompts():
    names = ("SIM_INSTRUCTOR_SYSTEM_PROMPT",
             "SIM_INSTRUCTOR_INITIAL_DECISION_TEMPLATE",
             "SIM_INSTRUCTOR_CHAT_TURN_TEMPLATE",
             "SIM_INSTRUCTOR_FINAL_DECISION_TEMPLATE",
             "SIM_ASSISTANT_SYSTEM_PROMPT")
    for n in names:
        _ORIGINALS[n] = getattr(_prompts_module, n)
        setattr(_prompts_module, n, _swap_phrase(_ORIGINALS[n], _PHRASES))

def restore_prompts():
    for n, v in _ORIGINALS.items():
        setattr(_prompts_module, n, v)


def run_one_uncalib(instructor_llm, cell, run_idx, llm_configs):
    out_path = uncalib_log_path(instructor_llm, cell["course_id"], run_idx)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if RESUME_SKIP_EXISTING and os.path.exists(out_path):
        return ("skip", out_path, None)
    try:
        db, all_students, gt_grades = _get_scenario_data(cell["dataset"], cell["course_file"])
        sample_seed = hash(
            f"sample_uncalib_{cell['dataset']}_{cell['course_id']}_{run_idx}"
        ) & 0xFFFFFFFF
        students = stratified_sample(
            all_students, gt_grades,
            sample_size=SAMPLE_SIZE, n_at_risk=N_AT_RISK, seed=sample_seed,
        )
        grades_lookup = build_grades_lookup(students)
        episode_seed = hash(
            f"uncalib_{instructor_llm}_{cell['dataset']}_{cell['course_id']}_{run_idx}"
        ) & 0xFFFFFFFF
        log = run_episode(
            assistant_llm_key=ASSISTANT_LLM_KEY,
            instructor_llm_key=instructor_llm,
            llm_configs=llm_configs,
            db=db, students=students, grades_lookup=grades_lookup, gt_grades=gt_grades,
            course_id=cell["course_id"], course_file=cell["course_file"],
            week=cell["week"], feature_set=cell["feature_set"],
            target_acc=None,           # <-- key: no noise injection
            seed=episode_seed,
            max_chat_turns=MAX_CHAT_TURNS,
            condition_id=f"{instructor_llm}_uncalib_{cell['course_id']}",
        )
        log["dataset"]      = cell["dataset"]
        log["run_idx"]      = run_idx
        log["sample_seed"]  = sample_seed
        log["episode_seed"] = episode_seed
        log["uncalibrated"] = True
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)
        return ("ok", out_path, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        err_path = out_path + ".error"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(err)
        return ("err", err_path, err)


def main():
    plan = []
    for instr in UNCALIB_INSTRUCTORS:
        for cell in UNCALIB_CELLS:
            for run_idx in range(UNCALIB_RUNS_PER_CELL):
                plan.append((instr, cell, run_idx))
    print(f"Uncalibrated sweep: {len(plan)} episodes "
          f"({len(UNCALIB_INSTRUCTORS)} models x {len(UNCALIB_CELLS)} courses "
          f"x {UNCALIB_RUNS_PER_CELL} runs)")
    print(f"Logs -> {UNCALIB_LOGS_ROOT}")
    print(f"target_accuracy = None (raw k-NN output, instructor told accuracy is unknown)\n")

    patch_prompts()  # in-memory swap, restored at the end via try/finally
    try:
        llm_configs = resolved_llm_configs()
        n_ok = n_skip = n_err = 0
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EPISODES) as ex:
            futures = {
                ex.submit(run_one_uncalib, instr, cell, ridx, llm_configs):
                    (instr, cell, ridx)
                for instr, cell, ridx in plan
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                instr, cell, ridx = futures[fut]
                status, path, err = fut.result()
                tag = f"[{i:>3}/{len(plan)}] {instr:<20} {cell['course_id']:<10} run{ridx:02d}"
                if status == "ok":     n_ok   += 1; print(f"  {tag}  OK")
                elif status == "skip": n_skip += 1; print(f"  {tag}  skip (exists)")
                else:                  n_err  += 1; print(f"  {tag}  ERROR -> {path}")
        print(f"\nDone. OK={n_ok}  skip={n_skip}  err={n_err}")
    finally:
        restore_prompts()


if __name__ == "__main__":
    main()
