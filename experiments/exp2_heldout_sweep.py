"""Held-out generalization sweep: 5 instructors x 5 OULAD held-out cells x 2 runs = 50 episodes.

One command:
    python -m experiments.exp2_heldout_sweep
"""
import os
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add repo root to path so 'experiments' and 'sim' are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp2_heldout_config import (
    HELDOUT_CELLS, HELDOUT_INSTRUCTORS, HELDOUT_RUNS_PER_CELL,
    heldout_log_path, HELDOUT_LOGS_ROOT,
)
from experiments.exp2_config import (
    ASSISTANT_LLM_KEY, SAMPLE_SIZE, N_AT_RISK, MAX_CHAT_TURNS,
    MAX_CONCURRENT_EPISODES, RESUME_SKIP_EXISTING, resolved_llm_configs,
)
from experiments.exp2_sim_sweep import _get_scenario_data, stratified_sample
from sim.episode import run_episode
from tools import build_grades_lookup


def run_one_heldout(instructor_llm, cell, run_idx, llm_configs):
    out_path = heldout_log_path(instructor_llm, cell["dataset"], cell["target_accuracy"], run_idx)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if RESUME_SKIP_EXISTING and os.path.exists(out_path):
        return ("skip", out_path, None)
    try:
        db, all_students, gt_grades = _get_scenario_data(cell["dataset"], cell["course_file"])
        sample_seed = hash(
            f"sample_heldout_{cell['dataset']}_{cell['course_id']}_{cell['target_accuracy']}_{run_idx}"
        ) & 0xFFFFFFFF
        students = stratified_sample(
            all_students, gt_grades,
            sample_size=SAMPLE_SIZE, n_at_risk=N_AT_RISK, seed=sample_seed,
        )
        grades_lookup = build_grades_lookup(students)
        episode_seed = hash(
            f"heldout_{instructor_llm}_{cell['dataset']}_{cell['course_id']}_{cell['target_accuracy']}_{run_idx}"
        ) & 0xFFFFFFFF
        log = run_episode(
            assistant_llm_key=ASSISTANT_LLM_KEY,
            instructor_llm_key=instructor_llm,
            llm_configs=llm_configs,
            db=db, students=students, grades_lookup=grades_lookup, gt_grades=gt_grades,
            course_id=cell["course_id"], course_file=cell["course_file"],
            week=cell["week"], feature_set=cell["feature_set"],
            target_acc=cell["target_accuracy"],
            seed=episode_seed, max_chat_turns=MAX_CHAT_TURNS,
        )
        log["dataset"]      = cell["dataset"]
        log["run_idx"]      = run_idx
        log["sample_seed"]  = sample_seed
        log["episode_seed"] = episode_seed
        log["heldout"]      = True
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
    for instr in HELDOUT_INSTRUCTORS:
        for cell in HELDOUT_CELLS:
            for run_idx in range(HELDOUT_RUNS_PER_CELL):
                plan.append((instr, cell, run_idx))
    print(f"Held-out sweep: {len(plan)} episodes "
          f"({len(HELDOUT_INSTRUCTORS)} models x {len(HELDOUT_CELLS)} cells "
          f"x {HELDOUT_RUNS_PER_CELL} runs)")
    print(f"Logs -> {HELDOUT_LOGS_ROOT}")
    print(f"Parallel workers: {MAX_CONCURRENT_EPISODES}\n")

    llm_configs = resolved_llm_configs()
    n_ok = n_skip = n_err = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EPISODES) as ex:
        futures = {
            ex.submit(run_one_heldout, instr, cell, ridx, llm_configs):
                (instr, cell, ridx)
            for instr, cell, ridx in plan
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            instr, cell, ridx = futures[fut]
            status, path, err = fut.result()
            tag = f"[{i:>3}/{len(plan)}] {instr:<20} {cell['dataset']}_{int(cell['target_accuracy']*100)}_{cell['course_id']}_run{ridx:02d}"
            if status == "ok":     n_ok   += 1; print(f"  {tag}  OK")
            elif status == "skip": n_skip += 1; print(f"  {tag}  skip (exists)")
            else:                  n_err  += 1; print(f"  {tag}  ERROR -> {path}")
    print(f"\nDone. OK={n_ok}  skip={n_skip}  err={n_err}")


if __name__ == "__main__":
    main()
