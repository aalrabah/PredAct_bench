"""
Exp 2 — Agent-to-Agent Simulator: sweep driver.

For every (enabled instructor LLM × cell × run_idx), runs one episode and
writes its log to disk. Idempotent: re-running skips runs whose log
already exists. Parallel: uses a ThreadPoolExecutor since episodes are
I/O-bound (LLM calls).

Usage:
    python -m experiments.exp2_sim_sweep                # full enabled grid
    python -m experiments.exp2_sim_sweep --dry-run      # print plan, do nothing
    python -m experiments.exp2_sim_sweep --instructor claude_haiku_4_5
    python -m experiments.exp2_sim_sweep --dataset predact_cs --accuracy 0.6
    python -m experiments.exp2_sim_sweep --runs 5       # override N_RUNS_PER_CELL
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from experiments.exp2_config import (
    ASSISTANT_LLM_KEY,
    CELLS,
    LLM_CONFIGS,
    MAX_CHAT_TURNS,
    MAX_CONCURRENT_EPISODES,
    N_AT_RISK,
    N_RUNS_PER_CELL,
    RESUME_SKIP_EXISTING,
    SAMPLE_SIZE,
    SIM_LOGS_ROOT,
    enabled_instructor_keys,
    log_path_for,
    resolved_llm_configs,
)

from sim.episode import load_scenario_data, run_episode, stratified_sample
from tools import build_grades_lookup


# -----------------------------------------------------------------------------
# Plan + filter
# -----------------------------------------------------------------------------

def build_plan(instructors, cells, n_runs):
    """Cartesian product → list of (instructor, cell, run_idx)."""
    plan = []
    for instructor in instructors:
        for cell in cells:
            for run_idx in range(n_runs):
                plan.append((instructor, cell, run_idx))
    return plan


def filter_existing(plan):
    """Drop runs whose log already exists on disk (resume support)."""
    out = []
    skipped = 0
    for instructor, cell, run_idx in plan:
        path = log_path_for(instructor, cell["dataset"], cell["target_accuracy"], run_idx)
        if RESUME_SKIP_EXISTING and os.path.exists(path):
            skipped += 1
            continue
        out.append((instructor, cell, run_idx))
    return out, skipped


# -----------------------------------------------------------------------------
# Per-episode worker
# -----------------------------------------------------------------------------

# Cache scenario data per (dataset, course_file) so we don't re-load JSON
# for every run in the same cell.
_scenario_cache = {}
_cache_lock = None


def _get_scenario_data(dataset, course_file):
    key = (dataset, course_file)
    if key in _scenario_cache:
        return _scenario_cache[key]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dataset == "predact_cs":
        # load_scenario_data already loads the PredAct-CS training DB by default.
        db, all_students, _, gt_grades = load_scenario_data(course_file)
    elif dataset == "oulad":
        # OULAD has its own training DB and test sets.
        from tools import load_db
        train_db_path = os.path.join(project_root, "results", "oulad", "oulad_db_train.json")
        test_set_path = os.path.join(project_root, "results", "oulad", "test_sets", course_file)
        gt_path = os.path.join(project_root, "results", "oulad", "ground_truth_for_cutoff_data.json")
        db = load_db(train_db_path)
        with open(test_set_path, "r", encoding="utf-8") as f:
            test_set = json.load(f)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        all_students = test_set["students"]
        gt_grades = gt.get(course_file, {}).get("student_grades", {})
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    _scenario_cache[key] = (db, all_students, gt_grades)
    return _scenario_cache[key]


def run_one(instructor_llm, cell, run_idx, llm_configs):
    """Run a single episode and write its log."""
    out_path = log_path_for(instructor_llm, cell["dataset"], cell["target_accuracy"], run_idx)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if RESUME_SKIP_EXISTING and os.path.exists(out_path):
        return ("skip", out_path, None)

    try:
        db, all_students, gt_grades = _get_scenario_data(cell["dataset"], cell["course_file"])

        # Stratified sample with a per-run seed so each run sees a different subset.
        sample_seed = hash(f"sample_{cell['dataset']}_{cell['target_accuracy']}_{run_idx}") & 0xFFFFFFFF
        students = stratified_sample(
            all_students, gt_grades,
            sample_size=SAMPLE_SIZE,
            n_at_risk=N_AT_RISK,
            seed=sample_seed,
        )
        grades_lookup = build_grades_lookup(students)

        # Per-run noise/instructor seed.
        episode_seed = hash(
            f"{instructor_llm}_{cell['dataset']}_{cell['target_accuracy']}_{run_idx}"
        ) & 0xFFFFFFFF

        log = run_episode(
            assistant_llm_key=ASSISTANT_LLM_KEY,
            instructor_llm_key=instructor_llm,
            llm_configs=llm_configs,
            db=db,
            students=students,
            grades_lookup=grades_lookup,
            gt_grades=gt_grades,
            course_id=cell["course_id"],
            course_file=cell["course_file"],
            week=cell["week"],
            feature_set=cell["feature_set"],
            target_acc=cell["target_accuracy"],
            seed=episode_seed,
            max_chat_turns=MAX_CHAT_TURNS,
        )

        # Augment log with experiment metadata
        log["dataset"] = cell["dataset"]
        log["run_idx"] = run_idx
        log["sample_seed"] = sample_seed
        log["episode_seed"] = episode_seed

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)

        return ("ok", out_path, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        # Don't crash the sweep — log the error to a sidecar file.
        err_path = out_path + ".error"
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(err)
        return ("err", err_path, err)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructor", action="append",
                        help="Restrict to one or more instructor llm_key(s). Repeatable.")
    parser.add_argument("--dataset", choices=["predact_cs", "oulad"],
                        help="Restrict to one dataset.")
    parser.add_argument("--accuracy", type=float,
                        help="Restrict to one accuracy level (e.g. 0.6).")
    parser.add_argument("--runs", type=int, default=N_RUNS_PER_CELL,
                        help=f"Override N_RUNS_PER_CELL (default {N_RUNS_PER_CELL}).")
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT_EPISODES,
                        help=f"Concurrent episodes (default {MAX_CONCURRENT_EPISODES}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan, do not run.")
    args = parser.parse_args()

    # Resolve which instructors / cells to run.
    instructors = enabled_instructor_keys()
    if args.instructor:
        instructors = [k for k in instructors if k in args.instructor]
        if not instructors:
            print(f"No enabled instructor matches {args.instructor}.")
            return

    cells = list(CELLS)
    if args.dataset:
        cells = [c for c in cells if c["dataset"] == args.dataset]
    if args.accuracy is not None:
        cells = [c for c in cells if abs(c["target_accuracy"] - args.accuracy) < 1e-6]
    if not cells:
        print("No cells match the dataset/accuracy filters.")
        return

    plan = build_plan(instructors, cells, args.runs)
    plan, n_skipped = filter_existing(plan)

    print(f"Sweep plan:")
    print(f"  Instructors: {instructors}")
    print(f"  Cells: {len(cells)} ({[(c['dataset'], c['target_accuracy']) for c in cells]})")
    print(f"  Runs/cell:   {args.runs}")
    print(f"  Total episodes planned: {len(instructors) * len(cells) * args.runs}")
    print(f"  Already complete (skipped): {n_skipped}")
    print(f"  Episodes to run now:        {len(plan)}")
    print(f"  Workers (concurrency):      {args.workers}")
    print(f"  Logs root:   {SIM_LOGS_ROOT}")

    if args.dry_run:
        print("\n--dry-run: not executing.")
        return

    # Resolve API keys from .env
    llm_configs = resolved_llm_configs()
    # Sanity-check keys for active endpoints
    for instructor in instructors:
        cfg = llm_configs[instructor]
        if not cfg["api_key"]:
            print(f"  WARN: api_key for {instructor} is empty (env var: {cfg['api_key_env']})")
    if not llm_configs[ASSISTANT_LLM_KEY]["api_key"]:
        print(f"  ERROR: api_key for assistant {ASSISTANT_LLM_KEY} is empty.")
        sys.exit(1)

    if not plan:
        print("Nothing to do.")
        return

    # Run.
    t_start = time.time()
    n_ok = n_err = 0
    print(f"\nLaunching {len(plan)} episodes with {args.workers} workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one, instructor, cell, run_idx, llm_configs):
                (instructor, cell, run_idx)
            for instructor, cell, run_idx in plan
        }

        for i, fut in enumerate(as_completed(futures), 1):
            instructor, cell, run_idx = futures[fut]
            status, path, err = fut.result()
            tag = f"[{i:>4}/{len(plan)}] {instructor:<18} {cell['dataset']}_{int(cell['target_accuracy']*100)}_run{run_idx:04d}"
            if status == "ok":
                n_ok += 1
                print(f"{tag}  OK")
            elif status == "skip":
                pass  # silent skip
            else:
                n_err += 1
                first_line = err.splitlines()[0] if err else "?"
                print(f"{tag}  ERR  {first_line}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s. ok={n_ok} err={n_err} (skipped earlier: {n_skipped})")


if __name__ == "__main__":
    main()
