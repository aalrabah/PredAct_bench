"""Held-out generalization config: 5 OULAD modules NOT used in calibration,
spanning all 5 cutoffs. Reuses Exp 2's LLM_CONFIGS, ASSISTANT_LLM_KEY, etc."""
import os
from experiments.exp2_config import (
    LLM_CONFIGS, ASSISTANT_LLM_KEY,
    SAMPLE_SIZE, N_AT_RISK, MAX_CHAT_TURNS,
    MAX_CONCURRENT_EPISODES, RESUME_SKIP_EXISTING,
    PROJECT_ROOT, DATA_ROOT, resolved_llm_configs,
)

HELDOUT_CELLS = [
    {"dataset": "oulad", "target_accuracy": 0.40,
     "course_id": "BBB_2013B", "course_file": "BBB_2013B_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": 0.50,
     "course_id": "CCC_2014J", "course_file": "CCC_2014J_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": 0.60,
     "course_id": "DDD_2014B", "course_file": "DDD_2014B_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": 0.70,
     "course_id": "EEE_2013J", "course_file": "EEE_2013J_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": 0.80,
     "course_id": "GGG_2014B", "course_file": "GGG_2014B_week16.json",
     "week": 16, "feature_set": "full"},
]

HELDOUT_INSTRUCTORS = [
    "gpt5_5",
    "gemini_3_flash",
    "claude_opus_4_7",
    "deepseek_v4_flash",
    "gpt5_4_mini",
]

HELDOUT_RUNS_PER_CELL = 2

EXP2_HELDOUT_RESULTS_ROOT = os.path.join(DATA_ROOT, "results", "exp2_heldout")
HELDOUT_LOGS_ROOT         = os.path.join(EXP2_HELDOUT_RESULTS_ROOT, "sim_logs")
HELDOUT_AGGREGATE_CSV     = os.path.join(EXP2_HELDOUT_RESULTS_ROOT, "exp2_heldout_per_cell.csv")


def heldout_log_path(instructor_llm, dataset, accuracy, run_idx):
    acc_pct = int(round(accuracy * 100))
    return os.path.join(
        HELDOUT_LOGS_ROOT,
        instructor_llm,
        f"{dataset}_{acc_pct}",
        f"run_{run_idx:04d}.json",
    )
