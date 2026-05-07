"""Uncalibrated tools config: 5 OULAD modules, target_accuracy=None
(natural k-NN output, no noise injection).
Reuses Exp 2's LLM_CONFIGS, ASSISTANT_LLM_KEY, etc. — but writes to a
separate output dir so nothing collides."""
import os
from experiments.exp2_config import (
    LLM_CONFIGS, ASSISTANT_LLM_KEY,
    SAMPLE_SIZE, N_AT_RISK, MAX_CHAT_TURNS,
    MAX_CONCURRENT_EPISODES, RESUME_SKIP_EXISTING,
    PROJECT_ROOT, DATA_ROOT, resolved_llm_configs,
)

# 5 held-out OULAD modules (same as in exp2_heldout_config) — re-using them
# so the apples-to-apples comparison is on the same courses, just with vs without
# calibration. target_accuracy=None disables noise injection.
UNCALIB_CELLS = [
    {"dataset": "oulad", "target_accuracy": None,
     "course_id": "BBB_2013B", "course_file": "BBB_2013B_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": None,
     "course_id": "CCC_2014J", "course_file": "CCC_2014J_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": None,
     "course_id": "DDD_2014B", "course_file": "DDD_2014B_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": None,
     "course_id": "EEE_2013J", "course_file": "EEE_2013J_week16.json",
     "week": 16, "feature_set": "full"},
    {"dataset": "oulad", "target_accuracy": None,
     "course_id": "GGG_2014B", "course_file": "GGG_2014B_week16.json",
     "week": 16, "feature_set": "full"},
]

UNCALIB_INSTRUCTORS = [
    "gpt5_5",
    "gemini_3_flash",
    "claude_opus_4_7",
    "deepseek_v4_flash",
    "gpt5_4_mini",
]

UNCALIB_RUNS_PER_CELL = 2

EXP2_UNCALIB_RESULTS_ROOT = os.path.join(DATA_ROOT, "results", "exp2_uncalib")
UNCALIB_LOGS_ROOT         = os.path.join(EXP2_UNCALIB_RESULTS_ROOT, "sim_logs")
UNCALIB_AGGREGATE_CSV     = os.path.join(EXP2_UNCALIB_RESULTS_ROOT, "exp2_uncalib_per_cell.csv")


def uncalib_log_path(instructor_llm, course_id, run_idx):
    """Logs go under instructor/oulad_<course>/run_NNNN.json (no accuracy in path)."""
    return os.path.join(
        UNCALIB_LOGS_ROOT,
        instructor_llm,
        f"oulad_{course_id}",
        f"run_{run_idx:04d}.json",
    )
