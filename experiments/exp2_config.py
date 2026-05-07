"""
Exp 2 — Agent-to-Agent Simulator: configuration.

Single source of truth for what cells to run, which models to use, and where
logs land. Imported by exp2_sim_sweep.py (driver), exp2_aggregate.py
(metrics rollup), and visualize_exp2.py (plots).

Pilot scope:
  - Assistant fixed at gpt-4o-mini.
  - 4 closed-source instructors (GPT-4o-mini family + Claude family).
  - 6 open-source instructors as PLACEHOLDERS — flip enabled=True when ready.
  - 6 cells: 3 accuracies × 2 datasets, each pinned to a (course, week) combo
    where natural k-NN accuracy lands near the target (verified from exp1_raw.csv).
  - 100 runs per cell.
  - Sample 30 students per scenario, 5 forced at-risk (stratified).
"""

import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("PREDACT_DATA_ROOT", PROJECT_ROOT)


# =============================================================================
# ASSISTANT (fixed across all cells)
# =============================================================================

ASSISTANT_LLM_KEY = "gpt4o_mini"


# =============================================================================
# LLM CONFIGS — endpoints, models, API keys
# =============================================================================
# enabled=False entries are placeholders — sweep skips them. Flip to True
# when you've wired up the endpoint + key for that model.

LLM_CONFIGS = {
    # ---- Assistant (always runs) ---------------------------------------------
    "gpt4o_mini": {
        "enabled": True,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "extra_body": {},
    },

    # ---- Closed-source instructors (active) ---------------------------------
    "gpt5_4_mini": {
        "enabled": True,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4-mini",
        "extra_body": {},
        "supports_temperature": False,            # gpt-5 family rejects non-default temperature
        "uses_max_completion_tokens": True,       # gpt-5 family rejects max_tokens
    },
    "gpt5_5": {
        "enabled": True,
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.5",
        "extra_body": {},
        "supports_temperature": False,
        "uses_max_completion_tokens": True,
    },
    "claude_opus_4_7": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "anthropic/claude-opus-4.7",
        "extra_body": {},
    },
    "claude_haiku_4_5": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "anthropic/claude-haiku-4.5",
        "extra_body": {},
    },

    # ---- Open-source instructors (PLACEHOLDERS — enabled=False) -------------
    # Flip enabled=True and fill in the api_base / api_key_env when ready.
    "qwen_9b": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen3.5-9b",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "qwen_35b": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen3.5-35b-a3b",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "mistral_small_24b": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "mistralai/mistral-small-3.2-24b-instruct",
        "extra_body": {},
    },
    "ministral_3_14b": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "mistralai/ministral-14b-2512",
        "extra_body": {},
    },
    "deepseek_v4_flash": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "deepseek/deepseek-v4-flash",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "deepseek_v4_pro": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "deepseek/deepseek-v4-pro",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "gemini_3_1_pro": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "google/gemini-3.1-pro-preview",
        "extra_body": {},
    },
    "gemini_3_flash": {
        "enabled": True,
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "google/gemini-3-flash-preview",
        "extra_body": {},
    },
}


# Instructors to sweep over (just the keys; sweep filters by enabled flag).
INSTRUCTOR_LLM_KEYS = [
    "gpt4o_mini",
    "gpt5_4_mini",
    "gpt5_5",
    "claude_opus_4_7",
    "claude_haiku_4_5",
    "qwen_9b",
    "qwen_35b",
    "mistral_small_24b",
    "ministral_3_14b",
    "deepseek_v4_flash",
    "deepseek_v4_pro",
    "gemini_3_1_pro",
    "gemini_3_flash",
]


# =============================================================================
# CELLS — (dataset, accuracy) → (course, course_file, week, feature_set)
# =============================================================================
# Each cell is one experimental condition. Course/week chosen so the natural
# k-NN accuracy lands near the target (verified from results/exp1/exp1_raw.csv).

CELLS = [
    # UIUC — week 8 everywhere (40/60/80 match the human study)
    {
        "dataset": "uiuc", "target_accuracy": 0.40,
        "course_id": "CS 374", "course_file": "CS374_week8.json",
        "week": 8, "feature_set": "full",
    },
    {
        "dataset": "uiuc", "target_accuracy": 0.50,
        "course_id": "CS 450", "course_file": "CS450_week8.json",
        "week": 8, "feature_set": "full",
    },
    {
        "dataset": "uiuc", "target_accuracy": 0.60,
        "course_id": "CS 461", "course_file": "CS461_week8.json",
        "week": 8, "feature_set": "full",
    },
    {
        "dataset": "uiuc", "target_accuracy": 0.70,
        "course_id": "CS 105", "course_file": "CS105_week8.json",
        "week": 8, "feature_set": "full",
    },
    {
        "dataset": "uiuc", "target_accuracy": 0.80,
        "course_id": "CS 421", "course_file": "CS421_week8.json",
        "week": 8, "feature_set": "full",
    },
    # OULAD — per-(course, week) anchors near each target
    {
        "dataset": "oulad", "target_accuracy": 0.40,
        "course_id": "AAA_2013J", "course_file": "AAA_2013J_week16.json",
        "week": 16, "feature_set": "full",
    },
    {
        "dataset": "oulad", "target_accuracy": 0.50,
        "course_id": "AAA_2013J", "course_file": "AAA_2013J_week8.json",
        "week": 8, "feature_set": "full",
    },
    {
        "dataset": "oulad", "target_accuracy": 0.60,
        "course_id": "AAA_2013J", "course_file": "AAA_2013J_week20.json",
        "week": 20, "feature_set": "full",
    },
    {
        "dataset": "oulad", "target_accuracy": 0.70,
        "course_id": "AAA_2014J", "course_file": "AAA_2014J_week17.json",
        "week": 17, "feature_set": "full",
    },
    {
        "dataset": "oulad", "target_accuracy": 0.80,
        "course_id": "FFF_2013B", "course_file": "FFF_2013B_week32.json",
        "week": 32, "feature_set": "full",
    },
]


# =============================================================================
# Episode-level knobs
# =============================================================================

N_RUNS_PER_CELL  = 10    # episodes per (instructor, dataset, accuracy) cell
SAMPLE_SIZE      = 30    # students sampled per episode
N_AT_RISK        = 5     # forced at-risk count in the stratified sample
MAX_CHAT_TURNS   = 10    # max instructor↔assistant turns per episode


# =============================================================================
# Sweep-level knobs
# =============================================================================

MAX_CONCURRENT_EPISODES = 20   # ThreadPoolExecutor workers
RESUME_SKIP_EXISTING    = True # skip runs whose log already exists


# =============================================================================
# Output paths
# =============================================================================

EXP2_RESULTS_ROOT = os.path.join(DATA_ROOT, "results", "exp2")
SIM_LOGS_ROOT     = os.path.join(EXP2_RESULTS_ROOT, "sim_logs")
AGGREGATE_CSV     = os.path.join(EXP2_RESULTS_ROOT, "exp2_per_cell.csv")


def log_path_for(instructor_llm, dataset, accuracy, run_idx):
    """Where one episode's log file lands."""
    acc_pct = int(round(accuracy * 100))
    return os.path.join(
        SIM_LOGS_ROOT,
        instructor_llm,
        f"{dataset}_{acc_pct}",
        f"run_{run_idx:04d}.json",
    )


def enabled_instructor_keys():
    """Keys in INSTRUCTOR_LLM_KEYS whose config has enabled=True."""
    return [k for k in INSTRUCTOR_LLM_KEYS if LLM_CONFIGS.get(k, {}).get("enabled")]


def resolved_llm_configs():
    """Resolve api_key_env → actual key from os.environ. Returns the dict
    LLM_CONFIGS expects (api_key field populated)."""
    out = {}
    for key, cfg in LLM_CONFIGS.items():
        out[key] = dict(cfg)
        env_var = cfg.get("api_key_env")
        out[key]["api_key"] = os.environ.get(env_var, "") if env_var and env_var != "TODO" else ""
    return out
