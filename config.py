"""
PredAct Benchmark - Configuration
All tunable parameters in one place.
"""

# =============================================================================
# MODEL & INFERENCE
# =============================================================================

# vLLM endpoint (local server)
VLLM_BASE_URL = "http://localhost:8000/v1"

# Models for each agent (can be the same or different)
AGENT_USER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
AGENT_SYSTEM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Generation parameters
TEMPERATURE = 0.7
MAX_TOKENS = 1024
TOP_P = 0.95

# =============================================================================
# DIALOGUE CONTROL
# =============================================================================

# Maximum turns per dialogue (one turn = user + system pair)
MAX_TURNS = 6

# Minimum turns before allowing dialogue to end
MIN_TURNS = 3

# =============================================================================
# MATCHING & PREDICTION
# =============================================================================

# Per-component score tolerance for nearest-neighbor matching (percentage points)
MATCH_TOLERANCE = 2.0

# If no matches found, widen tolerance incrementally by this amount
MATCH_TOLERANCE_STEP = 1.0

# Maximum tolerance before giving up on matching
MATCH_TOLERANCE_MAX = 10.0

# Minimum number of matched historical students to make a prediction
MIN_MATCHES = 3

# =============================================================================
# RISK MAPPING
# =============================================================================

# Predicted grade → failure_risk level
RISK_MAPPING = {
    "a": None,
    "b": None,
    "c": "medium",
    "d": "high",
    "f": "critical",
}

# Grade to GPA mapping for numeric calculations
GRADE_TO_GPA = {
    "a": 4.0,
    "b": 3.0,
    "c": 2.0,
    "d": 1.0,
    "f": 0.0,
}

# =============================================================================
# INTERVENTION LOGIC
# =============================================================================

# Risk level → default intervention type
DEFAULT_INTERVENTION = {
    "low": "monitor_only",
    "medium": "check_in_message",
    "high": "tutoring_referral",
    "critical": "advising_referral",
}

# Risk level → default intervention goal
DEFAULT_INTERVENTION_GOAL = {
    "low": "reduce_failure_risk",
    "medium": "improve_engagement",
    "high": "improve_concept_mastery",
    "critical": "connect_to_resources",
}

# Risk level → default contact mode
DEFAULT_CONTACT_MODE = {
    "low": "no_contact",
    "medium": "email",
    "high": "email",
    "critical": "advisor_referral",
}

# =============================================================================
# FILE PATHS
# =============================================================================

CS_DB_PATH = "cs_db.json"
ONTOLOGY_PATH = "ontology.json"
OUTPUT_DATA_PATH = "data.json"
LOGS_DIR = "logs/"

# =============================================================================
# EVALUATION
# =============================================================================

# Tolerance for open_numeric slot evaluation
EVAL_GPA_TOLERANCE = 0.2          # ±0.2 for average_gpa
EVAL_COUNT_TOLERANCE = 1          # ±1 for flagged_student_count, missing_assignments