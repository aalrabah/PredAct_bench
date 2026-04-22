"""
PredAct Benchmark - Human Study Interface

Ten scenarios (all week 8) organized into 4 blocks, each followed by a questionnaire:

  Block 1: no_agent        - CS 225, week 8 (25 students)
  Block 2: GPT-4o-mini    - CS 374 (40%), CS 461 (60%), CS 421 (80%)
  Block 3: Qwen3.5-9B     - CS 374 (40%), CS 461 (60%), CS 421 (80%)
  Block 4: Qwen3.5-35B-A3B - CS 374 (40%), CS 461 (60%), CS 421 (80%)

Same courses repeat across LLM blocks (within-subjects on LLM with course held
constant). Each (LLM x accuracy) cell uses an independent noise seed
hash(cond['id']), so the flagged 5-student subset is independent across LLM arms.

Usage:
    streamlit run app.py
"""

import os
import re
import json
import time
from datetime import datetime
import random
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from tools import (
    load_db,
    build_grades_lookup,
    get_student_grades,
    recalculate_grade,
    get_assignment_stats,
    filter_students,
    filter_students_by_grade,
    minimum_score_needed,
    list_all_assignments,
    predict_final_grade_for_student,
    suggest_intervention_for_student,
    get_course_syllabus,
    get_class_average,
    simulate_uniform_remaining,
)


# =============================================================================
# CONFIG
# =============================================================================

TRAIN_DB_PATH = "/home/alrabah2/PredAct_bench/results/uiuc/cs_db_train.json"
TEST_SETS_DIR = "/home/alrabah2/PredAct_bench/results/uiuc/test_sets"
GROUND_TRUTH_PATH = "/home/alrabah2/PredAct_bench/results/uiuc/ground_truth_for_cutoff_data.json"
LOGS_OUTPUT_DIR = "/home/alrabah2/PredAct_bench/study_logs"

# Per-LLM provider configuration. All OpenAI-compatible endpoints.
LLM_CONFIGS = {
    "gpt4o_mini": {
        "api_base": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": "gpt-4o-mini",
        "extra_body": {},
    },
    "qwen_9b": {
        "api_base": "https://api.together.xyz/v1",
        "api_key": os.environ.get("TOGETHER_API_KEY", ""),
        "model": "Qwen/Qwen3.5-9B",
        "extra_body": {},
    },
    "qwen_35b": {
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "model": "qwen/qwen3.5-35b-a3b",
        "extra_body": {"reasoning": {"enabled": False}},  # disable thinking for speed
    },
}

N_AT_RISK = 5
SAMPLE_SEED = 42

STUDENT_ID_REGEX = re.compile(r"syn_\d{6}")

# Shared drill-down section for agent scenarios
AGENT_DRILL_DOWN_GUIDE = (
    "3. **Drill down in chat.** Use the chat to investigate each flagged student. "
    "Some things you can ask the agent:\n\n"
    "   - **Grade history:** \"Show me the full grades for the flagged students each with the confidence scores\" "
    "(returns every graded and remaining assignment with weights)\n"
    "   - **Class comparison:** \"What's the class average?\" or \"How does syn_xxxxxx "
    "compare to the class average?\"\n"
    "   - **Assignment stats:** \"What are the class stats for Midterm 1?\"\n"
    "   - **Counterfactuals (uniform):** \"What if syn_xxxxxx scores 80 on all "
    "remaining work?\" — the agent will simulate it.\n"
    "   - **Counterfactuals (mixed):** \"What if they get 70 on the Final and 90 on Lab 10?\"\n"
    "   - **Minimum score needed:** \"What minimum score on remaining work does "
    "syn_xxxxxx need to pass with a C?\"\n"
    "   - **Intervention suggestions:** \"What interventions would you recommend "
    "for syn_xxxxxx?\"\n"
    "   - **Re-prediction:** \"Re-run the prediction for syn_xxxxxx and give me "
    "the confidence.\"\n\n"
    "   Use these to verify whether the agent's flag is real or noise.\n\n"
    "4. **Final decision.** Revise flags as needed and submit.\n\n"
)


def build_agent_intro(course_id, week, pct):
    """Build the standard intro for an agent scenario at given accuracy."""
    return (
        f"This is **week {week}** of **{course_id}**.\n\n"
        f"**An AI agent is available.** The agent's grade-prediction tool is about "
        f"**{pct}% accurate** in this course. Its lookup tools (grades, assignments, "
        f"class stats, counterfactuals) are always correct. Only its *predictions* "
        f"can be wrong.\n\n"
        f"**You will NOT see student data directly.** You can only learn about students "
        f"through the agent.\n\n"
        f"---\n\n"
        f"### Workflow\n\n"
        f"1. **See the agent's flagged students.** Click the button to reveal the "
        f"students the agent flagged as at-risk (predicted D or F). D is 60-69, F is below 60.\n\n"
        f"2. **Initial decision.** For each flagged student, decide Flag or Not Flag "
        f"based only on what the agent reports. You cannot use the chat yet. Submit.\n\n"
        + AGENT_DRILL_DOWN_GUIDE +
        f"---\n\n"
        f"**Definition of at-risk:** The student will finish the course with a D or F.\n\n"
        f"Remember: the agent is about ~{pct}% accurate in this course."
    )


# Courses used in all agent scenarios
AGENT_COURSES = [
    # (course_id, course_file, sample_size, target_accuracy)
    ("CS 374", "CS374_week8.json", 195, 0.40),
    ("CS 461", "CS461_week8.json", 147, 0.60),
    ("CS 421", "CS421_week8.json", 265, 0.80),
]

# LLM block definitions: (prefix for cond_id, LLM_CONFIGS key, block number)
LLM_BLOCKS = [
    ("gpt", "gpt4o_mini", 2),
    ("q9b", "qwen_9b",    3),
    ("q35", "qwen_35b",   4),
]


# =============================================================================
# CONDITIONS
# =============================================================================

# Scenario 1: no-agent baseline (block 1)
CONDITIONS = [
    {
        "id": "no_agent",
        "block": 1,
        "title": "Scenario 1 - No AI Agent",
        "course_id": "CS 225",
        "course_file": "CS225_week8.json",
        "week": 8,
        "feature_set": None,
        "has_agent": False,
        "sample_size": 25,
        "target_accuracy": None,
        "llm": None,
        "intro": (
            "This is **week 8** of **CS 225**. You have a set of students to review.\n\n"
            "**No AI agent is available here.** All student data is shown to you directly "
            "as cards — each with the student's weighted average, submission rate, and a "
            "breakdown of every assignment.\n\n"
            "**Your task:** Click **Flag** on every student you believe will finish the "
            "semester with a **D or F**. Click the button again to unflag. You must flag "
            "at least one student to submit.\n\n"
            "**Definition of at-risk:** The student will finish the course with a D or F."
        ),
    },
]

# Scenarios 2-10: three LLM blocks x three accuracy tiers
_scenario_num = 2
for prefix, llm_key, block_num in LLM_BLOCKS:
    for course_id, course_file, sample_size, target_acc in AGENT_COURSES:
        pct = int(target_acc * 100)
        CONDITIONS.append({
            "id": f"{prefix}_{pct}",
            "block": block_num,
            "title": f"Scenario {_scenario_num} - AI Agent (~{pct}% accurate)",
            "course_id": course_id,
            "course_file": course_file,
            "week": 8,
            "feature_set": "full",
            "has_agent": True,
            "sample_size": sample_size,
            "target_accuracy": target_acc,
            "llm": llm_key,
            "intro": build_agent_intro(course_id, 8, pct),
        })
        _scenario_num += 1


# =============================================================================
# BLOCK QUESTIONNAIRES
# =============================================================================

BLOCK_QUESTIONS = {
    1: [
        "I felt confident in my decisions in Scenario 1 (no AI agent).",
        "Reviewing students without any AI tool was manageable.",
        "I would have wanted an AI assistant during this scenario.",
    ],
    2: [
        "I felt confident in my decisions in Scenario 2 (~40% accurate agent).",
        "I felt confident in my decisions in Scenario 3 (~60% accurate agent).",
        "I felt confident in my decisions in Scenario 4 (~80% accurate agent).",
        "The AI agent in Scenarios 2-4 was a useful collaborator.",
        "I could tell when the agent's predictions were wrong in Scenarios 2-4.",
        "I trusted the agent more as its stated accuracy increased.",
    ],
    3: [
        "I felt confident in my decisions in Scenario 5 (~40% accurate agent).",
        "I felt confident in my decisions in Scenario 6 (~60% accurate agent).",
        "I felt confident in my decisions in Scenario 7 (~80% accurate agent).",
        "The AI agent in Scenarios 5-7 was a useful collaborator.",
        "I could tell when the agent's predictions were wrong in Scenarios 5-7.",
        "I trusted the agent more as its stated accuracy increased.",
    ],
    4: [
        "I felt confident in my decisions in Scenario 8 (~40% accurate agent).",
        "I felt confident in my decisions in Scenario 9 (~60% accurate agent).",
        "I felt confident in my decisions in Scenario 10 (~80% accurate agent).",
        "The AI agent in Scenarios 8-10 was a useful collaborator.",
        "I could tell when the agent's predictions were wrong in Scenarios 8-10.",
        "I trusted the agent more as its stated accuracy increased.",
        "Across the study, the different AI agents felt noticeably different from each other.",
        "I would use an AI agent like this in my own courses.",
    ],
}

BLOCK_TITLES = {
    1: "Block 1 Questionnaire (after Scenario 1)",
    2: "Block 2 Questionnaire (after Scenarios 2-4)",
    3: "Block 3 Questionnaire (after Scenarios 5-7)",
    4: "Final Questionnaire (after Scenarios 8-10)",
}


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def get_db():
    return load_db(TRAIN_DB_PATH)


@st.cache_data
def load_test_set(course_file):
    path = os.path.join(TEST_SETS_DIR, course_file)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_ground_truth():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_students_and_lookup(course_file, sample_size, n_at_risk=N_AT_RISK, seed=SAMPLE_SEED):
    """Stratified sample with fixed size: n_at_risk truly at-risk + (sample_size - n_at_risk) not-at-risk."""
    import random
    test_set = load_test_set(course_file)
    all_students = test_set["students"]

    gt = load_ground_truth()
    test_filename = os.path.basename(course_file)
    gt_entry = gt.get(test_filename, {})
    student_grades = gt_entry.get("student_grades", {})

    at_risk_pool = [s for s in all_students
                    if student_grades.get(s["student_id"], "").lower() in ("d", "f")]
    not_at_risk_pool = [s for s in all_students
                        if student_grades.get(s["student_id"], "").lower() in ("a", "b", "c")]

    rng = random.Random(seed)

    at_risk_sorted = sorted(at_risk_pool, key=lambda s: s["student_id"])
    rng.shuffle(at_risk_sorted)
    sampled_at_risk = at_risk_sorted[:n_at_risk]

    n_not_at_risk = sample_size - n_at_risk
    not_at_risk_sorted = sorted(not_at_risk_pool, key=lambda s: s["student_id"])
    rng.shuffle(not_at_risk_sorted)
    sampled_not_at_risk = not_at_risk_sorted[:n_not_at_risk]

    students = sampled_at_risk + sampled_not_at_risk
    rng.shuffle(students)

    lookup = build_grades_lookup(students)
    return students, lookup


@st.cache_data
def build_student_summary(course_file, sample_size):
    students, lookup = get_students_and_lookup(course_file, sample_size)
    summary = []
    for s in students:
        sid = s["student_id"]
        rec = get_student_grades(sid, lookup)
        if "error" in rec:
            continue
        avg = rec["weighted_average"]
        submitted = sum(1 for a in rec["assignments"] if a["score"] is not None)
        total = len(rec["assignments"])
        summary.append({
            "student_id": sid,
            "weighted_average": avg,
            "submitted": submitted,
            "total": total,
            "missing": total - submitted,
            "assignments": rec["assignments"],
        })
    return summary


# =============================================================================
# SYLLABUS
# =============================================================================

def render_syllabus_inline(course_id):
    db = get_db()
    syl_result = get_course_syllabus(db, course_id)
    syllabus = syl_result.get("assignments", [])

    if not syllabus:
        st.caption("No syllabus available.")
        return

    rows = [
        f"| {a['name']} | W{a['week']} | {a['type']} | {a['weight']:.2%} |"
        for a in syllabus
    ]
    table = "| Assignment | Week | Type | Weight |\n|---|---|---|---|\n" + "\n".join(rows)
    st.markdown(table)


# =============================================================================
# AGENT TOOLS
# =============================================================================

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "list_students",
        "description": "List every student ID in this class. Returns student_ids (the list) and total_count (the number of students). Always use total_count for the count — do not count the list manually.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_student_grades",
        "description": "Look up a student's full grade record (all assignment scores, weights, weeks, weighted average).",
        "parameters": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "predict_final_grade_for_student",
        "description": "Predict this student's final letter grade using the k-NN tool. Returns a confidence score.",
        "parameters": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "predict_all_students",
        "description": "Return predictions with confidence for ALL students, sorted by at-risk (D/F) confidence.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "suggest_intervention_for_student",
        "description": "Return a profile + intervention suggestions for a student.",
        "parameters": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "recalculate_grade",
        "description": "Recalculate a student's weighted average. Supports dropping, overriding existing scores, and simulating future scores via simulate_remaining. For 'what if they score X on ALL remaining assignments', prefer simulate_uniform_remaining instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "drop": {"type": "array", "items": {"type": "string"}},
                "override": {"type": "object", "additionalProperties": {"type": "number"}},
                "simulate_remaining": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Dict mapping assignment name to simulated score. Example: {\"Final Exam\": 85}",
                },
            },
            "required": ["student_id"],
        },
    }},
    {"type": "function", "function": {
        "name": "simulate_uniform_remaining",
        "description": "Counterfactual: assume the student scores a uniform score on EVERY remaining (not-yet-graded) assignment. Returns the new weighted average AND letter grade. Use this any time the instructor asks 'what if they score X on the rest' or similar. This is the ONLY correct way to compute such counterfactuals — do not do the math manually.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "uniform_score": {"type": "number", "description": "Score to apply to every remaining assignment, e.g. 80"},
            },
            "required": ["student_id", "uniform_score"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_assignment_stats",
        "description": "Class-wide stats for an assignment.",
        "parameters": {"type": "object", "properties": {"assignment_name": {"type": "string"}}, "required": ["assignment_name"]},
    }},
    {"type": "function", "function": {
        "name": "list_all_assignments",
        "description": "List every assignment students have attempted so far.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_course_syllabus",
        "description": "Full course syllabus (all weeks), with each assignment marked graded or remaining.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_class_average",
        "description": "Overall class average weighted grade across all students.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "minimum_score_needed",
        "description": "Calculate the minimum uniform score a student needs on ALL remaining assignments to reach a target letter grade (A/B/C/D). Use this instead of brute-forcing simulate_uniform_remaining.",
        "parameters": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "target_grade": {"type": "string", "description": "Target letter grade: A, B, C, or D"},
            },
            "required": ["student_id", "target_grade"],
        },
    }},
]


def predict_all_students_for_scenario(cond, grades_lookup, db, students):
    import random
    results = []
    per_student_cache = {}
    target_acc = cond.get("target_accuracy")
    all_grades = ["a", "b", "c", "d", "f"]

    # Load ground truth to know which predictions are right/wrong
    gt = load_ground_truth()
    gt_grades = gt.get(os.path.basename(cond["course_file"]), {}).get("student_grades", {})

    # Step 1: get real predictions
    raw_preds = []
    for s in students:
        sid = s["student_id"]
        pred = predict_final_grade_for_student(
            db, cond["course_id"], s,
            up_to_week=cond["week"], feature_set=cond["feature_set"],
        )
        if "error" in pred:
            continue
        pred = dict(pred)  # defensive copy — don't mutate any shared dict from tools.py
        truth = gt_grades.get(sid, "").lower()
        is_correct = pred["predicted_grade"].lower() == truth
        raw_preds.append((sid, pred, truth, is_correct))

    # Step 2: apply noise to hit target accuracy
    if target_acc is not None and raw_preds:
        rng = random.Random(hash(cond["id"]) & 0xFFFFFFFF)
        total = len(raw_preds)
        target_correct = round(target_acc * total)
        currently_correct = sum(1 for _, _, _, c in raw_preds if c)

        if currently_correct > target_correct:
            # Too many correct - flip some correct ones to wrong
            correct_preds = [p for p in raw_preds if p[3]]
            rng.shuffle(correct_preds)
            n_to_flip = currently_correct - target_correct
            for sid, pred, truth, _ in correct_preds[:n_to_flip]:
                wrong = [g for g in all_grades if g != truth]
                pred["predicted_grade"] = rng.choice(wrong)
                pred["failure_risk"] = "high" if pred["predicted_grade"] in ("d", "f") else None
        elif currently_correct < target_correct:
            # Too few correct - flip some wrong ones to correct
            wrong_preds = [p for p in raw_preds if not p[3]]
            rng.shuffle(wrong_preds)
            n_to_flip = target_correct - currently_correct
            for sid, pred, truth, _ in wrong_preds[:n_to_flip]:
                if truth:  # only if we have ground truth
                    pred["predicted_grade"] = truth
                    pred["failure_risk"] = "high" if truth in ("d", "f") else None

    # Step 3: build results
    for sid, pred, truth, _ in raw_preds:
        per_student_cache[sid] = pred
        driver = compute_primary_driver(sid, grades_lookup)
        results.append({
            "student_id": sid,
            "predicted_grade": pred["predicted_grade"].upper(),
            "confidence": round(pred["confidence"] * 100, 1),
            "failure_risk": pred["failure_risk"],
            "primary_driver": driver,
        })

    try:
        st.session_state.prediction_cache[cond["id"]] = per_student_cache
    except Exception:
        pass

    at_risk = [r for r in results if r["predicted_grade"] in ("D", "F")]

    # Stratified sample: spread across confidence levels
    at_risk.sort(key=lambda r: r["confidence"])
    n = len(at_risk)
    if n <= 5:
        picked = at_risk
    else:
        indices = [int(i * (n - 1) / 4) for i in range(5)]
        picked = [at_risk[i] for i in indices]
    at_risk = picked
    at_risk.sort(key=lambda r: -r["confidence"])  # display highest first

    acc_pct = f"{int(target_acc * 100)}%" if target_acc is not None else "—"

    return {
        "at_risk_students": at_risk,
        "at_risk_count": len(at_risk),
        "total_students": len(results),
        "note": f"Predictions roughly {acc_pct} accurate in this course.",
    }


def compute_primary_driver(student_id, grades_lookup):
    if student_id not in grades_lookup:
        return "unknown"
    scores = grades_lookup[student_id]
    missing = [(name, info) for name, info in scores.items()
               if info.get("score") is None and info.get("weight", 0) > 0]
    if missing:
        missing.sort(key=lambda x: -x[1]["weight"])
        worst_name, worst_info = missing[0]
        return f"Missing {worst_name} (weight {worst_info['weight']:.0%})"

    damaged = []
    for name, info in scores.items():
        score = info.get("score")
        weight = info.get("weight", 0)
        if score is None or weight <= 0:
            continue
        damage = weight * (100 - score) / 100
        damaged.append((name, score, weight, damage))

    if damaged:
        damaged.sort(key=lambda x: -x[3])
        worst_name, worst_score, worst_weight, _ = damaged[0]
        return f"Low on {worst_name} ({worst_weight:.0%} weight, score {worst_score:.0f})"

    return "insufficient data"


def execute_tool(tool_name, args, cond, grades_lookup, db, students):
    print(f"[TOOL CALL] {tool_name}({args})", flush=True)
    try:
        course_id = cond["course_id"]
        week = cond["week"]
        feature_set = cond["feature_set"]
        cid = cond["id"]

        if tool_name == "list_students":
            sids = sorted(grades_lookup.keys())
            result = {"student_ids": sids, "total_count": len(sids)}
        elif tool_name == "get_student_grades":
            result = get_student_grades(args["student_id"], grades_lookup)
        elif tool_name == "predict_final_grade_for_student":
            sid = args["student_id"]
            cached = st.session_state.prediction_cache.get(cid, {}).get(sid)
            if cached is not None:
                print(f"[CACHE HIT] predict_final_grade_for_student({sid})", flush=True)
                result = cached
            else:
                rec = next((s for s in students if s["student_id"] == sid), None)
                if rec is None:
                    result = {"error": f"Student {sid} not in this class"}
                else:
                    result = predict_final_grade_for_student(
                        db, course_id, rec,
                        up_to_week=week, feature_set=feature_set,
                    )
                    if "error" not in result:
                        st.session_state.prediction_cache.setdefault(cid, {})[sid] = result
        elif tool_name == "predict_all_students":
            cached = st.session_state.all_predictions_cache.get(cid)
            if cached is not None:
                print(f"[CACHE HIT] predict_all_students", flush=True)
                result = cached
            else:
                result = predict_all_students_for_scenario(cond, grades_lookup, db, students)
                st.session_state.all_predictions_cache[cid] = result
        elif tool_name == "suggest_intervention_for_student":
            result = suggest_intervention_for_student(args["student_id"], grades_lookup)
        elif tool_name == "recalculate_grade":
            full_syl = get_course_syllabus(db, course_id, current_week=week)
            full_syllabus = full_syl.get("assignments", [])
            result = recalculate_grade(
                args["student_id"], grades_lookup,
                drop=args.get("drop"), override=args.get("override"),
                simulate_remaining=args.get("simulate_remaining"),
                full_syllabus=full_syllabus,
            )
        elif tool_name == "simulate_uniform_remaining":
            result = simulate_uniform_remaining(
                db, course_id, week,
                args["student_id"], grades_lookup,
                args["uniform_score"],
            )
        elif tool_name == "get_assignment_stats":
            result = get_assignment_stats(args["assignment_name"], grades_lookup)
        elif tool_name == "list_all_assignments":
            result = list_all_assignments(grades_lookup)
        elif tool_name == "get_course_syllabus":
            result = get_course_syllabus(db, course_id, current_week=week)
        elif tool_name == "get_class_average":
            result = get_class_average(grades_lookup)
        elif tool_name == "minimum_score_needed":
            result = minimum_score_needed(
                args["student_id"], args["target_grade"], grades_lookup,
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        result_preview = json.dumps(result, ensure_ascii=False, default=str)[:200]
        print(f"[TOOL RESULT] {tool_name} -> {result_preview}", flush=True)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[TOOL ERROR] {tool_name}: {e}", flush=True)
        return json.dumps({"error": str(e)})


def extract_student_ids_from_text(text):
    if not text:
        return set()
    return set(STUDENT_ID_REGEX.findall(text))


def get_llm_response(messages, cond, grades_lookup, db, students):
    """Dispatch to the correct LLM provider based on cond['llm']."""
    llm_key = cond.get("llm")
    if not llm_key:
        return "[No LLM configured for this scenario.]", set()
    if llm_key not in LLM_CONFIGS:
        return f"[Unknown LLM key: {llm_key}]", set()

    cfg = LLM_CONFIGS[llm_key]
    if not cfg["api_key"]:
        return f"[API key missing for {llm_key}. Check your .env file.]", set()

    target_acc = cond.get("target_accuracy")
    accuracy_str = f"{int(target_acc * 100)}%" if target_acc is not None else "unknown"

    system_msg = {
        "role": "system",
        "content": (
            f"You are an academic advising assistant for {cond['course_id']}, currently at week {cond['week']}.\n\n"
            f"The instructor is reviewing a set of students and has already been shown the list of students "
            f"the prediction tool flagged as at-risk. They are now investigating those flagged students in detail.\n\n"
            f"RULES:\n"
            f"- Always call tools for any question about students, grades, or class data. Never guess.\n"
            f"- When asked about a specific student, ALWAYS include BOTH graded assignments AND remaining "
            f"assignments (with weights and weeks). The instructor needs to reason about counterfactuals.\n"
            f"- For counterfactuals like 'what if they score X on all remaining assignments', ALWAYS use the "
            f"simulate_uniform_remaining tool with student_id and uniform_score. Do NOT do the math manually. "
            f"For mixed-score counterfactuals (e.g. 'what if they get 80 on the midterm and 90 on the final'), "
            f"use recalculate_grade with simulate_remaining.\n"
            f"- For 'what's the minimum score needed to reach a grade' questions, use minimum_score_needed. "
            f"Do NOT brute-force by calling simulate_uniform_remaining repeatedly.\n"
            f"- When asked about class size or how many students, use list_students and read total_count. "
            f"Never count the list manually.\n"
            f"- Be concise and data-driven.\n"
            f"- Predictions in this course are about {accuracy_str} accurate. "
            f"Remind the instructor when appropriate that low-confidence predictions are more likely wrong."
        ),
    }

    client = OpenAI(base_url=cfg["api_base"], api_key=cfg["api_key"])
    full_messages = [system_msg] + messages
    surfaced = set()

    for _ in range(20):
        kwargs = dict(
            model=cfg["model"],
            messages=full_messages,
            tools=TOOL_DEFINITIONS,
            temperature=0.1,
            max_tokens=1400,
        )
        if cfg["extra_body"]:
            kwargs["extra_body"] = cfg["extra_body"]

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            return f"[LLM request failed for {cfg['model']}: {type(e).__name__}: {e}]", surfaced

        msg = response.choices[0].message

        if not msg.tool_calls:
            text = msg.content or "[No response]"
            surfaced |= extract_student_ids_from_text(text)
            return text, surfaced

        full_messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            if "student_id" in args:
                surfaced.add(args["student_id"])
            result = execute_tool(tc.function.name, args, cond, grades_lookup, db, students)
            if tc.function.name == "predict_all_students":
                try:
                    parsed = json.loads(result)
                    for s in parsed.get("at_risk_students", []):
                        surfaced.add(s["student_id"])
                except Exception:
                    pass
            elif tc.function.name in (
                "get_student_grades", "predict_final_grade_for_student",
                "suggest_intervention_for_student", "recalculate_grade",
                "simulate_uniform_remaining", "minimum_score_needed",
            ):
                try:
                    parsed = json.loads(result)
                    surfaced |= extract_student_ids_from_text(json.dumps(parsed))
                except Exception:
                    pass
            full_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "[Tool-call loop exceeded]", surfaced


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        "page": "consent",
        "participant": None,
        "condition_idx": 0,
        "initial_decisions": {},
        "final_decisions": {},
        "agent_flagged_in_scenario": {},
        "agent_flag_details": {},
        "initial_submitted": {},
        "chat_history": {},
        "surfaced": {},
        "start_times": {},
        "done_conditions": [],
        "prediction_cache": {},         # {cond_id: {student_id: raw pred dict}}
        "all_predictions_cache": {},    # {cond_id: full predict_all_students result}
        "block_responses": {},          # {block_num: {q1: 3, q2: 4, ...}}
        "open_feedback": "",
        "pending_block": None,          # block_num awaiting questionnaire
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def save_participant_log():
    os.makedirs(LOGS_OUTPUT_DIR, exist_ok=True)
    gt = load_ground_truth()

    log = {
        "participant": st.session_state.participant,
        "completed_at": datetime.now().isoformat(),
        "conditions": [],
    }
    for cond in CONDITIONS:
        cid = cond["id"]
        initial = st.session_state.initial_decisions.get(cid, {})
        final = st.session_state.final_decisions.get(cid, {})
        agent_flagged = st.session_state.agent_flagged_in_scenario.get(cid, [])
        agent_details = st.session_state.agent_flag_details.get(cid, {})

        test_file = os.path.basename(cond["course_file"])
        gt_grades = gt.get(test_file, {}).get("student_grades", {})

        per_student = {}
        all_sids = set(initial.keys()) | set(final.keys())
        for sid in all_sids:
            truth_grade = gt_grades.get(sid, "").lower()
            per_student[sid] = {
                "agent_flagged": sid in agent_flagged,
                "agent_details": agent_details.get(sid, None),
                "initial_decision": initial.get(sid, None),
                "final_decision": final.get(sid, {}).get("decision", None),
                "ground_truth_grade": truth_grade,
                "is_at_risk": truth_grade in ("d", "f"),
            }

        log["conditions"].append({
            "condition_id": cid,
            "block": cond.get("block"),
            "course_id": cond["course_id"],
            "week": cond["week"],
            "has_agent": cond["has_agent"],
            "feature_set": cond["feature_set"],
            "sample_size": cond.get("sample_size"),
            "llm": cond.get("llm"),
            "target_accuracy": cond.get("target_accuracy"),
            "agent_flagged_students": agent_flagged,
            "per_student": per_student,
            "chat_history": st.session_state.chat_history.get(cid, []),
            "duration_seconds": st.session_state.get(f"{cid}_duration", 0),
        })

    log["block_responses"] = st.session_state.get("block_responses", {})
    log["open_feedback"] = st.session_state.get("open_feedback", "")

    # Stable filename per participant so intermediate saves overwrite cleanly.
    name_clean = st.session_state.participant["name"].replace(" ", "_")
    session_stamp = st.session_state.participant.get("session_stamp")
    if not session_stamp:
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.participant["session_stamp"] = session_stamp
    fname = f"{name_clean}_{session_stamp}.json"
    path = os.path.join(LOGS_OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return path


# =============================================================================
# PAGES
# =============================================================================

def page_consent():
    st.title("PredAct - Human Study")
    st.markdown(f"""
    ### Welcome

    You will complete **10 scenarios** grouped into **4 blocks**. After each block
    you will answer a short questionnaire. Plan for roughly **2.5-3 hours total**;
    feel free to take a short break between blocks.

    In each scenario, your task is to **flag students you believe will finish with
    a D or F**.

    - **Block 1 (Scenario 1):** No AI agent. Student data shown directly.
    - **Block 2 (Scenarios 2-4):** AI agent at 40%, 60%, 80% stated accuracy.
    - **Block 3 (Scenarios 5-7):** A different AI agent at 40%, 60%, 80%.
    - **Block 4 (Scenarios 8-10):** A different AI agent at 40%, 60%, 80%.

    Your chat, decisions, and timing will be logged for research. Your progress is
    saved automatically after every block.
    """)

    st.markdown("---")
    st.markdown("**Name and email are required to proceed.**")
    name = st.text_input("Full Name *")
    email = st.text_input("Email *")
    consent = st.checkbox("I consent to having my interactions logged for research purposes.")

    name_ok = bool(name and name.strip())
    email_ok = bool(email and email.strip() and "@" in email and "." in email.split("@")[-1])
    can_proceed = name_ok and email_ok and consent

    if not name_ok and name:
        st.warning("Please enter your full name.")
    if not email_ok and email:
        st.warning("Please enter a valid email address.")

    if st.button("Start Study", type="primary", disabled=not can_proceed):
        st.session_state.participant = {
            "name": name.strip(), "email": email.strip(),
            "consented_at": datetime.now().isoformat(),
            "session_stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        st.session_state.page = "scenario_intro"
        st.rerun()


def page_scenario_intro():
    idx = st.session_state.condition_idx
    if idx >= len(CONDITIONS):
        # Safety net — normally we route directly to "done" from the last block
        # questionnaire.
        st.session_state.page = "done"
        st.rerun()
        return

    cond = CONDITIONS[idx]
    st.title(cond["title"])
    st.markdown(f"*Scenario {idx + 1} of {len(CONDITIONS)} — Block {cond['block']}*")
    st.markdown("---")
    st.markdown(cond["intro"])
    st.markdown("---")

    if st.button("Begin Scenario", type="primary"):
        cid = cond["id"]
        st.session_state.initial_decisions.setdefault(cid, {})
        st.session_state.final_decisions.setdefault(cid, {})
        st.session_state.agent_flagged_in_scenario.setdefault(cid, [])
        st.session_state.agent_flag_details.setdefault(cid, {})
        st.session_state.initial_submitted.setdefault(cid, False)
        st.session_state.chat_history.setdefault(cid, [])
        st.session_state.surfaced.setdefault(cid, set())
        st.session_state.start_times[cid] = time.time()
        st.session_state.page = "scenario_work"
        st.rerun()


# -----------------------------------------------------------------------------
# NO-AGENT CONDITION
# -----------------------------------------------------------------------------

def render_no_agent_cards(cond):
    cid = cond["id"]
    final_decisions = st.session_state.final_decisions[cid]
    summary = build_student_summary(cond["course_file"], cond["sample_size"])

    st.markdown(f"### Students ({len(summary)})")
    cols_per_row = 4
    for i in range(0, len(summary), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(summary):
                continue
            s = summary[i + j]
            with col:
                render_no_agent_card(s, cid, final_decisions)


def render_no_agent_card(student_info, cond_id, final_decisions):
    sid = student_info["student_id"]
    current = final_decisions.get(sid, {})
    is_flagged = current.get("decision") == "accept"

    border = "#d62728" if is_flagged else "#cccccc"
    bg = "#fff5f5" if is_flagged else "#fafafa"
    status = "FLAGGED" if is_flagged else "NOT FLAGGED"
    status_color = "#d62728" if is_flagged else "#888"

    avg = student_info.get('weighted_average')
    avg_str = f"{avg:.1f}" if avg is not None else "-"
    submitted = student_info.get('submitted', 0)
    total = student_info.get('total', 0)

    st.markdown(
        f"""
        <div style="border: 2px solid {border}; background: {bg};
                    border-radius: 8px; padding: 14px; margin-bottom: 6px;">
            <div style='font-weight:700; font-size:16px;'>{sid}</div>
            <div style='font-size:14px; color:#555; margin-top:6px;'>Avg: <b>{avg_str}</b> | Submitted {submitted}/{total}</div>
            <div style='font-size:13px; color:{status_color}; font-weight:700; margin-top:8px;'>{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Assignments"):
        for a in student_info.get("assignments", []):
            sc = a["score"]
            sc_str = f"{sc:.1f}" if sc is not None else "-"
            st.markdown(f"- **{a['name']}** (W{a['week']}, {a['type']}, {a['weight']:.2f}): `{sc_str}`")

    btn_label = "Unflag" if is_flagged else "Flag"
    btn_type = "primary" if is_flagged else "secondary"
    if st.button(btn_label, key=f"flag_{cond_id}_{sid}", type=btn_type, use_container_width=True):
        if is_flagged:
            del final_decisions[sid]
        else:
            final_decisions[sid] = {"decision": "accept"}
        st.rerun()


# -----------------------------------------------------------------------------
# AGENT CONDITION STAGES
# -----------------------------------------------------------------------------

def render_reveal_stage(cond):
    cid = cond["id"]
    st.markdown("### Ready to begin")
    st.markdown(
        "Click below to see which students the AI agent has flagged as at-risk. "
        "You will then make an **initial decision** for each flagged student."
    )
    if st.button("Show me the agent's flagged students", type="primary"):
        db = get_db()
        students, lookup = get_students_and_lookup(cond["course_file"], cond["sample_size"])
        result = predict_all_students_for_scenario(cond, lookup, db, students)
        print(f"[REVEAL] at-risk count: {len(result.get('at_risk_students', []))}", flush=True)
        print(f"[REVEAL] total: {result.get('total_students')}", flush=True)

        st.session_state.all_predictions_cache[cid] = result

        agent_flagged = [s["student_id"] for s in result.get("at_risk_students", [])]
        agent_details = {
            s["student_id"]: {
                "predicted_grade": s["predicted_grade"],
                "confidence": s["confidence"],
                "primary_driver": s["primary_driver"],
            }
            for s in result.get("at_risk_students", [])
        }
        st.session_state.agent_flagged_in_scenario[cid] = agent_flagged
        st.session_state.agent_flag_details[cid] = agent_details
        st.rerun()


def render_initial_decision_stage(cond):
    cid = cond["id"]
    agent_flagged = st.session_state.agent_flagged_in_scenario[cid]
    agent_details = st.session_state.agent_flag_details[cid]
    initial = st.session_state.initial_decisions[cid]

    st.markdown("### Initial Decision")
    st.markdown(
        f"The agent has flagged **{len(agent_flagged)} students** as at-risk. "
        "Without using the chat, click **Flag** on the ones you agree with. "
        "Leave the rest unflagged. Click a flagged card again to unflag it."
    )

    if not agent_flagged:
        st.warning("The agent did not flag any students. You can skip to chat.")
        if st.button("Open Chat", type="primary"):
            st.session_state.initial_submitted[cid] = True
            st.rerun()
        return

    cols_per_row = 3
    for i in range(0, len(agent_flagged), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(agent_flagged):
                continue
            sid = agent_flagged[i + j]
            details = agent_details.get(sid, {})
            with col:
                render_initial_card(sid, details, cid, initial)

    # Default un-decided students to "reject" so they have a stored value
    for sid in agent_flagged:
        if sid not in initial:
            initial[sid] = "reject"

    st.markdown("---")
    n_flagged_by_user = sum(1 for v in initial.values() if v == "accept")
    st.markdown(f"**Flagged by you:** {n_flagged_by_user} / {len(agent_flagged)}")

    if st.button("Submit Initial Decisions & Open Chat", type="primary"):
        st.session_state.initial_submitted[cid] = True
        for sid, dec in initial.items():
            if sid not in st.session_state.final_decisions[cid]:
                st.session_state.final_decisions[cid][sid] = {"decision": dec}
        st.rerun()


def render_initial_card(sid, details, cond_id, initial):
    current = initial.get(sid)
    is_flag = current == "accept"

    border = "#d62728" if is_flag else "#cccccc"
    bg = "#fff5f5" if is_flag else "#fafafa"

    pred = details.get("predicted_grade", "?")
    conf = details.get("confidence", 0)
    driver = details.get("primary_driver", "")

    st.markdown(
        f"""
        <div style="border: 2px solid {border}; background: {bg};
                    border-radius: 8px; padding: 12px; margin-bottom: 6px;">
            <div style='font-weight:700; font-size:15px;'>{sid}</div>
            <div style='font-size:13px; color:#555; margin-top:4px;'>
                Agent: <b>{pred}</b> ({conf:.0f}% conf)
            </div>
            <div style='font-size:12px; color:#777; margin-top:3px;'>{driver}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_label = "Unflag" if is_flag else "Flag"
    btn_type = "primary" if is_flag else "secondary"
    if st.button(btn_label, key=f"init_{cond_id}_{sid}",
                 type=btn_type, use_container_width=True):
        if is_flag:
            initial[sid] = "reject"
        else:
            initial[sid] = "accept"
        st.rerun()


def render_chat_and_final_stage(cond):
    cid = cond["id"]
    db = get_db()
    students, lookup = get_students_and_lookup(cond["course_file"], cond["sample_size"])
    agent_flagged = st.session_state.agent_flagged_in_scenario[cid]
    agent_details = st.session_state.agent_flag_details[cid]
    initial = st.session_state.initial_decisions[cid]
    final = st.session_state.final_decisions[cid]
    history = st.session_state.chat_history[cid]

    chat_col, panel_col = st.columns([7, 3])

    with chat_col:
        st.markdown("### Chat with AI Agent")
        chat_container = st.container(height=600)
        with chat_container:
            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
        if prompt := st.chat_input("Ask the agent..."):
            history.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                resp, _ = get_llm_response(history, cond, lookup, db, students)
            history.append({"role": "assistant", "content": resp})
            st.rerun()

    with panel_col:
        n_flagged = sum(1 for d in final.values() if d.get("decision") == "accept")
        st.markdown(f"### Flagged ({n_flagged})")
        st.caption(f"Total agent-flagged: {len(agent_flagged)}")

        if not agent_flagged:
            st.info("No students were flagged by the agent.")
            return

        for sid in agent_flagged:
            details = agent_details.get(sid, {})
            current = final.get(sid, {})
            is_flagged = current.get("decision") == "accept"
            initial_dec = initial.get(sid)

            dot = "🔴" if is_flagged else "⚪"
            initial_label = "🟥" if initial_dec == "accept" else "⬜"

            with st.container(border=True):
                st.markdown(
                    f"**{dot} {sid}**  \n"
                    f"<span style='font-size:11px; color:#777;'>"
                    f"Agent: {details.get('predicted_grade','?')} ({details.get('confidence',0):.0f}%) | "
                    f"Initial: {initial_label}</span>",
                    unsafe_allow_html=True,
                )

                btn_label = "Unflag" if is_flagged else "Flag"
                btn_type = "primary" if is_flagged else "secondary"
                if st.button(btn_label, key=f"final_{cid}_{sid}",
                             type=btn_type, use_container_width=True):
                    if is_flagged:
                        final[sid] = {"decision": "reject"}
                    else:
                        final[sid] = {"decision": "accept"}
                    st.rerun()


# -----------------------------------------------------------------------------
# SCENARIO WORK PAGE
# -----------------------------------------------------------------------------

def page_scenario_work():
    idx = st.session_state.condition_idx
    cond = CONDITIONS[idx]
    cid = cond["id"]

    st.title(cond["title"])
    st.caption(f"Scenario {idx + 1} of {len(CONDITIONS)} — {cond['course_id']}, week {cond['week']}")

    flagged_count = sum(
        1 for d in st.session_state.final_decisions.get(cid, {}).values()
        if d.get("decision") == "accept"
    )
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown(f"**Flagged so far:** {flagged_count} students")
    with top_right:
        can_submit = True
        if cond["has_agent"] and not st.session_state.initial_submitted.get(cid, False):
            can_submit = False

        if st.button("Submit & Continue", type="primary", disabled=not can_submit):
            final = st.session_state.final_decisions.get(cid, {})
            flagged = [sid for sid, d in final.items() if d.get("decision") == "accept"]

            if len(flagged) < 1:
                st.error("You must flag at least one student before submitting.")
            else:
                duration = time.time() - st.session_state.start_times[cid]
                st.session_state[f"{cid}_duration"] = round(duration)
                st.session_state.done_conditions.append(cid)

                current_block = cond["block"]
                next_idx = idx + 1
                next_block = CONDITIONS[next_idx]["block"] if next_idx < len(CONDITIONS) else None

                st.session_state.condition_idx = next_idx

                if next_block != current_block:
                    # Block boundary (or last scenario) -> questionnaire
                    st.session_state.pending_block = current_block
                    st.session_state.page = "block_questionnaire"
                else:
                    st.session_state.page = "scenario_intro"
                st.rerun()

    st.markdown("---")

    intro_col, syl_col = st.columns([1, 1])
    with intro_col:
        st.markdown("#### Scenario Instructions")
        with st.container(border=True):
            st.markdown(cond["intro"])
    with syl_col:
        st.markdown("#### Course Syllabus")
        with st.container(border=True):
            render_syllabus_inline(cond["course_id"])

    st.markdown("---")

    if not cond["has_agent"]:
        render_no_agent_cards(cond)
    else:
        if not st.session_state.agent_flagged_in_scenario.get(cid):
            render_reveal_stage(cond)
        elif not st.session_state.initial_submitted.get(cid, False):
            render_initial_decision_stage(cond)
        else:
            render_chat_and_final_stage(cond)


# -----------------------------------------------------------------------------
# BLOCK QUESTIONNAIRE
# -----------------------------------------------------------------------------

def page_block_questionnaire():
    block_num = st.session_state.get("pending_block")
    if block_num is None:
        st.session_state.page = "scenario_intro"
        st.rerun()
        return

    is_last_block = (block_num == 4)

    st.title(BLOCK_TITLES.get(block_num, f"Block {block_num} Questionnaire"))
    if is_last_block:
        st.markdown(f"Thank you, **{st.session_state.participant['name']}**. Almost done!")
    else:
        st.markdown("A few quick questions before you continue to the next block.")
    st.markdown("---")

    st.markdown("**Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree):**")
    questions = BLOCK_QUESTIONS.get(block_num, [])
    labels = ["1\nStrongly\nDisagree", "2", "3\nNeutral", "4", "5\nStrongly\nAgree"]

    # Pre-load any prior answers for this block
    prior = st.session_state.block_responses.get(block_num, {})
    responses = {}
    for i, q in enumerate(questions):
        st.markdown(f"**{q}**")
        default_idx = prior.get(f"q{i+1}", 3) - 1  # Likert 1-5 -> index 0-4
        choice = st.radio(
            label=q, options=[1, 2, 3, 4, 5],
            format_func=lambda x: labels[x - 1],
            horizontal=True,
            index=default_idx,
            key=f"lk_b{block_num}_{i}",
            label_visibility="collapsed",
        )
        responses[f"q{i+1}"] = choice
        st.markdown("")

    feedback_text = ""
    if is_last_block:
        st.markdown("---")
        feedback_text = st.text_area(
            "Any comments or feedback (optional)?",
            value=st.session_state.get("open_feedback", ""),
            height=150,
        )

    st.markdown("---")
    btn_label = "Submit & Finish" if is_last_block else "Submit & Continue to Next Block"
    if st.button(btn_label, type="primary"):
        st.session_state.block_responses[block_num] = responses
        if is_last_block:
            st.session_state.open_feedback = feedback_text

        # Auto-save log after every block (fail-safe for long sessions)
        try:
            path = save_participant_log()
            st.session_state.saved_path = path
            print(f"[LOG] Block {block_num} complete. Saved to {path}", flush=True)
        except Exception as e:
            print(f"[LOG ERROR] {e}", flush=True)

        st.session_state.pending_block = None

        if st.session_state.condition_idx >= len(CONDITIONS):
            st.session_state.page = "done"
        else:
            st.session_state.page = "scenario_intro"
        st.rerun()


# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------

def page_done():
    st.title("Thank you!")
    st.balloons()
    st.markdown(f"Saved to `{st.session_state.get('saved_path', '')}`.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(page_title="PredAct - Human Study", layout="wide")
    init_state()

    page = st.session_state.page
    if page == "consent":
        page_consent()
    elif page == "scenario_intro":
        page_scenario_intro()
    elif page == "scenario_work":
        page_scenario_work()
    elif page == "block_questionnaire":
        page_block_questionnaire()
    elif page == "done":
        page_done()
    else:
        st.session_state.page = "consent"
        st.rerun()


if __name__ == "__main__":
    main()