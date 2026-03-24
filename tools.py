"""
PredAct Benchmark - Tools
Deterministic computation logic that Agent 2 calls.
No LLM involved — pure data operations against cs_db.json.
"""

import json
from collections import Counter
from config import (
    CS_DB_PATH,
    MATCH_TOLERANCE,
    MATCH_TOLERANCE_STEP,
    MATCH_TOLERANCE_MAX,
    MIN_MATCHES,
    RISK_MAPPING,
    GRADE_TO_GPA,
    DEFAULT_INTERVENTION,
    DEFAULT_INTERVENTION_GOAL,
    DEFAULT_CONTACT_MODE,
)


# =============================================================================
# MAPPINGS (raw data values → ontology values)
# =============================================================================

# Course ID → ontology course_name
COURSE_NAME_MAP = {
    "CS 100": "intro_programming",
    "CS 101": "intro_programming",
    "CS 105": "intro_programming",
    "CS 124": "intro_programming",
    "CS 125": "intro_programming",
    "CS 128": "intro_programming",
    "CS 173": "discrete_math",
    "CS 225": "data_structures",
    "CS 233": "other",
    "CS 241": "other",
    "CS 340": "other",
    "CS 341": "other",
    "CS 357": "other",
    "CS 374": "other",
    "CS 410": "other",
    "CS 411": "other",
    "CS 421": "other",
    "CS 425": "other",
    "CS 440": "other",
    "CS 450": "other",
    "CS 461": "other",
    "CS 475": "other",
    "CS 498": "other",
    "CS 512": "other",
    "CS 519": "other",
    "MATH 220": "calculus_i",
    "MATH 221": "calculus_ii",
    "MATH 231": "calculus_ii",
    "MATH 241": "calculus_ii",
    "MATH 257": "linear_algebra",
    "MATH 415": "linear_algebra",
    "MATH 416": "linear_algebra",
    "STAT 100": "statistics",
    "STAT 200": "statistics",
    "STAT 400": "statistics",
    "CHEM 102": "general_chemistry",
    "CHEM 104": "general_chemistry",
    "PHYS 211": "general_physics",
    "PHYS 212": "general_physics",
    "RHET 105": "academic_writing",
}

# Assignment type abbreviations → ontology values
ASSIGNMENT_TYPE_MAP = {
    "hw": "homework",
    "homework": "homework",
    "quiz": "quiz",
    "exam": "midterm",
    "midterm": "midterm",
    "final": "final",
    "project": "project",
    "mp": "project",
    "lab": "lab",
    "discussion": "participation",
    "lecture": "participation",
    "attendance": "participation",
    "essay": "essay",
    "presentation": "presentation",
    "participation": "participation",
    "other": "unknown",
}

# Class-level grade trends (does NOT include "fluctuating" — that's student-level only)
CLASS_TREND_MAP = {
    "improving": "improving",
    "stable": "stable",
    "declining": "declining",
    "fluctuating": "polarized",  # class-level equivalent
    "unknown": "unknown",
}


def map_course_name(course_id):
    """Map raw course_id to ontology course_name value."""
    return COURSE_NAME_MAP.get(course_id, "other")


def map_assignment_type(raw_type):
    """Map raw assignment type to ontology value."""
    if raw_type is None:
        return "unknown"
    return ASSIGNMENT_TYPE_MAP.get(raw_type.lower().strip(), "unknown")


def map_class_trend(trend):
    """Map a computed trend to class-level ontology value."""
    return CLASS_TREND_MAP.get(trend, "unknown")


# =============================================================================
# LOAD DATABASE
# =============================================================================

def load_db(path=None):
    """Load cs_db.json into memory."""
    path = path or CS_DB_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# COURSE LOOKUP
# =============================================================================

def lookup_course(db, course_id):
    """
    Find a course in the database by course_id.
    Returns (course_info, intervention, students) or (None, None, None).
    """
    for course in db:
        if course["course_id"] == course_id:
            return (
                course.get("course_info", {}),
                course.get("intervention", None),
                course.get("students", []),
            )
    return None, None, None


# =============================================================================
# SYLLABUS EXTRACTION
# =============================================================================

def get_syllabus(historical_students, up_to_week=None):
    """
    From historical students, extract all unique (name, type, weight) components.
    If up_to_week is specified, only include components up to that week.
    Returns a list of dicts: [{"name": ..., "type": ..., "weight": ..., "week": ...}]
    """
    components = {}
    for student in historical_students:
        for week_data in student.get("weeks", []):
            week_num = week_data["week"]
            if up_to_week and week_num > up_to_week:
                continue
            for activity in week_data.get("activities", []):
                key = activity["name"]
                if key not in components:
                    components[key] = {
                        "name": activity["name"],
                        "type": map_assignment_type(activity.get("type", "unknown")),
                        "weight": activity.get("weight", 0.0),
                        "week": week_num,
                    }
    return list(components.values())


# =============================================================================
# STUDENT SCORE EXTRACTION
# =============================================================================

def extract_scores(student_record):
    """
    From a student record (unseen or historical), extract a dict of
    {component_name: score} and total weight covered.
    """
    scores = {}
    total_weight = 0.0
    max_week = 0
    for week_data in student_record.get("weeks", []):
        week_num = week_data["week"]
        if week_num > max_week:
            max_week = week_num
        for activity in week_data.get("activities", []):
            name = activity["name"]
            scores[name] = activity.get("score", None)
            total_weight += activity.get("weight", 0.0)
    return scores, total_weight, max_week


# =============================================================================
# NEAREST NEIGHBOR MATCHING
# =============================================================================

def match_students(unseen_scores, historical_students, tolerance=None):
    """
    For an unseen student's scores, find historical students who scored
    within ±tolerance on the same components.

    Strategy: keep tolerance tight, drop components if needed.
    1. Try matching ALL components at ±tolerance
    2. Not enough matches? Drop the component with highest score variance
       (least discriminating) and try again
    3. Keep dropping until we find enough matches
    4. Minimum 2 components required — never drop below that

    Args:
        unseen_scores: dict {component_name: score}
        historical_students: list of student records from cs_db.json
        tolerance: percentage points (default from config)

    Returns:
        list of matched historical student records
    """
    tolerance = tolerance or MATCH_TOLERANCE

    # Get valid components (non-None scores)
    valid_components = [
        (name, score) for name, score in unseen_scores.items()
        if score is not None
    ]

    if not valid_components:
        return []

    # Sort components by score ascending — low scores are most predictive of risk
    # When we need to drop components, we drop from the high end first
    sorted_components = sorted(valid_components, key=lambda x: x[1])

    # Try matching with progressively fewer components
    # Start with all, minimum 2
    min_components = min(2, len(sorted_components))

    for num_components in range(len(sorted_components), min_components - 1, -1):
        # Use the num_components lowest-scoring components (most predictive)
        use_components = sorted_components[:num_components]
        component_dict = dict(use_components)

        matches = _find_matches(component_dict, historical_students, tolerance)

        if len(matches) >= MIN_MATCHES:
            return matches

    # Last resort: widen tolerance slightly with minimum components
    use_components = sorted_components[:min_components]
    component_dict = dict(use_components)
    current_tolerance = tolerance + MATCH_TOLERANCE_STEP

    while current_tolerance <= MATCH_TOLERANCE_MAX:
        matches = _find_matches(component_dict, historical_students, current_tolerance)
        if len(matches) >= MIN_MATCHES:
            return matches
        current_tolerance += MATCH_TOLERANCE_STEP

    # Return whatever we have
    return matches


def _find_matches(component_scores, historical_students, tolerance):
    """
    Find historical students who scored within ±tolerance on ALL
    specified components.
    """
    matches = []
    for hist_student in historical_students:
        hist_scores, _, _ = extract_scores(hist_student)

        all_match = True
        for component, unseen_score in component_scores.items():
            hist_score = hist_scores.get(component)
            if hist_score is None:
                all_match = False
                break
            if abs(hist_score - unseen_score) > tolerance:
                all_match = False
                break

        if all_match:
            matches.append(hist_student)

    return matches


# =============================================================================
# GRADE PREDICTION
# =============================================================================

def predict_grade(matched_students):
    """
    From matched historical students, predict grade by majority vote.
    Returns (predicted_grade, confidence, grade_distribution).
    """
    if not matched_students:
        return "unknown", 0.0, {}

    grades = [s.get("final_grade", "unknown") for s in matched_students]
    distribution = dict(Counter(grades))
    total = len(grades)

    # Majority vote
    predicted = max(distribution, key=distribution.get)
    confidence = distribution[predicted] / total

    return predicted.lower(), confidence, distribution


# =============================================================================
# MISSING ASSIGNMENTS
# =============================================================================

def count_missing(unseen_scores, syllabus_components):
    """
    Compare unseen student's submitted components against the expected
    syllabus components. Returns count of missing and list of missing names.
    """
    expected = {c["name"] for c in syllabus_components}
    submitted = set(unseen_scores.keys())
    missing = expected - submitted
    return len(missing), sorted(list(missing))


# =============================================================================
# GRADE TREND
# =============================================================================

def compute_grade_trend(student_record):
    """
    Look at scores across weeks and determine trend.
    Returns one of: improving, stable, declining, fluctuating, unknown.
    NOTE: "fluctuating" is valid for student-level only.
    """
    scores_by_week = []
    for week_data in sorted(student_record.get("weeks", []), key=lambda w: w["week"]):
        week_scores = [
            a.get("score", 0) for a in week_data.get("activities", [])
            if a.get("score") is not None
        ]
        if week_scores:
            scores_by_week.append(sum(week_scores) / len(week_scores))

    if len(scores_by_week) < 2:
        return "unknown"

    # Simple linear trend
    diffs = [scores_by_week[i+1] - scores_by_week[i] for i in range(len(scores_by_week) - 1)]
    avg_diff = sum(diffs) / len(diffs)

    # Check for fluctuation
    pos = sum(1 for d in diffs if d > 2)
    neg = sum(1 for d in diffs if d < -2)
    if pos > 0 and neg > 0 and abs(pos - neg) <= 1:
        return "fluctuating"

    if avg_diff > 2:
        return "improving"
    elif avg_diff < -2:
        return "declining"
    else:
        return "stable"


# =============================================================================
# INTERVENTION CHECK (OR-GATE)
# =============================================================================

def check_intervention(current_week, components_submitted, intervention_data):
    """
    OR-gate logic:
    - current_week >= atrisk_approx_week → intervene
    - components_submitted >= total_components → intervene

    Returns (should_intervene: bool, reason: str).
    """
    if intervention_data is None:
        return False, "no_intervention_data"

    approx_week = intervention_data.get("atrisk_approx_week")
    total_components = intervention_data.get("total_components")

    if approx_week is None and total_components is None:
        return False, "no_intervention_data"

    week_triggered = approx_week is not None and current_week >= approx_week
    component_triggered = total_components is not None and components_submitted >= total_components

    if week_triggered and component_triggered:
        return True, "both_week_and_components"
    elif week_triggered:
        return True, "past_intervention_week"
    elif component_triggered:
        return True, "sufficient_components"
    else:
        return False, "below_threshold"


# =============================================================================
# RISK MAPPING
# =============================================================================

def map_risk(predicted_grade):
    """Map predicted grade to failure_risk level."""
    return RISK_MAPPING.get(predicted_grade.lower(), None)


# =============================================================================
# DETERMINE FAILURE REASON
# =============================================================================

def determine_failure_reason(unseen_scores, missing_count, grade_trend):
    """
    Determine the primary reason for failure risk based on available signals.
    Returns a value from the ontology's failure_risk_reason list.
    """
    # If significant missing work, that's the primary reason
    if missing_count >= 2:
        return "missing_work"

    # If scores are consistently low across submitted work
    submitted_scores = [s for s in unseen_scores.values() if s is not None]
    if submitted_scores:
        avg = sum(submitted_scores) / len(submitted_scores)
        if avg < 60:
            return "academic_underpreparedness"

    # If declining trend
    if grade_trend == "declining":
        return "low_engagement"

    return "unclear"


# =============================================================================
# FULL PIPELINE - PROCESS ALL UNSEEN STUDENTS
# =============================================================================

def process_students(db, course_id, unseen_students):
    """
    Run the full pipeline for a batch of unseen students.

    Args:
        db: loaded cs_db.json
        course_id: the course to match against
        unseen_students: list of unseen student records

    Returns:
        dict with all results structured for belief state filling.
    """
    # Step 1: Course lookup
    course_info, intervention_data, historical_students = lookup_course(db, course_id)
    if course_info is None:
        return {"error": f"Course {course_id} not found in database"}

    # Step 2: Get syllabus from historical data
    # We need to figure out current_week from unseen data
    all_max_weeks = []
    for student in unseen_students:
        _, _, max_week = extract_scores(student)
        all_max_weeks.append(max_week)
    current_week = max(all_max_weeks) if all_max_weeks else 0

    syllabus = get_syllabus(historical_students, up_to_week=current_week)

    # Step 3: Process each student
    student_results = []
    for student in unseen_students:
        sid = student.get("student_id", "unknown")
        scores, weight_covered, max_week = extract_scores(student)

        # Match
        matches = match_students(scores, historical_students)

        # Predict
        predicted_grade, confidence, distribution = predict_grade(matches)

        # Risk
        risk = map_risk(predicted_grade)

        # Missing assignments
        missing_count, missing_names = count_missing(scores, syllabus)

        # Trend
        trend = compute_grade_trend(student)

        # Failure reason
        failure_reason = determine_failure_reason(scores, missing_count, trend)

        student_results.append({
            "student_id": sid,
            "predicted_grade": predicted_grade,
            "confidence": confidence,
            "match_count": len(matches),
            "grade_distribution": distribution,
            "failure_risk": risk,
            "failure_risk_reason": failure_reason if risk else "none",
            "missing_assignments_count": missing_count,
            "missing_assignments": missing_names,
            "grade_trend": trend,
            "weight_covered": round(weight_covered, 4),
        })

    # Step 4: Intervention check
    components_submitted = len(syllabus)  # expected components at this week
    should_intervene, intervention_reason = check_intervention(
        current_week, components_submitted, intervention_data
    )

    # Step 5: Group by risk level
    risk_groups = {}
    for result in student_results:
        risk = result["failure_risk"]
        if risk is None:
            risk_key = "no_risk"
        else:
            risk_key = f"{risk}_risk"

        if risk_key not in risk_groups:
            risk_groups[risk_key] = {
                "student_ids": [],
                "count": 0,
                "predicted_grade": result["predicted_grade"],
                "failure_risk": result["failure_risk"],
                "failure_risk_reasons": {},
                "missing_assignments": {},
                "grade_trends": {},
            }

        group = risk_groups[risk_key]
        group["student_ids"].append(result["student_id"])
        group["count"] += 1
        group["failure_risk_reasons"][result["student_id"]] = result["failure_risk_reason"]
        group["missing_assignments"][result["student_id"]] = result["missing_assignments_count"]
        group["grade_trends"][result["student_id"]] = result["grade_trend"]

    # Step 6: Compute class-level summary
    all_predicted = [r["predicted_grade"] for r in student_results]
    gpas = [GRADE_TO_GPA.get(g, 0.0) for g in all_predicted]
    avg_gpa = round(sum(gpas) / len(gpas), 2) if gpas else 0.0

    flagged = [r for r in student_results if r["failure_risk"] is not None]
    flagged_count = len(flagged)

    # Determine overall class trend — map to class-level ontology values
    all_trends = [r["grade_trend"] for r in student_results]
    trend_counts = Counter(all_trends)
    raw_trend = trend_counts.most_common(1)[0][0] if trend_counts else "unknown"
    overall_trend = map_class_trend(raw_trend)

    # Most common issue type among flagged students — use ontology values
    issue_types = []
    for r in flagged:
        if r["missing_assignments_count"] > 0:
            # Look at what types of assignments are missing
            for name in r["missing_assignments"]:
                for comp in syllabus:
                    if comp["name"] == name:
                        issue_types.append(comp["type"])  # already mapped in get_syllabus
        else:
            # Fall back to type of assignments with lowest scores
            unseen_student = next(
                (s for s in unseen_students if s.get("student_id") == r["student_id"]),
                None,
            )
            if unseen_student:
                scores, _, _ = extract_scores(unseen_student)
                for comp in syllabus:
                    if comp["name"] in scores and scores[comp["name"]] is not None:
                        if scores[comp["name"]] < 70:
                            issue_types.append(comp["type"])  # already mapped

    issue_counter = Counter(issue_types)
    common_issue = issue_counter.most_common(1)[0][0] if issue_counter else "none"

    # Step 7: Build intervention plan per risk group
    intervention_plan = {}
    if should_intervene:
        for risk_key, group in risk_groups.items():
            if group["failure_risk"] is None:
                continue
            risk_level = group["failure_risk"]

            # Per-student intervention type and goal
            per_student_type = {}
            per_student_goal = {}
            for sid in group["student_ids"]:
                reason = group["failure_risk_reasons"][sid]
                if reason == "missing_work":
                    per_student_type[sid] = "study_plan"
                    per_student_goal[sid] = "recover_missing_work"
                elif reason == "academic_underpreparedness":
                    per_student_type[sid] = "tutoring_referral"
                    per_student_goal[sid] = "improve_concept_mastery"
                elif reason == "low_engagement":
                    per_student_type[sid] = "check_in_message"
                    per_student_goal[sid] = "improve_engagement"
                else:
                    per_student_type[sid] = DEFAULT_INTERVENTION.get(risk_level, "monitor_only")
                    per_student_goal[sid] = DEFAULT_INTERVENTION_GOAL.get(risk_level, "reduce_failure_risk")

            intervention_plan[risk_key] = {
                "target_scope": "flagged_students",
                "student_ids": group["student_ids"],
                "intervention_type": per_student_type,
                "intervention_goal": per_student_goal,
                "priority": risk_level,
                "contact_mode": DEFAULT_CONTACT_MODE.get(risk_level, "email"),
                "follow_up_needed": "yes",
            }

    return {
        "course_id": course_id,
        "course_name": map_course_name(course_id),
        "course_info": course_info,
        "intervention_data": intervention_data,
        "current_week": current_week,
        "should_intervene": should_intervene,
        "intervention_reason": intervention_reason,
        "class_summary": {
            "average_gpa": avg_gpa,
            "grade_trend": overall_trend,
            "common_assignment_type_issue": common_issue,
            "flagged_student_count": flagged_count,
            "summary_scope": "whole_class",
        },
        "student_results": student_results,
        "risk_groups": risk_groups,
        "intervention_plan": intervention_plan,
    }