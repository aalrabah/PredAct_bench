"""
PredAct Benchmark - Tools
Deterministic computation logic that Agent 2 calls.
No LLM involved — pure data operations against cs_db.json.

V7 — Base: V6
     Change: OULAD uses UK threshold prediction as primary method.
             weighted_score → UK thresholds (70/60/50/40) → A/B/C/D/F
             Fallback to V6 k-NN only when no graded work available.
     Safety: Primary dataset path identical to V6. OULAD detected via department=="oulad".
     Result: 71.9% → 86.4% on 20 dialogues (+14.5%).

V8 — Base: V7
     Added for human study (per-student, on-demand tools):
       - predict_final_grade_for_student(student_id, course_id, up_to_week, feature_set)
           Per-student k-NN prediction. Supports feature sets "minimal" and "full"
           which naturally land around 60% and 85% accuracy depending on week.
       - suggest_intervention_for_student(student_id, grades_lookup)
           Returns a dict with student profile + a prompt string. Caller handles
           the actual LLM call. No ML training, no fake probabilities.
     Everything else is unchanged.
"""

import json
import math
from collections import Counter
from config import (
    CS_DB_PATH,
    MATCH_TOLERANCE,
    MIN_MATCHES,
    RISK_MAPPING,
    GRADE_TO_GPA,
    DEFAULT_INTERVENTION,
    DEFAULT_INTERVENTION_GOAL,
    DEFAULT_CONTACT_MODE,
)


MIN_SHARED_FRACTION = 0.35
MIN_COMPONENTS_FOR_MATCH = 3

SCORE_GRADE_THRESHOLDS = [
    (90, "a"),
    (80, "b"),
    (70, "c"),
    (60, "d"),
    (0,  "f"),
]

UK_GRADE_THRESHOLDS = [
    (70, "a"),
    (60, "b"),
    (50, "c"),
    (40, "d"),
    (0,  "f"),
]

ENGAGEMENT_TYPES = {
    "oucontent", "forumng", "homepage", "ouwiki", "resource",
    "subpage", "url", "glossary", "dataplus", "ouelluminate",
    "questionnaire", "page", "dualpane", "folder", "htmlactivity",
    "oucollaborate", "repeatactivity", "sharedsubpage",
}

ENGAGEMENT_WEIGHT = 0.75

COURSE_NAME_MAP = {
    "Course_01": "intro_programming", "Course_02": "intro_programming",
    "Course_03": "intro_programming", "Course_04": "intro_programming",
    "Course_05": "intro_programming", "Course_06": "intro_programming",
    "Course_07": "discrete_math", "Course A": "data_structures",
    "Course_09": "other", "Course_10": "other", "Course_11": "other",
    "Course_12": "other", "Course_13": "other", "Course B": "other",
    "Course_15": "other", "Course_16": "other", "Course D": "other",
    "Course_18": "other", "Course_19": "other", "Course_20": "other",
    "Course C": "other", "Course_22": "other", "Course_23": "other",
    "Course_24": "other", "Course_25": "other",
    "MATH 220": "calculus_i", "MATH 221": "calculus_ii",
    "MATH 231": "calculus_ii", "MATH 241": "calculus_ii",
    "MATH 257": "linear_algebra", "MATH 415": "linear_algebra",
    "MATH 416": "linear_algebra",
    "STAT 100": "statistics", "STAT 200": "statistics",
    "STAT 400": "statistics",
    "CHEM 102": "general_chemistry", "CHEM 104": "general_chemistry",
    "PHYS 211": "general_physics", "PHYS 212": "general_physics",
    "RHET 105": "academic_writing",
    "AAA": "social_sciences", "BBB": "social_sciences",
    "CCC": "stem_module", "DDD": "stem_module",
    "EEE": "stem_module", "FFF": "stem_module",
    "GGG": "social_sciences",
}

ASSIGNMENT_TYPE_MAP = {
    "hw": "homework", "homework": "homework", "quiz": "quiz",
    "exam": "midterm", "midterm": "midterm", "final": "final",
    "project": "project", "mp": "project", "lab": "lab",
    "discussion": "participation", "lecture": "participation",
    "attendance": "participation", "essay": "essay",
    "presentation": "presentation", "participation": "participation",
    "other": "unknown",
    "tma": "homework", "cma": "quiz", "examen": "final",
    "oucontent": "participation", "forumng": "participation",
    "homepage": "participation", "ouwiki": "participation",
    "resource": "participation", "subpage": "participation",
    "url": "participation", "glossary": "participation",
    "dataplus": "participation", "ouelluminate": "participation",
    "questionnaire": "participation", "page": "participation",
    "dualpane": "participation", "folder": "participation",
    "htmlactivity": "participation", "oucollaborate": "participation",
    "repeatactivity": "participation", "sharedsubpage": "participation",
    "externalquiz": "quiz",
}

CLASS_TREND_MAP = {
    "improving": "improving", "stable": "stable",
    "declining": "declining", "fluctuating": "polarized",
    "unknown": "unknown",
}


def map_course_name(course_id):
    return COURSE_NAME_MAP.get(course_id, "other")

def map_assignment_type(raw_type):
    if raw_type is None:
        return "unknown"
    return ASSIGNMENT_TYPE_MAP.get(raw_type.lower().strip(), "unknown")

def map_class_trend(trend):
    return CLASS_TREND_MAP.get(trend, "unknown")

def load_db(path=None):
    path = path or CS_DB_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def lookup_course(db, course_id):
    for course in db:
        if course["course_id"] == course_id:
            return (
                course.get("course_info", {}),
                course.get("intervention", None),
                course.get("students", []),
            )
    return None, None, None

def get_syllabus(historical_students, up_to_week=None):
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

def extract_scores(student_record):
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

def _is_engagement_activity(activity):
    raw_type = activity.get("type", "")
    if raw_type is None:
        return False
    return raw_type.lower().strip() in ENGAGEMENT_TYPES

def extract_engagement(student_record):
    total_clicks = 0
    active_weeks = set()
    weekly_clicks = {}
    for week_data in student_record.get("weeks", []):
        week_num = week_data["week"]
        week_vle = 0
        for activity in week_data.get("activities", []):
            if _is_engagement_activity(activity):
                score = activity.get("score", 0)
                if score is not None and score > 0:
                    total_clicks += score
                    week_vle += score
                    active_weeks.add(week_num)
        if week_vle > 0:
            weekly_clicks[week_num] = week_vle
    return {
        "total_clicks": total_clicks,
        "active_weeks": len(active_weeks),
        "weekly_clicks": weekly_clicks,
        "has_vle": total_clicks > 0,
    }

def detect_vle_data(historical_students, sample_size=20):
    for student in historical_students[:sample_size]:
        if extract_engagement(student)["has_vle"]:
            return True
    return False

def build_engagement_distribution(historical_students):
    return [extract_engagement(s) for s in historical_students]

def normalize_engagement(student_eng, hist_engagements):
    if not student_eng["has_vle"]:
        return None
    student_clicks = student_eng["total_clicks"]
    hist_clicks = [e["total_clicks"] for e in hist_engagements if e["has_vle"]]
    if not hist_clicks:
        return None
    below = sum(1 for c in hist_clicks if c < student_clicks)
    equal = sum(1 for c in hist_clicks if c == student_clicks)
    return (below + 0.5 * equal) / len(hist_clicks) * 100

def predict_from_engagement(unseen_eng, historical_students, hist_engagements):
    if not unseen_eng["has_vle"]:
        return "no_match", 0.0, 0
    unseen_clicks = unseen_eng["total_clicks"]
    unseen_active = unseen_eng["active_weeks"]
    unseen_total_weeks = max(1, max(unseen_eng["weekly_clicks"].keys())) if unseen_eng["weekly_clicks"] else 1
    unseen_active_ratio = unseen_active / unseen_total_weeks
    candidates = []
    for i, student in enumerate(historical_students):
        grade = student.get("final_grade", "unknown")
        if grade == "unknown" or i >= len(hist_engagements):
            continue
        hist_eng = hist_engagements[i]
        if not hist_eng["has_vle"]:
            continue
        hist_clicks = hist_eng["total_clicks"]
        hist_active = hist_eng["active_weeks"]
        hist_total_weeks = max(1, max(hist_eng["weekly_clicks"].keys())) if hist_eng["weekly_clicks"] else 1
        hist_active_ratio = hist_active / hist_total_weeks
        max_clicks = max(unseen_clicks, hist_clicks, 1)
        click_dist = abs(unseen_clicks - hist_clicks) / max_clicks * 100
        ratio_dist = abs(unseen_active_ratio - hist_active_ratio) * 100
        combined_dist = click_dist * 0.7 + ratio_dist * 0.3
        candidates.append((combined_dist, grade.lower()))
    if not candidates:
        return "no_match", 0.0, 0
    candidates.sort(key=lambda x: x[0])
    n = len(candidates)
    k = max(5, min(15, int(math.sqrt(n))))
    k = min(k, n)
    topk = candidates[:k]
    weighted_votes = {}
    total_w = 0.0
    for dist, grade in topk:
        w = 1.0 / (1.0 + dist)
        weighted_votes[grade] = weighted_votes.get(grade, 0.0) + w
        total_w += w
    predicted = max(weighted_votes, key=weighted_votes.get)
    confidence = weighted_votes[predicted] / total_w if total_w > 0 else 0.0
    return predicted, confidence, k

def extract_graded_scores(student_record):
    scores = {}
    total_weight = 0.0
    max_week = 0
    for week_data in student_record.get("weeks", []):
        week_num = week_data["week"]
        if week_num > max_week:
            max_week = week_num
        for activity in week_data.get("activities", []):
            if _is_engagement_activity(activity):
                continue
            name = activity["name"]
            scores[name] = activity.get("score", None)
            total_weight += activity.get("weight", 0.0)
    return scores, total_weight, max_week

def learn_grade_thresholds(historical_students, syllabus):
    grade_scores = {}
    for student in historical_students:
        grade = student.get("final_grade", "unknown")
        if grade == "unknown":
            continue
        grade = grade.lower()
        scores, _, _ = extract_scores(student)
        ws, w = compute_weighted_score(scores, syllabus)
        if ws is not None:
            grade_scores.setdefault(grade, []).append(ws)
    if not grade_scores:
        return SCORE_GRADE_THRESHOLDS
    grade_medians = {}
    for grade, score_list in grade_scores.items():
        sorted_scores = sorted(score_list)
        n = len(sorted_scores)
        if n % 2 == 0:
            median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        else:
            median = sorted_scores[n // 2]
        grade_medians[grade] = median
    sorted_grades = sorted(grade_medians.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_grades) < 2:
        return SCORE_GRADE_THRESHOLDS
    thresholds = []
    for i in range(len(sorted_grades)):
        grade = sorted_grades[i][0]
        if i == len(sorted_grades) - 1:
            thresholds.append((0, grade))
        else:
            midpoint = (sorted_grades[i][1] + sorted_grades[i + 1][1]) / 2
            thresholds.append((midpoint, grade))
    return thresholds

def compute_adaptive_tolerance(historical_students, syllabus, base_tolerance=None):
    base_tolerance = base_tolerance or MATCH_TOLERANCE
    component_scores = {}
    for student in historical_students:
        scores, _, _ = extract_graded_scores(student)
        for name, score in scores.items():
            if score is not None:
                component_scores.setdefault(name, []).append(score)
    if not component_scores:
        return base_tolerance
    std_devs = []
    for name, score_list in component_scores.items():
        if len(score_list) < 3:
            continue
        mean = sum(score_list) / len(score_list)
        variance = sum((s - mean) ** 2 for s in score_list) / len(score_list)
        std_devs.append(math.sqrt(variance))
    if not std_devs:
        return base_tolerance
    avg_std = sum(std_devs) / len(std_devs)
    adaptive = 0.5 * avg_std
    adaptive = max(3.0, min(adaptive, base_tolerance))
    return adaptive

def fallback_predict_from_distribution(weighted_score, historical_students, syllabus):
    if weighted_score is None:
        return "unknown", 0.0
    hist_pairs = []
    for student in historical_students:
        grade = student.get("final_grade", "unknown")
        if grade == "unknown":
            continue
        scores, _, _ = extract_scores(student)
        ws, w = compute_weighted_score(scores, syllabus)
        if ws is not None:
            hist_pairs.append((ws, grade.lower()))
    if not hist_pairs:
        for student in historical_students:
            grade = student.get("final_grade", "unknown")
            if grade == "unknown":
                continue
            scores, _, _ = extract_scores(student)
            raw_avg = compute_raw_average_score(scores)
            if raw_avg is not None:
                hist_pairs.append((raw_avg, grade.lower()))
    if not hist_pairs:
        return "unknown", 0.0
    hist_pairs.sort(key=lambda x: abs(x[0] - weighted_score))
    k = max(3, min(20, int(math.sqrt(len(hist_pairs)))))
    nearest = hist_pairs[:k]
    weighted_votes = {}
    total_w = 0.0
    for ws, grade in nearest:
        dist = abs(ws - weighted_score)
        w = 1.0 / (1.0 + dist)
        weighted_votes[grade] = weighted_votes.get(grade, 0.0) + w
        total_w += w
    predicted = max(weighted_votes, key=weighted_votes.get)
    confidence = weighted_votes[predicted] / total_w if total_w > 0 else 0.0
    return predicted, confidence

def class_prior_predict(historical_students):
    grades = [s.get("final_grade", "unknown") for s in historical_students]
    grades = [g.lower() for g in grades if g != "unknown"]
    if not grades:
        return "unknown", 0.0
    counter = Counter(grades)
    most_common_grade, count = counter.most_common(1)[0]
    return most_common_grade, count / len(grades)

def compute_raw_average_score(scores):
    valid = [s for s in scores.values() if s is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)

def compute_weighted_score(scores, syllabus):
    weight_lookup = {c["name"]: c.get("weight", 0.0) for c in syllabus}
    total_weighted = 0.0
    total_weight = 0.0
    for name, score in scores.items():
        if score is None:
            continue
        weight = weight_lookup.get(name, 0.0)
        if weight <= 0:
            continue
        total_weighted += score * weight
        total_weight += weight
    if total_weight <= 0:
        return None, 0.0
    return total_weighted / total_weight, total_weight

def score_to_grade(weighted_score):
    if weighted_score is None:
        return "unknown"
    for threshold, grade in SCORE_GRADE_THRESHOLDS:
        if weighted_score >= threshold:
            return grade
    return "f"

def score_to_grade_adaptive(weighted_score, thresholds):
    if weighted_score is None:
        return "unknown"
    for threshold, grade in thresholds:
        if weighted_score >= threshold:
            return grade
    return thresholds[-1][1] if thresholds else "f"

def uk_threshold_predict(weighted_score):
    """V7: Apply UK grade thresholds 70/60/50/40 → A/B/C/D/F."""
    if weighted_score is None:
        return None
    for threshold, grade in UK_GRADE_THRESHOLDS:
        if weighted_score >= threshold:
            return grade
    return "f"

def match_students(unseen_scores, historical_students, tolerance=None,
                   unseen_eng_score=None, hist_eng_scores=None,
                   use_engagement=False):
    tolerance = tolerance or MATCH_TOLERANCE
    valid_scores = {n: s for n, s in unseen_scores.items() if s is not None}
    if len(valid_scores) < MIN_COMPONENTS_FOR_MATCH:
        return [], []
    min_shared = max(1, int(len(valid_scores) * MIN_SHARED_FRACTION))
    matches = []
    distances = []
    for i, hist_student in enumerate(historical_students):
        hist_scores, _, _ = extract_graded_scores(hist_student)
        diffs = []
        for comp, unseen_score in valid_scores.items():
            hist_score = hist_scores.get(comp)
            if hist_score is not None:
                diffs.append(abs(hist_score - unseen_score))
        if len(diffs) < min_shared:
            continue
        score_dist = sum(diffs) / len(diffs)
        if (use_engagement and unseen_eng_score is not None
                and hist_eng_scores is not None
                and i < len(hist_eng_scores)
                and hist_eng_scores[i] is not None):
            eng_dist = abs(unseen_eng_score - hist_eng_scores[i])
            combined_dist = score_dist * (1 - ENGAGEMENT_WEIGHT) + eng_dist * ENGAGEMENT_WEIGHT
        else:
            combined_dist = score_dist
        shared_ratio = len(diffs) / max(len(valid_scores), 1)
        scale = 0.4 + 0.6 * shared_ratio
        effective_tolerance = tolerance * scale
        if combined_dist <= effective_tolerance:
            matches.append(hist_student)
            distances.append(combined_dist)
    return matches, distances

def predict_grade(matched_students, match_distances=None):
    if not matched_students:
        return "no_match", 0.0, {}
    grades = [s.get("final_grade", "unknown") for s in matched_students]
    if match_distances and len(match_distances) == len(matched_students):
        weighted_votes = {}
        total_weight = 0.0
        for grade, dist in zip(grades, match_distances):
            w = 1.0 / (1.0 + dist)
            weighted_votes[grade] = weighted_votes.get(grade, 0.0) + w
            total_weight += w
        predicted = max(weighted_votes, key=weighted_votes.get)
        confidence = weighted_votes[predicted] / total_weight if total_weight > 0 else 0.0
        distribution = dict(Counter(grades))
    else:
        distribution = dict(Counter(grades))
        total = len(grades)
        predicted = max(distribution, key=distribution.get)
        confidence = distribution[predicted] / total
    return predicted.lower(), confidence, distribution

def count_missing(unseen_scores, syllabus_components):
    expected = {c["name"] for c in syllabus_components}
    submitted = set(unseen_scores.keys())
    missing = expected - submitted
    return len(missing), sorted(list(missing))

def compute_grade_trend(student_record):
    scores_by_week = []
    for week_data in sorted(student_record.get("weeks", []), key=lambda w: w["week"]):
        week_scores = [a.get("score", 0) for a in week_data.get("activities", []) if a.get("score") is not None]
        if week_scores:
            scores_by_week.append(sum(week_scores) / len(week_scores))
    if len(scores_by_week) < 2:
        return "unknown"
    diffs = [scores_by_week[i+1] - scores_by_week[i] for i in range(len(scores_by_week) - 1)]
    avg_diff = sum(diffs) / len(diffs)
    pos = sum(1 for d in diffs if d > 5)
    neg = sum(1 for d in diffs if d < -5)
    if pos >= 2 and neg >= 2:
        return "fluctuating"
    if avg_diff > 5:
        return "improving"
    elif avg_diff < -5:
        return "declining"
    else:
        return "stable"

def check_intervention(current_week, components_submitted, intervention_data):
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

def map_risk(predicted_grade):
    return RISK_MAPPING.get(predicted_grade.lower(), None)

def determine_failure_reason(unseen_scores, missing_count, grade_trend, syllabus):
    comp_lookup = {}
    for comp in syllabus:
        comp_lookup[comp["name"]] = {"type": comp.get("type", "unknown"), "weight": comp.get("weight", 0.0)}
    if missing_count >= 2:
        missing_damage = {}
        worst_missing_name = "unknown"
        worst_missing_weight = 0.0
        for comp in syllabus:
            if comp["name"] not in unseen_scores:
                atype = comp.get("type", "unknown")
                weight = comp.get("weight", 0.0)
                missing_damage[atype] = missing_damage.get(atype, 0.0) + weight
                if weight > worst_missing_weight:
                    worst_missing_weight = weight
                    worst_missing_name = comp["name"]
        worst_type = max(missing_damage, key=missing_damage.get) if missing_damage else "unknown"
        return "missing_work", worst_type, worst_missing_name
    type_damage = {}
    comp_damage = {}
    for name, score in unseen_scores.items():
        if score is None:
            continue
        info = comp_lookup.get(name)
        if info is None:
            continue
        damage = info["weight"] * (100.0 - score) / 100.0
        type_damage[info["type"]] = type_damage.get(info["type"], 0.0) + damage
        comp_damage[name] = damage
    if type_damage:
        worst_type = max(type_damage, key=type_damage.get)
        worst_comp = max(comp_damage, key=comp_damage.get) if comp_damage else "unknown"
        return "low_weighted_scores", worst_type, worst_comp
    if grade_trend == "declining":
        return "declining_trend", "none", "none"
    return "unclear", "none", "none"


def process_students(db, course_id, unseen_students):
    course_info, intervention_data, historical_students = lookup_course(db, course_id)
    if course_info is None:
        return {"error": f"Course {course_id} not found in database"}

    all_max_weeks = []
    for student in unseen_students:
        _, _, max_week = extract_scores(student)
        all_max_weeks.append(max_week)
    current_week = max(all_max_weeks) if all_max_weeks else 0

    syllabus = get_syllabus(historical_students, up_to_week=current_week)
    learned_thresholds = learn_grade_thresholds(historical_students, syllabus)
    adaptive_tol = compute_adaptive_tolerance(historical_students, syllabus)

    # V7: Detect OULAD for UK threshold prediction
    is_oulad = course_info.get("department") == "oulad"

    has_vle = detect_vle_data(historical_students)
    hist_eng_scores = None
    hist_engagements = None
    if has_vle:
        hist_engagements = build_engagement_distribution(historical_students)
        hist_eng_scores = [normalize_engagement(eng, hist_engagements) for eng in hist_engagements]

    student_results = []
    for student in unseen_students:
        sid = student.get("student_id", "unknown")
        scores, weight_covered, max_week = extract_scores(student)
        graded_scores, _, _ = extract_graded_scores(student)

        predicted_grade = None
        confidence = 0.0
        distribution = {}
        match_count = 0

        # =================================================================
        # V7: OULAD — UK threshold as primary predictor
        # =================================================================
        if is_oulad:
            weighted_score, weight_used = compute_weighted_score(scores, syllabus)
            threshold_pred = uk_threshold_predict(weighted_score)

            if threshold_pred is not None:
                predicted_grade = threshold_pred
                confidence = 1.0
                distribution = {"uk_threshold": True}
            else:
                # No graded work — fall back to V6 k-NN
                unseen_eng_score = None
                unseen_eng = None
                if has_vle:
                    unseen_eng = extract_engagement(student)
                    if unseen_eng["has_vle"]:
                        unseen_eng_score = normalize_engagement(unseen_eng, hist_engagements)

                matches, distances = match_students(
                    graded_scores, historical_students, tolerance=adaptive_tol,
                    unseen_eng_score=unseen_eng_score,
                    hist_eng_scores=hist_eng_scores,
                    use_engagement=has_vle,
                )
                match_count = len(matches)
                predicted_grade, confidence, distribution = predict_grade(matches, distances)

                if predicted_grade == "no_match":
                    effective_score = compute_raw_average_score(scores)
                    if effective_score is not None:
                        predicted_grade, confidence = fallback_predict_from_distribution(
                            effective_score, historical_students, syllabus
                        )
                    if predicted_grade in ("unknown", "no_match") and effective_score is not None:
                        predicted_grade = score_to_grade_adaptive(effective_score, learned_thresholds)
                        confidence = 0.0
                    if predicted_grade in ("unknown", "no_match") and has_vle and unseen_eng is not None:
                        eng_grade, eng_conf, eng_k = predict_from_engagement(
                            unseen_eng, historical_students, hist_engagements
                        )
                        if eng_grade not in ("unknown", "no_match"):
                            predicted_grade = eng_grade
                            confidence = eng_conf
                    if predicted_grade in ("unknown", "no_match"):
                        predicted_grade, confidence = class_prior_predict(historical_students)
                    distribution = {"fallback_used": True}

        # =================================================================
        # V6 path (completely unchanged)
        # =================================================================
        else:
            unseen_eng_score = None
            unseen_eng = None
            if has_vle:
                unseen_eng = extract_engagement(student)
                if unseen_eng["has_vle"]:
                    unseen_eng_score = normalize_engagement(unseen_eng, hist_engagements)

            matches, distances = match_students(
                graded_scores, historical_students, tolerance=adaptive_tol,
                unseen_eng_score=unseen_eng_score,
                hist_eng_scores=hist_eng_scores,
                use_engagement=has_vle,
            )
            match_count = len(matches)
            predicted_grade, confidence, distribution = predict_grade(matches, distances)

            if predicted_grade == "no_match":
                weighted_score, weight_used = compute_weighted_score(scores, syllabus)
                effective_score = weighted_score
                if effective_score is None:
                    effective_score = compute_raw_average_score(scores)
                if effective_score is not None:
                    predicted_grade, confidence = fallback_predict_from_distribution(
                        effective_score, historical_students, syllabus
                    )
                if predicted_grade in ("unknown", "no_match") and effective_score is not None:
                    predicted_grade = score_to_grade_adaptive(effective_score, learned_thresholds)
                    confidence = 0.0
                if predicted_grade in ("unknown", "no_match") and has_vle and unseen_eng is not None:
                    eng_grade, eng_conf, eng_k = predict_from_engagement(
                        unseen_eng, historical_students, hist_engagements
                    )
                    if eng_grade not in ("unknown", "no_match"):
                        predicted_grade = eng_grade
                        confidence = eng_conf
                if predicted_grade in ("unknown", "no_match"):
                    predicted_grade, confidence = class_prior_predict(historical_students)
                distribution = {"fallback_used": True}

        risk = map_risk(predicted_grade)
        missing_count, missing_names = count_missing(scores, syllabus)
        trend = compute_grade_trend(student)
        failure_reason, weak_type, weak_comp = determine_failure_reason(scores, missing_count, trend, syllabus)

        student_results.append({
            "student_id": sid,
            "predicted_grade": predicted_grade,
            "confidence": confidence,
            "match_count": match_count,
            "grade_distribution": distribution,
            "failure_risk": risk,
            "failure_risk_reason": failure_reason if risk else "none",
            "weak_assignment_type": weak_type if risk else "none",
            "weak_component_name": weak_comp if risk else "none",
            "missing_assignments_count": missing_count,
            "missing_assignments": missing_names,
            "grade_trend": trend,
            "weight_covered": round(weight_covered, 4),
        })

    components_submitted = len(syllabus)
    should_intervene, intervention_reason = check_intervention(
        current_week, components_submitted, intervention_data
    )

    risk_groups = {}
    for result in student_results:
        risk = result["failure_risk"]
        risk_key = "no_risk" if risk is None else f"{risk}_risk"
        if risk_key not in risk_groups:
            risk_groups[risk_key] = {
                "student_ids": [], "count": 0, "predicted_grades": [],
                "predicted_grade": "", "failure_risk": result["failure_risk"],
                "failure_risk_reasons": {}, "weak_assignment_types": {},
                "weak_component_names": {}, "missing_assignments": {},
                "grade_trends": {},
            }
        group = risk_groups[risk_key]
        group["student_ids"].append(result["student_id"])
        group["count"] += 1
        group["predicted_grades"].append(result["predicted_grade"])
        group["failure_risk_reasons"][result["student_id"]] = result["failure_risk_reason"]
        group["weak_assignment_types"][result["student_id"]] = result["weak_assignment_type"]
        group["weak_component_names"][result["student_id"]] = result["weak_component_name"]
        group["missing_assignments"][result["student_id"]] = result["missing_assignments_count"]
        group["grade_trends"][result["student_id"]] = result["grade_trend"]

    for risk_key, group in risk_groups.items():
        grades = group.pop("predicted_grades")
        group["predicted_grade"] = Counter(grades).most_common(1)[0][0] if grades else "unknown"
        group["per_student_grades"] = dict(zip(group["student_ids"], grades))

    all_predicted = [r["predicted_grade"] for r in student_results]
    gpas = [GRADE_TO_GPA.get(g, 0.0) for g in all_predicted]
    avg_gpa = round(sum(gpas) / len(gpas), 2) if gpas else 0.0
    flagged = [r for r in student_results if r["failure_risk"] is not None]
    flagged_count = len(flagged)
    all_trends = [r["grade_trend"] for r in student_results]
    trend_counts = Counter(all_trends)
    raw_trend = trend_counts.most_common(1)[0][0] if trend_counts else "unknown"
    overall_trend = map_class_trend(raw_trend)
    issue_types = [r["weak_assignment_type"] for r in flagged if r["weak_assignment_type"] != "none"]
    issue_counter = Counter(issue_types)
    common_issue = issue_counter.most_common(1)[0][0] if issue_counter else "none"

    intervention_plan = {}
    if should_intervene:
        for risk_key, group in risk_groups.items():
            if group["failure_risk"] is None:
                continue
            risk_level = group["failure_risk"]
            per_student_type = {}
            per_student_goal = {}
            for sid in group["student_ids"]:
                reason = group["failure_risk_reasons"][sid]
                if reason == "missing_work":
                    per_student_type[sid] = "study_plan"
                    per_student_goal[sid] = "recover_missing_work"
                elif reason == "low_weighted_scores":
                    per_student_type[sid] = "tutoring_referral"
                    per_student_goal[sid] = "improve_concept_mastery"
                elif reason == "declining_trend":
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
        "syllabus": syllabus,
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


# =============================================================================
# V8 HUMAN STUDY TOOLS (per-student, on-demand)
# =============================================================================

def _truncate_student_to_week(student_record, up_to_week):
    """Return a copy of the student with only data up through up_to_week."""
    if up_to_week is None:
        return student_record
    truncated_weeks = [
        w for w in student_record.get("weeks", []) if w["week"] <= up_to_week
    ]
    return {
        "student_id": student_record.get("student_id", "unknown"),
        "weeks": truncated_weeks,
        "final_grade": student_record.get("final_grade"),
    }


def _filter_scores_by_feature_set(scores, syllabus, feature_set):
    """
    Apply a feature set filter to a score dict.

    Feature sets:
      - "minimal": keep graded assignments only (drop engagement/participation);
                   used for noisy-tool condition (paired with early weeks → ~60%)
      - "full":    keep everything; used for clean-tool condition
                   (paired with late weeks → ~85%)
    """
    if feature_set == "full":
        return dict(scores)

    if feature_set == "minimal":
        # Keep only non-engagement, non-participation components with real weight
        allowed = set()
        for comp in syllabus:
            atype = comp.get("type", "unknown")
            if atype in ("participation",):
                continue
            if atype in ("unknown",):
                continue
            allowed.add(comp["name"])
        return {name: score for name, score in scores.items() if name in allowed}

    # Unknown feature set → fall back to full
    return dict(scores)


def predict_final_grade_for_student(
    db,
    course_id,
    student_record,
    up_to_week=None,
    feature_set="full",
):
    """
    Per-student, on-demand grade prediction for the human study.

    Same k-NN algorithm as process_students(), but:
      - Runs on ONE student at a time
      - Truncates data to `up_to_week`
      - Applies `feature_set` ("minimal" or "full")

    The SAME k-NN is used for both operating points. The accuracy difference
    comes from the data we feed in, not from changing the algorithm.

    Returns:
      {
        "student_id": str,
        "predicted_grade": "a" | "b" | "c" | "d" | "f" | "unknown",
        "confidence": float,
        "failure_risk": None | "medium" | "high" | "critical",
        "up_to_week": int,
        "feature_set": str,
        "match_count": int,
        "grade_distribution": dict,
      }
    """
    course_info, intervention_data, historical_students = lookup_course(db, course_id)
    if course_info is None:
        return {"error": f"Course {course_id} not found in database"}

    # Truncate student to the requested week
    student = _truncate_student_to_week(student_record, up_to_week)
    sid = student.get("student_id", "unknown")

    # Build syllabus up to that week
    syllabus = get_syllabus(historical_students, up_to_week=up_to_week)
    learned_thresholds = learn_grade_thresholds(historical_students, syllabus)
    adaptive_tol = compute_adaptive_tolerance(historical_students, syllabus)

    is_oulad = course_info.get("department") == "oulad"

    # Extract scores, apply feature-set filter
    scores, weight_covered, max_week = extract_scores(student)
    graded_scores, _, _ = extract_graded_scores(student)

    scores = _filter_scores_by_feature_set(scores, syllabus, feature_set)
    graded_scores = _filter_scores_by_feature_set(graded_scores, syllabus, feature_set)

    has_vle = detect_vle_data(historical_students) and feature_set == "full"
    hist_engagements = None
    hist_eng_scores = None
    if has_vle:
        hist_engagements = build_engagement_distribution(historical_students)
        hist_eng_scores = [
            normalize_engagement(eng, hist_engagements) for eng in hist_engagements
        ]

    predicted_grade = None
    confidence = 0.0
    distribution = {}
    match_count = 0

    # OULAD: UK threshold first
    if is_oulad:
        weighted_score, _ = compute_weighted_score(scores, syllabus)
        threshold_pred = uk_threshold_predict(weighted_score)
        if threshold_pred is not None:
            predicted_grade = threshold_pred
            confidence = 1.0
            distribution = {"uk_threshold": True}

    # Primary dataset path, or OULAD with no graded work
    if predicted_grade is None:
        unseen_eng_score = None
        unseen_eng = None
        if has_vle:
            unseen_eng = extract_engagement(student)
            if unseen_eng["has_vle"]:
                unseen_eng_score = normalize_engagement(unseen_eng, hist_engagements)

        matches, distances = match_students(
            graded_scores,
            historical_students,
            tolerance=adaptive_tol,
            unseen_eng_score=unseen_eng_score,
            hist_eng_scores=hist_eng_scores,
            use_engagement=has_vle,
        )
        match_count = len(matches)
        predicted_grade, confidence, distribution = predict_grade(matches, distances)

        if predicted_grade == "no_match":
            weighted_score, _ = compute_weighted_score(scores, syllabus)
            effective_score = weighted_score
            if effective_score is None:
                effective_score = compute_raw_average_score(scores)
            if effective_score is not None:
                predicted_grade, confidence = fallback_predict_from_distribution(
                    effective_score, historical_students, syllabus
                )
            if predicted_grade in ("unknown", "no_match") and effective_score is not None:
                predicted_grade = score_to_grade_adaptive(effective_score, learned_thresholds)
                confidence = 0.0
            if predicted_grade in ("unknown", "no_match") and has_vle and unseen_eng is not None:
                eng_grade, eng_conf, _ = predict_from_engagement(
                    unseen_eng, historical_students, hist_engagements
                )
                if eng_grade not in ("unknown", "no_match"):
                    predicted_grade = eng_grade
                    confidence = eng_conf
            if predicted_grade in ("unknown", "no_match"):
                predicted_grade, confidence = class_prior_predict(historical_students)
            distribution = {"fallback_used": True}

    risk = map_risk(predicted_grade)

    return {
        "student_id": sid,
        "predicted_grade": predicted_grade,
        "confidence": round(confidence, 4),
        "failure_risk": risk,
        "up_to_week": up_to_week,
        "feature_set": feature_set,
        "match_count": match_count,
        "grade_distribution": distribution,
    }


def suggest_intervention_for_student(student_id, grades_lookup):
    """
    Build a prompt + structured context for an LLM to generate intervention options.

    This tool does NOT call the LLM itself. It returns:
      - a student profile summary (grades, patterns, concerns)
      - a formatted prompt the caller can send to any LLM
    The caller (orchestrator or Streamlit app) handles the actual LLM call.

    This keeps tools.py free of LLM dependencies and honest — we don't pretend
    this is a trained ML tool. It's LLM-generated suggestions, nothing more.

    Returns:
      {
        "student_id": str,
        "profile": {...},     # structured facts about the student
        "prompt": str,        # ready-to-send LLM prompt
      }
      or {"error": "..."} if student not found.
    """
    if student_id not in grades_lookup:
        return {"error": f"Student {student_id} not found."}

    scores = grades_lookup[student_id]

    # Compute summary stats
    total_weighted = 0.0
    total_weight = 0.0
    missing_assignments = []
    low_scores = []
    scores_by_week = {}

    for name, info in scores.items():
        score = info.get("score")
        weight = info.get("weight", 0.0)
        week = info.get("week", 0)
        atype = info.get("type", "unknown")

        if score is None:
            missing_assignments.append({"name": name, "type": atype, "week": week})
        elif weight > 0:
            total_weighted += score * weight
            total_weight += weight
            if score < 65:
                low_scores.append({"name": name, "score": score, "type": atype, "week": week})

            scores_by_week.setdefault(week, []).append(score)

    weighted_avg = round(total_weighted / total_weight, 2) if total_weight > 0 else None

    # Simple trend: compare first half vs second half weekly averages
    trend = "unknown"
    if scores_by_week:
        weeks = sorted(scores_by_week.keys())
        if len(weeks) >= 4:
            mid = len(weeks) // 2
            early_avg = sum(
                sum(scores_by_week[w]) / len(scores_by_week[w]) for w in weeks[:mid]
            ) / mid
            late_avg = sum(
                sum(scores_by_week[w]) / len(scores_by_week[w]) for w in weeks[mid:]
            ) / (len(weeks) - mid)
            diff = late_avg - early_avg
            if diff > 5:
                trend = "improving"
            elif diff < -5:
                trend = "declining"
            else:
                trend = "stable"

    profile = {
        "student_id": student_id,
        "weighted_average": weighted_avg,
        "missing_assignment_count": len(missing_assignments),
        "missing_assignments": missing_assignments,
        "low_score_count": len(low_scores),
        "low_scores": low_scores,
        "trend": trend,
    }

    # Build a prompt the caller can send to any LLM
    # Format the weak assignments with specifics
    weak_detail = "\n".join([
        f"  - {x['name']} (week {x['week']}, type {x['type']}): scored {x['score']}"
        for x in low_scores[:5]
    ]) if low_scores else "  (none)"

    missing_detail = "\n".join([
        f"  - {x['name']} (week {x['week']}, type {x['type']})"
        for x in missing_assignments[:5]
    ]) if missing_assignments else "  (none)"

    prompt_parts = [
        f"Student {student_id} is at risk. Profile:",
        f"- Weighted average: {weighted_avg if weighted_avg is not None else 'N/A'}",
        f"- Grade trend: {trend}",
        f"- Weak assignments (score <65):",
        weak_detail,
        f"- Missing assignments:",
        missing_detail,
        "",
        "Suggest 2-3 SPECIFIC interventions grounded in this student's actual weak "
        "areas. Reference the specific assignment names and types above. Avoid "
        "generic advice like 'one-on-one meeting' unless you tie it to a specific "
        "topic or assignment. Keep each suggestion to one sentence. The instructor "
        "decides what to apply.",
    ]
    prompt = "\n".join(prompt_parts)

    return {
        "student_id": student_id,
        "profile": profile,
        "prompt": prompt,
    }


# =============================================================================
# DRILL-DOWN TOOLS (for human study / Streamlit app)
# =============================================================================

def build_grades_lookup(grades_data):
    """
    Build a compact lookup of ALL students' per-assignment scores from a grades file.
    Format: {student_id: {assignment_name: {score, weight, type, week}, ...}, ...}
    """
    if not grades_data:
        return {}

    if isinstance(grades_data, dict):
        students = grades_data.get("students", [])
    elif isinstance(grades_data, list):
        students = grades_data
    else:
        return {}

    lookup = {}
    for student in students:
        sid = student.get("student_id", "unknown")
        scores = {}
        for week_data in student.get("weeks", []):
            week_num = week_data["week"]
            for activity in week_data.get("activities", []):
                name = activity.get("name", "unknown")
                scores[name] = {
                    "score": activity.get("score", None),
                    "weight": activity.get("weight", 0.0),
                    "type": activity.get("type", "unknown"),
                    "week": week_num,
                }
        lookup[sid] = scores

    return lookup


def get_student_grades(student_id, grades_lookup):
    """
    Get a student's full grade record with weighted average.
    """
    if student_id not in grades_lookup:
        return {"error": f"Student {student_id} not found."}

    scores = grades_lookup[student_id]
    total_weighted = 0.0
    total_weight = 0.0

    assignments = []
    for name, info in sorted(scores.items(), key=lambda x: x[1].get("week", 0)):
        score = info.get("score")
        weight = info.get("weight", 0.0)
        assignments.append({
            "name": name,
            "score": score,
            "weight": round(weight, 4),
            "type": info.get("type", "unknown"),
            "week": info.get("week", 0),
        })
        if score is not None and weight > 0:
            total_weighted += score * weight
            total_weight += weight

    weighted_avg = round(total_weighted / total_weight, 2) if total_weight > 0 else None

    return {
        "student_id": student_id,
        "assignments": assignments,
        "weighted_average": weighted_avg,
        "total_weight_covered": round(total_weight, 4),
    }


def recalculate_grade(student_id, grades_lookup, drop=None, override=None,
                     simulate_remaining=None, full_syllabus=None):
    """
    Recalculate a student's weighted average with modifications.

    Args:
        drop: list of assignment names to exclude
        override: dict of {assignment_name: new_score} for existing assignments
        simulate_remaining: dict of {assignment_name: score} for FUTURE assignments
                            not yet in the gradebook
        full_syllabus: list of all course assignments (from get_course_syllabus)
                       needed when simulate_remaining is used, to get weights
    """
    if student_id not in grades_lookup:
        return {"error": f"Student {student_id} not found."}

    drop = drop or []
    override = override or {}
    simulate_remaining = simulate_remaining or {}
    # Agent sometimes sends a list of {assignment_name, score} instead of a dict
    if isinstance(simulate_remaining, list):
        simulate_remaining = {
            item.get("assignment_name") or item.get("name"): item.get("score")
            for item in simulate_remaining
            if isinstance(item, dict)
        }
    scores = grades_lookup[student_id]

    original_weighted = 0.0
    original_weight = 0.0
    new_weighted = 0.0
    new_weight = 0.0
    changes = []

    # Original average (based on existing graded work only)
    for name, info in scores.items():
        score = info.get("score")
        weight = info.get("weight", 0.0)
        if score is not None and weight > 0:
            original_weighted += score * weight
            original_weight += weight

    # Recalculated: apply drops / overrides on existing scores
    for name, info in scores.items():
        score = info.get("score")
        weight = info.get("weight", 0.0)

        if name in drop:
            changes.append({"assignment": name, "action": "dropped", "original_score": score})
            continue

        if name in override:
            new_score = override[name]
            changes.append({
                "assignment": name, "action": "overridden",
                "original_score": score, "new_score": new_score,
            })
            if weight > 0:
                new_weighted += new_score * weight
                new_weight += weight
        else:
            if score is not None and weight > 0:
                new_weighted += score * weight
                new_weight += weight

    # Add simulated future scores using full syllabus weights
    if simulate_remaining and full_syllabus:
        weight_lookup = {a["name"]: a.get("weight", 0.0) for a in full_syllabus}
        existing_names = set(scores.keys())
        for name, sim_score in simulate_remaining.items():
            if name in existing_names:
                continue  # can't simulate an already-graded assignment, use override instead
            weight = weight_lookup.get(name, 0.0)
            if weight > 0:
                new_weighted += sim_score * weight
                new_weight += weight
                changes.append({
                    "assignment": name, "action": "simulated",
                    "simulated_score": sim_score, "weight": weight,
                })

    original_avg = round(original_weighted / original_weight, 2) if original_weight > 0 else None
    new_avg = round(new_weighted / new_weight, 2) if new_weight > 0 else None

    return {
        "student_id": student_id,
        "original_weighted_average": original_avg,
        "recalculated_weighted_average": new_avg,
        "difference": round(new_avg - original_avg, 2) if original_avg and new_avg else None,
        "changes": changes,
    }


def get_assignment_stats(assignment_name, grades_lookup):
    """
    Get class-wide statistics for a specific assignment.
    """
    scores = []
    for sid, student_scores in grades_lookup.items():
        if assignment_name in student_scores:
            score = student_scores[assignment_name].get("score")
            if score is not None:
                scores.append(score)

    if not scores:
        return {"error": f"Assignment '{assignment_name}' not found or no scores available."}

    avg = round(sum(scores) / len(scores), 2)
    min_score = round(min(scores), 2)
    max_score = round(max(scores), 2)

    buckets = {"90-100": 0, "80-89": 0, "70-79": 0, "60-69": 0, "below_60": 0}
    for s in scores:
        if s >= 90:
            buckets["90-100"] += 1
        elif s >= 80:
            buckets["80-89"] += 1
        elif s >= 70:
            buckets["70-79"] += 1
        elif s >= 60:
            buckets["60-69"] += 1
        else:
            buckets["below_60"] += 1

    return {
        "assignment": assignment_name,
        "student_count": len(scores),
        "average": avg,
        "min": min_score,
        "max": max_score,
        "distribution": buckets,
    }


def filter_students(assignment_name, threshold, grades_lookup, direction="below"):
    """
    Find students who scored below (or above) a threshold on a specific assignment.
    """
    matches = []
    for sid, student_scores in grades_lookup.items():
        if assignment_name not in student_scores:
            continue
        score = student_scores[assignment_name].get("score")
        if score is None:
            continue

        if direction == "below" and score < threshold:
            matches.append({"student_id": sid, "score": round(score, 2)})
        elif direction == "above" and score > threshold:
            matches.append({"student_id": sid, "score": round(score, 2)})

    matches.sort(key=lambda x: x["score"], reverse=(direction == "above"))

    return {
        "assignment": assignment_name,
        "threshold": threshold,
        "direction": direction,
        "count": len(matches),
        "students": matches,
    }


def filter_students_by_grade(target_grade, grades_lookup):
    """
    Find students whose cumulative weighted average falls in a letter grade range.
    A: >=90, B: 80-89, C: 70-79, D: 60-69, F: <60

    NOTE: This is based on CURRENT weighted average, not a prediction.
    For predicted final grades, use predict_final_grade_for_student().
    """
    grade_ranges = {
        "a": (90, 100),
        "b": (80, 89.99),
        "c": (70, 79.99),
        "d": (60, 69.99),
        "f": (0, 59.99),
    }

    target = target_grade.lower()
    if target not in grade_ranges:
        return {"error": f"Unknown grade: {target_grade}. Use A, B, C, D, or F."}

    low, high = grade_ranges[target]
    matches = []

    for sid, student_scores in grades_lookup.items():
        total_weighted = 0.0
        total_weight = 0.0
        for name, info in student_scores.items():
            score = info.get("score")
            weight = info.get("weight", 0.0)
            if score is not None and weight > 0:
                total_weighted += score * weight
                total_weight += weight

        if total_weight == 0:
            continue

        avg = round(total_weighted / total_weight, 2)
        if low <= avg <= high:
            matches.append({"student_id": sid, "weighted_average": avg})

    matches.sort(key=lambda x: x["weighted_average"], reverse=True)

    return {
        "target_grade": target_grade.upper(),
        "range": f"{low}-{high}",
        "count": len(matches),
        "students": matches,
    }


def minimum_score_needed(student_id, target_grade, grades_lookup):
    """
    Calculate the minimum score needed on remaining assignments to reach a target grade.
    """
    grade_thresholds = {"a": 90, "b": 80, "c": 70, "d": 60}
    target_threshold = grade_thresholds.get(target_grade.lower())
    if target_threshold is None:
        return {"error": f"Invalid target grade: {target_grade}. Use a, b, c, or d."}

    if student_id not in grades_lookup:
        return {"error": f"Student {student_id} not found."}

    scores = grades_lookup[student_id]

    current_weighted = 0.0
    current_weight = 0.0
    remaining_weight = 0.0
    remaining_names = []

    for name, info in scores.items():
        score = info.get("score")
        weight = info.get("weight", 0.0)
        if score is not None and weight > 0:
            current_weighted += score * weight
            current_weight += weight
        elif score is None and weight > 0:
            remaining_weight += weight
            remaining_names.append(name)

    total_weight = current_weight + remaining_weight
    if total_weight <= 0 or remaining_weight <= 0:
        return {
            "student_id": student_id,
            "error": "No remaining assignments with weight found.",
            "current_weighted_average": round(current_weighted / current_weight, 2) if current_weight > 0 else None,
        }

    needed_score = (target_threshold * total_weight - current_weighted) / remaining_weight
    needed_score = round(needed_score, 2)

    return {
        "student_id": student_id,
        "target_grade": target_grade.upper(),
        "target_threshold": target_threshold,
        "current_weighted_average": round(current_weighted / current_weight, 2) if current_weight > 0 else None,
        "current_weight_covered": round(current_weight, 4),
        "remaining_weight": round(remaining_weight, 4),
        "remaining_assignments": remaining_names,
        "minimum_score_needed": needed_score,
        "achievable": needed_score <= 100,
    }

def get_course_syllabus(db, course_id, current_week=None):
    """
    Return the FULL course syllabus from the training DB (all weeks, not truncated).

    For the human study: the agent uses this to tell the instructor what
    assignments are still remaining in the semester.

    Args:
        db: the training database (loaded from cs_db_train.json)
        course_id: e.g. "Course C"
        current_week: if provided, each assignment is marked "graded" or "remaining"

    Returns:
        {
          "course_id": str,
          "current_week": int or None,
          "assignments": [
              {name, type, weight, week, status: "graded"|"remaining"}
          ],
          "total_weight": float,
          "graded_weight": float,
          "remaining_weight": float,
        }
    """
    course_info, _, historical_students = lookup_course(db, course_id)
    if course_info is None:
        return {"error": f"Course {course_id} not found in training DB"}

    # Pull the full syllabus (no week cap)
    full_syllabus = get_syllabus(historical_students, up_to_week=None)

    assignments = []
    total_weight = 0.0
    graded_weight = 0.0
    remaining_weight = 0.0

    for comp in sorted(full_syllabus, key=lambda c: (c["week"], c["name"])):
        week = comp.get("week", 0)
        weight = comp.get("weight", 0.0)
        if current_week is not None and week <= current_week:
            status = "graded"
            graded_weight += weight
        else:
            status = "remaining"
            remaining_weight += weight
        total_weight += weight

        assignments.append({
            "name": comp["name"],
            "type": comp.get("type", "unknown"),
            "weight": round(weight, 4),
            "week": week,
            "status": status,
        })

    return {
        "course_id": course_id,
        "current_week": current_week,
        "assignments": assignments,
        "total_weight": round(total_weight, 4),
        "graded_weight": round(graded_weight, 4),
        "remaining_weight": round(remaining_weight, 4),
    }

def list_all_assignments(grades_lookup):
    """
    List all unique assignments across all students with their types and weights.
    """
    assignments = {}
    for sid, student_scores in grades_lookup.items():
        for name, info in student_scores.items():
            if name not in assignments:
                assignments[name] = {
                    "name": name,
                    "type": info.get("type", "unknown"),
                    "weight": info.get("weight", 0.0),
                    "week": info.get("week", 0),
                }

    sorted_assignments = sorted(assignments.values(), key=lambda x: x["week"])
    return {"assignments": sorted_assignments, "count": len(sorted_assignments)}

def get_class_average(grades_lookup):
    """Compute the class-wide average weighted grade."""
    averages = []
    for sid in grades_lookup:
        rec = get_student_grades(sid, grades_lookup)
        if "error" in rec:
            continue
        avg = rec.get("weighted_average")
        if avg is not None:
            averages.append(avg)
    if not averages:
        return {"error": "No students with graded work."}
    return {
        "class_average": round(sum(averages) / len(averages), 2),
        "student_count": len(averages),
        "min": round(min(averages), 2),
        "max": round(max(averages), 2),
    }

def simulate_uniform_remaining(db, course_id, current_week, student_id, grades_lookup, uniform_score):
    """
    Guarantee-style counterfactual: assume the student scores `uniform_score` on every
    remaining (not-yet-graded) assignment. Returns new weighted average + letter grade.

    Args:
        db: training DB (from load_db)
        course_id: course ID string
        current_week: current week cutoff
        student_id: student to simulate
        grades_lookup: existing grades lookup
        uniform_score: score to apply to all remaining assignments (e.g. 80)
    """
    if student_id not in grades_lookup:
        return {"error": f"Student {student_id} not found."}

    # Get full syllabus to know what's remaining
    syl_result = get_course_syllabus(db, course_id, current_week=current_week)
    full_syllabus = syl_result.get("assignments", [])
    remaining = [a for a in full_syllabus if a.get("status") == "remaining"]

    if not remaining:
        return {
            "student_id": student_id,
            "note": "No remaining assignments — everything is already graded.",
            "current_weighted_average": None,
            "simulated_weighted_average": None,
        }

    # Build simulate_remaining dict
    simulate_remaining = {a["name"]: uniform_score for a in remaining}

    # Call recalculate with the dict
    result = recalculate_grade(
        student_id, grades_lookup,
        simulate_remaining=simulate_remaining,
        full_syllabus=full_syllabus,
    )

    # Add letter grade
    avg = result.get("recalculated_weighted_average")
    if avg is not None:
        if avg >= 90:
            letter = "A"
        elif avg >= 80:
            letter = "B"
        elif avg >= 70:
            letter = "C"
        elif avg >= 60:
            letter = "D"
        else:
            letter = "F"
        result["simulated_letter_grade"] = letter

    result["uniform_score_applied"] = uniform_score
    result["n_remaining_assignments"] = len(remaining)
    result["remaining_assignment_names"] = [a["name"] for a in remaining]

    return result