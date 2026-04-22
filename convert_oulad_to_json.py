"""
PredAct Benchmark - OULAD Converter

Reads raw OULAD CSVs and produces `oulad_db.json` in the same schema as UIUC's
`cs_db.json`, so the existing split_data.py can be used on it directly.

Schema:
    [
      {
        "course_id": "AAA_2013J",
        "course_info": {...},
        "students": [
          {
            "student_id": "12345",
            "weeks": [
              {"week": 1, "activities": [
                  {"name": "TMA01", "type": "tma", "weight": 0.10, "score": 82.0},
                  {"name": "VLE_clicks", "type": "engagement", "weight": 0.0, "score": 47}
              ]}
            ],
            "final_grade": "a"
          }
        ]
      },
      ...
    ]

Design choices (confirmed with user, per OULAD docs):
  - Each (module, presentation) is a separate course -> 22 courses total.
  - Final grade mapped from exam score using UK thresholds (70/60/50/40 = A/B/C/D/F).
    Students without exam score fall back to their weighted-continuous-assessment score.
    Withdrawn students are marked "f".
  - Weekly activities include:
      TMA/CMA/Exam assessment submissions (placed in the week containing their date)
      VLE click aggregation per week (name = "VLE_clicks", type = "engagement")
  - Each week is a 7-day bucket starting at day 0 (presentation start).
    Day 0-6 = week 1, day 7-13 = week 2, etc.

Usage:
    python convert_oulad_to_json.py
    python convert_oulad_to_json.py --oulad-dir /custom/path --output /custom/out.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict

DEFAULT_OULAD_DIR = "/home/alrabah2/PredAct_bench/results/oulad"
DEFAULT_OUTPUT = "/home/alrabah2/PredAct_bench/results/oulad/oulad_db.json"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def day_to_week(day):
    """Convert day (relative to presentation start) to 1-indexed week number.
    Day 0-6 = week 1, Day 7-13 = week 2, etc. Days < 0 are clamped to week 1."""
    if day is None:
        return None
    try:
        d = int(day)
    except (TypeError, ValueError):
        return None
    if d < 0:
        return 1
    return (d // 7) + 1


def score_to_letter(score):
    """UK grading: >=70 A, >=60 B, >=50 C, >=40 D, else F."""
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s >= 70:
        return "a"
    if s >= 60:
        return "b"
    if s >= 50:
        return "c"
    if s >= 40:
        return "d"
    return "f"


def safe_float(v):
    try:
        return float(v) if v not in (None, "", "NA") else None
    except (TypeError, ValueError):
        return None


def safe_int(v):
    try:
        return int(v) if v not in (None, "", "NA") else None
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# Load raw CSVs
# -----------------------------------------------------------------------------

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -----------------------------------------------------------------------------
# Build DB
# -----------------------------------------------------------------------------

def build_oulad_db(oulad_dir):
    print("Loading OULAD CSVs...")
    assessments = load_csv(os.path.join(oulad_dir, "assessments.csv"))
    student_info = load_csv(os.path.join(oulad_dir, "studentInfo.csv"))
    student_assessment = load_csv(os.path.join(oulad_dir, "studentAssessment.csv"))
    student_vle = load_csv(os.path.join(oulad_dir, "studentVle.csv"))

    print(f"  assessments: {len(assessments)}")
    print(f"  studentInfo: {len(student_info)}")
    print(f"  studentAssessment: {len(student_assessment)}")
    print(f"  studentVle: {len(student_vle)}")

    # Index assessments by id_assessment
    # Each assessment belongs to one (module, presentation)
    assessment_meta = {}
    for a in assessments:
        aid = a["id_assessment"]
        assessment_meta[aid] = {
            "code_module": a["code_module"],
            "code_presentation": a["code_presentation"],
            "assessment_type": a["assessment_type"],
            "date": safe_int(a.get("date")),
            "weight": safe_float(a.get("weight")) or 0.0,
        }

    # Build per-course list of assessments (for course_info / reference)
    course_assessments = defaultdict(list)  # (module, presentation) -> list of meta
    for aid, meta in assessment_meta.items():
        key = (meta["code_module"], meta["code_presentation"])
        course_assessments[key].append({
            "id_assessment": aid,
            "assessment_type": meta["assessment_type"],
            "date": meta["date"],
            "weight": meta["weight"],
        })

    # Index student assessment submissions:
    # (module, presentation, student) -> list of {id_assessment, score, date_submitted}
    student_submissions = defaultdict(list)
    for sa in student_assessment:
        aid = sa["id_assessment"]
        meta = assessment_meta.get(aid)
        if meta is None:
            continue
        key = (meta["code_module"], meta["code_presentation"], sa["id_student"])
        student_submissions[key].append({
            "id_assessment": aid,
            "score": safe_float(sa.get("score")),
            "date_submitted": safe_int(sa.get("date_submitted")),
            "assessment_type": meta["assessment_type"],
            "date": meta["date"],
            "weight": meta["weight"],
        })

    # Aggregate VLE clicks per (module, presentation, student, week)
    # Key: (module, presentation, student) -> {week -> total_clicks}
    student_vle_clicks = defaultdict(lambda: defaultdict(int))
    for v in student_vle:
        key = (v["code_module"], v["code_presentation"], v["id_student"])
        week = day_to_week(v.get("date"))
        clicks = safe_int(v.get("sum_click")) or 0
        if week is not None:
            student_vle_clicks[key][week] += clicks

    # Build students per course (= module+presentation)
    print("\nBuilding courses...")
    courses = defaultdict(lambda: {"students": [], "course_info": {}})

    for si in student_info:
        module = si["code_module"]
        presentation = si["code_presentation"]
        sid = si["id_student"]
        final_result = (si.get("final_result") or "").strip()

        course_id = f"{module}_{presentation}"
        course_key = (module, presentation)

        # Get this student's assessment submissions
        subs = student_submissions.get((module, presentation, sid), [])

        # Compute weighted continuous assessment score for fallback final-grade mapping
        cont_subs = [s for s in subs if s["assessment_type"] in ("TMA", "CMA")]
        total_weight = sum(s["weight"] for s in cont_subs if s["score"] is not None)
        if total_weight > 0:
            ca_score = sum(s["weight"] * s["score"] for s in cont_subs if s["score"] is not None) / total_weight
        else:
            ca_score = None

        # Exam score (if any)
        exam_subs = [s for s in subs if s["assessment_type"] == "Exam" and s["score"] is not None]
        exam_score = exam_subs[0]["score"] if exam_subs else None

        # Determine final letter grade
        # Priority: exam score (if present) -> continuous assessment (if present) -> withdrawn=f / fail=f / distinction=a / pass=c
        if final_result == "Withdrawn":
            final_grade = "f"
        elif exam_score is not None:
            final_grade = score_to_letter(exam_score)
        elif ca_score is not None:
            final_grade = score_to_letter(ca_score)
        else:
            # Use final_result as a last resort
            mapping = {"Distinction": "a", "Pass": "c", "Fail": "f"}
            final_grade = mapping.get(final_result, None)

        if final_grade is None:
            continue  # skip students with no determinable grade

        # Build weekly activity structure
        # Week -> list of activities
        weeks_dict = defaultdict(list)

        # 1. Add assessment submissions (placed in week of assessment's due date)
        for sub in subs:
            if sub["date"] is None:
                continue
            week = day_to_week(sub["date"])
            if week is None or week < 1:
                continue
            name = f"{sub['assessment_type']}_{sub['id_assessment']}"
            weight_frac = (sub["weight"] or 0.0) / 100.0  # convert percent to fraction
            activity = {
                "name": name,
                "type": sub["assessment_type"].lower(),
                "weight": round(weight_frac, 4),
            }
            if sub["score"] is not None:
                activity["score"] = sub["score"]
            weeks_dict[week].append(activity)

        # 2. Add weekly VLE click aggregation
        vle_by_week = student_vle_clicks.get((module, presentation, sid), {})
        for week, total_clicks in vle_by_week.items():
            if week < 1:
                continue
            weeks_dict[week].append({
                "name": "VLE_clicks",
                "type": "engagement",
                "weight": 0.0,
                "score": float(total_clicks),
            })

        # Sort weeks and flatten
        weeks = []
        for w in sorted(weeks_dict.keys()):
            weeks.append({"week": w, "activities": weeks_dict[w]})

        if not weeks:
            continue  # skip students with no recorded activity

        courses[course_key]["students"].append({
            "student_id": sid,
            "weeks": weeks,
            "final_grade": final_grade,
        })

        # Fill course_info once (aggregate grade distribution later if needed)
        if not courses[course_key]["course_info"]:
            # We'll fill per-course stats at the end
            pass

    # Compute per-course aggregates (avg_gpa, pct_A/B/C/D/F)
    gpa_points = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0, "f": 0.0}
    for course_key, info in courses.items():
        students = info["students"]
        if not students:
            continue
        grades = [s["final_grade"] for s in students]
        n = len(grades)
        course_info = {
            "avg_gpa": round(sum(gpa_points[g] for g in grades) / n, 3),
            "pct_A": round(grades.count("a") / n, 4),
            "pct_B": round(grades.count("b") / n, 4),
            "pct_C": round(grades.count("c") / n, 4),
            "pct_D": round(grades.count("d") / n, 4),
            "pct_F": round(grades.count("f") / n, 4),
            "grading_scale_A": 70.0,  # UK threshold for Distinction/A
        }
        info["course_info"] = course_info

    # Build output list sorted by course_id
    output = []
    for (module, presentation), data in sorted(courses.items()):
        course_id = f"{module}_{presentation}"
        output.append({
            "course_id": course_id,
            "course_info": data["course_info"],
            "students": data["students"],
        })

    return output


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert raw OULAD CSVs to PredAct JSON format.")
    parser.add_argument("--oulad-dir", default=DEFAULT_OULAD_DIR,
                        help=f"Directory with raw OULAD CSVs (default: {DEFAULT_OULAD_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    db = build_oulad_db(args.oulad_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    total_students = sum(len(c["students"]) for c in db)
    print(f"\n{'=' * 60}")
    print(f"OULAD CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Courses (module,presentation): {len(db)}")
    print(f"  Total students:                {total_students}")
    print(f"  Output:                        {args.output}")
    print()

    # Per-course breakdown
    print(f"  Per-course student counts:")
    for c in db:
        print(f"    {c['course_id']}: {len(c['students'])} students")


if __name__ == "__main__":
    main()