"""
OULAD → PredAct Converter (V3 — UK Grading)
Computes final grades from weighted assessment scores using UK thresholds.
Weighted average is computed over SUBMITTED assessments only.
Unsubmitted assessments are ignored (not counted as 0).

UK grading:
  70+   → A (First-Class Honours)
  60-69 → B (Upper Second / 2:1)
  50-59 → C (Lower Second / 2:2)
  40-49 → D (Third-Class Honours)
  <40   → F (Fail)

Withdrawn students are skipped.

Usage:
    python oulad_converter.py --oulad-dir /path/to/oulad/ --output cs_db_oulad.json
"""

import csv
import json
import os
import argparse
from collections import defaultdict, Counter


# =============================================================================
# UK GRADE THRESHOLDS
# =============================================================================

UK_THRESHOLDS = [
    (70, "A"),
    (60, "B"),
    (50, "C"),
    (40, "D"),
    (0,  "F"),
]


def score_to_uk_grade(weighted_avg):
    if weighted_avg is None:
        return None
    for threshold, grade in UK_THRESHOLDS:
        if weighted_avg >= threshold:
            return grade
    return "F"


# =============================================================================
# MAPPINGS
# =============================================================================

ASSESSMENT_TYPE_MAP = {
    "TMA": "homework",
    "CMA": "quiz",
    "Exam": "final",
}


# =============================================================================
# HELPERS
# =============================================================================

def load_csv(filepath):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def day_to_week(day_str):
    try:
        day = int(float(day_str))
    except (ValueError, TypeError):
        return 1
    if day < 0:
        day = 0
    return (day // 7) + 1


# =============================================================================
# CONVERTER
# =============================================================================

def convert_oulad(oulad_dir, output_path):
    print(f"Loading OULAD data from {oulad_dir}...")

    assessments_raw = load_csv(os.path.join(oulad_dir, "assessments.csv"))
    student_assessments_raw = load_csv(os.path.join(oulad_dir, "studentAssessment.csv"))
    student_info_raw = load_csv(os.path.join(oulad_dir, "studentInfo.csv"))
    courses_raw = load_csv(os.path.join(oulad_dir, "courses.csv"))
    vle_raw = load_csv(os.path.join(oulad_dir, "vle.csv"))
    student_vle_raw = load_csv(os.path.join(oulad_dir, "studentVle.csv"))

    print(f"  Assessments: {len(assessments_raw)}")
    print(f"  Student submissions: {len(student_assessments_raw)}")
    print(f"  Student records: {len(student_info_raw)}")
    print(f"  Courses: {len(courses_raw)}")
    print(f"  VLE materials: {len(vle_raw)}")
    print(f"  Student VLE interactions: {len(student_vle_raw)}")

    # =========================================================================
    # Step 1: Build assessment lookup
    # =========================================================================
    assessment_lookup = {}
    type_counts = defaultdict(int)
    course_assessments = defaultdict(list)

    for row in assessments_raw:
        aid = row["id_assessment"].strip()
        module = row["code_module"].strip()
        pres = row["code_presentation"].strip()
        atype = row["assessment_type"].strip()
        weight = float(row["weight"]) if row["weight"].strip() else 0.0
        date = row["date"].strip() if row["date"].strip() else None

        course_key = f"{module}_{pres}"
        type_count_key = f"{course_key}_{atype}"
        type_counts[type_count_key] += 1
        name = f"{atype}_{type_counts[type_count_key]}"

        assessment_lookup[aid] = {
            "module": module,
            "presentation": pres,
            "course_key": course_key,
            "type": ASSESSMENT_TYPE_MAP.get(atype, "unknown"),
            "weight": weight,
            "date": date,
            "name": name,
            "week": day_to_week(date) if date else None,
        }

        course_assessments[course_key].append({
            "id": aid,
            "weight": weight,
            "name": name,
            "type": ASSESSMENT_TYPE_MAP.get(atype, "unknown"),
        })

    print(f"\n  Course assessment structure:")
    for ck in sorted(course_assessments.keys()):
        assessments = course_assessments[ck]
        total_weight = sum(a["weight"] for a in assessments)
        names = [f"{a['name']}({a['weight']}%)" for a in assessments]
        print(f"    {ck}: {len(assessments)} assessments, total_weight={total_weight:.0f}%")
        print(f"      {', '.join(names)}")

    # =========================================================================
    # Step 2: Build VLE material lookup
    # =========================================================================
    vle_lookup = {}
    for row in vle_raw:
        site_id = row["id_site"].strip()
        activity_type = row["activity_type"].strip()
        vle_lookup[site_id] = activity_type

    # =========================================================================
    # Step 3: Identify non-withdrawn students
    # =========================================================================
    valid_students = set()
    withdrawn_count = 0

    for row in student_info_raw:
        module = row["code_module"].strip()
        pres = row["code_presentation"].strip()
        sid = row["id_student"].strip()
        result = row["final_result"].strip()

        if result == "Withdrawn":
            withdrawn_count += 1
            continue

        valid_students.add((module, pres, sid))

    print(f"\n  Valid students (non-Withdrawn): {len(valid_students)}")
    print(f"  Withdrawn students (skipped): {withdrawn_count}")

    # =========================================================================
    # Step 4: Build student submission scores
    # student_scores[(module, pres, sid)][assessment_id] = (score, weight)
    # =========================================================================
    student_scores = defaultdict(dict)

    for row in student_assessments_raw:
        aid = row["id_assessment"].strip()
        sid = row["id_student"].strip()
        score_str = row["score"].strip() if row["score"].strip() else None

        if aid not in assessment_lookup:
            continue

        info = assessment_lookup[aid]
        module = info["module"]
        pres = info["presentation"]

        if (module, pres, sid) not in valid_students:
            continue

        if score_str is None:
            continue

        try:
            score = float(score_str)
        except ValueError:
            continue

        student_scores[(module, pres, sid)][aid] = (score, info["weight"])

    print(f"  Students with submissions: {len(student_scores)}")

    # =========================================================================
    # Step 5: Compute UK grades from SUBMITTED assessments only
    # weighted_avg = sum(score_i * weight_i) / sum(weight_i)
    # Only includes assessments the student actually submitted
    # =========================================================================
    student_grades = {}
    grade_details = defaultdict(list)
    no_submission_count = 0

    for (module, pres, sid) in valid_students:
        submissions = student_scores.get((module, pres, sid), {})

        if not submissions:
            no_submission_count += 1
            continue

        total_weighted_score = 0.0
        total_weight = 0.0

        for aid, (score, weight) in submissions.items():
            if weight > 0:
                total_weighted_score += score * weight
                total_weight += weight

        if total_weight <= 0:
            # Student only submitted 0-weight assessments, use raw avg
            scores_only = [s for s, w in submissions.values()]
            if scores_only:
                weighted_avg = sum(scores_only) / len(scores_only)
            else:
                no_submission_count += 1
                continue
        else:
            weighted_avg = total_weighted_score / total_weight

        grade = score_to_uk_grade(weighted_avg)
        student_grades[(module, pres, sid)] = grade

        course_key = f"{module}_{pres}"
        grade_details[course_key].append({
            "sid": sid,
            "weighted_avg": round(weighted_avg, 2),
            "grade": grade,
            "num_submitted": len(submissions),
        })

    # Print grade distribution
    all_grades = list(student_grades.values())
    grade_dist = Counter(all_grades)
    print(f"\n  UK Grade Distribution (from submitted assessments):")
    print(f"    Total graded students: {len(all_grades)}")
    print(f"    No submissions (skipped): {no_submission_count}")
    for g in ["A", "B", "C", "D", "F"]:
        count = grade_dist.get(g, 0)
        pct = count / len(all_grades) * 100 if all_grades else 0
        print(f"    {g}: {count:>5} ({pct:>5.1f}%)")

    # Per-course breakdown
    print(f"\n  Per-course grade distribution:")
    for ck in sorted(grade_details.keys()):
        details = grade_details[ck]
        avgs = [d["weighted_avg"] for d in details]
        grades = [d["grade"] for d in details]
        dist = Counter(grades)
        avg_score = sum(avgs) / len(avgs) if avgs else 0
        avg_submitted = sum(d["num_submitted"] for d in details) / len(details) if details else 0
        dist_str = ", ".join(f"{g}:{dist.get(g, 0)}" for g in ["A", "B", "C", "D", "F"])
        print(f"    {ck}: {len(details)} students, avg_score={avg_score:.1f}, "
              f"avg_submitted={avg_submitted:.1f}, {dist_str}")

    # =========================================================================
    # Step 6: Build student assessment submissions (for weekly data)
    # =========================================================================
    student_submissions = defaultdict(list)

    for row in student_assessments_raw:
        aid = row["id_assessment"].strip()
        sid = row["id_student"].strip()
        score_str = row["score"].strip() if row["score"].strip() else None
        date_submitted = row["date_submitted"].strip() if row["date_submitted"].strip() else None

        if aid not in assessment_lookup:
            continue

        info = assessment_lookup[aid]
        module = info["module"]
        pres = info["presentation"]

        if (module, pres, sid) not in student_grades:
            continue

        if score_str is None:
            continue

        try:
            score = float(score_str)
        except ValueError:
            continue

        if date_submitted:
            week = day_to_week(date_submitted)
        elif info["week"]:
            week = info["week"]
        else:
            week = 1

        student_submissions[(module, pres, sid)].append({
            "name": info["name"],
            "type": info["type"],
            "weight": info["weight"],
            "score": round(score, 2),
            "week": week,
        })

    # =========================================================================
    # Step 7: Build student VLE engagement data
    # =========================================================================
    student_vle = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    vle_loaded = 0
    vle_skipped = 0

    for row in student_vle_raw:
        module = row["code_module"].strip()
        pres = row["code_presentation"].strip()
        sid = row["id_student"].strip()
        site_id = row["id_site"].strip()
        date = row["date"].strip()
        clicks = int(row["sum_click"].strip()) if row["sum_click"].strip() else 0

        if (module, pres, sid) not in student_grades:
            vle_skipped += 1
            continue

        activity_type = vle_lookup.get(site_id, "unknown")
        week = day_to_week(date)

        student_vle[(module, pres, sid)][week][activity_type] += clicks
        vle_loaded += 1

    print(f"\n  VLE interactions loaded: {vle_loaded}")
    print(f"  VLE skipped (no grade): {vle_skipped}")

    # =========================================================================
    # Step 8: Build course length lookup
    # =========================================================================
    course_lengths = {}
    for row in courses_raw:
        module = row["code_module"].strip()
        pres = row["code_presentation"].strip()
        length_days = int(row["module_presentation_length"].strip()) if row["module_presentation_length"].strip() else 0
        course_lengths[f"{module}_{pres}"] = length_days

    # =========================================================================
    # Step 9: Merge assessments + VLE into student records
    # =========================================================================
    courses = defaultdict(lambda: {
        "students": {},
        "module": None,
        "presentation": None,
    })

    all_student_keys = set(student_submissions.keys()) | set(student_vle.keys())
    all_student_keys = {k for k in all_student_keys if k in student_grades}

    for (module, pres, sid) in all_student_keys:
        course_key = f"{module}_{pres}"
        grade = student_grades[(module, pres, sid)]

        if courses[course_key]["module"] is None:
            courses[course_key]["module"] = module
            courses[course_key]["presentation"] = pres

        all_weeks = defaultdict(list)

        for sub in student_submissions.get((module, pres, sid), []):
            all_weeks[sub["week"]].append({
                "name": sub["name"],
                "type": sub["type"],
                "weight": sub["weight"],
                "score": sub["score"],
            })

        vle_data = student_vle.get((module, pres, sid), {})
        for week_num, type_clicks in vle_data.items():
            for activity_type, clicks in type_clicks.items():
                all_weeks[week_num].append({
                    "name": f"vle_{activity_type}_w{week_num}",
                    "type": activity_type,
                    "weight": 0.0,
                    "score": clicks,
                })

        week_records = []
        for week_num in sorted(all_weeks.keys()):
            week_records.append({
                "week": week_num,
                "activities": all_weeks[week_num],
            })

        courses[course_key]["students"][sid] = {
            "student_id": f"oulad_{sid}",
            "final_grade": grade,
            "weeks": week_records,
        }

    # =========================================================================
    # Step 10: Build intervention data
    # =========================================================================
    def build_intervention(course_key):
        length_days = course_lengths.get(course_key, 0)
        length_weeks = max(1, length_days // 7)
        approx_week = max(1, int(length_weeks * 0.6))
        return {
            "atrisk_approx_week": approx_week,
            "total_components": None,
        }

    # =========================================================================
    # Step 11: Build final output
    # =========================================================================
    output = []
    for course_key in sorted(courses.keys()):
        cdata = courses[course_key]
        students_dict = cdata["students"]

        if len(students_dict) < 5:
            print(f"  Skipping {course_key}: only {len(students_dict)} students")
            continue

        students_list = list(students_dict.values())
        intervention = build_intervention(course_key)

        gpa_map = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
        gpas = [gpa_map.get(s["final_grade"], 0.0) for s in students_list]
        avg_gpa = round(sum(gpas) / len(gpas), 2) if gpas else 0.0

        grade_dist = dict(Counter(s["final_grade"] for s in students_list))

        vle_students = 0
        total_vle_activities = 0
        for s in students_list:
            has_vle = False
            for w in s["weeks"]:
                for a in w["activities"]:
                    if a.get("type", "") in {
                        "oucontent", "forumng", "homepage", "ouwiki",
                        "resource", "subpage", "url", "glossary",
                        "dataplus", "ouelluminate", "questionnaire",
                        "page", "dualpane", "folder", "htmlactivity",
                        "oucollaborate", "repeatactivity", "sharedsubpage",
                    }:
                        has_vle = True
                        total_vle_activities += 1
            if has_vle:
                vle_students += 1

        output.append({
            "course_id": course_key,
            "course_info": {
                "department": "oulad",
                "level": cdata["module"],
                "term": cdata["presentation"],
                "average_gpa": avg_gpa,
                "total_students": len(students_list),
                "grade_distribution": grade_dist,
                "vle_students": vle_students,
                "vle_activities": total_vle_activities,
            },
            "intervention": intervention,
            "students": students_list,
        })

    output.sort(key=lambda c: c["course_id"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*50}")
    total_students = sum(len(c["students"]) for c in output)
    total_vle = sum(c["course_info"].get("vle_students", 0) for c in output)
    print(f"  Courses: {len(output)}")
    print(f"  Total students: {total_students}")
    print(f"  Students with VLE data: {total_vle}")
    for c in output:
        n = len(c["students"])
        gi = c["course_info"]["grade_distribution"]
        vle_n = c["course_info"].get("vle_students", 0)
        dist_str = ", ".join(f"{g}:{gi.get(g, 0)}" for g in ["A", "B", "C", "D", "F"])
        print(f"  {c['course_id']}: {n} students, {dist_str}, VLE: {vle_n}/{n}")
    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert OULAD to PredAct format (UK grading)")
    parser.add_argument("--oulad-dir", required=True, help="Directory containing OULAD CSVs")
    parser.add_argument("--output", default="cs_db_oulad.json", help="Output path")
    args = parser.parse_args()
    convert_oulad(args.oulad_dir, args.output)


if __name__ == "__main__":
    main()