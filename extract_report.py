"""
PredAct - Report Extractor
Extracts a clean report JSON from a dialogue entry in data.json.
Also builds a student grades lookup from the grades log file
so the system can answer questions about ANY student.

Outputs:
  1. Report JSON (class overview, risk groups, intervention)
  2. Grades lookup JSON (all students' per-assignment scores)

Usage:
    python extract_report.py --data results/dataset/data_test.json --dlg DLG_0072.json --grades results/dataset/logs/DLG_0072_grades.json --output reports/week5_report.json
"""

import json
import argparse
import os
import sys

# Add parent dir so we can import tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import build_grades_lookup


def extract_report(data, dlg_id):
    """
    Extract a structured report from a dialogue entry.
    """
    if dlg_id not in data:
        print(f"ERROR: {dlg_id} not found in data")
        return None

    entry = data[dlg_id]
    goal = entry.get("goal", {})
    log = entry.get("log", [])

    # Get final belief state
    final_meta = {}
    for turn in reversed(log):
        if turn.get("metadata") and turn["metadata"] != {}:
            final_meta = turn["metadata"]
            break

    if not final_meta:
        print(f"ERROR: No belief state found in {dlg_id}")
        return None

    class_ctx = final_meta.get("class_context", {})
    class_summary = final_meta.get("class_summary", {})
    student_status = final_meta.get("student_status", {})
    intervention = final_meta.get("intervention", {})

    # Build class overview
    report = {
        "dialogue_id": dlg_id,
        "class_overview": {
            "course_name": goal.get("class_context", {}).get("course_name", "?"),
            "course_department": class_ctx.get("course_department", "?"),
            "course_level": class_ctx.get("course_level", "?"),
            "term": class_ctx.get("term", "?"),
            "week": class_ctx.get("week", "?"),
            "total_students": goal.get("student_count", 0),
            "average_gpa": class_summary.get("average_gpa", "?"),
            "grade_trend": class_summary.get("grade_trend", "?"),
            "common_issue": class_summary.get("common_assignment_type_issue", "none"),
            "flagged_student_count": class_summary.get("flagged_student_count", 0),
        },
        "risk_groups": {},
        "intervention": intervention,
    }

    # Build risk groups with per-student details
    for risk_key, group in student_status.items():
        if risk_key == "no_risk":
            report["risk_groups"]["no_risk"] = {
                "count": group.get("count", 0),
                "predicted_grade": group.get("predicted_grade", "?"),
            }
            continue

        flagged_details = []
        student_ids = group.get("student_ids", [])
        reasons = group.get("failure_risk_reasons", {})
        missing = group.get("missing_assignments", {})

        for sid in student_ids:
            student_entry = {
                "student_id": sid,
                "predicted_grade": group.get("per_student_grades", {}).get(sid, group.get("predicted_grade", "?")),
                "failure_risk": group.get("failure_risk", "?"),
                "failure_risk_reason": reasons.get(sid, "unknown"),
                "missing_assignments": missing.get(sid, 0),
            }
            flagged_details.append(student_entry)

        report["risk_groups"][risk_key] = {
            "count": group.get("count", 0),
            "predicted_grade": group.get("predicted_grade", "?"),
            "failure_risk": group.get("failure_risk", "?"),
            "common_grade_trend": group.get("common_grade_trend", "?"),
            "common_assignment_type": group.get("common_assignment_type", "?"),
            "students": flagged_details,
        }

    return report


def main():
    parser = argparse.ArgumentParser(description="Extract PredAct report from dialogue")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--dlg", required=True, help="Dialogue ID (e.g. DLG_0072.json)")
    parser.add_argument("--grades", default=None, help="Path to grades log file (e.g. DLG_0072_grades.json)")
    parser.add_argument("--output", default=None, help="Output path for report JSON")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract report
    report = extract_report(data, args.dlg)
    if report is None:
        return

    # Build and save grades lookup if grades file provided
    if args.grades and os.path.exists(args.grades):
        print(f"Loading grades from {args.grades}...")
        with open(args.grades, "r", encoding="utf-8") as f:
            grades_data = json.load(f)

        grades_lookup = build_grades_lookup(grades_data)
        print(f"Built grades lookup for {len(grades_lookup)} students")

        # Save grades lookup alongside the report
        if args.output:
            grades_output = args.output.replace("_report.json", "_grades_lookup.json")
            os.makedirs(os.path.dirname(grades_output), exist_ok=True)
            with open(grades_output, "w", encoding="utf-8") as f:
                json.dump(grades_lookup, f, indent=2, ensure_ascii=False)
            print(f"Grades lookup saved to {grades_output}")
    else:
        print("No grades file provided — grades lookup not built")

    # Print summary
    overview = report["class_overview"]
    print(f"\n=== REPORT: {overview['course_name']} {overview['week']} ===")
    print(f"  Students: {overview['total_students']}")
    print(f"  Avg GPA: {overview['average_gpa']}")
    print(f"  Trend: {overview['grade_trend']}")
    print(f"  Flagged: {overview['flagged_student_count']}")
    print(f"  Common issue: {overview['common_issue']}")

    print(f"\n  Risk Groups:")
    for risk_key, group in report["risk_groups"].items():
        if risk_key == "no_risk":
            print(f"    no_risk: {group['count']} students, grade={group['predicted_grade']}")
        else:
            print(f"    {risk_key}: {group['count']} students, grade={group['predicted_grade']}, risk={group['failure_risk']}")
            for s in group.get("students", []):
                print(f"      - {s['student_id']}: reason={s['failure_risk_reason']}, missing={s['missing_assignments']}")

    print(f"\n  Intervention: {list(report['intervention'].keys())}")

    # Save report
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()