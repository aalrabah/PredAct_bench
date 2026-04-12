"""
PredAct - Report Extractor
Extracts a clean report JSON from a dialogue entry in data.json,
optionally enriched with flagged student SQL submission data.

Usage:
    python extract_report.py --data results/uiuc/data_test.json --dlg DLG_0072.json --student-data CS-411/flagged_students/week5_flagged_students.json --output CS-411/reports/week5_report.json
"""

import json
import argparse
import os


def extract_report(data, dlg_id, student_data=None):
    """
    Extract a structured report from a dialogue entry.

    Args:
        data: full data.json dict
        dlg_id: dialogue ID (e.g. "DLG_0072.json")
        student_data: optional list of flagged student SQL submissions

    Returns:
        clean report dict
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

            # Enrich with SQL submission data if available
            if student_data:
                sql_match = _find_student_sql_data(sid, student_data)
                if sql_match:
                    student_entry["sql_submissions"] = sql_match

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


def _find_student_sql_data(synthetic_id, student_data):
    """
    Find SQL submission data for a synthetic student ID.
    Uses position-based mapping: first flagged synthetic ID maps to
    first real student in student_data, etc.

    This is a placeholder — the real mapping file will be created next.
    Returns None if no mapping exists yet.
    """
    # For now, return None. Mapping will be added later.
    return None


def load_student_data(path):
    """Load flagged student SQL submission data."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Extract PredAct report from dialogue")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--dlg", required=True, help="Dialogue ID (e.g. DLG_0072.json)")
    parser.add_argument("--student-data", default=None, help="Path to flagged student SQL data")
    parser.add_argument("--output", default=None, help="Output path for report JSON")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load student data if provided
    student_data = load_student_data(args.student_data)
    if student_data:
        print(f"Loaded {len(student_data)} flagged student records")

    # Extract report
    report = extract_report(data, args.dlg, student_data)
    if report is None:
        return

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

    # Save if output specified
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()