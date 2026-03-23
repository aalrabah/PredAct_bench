"""
Convert student records CSV + intervention timing CSV into a consolidated
MultiWOZ-style JSON file, grouped by course.

Usage:
    python convert_to_json.py --students students.csv --interventions interventions.csv --output db.json
"""

import argparse
import csv
import json
import re
from collections import defaultdict


def parse_students(filepath):
    """Parse the wide-format student CSV into structured records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            student_id = row["student_id"].strip()
            course_id = row["course_id"].strip()

            # Course-level info
            course_info = {}
            for key, json_key in [
                ("course_avg_gpa", "avg_gpa"),
                ("course_pct_A", "pct_A"),
                ("course_pct_B", "pct_B"),
                ("course_pct_C", "pct_C"),
                ("course_pct_D", "pct_D"),
                ("course_pct_F", "pct_F"),
                ("grading_scale_A", "grading_scale_A"),
            ]:
                val = row.get(key, "").strip()
                if val:
                    course_info[json_key] = float(val)

            # Parse weekly slots
            weeks = []
            for week_num in range(1, 17):
                activities = []
                for slot_num in range(1, 8):
                    prefix = f"week_{week_num}_slot_{slot_num}"
                    name = row.get(f"{prefix}_name", "").strip()
                    if not name:
                        continue
                    activity = {"name": name}
                    atype = row.get(f"{prefix}_type", "").strip()
                    if atype:
                        activity["type"] = atype
                    weight = row.get(f"{prefix}_weight", "").strip()
                    if weight:
                        activity["weight"] = float(weight)
                    score = row.get(f"{prefix}_score", "").strip()
                    if score:
                        activity["score"] = float(score)
                    activities.append(activity)
                if activities:
                    weeks.append({"week": week_num, "activities": activities})

            final_grade = row.get("final_grade", "").strip() or None

            records.append({
                "student_id": student_id,
                "course_id": course_id,
                "course_info": course_info,
                "weeks": weeks,
                "final_grade": final_grade,
            })
    return records


def parse_interventions(filepath):
    """Parse the intervention timing CSV into a dict keyed by course_id."""
    interventions = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            course_id = row["course_id"].strip()

            total = row.get("total_components", "").strip()
            approx_week = row.get("atrisk_approx_week", "").strip()
            components_raw = row.get("atrisk_components", "").strip()

            # Skip courses with no intervention data
            if not approx_week:
                interventions[course_id] = None
                continue

            components = [c.strip() for c in components_raw.split(",") if c.strip()] if components_raw else []

            interventions[course_id] = {
                "total_components": int(total) if total else None,
                "atrisk_approx_week": int(approx_week),
                "atrisk_components": components,
            }
    return interventions


def build_db(students_path, interventions_path):
    """Merge both datasets into course-grouped JSON."""
    records = parse_students(students_path)
    interventions = parse_interventions(interventions_path)

    # Group students by course
    courses = defaultdict(lambda: {"course_info": {}, "students": []})

    for rec in records:
        cid = rec["course_id"]
        # Set course_info from first student seen (identical across students)
        if not courses[cid]["course_info"]:
            courses[cid]["course_info"] = rec["course_info"]

        courses[cid]["students"].append({
            "student_id": rec["student_id"],
            "weeks": rec["weeks"],
            "final_grade": rec["final_grade"],
        })

    # Build final output
    output = []
    for course_id, data in sorted(courses.items()):
        entry = {
            "course_id": course_id,
            "course_info": data["course_info"],
            "intervention": interventions.get(course_id, None),
            "students": data["students"],
        }
        output.append(entry)

    return output


def main():
    parser = argparse.ArgumentParser(description="Convert CSVs to MultiWOZ-style JSON")
    parser.add_argument("--students", required=True, help="Path to student records CSV (tab-separated)")
    parser.add_argument("--interventions", required=True, help="Path to intervention timing CSV (tab-separated)")
    parser.add_argument("--output", default="db.json", help="Output JSON path")
    args = parser.parse_args()

    db = build_db(args.students, args.interventions)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    # Summary
    total_students = sum(len(c["students"]) for c in db)
    print(f"Done! {len(db)} courses, {total_students} student records → {args.output}")


if __name__ == "__main__":
    main()

    