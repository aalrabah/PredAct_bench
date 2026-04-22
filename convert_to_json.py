"""
Convert student records CSV into a consolidated JSON file, grouped by course.

Removed: intervention timing CSV (interventions are no longer hardcoded rules;
         the instructor decides intervention in the human study, and the new
         agent-to-agent pipeline does not use intervention timing).

Usage:
    python convert_to_json.py --students synthetic_students.csv
    python convert_to_json.py --students synthetic_students.csv --output /custom/path/cs_db.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict


DEFAULT_OUTPUT_DIR = "/home/alrabah2/PredAct_bench/results/uiuc"


def parse_students(filepath):
    """Parse the wide-format student CSV into structured records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            student_id = row["student_id"].strip()
            course_id = row["course_id"].strip()

            # Course-level info (same across all rows of a course)
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

            # Parse weekly slots (up to 16 weeks, up to 7 slots per week)
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


def build_db(students_path):
    """Merge student records into course-grouped JSON."""
    records = parse_students(students_path)

    # Group students by course
    courses = defaultdict(lambda: {"course_info": {}, "students": []})

    for rec in records:
        cid = rec["course_id"]
        if not courses[cid]["course_info"]:
            courses[cid]["course_info"] = rec["course_info"]

        courses[cid]["students"].append({
            "student_id": rec["student_id"],
            "weeks": rec["weeks"],
            "final_grade": rec["final_grade"],
        })

    output = []
    for course_id, data in sorted(courses.items()):
        entry = {
            "course_id": course_id,
            "course_info": data["course_info"],
            "students": data["students"],
        }
        output.append(entry)

    return output


def main():
    parser = argparse.ArgumentParser(description="Convert student CSV to JSON grouped by course")
    parser.add_argument("--students", required=True, help="Path to student records CSV")
    parser.add_argument(
        "--output",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "cs_db.json"),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_DIR}/cs_db.json)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    db = build_db(args.students)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    total_students = sum(len(c["students"]) for c in db)
    print(f"Done! {len(db)} courses, {total_students} student records -> {args.output}")


if __name__ == "__main__":
    main()