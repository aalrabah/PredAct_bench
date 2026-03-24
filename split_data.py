"""
PredAct Benchmark - Data Splitter
Splits cs_db_original.json into:
  1. cs_db.json         — historical knowledge base (80% of students per course)
  2. logs/              — truncated test student records at varying week cutoffs
  3. ground_truth.json  — full records of test students for evaluation

Usage:
    python split_data.py --input cs_db_original.json --output-dir .
"""

import json
import os
import argparse
import random
from collections import defaultdict
from config import RISK_MAPPING


# Week cutoffs for temporal diversity
CUTOFF_WEEKS = [3, 5, 7, 9, 11, 13]

# Train/test split ratio
TRAIN_RATIO = 0.8

# Random seed for reproducibility
SEED = 42


def truncate_student(student, cutoff_week):
    """
    Remove all week data after cutoff_week.
    Returns truncated student record (without final_grade).
    """
    truncated_weeks = []
    for week_data in student.get("weeks", []):
        if week_data["week"] <= cutoff_week:
            truncated_weeks.append(week_data)

    return {
        "student_id": student["student_id"],
        "weeks": truncated_weeks,
    }


def split_data(input_path, output_dir):
    """Run the full split pipeline."""
    random.seed(SEED)

    # Load original data
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        original_db = json.load(f)

    print(f"  Found {len(original_db)} courses")

    # Output containers
    historical_db = []
    ground_truth = {}
    dialogue_counter = 0

    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    stats = {
        "total_courses": len(original_db),
        "total_students": 0,
        "historical_students": 0,
        "test_students": 0,
        "dialogues_created": 0,
    }

    for course in original_db:
        course_id = course["course_id"]
        course_info = course.get("course_info", {})
        intervention = course.get("intervention", None)
        students = course.get("students", [])

        stats["total_students"] += len(students)

        if len(students) < 5:
            # Too few students to split meaningfully, keep all as historical
            print(f"  {course_id}: only {len(students)} students, keeping all as historical")
            historical_db.append(course)
            stats["historical_students"] += len(students)
            continue

        # Shuffle and split
        shuffled = list(students)
        random.shuffle(shuffled)

        split_idx = max(1, int(len(shuffled) * TRAIN_RATIO))
        train_students = shuffled[:split_idx]
        test_students = shuffled[split_idx:]

        # Ensure at least 1 test student
        if not test_students:
            test_students = [train_students.pop()]

        print(f"  {course_id}: {len(train_students)} historical, {len(test_students)} test")
        stats["historical_students"] += len(train_students)
        stats["test_students"] += len(test_students)

        # Add train students to historical db
        historical_db.append({
            "course_id": course_id,
            "course_info": course_info,
            "intervention": intervention,
            "students": train_students,
        })

        # Create log files at each cutoff week
        for cutoff in CUTOFF_WEEKS:
            # Check if any test student has data at or before this cutoff
            truncated_students = []
            for student in test_students:
                truncated = truncate_student(student, cutoff)
                # Only include if they have at least 1 activity after truncation
                if truncated["weeks"]:
                    truncated_students.append(truncated)

            if not truncated_students:
                # No students have data at this cutoff, skip
                continue

            dialogue_counter += 1
            dlg_id = f"DLG_{dialogue_counter:04d}"

            # Write log file
            log_data = {
                "course_id": course_id,
                "cutoff_week": cutoff,
                "students": truncated_students,
            }
            log_path = os.path.join(logs_dir, f"{dlg_id}_grades.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            # Add to ground truth
            gt_students = {}
            for student in test_students:
                sid = student["student_id"]
                # Check if this student was included in the truncated set
                if any(t["student_id"] == sid for t in truncated_students):
                    gt_students[sid] = {
                        "final_grade": student.get("final_grade", "unknown"),
                        "full_weeks": student.get("weeks", []),
                    }

            # Check if intervention should be triggered:
            # 1. Course has intervention data
            # 2. We're at or past the threshold week
            # 3. At least one student's actual grade maps to a risk level
            week_threshold_met = (
                intervention is not None
                and intervention.get("atrisk_approx_week") is not None
                and cutoff >= intervention.get("atrisk_approx_week", 999)
            )
            has_at_risk_students = any(
                RISK_MAPPING.get(info["final_grade"].lower()) is not None
                for info in gt_students.values()
            )

            ground_truth[f"{dlg_id}.json"] = {
                "course_id": course_id,
                "cutoff_week": cutoff,
                "student_grades": {
                    sid: info["final_grade"]
                    for sid, info in gt_students.items()
                },
                "full_student_records": gt_students,
                "intervention_triggered": week_threshold_met and has_at_risk_students,
            }

            stats["dialogues_created"] += 1

    # Write historical db
    db_path = os.path.join(output_dir, "cs_db.json")
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(historical_db, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {db_path} ({stats['historical_students']} students)")

    # Write ground truth
    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    print(f"Wrote {gt_path} ({len(ground_truth)} dialogue entries)")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"SPLIT SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total courses:      {stats['total_courses']}")
    print(f"  Total students:     {stats['total_students']}")
    print(f"  Historical (train): {stats['historical_students']}")
    print(f"  Test (held out):    {stats['test_students']}")
    print(f"  Dialogues created:  {stats['dialogues_created']}")
    print(f"  Cutoff weeks used:  {CUTOFF_WEEKS}")
    print(f"\nOutput files:")
    print(f"  {db_path}")
    print(f"  {gt_path}")
    print(f"  {logs_dir}/ ({stats['dialogues_created']} files)")


def main():
    parser = argparse.ArgumentParser(description="Split cs_db into train/test")
    parser.add_argument("--input", required=True, help="Path to original cs_db.json")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    split_data(args.input, args.output_dir)

if __name__ == "__main__":
    main()