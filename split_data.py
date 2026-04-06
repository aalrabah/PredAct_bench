"""
PredAct Benchmark - Train/Test Splitter

Splits cs_db.json into:
  1. cs_db_train.json — historical students only (for matching)
  2. logs/*.json — grade log files for unseen students (for prediction)
  3. ground_truth.json — ground truth for evaluation

The split ensures NO overlap between historical and unseen students.

For each course, creates multiple dialogues at different cutoff weeks
(early, mid, late) to test temporal generalization.

Usage:
    python split_data.py --db cs_db.json --output-dir split_output/ --test-ratio 0.2 --seed 42
"""

import json
import os
import random
import argparse
import math
from collections import defaultdict, Counter


def get_student_max_week(student):
    """Get the last week with data for a student."""
    max_week = 0
    for week_data in student.get("weeks", []):
        if week_data["week"] > max_week:
            max_week = week_data["week"]
    return max_week


def get_student_components(student):
    """Get all unique component names for a student."""
    components = set()
    for week_data in student.get("weeks", []):
        for activity in week_data.get("activities", []):
            components.add(activity["name"])
    return components


def truncate_student_to_week(student, cutoff_week):
    """
    Create a copy of the student record with only data up to cutoff_week.
    This simulates what we'd know at that point in the semester.
    """
    truncated_weeks = []
    for week_data in student.get("weeks", []):
        if week_data["week"] <= cutoff_week:
            truncated_weeks.append(week_data)

    return {
        "student_id": student["student_id"],
        "weeks": truncated_weeks,
        # NOTE: final_grade is NOT included in the truncated version
        # (it's what we're trying to predict)
    }


def compute_cutoff_weeks(course_students, intervention_data):
    """
    Determine cutoff weeks for early/mid/late dialogues.

    Strategy:
    - Find the overall week range from all students
    - Early: ~15-20% through the course
    - Mid: ~40-50% through
    - Late: multiple points from 60% onward

    Also uses intervention week if available.
    """
    all_max_weeks = [get_student_max_week(s) for s in course_students]
    if not all_max_weeks:
        return []

    course_max_week = max(all_max_weeks)
    if course_max_week < 4:
        # Too short for meaningful splits
        return [max(1, course_max_week)]

    cutoffs = set()

    # Early: week 2 or ~15% through
    early = max(2, int(course_max_week * 0.15))
    cutoffs.add(early)

    # Mid: ~40% through
    mid = max(early + 2, int(course_max_week * 0.40))
    cutoffs.add(mid)

    # Late: multiple points
    for pct in [0.60, 0.75, 0.90]:
        late = int(course_max_week * pct)
        if late > mid:
            cutoffs.add(late)

    # Add intervention week if available
    if intervention_data:
        atrisk_week = intervention_data.get("atrisk_approx_week")
        if atrisk_week and atrisk_week > early:
            cutoffs.add(atrisk_week)

    # Add a very late cutoff (near end)
    final = max(cutoffs) + max(2, int(course_max_week * 0.1))
    if final <= course_max_week:
        cutoffs.add(final)

    return sorted(cutoffs)


def split_course(course_data, test_ratio, rng):
    """
    Split a single course's students into train and test sets.

    Returns (train_students, test_students)
    """
    students = course_data.get("students", [])

    # Filter students with valid grades
    valid_students = [s for s in students if s.get("final_grade") not in (None, "unknown", "")]

    if len(valid_students) < 10:
        # Too few students — use all for training, no test
        return valid_students, []

    # Stratified split: maintain grade distribution
    grade_groups = defaultdict(list)
    for s in valid_students:
        grade_groups[s["final_grade"]].append(s)

    train = []
    test = []

    for grade, group in grade_groups.items():
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio))

        # Ensure at least some remain for training
        if len(group) - n_test < 2:
            n_test = max(0, len(group) - 2)

        test.extend(group[:n_test])
        train.extend(group[n_test:])

    return train, test


def build_grade_log(course_id, test_students, cutoff_week):
    """
    Build a grade log file for a batch of test students at a given cutoff week.
    Each student's record is truncated to only include data up to cutoff_week.
    """
    truncated = []
    for student in test_students:
        # Only include students who have data at or before the cutoff
        truncated_record = truncate_student_to_week(student, cutoff_week)
        if truncated_record["weeks"]:  # has at least some data
            truncated.append(truncated_record)

    if not truncated:
        return None

    return {
        "course_id": course_id,
        "students": truncated,
    }


def build_ground_truth_entry(course_id, test_students, cutoff_week, dlg_id, intervention_data):
    """
    Build ground truth for one dialogue.
    Format matches what evaluate.py expects:
      - student_grades: dict {student_id: grade}
      - intervention_triggered: bool
      - cutoff_week: int
      - course_id: str
    """
    student_grades = {}
    full_student_records = {}

    for student in test_students:
        # Check student has data before cutoff
        has_data = any(w["week"] <= cutoff_week for w in student.get("weeks", []))
        if not has_data:
            continue

        sid = student["student_id"]
        grade = student.get("final_grade", "unknown")
        if grade and grade != "unknown":
            student_grades[sid] = grade.lower()
            full_student_records[sid] = {
                "final_grade": grade.lower(),
                "full_weeks": student.get("weeks", []),
            }

    if not student_grades:
        return None

    # Determine if intervention should be triggered at this cutoff
    intervention_triggered = False
    if intervention_data:
        approx_week = intervention_data.get("atrisk_approx_week")
        if approx_week is not None and cutoff_week >= approx_week:
            intervention_triggered = True

    return {
        "course_id": course_id,
        "cutoff_week": cutoff_week,
        "student_grades": student_grades,
        "full_student_records": full_student_records,
        "intervention_triggered": intervention_triggered,
    }


def main():
    parser = argparse.ArgumentParser(description="Split PredAct data into train/test")
    parser.add_argument("--db", required=True, help="Path to cs_db.json (combined UIUC + OULAD)")
    parser.add_argument("--output-dir", default="split_output", help="Output directory")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction for test (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load database
    print(f"Loading {args.db}...")
    with open(args.db, "r", encoding="utf-8") as f:
        db = json.load(f)
    print(f"  {len(db)} courses loaded")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Process each course
    train_db = []
    all_ground_truth = {}
    dlg_counter = 0
    total_train = 0
    total_test = 0

    stats = {
        "courses_with_test": 0,
        "courses_skipped": 0,
        "dialogues_created": 0,
    }

    for course_data in db:
        course_id = course_data["course_id"]
        intervention = course_data.get("intervention")

        # Split students
        train_students, test_students = split_course(course_data, args.test_ratio, rng)

        total_train += len(train_students)
        total_test += len(test_students)

        # Build training DB entry (historical students only)
        train_entry = {
            "course_id": course_id,
            "course_info": course_data.get("course_info", {}),
            "intervention": intervention,
            "students": train_students,
        }
        train_db.append(train_entry)

        if not test_students:
            stats["courses_skipped"] += 1
            print(f"  {course_id}: {len(train_students)} train, 0 test (skipped)")
            continue

        stats["courses_with_test"] += 1

        # Determine cutoff weeks
        cutoff_weeks = compute_cutoff_weeks(test_students, intervention)

        if not cutoff_weeks:
            print(f"  {course_id}: {len(train_students)} train, {len(test_students)} test, no valid cutoffs")
            continue

        # Create grade log files at each cutoff
        course_dlgs = 0
        for cutoff_week in cutoff_weeks:
            dlg_counter += 1
            dlg_id = f"DLG_{dlg_counter:04d}.json"

            # Build grade log (truncated student data)
            grade_log = build_grade_log(course_id, test_students, cutoff_week)
            if grade_log is None:
                dlg_counter -= 1
                continue

            # Save grade log
            log_filename = dlg_id.replace(".json", "_grades.json")
            log_path = os.path.join(logs_dir, log_filename)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(grade_log, f, indent=2, ensure_ascii=False)

            # Build ground truth
            gt = build_ground_truth_entry(course_id, test_students, cutoff_week, dlg_id, intervention)
            if gt:
                all_ground_truth[dlg_id] = gt

            course_dlgs += 1
            stats["dialogues_created"] += 1

        print(f"  {course_id}: {len(train_students)} train, {len(test_students)} test, "
              f"{len(cutoff_weeks)} cutoffs → {course_dlgs} dialogues")

    # Save training DB
    train_db_path = os.path.join(args.output_dir, "cs_db.json")
    with open(train_db_path, "w", encoding="utf-8") as f:
        json.dump(train_db, f, indent=2, ensure_ascii=False)

    # Save ground truth
    gt_path = os.path.join(args.output_dir, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(all_ground_truth, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Seed: {args.seed}")
    print(f"  Test ratio: {args.test_ratio}")
    print(f"  Total courses: {len(db)}")
    print(f"  Courses with test data: {stats['courses_with_test']}")
    print(f"  Courses skipped (too few): {stats['courses_skipped']}")
    print(f"  Total train students: {total_train}")
    print(f"  Total test students: {total_test}")
    print(f"  Total dialogues: {stats['dialogues_created']}")
    print(f"\nOutputs:")
    print(f"  Training DB:    {train_db_path}")
    print(f"  Grade logs:     {logs_dir}/ ({stats['dialogues_created']} files)")
    print(f"  Ground truth:   {gt_path}")
    print(f"\nUsage:")
    print(f"  1. Point config.py CS_DB_PATH to: {train_db_path}")
    print(f"  2. Point config.py LOGS_DIR to:   {logs_dir}")
    print(f"  3. Run orchestrator.py")
    print(f"  4. Evaluate with: python evaluate.py --ground-truth {gt_path}")


if __name__ == "__main__":
    main()