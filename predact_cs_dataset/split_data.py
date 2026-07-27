"""
PredAct Benchmark - Train/Test Splitter

Splits cs_db.json into:
  1. cs_db_train.json                  - training students only (for k-NN reference)
  2. test_sets/{course}_week{N}.json   - truncated test students at each cutoff week
  3. ground_truth_for_cutoff_data.json - answer key (final grades) for every test set

Everything saves to <project-root>/results/predact_cs by default.

The split ensures NO overlap between training and test students, so k-NN
cannot leak by matching a test student to themselves.

Removed: intervention_triggered logic (instructor decides intervention,
         not a hardcoded rule).

Usage:
    python split_data.py
    python split_data.py --db /custom/cs_db.json --output-dir /custom/out
    python split_data.py --test-ratio 0.2 --seed 42
"""

import json
import os
import random
import argparse
from collections import defaultdict, Counter


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "results", "predact_cs")
DEFAULT_DB = os.path.join(DEFAULT_DIR, "cs_db.json")


def get_student_max_week(student):
    """Last week with any data for this student."""
    max_week = 0
    for week_data in student.get("weeks", []):
        if week_data["week"] > max_week:
            max_week = week_data["week"]
    return max_week


def truncate_student_to_week(student, cutoff_week):
    """
    Return a copy of the student with only data up to cutoff_week.
    final_grade is NOT included - that's what we're predicting.
    """
    truncated_weeks = [
        w for w in student.get("weeks", []) if w["week"] <= cutoff_week
    ]
    return {
        "student_id": student["student_id"],
        "weeks": truncated_weeks,
    }


def compute_cutoff_weeks(course_students):
    """
    Return every week from 1 to the course's last recorded week.
    Full granularity lets you plot the accuracy curve week-by-week.
    """
    all_max_weeks = [get_student_max_week(s) for s in course_students]
    if not all_max_weeks:
        return []

    course_max_week = max(all_max_weeks)
    if course_max_week < 1:
        return []

    # Take every week from 1 to course_max_week.
    # This gives full granularity for the noise/accuracy curve.
    return list(range(1, course_max_week + 1))


def split_course(course_data, test_ratio, rng):
    """
    Stratified split by final grade (keeps grade distribution balanced
    between train and test). Returns (train_students, test_students).
    """
    students = course_data.get("students", [])
    valid = [
        s for s in students
        if s.get("final_grade") not in (None, "unknown", "")
    ]

    if len(valid) < 10:
        return valid, []

    grade_groups = defaultdict(list)
    for s in valid:
        grade_groups[s["final_grade"]].append(s)

    train, test = [], []
    for grade, group in grade_groups.items():
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio))
        # Leave at least 2 in train per grade so k-NN has neighbors
        if len(group) - n_test < 2:
            n_test = max(0, len(group) - 2)

        test.extend(group[:n_test])
        train.extend(group[n_test:])

    return train, test


def safe_filename(course_id):
    """Sanitize course_id for use in file names."""
    return course_id.replace(" ", "").replace("/", "_").replace("\\", "_")


def build_test_set(course_id, test_students, cutoff_week):
    """
    Build one test set: truncated student records at cutoff_week.
    Course_id and cutoff_week included so the file is self-describing.
    """
    truncated = []
    for student in test_students:
        rec = truncate_student_to_week(student, cutoff_week)
        if rec["weeks"]:
            truncated.append(rec)

    if not truncated:
        return None

    return {
        "course_id": course_id,
        "cutoff_week": cutoff_week,
        "students": truncated,
    }


def build_ground_truth_entry(course_id, test_students, cutoff_week):
    """
    Build answer key for one test set.
    """
    student_grades = {}
    for student in test_students:
        has_data = any(w["week"] <= cutoff_week for w in student.get("weeks", []))
        if not has_data:
            continue
        sid = student["student_id"]
        grade = student.get("final_grade", "unknown")
        if grade and grade != "unknown":
            student_grades[sid] = grade.lower()

    if not student_grades:
        return None

    return {
        "course_id": course_id,
        "cutoff_week": cutoff_week,
        "student_grades": student_grades,
    }


def main():
    parser = argparse.ArgumentParser(description="Split cs_db.json into train + test sets")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"Path to cs_db.json (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DIR,
        help=f"Output directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction for test (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.db}...")
    with open(args.db, "r", encoding="utf-8") as f:
        db = json.load(f)
    print(f"  {len(db)} courses loaded")

    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    test_sets_dir = os.path.join(args.output_dir, "test_sets")
    os.makedirs(test_sets_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "cs_db_train.json")
    gt_path = os.path.join(args.output_dir, "ground_truth_for_cutoff_data.json")

    train_db = []
    all_ground_truth = {}
    total_train = 0
    total_test = 0

    stats = {
        "courses_with_test": 0,
        "courses_skipped": 0,
        "test_sets_created": 0,
    }

    for course_data in db:
        course_id = course_data["course_id"]
        cid_safe = safe_filename(course_id)

        train_students, test_students = split_course(course_data, args.test_ratio, rng)

        total_train += len(train_students)
        total_test += len(test_students)

        # Training DB entry (no intervention, no test students)
        train_entry = {
            "course_id": course_id,
            "course_info": course_data.get("course_info", {}),
            "students": train_students,
        }
        train_db.append(train_entry)

        if not test_students:
            stats["courses_skipped"] += 1
            print(f"  {course_id}: {len(train_students)} train, 0 test (skipped)")
            continue

        stats["courses_with_test"] += 1

        cutoff_weeks = compute_cutoff_weeks(test_students)
        if not cutoff_weeks:
            print(f"  {course_id}: {len(train_students)} train, {len(test_students)} test, no valid cutoffs")
            continue

        course_test_sets = 0
        for cutoff_week in cutoff_weeks:
            test_set = build_test_set(course_id, test_students, cutoff_week)
            if test_set is None:
                continue

            # File name: e.g. C2-04_week4.json
            test_filename = f"{cid_safe}_week{cutoff_week}.json"
            test_path = os.path.join(test_sets_dir, test_filename)
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump(test_set, f, indent=2, ensure_ascii=False)

            gt = build_ground_truth_entry(course_id, test_students, cutoff_week)
            if gt:
                all_ground_truth[test_filename] = gt

            course_test_sets += 1
            stats["test_sets_created"] += 1

        print(f"  {course_id}: {len(train_students)} train, {len(test_students)} test, "
              f"{len(cutoff_weeks)} cutoffs -> {course_test_sets} test sets")

    # Save training DB
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_db, f, indent=2, ensure_ascii=False)

    # Save ground truth
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(all_ground_truth, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SPLIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Seed:                  {args.seed}")
    print(f"  Test ratio:            {args.test_ratio}")
    print(f"  Total courses:         {len(db)}")
    print(f"  Courses with test:     {stats['courses_with_test']}")
    print(f"  Courses skipped:       {stats['courses_skipped']}")
    print(f"  Total train students:  {total_train}")
    print(f"  Total test students:   {total_test}")
    print(f"  Total test sets:       {stats['test_sets_created']}")
    print(f"\nOutputs in {args.output_dir}:")
    print(f"  Training DB:       cs_db_train.json")
    print(f"  Test sets dir:     test_sets/ ({stats['test_sets_created']} files)")
    print(f"  Ground truth:      ground_truth_for_cutoff_data.json")


if __name__ == "__main__":
    main()