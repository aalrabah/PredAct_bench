"""
PredAct - Map Synthetic Students to Real Students
Enriches the report with:
  - Full grade records for each flagged student (always, from grades lookup)
  - Mapped SQL submissions + quiz reference data (optional, if student-data and quizzes provided)

Usage (with SQL data — e.g. week 5):
    python map_students.py --report CS-411/reports/week5_report.json --grades CS-411/reports/week5_grades_lookup.json --student-data CS-411/flagged_students/week5_flagged_students.json --quizzes CS-411/assignments/ --output CS-411/reports/week5_report_enriched.json

Usage (grades only — e.g. week 8):
    python map_students.py --report CS-411/reports/week8_report.json --grades CS-411/reports/week8_grades_lookup.json --output CS-411/reports/week8_report_enriched.json
"""

import json
import argparse
import os
import glob


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_quizzes(quizzes_dir):
    """
    Load all quiz JSON files from a directory.
    Returns a lookup: sql_qN -> {question_id, quiz, text, correct_solution, concept_tags}
    """
    questions = {}
    if not quizzes_dir or not os.path.exists(quizzes_dir):
        return questions

    quiz_files = glob.glob(os.path.join(quizzes_dir, "quiz_*.json"))

    for quiz_file in sorted(quiz_files):
        quiz = load_json(quiz_file)
        quiz_id = quiz.get("quiz_id", os.path.basename(quiz_file))

        for q in quiz.get("questions", []):
            qid = q["question_id"]
            key = f"sql_q{qid[1:]}"
            questions[key] = {
                "question_id": qid,
                "quiz": quiz_id,
                "week": quiz.get("week", "?"),
                "text": q["text"],
                "correct_solution": q["correct_solution"],
                "concept_tags": q["concept_tags"],
            }

    return questions


def get_flagged_students_from_report(report):
    """
    Extract flagged synthetic student IDs from report, ordered by risk severity.
    Critical first, then high, medium, unknown.
    """
    risk_priority = ["critical_risk", "high_risk", "medium_risk", "unknown_risk"]
    flagged = []

    for risk_key in risk_priority:
        group = report.get("risk_groups", {}).get(risk_key)
        if not group:
            continue
        for student in group.get("students", []):
            flagged.append({
                "student_id": student["student_id"],
                "risk_key": risk_key,
                "failure_risk": group.get("failure_risk", "?"),
                "predicted_grade": student.get("predicted_grade", "?"),
            })

    return flagged


def rank_real_students(student_data):
    """
    Rank real students by total false submissions (most first).
    """
    ranked = sorted(student_data, key=lambda s: s.get("total_false_submissions", 0), reverse=True)
    return ranked


def build_mapping(flagged_synthetic, real_students):
    """
    Map synthetic IDs to real students.
    Worst real student (most errors) → highest risk synthetic student.
    """
    mapping = {}
    n = min(len(flagged_synthetic), len(real_students))

    for i in range(n):
        syn_id = flagged_synthetic[i]["student_id"]
        real = real_students[i]
        mapping[syn_id] = {
            "real_student_id": real["student_id"],
            "total_false_submissions": real["total_false_submissions"],
            "risk_key": flagged_synthetic[i]["risk_key"],
            "failure_risk": flagged_synthetic[i]["failure_risk"],
            "predicted_grade": flagged_synthetic[i]["predicted_grade"],
        }

    return mapping


def build_sql_summary(real_student, questions_lookup):
    """
    Build a clean SQL submission summary for a real student.
    Includes question text, correct solution, and concept tags from quiz data.
    """
    summary = {}
    for question_id, qdata in real_student.get("questions", {}).items():
        entry = {
            "quiz": qdata.get("quiz", "?"),
            "false_attempts": qdata.get("false_attempts", 0),
            "submissions": qdata.get("submissions", []),
        }

        if question_id in questions_lookup:
            ref = questions_lookup[question_id]
            entry["question_text"] = ref["text"]
            entry["correct_solution"] = ref["correct_solution"]
            entry["concept_tags"] = ref["concept_tags"]

        summary[question_id] = entry

    return summary


def build_grade_record(student_id, grades_lookup):
    """
    Build a full grade record for a student from the grades lookup.
    Returns assignments list sorted by week + weighted average.
    """
    if not grades_lookup or student_id not in grades_lookup:
        return None

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
        "assignments": assignments,
        "weighted_average": weighted_avg,
        "total_weight_covered": round(total_weight, 4),
    }


def enrich_report(report, grades_lookup, mapping=None, real_students_lookup=None, questions_lookup=None):
    """
    Add to each flagged student:
      - Full grade record (always, from grades lookup)
      - SQL submission data + quiz reference (optional, if mapping and real data provided)
    """
    # Add quiz reference at report level if available
    if questions_lookup:
        report["quiz_reference"] = questions_lookup

    # Enrich each flagged student
    for risk_key, group in report.get("risk_groups", {}).items():
        if risk_key == "no_risk":
            continue

        for student in group.get("students", []):
            sid = student["student_id"]

            # Always add grade record
            grade_record = build_grade_record(sid, grades_lookup)
            if grade_record:
                student["grade_record"] = grade_record

            # Add SQL submissions if mapping exists
            if mapping and sid in mapping and real_students_lookup and questions_lookup:
                real_id = mapping[sid]["real_student_id"]
                real_student = real_students_lookup.get(real_id)

                student["mapped_real_student"] = real_id

                if real_student:
                    student["sql_submissions"] = build_sql_summary(real_student, questions_lookup)

    return report


def main():
    parser = argparse.ArgumentParser(description="Enrich report with grade records and optionally SQL submissions")
    parser.add_argument("--report", required=True, help="Path to report JSON")
    parser.add_argument("--grades", required=True, help="Path to grades lookup JSON (all students)")
    parser.add_argument("--student-data", default=None, help="Optional: path to flagged student SQL submissions")
    parser.add_argument("--quizzes", default=None, help="Optional: path to assignments directory with quiz JSONs")
    parser.add_argument("--output", required=True, help="Output path for enriched report")
    parser.add_argument("--save-mapping", default=None, help="Optional: save mapping to JSON")
    args = parser.parse_args()

    # Load required data
    report = load_json(args.report)
    grades_lookup = load_json(args.grades)
    print(f"Loaded grades for {len(grades_lookup)} students")

    # Get flagged students
    flagged_synthetic = get_flagged_students_from_report(report)
    print(f"\nFound {len(flagged_synthetic)} flagged synthetic students:")
    for s in flagged_synthetic:
        sid = s['student_id']
        in_grades = sid in grades_lookup
        print(f"  {sid} ({s['risk_key']}, grade={s['predicted_grade']}) — in grades: {in_grades}")

    # Load optional SQL data
    mapping = None
    real_lookup = None
    questions_lookup = None

    if args.student_data and os.path.exists(args.student_data):
        student_data = load_json(args.student_data)
        print(f"\nLoaded {len(student_data)} real student SQL records")

        # Load quiz data
        questions_lookup = load_quizzes(args.quizzes)
        if questions_lookup:
            print(f"Loaded {len(questions_lookup)} questions from quizzes:")
            for key, q in questions_lookup.items():
                print(f"  {key} ({q['quiz']}): {q['concept_tags']}")

        # Rank and map
        ranked_real = rank_real_students(student_data)
        print(f"\nTop {len(flagged_synthetic)} real students by false submissions:")
        for r in ranked_real[:len(flagged_synthetic)]:
            print(f"  {r['student_id']}: {r['total_false_submissions']} wrong submissions")

        mapping = build_mapping(flagged_synthetic, ranked_real)
        print(f"\nMapping:")
        for syn_id, info in mapping.items():
            print(f"  {syn_id} ({info['risk_key']}) -> {info['real_student_id']} ({info['total_false_submissions']} errors)")

        real_lookup = {s["student_id"]: s for s in student_data}
    else:
        print("\nNo SQL submission data provided — enriching with grades only")

    # Enrich report
    enriched = enrich_report(report, grades_lookup, mapping, real_lookup, questions_lookup)

    # Verify
    print(f"\nVerification:")
    for risk_key, group in enriched.get("risk_groups", {}).items():
        if risk_key == "no_risk":
            continue
        for student in group.get("students", []):
            sid = student["student_id"]
            has_grades = "grade_record" in student
            has_sql = "sql_submissions" in student
            avg = student.get("grade_record", {}).get("weighted_average", "N/A")
            print(f"  {sid}: grades={has_grades} (avg={avg}), sql={has_sql}")

    # Save
    save_json(enriched, args.output)
    print(f"\nEnriched report saved to {args.output}")

    if args.save_mapping and mapping:
        save_json(mapping, args.save_mapping)
        print(f"Mapping saved to {args.save_mapping}")


if __name__ == "__main__":
    main()