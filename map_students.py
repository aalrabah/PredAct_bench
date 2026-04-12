"""
PredAct - Map Synthetic Students to Real Students
Maps synthetic flagged student IDs to real student SQL submission data.
Enriches the report with mapped SQL submissions AND quiz reference data
(questions, correct solutions, concept tags).

Usage:
    python map_students.py --report CS-411/reports/week5_report.json --student-data CS-411/flagged_students/week5_flagged_students.json --quizzes CS-411/assignments/ --output CS-411/reports/week5_report_enriched.json
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
    quiz_files = glob.glob(os.path.join(quizzes_dir, "quiz_*.json"))

    for quiz_file in sorted(quiz_files):
        quiz = load_json(quiz_file)
        quiz_id = quiz.get("quiz_id", os.path.basename(quiz_file))

        for q in quiz.get("questions", []):
            qid = q["question_id"]
            # Convert Q5 -> sql_q5, Q11 -> sql_q11
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

        # Attach quiz reference data
        if question_id in questions_lookup:
            ref = questions_lookup[question_id]
            entry["question_text"] = ref["text"]
            entry["correct_solution"] = ref["correct_solution"]
            entry["concept_tags"] = ref["concept_tags"]

        summary[question_id] = entry

    return summary


def enrich_report(report, mapping, real_students_lookup, questions_lookup):
    """
    Add SQL submission data and quiz reference to each flagged student in the report.
    Also adds the full quiz reference to the report for system-level access.
    """
    # Add quiz reference at report level
    report["quiz_reference"] = questions_lookup

    # Enrich each flagged student
    for risk_key, group in report.get("risk_groups", {}).items():
        if risk_key == "no_risk":
            continue

        for student in group.get("students", []):
            sid = student["student_id"]
            if sid not in mapping:
                continue

            real_id = mapping[sid]["real_student_id"]
            real_student = real_students_lookup.get(real_id)
            if not real_student:
                continue

            student["mapped_real_student"] = real_id
            student["sql_submissions"] = build_sql_summary(real_student, questions_lookup)

    return report


def main():
    parser = argparse.ArgumentParser(description="Map synthetic to real students and enrich report")
    parser.add_argument("--report", required=True, help="Path to report JSON")
    parser.add_argument("--student-data", required=True, help="Path to week5_flagged_students.json")
    parser.add_argument("--quizzes", required=True, help="Path to assignments directory with quiz JSONs")
    parser.add_argument("--output", required=True, help="Output path for enriched report")
    parser.add_argument("--save-mapping", default=None, help="Optional: save mapping to JSON")
    args = parser.parse_args()

    # Load data
    report = load_json(args.report)
    student_data = load_json(args.student_data)

    # Load quiz data
    questions_lookup = load_quizzes(args.quizzes)
    print(f"Loaded {len(questions_lookup)} questions from quizzes:")
    for key, q in questions_lookup.items():
        print(f"  {key} ({q['quiz']}): {q['concept_tags']}")

    # Get flagged synthetic students ordered by risk
    flagged_synthetic = get_flagged_students_from_report(report)
    print(f"\nFound {len(flagged_synthetic)} flagged synthetic students:")
    for s in flagged_synthetic:
        print(f"  {s['student_id']} ({s['risk_key']}, grade={s['predicted_grade']})")

    # Rank real students by error count
    ranked_real = rank_real_students(student_data)
    print(f"\nTop {len(flagged_synthetic)} real students by false submissions:")
    for r in ranked_real[:len(flagged_synthetic)]:
        print(f"  {r['student_id']}: {r['total_false_submissions']} wrong submissions")

    # Build mapping
    mapping = build_mapping(flagged_synthetic, ranked_real)
    print(f"\nMapping:")
    for syn_id, info in mapping.items():
        print(f"  {syn_id} ({info['risk_key']}) -> {info['real_student_id']} ({info['total_false_submissions']} errors)")

    # Build real student lookup
    real_lookup = {s["student_id"]: s for s in student_data}

    # Enrich report
    enriched = enrich_report(report, mapping, real_lookup, questions_lookup)

    # Save enriched report
    save_json(enriched, args.output)
    print(f"\nEnriched report saved to {args.output}")

    # Save mapping if requested
    if args.save_mapping:
        save_json(mapping, args.save_mapping)
        print(f"Mapping saved to {args.save_mapping}")


if __name__ == "__main__":
    main()