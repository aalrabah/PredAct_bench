"""
Scan data.json and rank dialogues by flagged_student_count.
Helps pick interesting scenarios for the human study.

Usage: python scan_dialogues.py [--data path/to/data.json] [--top N]
"""

import json
import argparse
from pathlib import Path


def extract_report_summary(dlg_id, entry):
    """Extract key report fields from a dialogue entry."""
    goal = entry.get("goal", {})
    log = entry.get("log", [])

    # Get final belief state from last log entry
    final_meta = {}
    for turn in reversed(log):
        if turn.get("metadata"):
            final_meta = turn["metadata"]
            break

    class_ctx = goal.get("class_context", {})
    class_summary = final_meta.get("class_summary", {})
    student_status = final_meta.get("student_status", {})
    intervention = final_meta.get("intervention", {})

    # Count risk groups (excluding no_risk)
    risk_groups = {}
    for key, group in student_status.items():
        if key == "no_risk":
            continue
        risk_groups[key] = {
            "count": group.get("count", 0),
            "predicted_grade": group.get("predicted_grade", "?"),
            "failure_risk": group.get("failure_risk", "?"),
            "student_ids": group.get("student_ids", []),
        }

    return {
        "dlg_id": dlg_id,
        "course": class_ctx.get("course_name", "?"),
        "week": class_ctx.get("week", "?"),
        "student_count": goal.get("student_count", 0),
        "avg_gpa": class_summary.get("average_gpa", "?"),
        "grade_trend": class_summary.get("grade_trend", "?"),
        "flagged_count": class_summary.get("flagged_student_count", 0),
        "common_issue": class_summary.get("common_assignment_type_issue", "none"),
        "risk_groups": risk_groups,
        "intervention": intervention,
        "total_turns": len(log),
    }


def main():
    parser = argparse.ArgumentParser(description="Scan PredAct dialogues and rank by flagged students")
    parser.add_argument("--data", type=str, default="data.json", help="Path to data.json")
    parser.add_argument("--top", type=int, default=None, help="Show only top N results")
    parser.add_argument("--min-flagged", type=int, default=0, help="Only show dialogues with >= N flagged students")
    parser.add_argument("--export", type=str, default=None, help="Export results to JSON file")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Found {len(data)} dialogues\n")

    # Extract summaries
    summaries = []
    for dlg_id, entry in data.items():
        summary = extract_report_summary(dlg_id, entry)
        summaries.append(summary)

    # Filter
    if args.min_flagged > 0:
        summaries = [s for s in summaries if s["flagged_count"] >= args.min_flagged]

    # Sort by flagged_count descending, then by number of risk groups
    summaries.sort(key=lambda s: (s["flagged_count"], len(s["risk_groups"])), reverse=True)

    if args.top:
        summaries = summaries[:args.top]

    # Print
    print(f"{'Rank':<5} {'DLG ID':<20} {'Course':<12} {'Week':<8} {'Students':<10} {'Flagged':<8} {'GPA':<6} {'Trend':<12} {'Issue':<12} {'Risk Groups'}")
    print("-" * 120)

    for i, s in enumerate(summaries, 1):
        risk_str = ", ".join(f"{k}({v['count']})" for k, v in s["risk_groups"].items()) or "none"
        print(f"{i:<5} {s['dlg_id']:<20} {s['course']:<12} {s['week']:<8} {s['student_count']:<10} {s['flagged_count']:<8} {str(s['avg_gpa']):<6} {s['grade_trend']:<12} {s['common_issue']:<12} {risk_str}")

    # Export if requested
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"\nExported {len(summaries)} results to {args.export}")

    # Quick stats
    total_flagged = sum(s["flagged_count"] for s in summaries)
    with_flags = sum(1 for s in summaries if s["flagged_count"] > 0)
    print(f"\nStats: {with_flags}/{len(summaries)} dialogues have flagged students, {total_flagged} total flagged")


if __name__ == "__main__":
    main()