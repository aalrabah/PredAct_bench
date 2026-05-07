"""
Evaluate ONE sim episode log (or a directory of them) using the same metrics
defined in evaluate_human_study.py — F1, Trajectory-RAIR, Trajectory-RSR.

The sim log already uses the human-study `per_student` schema, so we reuse
the metric functions directly. No participant-style wrapping needed.

Usage:
    python -m sim.evaluate_episode sim/smoke_episode_log__qwen_35b_x_gpt4o_mini.json
    python -m sim.evaluate_episode sim_logs/  # process every .json in the dir
"""

import argparse
import json
import os
import sys

from evaluate_human_study import (
    compute_decision_quality,
    compute_trajectory_metrics,
)


def evaluate_one(log):
    """Compute decision-quality + trajectory metrics for one sim log."""
    per_student = log.get("per_student", {}) or {}

    all_sids = set(per_student.keys())
    at_risk_set = {sid for sid, rec in per_student.items() if rec.get("is_at_risk")}
    final_flagged = {sid for sid, rec in per_student.items()
                     if rec.get("final_decision") == "accept"}
    initial_flagged = {sid for sid, rec in per_student.items()
                       if rec.get("initial_decision") == "accept"}

    final_metrics = compute_decision_quality(final_flagged, all_sids, at_risk_set)
    initial_metrics = (
        compute_decision_quality(initial_flagged, all_sids, at_risk_set)
        if initial_flagged else None
    )
    trajectory = compute_trajectory_metrics(per_student)

    return {
        "condition_id": log.get("condition_id"),
        "course_id": log.get("course_id"),
        "target_accuracy": log.get("target_accuracy"),
        "assistant_llm": log.get("assistant_llm"),
        "instructor_llm": log.get("instructor_llm"),
        "n_students": len(all_sids),
        "n_at_risk": len(at_risk_set),
        "n_initial_flagged": len(initial_flagged),
        "n_final_flagged": len(final_flagged),
        "initial": initial_metrics,
        "final": final_metrics,
        "trajectory": trajectory,
        "per_student": per_student,
        "agent_flagged_sids": log.get("agent_flagged_sids", []),
        "at_risk_set": at_risk_set,
    }


def _mark(decision_is_flag, truth_is_at_risk):
    """Return ✓ if the decision matches truth, ✗ otherwise."""
    return "✓" if decision_is_flag == truth_is_at_risk else "✗"


def print_breakdown(report, indent=""):
    """Show per-student truth + each decision-maker's call, side by side."""
    per_student = report["per_student"]
    flagged_sids = report["agent_flagged_sids"]
    at_risk_set = report["at_risk_set"]

    print(f"{indent}Legend: 'flag' = student marked at-risk, 'no_flag' = student NOT marked.")
    print(f"{indent}        ✓ = decision matches truth, ✗ = decision wrong.")
    print()
    print(f"{indent}Students the AGENT flagged (these are what the instructor saw):")
    header = f"  {'student_id':<14} {'truth':<10} {'AGENT':<10} {'INITIAL':<14} {'FINAL':<14}"
    print(f"{indent}{header}")
    print(f"{indent}  " + "-" * (len(header) - 2))

    counts = {"agent_correct": 0, "agent_wrong": 0,
              "initial_correct": 0, "initial_wrong": 0,
              "final_correct": 0, "final_wrong": 0}

    for sid in flagged_sids:
        rec = per_student.get(sid, {})
        truth_is_at_risk = bool(rec.get("is_at_risk"))
        truth = "AT-RISK" if truth_is_at_risk else "safe"

        # Agent flagged everyone in this list
        agent_flag = True
        agent_str = f"flag {_mark(agent_flag, truth_is_at_risk)}"

        # Instructor initial / final
        initial = rec.get("initial_decision")
        final = rec.get("final_decision")
        initial_flag = (initial == "accept")
        final_flag = (final == "accept")
        initial_word = "flag" if initial_flag else ("no_flag" if initial == "reject" else "-")
        final_word   = "flag" if final_flag   else ("no_flag" if final   == "reject" else "-")
        initial_str  = f"{initial_word} {_mark(initial_flag, truth_is_at_risk)}" if initial else "-"
        final_str    = f"{final_word} {_mark(final_flag, truth_is_at_risk)}"     if final   else "-"

        # Tallies
        if agent_flag == truth_is_at_risk: counts["agent_correct"] += 1
        else:                              counts["agent_wrong"] += 1
        if initial:
            if initial_flag == truth_is_at_risk: counts["initial_correct"] += 1
            else:                                counts["initial_wrong"] += 1
        if final:
            if final_flag == truth_is_at_risk: counts["final_correct"] += 1
            else:                              counts["final_wrong"] += 1

        print(f"{indent}  {sid:<14} {truth:<10} {agent_str:<10} {initial_str:<14} {final_str:<14}")

    n = len(flagged_sids)
    print()
    print(f"{indent}Summary across the {n} agent-flagged students:")
    print(f"{indent}  AGENT was correct {counts['agent_correct']}/{n}, wrong {counts['agent_wrong']}/{n}")
    print(f"{indent}  INITIAL instructor was correct {counts['initial_correct']}/{n}, wrong {counts['initial_wrong']}/{n}")
    print(f"{indent}  FINAL   instructor was correct {counts['final_correct']}/{n}, wrong {counts['final_wrong']}/{n}")

    # Students truly at-risk that the agent didn't surface — invisible to instructor.
    missed_by_agent = at_risk_set - set(flagged_sids)
    if missed_by_agent:
        print()
        print(f"{indent}Truly at-risk but NOT flagged by the agent (instructor never saw these):")
        for sid in sorted(missed_by_agent):
            print(f"{indent}  {sid}")


def print_one(report, indent=""):
    cid = report["condition_id"] or "(unknown)"
    inst = report.get("instructor_llm") or "?"
    asst = report.get("assistant_llm") or "?"
    acc = report.get("target_accuracy")
    acc_str = f"{int(acc*100)}%" if acc is not None else "?"

    print(f"{indent}{cid}  (instructor={inst}, assistant={asst}, accuracy={acc_str})")
    print(f"{indent}  Students: {report['n_students']} total, "
          f"{report['n_at_risk']} truly at-risk, "
          f"{report['n_initial_flagged']} initial-flagged, "
          f"{report['n_final_flagged']} final-flagged")

    f = report["final"]
    print(f"{indent}  FINAL    P={f['precision']:.3f}  R={f['recall']:.3f}  "
          f"F1={f['f1']:.3f}  (TP={f['tp']} FP={f['fp']} FN={f['fn']} TN={f['tn']})")

    i = report["initial"]
    if i is not None:
        print(f"{indent}  INITIAL  P={i['precision']:.3f}  R={i['recall']:.3f}  F1={i['f1']:.3f}")

    t = report["trajectory"]
    rair = t["trajectory_rair"]
    rsr = t["trajectory_rsr"]
    rair_str = f"{rair:.3f}" if rair is not None else "n/a"
    rsr_str = f"{rsr:.3f}" if rsr is not None else "n/a"
    print(f"{indent}  TRAJ     RAIR={rair_str} ({t['rair_numerator']}/{t['rair_denominator']})  "
          f"RSR={rsr_str} ({t['rsr_numerator']}/{t['rsr_denominator']})")


def collect_log_paths(target):
    if os.path.isdir(target):
        return sorted(
            os.path.join(target, f) for f in os.listdir(target)
            if f.endswith(".json")
        )
    return [target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a sim log file or a directory of them.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-student truth + decisions for each log.")
    args = parser.parse_args()

    paths = collect_log_paths(args.path)
    if not paths:
        print(f"No .json files found at {args.path}")
        sys.exit(1)

    print(f"Evaluating {len(paths)} log(s) from {args.path}")
    print("=" * 90)

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue
        report = evaluate_one(log)
        print(f"\nFile: {os.path.basename(path)}")
        print_one(report, indent="  ")
        if args.verbose:
            print()
            print_breakdown(report, indent="  ")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
