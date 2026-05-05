"""
PredAct Benchmark - Evaluation Script

Reads participant logs from study_logs/ and computes:

Exp 2 — Decision Quality (final decisions vs ground truth)
  Metrics: precision, recall, F1 (based on FINAL decisions)
  Also computed for INITIAL decisions as a baseline.

Exp 3 — Trajectory-Level Override (initial -> chat -> final)
  Uses the proper Schemmer et al. 2023 definition:

    Trajectory-RAIR:
      Of cases where instructor was INITIALLY WRONG and AGENT was RIGHT,
      how often did the instructor correctly switch to the AGENT'S answer?

    Trajectory-RSR:
      Of cases where instructor was INITIALLY RIGHT and AGENT was WRONG,
      how often did the instructor correctly STICK with their own answer?

  Scope: only students the AGENT flagged (same as human study UI).

Usage:
    python evaluate_study.py
    python evaluate_study.py --logs-dir study_logs --out results_summary.csv
"""

import os
import json
import csv
import argparse
from collections import defaultdict


# =============================================================================
# METRIC HELPERS
# =============================================================================

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_decision_quality(flagged_set, all_sids, at_risk_set):
    """Given a set of flagged sids, compute TP/FP/FN/TN + precision/recall/F1."""
    tp = len(flagged_set & at_risk_set)
    fp = len(flagged_set - at_risk_set)
    fn = len(at_risk_set - flagged_set)
    tn = len(all_sids) - tp - fp - fn
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_trajectory_metrics(per_student):
    """
    Proper Schemmer et al. 2023 Trajectory-RAIR / Trajectory-RSR.

    For each student where the agent flagged AND the instructor made both
    an initial and final decision, classify by:
      - initial correctness (initial vs truth)
      - agent correctness (agent flag vs truth)
      - final correctness (final vs truth)

    RAIR = (initial wrong, agent right, final right) / (initial wrong, agent right, any final)
    RSR  = (initial right, agent wrong, final right) / (initial right, agent wrong, any final)

    All decisions are binary: accept (flag as at-risk) or reject (not).
    Agent "right" = agent flagged AND student truly at-risk.
    Agent flagged everyone in this set (by construction of the UI).
    So: agent right iff student truly at-risk, agent wrong iff not truly at-risk.
    """
    # Initialize buckets
    rair_numerator = 0    # initial wrong, agent right, final right
    rair_denominator = 0  # initial wrong, agent right (any final)
    rsr_numerator = 0     # initial right, agent wrong, final right
    rsr_denominator = 0   # initial right, agent wrong (any final)

    # Also keep the simple 4-bucket classification for reference
    correct_follow = 0
    bad_follow = 0
    correct_override = 0
    bad_override = 0

    considered = 0

    for sid, rec in per_student.items():
        if not rec.get("agent_flagged"):
            continue
        initial = rec.get("initial_decision")
        final = rec.get("final_decision")
        is_at_risk = rec.get("is_at_risk")

        if initial is None or final is None or is_at_risk is None:
            continue

        considered += 1

        # Decisions as booleans (True = flagged as at-risk)
        initial_flagged = initial == "accept"
        final_flagged = final == "accept"
        # Instructor correct if their decision matches truth
        initial_correct = initial_flagged == is_at_risk
        final_correct = final_flagged == is_at_risk
        # Agent flagged = True here (we only look at agent-flagged students).
        # Agent correct iff the student truly is at-risk.
        agent_correct = is_at_risk

        # RAIR: initial wrong, agent right
        if (not initial_correct) and agent_correct:
            rair_denominator += 1
            if final_correct:
                rair_numerator += 1

        # RSR: initial right, agent wrong
        if initial_correct and (not agent_correct):
            rsr_denominator += 1
            if final_correct:
                rsr_numerator += 1

        # 4-bucket classification based on final decision
        if final_flagged and is_at_risk:
            correct_follow += 1
        elif final_flagged and not is_at_risk:
            bad_follow += 1
        elif (not final_flagged) and (not is_at_risk):
            correct_override += 1
        elif (not final_flagged) and is_at_risk:
            bad_override += 1

    rair = rair_numerator / rair_denominator if rair_denominator > 0 else None
    rsr = rsr_numerator / rsr_denominator if rsr_denominator > 0 else None

    return {
        "agent_flagged_count": considered,
        "trajectory_rair": round(rair, 4) if rair is not None else None,
        "trajectory_rsr": round(rsr, 4) if rsr is not None else None,
        "rair_numerator": rair_numerator,
        "rair_denominator": rair_denominator,
        "rsr_numerator": rsr_numerator,
        "rsr_denominator": rsr_denominator,
        "correct_follow": correct_follow,
        "bad_follow": bad_follow,
        "correct_override": correct_override,
        "bad_override": bad_override,
    }


# =============================================================================
# PROCESS ONE PARTICIPANT
# =============================================================================

def process_participant(log):
    participant = log.get("participant", {})
    name = participant.get("name", "unknown")
    email = participant.get("email", "unknown")

    rows = []
    for cond in log.get("conditions", []):
        cond_id = cond.get("condition_id")
        has_agent = cond.get("has_agent", False)
        per_student = cond.get("per_student", {}) or {}

        # Pull sid sets from the log itself (no need for external ground truth)
        all_sids = set(per_student.keys())
        at_risk_set = {sid for sid, rec in per_student.items() if rec.get("is_at_risk")}

        final_flagged = {
            sid for sid, rec in per_student.items()
            if rec.get("final_decision") == "accept"
        }
        initial_flagged = {
            sid for sid, rec in per_student.items()
            if rec.get("initial_decision") == "accept"
        }

        # Exp 2 — decision quality
        final_metrics = compute_decision_quality(final_flagged, all_sids, at_risk_set)
        if initial_flagged:
            initial_metrics = compute_decision_quality(initial_flagged, all_sids, at_risk_set)
        else:
            initial_metrics = None

        # Exp 3 — trajectory metrics (agent scenarios only)
        if has_agent:
            trajectory = compute_trajectory_metrics(per_student)
        else:
            trajectory = {
                "agent_flagged_count": None,
                "trajectory_rair": None,
                "trajectory_rsr": None,
                "rair_numerator": None,
                "rair_denominator": None,
                "rsr_numerator": None,
                "rsr_denominator": None,
                "correct_follow": None,
                "bad_follow": None,
                "correct_override": None,
                "bad_override": None,
            }

        row = {
            "participant_name": name,
            "participant_email": email,
            "condition_id": cond_id,
            "course_id": cond.get("course_id"),
            "week": cond.get("week"),
            "has_agent": has_agent,
            "feature_set": cond.get("feature_set"),
            "duration_seconds": cond.get("duration_seconds"),
            "n_students_in_scenario": len(all_sids),
            "n_truly_at_risk": len(at_risk_set),
            "n_initial_flagged": len(initial_flagged) if initial_flagged else 0,
            "n_final_flagged": len(final_flagged),
            # Final decision quality
            "final_tp": final_metrics["tp"],
            "final_fp": final_metrics["fp"],
            "final_fn": final_metrics["fn"],
            "final_tn": final_metrics["tn"],
            "final_precision": final_metrics["precision"],
            "final_recall": final_metrics["recall"],
            "final_f1": final_metrics["f1"],
            # Initial decision quality (agent scenarios only)
            "initial_precision": initial_metrics["precision"] if initial_metrics else None,
            "initial_recall": initial_metrics["recall"] if initial_metrics else None,
            "initial_f1": initial_metrics["f1"] if initial_metrics else None,
            # Trajectory metrics
            **trajectory,
        }
        rows.append(row)

    return rows


# =============================================================================
# AGGREGATE SUMMARY
# =============================================================================

def print_summary(all_rows):
    by_cond = defaultdict(list)
    for r in all_rows:
        by_cond[r["condition_id"]].append(r)

    print()
    print("=" * 100)
    print("AGGREGATE SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Condition':<12}{'N':<4}"
          f"{'Init F1':<10}{'Final F1':<10}"
          f"{'Precision':<11}{'Recall':<9}"
          f"{'RAIR':<8}{'RSR':<8}")
    print("-" * 100)

    for cond_id in ("no_agent", "tool_60", "tool_85"):
        rows = by_cond.get(cond_id, [])
        if not rows:
            continue
        n = len(rows)

        final_f1 = sum(r["final_f1"] for r in rows) / n
        final_precision = sum(r["final_precision"] for r in rows) / n
        final_recall = sum(r["final_recall"] for r in rows) / n

        init_f1_vals = [r["initial_f1"] for r in rows if r["initial_f1"] is not None]
        init_f1 = sum(init_f1_vals) / len(init_f1_vals) if init_f1_vals else None

        rair_vals = [r["trajectory_rair"] for r in rows if r["trajectory_rair"] is not None]
        rsr_vals = [r["trajectory_rsr"] for r in rows if r["trajectory_rsr"] is not None]
        rair = sum(rair_vals) / len(rair_vals) if rair_vals else None
        rsr = sum(rsr_vals) / len(rsr_vals) if rsr_vals else None

        init_f1_str = f"{init_f1:.3f}" if init_f1 is not None else "-"
        rair_str = f"{rair:.3f}" if rair is not None else "-"
        rsr_str = f"{rsr:.3f}" if rsr is not None else "-"

        print(f"{cond_id:<12}{n:<4}"
              f"{init_f1_str:<10}{final_f1:<10.3f}"
              f"{final_precision:<11.3f}{final_recall:<9.3f}"
              f"{rair_str:<8}{rsr_str:<8}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        default="study_logs",
    )
    parser.add_argument(
        "--out",
        default="results_summary.csv",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"ERROR: {args.logs_dir} does not exist")
        return

    log_files = sorted(f for f in os.listdir(args.logs_dir) if f.endswith(".json"))
    print(f"Found {len(log_files)} participant log files in {args.logs_dir}")

    all_rows = []
    for fname in log_files:
        path = os.path.join(args.logs_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue

        if "conditions" not in log:
            continue

        rows = process_participant(log)
        all_rows.extend(rows)
        name = log.get("participant", {}).get("name", "?")
        print(f"  {fname} -> {name} ({len(rows)} conditions)")

    if not all_rows:
        print("No data to summarize.")
        return

    # Write CSV
    fieldnames = list(all_rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written to {args.out}")

    print_summary(all_rows)


if __name__ == "__main__":
    main()