"""
PredAct Benchmark - Evaluation Script

Reads participant logs from study_logs/ and computes:

Exp 2 — Decision Quality (final decisions vs ground truth)
  Metrics: precision, recall, F1 (based on FINAL decisions)
  Also computed for INITIAL decisions as a baseline.

Exp 3 — Trajectory-Level Override (initial -> chat -> final)
  Schemmer et al. 2023 RAIR / RSR.

Outputs:
  - results_per_participant.csv (one row per participant x condition)
  - results_pooled.csv          (one row per (LLM x target_accuracy) cell, pooled)
"""

import os
import json
import csv
import argparse
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# METRIC HELPERS
# =============================================================================

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_decision_quality(flagged_set, all_sids, at_risk_set):
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
    rair_numerator = rair_denominator = 0
    rsr_numerator = rsr_denominator = 0
    correct_follow = bad_follow = correct_override = bad_override = 0
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
        initial_flagged = initial == "accept"
        final_flagged = final == "accept"
        initial_correct = initial_flagged == is_at_risk
        final_correct = final_flagged == is_at_risk
        agent_correct = is_at_risk  # agent flagged this student by construction

        if (not initial_correct) and agent_correct:
            rair_denominator += 1
            if final_correct:
                rair_numerator += 1
        if initial_correct and (not agent_correct):
            rsr_denominator += 1
            if final_correct:
                rsr_numerator += 1

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

        all_sids = set(per_student.keys())
        at_risk_set = {sid for sid, rec in per_student.items() if rec.get("is_at_risk")}
        final_flagged = {sid for sid, rec in per_student.items() if rec.get("final_decision") == "accept"}
        initial_flagged = {sid for sid, rec in per_student.items() if rec.get("initial_decision") == "accept"}

        final_metrics = compute_decision_quality(final_flagged, all_sids, at_risk_set)
        initial_metrics = compute_decision_quality(initial_flagged, all_sids, at_risk_set) if initial_flagged else None

        if has_agent:
            trajectory = compute_trajectory_metrics(per_student)
        else:
            trajectory = {
                "agent_flagged_count": None,
                "trajectory_rair": None, "trajectory_rsr": None,
                "rair_numerator": 0, "rair_denominator": 0,
                "rsr_numerator": 0, "rsr_denominator": 0,
                "correct_follow": 0, "bad_follow": 0,
                "correct_override": 0, "bad_override": 0,
            }

        row = {
            "participant_name": name,
            "participant_email": email,
            "condition_id": cond_id,
            "course_id": cond.get("course_id"),
            "week": cond.get("week"),
            "has_agent": has_agent,
            "llm": cond.get("llm"),
            "target_accuracy": cond.get("target_accuracy"),
            "feature_set": cond.get("feature_set"),
            "duration_seconds": cond.get("duration_seconds"),
            "n_students_in_scenario": len(all_sids),
            "n_truly_at_risk": len(at_risk_set),
            "n_initial_flagged": len(initial_flagged) if initial_flagged else 0,
            "n_final_flagged": len(final_flagged),
            "final_tp": final_metrics["tp"],
            "final_fp": final_metrics["fp"],
            "final_fn": final_metrics["fn"],
            "final_tn": final_metrics["tn"],
            "final_precision": final_metrics["precision"],
            "final_recall": final_metrics["recall"],
            "final_f1": final_metrics["f1"],
            "initial_precision": initial_metrics["precision"] if initial_metrics else None,
            "initial_recall": initial_metrics["recall"] if initial_metrics else None,
            "initial_f1": initial_metrics["f1"] if initial_metrics else None,
            **trajectory,
        }
        rows.append(row)

    return rows


# =============================================================================
# POOLED AGGREGATION
# =============================================================================

def aggregate_pooled(all_rows):
    """Pool raw counts across participants per (llm, target_accuracy) cell."""
    by_cell = defaultdict(lambda: {
        "n_participants": 0,
        "final_tp": 0, "final_fp": 0, "final_fn": 0, "final_tn": 0,
        "rair_numerator": 0, "rair_denominator": 0,
        "rsr_numerator": 0, "rsr_denominator": 0,
        "correct_follow": 0, "bad_follow": 0,
        "correct_override": 0, "bad_override": 0,
    })

    for r in all_rows:
        key = (r["llm"], r["target_accuracy"]) if r.get("has_agent") else ("no_agent", None)
        b = by_cell[key]
        b["n_participants"] += 1
        for k in ("final_tp", "final_fp", "final_fn", "final_tn",
                  "correct_follow", "bad_follow", "correct_override", "bad_override",
                  "rair_numerator", "rair_denominator", "rsr_numerator", "rsr_denominator"):
            b[k] += r.get(k) or 0

    out = []
    for (llm, ta), b in sorted(by_cell.items(), key=lambda x: (x[0][0] or "", x[0][1] or 0)):
        p, r, f1 = precision_recall_f1(b["final_tp"], b["final_fp"], b["final_fn"])
        rair = b["rair_numerator"] / b["rair_denominator"] if b["rair_denominator"] > 0 else None
        rsr = b["rsr_numerator"] / b["rsr_denominator"] if b["rsr_denominator"] > 0 else None
        total = b["correct_follow"] + b["bad_follow"] + b["correct_override"] + b["bad_override"]

        out.append({
            "llm": llm,
            "target_accuracy": ta,
            "n_participants": b["n_participants"],
            "pooled_tp": b["final_tp"],
            "pooled_fp": b["final_fp"],
            "pooled_fn": b["final_fn"],
            "pooled_tn": b["final_tn"],
            "pooled_precision": round(p, 4),
            "pooled_recall": round(r, 4),
            "pooled_f1": round(f1, 4),
            "pooled_rair": round(rair, 4) if rair is not None else None,
            "rair_n": b["rair_denominator"],
            "pooled_rsr": round(rsr, 4) if rsr is not None else None,
            "rsr_n": b["rsr_denominator"],
            "correct_follow": b["correct_follow"],
            "bad_follow": b["bad_follow"],
            "correct_override": b["correct_override"],
            "bad_override": b["bad_override"],
            "total_decisions": total,
            "pct_correct_follow": round(b["correct_follow"] / total * 100, 1) if total else None,
            "pct_bad_follow": round(b["bad_follow"] / total * 100, 1) if total else None,
            "pct_correct_override": round(b["correct_override"] / total * 100, 1) if total else None,
            "pct_bad_override": round(b["bad_override"] / total * 100, 1) if total else None,
        })
    return out


# =============================================================================
# SUMMARY PRINTERS
# =============================================================================

def print_per_condition_summary(all_rows):
    by_cond = defaultdict(list)
    for r in all_rows:
        by_cond[r["condition_id"]].append(r)

    print("\n" + "=" * 112)
    print("PER-CONDITION SUMMARY (averaged over participants)")
    print("=" * 112)
    print(f"{'Condition':<12}{'N':<4}{'Init F1':<10}{'Final F1':<10}{'F1 Std':<9}{'Precision':<11}{'Recall':<9}{'RAIR':<8}{'RSR':<8}")
    print("-" * 112)

    cond_order = ["no_agent",
                  "gpt_40", "gpt_60", "gpt_80",
                  "q9b_40", "q9b_60", "q9b_80",
                  "q35_40", "q35_60", "q35_80"]

    for cond_id in cond_order:
        rows = by_cond.get(cond_id, [])
        if not rows:
            continue
        n = len(rows)
        final_f1_vals = [r["final_f1"] for r in rows]
        final_f1 = sum(final_f1_vals) / n
        final_f1_std = np.std(final_f1_vals)
        final_p = sum(r["final_precision"] for r in rows) / n
        final_r = sum(r["final_recall"] for r in rows) / n
        init_vals = [r["initial_f1"] for r in rows if r["initial_f1"] is not None]
        init_f1 = sum(init_vals) / len(init_vals) if init_vals else None
        rair_vals = [r["trajectory_rair"] for r in rows if r["trajectory_rair"] is not None]
        rsr_vals = [r["trajectory_rsr"] for r in rows if r["trajectory_rsr"] is not None]
        rair = sum(rair_vals) / len(rair_vals) if rair_vals else None
        rsr = sum(rsr_vals) / len(rsr_vals) if rsr_vals else None

        print(f"{cond_id:<12}{n:<4}"
              f"{(f'{init_f1:.3f}' if init_f1 is not None else '-'):<10}"
              f"{final_f1:<10.3f}{final_f1_std:<9.3f}{final_p:<11.3f}{final_r:<9.3f}"
              f"{(f'{rair:.3f}' if rair is not None else '-'):<8}"
              f"{(f'{rsr:.3f}' if rsr is not None else '-'):<8}")
    print()


def print_pooled_summary(pooled):
    print("\n" + "=" * 110)
    print("POOLED AGGREGATION (counts pooled across participants per cell)")
    print("=" * 110)
    print(f"{'LLM':<14}{'Tgt':<6}{'N':<4}{'F1':<8}{'P':<8}{'R':<8}{'RAIR':<14}{'RSR':<14}")
    print("-" * 110)
    for c in pooled:
        ta = f"{c['target_accuracy']:.1f}" if c['target_accuracy'] is not None else "-"
        rair = f"{c['pooled_rair']:.3f} (n={c['rair_n']})" if c['pooled_rair'] is not None else "-"
        rsr = f"{c['pooled_rsr']:.3f} (n={c['rsr_n']})" if c['pooled_rsr'] is not None else "-"
        print(f"{c['llm']:<14}{ta:<6}{c['n_participants']:<4}"
              f"{c['pooled_f1']:<8.3f}{c['pooled_precision']:<8.3f}{c['pooled_recall']:<8.3f}"
              f"{rair:<14}{rsr:<14}")
    print()

    print("=" * 110)
    print("OVERRIDE BEHAVIOR (% of agent-flagged decisions, pooled)")
    print("  Correct-Follow: agent right, human kept flag, student WAS at-risk")
    print("  Bad-Follow:     agent wrong, human kept flag, student NOT at-risk (false positive together)")
    print("  Correct-Override: agent wrong, human dismissed flag, student NOT at-risk")
    print("  Bad-Override:   agent right, human dismissed flag, student WAS at-risk (DANGEROUS MISS)")
    print("=" * 110)
    print(f"{'LLM':<14}{'Tgt':<6}{'N-dec':<7}{'CorrFol':<10}{'BadFol':<10}{'CorrOvr':<10}{'BadOvr':<10}")
    print("-" * 110)
    for c in pooled:
        if c['total_decisions'] == 0:
            continue
        ta = f"{c['target_accuracy']:.1f}" if c['target_accuracy'] is not None else "-"
        cf = f"{c['pct_correct_follow']}%"
        bf = f"{c['pct_bad_follow']}%"
        co = f"{c['pct_correct_override']}%"
        bo = f"{c['pct_bad_override']}%"
        print(f"{c['llm']:<14}{ta:<6}{c['total_decisions']:<7}{cf:<10}{bf:<10}{co:<10}{bo:<10}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=os.path.join(PROJECT_ROOT, "study_logs"))
    parser.add_argument("--out-per-participant",
                        default=os.path.join(PROJECT_ROOT, "results_per_participant.csv"))
    parser.add_argument("--out-pooled",
                        default=os.path.join(PROJECT_ROOT, "results_pooled.csv"))
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

    with open(args.out_per_participant, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-participant CSV: {args.out_per_participant}")

    pooled = aggregate_pooled(all_rows)
    with open(args.out_pooled, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pooled[0].keys()))
        writer.writeheader()
        writer.writerows(pooled)
    print(f"Pooled CSV:          {args.out_pooled}")

    print_per_condition_summary(all_rows)
    print_pooled_summary(pooled)


if __name__ == "__main__":
    main()
