"""
Exp 2 — Calibration Audit.

Scans every sim log under SIM_LOGS_ROOT and verifies that each episode's
calibrated predictions actually hit the target accuracy. Reports any drift.

Usage:
    python -m experiments.exp2_verify_calibration
    python -m experiments.exp2_verify_calibration --tolerance 0.02
"""

import argparse
import glob
import json
import os
from collections import defaultdict

from experiments.exp2_config import SIM_LOGS_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=SIM_LOGS_ROOT)
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Max allowed |achieved - target| before flagging drift.")
    args = parser.parse_args()

    if not os.path.isdir(args.logs):
        print(f"No logs directory at {args.logs}")
        return

    files = sorted(glob.glob(os.path.join(args.logs, "**", "*.json"), recursive=True))
    print(f"Auditing {len(files)} logs in {args.logs}")

    by_cell = defaultdict(list)   # (instructor, dataset, target) -> list of achieved
    drifts = []
    missing_stats = []
    no_predictions = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue

        stats = log.get("calibration_stats")
        if not stats:
            missing_stats.append(path)
            continue

        target = stats.get("target_accuracy")
        achieved = stats.get("achieved_accuracy")
        n_pred = stats.get("n_predicted", 0)

        instructor = log.get("instructor_llm")
        dataset = log.get("dataset")
        cell_key = (instructor, dataset, target)

        if achieved is None:
            no_predictions.append((path, stats))
            continue

        by_cell[cell_key].append(achieved)

        drift = abs(achieved - target) if target is not None else 0
        if drift > args.tolerance:
            drifts.append((path, target, achieved, drift, n_pred))

    # Summary
    print(f"\n{'cell':<48}{'n_runs':<8}{'target':<10}{'achieved (mean)':<18}{'achieved (range)'}")
    print("-" * 110)
    for (instructor, dataset, target), achieved_list in sorted(by_cell.items(),
                                                                key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2] or 0)):
        n = len(achieved_list)
        mean = sum(achieved_list) / n
        lo = min(achieved_list)
        hi = max(achieved_list)
        cell_label = f"{instructor or '?'} | {dataset or '?'}"
        print(f"{cell_label:<48}{n:<8}{target:<10.3f}{mean:<18.4f}{lo:.4f}–{hi:.4f}")

    if drifts:
        print(f"\n{len(drifts)} log(s) drifted more than {args.tolerance:.3f} from target:")
        for path, target, achieved, drift, n_pred in drifts[:20]:
            print(f"  {os.path.basename(path)}  target={target:.3f}  achieved={achieved:.4f}  drift={drift:.4f}  n_pred={n_pred}")
        if len(drifts) > 20:
            print(f"  ... and {len(drifts) - 20} more")
    else:
        print(f"\n✓ All achieved accuracies within ±{args.tolerance} of target.")

    if no_predictions:
        print(f"\n{len(no_predictions)} log(s) had NO predictions (k-NN errored on every student):")
        for path, stats in no_predictions[:10]:
            print(f"  {os.path.basename(path)}  errors={stats.get('n_errors')}/{stats.get('n_students_input')}")

    if missing_stats:
        print(f"\n{len(missing_stats)} log(s) had no calibration_stats field (older logs from before this audit):")
        for path in missing_stats[:5]:
            print(f"  {os.path.basename(path)}")
        if len(missing_stats) > 5:
            print(f"  ... and {len(missing_stats) - 5} more")


if __name__ == "__main__":
    main()
