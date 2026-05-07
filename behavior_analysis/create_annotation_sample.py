"""
Creates 25 anonymized conversation CSVs for human annotation.

  12 from results/exp2/sim_logs (predact_cs dataset only), randomly sampled
  13 from study_logs (one condition per participant, randomly sampled)

Each CSV has columns: turn, instructor_turn, assistant_turn
Plus a single SCORE row at the bottom with score_1 and score_2 for the annotator.

Outputs (all in behavior_analysis/annotation_sample/):
  log_001.csv ... log_025.csv   — anonymized conversation files
  log_mapping.csv               — maps log ID -> source file + condition
"""

import argparse, csv, glob, json, os, random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_LOGS     = os.path.join(PROJECT_ROOT, "results", "exp2", "sim_logs", "**", "*.json")
HUMAN_LOGS   = os.path.join(PROJECT_ROOT, "study_logs", "*.json")

RUBRIC_SCORE1 = (
    "Score 1 — Verification quality: "
    "0=decided without verifying | "
    "1=verified some flagged students | "
    "2=verified every flagged student"
)
RUBRIC_SCORE2 = (
    "Score 2 — Question quality: "
    "0=vague/repeated/generic | "
    "1=specific but no follow-up | "
    "2=specific + builds on prior answers + follow-ups"
)


def write_log_csv(path, turns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["turn", "instructor_turn", "assistant_turn", "score_1", "score_2"])
        for i, (inst, asst) in enumerate(turns, 1):
            w.writerow([i, inst, asst, "", ""])
        # blank row separator
        w.writerow(["", "", "", "", ""])
        # rubric rows
        w.writerow(["RUBRIC", RUBRIC_SCORE1, "", "", ""])
        w.writerow(["RUBRIC", RUBRIC_SCORE2, "", "", ""])
        # score row for annotator to fill in
        w.writerow(["SCORE", "", "", "", ""])


def pair_turns(messages, instructor_role, assistant_role):
    """Pair instructor and assistant messages in order."""
    pairs = []
    inst_msgs = [m["content"] for m in messages if m["role"] == instructor_role]
    asst_msgs = [m["content"] for m in messages if m["role"] == assistant_role]
    for i in range(max(len(inst_msgs), len(asst_msgs))):
        inst = inst_msgs[i] if i < len(inst_msgs) else ""
        asst = asst_msgs[i] if i < len(asst_msgs) else ""
        pairs.append((inst, asst))
    return pairs


def collect_sim_candidates():
    candidates = []
    for path in sorted(glob.glob(SIM_LOGS, recursive=True)):
        with open(path) as f:
            d = json.load(f)
        if d.get("dataset") != "predact_cs":
            continue
        candidates.append({
            "source":    "llm",
            "path":      path,
            "condition": f"{d.get('instructor_llm')}_{d.get('target_accuracy')}",
            "messages":  d.get("dialogue_history", []),
            "inst_role": "instructor",
            "asst_role": "assistant",
        })
    return candidates


def collect_human_candidates():
    candidates = []
    for path in sorted(glob.glob(HUMAN_LOGS)):
        with open(path) as f:
            log = json.load(f)
        name = log.get("participant", {}).get("name", os.path.basename(path))
        for cond in log.get("conditions", []):
            if not cond.get("has_agent"):
                continue
            candidates.append({
                "source":      "human",
                "path":        path,
                "condition":   cond.get("condition_id"),
                "participant": name,
                "messages":    cond.get("chat_history", []),
                "inst_role":   "user",
                "asst_role":   "assistant",
            })
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", default="", help="subfolder inside annotation_sample/")
    parser.add_argument("--seed",   type=int, default=99)
    args = parser.parse_args()

    random.seed(args.seed)
    base = os.path.join(PROJECT_ROOT, "behavior_analysis", "annotation_sample")
    OUT_DIR = os.path.join(base, args.subdir) if args.subdir else base
    os.makedirs(OUT_DIR, exist_ok=True)

    sim_candidates   = collect_sim_candidates()
    human_candidates = collect_human_candidates()

    print(f"Available sim (predact_cs): {len(sim_candidates)}")
    print(f"Available human:      {len(human_candidates)}")

    sim_sample   = random.sample(sim_candidates,   12)
    human_sample = random.sample(human_candidates, 13)
    all_samples  = sim_sample + human_sample
    random.shuffle(all_samples)  # mix so IDs don't reveal source

    mapping_rows = []
    for idx, item in enumerate(all_samples, 1):
        log_id   = f"log_{idx:03d}"
        csv_path = os.path.join(OUT_DIR, f"{log_id}.csv")

        turns = pair_turns(item["messages"], item["inst_role"], item["asst_role"])
        write_log_csv(csv_path, turns)

        mapping_rows.append({
            "log_id":        log_id,
            "source":        item["source"],
            "original_path": item["path"],
            "condition":     item["condition"],
            "participant":   item.get("participant", ""),
            "n_turns":       len(turns),
        })

    # Write mapping file
    mapping_path = os.path.join(OUT_DIR, "log_mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["log_id", "source", "original_path",
                                           "condition", "participant", "n_turns"])
        w.writeheader()
        w.writerows(mapping_rows)

    print(f"\nWritten to {OUT_DIR}/")
    print(f"  {len(all_samples)} log CSVs")
    print(f"  log_mapping.csv")
    for r in mapping_rows:
        print(f"  {r['log_id']}  {r['source']:<6}  {r['condition']:<40}  {r['n_turns']} turns")


if __name__ == "__main__":
    main()
