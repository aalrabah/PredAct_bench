"""
Reads annotation scores from all participant_* folders inside annotation_sample/,
handling two different fill-in formats:

  Format A (e.g. participant_1): scores in the SCORE row, columns 4 & 5
      SCORE,,,<score_1>,<score_2>

  Format B (e.g. participant_2): scores in the RUBRIC rows, column 3
      RUBRIC,Score 1 ...,<score_1>,,
      RUBRIC,Score 2 ...,<score_2>,,

Joins with each folder's log_mapping.csv and outputs:
  behavior_analysis/annotation_scores.csv
"""

import csv, glob, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR   = os.path.join(PROJECT_ROOT, "behavior_analysis", "annotation_sample")
OUT_PATH     = os.path.join(PROJECT_ROOT, "behavior_analysis", "annotation_scores.csv")


def extract_scores(csv_path):
    """Return (score_1, score_2) from a log CSV, handling both formats."""
    score_1 = score_2 = ""
    rubric_scores = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            tag = row[0].strip().upper()

            if tag == "SCORE":
                # Format A: SCORE,,,<s1>,<s2>
                s1 = row[3].strip() if len(row) > 3 else ""
                s2 = row[4].strip() if len(row) > 4 else ""
                if s1 or s2:
                    score_1, score_2 = s1, s2

            elif tag == "RUBRIC":
                # Format B: RUBRIC,<desc>,<score>,,
                val = row[2].strip() if len(row) > 2 else ""
                if val:
                    rubric_scores.append(val)

    # Format B wins if SCORE row was empty but RUBRIC rows had values
    if (not score_1 and not score_2) and len(rubric_scores) >= 2:
        score_1, score_2 = rubric_scores[0], rubric_scores[1]
    elif (not score_1 and not score_2) and len(rubric_scores) == 1:
        score_1 = rubric_scores[0]

    return score_1, score_2


def load_mapping(folder):
    mapping_path = os.path.join(folder, "log_mapping.csv")
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mapping[row["log_id"]] = row
    return mapping


def main():
    rows = []
    participant_dirs = sorted(glob.glob(os.path.join(SAMPLE_DIR, "participant_*")))

    for pdir in participant_dirs:
        participant = os.path.basename(pdir)
        mapping = load_mapping(pdir)

        for csv_path in sorted(glob.glob(os.path.join(pdir, "log_[0-9]*.csv"))):
            log_id = os.path.splitext(os.path.basename(csv_path))[0]
            score_1, score_2 = extract_scores(csv_path)
            meta = mapping.get(log_id, {})
            rows.append({
                "participant":   participant,
                "log_id":        log_id,
                "score_1":       score_1,
                "score_2":       score_2,
                "source":        meta.get("source", ""),
                "condition":     meta.get("condition", ""),
                "human_participant": meta.get("participant", ""),
                "n_turns":       meta.get("n_turns", ""),
            })

    fieldnames = ["participant", "log_id", "score_1", "score_2",
                  "source", "condition", "human_participant", "n_turns"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Written {len(rows)} rows to {OUT_PATH}")
    for r in rows:
        print(f"  {r['participant']}  {r['log_id']}  s1={r['score_1']}  s2={r['score_2']}  {r['source']:<6}  {r['condition']}")


if __name__ == "__main__":
    main()
