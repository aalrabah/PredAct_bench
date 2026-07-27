# PredAct-CS Dataset

PredAct-CS is the synthetic student-records dataset of PredAct-Bench: 53,401 students
across 60 computer science courses at a large public research university. Each course
provides a real syllabus structure (assignment names, weights, weekly schedule) and each
student a real final letter grade; weekly per-assignment scores are synthetically
generated to produce realistic trajectories that terminate at each student's real final
grade. Course identifiers are anonymized (e.g., `C4-20`); no real course numbers, student
identifiers, or personal information appear anywhere in the data.

## Files

| File | Description |
|---|---|
| `students_data.csv` | Source dataset. One row per student: course id, course-level grade statistics, and weekly assignment slots (name, type, weight, score) for up to 16 weeks. |
| `cs_db.json.gz` | Full benchmark database built from `students_data.csv` (all 60 courses, grouped by course). |
| `cs_db_train.json.gz` | Training pool only — the historical students the k-NN predictor searches. No overlap with test students. |
| `test_sets/` | 776 evaluation files, one per (course, cutoff week): `C3-01_week8.json` contains that course's test students truncated at week 8. |
| `ground_truth_for_cutoff_data.json` | Answer key: final grades for every test set. |
| `convert_to_json.py` | Builds `cs_db.json` from `students_data.csv`. |
| `split_data.py` | Builds the train/test split, test sets, and ground truth from `cs_db.json`. |

## Usage

The released JSON files are exactly the ones used in the paper's experiments (course
identifiers renamed). To rebuild from source:

```
python convert_to_json.py --students students_data.csv --output cs_db.json
python split_data.py --db cs_db.json --output-dir .
```

Decompress the databases with `gunzip cs_db.json.gz cs_db_train.json.gz`.

## License

Released under CC BY 4.0.
