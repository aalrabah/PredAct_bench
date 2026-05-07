# PredAct

PredAct is a benchmark for evaluating LLM-based instructor agents that predict student performance and recommend interventions from gradebook data. Agents are tested on two datasets: **PredAct-CS** (a real CS course dataset) and **OULAD** (the Open University Learning Analytics Dataset).

---

## Project Structure

```
PredAct/
├── app.py              # Human study UI (Streamlit)
├── tools.py            # Agent tool definitions
├── prompts.py          # Prompt templates
├── config.py           # Paths and constants
├── state.py            # Dialogue state management
├── tod.py              # Task-oriented dialogue engine
│
├── data/               # Data pipeline scripts
│   ├── data_generator.py           # Generate synthetic OULAD-format data
│   ├── convert_to_json.py          # Convert PredAct-CS raw data → JSON
│   ├── convert_oulad_to_json.py    # Convert OULAD CSVs → JSON
│   ├── split_data.py               # Split data into train/test sets
│   └── oulad_converter.py          # OULAD format utilities
│
├── sim/                # Simulation framework
│   ├── episode.py          # Single agent episode
│   ├── instructor_agent.py # LLM instructor agent
│   ├── assistant_agent.py  # Student-side assistant
│   ├── accuracy_injector.py
│   ├── evaluate_episode.py
│   └── schemas.py
│
├── experiments/        # Experiment sweep scripts
│   ├── exp1_tool_characterization.py
│   ├── exp2_sim_sweep.py
│   ├── exp2_aggregate.py
│   └── ...
│
├── eval/               # Evaluation scripts
│   ├── evaluate.py
│   ├── evaluate_human_study.py
│   └── evaluate_human_study_v2.py
│
├── plots/              # Figure generation (18 scripts)
│   └── plot_*.py
│
├── behavior_analysis/  # Agent behavior analysis
├── results/            # Generated experiment outputs
└── figures/            # Generated figures
```

---

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Set your API key in a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

---

## Quickstart

### 1. Prepare data

**OULAD** (public dataset — download separately, then convert):
```bash
python data/convert_oulad_to_json.py --oulad-dir data/oulad --output results/oulad/oulad_db.json
```

**PredAct-CS** (convert raw gradebook CSV to JSON, then split):
```bash
python data/convert_to_json.py
python data/split_data.py
```

Or generate synthetic data for testing:
```bash
python data/data_generator.py
```

### 2. Run the human study UI

```bash
streamlit run app.py
```

### 3. Run experiments

```bash
# Experiment 1 — tool characterization
python experiments/exp1_tool_characterization.py

# Experiment 2 — LLM sweep (edit exp2_config.py for model/dataset settings first)
python experiments/exp2_sim_sweep.py
python experiments/exp2_aggregate.py
```

### 4. Evaluate results

```bash
# Agent evaluation
python eval/evaluate.py

# Human study evaluation
python eval/evaluate_human_study_v2.py --logs-dir study_logs
```

### 5. Generate figures

Each plot script reads from `results/` and writes to `figures/`:

```bash
python plots/plot_f1_lines_faceted.py
python plots/plot_override.py
python plots/plot_rair_slope.py
# ... (run any plot_*.py independently)
```

---
