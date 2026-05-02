# ProactBench

Benchmark code and data-generation pipeline for
**ProactBench: What the User Didn't Ask** — measuring conversational
proactivity in multi-turn LLM dialogues.

The benchmark decomposes proactivity into three phase-tied types —
**Emergent**, **Critical**, and **Recovery** — and evaluates them with a
three-agent architecture (Planner, User Agent, Evaluated Model) whose
information asymmetries defend against rubric leakage, post-hoc
rationalization, style-confounded scoring, and information dumps.

## Dataset

The released benchmark lives in [`dataset/`](dataset/):

| File | Rows | Description |
|---|---|---|
| `final_dialogues.jsonl` | 198 | **Main benchmark corpus** — 198 curated dialogues, 624 trigger points (201 Emergent / 232 Critical / 191 Recovery), each scored at curation time. |
| `validated_blueprints.jsonl` | 210 | Audit-passing blueprints (input to the dialogue rollout). |
| `blueprints.jsonl` | 250 | All generated blueprints (pre-audit). |
| `validation_results.jsonl` | 250 | Independent-judge audit decisions. |
| `tasks.jsonl` | 50 | Per-persona scenarios (Stage 1 output). |
| `selected_tasks.jsonl` | 50 | Curated subset of `tasks.jsonl`. |
| [`human_eval/`](dataset/human_eval/) | 18 raters / 275 ratings / 60 items | Human-validation study results (Prolific, pseudonymized). Per-annotator rating files, the 60-item stratified subsample, annotator briefing, instructions, and the analysis script that reproduces Krippendorff's $\alpha$ and Cohen's $\kappa_{\text{quad}}$ from the paper. |

The dataset ships with [Croissant 1.0](http://mlcommons.org/croissant/)
metadata at [`dataset/metadata.json`](dataset/metadata.json) and a full
datasheet at [`dataset/DATASHEET.md`](dataset/DATASHEET.md). The schema for
every file is documented in
[`docs/DATA_SCHEMAS.md`](docs/DATA_SCHEMAS.md), with `proactbench/types.py`
as the canonical Pydantic source.

The benchmark is released under the Apache 2.0 License; persona-derived
content inherits the upstream Nemotron-Personas-USA CC-BY-4.0 license. After
acceptance, the dataset will additionally be hosted at a long-term artifact
registry (HuggingFace Datasets) for discoverability.

```python
import json
from pathlib import Path

dialogues = [json.loads(l) for l in
             Path("dataset/final_dialogues.jsonl").read_text().splitlines() if l.strip()]
print(len(dialogues), "dialogues")  # 198
```

## Install

```bash
pip install -e .
```

Python ≥ 3.10.

You need API credentials for at least the models you actually call. Each
client picks its key up from the corresponding env var:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export ANTHROPIC_API_KEY=sk-ant-...
```

The `make_client(model)` factory routes on the model name:

| Model prefix | Client |
|---|---|
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI |
| `claude-*` | Anthropic |
| contains `gemini` | Gemini |
| anything + `base_url=...` | OpenAI-compatible endpoint (e.g. a vLLM server) |

## Quick smoke test

```bash
# OpenAI-only: cheap, single key needed.
python scripts/smoke_test.py --validate-model o4-mini --eval-model gpt-4o

# Mixed-provider default (requires GEMINI_API_KEY):
python scripts/smoke_test.py
```

Runs the full pipeline on one persona and one blueprint end-to-end
(task → blueprint → validation → online eval → offline eval), using small
models and `reasoning_effort=low` for speed. See
[`scripts/smoke_test.py`](scripts/smoke_test.py) for the exact command and
expected output. On success you should see five `stage N OK` lines ending with
`✓ SMOKE TEST PASSED`.

## Package layout

```
proactbench/
├── clients.py              # Minimal OpenAI/Gemini/Anthropic wrappers
├── types.py                # All Pydantic models (runtime + synthesis)
├── data.py                 # Persona / blueprint / task loaders
├── styles.py               # 24 CSI communication styles
├── prompts/
│   ├── runtime.py          # Planner + User Agent system prompts
│   └── synthesis.py        # Task / blueprint / validation prompts
├── synthesis/
│   ├── generate_tasks.py       # Stage 1 — scenarios from personas
│   ├── generate_blueprints.py  # Stage 2 — blueprints from scenarios × styles
│   └── validate_blueprints.py  # Stage 3 — independent-judge audit
└── evaluation/
    ├── online.py           # Full three-agent dialogue loop
    └── offline.py          # Rerun model at trigger points, score with judge
```

## End-to-end pipeline

Each stage is a module with a CLI. A full run produces personas → tasks →
blueprints → validated_blueprints → dialogues → offline scores.

### 1. Synthesis

> **Note on model roles.** Throughout the paper's main configuration, GPT-5.4
> serves as the **Planner**, **User Agent**, and **offline judge**. The
> *evaluated* model — the one whose proactivity is being scored — is specified
> separately via `--eval-model` in the offline-evaluation command (the paper
> reports 16 evaluated models, with GPT-5.5 leading). Don't confuse the
> orchestration model with the evaluated model.

```bash
# Stage 1: scenarios (hidden goal + anchors + ideal trajectory).
# `--model` here is the Planner.
python -m proactbench.synthesis.generate_tasks \
  --num-personas 50 \
  --output-path data/tasks.jsonl \
  --model gpt-5.4 \
  --num-scenarios 5

# Stage 2: turn-by-turn blueprints × communication styles
python -m proactbench.synthesis.generate_blueprints \
  --tasks-path data/tasks.jsonl \
  --num-personas 50 \
  --output-path data/blueprints.jsonl \
  --model gpt-5.4 \
  --num-styles-per-task 10

# Stage 3: independent-judge audit; writes validated_blueprints.jsonl alongside
python -m proactbench.synthesis.validate_blueprints \
  --blueprints-path data/blueprints.jsonl \
  --tasks-path data/tasks.jsonl \
  --num-personas 50 \
  --output-path data/validation_results.jsonl \
  --model gemini-2.5-pro
```

### 2. Online evaluation (full three-agent dialogue)

```python
from pathlib import Path
from proactbench.evaluation import run_online_eval

run_online_eval(
    blueprints_path=Path("data/validated_blueprints.jsonl"),
    output_path=Path("output/online_eval.jsonl"),
    evaluated_model="gemini-2.5-pro",
    planner_model="gpt-5.4",
    user_agent_model="gpt-5.4",
    tasks_path=Path("data/tasks.jsonl"),
    num_personas=50,
    num_turns=10,
    num_threads=4,
)
```

### 3. Offline evaluation (rescore existing dialogues with a new model)

```python
from pathlib import Path
from proactbench.evaluation import run_offline_eval

run_offline_eval(
    results_path=Path("output/online_eval.jsonl"),
    output_path=Path("output/offline_gpt55_eval.jsonl"),
    eval_model="gpt-5.5",          # the model under evaluation (the paper's headline)
    judge_model="gpt-5.4",         # judge stays GPT-5.4 in the main configuration
    num_threads=4,
)
```

Equivalently via CLI:

```bash
python -m proactbench.evaluation.offline \
  --results-path output/online_eval.jsonl \
  --output-path output/offline_gemini_as_eval.jsonl \
  --eval-model gemini-2.5-pro \
  --judge-model gpt-5.4
```

## Data formats

| File | Shape |
|---|---|
| `tasks.jsonl` | one row per persona: `{"uuid": ..., "<category>_scenarios": [scenario, ...]}` for each of 5 categories |
| `blueprints.jsonl` | one row per (persona, scenario, style) triple: full `BlueprintOutput` |
| `validated_blueprints.jsonl` | subset of `blueprints.jsonl` whose audit decision is PASS |
| `online_eval.jsonl` / `offline_eval.jsonl` | one row per dialogue: trigger points, rubrics, scores, token usage |

Pydantic schemas for all of the above live in [`proactbench/types.py`](proactbench/types.py).

## Citation

```bibtex
@inproceedings{anonymous2026proactbench,
  title={ProactBench: What the User Didn't Ask},
  author={Anonymous},
  booktitle={Under review at NeurIPS 2026 (Datasets and Benchmarks)},
  year={2026}
}
```

(BibTeX will be updated with author/venue details on acceptance.)

## License

Apache 2.0 — see [LICENSE](LICENSE). Persona-derived content inherits the upstream Nemotron-Personas-USA CC-BY-4.0 license.
