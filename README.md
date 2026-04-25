# ProactBench

Benchmark code and data-generation pipeline for
**ProactBench: What the User Didn't Ask** — measuring conversational
proactivity in multi-turn LLM dialogues.

The benchmark decomposes proactivity into three phase-tied types —
**Emergent**, **Critical**, and **Recovery** — and evaluates them with a
three-agent architecture (Planner, User Agent, Evaluated Model) whose
information asymmetries defend against rubric leakage, post-hoc
rationalization, style-confounded scoring, and information dumps.

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

```bash
# Stage 1: scenarios (hidden goal + anchors + ideal trajectory)
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
    output_path=Path("output/offline_gemini_as_eval.jsonl"),
    eval_model="gemini-2.5-pro",  # new model under evaluation
    judge_model="gpt-5.4",
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



## License

MIT — see [LICENSE](LICENSE).
