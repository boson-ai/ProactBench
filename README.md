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
| `final_dialogues.jsonl` | 198 | **Main benchmark corpus** — 198 curated dialogues, 624 trigger points (201 Emergent / 232 Critical / 191 Recovery). Each trigger carries the Planner-authored rubric (pass/partial/fail criteria) but no curation-time judge score; per-model scored outputs live in [`eval/`](dataset/eval/). |
| `validated_blueprints.jsonl` | 207 | Audit-passing blueprints rolled forward to the dialogue stage. (210 of 250 received a PASS audit decision; 3 were dropped at the validated-blueprint write step due to post-audit format-validation failures.) |
| `blueprints.jsonl` | 250 | All generated blueprints (pre-audit). |
| `validation_results.jsonl` | 250 | Independent-judge audit decisions. |
| `tasks.jsonl` | 50 | Per-persona scenarios (Stage 1 output). |
| `selected_tasks.jsonl` | 19 | Curated subset of `tasks.jsonl` — 19 personas containing 25 scenarios that fed Stage 2 blueprint generation. |
| [`human_eval/`](dataset/human_eval/) | 18 raters / 275 ratings / 60 items | Human-validation study results (Prolific, pseudonymized). Per-annotator rating files, the 60-item stratified subsample, annotator briefing, instructions, and the analysis script that reproduces Krippendorff's $\alpha$ and Cohen's $\kappa_{\text{quad}}$ from the paper. |
| [`eval/`](dataset/eval/) | 16 models × (198 dialogues + metrics) | Per-model evaluation outputs: full assistant responses regenerated at each trigger turn, judge scores with rationales and evidence quotes, plus aggregate metrics. The raw inputs that feed every per-model number, ranking, and ablation in the paper. ~65 MB total. |

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

Load with plain JSONL parsing:

```python
import json
from pathlib import Path

dialogues = [json.loads(l) for l in
             Path("dataset/final_dialogues.jsonl").read_text().splitlines() if l.strip()]
print(len(dialogues), "dialogues")  # 198
```

Or via HuggingFace `datasets` (v3+, no custom loader script needed):

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="dataset/final_dialogues.jsonl", split="train")
print(len(ds), "dialogues")  # 198
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

Runs the evaluation pipeline on a small subset of dialogues end-to-end,
using a cheap evaluated model and a cheap judge model with
`reasoning_effort=low` for speed. See
[`scripts/smoke_test.py`](scripts/smoke_test.py) for the exact command and
expected output.

## Package layout

```
proactbench/
├── __init__.py             # Package entry, exposes `run_eval`
├── clients.py              # Minimal OpenAI / Gemini / Anthropic wrappers
├── types.py                # Pydantic models for triggers and judge output
├── evaluation.py           # Evaluation loop: rerun a model at trigger points + judge
└── prompts/
    ├── __init__.py
    └── runtime.py          # User Agent / judge system prompts and message builders
```

The released code includes only the **evaluation** pipeline. The synthesis
pipeline that produced the released `dataset/` artefacts (personas → scenarios
→ blueprints → validated blueprints → dialogues) is documented in the paper's
appendix and the dataset's [DATASHEET.md](dataset/DATASHEET.md), but the
generation code is not redistributed; the released corpus in `dataset/` is the
canonical artefact.

## Evaluation

> **Note on model roles.** Throughout the paper's main configuration, GPT-5.4
> serves as the **judge**. The *evaluated* model — the one whose proactivity
> is being scored — is specified separately via `--eval-model` in the
> evaluation command (the paper reports 16 evaluated models, with GPT-5.5
> leading). Don't confuse the judge model with the evaluated model.

The evaluation loop reads dialogues from
[`dataset/final_dialogues.jsonl`](dataset/final_dialogues.jsonl) (each
dialogue carries the Planner-authored rubric per trigger but no
curation-time judge labels), regenerates the assistant response at each
trigger turn with the model under evaluation, and has the judge apply the
original rubric in neutral mode (persona / style hidden at scoring time).

Python API:

```python
from pathlib import Path
from proactbench import run_eval

run_eval(
    results_path=Path("dataset/final_dialogues.jsonl"),
    output_path=Path("output/gpt55_eval.jsonl"),
    eval_model="gpt-5.5",          # the model under evaluation (paper's headline)
    judge_model="gpt-5.4",         # judge stays GPT-5.4 in the main configuration
    num_threads=4,
)
```

Equivalently via CLI:

```bash
python -m proactbench.evaluation \
  --results-path dataset/final_dialogues.jsonl \
  --output-path output/gpt55_eval.jsonl \
  --eval-model gpt-5.5 \
  --judge-model gpt-5.4
```

The reference per-model outputs from the paper's main sweep (16 evaluated
models × 198 dialogues each) are released in
[`dataset/eval/`](dataset/eval/) for direct comparison.

## Data formats

| File | Shape |
|---|---|
| `tasks.jsonl` | one row per persona: `{"uuid": ..., "<category>_scenarios": [scenario, ...]}` for each of 5 categories |
| `blueprints.jsonl` | one row per (persona, scenario, style) triple: full `BlueprintOutput` |
| `validated_blueprints.jsonl` | subset of `blueprints.jsonl` whose audit decision is PASS |
| `online_eval.jsonl` / `eval.jsonl` | one row per dialogue: trigger points, rubrics, scores, token usage |

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
