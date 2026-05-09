# ProactBench

Code and data for **ProactBench: Beyond What The User Asked For** — measuring
conversational proactivity in multi-turn LLM dialogues.

The benchmark decomposes proactivity into three phase-tied trigger types —
**Emergent**, **Critical**, and **Recovery** — and ships as (i) a curated
dialogue corpus with prospective per-trigger rubrics and (ii) an offline
evaluation pipeline that re-runs any model under test at each trigger turn
and scores the response against the rubric via an LLM judge.

## Dataset

The released benchmark lives in [`dataset/`](dataset/):

| File | Rows | Description |
|---|---|---|
| `final_dialogues.jsonl` | 198 | **The benchmark corpus.** 198 curated dialogues, 624 trigger points (201 Emergent / 232 Critical / 191 Recovery). Each trigger carries the Planner-authored rubric (pass / partial / fail criteria) but no curation-time judge label; per-(model, trigger) labels are produced at run time by the offline judge in [`proactbench/evaluation.py`](proactbench/evaluation.py). |

The dataset ships with [Croissant 1.1](http://mlcommons.org/croissant/)
metadata at [`dataset/metadata.json`](dataset/metadata.json) and a Gebru-style
datasheet at [`dataset/DATASHEET.md`](dataset/DATASHEET.md). The schema for
`final_dialogues.jsonl` and the offline-evaluation output is documented in
[`docs/DATA_SCHEMAS.md`](docs/DATA_SCHEMAS.md), with
[`proactbench/types.py`](proactbench/types.py) as the canonical Pydantic
source of truth.

The benchmark is released under the Apache 2.0 licence; persona-derived
content inherits the upstream Nemotron-Personas-USA CC-BY-4.0 licence.

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

## What the released code does

The released code is the **offline evaluation pipeline** only. Given the
released corpus, it:

1. Reads each dialogue from `dataset/final_dialogues.jsonl`.
2. At every trigger turn, regenerates the assistant response using the
   model under evaluation (the conversation history up to that turn is the
   context).
3. Scores that response with the judge against the trigger's prospective
   rubric (PASS / PARTIAL / FAIL, plus rationale and verbatim evidence
   quote).
4. Writes one JSONL row per dialogue with the regenerated responses,
   judge labels, and per-agent token usage.

The synthesis-stage code (curation-time Planner / User Agent loop, scenario
and blueprint generation, blueprint-audit judge) is **not** redistributed.
The released corpus in `dataset/` is the canonical artefact every paper
number is computed against; the synthesis methodology is described in the
paper's appendix.

## Install

```bash
pip install -e .
```

Python ≥ 3.10.

You need API credentials for at least the evaluated model and the judge.
Each client picks its key up from the corresponding env var:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export ANTHROPIC_API_KEY=sk-ant-...
```

`make_client(model)` routes by name:

| Model name | Client |
|---|---|
| `gpt-*`, `o1*`, `o3*`, `o4*` | OpenAI |
| `claude-*` | Anthropic |
| contains `gemini` | Gemini |
| `kimi*`, `moonshot*`, `deepseek*` | Their respective OpenAI-compatible endpoints |
| anything else + `base_url=...` | OpenAI-compatible endpoint (e.g. a vLLM server) |

## Quick smoke test

```bash
python scripts/smoke_test.py --eval-model gpt-4o --judge-model gpt-4o
```

Reads the first 2 dialogues from `dataset/final_dialogues.jsonl`, regenerates
each trigger response with the evaluated model, scores it with the judge,
and verifies the output shape end-to-end. Defaults to OpenAI for both
roles (only `OPENAI_API_KEY` needed). See [`scripts/smoke_test.py`](scripts/smoke_test.py)
for full options.

## Package layout

```
proactbench/
├── __init__.py             # Package entry, exposes run_eval and make_client
├── clients.py              # Minimal OpenAI / Gemini / Anthropic / OpenAI-compatible wrappers
├── types.py                # Pydantic models: EvaluationRubric, EvaluationResult, TriggerPoint, JudgeOutput
├── evaluation.py           # Offline-eval loop: rerun a model at trigger points + judge against the rubric
└── prompts/
    ├── __init__.py
    └── runtime.py          # JUDGE_SYSTEM_TEMPLATE + build_judge_eval_message
```

## Evaluation

> **Note on model roles.** The paper's main configuration uses GPT-5.4 as
> the **judge**. The *evaluated* model — the one whose proactivity is being
> scored — is specified separately via `--eval-model`. Don't confuse the
> two.

The offline judge sees only the prospective rubric and the dialogue history
ending with the regenerated assistant response — no persona, no
communication style, no blueprint, no scenario package. This is the
information asymmetry that defends the benchmark against rubric leakage and
style-confounded scoring (see paper Section 3.2).

Python API:

```python
from pathlib import Path
from proactbench import run_eval

run_eval(
    results_path=Path("dataset/final_dialogues.jsonl"),
    output_path=Path("output/gpt55_eval.jsonl"),
    eval_model="gpt-5.5",          # the model under evaluation
    judge_model="gpt-5.4",         # judge stays GPT-5.4 in the paper's main config
    num_threads=4,
)
```

CLI:

```bash
python -m proactbench.evaluation \
  --results-path dataset/final_dialogues.jsonl \
  --output-path output/gpt55_eval.jsonl \
  --eval-model gpt-5.5 \
  --judge-model gpt-5.4
```

To use a vLLM (or any OpenAI-compatible) endpoint for either role:

```bash
python -m proactbench.evaluation \
  --results-path dataset/final_dialogues.jsonl \
  --output-path output/my_eval.jsonl \
  --eval-model my-served-model \
  --eval-base-url http://localhost:8000/v1 \
  --judge-model gpt-5.4
```

## Output format

The offline-evaluation pipeline writes one row per dialogue. Each row carries
the regenerated assistant responses, the judge's labels (with rationale and
verbatim evidence), per-trigger-type Pass/Partial/Fail/Skipped counts, and
per-agent token usage. Full schema:
[`docs/DATA_SCHEMAS.md`](docs/DATA_SCHEMAS.md).

Aggregation convention: `Pass=1.0`, `Partial=0.5`, `Fail=0.0` for the
weighted score; pass rate counts only `PASS`.

## Citation

```bibtex
@inproceedings{anonymous2026proactbench,
  title={ProactBench: Beyond What The User Asked For},
  author={Anonymous},
  booktitle={Under review at NeurIPS 2026 (Datasets and Benchmarks)},
  year={2026}
}
```

(BibTeX will be updated with author and venue details on acceptance.)

## Licence

Apache 2.0 — see [LICENSE](LICENSE). Persona-derived content inherits the
upstream Nemotron-Personas-USA CC-BY-4.0 licence.
