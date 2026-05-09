# Data schemas

ProactBench distributes one JSONL artefact and the offline-evaluation
pipeline writes one. Both are documented here. The Pydantic source of truth
lives in [`proactbench/types.py`](../proactbench/types.py).

## `final_dialogues.jsonl` — the released benchmark corpus

Hosted on HuggingFace at
[`bosonai/proactbench`](https://huggingface.co/datasets/bosonai/proactbench).

198 rows; one per curated dialogue. Each row carries:

```json
{
  "uuid": "531a542f-...",
  "unique_id_eval": "BP_PROFESSIONAL_01_VQ__style9__531a542f-...",
  "blueprint_id": "BP_PROFESSIONAL_01_VQ",
  "scenario_id": "PROFESSIONAL_01",
  "category_key": "professional_persona",
  "style_combination_index": 9,
  "evaluated_model": "gemini-2.5-pro",
  "num_turns_completed": 8,
  "trigger_points": [
    {
      "turn": 2,
      "evaluation_rubric": {
        "type": "EMERGENT",
        "pass_criteria": "...",
        "partial_criteria": "...",
        "fail_criteria": "..."
      }
    },
    ...
  ],
  "turn_records": [
    {
      "turn": 1,
      "planner": { ... },
      "user_message": "...",
      "assistant_response": "..."
    },
    ...
  ],
  "token_usage": {
    "planner":    {"prompt_tokens": ..., "completion_tokens": ..., "calls": ...},
    "user_agent": {"prompt_tokens": ..., "completion_tokens": ..., "calls": ...},
    "assistant":  {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
  }
}
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `uuid` | string | Original generation UUID. **Not unique on its own** — use `unique_id_eval` as the primary key when joining across files. |
| `unique_id_eval` | string | Composite primary key formatted as `{blueprint_id}__style{style_combination_index}__{uuid}`. Guaranteed unique across the corpus. |
| `blueprint_id` | string | Source-blueprint identifier (e.g. `BP_PROFESSIONAL_01_EVA`). |
| `scenario_id` | string | Source-scenario identifier (e.g. `PROFESSIONAL_01`). |
| `category_key` | enum | One of `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`. |
| `style_combination_index` | int | 1–24, indexing the CSI factorial style. The 24 retained binary combinations of the 6 CSI dimensions are listed in the paper's Appendix C. |
| `evaluated_model` | string | Model whose responses populated the dialogue at curation time. `gemini-2.5-pro` for the entire released corpus. |
| `num_turns_completed` | int | 5–10. Minimum enforced by the curation loop. |
| `trigger_points` | array | One entry per trigger turn. Each entry has only `turn` (1-indexed) and `evaluation_rubric`. **No curation-time judge label is included.** Per-(model, trigger) labels are produced at run time by the offline judge in [`proactbench/evaluation.py`](../proactbench/evaluation.py). |
| `trigger_points[i].evaluation_rubric` | object | `{type, pass_criteria, partial_criteria, fail_criteria}`, authored prospectively by the Planner before the assistant responded. `type` ∈ {`EMERGENT`, `CRITICAL`, `RECOVERY`}. |
| `turn_records` | array | Per-turn record. Each entry has `turn`, `user_message`, `assistant_response`, plus a `planner` sub-object capturing the curation-time Planner's state at that turn. The offline pipeline reads only `turn` / `user_message` / `assistant_response`; the `planner` block is preserved for inspection but otherwise ignored. |
| `token_usage` | object | Per-agent (`planner`, `user_agent`, `assistant`) prompt / completion / call counts from the curation run. Informational only; the offline pipeline writes its own `assistant` + `judge` block in the output. (Note: in the released corpus the curation `assistant` counts are zero because the curation-time evaluated model was tracked separately.) |

The Pydantic models that the offline pipeline parses these against are
`EvaluationRubric` and `TriggerPoint` in
[`proactbench/types.py`](../proactbench/types.py).

## Offline-evaluation output

`proactbench.run_eval` writes one row per dialogue to the user-supplied
output path. The shape mirrors `final_dialogues.jsonl` plus the freshly
produced judge labels and per-turn diff records:

```json
{
  "blueprint_id": "...",
  "scenario_id": "...",
  "uuid": "...",
  "category_key": "...",
  "style_combination_index": 9,
  "evaluated_model": "gpt-5.5",
  "source_evaluated_model": "gemini-2.5-pro",
  "num_trigger_points": 3,
  "trigger_stats": {
    "EMERGENT": {"PASS": 1, "PARTIAL": 0, "FAIL": 0, "SKIPPED": 0},
    "CRITICAL": {"PASS": 1, "PARTIAL": 0, "FAIL": 0, "SKIPPED": 0},
    "RECOVERY": {"PASS": 0, "PARTIAL": 1, "FAIL": 0, "SKIPPED": 0}
  },
  "trigger_points": [
    {
      "turn": 2,
      "evaluation_rubric": {"type": "EMERGENT", "pass_criteria": "...", ...},
      "evaluation_result": {
        "status": "EVALUATED",
        "score": "PASS",
        "rationale": "...",
        "evidence": "<verbatim quote from the assistant>"
      }
    },
    ...
  ],
  "turn_records": [
    {
      "turn": 2,
      "user_message": "...",
      "assistant_response": "<regenerated by the model under evaluation>",
      "original_assistant_response": "<curation-time response from gemini-2.5-pro>"
    },
    ...
  ],
  "token_usage": {
    "assistant": {"prompt_tokens": ..., "completion_tokens": ..., "calls": ...},
    "judge":     {"prompt_tokens": ..., "completion_tokens": ..., "calls": ...}
  }
}
```

### Field reference (offline output)

| Field | Type | Description |
|---|---|---|
| `evaluated_model` | string | The model under test (matches the value passed to `run_eval`). |
| `source_evaluated_model` | string | Model that produced the curation-time conversation (`gemini-2.5-pro` for the released corpus). |
| `num_trigger_points` | int | Number of trigger points scored in this dialogue. |
| `trigger_stats` | object | Per-trigger-type Pass / Partial / Fail / Skipped counts. |
| `trigger_points[i].evaluation_result` | object | The judge's label. Schema: `JudgeOutput.evaluation_result` — `{status, score, rationale, evidence}`. `status` ∈ {`EVALUATED`, `SKIPPED`}; `score` ∈ {`PASS`, `PARTIAL`, `FAIL`}; `evidence` is a verbatim quote from the regenerated response. |
| `turn_records[i].assistant_response` | string | Freshly regenerated response from the evaluated model. |
| `turn_records[i].original_assistant_response` | string | The curation-time response from `source_evaluated_model`, included for diff against the freshly regenerated one. |
| `token_usage` | object | Per-agent (`assistant`, `judge`) prompt and completion token counts plus call counts. The curation-time `planner` / `user_agent` blocks are not present (those agents only run at curation time). |

The Pydantic types are `JudgeOutput` and `EvaluationResult` in
[`proactbench/types.py`](../proactbench/types.py).

## Score semantics

Trigger scores are one of `PASS`, `PARTIAL`, `FAIL`, or `SKIPPED` (the last
when the pipeline could not reach the trigger turn — for example if the
evaluated model truncated the dialogue early).

Aggregation conventions:

- **Weighted score**: `Pass=1.0`, `Partial=0.5`, `Fail=0.0`, `Skipped`
  excluded from the denominator.
- **Pass rate**: counts only `PASS` over all evaluated triggers.

Per-trigger-type and overall headline numbers are reported with both
metrics in the paper.
