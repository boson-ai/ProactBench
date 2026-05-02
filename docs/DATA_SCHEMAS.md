# Data schemas

Every JSONL artifact produced by ProactBench is one row per line with a
stable schema. Pydantic models for every structure live in
[`proactbench/types.py`](../proactbench/types.py).

The released benchmark corpus lives in [`dataset/`](../dataset/). Six files:

| File | Stage | Rows |
|---|---|---|
| `tasks.jsonl` | 1 — scenario synthesis | 50 (one per persona) |
| `blueprints.jsonl` | 2 — turn-by-turn plans | 250 |
| `validation_results.jsonl` | 3 — independent-judge audit decisions | 250 |
| `validated_blueprints.jsonl` | 3 — audit-passing subset | 210 |
| `selected_tasks.jsonl` | curated subset of `tasks.jsonl` | 50 |
| `final_dialogues.jsonl` | 4 — main benchmark corpus | **198** |

The dataset is also indexed by [`dataset/metadata.json`](../dataset/metadata.json)
in the [Croissant 1.0](http://mlcommons.org/croissant/) schema for
machine-readable discovery, and described in [`dataset/DATASHEET.md`](../dataset/DATASHEET.md).

## `tasks.jsonl` — Stage 1 output

Each row is one persona; its `<category>_scenarios` field holds the list of
`ProactiveScenario` objects generated for that life domain.

```json
{
  "uuid": "531a542f-…",
  "professional_persona_scenarios": [ {"scenario_id": "PROFESSIONAL_01", …} ],
  "sports_persona_scenarios":       [ {"scenario_id": "SPORTS_01", …} ],
  "arts_persona_scenarios":         [ {"scenario_id": "ARTS_01", …} ],
  "travel_persona_scenarios":       [ {"scenario_id": "TRAVEL_01", …} ],
  "culinary_persona_scenarios":     [ {"scenario_id": "CULINARY_01", …} ]
}
```

Each scenario:

```json
{
  "scenario_id": "PROFESSIONAL_01",
  "hidden_main_goal": "...",
  "explicit_trigger": "...",
  "implicit_anchors": ["...", "..."],
  "proactive_subtasks": [ {"task": "...", "logic": "..."} ],
  "ideal_assistant_trajectory": [
    {"step": 1, "type": "Reactive", "description": "...", "grounding": null},
    …
  ],
  "persona_alignment_check": "..."
}
```

## `blueprints.jsonl` / `validated_blueprints.jsonl` — Stage 2 / Stage 3 output

```json
{
  "uuid": "...",
  "category_key": "professional_persona",
  "persona_uuid": "...",
  "scenario_id": "PROFESSIONAL_01",
  "style_combination_index": 9,
  "blueprint_id": "BP_PROFESSIONAL_01_VQ",
  "strategic_overview": "...",
  "interaction_roadmap": [
    {
      "turn": 1,
      "phase": "EMERGENT",
      "strategic_objective": "...",
      "anchors_to_reveal": ["..."],
      "evaluation_checkpoint": {"is_trigger": true, "type": "EMERGENT",
                                "expected_inference": "..."},
      "tactical_instructions": "...",
      "reaction_logic": {"on_proactivity": "...", "on_reactivity": "..."}
    },
    ...
  ],
  "style_guardrails": "..."
}
```

`validated_blueprints.jsonl` is the subset whose `audit_decision == "PASS"`
in the matching `validation_results.jsonl`.

## `validation_results.jsonl` — Stage 3 audit decisions

```json
{
  "blueprint_id": "BP_PROFESSIONAL_01_VQ",
  "audit_decision": "PASS",
  "audit_judge_model": "gemini-2.5-pro",
  "blank_slate_integrity": {"score": "PASS", "rationale": "..."},
  "logical_necessity":     {"score": "PASS", "rationale": "..."},
  "persona_alignment":     {"score": "PASS", "rationale": "..."},
  "rubric_clarity":        {"score": "PASS", "rationale": "..."}
}
```

## `final_dialogues.jsonl` — main benchmark corpus

The released benchmark. One row per curated dialogue (198 rows).

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
  "trigger_stats": {"EMERGENT": {"PASS": 1, "PARTIAL": 0, "FAIL": 0, "SKIPPED": 0}, ...},
  "trigger_points": [
    {
      "turn": 2,
      "evaluation_rubric": {
        "type": "EMERGENT",
        "pass_criteria": "...",
        "partial_criteria": "...",
        "fail_criteria": "..."
      },
      "evaluation_result": {
        "status": "EVALUATED",
        "score": "PASS",
        "rationale": "...",
        "evidence": "\"…\""
      }
    },
    ...
  ],
  "turn_records": [
    {"turn": 1, "planner": {...}, "user_message": "...", "assistant_response": "..."},
    ...
  ],
  "token_usage": {
    "planner":    {"prompt_tokens": ..., "completion_tokens": ..., "calls": ...},
    "user_agent": {...},
    "assistant":  {...}
  }
}
```

### Field reference (final_dialogues.jsonl)

| Field | Type | Description |
|---|---|---|
| `uuid` | string | Original generation UUID. **Note:** not unique across the corpus on its own — use `unique_id_eval` as the primary key. |
| `unique_id_eval` | string | Composite primary key formatted as `{blueprint_id}__style{style_combination_index}__{uuid}`. **Guaranteed unique across the corpus**; use this when joining with offline-eval outputs or per-trigger analyses. |
| `blueprint_id` | string | Source blueprint identifier (e.g. `BP_PROFESSIONAL_01_EVA`). |
| `scenario_id` | string | Source scenario identifier (e.g. `PROFESSIONAL_01`). |
| `category_key` | enum | One of `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`. |
| `style_combination_index` | int | 1–24, indexing the CSI factorial style. See [`proactbench/styles.py`](../proactbench/styles.py). |
| `evaluated_model` | string | Model whose responses populated the dialogue at curation time (Gemini-2.5-Pro for the released corpus). |
| `num_turns_completed` | int | 5–10. Min enforced; max bounded by the planner's stop condition. |
| `trigger_stats` | object | Per-trigger-type tally of PASS / PARTIAL / FAIL / SKIPPED. |
| `trigger_points` | array | One entry per trigger turn: rubric (pass/partial/fail criteria), evaluation result (score, rationale, verbatim evidence quote). |
| `turn_records` | array | Per-turn record: planner state, user message, assistant response. |
| `token_usage` | object | Per-agent prompt/completion token counts. |

## Online vs offline evaluation outputs

`online_eval.jsonl` and `offline_eval.jsonl` use the same top-level schema as
`final_dialogues.jsonl` but with two differences in the offline form:

* `token_usage` has only `assistant` and `judge` (no `planner` / `user_agent`).
* Each `turn_records` entry also carries `original_assistant_response`, the
  response from the seed curation model, so you can diff what the new model
  said vs. what the dialogue was originally curated against.

## Score semantics

Trigger scores are one of `PASS`, `PARTIAL`, `FAIL`, or `SKIPPED` (the last
when the pipeline couldn't reach the trigger turn). Aggregation uses the
convention `Pass=1.0`, `Partial=0.5`, `Fail=0.0` for the "weighted score".

## Human-evaluation data (`dataset/human_eval/`)

### `dataset/human_eval/results/A{NN}.jsonl`

One file per pseudonymized annotator (`A01`–`A18`). Each line is one rating:

| Field | Type | Description |
|---|---|---|
| `annotator_id` | string | Pseudonym (`A01`…`A18`); the underlying Prolific worker ID is not redistributed. |
| `item_id` | string | Stable identifier for the rated trigger; matches `item_id` in `sample_items.json`. |
| `score` | string | One of `Pass`, `Partial`, `Fail`. |
| `rationale` | string | Annotator's free-text justification (mandatory). |
| `confidence` | int | Self-reported 1–5 Likert. |
| `time_spent_seconds` | float | Render → submit time for this item. |
| `timestamp` | ISO 8601 | Annotator's local-time submission. |
| `server_received_at` | ISO 8601 | UTC server-side receipt time. |

### `dataset/human_eval/sample_items.json`

A JSON array of 60 trigger points (the rated subsample). Each item:

| Field | Type | Description |
|---|---|---|
| `item_id` | string | Primary key; matches `item_id` in `results/A*.jsonl`. |
| `dialogue_id` | string | Source dialogue UUID from `final_dialogues.jsonl`. |
| `trigger_type` | string | `EMERGENT` / `CRITICAL` / `RECOVERY`. |
| `turn_index` | int | 1-indexed trigger turn within the dialogue. |
| `evaluated_model` | string | Display name of the model whose response is being rated. |
| `judge_score` | string | The GPT-5.4 offline-judge score (held out from the rating UI; used post-hoc for the human–judge comparison). |
| `judge_rationale` | string | Judge's rationale (held out from rating UI). |
| `judge_evidence` | string | Judge's pointed evidence quote (held out from rating UI). |
| `persona_category` | string | `PROFESSIONAL` / `CULINARY` / `ARTS` / `TRAVEL` / `SPORTS`. |
| `communication_style` | string | One of the 24 CSI styles. |
| `dialogue_history` | list of `{role, content}` | Conversation up to and including the trigger user message. |
| `user_message` | string | The trigger user message. |
| `anchors_disclosed` | list of strings | Explicit list of facts disclosed up to the trigger. |
| `model_response` | string | The assistant response under evaluation. |
| `rubric` | object | `{type, pass_criteria, partial_criteria, fail_criteria}` — Planner-authored before the model spoke. |
