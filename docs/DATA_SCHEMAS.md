# Data schemas

Every JSONL artifact produced by ProactBench is one row per line with a
stable schema. Pydantic models for every structure live in
[`proactbench/types.py`](../proactbench/types.py).

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

## `online_eval.jsonl` / `offline_eval.jsonl` — evaluation outputs

```json
{
  "blueprint_id": "BP_PROFESSIONAL_01_VQ",
  "scenario_id": "PROFESSIONAL_01",
  "uuid": "...",
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

Offline records use the same top-level schema but with two differences:

* `token_usage` has only `assistant` and `judge` (no `planner` / `user_agent`)
* each `turn_records` entry also carries `original_assistant_response`, the
  response from the seed curation model, so you can diff what the new model
  said vs. what the dialogue was originally curated against.

## Score semantics

Trigger scores are one of `PASS`, `PARTIAL`, `FAIL`, or `SKIPPED` (the last
when the pipeline couldn't reach the trigger turn). Aggregation uses the
convention `Pass=1.0`, `Partial=0.5`, `Fail=0.0` for the "weighted score".
