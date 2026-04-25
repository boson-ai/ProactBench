"""Stage 3: validate blueprints with an independent judge model.

Outputs per-blueprint audit decisions (PASS / NEEDS_REFINEMENT / FAIL) and
writes the PASSing subset to a companion ``validated_blueprints.jsonl`` file.

We recommend using a different model family from the one used in stage 2 to
avoid same-model bias in the audit (e.g., Gemini to audit GPT-generated
blueprints, or vice versa).

Usage:
    python -m proactbench.synthesis.validate_blueprints \\
        --blueprints-path data/blueprints.jsonl \\
        --tasks-path data/tasks.jsonl \\
        --personas-path personas.jsonl \\
        --output-path data/validation_results.jsonl \\
        --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..clients import make_client
from ..data import build_global_persona, load_personas
from ..prompts import PERSONA_CATEGORIES, ValidationPromptConfig
from ..styles import STYLES_BY_INDEX
from ..types import ValidationOutput
from . import run_prompts_parallel


def _build_scenario_index(task_rows: list[dict]) -> dict[tuple, dict]:
    """Build a ``(uuid, scenario_id) -> scenario`` index across all task rows."""
    index: dict[tuple, dict] = {}
    for row in task_rows:
        uuid = row["uuid"]
        for cat in PERSONA_CATEGORIES:
            for s in row.get(f"{cat}_scenarios") or []:
                index[(uuid, s["scenario_id"])] = s
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blueprints-path", type=Path, required=True)
    ap.add_argument("--tasks-path", type=Path, required=True)
    ap.add_argument("--personas-path", type=Path, default=None)
    ap.add_argument("--num-personas", type=int, default=50)
    ap.add_argument("--output-path", type=Path, required=True)

    ap.add_argument("--model", type=str, default="gemini-2.5-pro")
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--reasoning-effort", type=str, default="low",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-new-tokens", type=int, default=32768)

    ap.add_argument("--num-threads", type=int, default=10)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    personas_by_uuid = load_personas(
        personas_path=args.personas_path, num_personas=args.num_personas,
    )

    with open(args.tasks_path) as f:
        task_rows = [json.loads(l) for l in f if l.strip()]
    scenario_index = _build_scenario_index(task_rows)

    with open(args.blueprints_path) as f:
        blueprint_rows = [json.loads(l) for l in f if l.strip()]
    print(f"{len(personas_by_uuid)} personas, {len(task_rows)} tasks, {len(blueprint_rows)} blueprints")

    client = make_client(args.model, base_url=args.base_url)
    gen_config = dict(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    prompt_config = ValidationPromptConfig.create()

    metadata: list[dict] = []
    prompt_pairs = []
    skipped = 0
    for bp in blueprint_rows:
        uuid = bp["uuid"]
        scenario_id = bp["scenario_id"]
        style_index = bp["style_combination_index"]
        persona_row = personas_by_uuid.get(uuid)
        scenario = scenario_index.get((uuid, scenario_id))
        style = STYLES_BY_INDEX.get(style_index)
        if persona_row is None or scenario is None or style is None:
            skipped += 1
            continue

        global_persona = build_global_persona(persona_row)
        blueprint_payload = {
            k: v for k, v in bp.items()
            if k not in {"uuid", "category_key", "persona_uuid", "scenario_id", "style_combination_index"}
        }
        metadata.append({
            "uuid": uuid, "category_key": bp["category_key"],
            "scenario_id": scenario_id, "style_combination_index": style_index,
            "blueprint_id": bp.get("blueprint_id"),
        })
        prompt_pairs.append(prompt_config.format(
            global_persona=global_persona,
            communication_style=style.format(),
            scenario=scenario,
            blueprint=blueprint_payload,
        ))

    if skipped:
        print(f"Skipped {skipped} blueprints with missing persona/scenario/style")
    print(f"Validating {len(prompt_pairs)} blueprints")

    results = run_prompts_parallel(
        client=client, prompts=prompt_pairs, model=args.model,
        response_format=ValidationOutput, gen_config=gen_config,
        num_threads=args.num_threads, show_progress=not args.no_progress,
        desc="validate_blueprints",
    )

    os.makedirs(args.output_path.parent, exist_ok=True)
    saved = failed = 0
    passed_keys = set()
    with open(args.output_path, "w") as f:
        for item, result in zip(metadata, results):
            if result is None:
                failed += 1
                continue
            if result.audit_decision == "PASS":
                passed_keys.add((item["uuid"], item["blueprint_id"]))
            row = {**item, **result.model_dump()}
            f.write(json.dumps(row) + "\n")
            saved += 1
    print(f"Wrote {saved} validations (failed: {failed}) → {args.output_path}")

    passed_path = args.output_path.with_name(
        args.output_path.stem.replace("validation_results", "validated_blueprints") + ".jsonl"
    )
    passed_count = 0
    with open(passed_path, "w") as f:
        for bp in blueprint_rows:
            if (bp.get("uuid"), bp.get("blueprint_id")) in passed_keys:
                f.write(json.dumps(bp) + "\n")
                passed_count += 1
    print(f"Wrote {passed_count} passing blueprints → {passed_path}")

    metrics = {"total": saved, "pass": 0, "needs_refinement": 0, "fail": 0}
    for item, r in zip(metadata, results):
        if r is not None:
            metrics[r.audit_decision.lower()] = metrics.get(r.audit_decision.lower(), 0) + 1
    metrics["pass_rate"] = round(metrics["pass"] / saved, 4) if saved else 0
    metrics_path = args.output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary: pass={metrics['pass']}  needs_refinement={metrics['needs_refinement']}  fail={metrics['fail']}  pass_rate={metrics['pass_rate']:.1%}")


if __name__ == "__main__":
    main()
