"""Stage 2: generate interaction blueprints from scenarios × communication styles.

Each blueprint is a turn-by-turn plan that the Planner follows at eval time.

Usage:
    python -m proactbench.synthesis.generate_blueprints \\
        --tasks-path data/tasks.jsonl \\
        --personas-path personas.jsonl \\
        --output-path data/blueprints.jsonl \\
        --model gpt-5.4 \\
        --num-styles-per-task 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from ..clients import make_client
from ..data import build_global_persona, load_personas
from ..prompts import PERSONA_CATEGORIES, BlueprintPromptConfig
from ..styles import COMMUNICATION_STYLES, STYLES_BY_INDEX
from ..types import BlueprintOutput
from . import run_prompts_parallel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-path", type=Path, required=True)
    ap.add_argument("--personas-path", type=Path, default=None)
    ap.add_argument("--num-personas", type=int, default=50)
    ap.add_argument("--output-path", type=Path, required=True)

    ap.add_argument("--model", type=str, default="gpt-5.4")
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", type=str, default="low",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-new-tokens", type=int, default=32768)

    ap.add_argument("--style-ids", type=int, nargs="+", default=None)
    ap.add_argument("--num-styles-per-task", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-threads", type=int, default=10)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    personas_by_uuid = load_personas(
        personas_path=args.personas_path, num_personas=args.num_personas,
    )
    print(f"Loaded {len(personas_by_uuid)} personas")

    task_rows: list[dict] = []
    with open(args.tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                task_rows.append(json.loads(line))
    print(f"Loaded {len(task_rows)} tasks")

    client = make_client(args.model, base_url=args.base_url)
    gen_config = dict(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    prompt_config = BlueprintPromptConfig.create()
    category_keys = list(PERSONA_CATEGORIES.keys())

    if args.style_ids:
        available_styles = [STYLES_BY_INDEX[i] for i in args.style_ids if i in STYLES_BY_INDEX]
        if not available_styles:
            raise SystemExit(f"No valid styles in {args.style_ids}")
    else:
        available_styles = COMMUNICATION_STYLES

    num_styles_per_task = args.num_styles_per_task
    if num_styles_per_task is not None and num_styles_per_task > len(available_styles):
        num_styles_per_task = None

    rng = random.Random(args.seed)

    metadata: list[dict] = []
    prompt_pairs = []
    skipped = 0
    for task_row in task_rows:
        uuid = task_row["uuid"]
        persona_row = personas_by_uuid.get(uuid)
        if persona_row is None:
            skipped += 1
            continue
        global_persona = build_global_persona(persona_row)

        for category_key in category_keys:
            scenarios = task_row.get(f"{category_key}_scenarios") or []
            for scenario in scenarios:
                task_styles = (rng.sample(available_styles, num_styles_per_task)
                               if num_styles_per_task else available_styles)
                for style in task_styles:
                    metadata.append({
                        "uuid": uuid, "category_key": category_key,
                        "scenario_id": scenario["scenario_id"],
                        "style_combination_index": style.index,
                    })
                    prompt_pairs.append(prompt_config.format(
                        global_persona=global_persona,
                        communication_style=style.format(),
                        scenario=scenario,
                    ))

    print(f"Generating {len(prompt_pairs)} blueprints (skipped {skipped} tasks with no persona)")

    results = run_prompts_parallel(
        client=client, prompts=prompt_pairs, model=args.model,
        response_format=BlueprintOutput, gen_config=gen_config,
        num_threads=args.num_threads, show_progress=not args.no_progress,
        desc="generate_blueprints",
    )

    os.makedirs(args.output_path.parent, exist_ok=True)
    saved = failed = 0
    with open(args.output_path, "w") as f:
        for item, result in zip(metadata, results):
            if result is None:
                failed += 1
                continue
            result.persona_uuid = item["uuid"]
            result.scenario_id = item["scenario_id"]
            result.style_combination_index = item["style_combination_index"]
            row = {
                "uuid": item["uuid"], "category_key": item["category_key"],
                **result.model_dump(),
            }
            f.write(json.dumps(row) + "\n")
            saved += 1

    print(f"Wrote {saved} blueprints (failed: {failed}) → {args.output_path}")


if __name__ == "__main__":
    main()
