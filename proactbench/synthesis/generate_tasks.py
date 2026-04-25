"""Stage 1: generate persona-grounded proactivity scenarios (tasks).

Given N personas × 5 life-domain categories, produce structured proactive
scenarios (hidden_main_goal, explicit_trigger, implicit_anchors, …) that form
the substrate for later blueprint generation.

Usage:
    python -m proactbench.synthesis.generate_tasks \\
        --personas-path personas.jsonl \\
        --output-path data/tasks.jsonl \\
        --model gpt-5.4 \\
        --num-scenarios 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

from ..clients import make_client
from ..data import load_personas
from ..prompts import PERSONA_CATEGORIES, PersonaProactivityScenarioPromptConfig
from ..types import PersonaCategoryScenarios
from . import run_prompts_parallel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--personas-path", type=Path, default=None,
                    help="JSONL of raw persona rows. If omitted, streams from HF.")
    ap.add_argument("--num-personas", type=int, default=50,
                    help="How many personas to stream if --personas-path is not given.")
    ap.add_argument("--output-path", type=Path, required=True,
                    help="Output JSONL path.")
    ap.add_argument("--num-scenarios", type=int, default=5,
                    help="Scenarios per persona × category.")
    ap.add_argument("--model", type=str, default="gpt-5.4")
    ap.add_argument("--base-url", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", type=str, default="medium",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--num-threads", type=int, default=10)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    personas_by_uuid = load_personas(
        personas_path=args.personas_path,
        num_personas=args.num_personas,
    )
    personas = list(personas_by_uuid.values())
    print(f"Loaded {len(personas)} personas")

    client = make_client(args.model, base_url=args.base_url)
    gen_config = dict(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    prompt_config = PersonaProactivityScenarioPromptConfig.create()
    category_keys = list(PERSONA_CATEGORIES.keys())

    metadata = [
        {"persona_idx": i, "uuid": row["uuid"], "category_key": cat}
        for i, row in enumerate(personas)
        for cat in category_keys
    ]
    prompts = [
        prompt_config.format(
            row=personas[item["persona_idx"]],
            category_key=item["category_key"],
            num_scenarios=args.num_scenarios,
        )
        for item in metadata
    ]

    results = run_prompts_parallel(
        client=client, prompts=prompts, model=args.model,
        response_format=PersonaCategoryScenarios, gen_config=gen_config,
        num_threads=args.num_threads, show_progress=not args.no_progress,
        desc="generate_tasks",
    )

    # Assemble output: one row per persona, one column per category.
    rows_by_uuid: dict[str, dict] = {}
    for item, result in zip(metadata, results):
        uuid = item["uuid"]
        if uuid not in rows_by_uuid:
            rows_by_uuid[uuid] = {"uuid": uuid}
        col = f"{item['category_key']}_scenarios"
        if result is not None:
            rows_by_uuid[uuid][col] = [s.model_dump() for s in result.scenarios]
        else:
            rows_by_uuid[uuid][col] = []

    os.makedirs(args.output_path.parent, exist_ok=True)
    with open(args.output_path, "w") as f:
        for row in rows_by_uuid.values():
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows_by_uuid)} persona rows → {args.output_path}")


if __name__ == "__main__":
    main()
