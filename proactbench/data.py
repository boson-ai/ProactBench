"""Data-loading utilities for ProactBench (personas, blueprints, tasks)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Optional


# ── Persona rendering ─────────────────────────────────────────────────────────

def build_global_persona(row: dict) -> str:
    """Assemble a multi-aspect persona string from a raw Nemotron-Personas row."""
    label_map = {
        "professional_persona": "Professional / Career",
        "sports_persona":       "Sports & Fitness",
        "arts_persona":         "Arts & Culture",
        "travel_persona":       "Travel & Exploration",
        "culinary_persona":     "Culinary & Food",
    }
    parts: list[str] = []
    persona_summary = (row.get("persona") or "").strip()
    if persona_summary:
        parts.append(f"[Core Personality]\n{persona_summary}")
    for col, label in label_map.items():
        text = (row.get(col) or "").strip()
        if text:
            parts.append(f"[{label}]\n{text}")
    return "\n\n".join(parts)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_personas(
    personas_path: Optional[Path] = None,
    num_personas: Optional[int] = None,
    hf_dataset_name: str = "nvidia/Nemotron-Personas-USA",
    hf_local_path: Optional[str] = None,
) -> dict[str, dict]:
    """Return a ``{uuid -> persona_row}`` mapping.

    Either load from a local JSONL file (``personas_path``), or stream
    ``num_personas`` rows from HuggingFace.  The HF path supports both a
    normal ``datasets.load_dataset(hf_dataset_name)`` call and a pre-cached
    local copy via ``hf_local_path`` (e.g. an internal fsx cache).
    """
    if personas_path is not None:
        rows: list[dict] = []
        with open(personas_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        source = hf_local_path or hf_dataset_name
        hf_ds = load_dataset(source, split="train", streaming=True)
        rows = list(itertools.islice(hf_ds, num_personas))
    return {row["uuid"]: row for row in rows}


def load_blueprints(blueprints_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(blueprints_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(tasks_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["uuid"]] = row
    return rows


# ── Scenario package extraction ───────────────────────────────────────────────

_CATEGORY_TO_SCENARIOS_KEY = {
    "professional_persona": "professional_persona_scenarios",
    "sports_persona": "sports_persona_scenarios",
    "arts_persona": "arts_persona_scenarios",
    "travel_persona": "travel_persona_scenarios",
    "culinary_persona": "culinary_persona_scenarios",
}


def extract_scenario_package(task_row: dict, category_key: str, scenario_id: str) -> dict:
    """Pull the hidden_main_goal / explicit_trigger / proactive_subtasks /
    ideal_assistant_trajectory fields out of a task row for the Planner."""
    scenarios_key = _CATEGORY_TO_SCENARIOS_KEY.get(category_key, f"{category_key}_scenarios")
    scenarios = task_row.get(scenarios_key, [])
    scenario = next((s for s in scenarios if s.get("scenario_id") == scenario_id), None)
    if scenario is None:
        return {}
    return {
        "hidden_main_goal": scenario.get("hidden_main_goal"),
        "explicit_trigger": scenario.get("explicit_trigger"),
        "proactive_subtasks": scenario.get("proactive_subtasks"),
        "ideal_assistant_trajectory": scenario.get("ideal_assistant_trajectory"),
    }
