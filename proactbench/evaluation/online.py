"""Online evaluation: run the full three-agent loop on a blueprint.

Produces one JSONL record per blueprint, containing the complete dialogue
history, all trigger points with rubrics and scores, and per-agent token
usage.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..clients import (
    AnthropicClient, GeminiClient, OpenAIClient,
    _is_openai_reasoning, make_client,
)
from ..data import (
    build_global_persona, extract_scenario_package,
    load_blueprints, load_personas, load_tasks,
)
from ..prompts import (
    PLANNER_SYSTEM_TEMPLATE, USER_AGENT_SYSTEM_TEMPLATE,
    build_planner_user_message, build_user_agent_eval_message, build_user_agent_user_message,
)
from ..styles import STYLES_BY_INDEX
from ..types import (
    PlannerOutput, TriggerPoint, UserAgentOutput,
)


# ── Default generation configs ────────────────────────────────────────────────

PLANNER_GEN_CONFIG = dict(temperature=0.1, max_new_tokens=32768,
                          reasoning_effort="medium")
USER_AGENT_GEN_CONFIG = dict(temperature=0.7, max_new_tokens=32768,
                             reasoning_effort="medium")
EVAL_MODEL_GEN_CONFIG = dict(temperature=0.7, max_new_tokens=8192, top_p=1.0)


def _sanitize_gen_config(gen_config: dict, model_name: str) -> dict:
    """Provider-specific quirks. Reasoning models need temperature=1.0 and
    reject top_p; Gemini thinking models ditto."""
    if _is_openai_reasoning(model_name) or "gemini-2.5-pro" in model_name:
        out = dict(gen_config)
        out["temperature"] = 1.0
        out["top_p"] = None
        if out.get("max_new_tokens", 0) < 32768:
            out["max_new_tokens"] = 32768
        return out
    return gen_config


# ── Agent call helpers ────────────────────────────────────────────────────────

def _structured_call(client, model: str, system: str, user: str, response_format,
                     gen_config: dict):
    """Shared wrapper: send a system+user message pair, parse result into
    ``response_format``, and return (parsed, usage)."""
    return client.chat_structured(
        model=model,
        messages=[{"role": "user", "content": user}],
        system=system,
        response_format=response_format,
        return_usage=True,
        **gen_config,
    )


def _call_planner(system: str, turn: int, history: list[dict],
                  client, model: str,
                  trigger_points: Optional[list[TriggerPoint]] = None):
    count = len(trigger_points) if trigger_points is not None else 0
    user = build_planner_user_message(turn, history, trigger_points, count)
    gen = _sanitize_gen_config(dict(PLANNER_GEN_CONFIG), model)
    try:
        result, usage = _structured_call(client, model, system, user, PlannerOutput, gen)
        return result, usage
    except Exception as e:
        print(f"[Planner] turn {turn} error: {e}", flush=True)
        return None, None


def _call_user_agent(system: str, planner_out: PlannerOutput, history: list[dict],
                     client, model: str,
                     pending_trigger_point: Optional[TriggerPoint] = None):
    user = build_user_agent_user_message(planner_out, history, pending_trigger_point)
    gen = _sanitize_gen_config(dict(USER_AGENT_GEN_CONFIG), model)
    try:
        result, usage = _structured_call(client, model, system, user, UserAgentOutput, gen)
        return result, usage
    except Exception as e:
        print(f"[UserAgent] error: {e}", flush=True)
        return None, None


def _call_user_agent_eval_only(system: str, trigger: TriggerPoint, history: list[dict],
                               client, model: str):
    user = build_user_agent_eval_message(trigger, history)
    gen = _sanitize_gen_config(dict(USER_AGENT_GEN_CONFIG), model)
    try:
        result, usage = _structured_call(client, model, system, user, UserAgentOutput, gen)
        return result, usage
    except Exception as e:
        print(f"[UserAgent-Eval] error: {e}", flush=True)
        return None, None


def _call_evaluated_model(history: list[dict], user_message: str,
                          client, model: str, system: str, gen_config: dict):
    """Call the model under evaluation with the full conversation history."""
    messages = list(history) + [{"role": "user", "content": user_message}]
    try:
        result = client.chat(
            model=model, messages=messages, system=system or None,
            return_usage=True, **gen_config,
        )
        if isinstance(result, tuple):
            return result
        return result, None
    except Exception as e:
        print(f"[EvaluatedModel] error: {e}", flush=True)
        return None, None


# ── Trigger-type tally ────────────────────────────────────────────────────────

def _compute_trigger_stats(trigger_points: list[TriggerPoint]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for tp in trigger_points:
        rubric_type = tp.evaluation_rubric.type
        if tp.evaluation_result is None or tp.evaluation_result.status != "EVALUATED":
            score = "SKIPPED"
        else:
            score = tp.evaluation_result.score
        stats.setdefault(rubric_type, {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "SKIPPED": 0})
        stats[rubric_type][score] = stats[rubric_type].get(score, 0) + 1
    return stats


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_single_dialogue(
    blueprint_row: dict,
    personas_by_uuid: dict[str, dict],
    planner_client,
    planner_model: str,
    user_agent_client,
    user_agent_model: str,
    model_client,
    model_name: str,
    eval_system_prompt: str,
    eval_gen_config: dict,
    num_turns: int,
    tasks_by_uuid: Optional[dict[str, dict]] = None,
) -> Optional[dict]:
    """Execute one full three-agent dialogue against a single blueprint.

    Returns ``None`` if the persona or style cannot be resolved, otherwise a
    dict with the schema used by the paper's benchmark JSONL.
    """
    uuid = blueprint_row.get("uuid", "")
    persona_row = personas_by_uuid.get(uuid)
    if persona_row is None:
        print(f"[warn] no persona for uuid={uuid}; skipping {blueprint_row.get('blueprint_id')}")
        return None

    style_idx = blueprint_row.get("style_combination_index", -1)
    style = STYLES_BY_INDEX.get(style_idx)
    if style is None:
        print(f"[warn] unknown style_combination_index={style_idx}; skipping")
        return None

    global_persona = build_global_persona(persona_row)
    style_text = style.format()
    blueprint_str = json.dumps(blueprint_row, indent=2)

    scenario_pkg: dict = {}
    if tasks_by_uuid is not None:
        task_row = tasks_by_uuid.get(uuid)
        if task_row is not None:
            scenario_pkg = extract_scenario_package(
                task_row,
                blueprint_row.get("category_key", ""),
                blueprint_row.get("scenario_id", ""),
            )
    scenario_package_str = json.dumps(scenario_pkg, indent=2) if scenario_pkg else "(not provided)"

    planner_system = PLANNER_SYSTEM_TEMPLATE.format(
        persona=global_persona,
        scenario_package=scenario_package_str,
        blueprint=blueprint_str,
    )
    user_agent_system = USER_AGENT_SYSTEM_TEMPLATE.format(
        persona=global_persona,
        style=style_text,
    )

    dialogue_history: list[dict] = []
    turn_records: list[dict] = []
    trigger_points: list[TriggerPoint] = []
    pending_trigger: Optional[TriggerPoint] = None
    token_usage = {
        "planner": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "user_agent": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "assistant": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
    }

    def _accum(agent: str, usage: Optional[dict]):
        if usage is not None:
            token_usage[agent]["prompt_tokens"] += usage.get("prompt_tokens", 0)
            token_usage[agent]["completion_tokens"] += usage.get("completion_tokens", 0)
            token_usage[agent]["calls"] += 1

    for turn in range(1, num_turns + 1):
        planner_trigger_view = trigger_points + ([pending_trigger] if pending_trigger else [])
        planner_out, planner_usage = _call_planner(
            planner_system, turn, dialogue_history,
            planner_client, planner_model, planner_trigger_view,
        )
        _accum("planner", planner_usage)
        if planner_out is None:
            break

        if planner_out.stop_conversation and not planner_out.is_trigger_point:
            break

        user_agent_out, ua_usage = _call_user_agent(
            user_agent_system, planner_out, dialogue_history,
            user_agent_client, user_agent_model,
            pending_trigger_point=pending_trigger,
        )
        _accum("user_agent", ua_usage)
        if user_agent_out is None:
            break

        if pending_trigger is not None:
            pending_trigger.evaluation_result = user_agent_out.evaluation_result
            trigger_points.append(pending_trigger)
            pending_trigger = None

        assistant_response, assistant_usage = _call_evaluated_model(
            dialogue_history, user_agent_out.user_message,
            model_client, model_name, eval_system_prompt, eval_gen_config,
        )
        _accum("assistant", assistant_usage)
        if assistant_response is None:
            break

        dialogue_history.append({"role": "user", "content": user_agent_out.user_message})
        dialogue_history.append({"role": "assistant", "content": assistant_response})
        turn_records.append({
            "turn": turn,
            "planner": planner_out.model_dump(),
            "user_message": user_agent_out.user_message,
            "assistant_response": assistant_response,
        })

        if planner_out.is_trigger_point:
            pending_trigger = TriggerPoint(
                turn=turn,
                evaluation_rubric=planner_out.evaluation_rubric,
            )

        if planner_out.stop_conversation:
            break

    # If a trigger was declared on the last turn it still needs evaluation.
    if pending_trigger is not None:
        eval_out, eval_usage = _call_user_agent_eval_only(
            user_agent_system, pending_trigger, dialogue_history,
            user_agent_client, user_agent_model,
        )
        _accum("user_agent", eval_usage)
        if eval_out is not None:
            pending_trigger.evaluation_result = eval_out.evaluation_result
        trigger_points.append(pending_trigger)

    return {
        "blueprint_id": blueprint_row.get("blueprint_id", ""),
        "scenario_id": blueprint_row.get("scenario_id", ""),
        "uuid": uuid,
        "category_key": blueprint_row.get("category_key", ""),
        "style_combination_index": style_idx,
        "evaluated_model": model_name,
        "num_turns_completed": len(turn_records),
        "trigger_stats": _compute_trigger_stats(trigger_points),
        "trigger_points": [tp.model_dump() for tp in trigger_points],
        "turn_records": turn_records,
        "token_usage": token_usage,
    }


# ── Batch driver ──────────────────────────────────────────────────────────────

def run_online_eval(
    blueprints_path: Path,
    output_path: Path,
    evaluated_model: str,
    planner_model: str = "gpt-5.4",
    user_agent_model: str = "gpt-5.4",
    personas_path: Optional[Path] = None,
    tasks_path: Optional[Path] = None,
    num_turns: int = 10,
    num_threads: int = 4,
    num_personas: Optional[int] = None,
    num_samples: Optional[int] = None,
    eval_system_prompt: str = "",
    eval_gen_config: Optional[dict] = None,
    eval_base_url: Optional[str] = None,
    planner_base_url: Optional[str] = None,
):
    """Run the online three-agent benchmark on every blueprint in ``blueprints_path``.

    One JSONL record per completed dialogue is appended to ``output_path``.
    """
    blueprints = load_blueprints(blueprints_path)
    if num_samples is not None:
        blueprints = blueprints[:num_samples]

    personas_by_uuid = load_personas(personas_path=personas_path,
                                     num_personas=num_personas)
    tasks_by_uuid = load_tasks(tasks_path) if tasks_path is not None else None

    planner_client = make_client(planner_model, base_url=planner_base_url)
    user_agent_client = (planner_client if user_agent_model == planner_model
                         else make_client(user_agent_model, base_url=planner_base_url))
    model_client = make_client(evaluated_model, base_url=eval_base_url)

    eval_gen = dict(eval_gen_config or EVAL_MODEL_GEN_CONFIG)
    eval_gen = _sanitize_gen_config(eval_gen, evaluated_model)

    worker = partial(
        run_single_dialogue,
        personas_by_uuid=personas_by_uuid,
        planner_client=planner_client,
        planner_model=planner_model,
        user_agent_client=user_agent_client,
        user_agent_model=user_agent_model,
        model_client=model_client,
        model_name=evaluated_model,
        eval_system_prompt=eval_system_prompt,
        eval_gen_config=eval_gen,
        num_turns=num_turns,
        tasks_by_uuid=tasks_by_uuid,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_threads) as pool, open(output_path, "w") as f:
        for result in tqdm(pool.map(worker, blueprints), total=len(blueprints),
                           desc=f"Online eval [{evaluated_model}]"):
            if result is not None:
                f.write(json.dumps(result) + "\n")
                f.flush()
