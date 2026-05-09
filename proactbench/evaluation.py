"""ProactBench evaluation: rerun the evaluated model at each trigger point in
a curated dialogue, then score with a judge.

Loads dialogues from the HuggingFace dataset ``bosonai/proactbench`` by
default (a local JSONL path can be supplied to override). Each dialogue
carries the Planner-authored rubric per trigger but no curation-time judge
labels. The pipeline uses the full conversation history up to each trigger
turn as context, regenerates the assistant response from the model under
evaluation, and has the judge apply the original rubric. The judge operates
in neutral mode: persona and style are not exposed at scoring time."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .clients import _is_openai_reasoning, make_client
from .prompts import JUDGE_SYSTEM_TEMPLATE, build_judge_eval_message
from .types import EvaluationRubric, JudgeOutput, TriggerPoint


# ── Generation configs ────────────────────────────────────────────────────────

JUDGE_GEN_CONFIG = dict(temperature=0.7, max_new_tokens=32768, top_p=1.0,
                        reasoning_effort="medium")
EVAL_MODEL_GEN_CONFIG = dict(temperature=0.7, max_new_tokens=8192, top_p=1.0)


def _sanitize_gen_config(gen_config: dict, model_name: str) -> dict:
    if _is_openai_reasoning(model_name) or "gemini-2.5-pro" in model_name:
        out = dict(gen_config)
        out["temperature"] = 1.0
        out["top_p"] = None
        if out.get("max_new_tokens", 0) < 32768:
            out["max_new_tokens"] = 32768
        return out
    return gen_config


# ── I/O helpers ───────────────────────────────────────────────────────────────

DEFAULT_HF_REPO = "bosonai/proactbench"
DEFAULT_HF_FILE = "final_dialogues.jsonl"


def load_dialogues(path: Optional[Path] = None) -> list[dict]:
    """Load the benchmark corpus.

    With ``path=None`` (the default), downloads ``final_dialogues.jsonl`` from
    the canonical HuggingFace dataset (``bosonai/proactbench``) via
    ``huggingface_hub.hf_hub_download``. The file is cached locally on first
    fetch and reused on subsequent runs.

    Pass an explicit ``path`` to load a local JSONL file instead (useful for
    testing against a modified or subset corpus).
    """
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=DEFAULT_HF_REPO,
            filename=DEFAULT_HF_FILE,
            repo_type="dataset",
        )
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _reconstruct_history(turn_records: list[dict], trigger_turn: int):
    """Walk turn_records in order; return (history before trigger_turn, user_message_at_trigger_turn)."""
    history: list[dict] = []
    user_message: Optional[str] = None
    for tr in turn_records:
        if tr["turn"] < trigger_turn:
            history.append({"role": "user", "content": tr["user_message"]})
            history.append({"role": "assistant", "content": tr["assistant_response"]})
        elif tr["turn"] == trigger_turn:
            user_message = tr["user_message"]
            break
    return history, user_message


def _compute_trigger_stats(trigger_points: list[TriggerPoint]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for tp in trigger_points:
        t = tp.evaluation_rubric.type
        if tp.evaluation_result is None or tp.evaluation_result.status != "EVALUATED":
            score = "SKIPPED"
        else:
            score = tp.evaluation_result.score
        stats.setdefault(t, {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "SKIPPED": 0})
        stats[t][score] = stats[t].get(score, 0) + 1
    return stats


# ── Agent calls ───────────────────────────────────────────────────────────────

def _call_eval_model(history: list[dict], user_message: str,
                    client, model: str, system: str, gen_config: dict):
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
        print(f"[EvalModel] error: {e}", flush=True)
        return None, None


def _call_judge(trigger: TriggerPoint, history_including_response: list[dict],
                client, model: str, system: str, gen_config: dict):
    user = build_judge_eval_message(trigger, history_including_response)
    try:
        result = client.chat_structured(
            model=model,
            messages=[{"role": "user", "content": user}],
            system=system,
            response_format=JudgeOutput,
            return_usage=True,
            **gen_config,
        )
        if isinstance(result, tuple):
            return result
        return result, None
    except Exception as e:
        print(f"[Judge] error: {e}", flush=True)
        return None, None


# ── Per-dialogue driver ───────────────────────────────────────────────────────

def run_dialogue_eval(
    dialogue: dict,
    eval_client,
    eval_model: str,
    eval_system_prompt: str,
    eval_gen_config: dict,
    judge_client,
    judge_model: str,
    judge_system_prompt: str,
    judge_gen_config: dict,
) -> Optional[dict]:
    """Re-evaluate every trigger point in one dialogue against a fresh model."""
    turn_records = dialogue.get("turn_records", [])
    original_triggers = dialogue.get("trigger_points", [])
    bp_id = dialogue.get("blueprint_id", "?")

    if not original_triggers:
        return None

    new_trigger_points: list[TriggerPoint] = []
    new_turn_records: list[dict] = []
    token_usage = {
        "assistant": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "judge": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
    }

    def _accum(key, usage):
        if usage:
            token_usage[key]["prompt_tokens"] += usage.get("prompt_tokens", 0)
            token_usage[key]["completion_tokens"] += usage.get("completion_tokens", 0)
            token_usage[key]["calls"] += 1

    for tp_data in original_triggers:
        turn = tp_data["turn"]
        rubric_data = tp_data["evaluation_rubric"]

        history, user_message = _reconstruct_history(turn_records, turn)
        if user_message is None:
            print(f"  [{bp_id}] no user message at turn {turn}; skipping trigger", flush=True)
            continue

        response, eval_usage = _call_eval_model(
            history, user_message, eval_client, eval_model,
            eval_system_prompt, eval_gen_config,
        )
        _accum("assistant", eval_usage)
        if response is None:
            continue

        judge_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response},
        ]
        tp = TriggerPoint(turn=turn, evaluation_rubric=EvaluationRubric(**rubric_data))

        judge_out, judge_usage = _call_judge(
            tp, judge_history, judge_client, judge_model,
            judge_system_prompt, judge_gen_config,
        )
        _accum("judge", judge_usage)
        if judge_out and judge_out.evaluation_result:
            tp.evaluation_result = judge_out.evaluation_result

        new_trigger_points.append(tp)
        new_turn_records.append({
            "turn": turn,
            "user_message": user_message,
            "assistant_response": response,
            "original_assistant_response": next(
                (tr["assistant_response"] for tr in turn_records if tr["turn"] == turn),
                None,
            ),
        })

    return {
        "blueprint_id": dialogue.get("blueprint_id", ""),
        "scenario_id": dialogue.get("scenario_id", ""),
        "uuid": dialogue.get("uuid", ""),
        "category_key": dialogue.get("category_key", ""),
        "style_combination_index": dialogue.get("style_combination_index"),
        "evaluated_model": eval_model,
        "source_evaluated_model": dialogue.get("evaluated_model", ""),
        "num_trigger_points": len(new_trigger_points),
        "trigger_stats": _compute_trigger_stats(new_trigger_points),
        "trigger_points": [tp.model_dump() for tp in new_trigger_points],
        "turn_records": new_turn_records,
        "token_usage": token_usage,
    }


# ── Batch driver ──────────────────────────────────────────────────────────────

def run_eval(
    output_path: Path,
    eval_model: str,
    results_path: Optional[Path] = None,
    judge_model: str = "gpt-5.4",
    num_threads: int = 4,
    num_samples: Optional[int] = None,
    eval_system_prompt: str = "",
    eval_gen_config: Optional[dict] = None,
    judge_gen_config: Optional[dict] = None,
    eval_base_url: Optional[str] = None,
    judge_base_url: Optional[str] = None,
    gemini_think_budget: Optional[int] = None,
):
    """Rescore every dialogue in the corpus against a new evaluated model.

    With ``results_path=None`` (the default), the canonical corpus is fetched
    from the HuggingFace dataset ``bosonai/proactbench``. Pass an explicit
    path to use a local JSONL file instead.

    One JSONL record per dialogue is written to ``output_path``.
    """
    dialogues = load_dialogues(results_path)
    if num_samples is not None:
        dialogues = dialogues[:num_samples]

    # Only pass thinking_budget when the client actually accepts it (Gemini).
    eval_client_kwargs = {}
    if "gemini" in eval_model and gemini_think_budget is not None:
        eval_client_kwargs["thinking_budget"] = gemini_think_budget
    eval_client = make_client(eval_model, base_url=eval_base_url, **eval_client_kwargs)
    judge_client = make_client(judge_model, base_url=judge_base_url)

    eval_gen = _sanitize_gen_config(dict(eval_gen_config or EVAL_MODEL_GEN_CONFIG), eval_model)
    judge_gen = _sanitize_gen_config(dict(judge_gen_config or JUDGE_GEN_CONFIG), judge_model)

    judge_system = JUDGE_SYSTEM_TEMPLATE

    worker = partial(
        run_dialogue_eval,
        eval_client=eval_client, eval_model=eval_model,
        eval_system_prompt=eval_system_prompt, eval_gen_config=eval_gen,
        judge_client=judge_client, judge_model=judge_model,
        judge_system_prompt=judge_system, judge_gen_config=judge_gen,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_threads) as pool, open(output_path, "w") as f:
        for r in tqdm(pool.map(worker, dialogues), total=len(dialogues),
                      desc=f"Eval [{eval_model}]"):
            if r is not None:
                f.write(json.dumps(r) + "\n")
                f.flush()


def _main():
    import argparse
    ap = argparse.ArgumentParser(description=run_eval.__doc__)
    ap.add_argument("--results-path", type=Path, default=None,
                    help="Local JSONL path. If omitted, downloads "
                         f"{DEFAULT_HF_FILE} from the HF dataset "
                         f"{DEFAULT_HF_REPO}.")
    ap.add_argument("--output-path", required=True, type=Path)
    ap.add_argument("--eval-model", required=True)
    ap.add_argument("--judge-model", default="gpt-5.4")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--eval-base-url", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--gemini-think-budget", type=int, default=None)
    args = ap.parse_args()
    run_eval(
        results_path=args.results_path, output_path=args.output_path,
        eval_model=args.eval_model, judge_model=args.judge_model,
        num_threads=args.num_threads, num_samples=args.num_samples,
        eval_base_url=args.eval_base_url, judge_base_url=args.judge_base_url,
        gemini_think_budget=args.gemini_think_budget,
    )


if __name__ == "__main__":
    _main()
