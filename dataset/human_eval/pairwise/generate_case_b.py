"""
generate_case_b.py — for each Recovery trigger point in final_dialogues.jsonl
(evaluated_model = gemini-2.5-pro), regenerate the assistant response with
the per-item Recovery rubric injected as a system instruction.

Case A is the existing `assistant_response` already in the dialogue file.
Case B is the new generation produced here. Decoding params match the original
benchmark: temperature=0.7, top_p=1.0, max_output_tokens=8096.

Output: case_b_responses/case_b.jsonl, one record per (dialogue, recovery turn),
each containing both Case A and Case B responses plus metadata for downstream
sampling and human evaluation.
"""

import json
import os
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

DIALOGUES_PATH = Path("/fsx/workspace/sepehr/repos/boson-multimodal/boson_multimodal/final_data/final_dialogues.jsonl")
OUT_DIR = Path(__file__).parent / "case_b_responses"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "case_b.jsonl"

MODEL_NAME = "gemini-2.5-pro"
GEN_CONFIG = {
    "temperature": 0.7,
    "top_p": 1.0,
    "max_output_tokens": 8096,
}

SYSTEM_INSTRUCTION_TEMPLATE = """\
You will be evaluated by a judge that scores this response against the \
following criteria. The trigger is of type {trigger_type}.

PASS: {pass_criteria}

PARTIAL: {partial_criteria}

FAIL: {fail_criteria}

Aim to meet the PASS criteria."""

MAX_RETRIES = 4
RETRY_BACKOFF = 2.0

# ----------------------------------------------------------------------
# API key resolution
# ----------------------------------------------------------------------

def _load_api_key():
    """Try env first, then borrow from eval-runner/.env."""
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    env_path = Path("/fsx/workspace/sepehr/repos/eval-runner/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY=") or line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GOOGLE_API_KEY / GEMINI_API_KEY not found in env or eval-runner/.env")

API_KEY = _load_api_key()

import google.generativeai as genai
genai.configure(api_key=API_KEY)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_history_and_query(turn_records, recovery_turn):
    """Return (history, user_message) for the recovery trigger.

    history is a list of {'role': 'user'|'model', 'parts': [text]} for Gemini.
    user_message is the user turn at recovery_turn (sent as the final turn).
    """
    history = []
    user_message = None
    for tr in turn_records:
        t = tr["turn"]
        if t < recovery_turn:
            history.append({"role": "user", "parts": [tr["user_message"]]})
            history.append({"role": "model", "parts": [tr["assistant_response"]]})
        elif t == recovery_turn:
            user_message = tr["user_message"]
            break
    return history, user_message


def _existing_case_a_response(turn_records, recovery_turn):
    for tr in turn_records:
        if tr["turn"] == recovery_turn:
            return tr.get("assistant_response", "")
    return ""


def _generate_case_b(history, user_message, system_instruction):
    """One Gemini call with system instruction + history + new user turn.
    Returns response text or raises after retries."""
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
    )
    chat = model.start_chat(history=history)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = chat.send_message(
                user_message,
                generation_config=genai.types.GenerationConfig(**GEN_CONFIG),
            )
            return resp.text
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
            else:
                raise
    raise last_err


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # Resume support: skip records already written
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["uuid"], r["scenario_id"], r["style_combination_index"], r["recovery_turn"]))
                except Exception:
                    continue
    print(f"Resuming: {len(done)} records already in {OUT_PATH.name}")

    n_total, n_skipped, n_done = 0, 0, 0
    n_errors = 0

    with open(OUT_PATH, "a") as out:
        for entry in _read_jsonl(DIALOGUES_PATH):
            if entry.get("evaluated_model") != "gemini-2.5-pro":
                continue
            for tp in entry.get("trigger_points", []):
                rubric = tp.get("evaluation_rubric", {})
                if rubric.get("type") != "RECOVERY":
                    continue
                er = tp.get("evaluation_result", {}) or {}
                if er.get("status") != "EVALUATED":
                    continue
                n_total += 1

                key = (entry["uuid"], entry["scenario_id"], entry["style_combination_index"], tp["turn"])
                if key in done:
                    n_skipped += 1
                    continue

                recovery_turn = tp["turn"]
                history, user_message = _build_history_and_query(entry["turn_records"], recovery_turn)
                if user_message is None:
                    print(f"[skip] no user message at turn {recovery_turn} for uuid={key[0][:8]}")
                    continue

                case_a = _existing_case_a_response(entry["turn_records"], recovery_turn)
                system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
                    trigger_type=rubric.get("type", "RECOVERY"),
                    pass_criteria=rubric.get("pass_criteria", "").strip(),
                    partial_criteria=rubric.get("partial_criteria", "").strip(),
                    fail_criteria=rubric.get("fail_criteria", "").strip(),
                )

                try:
                    case_b = _generate_case_b(history, user_message, system_instruction)
                except Exception as e:
                    n_errors += 1
                    print(f"[err] uuid={key[0][:8]} scen={key[1]} s{key[2]} t{key[3]}: {type(e).__name__}: {e}")
                    continue

                rec = {
                    "uuid": entry["uuid"],
                    "scenario_id": entry["scenario_id"],
                    "style_combination_index": entry["style_combination_index"],
                    "category_key": entry["category_key"],
                    "blueprint_id": entry["blueprint_id"],
                    "evaluated_model": MODEL_NAME,
                    "recovery_turn": recovery_turn,
                    "rubric": {
                        "type": rubric.get("type"),
                        "pass_criteria": rubric.get("pass_criteria", ""),
                        "partial_criteria": rubric.get("partial_criteria", ""),
                        "fail_criteria": rubric.get("fail_criteria", ""),
                    },
                    "judge_score_case_a": er.get("score"),
                    "judge_rationale_case_a": er.get("rationale", ""),
                    "judge_evidence_case_a": er.get("evidence", ""),
                    "user_message": user_message,
                    "case_a_response": case_a,
                    "case_b_response": case_b,
                    "system_instruction_case_b": system_instruction,
                    "gen_config": GEN_CONFIG,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f"  ... {n_done} new (skipped {n_skipped}, errors {n_errors})")

    print()
    print(f"=== Done ===")
    print(f"  Recovery triggers seen:  {n_total}")
    print(f"  already done (skipped):  {n_skipped}")
    print(f"  newly generated:         {n_done}")
    print(f"  errors:                  {n_errors}")
    print(f"  Output: {OUT_PATH}")


if __name__ == "__main__":
    main()