"""End-to-end smoke test for the evaluation pipeline.

Reads the first 2 dialogues from ``release/final_dialogues.jsonl``, reruns
them with a cheap evaluated model and a cheap judge model, and verifies the
output JSONL has the expected per-dialogue records with trigger-point scores.

Usage:

    # Default (OpenAI-only, ~1 minute):
    python scripts/smoke_test.py

    # Override eval / judge models:
    python scripts/smoke_test.py --eval-model gpt-4o --judge-model gpt-4o

Exits non-zero and prints the failing stage on any error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proactbench import run_eval


def _log(msg: str) -> None:
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fail(stage: str, exc: Exception) -> int:
    print(f"[smoke FAIL] stage={stage}: {type(exc).__name__}: {exc}", flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-model", default="gpt-4o",
                    help="Evaluated model (cheap by default).")
    ap.add_argument("--judge-model", default="gpt-4o",
                    help="Judge model (cheap by default).")
    ap.add_argument("--num-samples", type=int, default=2,
                    help="Number of dialogues to evaluate.")
    ap.add_argument("--threads", type=int, default=8,
                    help="Concurrent API workers (default 8).")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[smoke] OPENAI_API_KEY not set", flush=True)
        return 1

    src = ROOT / "release" / "final_dialogues.jsonl"
    if not src.exists():
        print(f"[smoke] missing {src}", flush=True)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="proactbench_smoke_"))
    try:
        # Subsample for cheap test
        subset = tmp / "dialogues_subset.jsonl"
        rows = src.read_text().splitlines()[: args.num_samples]
        subset.write_text("\n".join(r for r in rows if r.strip()) + "\n")
        _log(f"sub-sample: {args.num_samples} dialogues -> {subset}")

        # Run eval
        out = tmp / "eval_out.jsonl"
        _log(f"running eval: eval={args.eval_model} judge={args.judge_model}")
        try:
            run_eval(
                results_path=subset,
                output_path=out,
                eval_model=args.eval_model,
                judge_model=args.judge_model,
                num_threads=args.threads,
            )
        except Exception as e:
            return _fail("run_eval", e)

        if not out.exists():
            print("[smoke FAIL] no output file produced", flush=True); return 1
        records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        if not records:
            print("[smoke FAIL] output file is empty", flush=True); return 1

        for r in records:
            for need in ("blueprint_id", "evaluated_model", "trigger_points"):
                if need not in r:
                    print(f"[smoke FAIL] record missing {need!r}", flush=True); return 1
            if r["evaluated_model"] != args.eval_model:
                print(f"[smoke FAIL] wrong evaluated_model in record: got {r['evaluated_model']!r}",
                      flush=True); return 1
            for tp in r["trigger_points"]:
                er = tp.get("evaluation_result") or {}
                if not er.get("score"):
                    print("[smoke FAIL] trigger missing judge score", flush=True); return 1

        n_trigs = sum(len(r["trigger_points"]) for r in records)
        _log(f"OK: {len(records)} records / {n_trigs} triggers, all carry valid judge scores")
        print("SMOKE TEST PASSED", flush=True)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
