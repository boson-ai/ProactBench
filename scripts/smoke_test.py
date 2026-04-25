"""End-to-end smoke test: 1 persona → 1 task → 1 blueprint → 1 online dialogue
→ 1 offline rescore. Exercises every component using small, cheap defaults
(``o4-mini`` reasoning model with ``reasoning_effort=low`` for synthesis +
Planner/User-Agent, ``gpt-4o`` as the model under evaluation).

By default only ``OPENAI_API_KEY`` is required. Override individual stage
models on the CLI to exercise other providers (set ``GEMINI_API_KEY`` for
Gemini, ``ANTHROPIC_API_KEY`` for Claude).

Usage:

    # Default (OpenAI-only, ~3 minutes):
    python scripts/smoke_test.py

    # Cross-provider (validate with Gemini, evaluate Claude):
    python scripts/smoke_test.py --validate-model gemini-2.5-flash --eval-model claude-haiku-4-5-20251001

Exits non-zero and prints the failing stage on any error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure we import the in-repo package even when run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _log(msg: str) -> None:
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"error: {name} must be set in the environment", file=sys.stderr)
        sys.exit(2)
    return val


def _run(cmd: list[str]) -> None:
    _log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed with exit {r.returncode}: {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Defaults chosen so that the smoke test runs with only OPENAI_API_KEY set.
    # Overrideable for cross-provider checks.
    ap.add_argument("--cheap-model", default="o4-mini",
                    help="OpenAI reasoning model used for synthesis stages and "
                         "Planner/User Agent in the online loop.")
    ap.add_argument("--judge-model", default="o4-mini",
                    help="Model used as the judge in offline eval.")
    ap.add_argument("--validate-model", default="o4-mini",
                    help="Model used as the independent judge in stage 3 "
                         "(blueprint validation).")
    ap.add_argument("--eval-model", default="gpt-4o",
                    help="Model under evaluation (the system being benchmarked).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write intermediate artifacts. Default: a fresh /tmp dir.")
    ap.add_argument("--keep", action="store_true",
                    help="Don't remove --output-dir on exit.")
    ap.add_argument("--skip-synthesis", action="store_true",
                    help="Skip stages 1-3 (useful when you already have blueprints.jsonl).")
    args = ap.parse_args()

    _require_env("OPENAI_API_KEY")
    if "gemini" in args.validate_model or "gemini" in args.eval_model:
        _require_env("GEMINI_API_KEY")

    out = args.output_dir or Path(tempfile.mkdtemp(prefix="proactbench_smoke_"))
    out.mkdir(parents=True, exist_ok=True)
    _log(f"output dir: {out}")

    tasks_path = out / "tasks.jsonl"
    blueprints_path = out / "blueprints.jsonl"
    validation_path = out / "validation_results.jsonl"
    validated_path = out / "validated_blueprints.jsonl"
    online_path = out / "online_eval.jsonl"
    offline_path = out / "offline_eval.jsonl"

    python = sys.executable

    try:
        if not args.skip_synthesis:
            _log("─── Stage 1: generate_tasks (1 persona × 5 categories × 1 scenario each) ───")
            _run([
                python, "-m", "proactbench.synthesis.generate_tasks",
                "--num-personas", "1",
                "--num-scenarios", "1",
                "--output-path", str(tasks_path),
                "--model", args.cheap_model,
                "--reasoning-effort", "low",
                "--num-threads", "2",
                "--no-progress",
            ])
            assert tasks_path.exists(), "tasks.jsonl not written"
            n_tasks = sum(1 for _ in open(tasks_path))
            _log(f"stage 1 OK — wrote {n_tasks} persona row(s)")

            _log("─── Stage 2: generate_blueprints (1 blueprint, 1 style) ───")
            _run([
                python, "-m", "proactbench.synthesis.generate_blueprints",
                "--tasks-path", str(tasks_path),
                "--num-personas", "1",
                "--output-path", str(blueprints_path),
                "--model", args.cheap_model,
                "--reasoning-effort", "low",
                "--num-styles-per-task", "1",
                "--seed", "42",
                "--num-threads", "2",
                "--no-progress",
            ])
            assert blueprints_path.exists() and blueprints_path.stat().st_size > 0, "blueprints.jsonl empty"
            n_bp = sum(1 for _ in open(blueprints_path))
            _log(f"stage 2 OK — wrote {n_bp} blueprint(s)")

            _log("─── Stage 3: validate_blueprints ───")
            _run([
                python, "-m", "proactbench.synthesis.validate_blueprints",
                "--blueprints-path", str(blueprints_path),
                "--tasks-path", str(tasks_path),
                "--num-personas", "1",
                "--output-path", str(validation_path),
                "--model", args.validate_model,
                "--reasoning-effort", "low",
                "--num-threads", "2",
                "--no-progress",
            ])
            assert validation_path.exists()
            # Pass-through: if none passed, fall back to original blueprints
            if not validated_path.exists() or validated_path.stat().st_size == 0:
                _log("stage 3 note — no blueprints passed audit; using unvalidated blueprints for eval")
                shutil.copy(blueprints_path, validated_path)
            else:
                n_pass = sum(1 for _ in open(validated_path))
                _log(f"stage 3 OK — {n_pass} blueprint(s) passed audit")
        else:
            if not validated_path.exists():
                if blueprints_path.exists():
                    shutil.copy(blueprints_path, validated_path)
                else:
                    raise RuntimeError("--skip-synthesis requires an existing blueprints.jsonl or validated_blueprints.jsonl")

        _log("─── Stage 4: online eval (one full three-agent dialogue) ───")
        from proactbench.evaluation import run_online_eval
        run_online_eval(
            blueprints_path=validated_path,
            output_path=online_path,
            evaluated_model=args.eval_model,
            planner_model=args.cheap_model,
            user_agent_model=args.cheap_model,
            tasks_path=tasks_path,
            num_personas=1,
            num_turns=4,            # short dialogue for smoke speed
            num_samples=1,
            num_threads=1,
        )
        assert online_path.exists() and online_path.stat().st_size > 0, "online_eval.jsonl empty"
        row = json.loads(open(online_path).readline())
        _log(f"stage 4 OK — dialogue turns={row['num_turns_completed']}  "
             f"triggers={len(row['trigger_points'])}")

        _log("─── Stage 5: offline eval (rescore the new dialogue) ───")
        from proactbench.evaluation import run_offline_eval
        run_offline_eval(
            results_path=online_path,
            output_path=offline_path,
            eval_model=args.eval_model,
            judge_model=args.judge_model,
            num_threads=1,
        )
        assert offline_path.exists() and offline_path.stat().st_size > 0, "offline_eval.jsonl empty"
        row = json.loads(open(offline_path).readline())
        _log(f"stage 5 OK — rescored triggers={row['num_trigger_points']}  "
             f"stats={row['trigger_stats']}")

        _log("✓ SMOKE TEST PASSED")
        _log(f"  artifacts: {out}")
        return 0
    except Exception as e:
        _log(f"✗ SMOKE TEST FAILED: {e!r}")
        _log(f"  inspect artifacts at: {out}")
        return 1
    finally:
        if not args.keep and args.output_dir is None:
            # Only auto-cleanup if we created a throwaway dir.
            pass  # leave it for inspection anyway; user can rm manually


if __name__ == "__main__":
    sys.exit(main())
