"""
build_sample.py — sample 80 Recovery trigger points (stratified by the Case A
judge score: 27 Pass / 27 Partial / 26 Fail) for the pairwise human study.

Reads case_b_responses/case_b.jsonl (produced by generate_case_b.py),
emits sample_items.json that the Prolific server consumes.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
CASE_B_PATH = ROOT / "case_b_responses" / "case_b.jsonl"
SAMPLE_OUT = ROOT / "sample_items.json"
LOG_OUT = ROOT / "build_sample.log"

N_TARGET = 80
# Recovery is hard for vanilla gemini-2.5-pro: most items are FAIL under the
# GPT-5.4 judge. Pool sizes (typical): PASS~23, PARTIAL~12, FAIL~156.
# Strategy: take ALL items in the small strata, sample from FAIL to reach
# N_TARGET. This guarantees full coverage of PASS/PARTIAL while leaving
# headroom in the abundant FAIL stratum.
STRATA_FLOOR = {"PASS": None, "PARTIAL": None, "FAIL": None}  # None = take all
SAMPLE_SEED = 42

LOG_BUF = []
def log(msg=""):
    print(msg)
    LOG_BUF.append(str(msg))


def main():
    if not CASE_B_PATH.exists():
        raise SystemExit(f"Missing {CASE_B_PATH} — run generate_case_b.py first.")

    pool = []
    with open(CASE_B_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            j = r.get("judge_score_case_a")
            if j not in {"PASS", "PARTIAL", "FAIL"}:
                continue
            pool.append(r)

    log(f"Loaded {len(pool)} Recovery items with Case A + Case B")
    by_stratum = defaultdict(list)
    for r in pool:
        by_stratum[r["judge_score_case_a"]].append(r)
    for s in ["PASS", "PARTIAL", "FAIL"]:
        log(f"  stratum {s}: {len(by_stratum.get(s, []))} items available")

    rng = random.Random(SAMPLE_SEED)
    chosen = []
    # Step 1: take all PASS + PARTIAL (small strata)
    for s in ["PASS", "PARTIAL"]:
        bucket = list(by_stratum.get(s, []))
        rng.shuffle(bucket)
        chosen.extend(bucket)
        log(f"  taking all {len(bucket)} items from stratum {s}")
    # Step 2: fill remaining slots from FAIL
    fail_bucket = list(by_stratum.get("FAIL", []))
    rng.shuffle(fail_bucket)
    n_remaining = N_TARGET - len(chosen)
    n_fail_to_take = min(n_remaining, len(fail_bucket))
    chosen.extend(fail_bucket[:n_fail_to_take])
    log(f"  taking {n_fail_to_take} of {len(fail_bucket)} FAIL items to reach target")

    log(f"\nFinal sample: {len(chosen)} items")
    log(f"  by stratum: {dict(Counter(it['judge_score_case_a'] for it in chosen))}")
    log(f"  by category: {dict(Counter(it['category_key'] for it in chosen))}")
    log(f"  by recovery_turn: {dict(Counter(it['recovery_turn'] for it in chosen))}")

    # Build the records the server will serve. Each item has both responses;
    # left/right randomization is performed at slate-creation time (per worker).
    items = []
    for r in chosen:
        item_id = f"{r['uuid']}__{r['scenario_id']}__s{r['style_combination_index']}__t{r['recovery_turn']}"
        items.append({
            "item_id": item_id,
            "uuid": r["uuid"],
            "scenario_id": r["scenario_id"],
            "style_combination_index": r["style_combination_index"],
            "category_key": r["category_key"],
            "blueprint_id": r["blueprint_id"],
            "evaluated_model": r["evaluated_model"],
            "recovery_turn": r["recovery_turn"],
            "judge_score_case_a": r["judge_score_case_a"],
            "judge_rationale_case_a": r["judge_rationale_case_a"],
            "user_message": r["user_message"],
            "case_a_response": r["case_a_response"],
            "case_b_response": r["case_b_response"],
            # rubric is hidden from annotators but kept here for analysis
            "rubric": r["rubric"],
            "system_instruction_case_b": r["system_instruction_case_b"],
        })

    with open(SAMPLE_OUT, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    log(f"\nWrote {len(items)} items to {SAMPLE_OUT}")

    with open(LOG_OUT, "w") as f:
        f.write("\n".join(LOG_BUF) + "\n")
    log(f"Log: {LOG_OUT}")


if __name__ == "__main__":
    main()