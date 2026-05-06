"""
analyze_pairwise.py — analyse the pairwise A-vs-B human study.

Reads results/annotator_*_results.jsonl and sample_items.json.
Reports:
  - Coverage (items per rater, ratings per item)
  - B-preference rate overall and per stratum (Pass/Partial/Fail), with
    item-clustered bootstrap 95% CIs
  - Two-sided exact binomial test against null = 0.5 (excluding ties)
  - Per-rater B-preference (sanity check on rater drift)
  - Effect of confidence and time-on-task on B-preference

Outputs:
  analysis_summary.txt
  analysis_stats.json
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
SAMPLE_PATH = ROOT / "sample_items.json"
OUT_TXT = ROOT / "analysis_summary.txt"
OUT_JSON = ROOT / "analysis_stats.json"

N_BOOT = 10_000
SEED = 2026

LOG_BUF = []
def p(s=""):
    print(s)
    LOG_BUF.append(str(s))


def load_ratings():
    rows = []
    for fp in sorted(RESULTS_DIR.glob("annotator_*_results.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def cluster_bootstrap_ci(values_by_item, stat_fn, B=N_BOOT, seed=SEED):
    """Cluster bootstrap: resample items (clusters) with replacement.
    values_by_item: dict item_id -> list of values (one per rater).
    stat_fn: function from list[value] -> scalar.
    Returns (point, lo, hi)."""
    items = list(values_by_item.keys())
    if not items:
        return float("nan"), float("nan"), float("nan")
    flat = [v for vs in values_by_item.values() for v in vs]
    point = stat_fn(flat)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(B):
        idx = rng.integers(0, len(items), size=len(items))
        flat_b = []
        for k in idx:
            flat_b.extend(values_by_item[items[k]])
        if not flat_b:
            continue
        s = stat_fn(flat_b)
        if not (isinstance(s, float) and math.isnan(s)):
            samples.append(s)
    if not samples:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main():
    if not SAMPLE_PATH.exists():
        raise SystemExit(f"Missing {SAMPLE_PATH}")
    with open(SAMPLE_PATH) as f:
        sample = {it["item_id"]: it for it in json.load(f)}

    ratings = load_ratings()
    p("=" * 70)
    p(f"Pairwise analysis — {len(ratings)} judgements collected")
    p("=" * 70)

    if not ratings:
        p("\n(no ratings yet — re-run after data collection)")
        return

    # Coverage
    by_item = defaultdict(list)
    for r in ratings:
        if r["item_id"] in sample:
            by_item[r["item_id"]].append(r)
    raters = sorted({r["annotator_id"] for r in ratings})
    p(f"\n=== Coverage ===")
    p(f"  total judgements: {len(ratings)}")
    p(f"  unique items rated: {len(by_item)} / {len(sample)}")
    cov = Counter(len(v) for v in by_item.values())
    for k in sorted(cov.keys()):
        p(f"    {k} ratings: {cov[k]} items")
    p(f"  raters: {len(raters)}")

    # Per-rater stats
    p(f"\n=== Per rater ===")
    per_rater = defaultdict(list)
    for r in ratings:
        per_rater[r["annotator_id"]].append(r)
    for pid in sorted(per_rater.keys()):
        rs = per_rater[pid]
        n = len(rs)
        n_b = sum(1 for r in rs if r["chosen_case"] == "B")
        n_a = sum(1 for r in rs if r["chosen_case"] == "A")
        n_t = sum(1 for r in rs if r["chosen_case"] == "tie")
        if n_a + n_b > 0:
            b_rate = n_b / (n_a + n_b)
        else:
            b_rate = float("nan")
        med_t = float(np.median([r.get("time_spent_seconds", 0) for r in rs]))
        mean_c = float(np.mean([r.get("confidence", 0) for r in rs]))
        p(f"  {pid}: n={n}  A={n_a} B={n_b} ties={n_t}  B-rate (excl ties)={b_rate:.2f}  med_time={med_t:.0f}s  mean_conf={mean_c:.2f}")

    # B-preference (excluding ties)
    p(f"\n=== B-preference rate (excluding ties) ===")
    # Build per-item list of {1 if chosen B, 0 if chosen A}; ties dropped
    b_by_item = defaultdict(list)
    for r in ratings:
        if r["chosen_case"] == "tie": continue
        b_by_item[r["item_id"]].append(1 if r["chosen_case"] == "B" else 0)

    flat = [v for vs in b_by_item.values() for v in vs]
    n_total = len(flat)
    n_b = sum(flat)
    n_a = n_total - n_b
    if n_total == 0:
        p("  (all judgements were ties — no statistic)")
    else:
        rate, lo, hi = cluster_bootstrap_ci(b_by_item, lambda xs: sum(xs) / len(xs))
        p(f"  Overall: n_A={n_a}, n_B={n_b} (n_total={n_total}, ties_dropped={len(ratings) - n_total})")
        p(f"  B-preference rate: {rate:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
        # Two-sided exact binomial test against 0.5
        try:
            res = stats.binomtest(n_b, n_total, p=0.5, alternative="two-sided")
            pval = res.pvalue
        except AttributeError:
            pval = stats.binom_test(n_b, n_total, p=0.5, alternative="two-sided")
        p(f"  Exact binomial test vs 0.5 (two-sided): p = {pval:.4g}")

    # Per stratum
    p(f"\n=== B-preference rate per Case-A judge stratum ===")
    strata = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        if r["chosen_case"] == "tie": continue
        s = sample[r["item_id"]].get("judge_score_case_a", "?")
        strata[s][r["item_id"]].append(1 if r["chosen_case"] == "B" else 0)
    for stratum in ["FAIL", "PARTIAL", "PASS"]:
        if stratum not in strata: continue
        rate, lo, hi = cluster_bootstrap_ci(strata[stratum], lambda xs: sum(xs) / len(xs))
        flat_s = [v for vs in strata[stratum].values() for v in vs]
        n_s = len(flat_s); n_b_s = sum(flat_s)
        if n_s == 0:
            continue
        try:
            res = stats.binomtest(n_b_s, n_s, p=0.5, alternative="two-sided")
            pval = res.pvalue
        except AttributeError:
            pval = stats.binom_test(n_b_s, n_s, p=0.5, alternative="two-sided")
        p(f"  Case-A judge = {stratum:<8} n={n_s:>3}  B-pref={rate:.3f} [{lo:.3f}, {hi:.3f}]  binom p={pval:.3g}")

    # Confidence-weighted look
    p(f"\n=== B-preference by confidence ===")
    by_conf_item = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        if r["chosen_case"] == "tie": continue
        c = r.get("confidence")
        if c not in (1,2,3,4,5): continue
        by_conf_item[c][r["item_id"]].append(1 if r["chosen_case"] == "B" else 0)
    for c in sorted(by_conf_item.keys()):
        flat_c = [v for vs in by_conf_item[c].values() for v in vs]
        if not flat_c: continue
        rate = sum(flat_c)/len(flat_c)
        p(f"  confidence {c}: n={len(flat_c)}  B-rate={rate:.3f}")

    # Per-item agreement (when 3+ raters per item, do they agree?)
    p(f"\n=== Inter-rater agreement on items with >=3 raters ===")
    items_3plus = {iid: vs for iid, vs in b_by_item.items() if len(vs) >= 3}
    if items_3plus:
        unanimous_b = sum(1 for vs in items_3plus.values() if all(v == 1 for v in vs))
        unanimous_a = sum(1 for vs in items_3plus.values() if all(v == 0 for v in vs))
        majority_b = sum(1 for vs in items_3plus.values() if sum(vs) > len(vs)/2)
        p(f"  items with >=3 non-tie ratings: {len(items_3plus)}")
        p(f"  unanimous B: {unanimous_b}")
        p(f"  unanimous A: {unanimous_a}")
        p(f"  majority B (>50%): {majority_b}")

    # Save
    stats_out = {
        "n_ratings": len(ratings),
        "n_items_rated": len(by_item),
        "raters": raters,
        "n_total_excl_ties": n_total,
        "n_a": n_a, "n_b": n_b,
        "b_rate": rate if n_total else None,
    }
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(LOG_BUF) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(stats_out, f, indent=2)
    p(f"\nSaved {OUT_TXT}")
    p(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()