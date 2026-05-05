"""
analyze_results.py — Post-study analysis.

Reads results/A*.jsonl (one per pseudonymized annotator), joins with sample_items.json,
and computes:
  - Coverage stats
  - Per-annotator stats (count, mean time, mean confidence)
  - Krippendorff's alpha per trigger type (ordinal) with 95% bootstrap CI
  - Cohen's kappa (majority-vote consensus vs judge) per trigger type with 95% CI
  - 3x3 confusion matrices per trigger type
  - Agreement-level distribution (all 3 agree / 2 agree / all differ)

Outputs:
  analysis_summary.txt
  analysis_stats.json
  confusion_matrix_emergent.csv
  confusion_matrix_critical.csv
  confusion_matrix_recovery.csv
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import krippendorff as _krip_pkg
    HAS_KRIP = True
except ImportError:
    HAS_KRIP = False

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
SAMPLE_PATH = ROOT / "sample_items.json"
OUT_TXT = ROOT / "analysis_summary.txt"
OUT_JSON = ROOT / "analysis_stats.json"

SCORE_ORDER = ["Fail", "Partial", "Pass"]          # ordinal: Fail < Partial < Pass
SCORE_TO_INT = {"Fail": 0, "Partial": 1, "Pass": 2}
TRIGGER_TYPES = ["EMERGENT", "CRITICAL", "RECOVERY"]

N_BOOTSTRAP = 10_000
BOOT_SEED = 2026

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_ratings():
    """Return list of dicts, one per human rating."""
    ratings = []
    for p in sorted(RESULTS_DIR.glob("A*.jsonl")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ratings.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return ratings


def load_sample():
    with open(SAMPLE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def build_long_df(ratings, sample):
    items = {it["item_id"]: it for it in sample}
    rows = []
    for r in ratings:
        iid = r.get("item_id")
        it = items.get(iid)
        if it is None:
            continue
        rows.append({
            "item_id": iid,
            "annotator_id": str(r.get("annotator_id")),
            "human_score": r.get("score"),
            "judge_score": it.get("judge_score"),
            "rationale": r.get("rationale", ""),
            "confidence": r.get("confidence"),
            "time_spent_seconds": r.get("time_spent_seconds"),
            "trigger_type": it.get("trigger_type"),
            "evaluated_model": it.get("evaluated_model"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_stats(df, sample):
    total_items = len(sample)
    by_item = df.groupby("item_id").size() if len(df) else pd.Series(dtype=int)
    n_with_3 = int((by_item == 3).sum())
    n_with_2 = int((by_item == 2).sum())
    n_with_1 = int((by_item == 1).sum())
    n_with_0 = total_items - len(by_item)
    return {
        "total_items": total_items,
        "items_with_3_ratings": n_with_3,
        "items_with_2_ratings": n_with_2,
        "items_with_1_rating": n_with_1,
        "items_with_0_ratings": n_with_0,
        "total_ratings": len(df),
    }


# ---------------------------------------------------------------------------
# Per-annotator stats
# ---------------------------------------------------------------------------

def per_annotator_stats(df):
    if df.empty:
        return {}
    g = df.groupby("annotator_id")
    out = {}
    for aid, sub in g:
        out[aid] = {
            "n_ratings": int(len(sub)),
            "mean_time_sec": float(sub["time_spent_seconds"].mean()) if len(sub) else None,
            "median_time_sec": float(sub["time_spent_seconds"].median()) if len(sub) else None,
            "mean_confidence": float(sub["confidence"].mean()) if len(sub) else None,
            "pct_pass": float((sub["human_score"] == "Pass").mean()),
            "pct_partial": float((sub["human_score"] == "Partial").mean()),
            "pct_fail": float((sub["human_score"] == "Fail").mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Krippendorff's alpha (ordinal)
# ---------------------------------------------------------------------------

def _build_reliability_matrix(df_subset, annotator_ids):
    """
    Rows = annotators, cols = items. Values are 0/1/2 ints or np.nan.
    Takes the last rating per (annotator, item) pair if duplicated.
    """
    items = sorted(df_subset["item_id"].unique())
    item_to_col = {iid: j for j, iid in enumerate(items)}
    mat = np.full((len(annotator_ids), len(items)), np.nan, dtype=float)
    for _, row in df_subset.iterrows():
        aid = str(row["annotator_id"])
        if aid not in annotator_ids:
            continue
        j = item_to_col[row["item_id"]]
        i = annotator_ids.index(aid)
        mat[i, j] = SCORE_TO_INT[row["human_score"]]
    return mat, items


def _alpha_ordinal_manual(reliability_matrix):
    """
    Implement Krippendorff's alpha for ordinal data from scratch.
    reliability_matrix: rows=coders, cols=units, NaN for missing.
    """
    M = np.array(reliability_matrix, dtype=float)
    n_coders, n_units = M.shape

    # Per-unit: list of observed values
    unit_values = [M[:, u][~np.isnan(M[:, u])].astype(int).tolist() for u in range(n_units)]
    # Drop units with <2 ratings
    valid = [v for v in unit_values if len(v) >= 2]
    if len(valid) < 2:
        return float("nan")

    # Build value frequency counts (nv and n)
    value_counts = Counter()
    for v in valid:
        for x in v:
            value_counts[x] += 1
    values = sorted(value_counts.keys())
    n = sum(value_counts.values())

    # Ordinal distance: d(c,k) = (sum of counts between c and k, plus half endpoint counts) squared
    # delta^2_{ck} = ( (n_c + n_k)/2 + sum_{g: c<g<k} n_g )^2
    def delta_sq(c, k):
        if c == k:
            return 0.0
        lo, hi = (c, k) if c < k else (k, c)
        between = sum(value_counts[g] for g in values if lo < g < hi)
        return ((value_counts[lo] + value_counts[hi]) / 2.0 + between) ** 2

    # Observed disagreement
    num = 0.0
    total_weight = 0.0
    for v in valid:
        mu = len(v)
        # weight 1/(mu - 1) per pair
        w = 1.0 / (mu - 1)
        total_weight += mu  # denominator accumulator
        for i in range(mu):
            for j in range(mu):
                if i == j:
                    continue
                num += w * delta_sq(v[i], v[j])

    # Do (observed) = num / (2 * sum mu)
    Do = num / (2.0 * sum(len(v) for v in valid))

    # Expected disagreement: sum over ordered pairs of all observed values
    De_num = 0.0
    for i, c in enumerate(values):
        for j, k in enumerate(values):
            De_num += value_counts[c] * value_counts[k] * delta_sq(c, k)
    De = De_num / (n * (n - 1)) if n > 1 else 0.0

    if De == 0:
        return 1.0 if Do == 0 else float("nan")
    return 1.0 - Do / De


def krippendorff_alpha_ordinal(reliability_matrix):
    """Prefer the `krippendorff` package if installed; otherwise fall back to manual."""
    # Normalize NaN handling
    M = np.array(reliability_matrix, dtype=float)
    if HAS_KRIP:
        try:
            return float(_krip_pkg.alpha(reliability_data=M, level_of_measurement="ordinal"))
        except Exception:
            pass
    return float(_alpha_ordinal_manual(M))


def bootstrap_alpha(df_sub, annotator_ids, n_boot=N_BOOTSTRAP, seed=BOOT_SEED):
    """
    Bootstrap by resampling units (items) with replacement.
    Returns (point_estimate, ci_low, ci_high).
    """
    # Use the full observed matrix for point estimate
    M_full, items = _build_reliability_matrix(df_sub, annotator_ids)
    point = krippendorff_alpha_ordinal(M_full)

    rng = np.random.default_rng(seed)
    n_items = len(items)
    if n_items < 2:
        return point, float("nan"), float("nan")

    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        M_boot = M_full[:, idx]
        a = krippendorff_alpha_ordinal(M_boot)
        if not (isinstance(a, float) and math.isnan(a)):
            samples.append(a)
    if not samples:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def majority_vote_consensus(df):
    """
    Per item_id, return the consensus label (majority vote).
    Ties (e.g. one vote for each of 3 categories) resolved by priority:
      Partial > Pass > Fail (middle wins ties, then Pass over Fail).
    Returns dict item_id -> consensus string.
    """
    out = {}
    tie_priority = {"Partial": 0, "Pass": 1, "Fail": 2}  # lower = preferred
    for iid, sub in df.groupby("item_id"):
        c = Counter(sub["human_score"].tolist())
        if not c:
            continue
        top = c.most_common()
        best_count = top[0][1]
        contenders = [lab for lab, cnt in top if cnt == best_count]
        if len(contenders) == 1:
            out[iid] = contenders[0]
        else:
            out[iid] = sorted(contenders, key=lambda x: tie_priority.get(x, 99))[0]
    return out


def cohen_kappa(y_true, y_pred, labels):
    """Plain Cohen's kappa, unweighted."""
    if len(y_true) == 0:
        return float("nan")
    N = len(y_true)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)))
    for a, b in zip(y_true, y_pred):
        if a not in label_to_idx or b not in label_to_idx:
            continue
        cm[label_to_idx[a], label_to_idx[b]] += 1
    po = np.trace(cm) / N
    row = cm.sum(axis=1) / N
    col = cm.sum(axis=0) / N
    pe = float(np.sum(row * col))
    if pe == 1:
        return 1.0 if po == 1 else float("nan")
    return float((po - pe) / (1 - pe))


def bootstrap_kappa(y_true, y_pred, labels, n_boot=N_BOOTSTRAP, seed=BOOT_SEED):
    point = cohen_kappa(y_true, y_pred, labels)
    rng = np.random.default_rng(seed + 1)
    N = len(y_true)
    if N < 2:
        return point, float("nan"), float("nan")
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        a = [y_true[i] for i in idx]
        b = [y_pred[i] for i in idx]
        k = cohen_kappa(a, b, labels)
        if not math.isnan(k):
            samples.append(k)
    if not samples:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_df(human, judge, labels):
    df = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for h, j in zip(human, judge):
        if h in labels and j in labels:
            df.loc[h, j] += 1
    df.index.name = "human_consensus"
    df.columns.name = "judge_score"
    return df


# ---------------------------------------------------------------------------
# Agreement-level distribution
# ---------------------------------------------------------------------------

def agreement_levels(df):
    """Per item, classify as all-3-agree / 2-agree / all-differ."""
    out = Counter()
    for iid, sub in df.groupby("item_id"):
        scores = sub["human_score"].tolist()
        c = Counter(scores)
        if len(c) == 1 and len(scores) >= 3:
            out["all_3_agree"] += 1
        elif len(scores) == 3 and max(c.values()) == 2:
            out["2_agree_1_differ"] += 1
        elif len(scores) == 3 and max(c.values()) == 1:
            out["all_differ"] += 1
        elif len(scores) == 2 and len(c) == 1:
            out["2_agree_only_2_ratings"] += 1
        elif len(scores) == 2:
            out["2_differ_only_2_ratings"] += 1
        elif len(scores) == 1:
            out["only_1_rating"] += 1
    return dict(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_ci(p, lo, hi):
    if isinstance(p, float) and math.isnan(p):
        return "nan"
    if isinstance(lo, float) and math.isnan(lo):
        return f"{p:.3f} [nan, nan]"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    sample = load_sample()
    ratings = load_ratings()
    df = build_long_df(ratings, sample)

    lines = []
    def p(s=""):
        print(s)
        lines.append(str(s))

    p("=" * 70)
    p("Proactivity Bench — Human Annotation Analysis")
    p("=" * 70)
    p(f"Loaded {len(sample)} items from sample")
    p(f"Loaded {len(ratings)} ratings from {len(list(RESULTS_DIR.glob('A*.jsonl')))} annotator file(s)")
    p("")

    # --- Coverage ---
    cov = coverage_stats(df, sample)
    p("=== Coverage ===")
    for k, v in cov.items():
        p(f"  {k}: {v}")
    p("")

    # --- Per-annotator ---
    pa = per_annotator_stats(df)
    p("=== Per-annotator stats ===")
    for aid in sorted(pa.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        s = pa[aid]
        p(f"  annotator {aid}: n={s['n_ratings']:3d}  "
          f"mean_time={s['mean_time_sec']:.1f}s  "
          f"mean_conf={s['mean_confidence']:.2f}  "
          f"pass/partial/fail = {s['pct_pass']:.2f}/{s['pct_partial']:.2f}/{s['pct_fail']:.2f}")
    p("")

    # --- Agreement-level distribution ---
    agree = agreement_levels(df)
    p("=== Agreement-level distribution (per item) ===")
    for k, v in agree.items():
        p(f"  {k}: {v}")
    p("")

    stats = {
        "coverage": cov,
        "per_annotator": pa,
        "agreement_levels": agree,
        "per_trigger_type": {},
    }

    # --- Krippendorff alpha & Cohen's kappa per trigger type ---
    annotator_ids = sorted(df["annotator_id"].unique().tolist()) if not df.empty else []
    p("=== Krippendorff's alpha (ordinal) + Cohen's kappa (consensus vs judge) per trigger type ===")
    if df.empty:
        p("  (no ratings yet)")
    for trig in TRIGGER_TYPES:
        sub = df[df["trigger_type"] == trig]
        if sub.empty:
            p(f"  {trig}: no data")
            stats["per_trigger_type"][trig] = {}
            continue
        alpha, alo, ahi = bootstrap_alpha(sub, annotator_ids)
        consensus = majority_vote_consensus(sub)
        item_judge = {it["item_id"]: it["judge_score"]
                      for it in sample if it["trigger_type"] == trig}
        human_list = []
        judge_list = []
        for iid, h in consensus.items():
            j = item_judge.get(iid)
            if j is None:
                continue
            human_list.append(h)
            judge_list.append(j)
        kappa, klo, khi = bootstrap_kappa(human_list, judge_list, SCORE_ORDER)
        cm = confusion_matrix_df(human_list, judge_list, SCORE_ORDER)

        p(f"  {trig}:")
        p(f"    n_items={sub['item_id'].nunique()}, n_ratings={len(sub)}")
        p(f"    Krippendorff alpha (ordinal): {fmt_ci(alpha, alo, ahi)}")
        p(f"    Cohen's kappa (consensus vs judge): {fmt_ci(kappa, klo, khi)}")
        p(f"    Confusion matrix (rows=human_consensus, cols=judge):")
        for row_label in SCORE_ORDER:
            row = cm.loc[row_label].tolist()
            p(f"      {row_label:<8} | {' '.join(f'{v:>4d}' for v in row)}")
        p(f"             {'':8}   {'  '.join(f'{c:>4s}' for c in SCORE_ORDER)}")

        cm_path = ROOT / f"confusion_matrix_{trig.lower()}.csv"
        cm.to_csv(cm_path)
        p(f"    -> saved {cm_path.name}")
        p("")

        stats["per_trigger_type"][trig] = {
            "n_items": int(sub["item_id"].nunique()),
            "n_ratings": int(len(sub)),
            "krippendorff_alpha": alpha,
            "krippendorff_alpha_ci95": [alo, ahi],
            "cohen_kappa_consensus_vs_judge": kappa,
            "cohen_kappa_ci95": [klo, khi],
            "confusion_matrix": {r: {c: int(cm.loc[r, c]) for c in SCORE_ORDER} for r in SCORE_ORDER},
        }

    # Save outputs
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved {OUT_TXT}")
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
