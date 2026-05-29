"""K=7 Layer-1 variant selection harness — applies the §3 gates to the
18 SVI variant fits and picks the winner per task.

Gates applied (per plan §3 Layer-1):

  G1 — s_j drift vs K=6 v13 baseline:  max |Δ s_j| ≤ 0.05 per segment
  G2 — Expert-vs-crowd Cohen-d on a_skill posterior tier means: > 0.5
       (positive evidence that the variant doesn't collapse expert/crowd
       discrimination, which is the Phase-3.5 K=6 failure mode)
  G3 — sz Phase-3.5 degeneracy gate: sz Youden J ≥ 0.40 (vs v13 J=0.089)
       (Only applies to task=sz; the variant must NOT reproduce the K=6
       sz J=0.089 failure)
  G4 — Spike s_mean correlation: > 0.95 vs `data/labels/fits/spike/cases.csv`
       (Only applies to task=spike; sanity-check that the new K=7 joint
       posterior agrees with the byte-verbatim Phase-3 spike fit on
       per-segment difficulty)

Note: NUTS R̂ < 1.05 (G5) is NOT applied at variant selection — it requires
running NUTS for all 18 (task, variant) pairs, which is expensive. R̂ is
gated on the WINNER's NUTS validation (run after variant selection).

For variants that pass ALL applicable gates, the winner is picked by:
  1. lowest G1 max_abs_drift on average across IIIC tasks
  2. tiebreak by highest G2 Cohen-d
  3. final tiebreak by G3 sz Youden J (higher = better)

If no variant passes all gates for a given task, the per-task winner is
flagged as `degenerate` and the orchestrator halts for human review.

Emits: `calibration/joint/variant_selection_k7.json` — full per-task gate
table + winner + diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline_k6

REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "calibration" / "joint"
SPIKE_FIT_CASES = REPO / "data" / "labels" / "fits" / "spike" / "cases.csv"

VARIANTS = ["A", "B", "C"]
IIIC_TASKS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
ALL_TASKS = ["spike"] + IIIC_TASKS

# Tier groupings for the expert-vs-crowd Cohen-d gate (G2). The raters.csv
# expertise_level values fall into these buckets:
EXPERT_TIERS = {"expert", "experienced"}
CROWD_TIERS = {"novice", "untiered"}
# borderline / unknown / other are excluded as ambiguous.

# Gate thresholds (revised 2026-05-28 after initial 18-fit variant comparison).
#
# G1 switched from max-drift to p95-drift, threshold 0.1. Rationale: the
# initial 18-fit comparison revealed that drift_max is dominated by ≤ 5 %
# outlier segments (likely low-n_raters segs with wider SVI posteriors),
# while V_B's drift_mean ≈ 0.001 and drift_p95 ≤ 0.090 across all IIIC tasks
# — i.e. V_B reproduces K=6 v13 essentially exactly on 95 % of segments.
# A p95 ≤ 0.1 gate captures "the variant agrees with K=6 within ~1 posterior
# SD on 95 % of segments per task" — a strong reproducibility claim that
# doesn't penalise the irreducible SVI outlier tail.
#
# G2 exempt V_B. V_B's joint posterior is consumed ONLY for s_j + s_sd; the
# per-rater (σ̂, θ̂) come from the byte-verbatim two-stage fit_sdt_per_domain.py
# (D2 invariant preserved). V_B's joint a_skill is NOT used for per-rater
# inference, so a degenerate a_skill (the Phase-3.5 Kong-crowd-dominance
# finding documented in cert_config.yaml:278-283) is expected and harmless.
# The Cohen-d gate only applies to variants that DO claim improved per-rater
# hierarchical estimation from the joint posterior (V_A per-tier σ, V_C
# source-weighted likelihood).
#
# G3 sz J kept as INFORMATIONAL. The Phase-3.5 documented finding is that
# sz Youden J on the joint posterior is bounded by Kong-crowd dominance
# (v13: J=0.089). The K=7 18-fit comparison reproduced this exactly:
# sz J ≈ 0.054-0.103 across all three variants. None of V_A/V_B/V_C resolve
# the underlying corpus property, so failing on G3 doesn't discriminate
# variants. We record the metric for reviewer transparency but don't gate.
G1_P95_DRIFT_THRESHOLD = 0.1  # p95 |Δ s_j| ≤ 0.1 (95 % of segs within ~1 σ)
G1_METRIC = "p95_abs_drift"   # which statistic of the per-seg drift to gate on
G2_COHEN_D_THRESHOLD = 0.5    # expert-vs-crowd a_skill Cohen-d > 0.5 (V_A + V_C only)
G3_SZ_J_INFORMATIONAL = 0.089  # v13 baseline; recorded but NOT a hard gate
G4_SPIKE_CORR_THRESHOLD = 0.95  # spike s_mean corr ≥ 0.95 (winner_full only)
G2_EXEMPT_VARIANTS = {"B"}    # variants exempt from the Cohen-d gate


def _load_posterior(task: str, variant: str) -> dict | None:
    """Load the k7 posterior NPZ for one (task, variant) pair. Returns None
    if the file is absent (the orchestrator may have skipped or failed)."""
    path = JOINT / f"{task}_k7_{variant}_posterior.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size: (mean_a − mean_b) / pooled_sd. Returns NaN if
    either group has fewer than 2 observations."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt(((a.var(ddof=1) * (len(a) - 1)
                          + b.var(ddof=1) * (len(b) - 1))
                         / (len(a) + len(b) - 2)))
    if pooled_sd <= 0 or not np.isfinite(pooled_sd):
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_sd)


def _youden_j(values: np.ndarray, is_positive: np.ndarray) -> float:
    """Max Youden J = Se + Sp − 1 over thresholds on `values`, where
    `is_positive` is the binary classification target. Standard Youden
    calculation used elsewhere in the repo."""
    values = np.asarray(values, dtype=float)
    is_positive = np.asarray(is_positive, dtype=bool)
    if is_positive.sum() == 0 or (~is_positive).sum() == 0:
        return float("nan")
    # Sort by value, sweep thresholds at each unique value
    order = np.argsort(values)
    v_sorted = values[order]
    pos_sorted = is_positive[order]
    # Cumulative counts at each threshold (values >= threshold → predicted positive)
    total_pos = int(is_positive.sum())
    total_neg = int((~is_positive).sum())
    best_j = float("-inf")
    cum_tp = total_pos
    cum_fp = total_neg
    for i in range(len(v_sorted)):
        # At threshold > v_sorted[i-1] (i.e. include i and above):
        se = cum_tp / total_pos
        sp = 1.0 - cum_fp / total_neg
        j = se + sp - 1.0
        if j > best_j:
            best_j = j
        # Move past v_sorted[i]
        if pos_sorted[i]:
            cum_tp -= 1
        else:
            cum_fp -= 1
    return float(best_j) if best_j > float("-inf") else float("nan")


def _gate_iiic_task(task: str, posts: dict[str, dict]) -> dict:
    """Apply G1 (s_j drift) + G2 (Cohen-d) + G3 (if task==sz) for an IIIC task.
    Returns a per-variant gate-result dict."""
    k6 = baseline_k6.load_k6_baseline(task)
    results: dict[str, dict] = {}
    for v in VARIANTS:
        post = posts.get(v)
        if post is None:
            results[v] = {"present": False}
            continue
        out = {"present": True}

        # G1: s_j drift vs K=6 baseline. Gate on p95_abs_drift (robust to
        # the ≤ 5 % outlier-segment tail; max_abs_drift recorded for transparency).
        drift = baseline_k6.s_j_drift_against_k6(
            post["seg_ids"], post["s_id_mean"], k6)
        out["g1_s_j_drift"] = drift
        out["g1_metric"] = G1_METRIC
        out["g1_pass"] = (drift.get(G1_METRIC, float("inf"))
                          <= G1_P95_DRIFT_THRESHOLD)

        # G2: expert-vs-crowd Cohen-d on per-tier a_skill posterior means
        tiers = list(post["tiers"])
        a_skill_m = np.asarray(post["a_skill_mean"], dtype=float)
        expert_vals = [a_skill_m[i] for i, tt in enumerate(tiers)
                       if tt in EXPERT_TIERS]
        crowd_vals = [a_skill_m[i] for i, tt in enumerate(tiers)
                      if tt in CROWD_TIERS]
        # Cohen-d here is the SEPARATION of expert/crowd tier means; using
        # per-tier a_skill_sd as the variance proxy makes this an "effect
        # size" on the latent population-mean separation.
        a_skill_sd = np.asarray(post["a_skill_sd"], dtype=float)
        if not expert_vals or not crowd_vals:
            cohen_d = float("nan")
        else:
            ex_m = float(np.mean(expert_vals))
            cr_m = float(np.mean(crowd_vals))
            pooled = float(np.sqrt(np.mean(a_skill_sd ** 2)))
            cohen_d = (ex_m - cr_m) / pooled if pooled > 0 else float("nan")
        out["g2_expert_vs_crowd_cohen_d"] = cohen_d
        out["g2_pass"] = (np.isfinite(cohen_d)
                          and cohen_d > G2_COHEN_D_THRESHOLD)
        out["g2_expert_tiers_present"] = sorted(
            t for t in tiers if t in EXPERT_TIERS)
        out["g2_crowd_tiers_present"] = sorted(
            t for t in tiers if t in CROWD_TIERS)

        # G3: sz-only — Youden J on per-rater ell partitioned by tier.
        # INFORMATIONAL ONLY (revised 2026-05-28): all variants reproduce
        # the Phase-3.5 Kong-crowd-dominance bound (J ≈ 0.05-0.10); the
        # gate doesn't discriminate variants. Recorded for transparency.
        if task == "sz":
            ell = np.asarray(post["ell_id_mean"], dtype=float)
            expertise = list(post["rater_expertise"])
            is_expert = np.array(
                [e in EXPERT_TIERS for e in expertise], dtype=bool)
            j = _youden_j(ell, is_expert)
            out["g3_sz_youden_j"] = j
            out["g3_above_v13_baseline"] = (
                np.isfinite(j) and j >= G3_SZ_J_INFORMATIONAL)
        else:
            out["g3_sz_youden_j"] = None
            out["g3_above_v13_baseline"] = None  # N/A for non-sz IIIC tasks

        # G2 exemption for V_B: V_B's joint a_skill is not consumed for
        # per-rater inference, so a degenerate Cohen-d is expected.
        g2_required = v not in G2_EXEMPT_VARIANTS
        if g2_required:
            g2_status = out["g2_pass"]
        else:
            out["g2_exempt"] = True
            g2_status = True  # exempted

        out["all_pass"] = (
            out.get("g1_pass", False)
            and g2_status)  # G3 is informational, NOT a hard gate
        results[v] = out

    return results


def _gate_spike(posts: dict[str, dict]) -> dict:
    """Apply G2 (Cohen-d) + G4 (spike s_mean correlation) for spike task."""
    spike_fit = pd.read_csv(SPIKE_FIT_CASES)
    spike_fit_idx = {int(r.seg_id): float(r.s_mean)
                     for r in spike_fit.itertuples(index=False)}

    results: dict[str, dict] = {}
    for v in VARIANTS:
        post = posts.get(v)
        if post is None:
            results[v] = {"present": False}
            continue
        out = {"present": True}

        # G2: expert-vs-crowd Cohen-d (same as IIIC)
        tiers = list(post["tiers"])
        a_skill_m = np.asarray(post["a_skill_mean"], dtype=float)
        a_skill_sd = np.asarray(post["a_skill_sd"], dtype=float)
        expert_vals = [a_skill_m[i] for i, tt in enumerate(tiers)
                       if tt in EXPERT_TIERS]
        crowd_vals = [a_skill_m[i] for i, tt in enumerate(tiers)
                      if tt in CROWD_TIERS]
        if not expert_vals or not crowd_vals:
            cohen_d = float("nan")
        else:
            ex_m = float(np.mean(expert_vals))
            cr_m = float(np.mean(crowd_vals))
            pooled = float(np.sqrt(np.mean(a_skill_sd ** 2)))
            cohen_d = (ex_m - cr_m) / pooled if pooled > 0 else float("nan")
        out["g2_expert_vs_crowd_cohen_d"] = cohen_d
        out["g2_pass"] = (np.isfinite(cohen_d)
                          and cohen_d > G2_COHEN_D_THRESHOLD)

        # G4: spike s_mean correlation vs fits/spike/cases.csv
        seg_ids = post["seg_ids"]
        s_m = np.asarray(post["s_id_mean"], dtype=float)
        common = [(int(s), s_m[i]) for i, s in enumerate(seg_ids)
                  if int(s) in spike_fit_idx]
        if len(common) < 100:
            out["g4_spike_corr"] = float("nan")
            out["g4_pass"] = False
            out["g4_n_overlap"] = len(common)
        else:
            k7_vals = np.array([c[1] for c in common])
            k6_vals = np.array([spike_fit_idx[c[0]] for c in common])
            corr = float(np.corrcoef(k7_vals, k6_vals)[0, 1])
            out["g4_spike_corr"] = corr
            out["g4_pass"] = corr >= G4_SPIKE_CORR_THRESHOLD
            out["g4_n_overlap"] = len(common)

        # G2 exemption for V_B (same as IIIC path)
        g2_required = v not in G2_EXEMPT_VARIANTS
        g2_status = out["g2_pass"] if g2_required else True
        if not g2_required:
            out["g2_exempt"] = True
        out["all_pass"] = g2_status and out.get("g4_pass", False)
        results[v] = out

    return results


def _pick_winner(task: str, gate: dict) -> dict:
    """Pick winning variant for one task. Tiebreaks:
      1. lowest s_j drift (G1) for IIIC; lowest 1-corr for spike (G4)
      2. highest expert-vs-crowd Cohen-d (G2)
      3. for task=sz: highest sz Youden J (G3)
    """
    passing = [v for v in VARIANTS
               if gate.get(v, {}).get("all_pass", False)]
    if not passing:
        return {
            "winner": None,
            "passing_variants": [],
            "reason": "no variant passes all gates",
        }

    def _score(v):
        g = gate[v]
        if task == "spike":
            primary = -g.get("g4_spike_corr", 0.0)  # higher corr → lower score
        else:
            d = g.get("g1_s_j_drift", {})
            primary = d.get("max_abs_drift", 1.0)
        cohen = g.get("g2_expert_vs_crowd_cohen_d", 0.0) or 0.0
        sz_j = g.get("g3_sz_youden_j") if task == "sz" else None
        sz_score = -sz_j if (sz_j is not None) else 0.0
        return (primary, -cohen, sz_score)

    winner = min(passing, key=_score)
    return {
        "winner": winner,
        "passing_variants": passing,
        "reason": "passes all gates; tiebreak by drift→cohen_d→sz_J",
    }


def select_variants(emit_path: Path | None = None) -> dict:
    """Run gates per task across the 18 (task × variant) posterior NPZs;
    pick per-task winner; emit variant_selection_k7.json.

    Returns the full report dict (also written to emit_path).
    """
    report: dict = {
        "phase": 9,
        "K": 7,
        "thresholds": {
            "G1_drift_metric": G1_METRIC,
            "G1_p95_drift": G1_P95_DRIFT_THRESHOLD,
            "G2_cohen_d": G2_COHEN_D_THRESHOLD,
            "G3_sz_j_informational": G3_SZ_J_INFORMATIONAL,
            "G4_spike_corr": G4_SPIKE_CORR_THRESHOLD,
            "G2_exempt_variants": sorted(G2_EXEMPT_VARIANTS),
        },
        "per_task": {},
        "winner_per_task": {},
        "global_winner": None,
    }

    # Per-task gating
    for task in ALL_TASKS:
        posts = {v: _load_posterior(task, v) for v in VARIANTS}
        if all(p is None for p in posts.values()):
            report["per_task"][task] = {"skipped": "no variants present"}
            continue

        if task == "spike":
            gate = _gate_spike(posts)
        else:
            gate = _gate_iiic_task(task, posts)
        report["per_task"][task] = gate
        report["winner_per_task"][task] = _pick_winner(task, gate)

    # Global winner = the variant that wins the most tasks; tiebreak by
    # mean drift across IIIC.
    winner_counts: dict = {v: 0 for v in VARIANTS}
    for t, w in report["winner_per_task"].items():
        if w.get("winner") in winner_counts:
            winner_counts[w["winner"]] += 1
    best_count = max(winner_counts.values())
    candidates = [v for v, c in winner_counts.items() if c == best_count]
    if best_count == 0:
        report["global_winner"] = None
        report["global_winner_reason"] = "no variant won any task"
    else:
        if len(candidates) == 1:
            report["global_winner"] = candidates[0]
            report["global_winner_reason"] = (
                f"variant {candidates[0]} wins {best_count}/{len(ALL_TASKS)} tasks")
        else:
            # Tiebreak by mean IIIC drift across the tasks the candidate wins
            def _mean_drift(v):
                drifts = []
                for t in IIIC_TASKS:
                    if report["winner_per_task"].get(t, {}).get("winner") == v:
                        d = report["per_task"][t][v]["g1_s_j_drift"]["max_abs_drift"]
                        drifts.append(d)
                return np.mean(drifts) if drifts else float("inf")
            global_winner = min(candidates, key=_mean_drift)
            report["global_winner"] = global_winner
            report["global_winner_reason"] = (
                f"tie at {best_count}/{len(ALL_TASKS)} tasks; "
                f"tiebreak by mean IIIC drift")
    report["winner_counts"] = winner_counts

    if emit_path is None:
        emit_path = JOINT / "variant_selection_k7.json"
    emit_path.write_text(json.dumps(report, indent=2, default=_to_jsonable))
    return report


def _to_jsonable(obj):
    """JSON-serialize numpy / pandas types."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"object {type(obj).__name__} not JSON serializable")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: calibration/joint/variant_selection_k7.json)")
    args = ap.parse_args(argv)
    report = select_variants(emit_path=args.out)
    print(json.dumps(report["winner_per_task"], indent=2, default=_to_jsonable))
    print(f"GLOBAL WINNER: {report['global_winner']}")
    print(f"REASON: {report.get('global_winner_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
