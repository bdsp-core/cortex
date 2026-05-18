"""F2.6 (2026-05-15) — SPARCNET split-half test-retest reliability.

Under the Multi-AUROC reframe, the relevant reliability quantity is the
reproducibility of the per-domain AUROC point estimate (not a binary
pass/fail verdict).  This mirrors the single-domain spike paper's
ICC(3,1) = 0.865 on σ̂, adapted to the AUROC scale and the six SPARCNET
domains.

Protocol (per domain d, per rater i with ≥ MIN_TRIALS annotations):
  1. Join the rater's (case_id, Y) history with Rasch difficulty c_mean
     from sparcnet_{d}_fit/c.csv; convert to probit:
        c_probit = c_mean * LOGIT_TO_PROBIT,   LOGIT_TO_PROBIT = 1/1.7
  2. Shuffle the rater's trials with a fixed seed; split into two halves.
  3. Fit (σ, θ) on each half via the spike-paper probit-lapse NLL
        p = λ + (1 − 2λ)·Φ((c_probit − θ)/σ),  λ = 0.025
     (L-BFGS-B on (θ, log σ); identical to train_val_split_and_fit.fit_rater).
  4. AUROC_half = Φ(√2 / √(exp(−2ℓ) + 1)),  ℓ = −log σ   (auroc.auroc_from_l).
  5. Per domain: ICC(3,1) and Pearson r on (AUROC_h1, AUROC_h2) across raters.

Acceptance (Multi-AUROC): ICC(3,1) ≥ 0.70 per domain is the target
(spike paper achieved 0.865 on σ̂; AUROC is a monotone transform so a
comparable or slightly attenuated value is expected).

Parallel over (domain, rater).  Bitwise-deterministic (fixed shuffle seed
+ single-thread BLAS).

Outputs:
  results/phase2_validation/sparcnet_test_retest.csv     (per rater-domain)
  results/phase2_validation/sparcnet_test_retest.md      (per-domain ICC)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_THIS_DIR_BOOT = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR_BOOT not in sys.path:
    sys.path.insert(0, _THIS_DIR_BOOT)
from _parallel import configure_blas_single_thread, parallel_map  # noqa: E402
configure_blas_single_thread()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

ENGINE_REPO = os.path.dirname(_THIS_DIR_BOOT)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)
from auroc import auroc_from_l  # noqa: E402
import engine_paths  # noqa: E402

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
SPIKE_PREPARED = ("/Users/elikeldsen/Documents/Research/spike-test-project/"
                  "ilae-skill-certification-test-main/data/prepared")
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

LOGIT_TO_PROBIT = 1.0 / 1.7   # CLAUDE.md critical invariant
LAMBDA = 0.025
MIN_TRIALS = 400              # ≥200 per half
SHUFFLE_SEED = 42

# Per-worker cache of the Rasch c maps (one dict per domain).
_C_MAPS: Dict[str, Dict[int, float]] = {}


def _c_probit_map(domain: str) -> Dict[int, float]:
    if domain not in _C_MAPS:
        cdf = pd.read_csv(
            os.path.join(SPIKE_PREPARED, f"sparcnet_{domain}_fit", "c.csv"))
        _C_MAPS[domain] = {
            int(r.case_id): float(r.c_mean) * LOGIT_TO_PROBIT
            for r in cdf.itertuples(index=False)
        }
    return _C_MAPS[domain]


def _probit_lapse_nll(params, c, y):
    """Spike-paper Eq. 2 NLL: p = λ + (1 − 2λ)·Φ((c − θ)/σ).

    params = (θ, log σ).  Identical to
    train_val_split_and_fit.probit_lapse_nll (which is already F0.1-correct).
    """
    t, log_sigma = params
    sigma = np.exp(log_sigma)
    z = (c - t) / sigma
    p = LAMBDA + (1.0 - 2.0 * LAMBDA) * norm.cdf(z)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_half(c: np.ndarray, y: np.ndarray):
    """Return (sigma, theta, converged) via L-BFGS-B on (θ, log σ)."""
    if len(y) < 20 or y.sum() == 0 or y.sum() == len(y):
        return (np.nan, np.nan, False)
    res = minimize(
        _probit_lapse_nll, np.array([0.5, np.log(0.5)]),
        args=(c, y), method="L-BFGS-B",
        bounds=[(-2.0, 2.0), (np.log(0.02), np.log(5.0))],
    )
    if not res.success:
        return (np.nan, np.nan, False)
    t_hat, log_sigma = res.x
    return (float(np.exp(log_sigma)), float(t_hat), True)


def _retest_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (domain, rater) → split-half (σ, θ, AUROC) for both halves."""
    domain = task["domain"]
    rater_id = task["rater_id"]
    cmap = _c_probit_map(domain)

    long_df = pd.read_csv(
        os.path.join(SPIKE_PREPARED, f"sparcnet_{domain}.csv"))
    sub = long_df[long_df["rater_id"] == rater_id]
    if len(sub) < MIN_TRIALS:
        return {"domain": domain, "rater_id": rater_id,
                "n_trials": int(len(sub)), "skipped": "too_few_trials"}

    case_ids = sub["case_id"].to_numpy()
    Y = sub["Y"].to_numpy(dtype=float)
    c = np.array([cmap.get(int(ci), np.nan) for ci in case_ids])
    ok = np.isfinite(c)
    c, Y = c[ok], Y[ok]
    if len(Y) < MIN_TRIALS:
        return {"domain": domain, "rater_id": rater_id,
                "n_trials": int(len(Y)), "skipped": "too_few_after_cjoin"}

    rng = np.random.default_rng(SHUFFLE_SEED + rater_id)
    perm = rng.permutation(len(Y))
    c, Y = c[perm], Y[perm]
    mid = len(Y) // 2
    s1, t1, ok1 = _fit_half(c[:mid], Y[:mid])
    s2, t2, ok2 = _fit_half(c[mid:], Y[mid:])
    if not (ok1 and ok2):
        return {"domain": domain, "rater_id": rater_id,
                "n_trials": int(len(Y)), "skipped": "fit_failed"}

    l1 = -np.log(s1)
    l2 = -np.log(s2)
    auroc1 = float(auroc_from_l(np.array([l1]))[0])
    auroc2 = float(auroc_from_l(np.array([l2]))[0])
    return {
        "domain": domain, "rater_id": rater_id,
        "n_trials": int(len(Y)),
        "sigma_h1": s1, "sigma_h2": s2,
        "theta_h1": t1, "theta_h2": t2,
        "auroc_h1": auroc1, "auroc_h2": auroc2,
        "skipped": None,
    }


def _icc_3_1(x1: np.ndarray, x2: np.ndarray) -> float:
    """Shrout & Fleiss (1979) ICC(3,1): two-way mixed, single measure,
    consistency.  Matches eval_reliability_splithalf.icc_3_1 in the spike repo.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n = len(x1)
    if n < 3:
        return float("nan")
    M = np.column_stack([x1, x2])
    grand = M.mean()
    row_means = M.mean(axis=1)
    col_means = M.mean(axis=0)
    SST = ((M - grand) ** 2).sum()
    SSR = 2.0 * ((row_means - grand) ** 2).sum()          # between-subjects
    SSC = n * ((col_means - grand) ** 2).sum()            # between-measures
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSE = SSE / (n - 1)
    denom = MSR + (2 - 1) * MSE
    return float((MSR - MSE) / denom) if denom != 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-workers", type=int, default=None)
    args = p.parse_args()

    print(f"\n=== F2.6 SPARCNET split-half test-retest ===", flush=True)

    # Build task list from the SDT-fit rosters (rater_id integer codes).
    tasks: List[Dict[str, Any]] = []
    for d in DOMAINS:
        # tidy in-repo sdt_fits (drop-in for the former per-domain
        # sparcnet_{d}_sdt_fits.csv — identical cols/dtypes/values).
        fits = engine_paths.sdt_fits_domain(d)
        for r in fits.itertuples(index=False):
            if int(r.n_trials) >= MIN_TRIALS:
                tasks.append({"domain": d, "rater_id": int(r.rater_id)})
    print(f"  {len(tasks)} (domain, rater) tasks across {len(DOMAINS)} domains",
          flush=True)

    t0 = time.time()
    results = parallel_map(
        _retest_worker, tasks, max_workers=args.max_workers,
        desc="retest fits", ordered=True, progress_every=40,
    )

    rows = [r for r in results
            if isinstance(r, dict) and "__error__" not in r]
    valid = [r for r in rows if r.get("skipped") is None]
    skipped = [r for r in rows if r.get("skipped") is not None]
    print(f"  {len(valid)} valid, {len(skipped)} skipped", flush=True)

    # Per-domain reliability
    print("\n=== Per-domain test-retest reliability ===", flush=True)
    per_domain = {}
    for d in DOMAINS:
        dd = [r for r in valid if r["domain"] == d]
        if len(dd) < 3:
            per_domain[d] = {"n": len(dd), "icc_3_1": None,
                             "pearson_r": None, "pass": False}
            print(f"  {d:5s}: n={len(dd)} (too few for ICC)", flush=True)
            continue
        a1 = np.array([r["auroc_h1"] for r in dd])
        a2 = np.array([r["auroc_h2"] for r in dd])
        icc = _icc_3_1(a1, a2)
        pear = float(np.corrcoef(a1, a2)[0, 1])
        # Also σ-scale reliability for cross-paper comparison with spike's 0.865
        s1 = np.array([r["sigma_h1"] for r in dd])
        s2 = np.array([r["sigma_h2"] for r in dd])
        icc_sigma = _icc_3_1(s1, s2)
        passed = icc >= 0.70
        per_domain[d] = {
            "n": len(dd),
            "icc_3_1_auroc": icc,
            "icc_3_1_sigma": icc_sigma,
            "pearson_r_auroc": pear,
            "pass": bool(passed),
        }
        print(f"  {d:5s}: n={len(dd):2d}  ICC(3,1)_AUROC={icc:.3f}  "
              f"ICC_σ={icc_sigma:.3f}  r={pear:.3f}  "
              f"{'PASS' if passed else 'BELOW 0.70'}", flush=True)

    n_pass = sum(1 for v in per_domain.values() if v.get("pass"))
    print(f"\n  {n_pass}/{len(DOMAINS)} domains ICC(3,1)_AUROC ≥ 0.70",
          flush=True)

    # Outputs
    csv_path = os.path.join(OUT_DIR, "sparcnet_test_retest.csv")
    pd.DataFrame(valid).to_csv(csv_path, index=False)
    md_path = os.path.join(OUT_DIR, "sparcnet_test_retest.md")
    with open(md_path, "w") as f:
        f.write("# F2.6 SPARCNET Split-Half Test-Retest\n\n")
        f.write(f"Split-half (seed={SHUFFLE_SEED}), probit-lapse refit "
                f"(F0.1 Eq. 2, λ={LAMBDA}), AUROC point-estimate "
                f"reliability.  {len(valid)} valid rater-domain pairs.\n\n")
        f.write("| Domain | n | ICC(3,1) AUROC | ICC(3,1) σ | Pearson r | "
                "Status |\n")
        f.write("|---|---|---|---|---|---|\n")
        for d in DOMAINS:
            v = per_domain[d]
            if v.get("icc_3_1_auroc") is None:
                f.write(f"| {d} | {v['n']} | n/a | n/a | n/a | — |\n")
            else:
                st = "✅ PASS" if v["pass"] else "⚠️ < 0.70"
                f.write(f"| {d} | {v['n']} | {v['icc_3_1_auroc']:.3f} | "
                        f"{v['icc_3_1_sigma']:.3f} | "
                        f"{v['pearson_r_auroc']:.3f} | {st} |\n")
        f.write(f"\n**{n_pass}/{len(DOMAINS)} domains pass ICC(3,1)_AUROC "
                f"≥ 0.70.**  Spike-paper single-domain reference: "
                f"ICC(3,1)_σ = 0.865.\n")
        f.write(f"\nTotal compute: {(time.time()-t0)/60:.1f} min.\n")
    with open(os.path.join(OUT_DIR, "sparcnet_test_retest.json"), "w") as f:
        json.dump({"per_domain": per_domain, "n_valid": len(valid),
                   "n_skipped": len(skipped),
                   "total_seconds": time.time() - t0}, f, indent=2)
    print(f"\n  wrote {csv_path}\n  wrote {md_path}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
