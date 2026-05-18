"""F2.7 (2026-05-15, rev) — lapse-rate misspecification bias (Wichmann-Hill).

Wichmann & Hill (2001) is about the bias a misspecified lapse rate
induces in the **SDT fit** (slope/discrimination), not about adaptive
testing.  The cleanest, fastest measurement of exactly that concern:

  For each (synthetic rater, domain, λ_true):
    1. Generate a FIXED set of n_obs (c_probit, y) observations under the
       true response model with lapse λ_true and the rater's true (σ, θ).
    2. Fit (σ̂, θ̂) by maximum likelihood ASSUMING the production lapse
       λ_assumed = 0.025  (the misspecified fit when λ_true ≠ 0.025).
    3. AUROC bias = AUROC(σ̂) − AUROC(σ_true).

The well-specified cell (λ_true = 0.025) is the control — its residual
bias is the finite-sample MLE noise floor; the misspecification effect is
the EXCESS bias of the other λ_true cells over that floor.

This is a pure fixed-design MLE study (no SMC, no adaptive selection):
~thousands of L-BFGS-B fits, seconds on 14 workers, bitwise-deterministic.

Acceptance: excess |AUROC bias| (misspecified − well-specified) ≤ 0.02
across the λ_true grid → the fixed-λ=0.025 assumption is robust enough
for Paper 1.

Outputs:
  results/phase2_validation/lapse_sensitivity.csv
  results/phase2_validation/lapse_sensitivity.md
"""
from __future__ import annotations

import argparse
import csv
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
from scipy.optimize import minimize  # noqa: E402
from scipy.special import ndtr  # noqa: E402

ENGINE_REPO = os.path.dirname(_THIS_DIR_BOOT)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)
from core_mcmc import load_fitted_Sigma, sample_prior_hier_K  # noqa: E402
from auroc import auroc_from_l  # noqa: E402

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
LAMBDA_TRUE_GRID = [0.01, 0.025, 0.05, 0.10]
LAMBDA_ASSUMED = 0.025
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)


def _p_yes(c, theta, sigma, lam):
    """Spike-paper Eq. 2 response model."""
    z = (c - theta) / sigma
    return lam + (1.0 - 2.0 * lam) * ndtr(z)


def _nll_assumed_lambda(params, c, y):
    """NLL fit ASSUMING λ = LAMBDA_ASSUMED (the misspecified objective)."""
    t, log_s = params
    p = _p_yes(c, t, np.exp(log_s), LAMBDA_ASSUMED)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (rater, λ_true) → per-domain AUROC bias under misspecified fit."""
    K = task["K"]
    lam_true = task["lambda_true"]
    n_obs = task["n_obs"]
    sigma_true = np.asarray(task["sigma_true"], dtype=float)   # (K,)
    theta_true = np.asarray(task["theta_true"], dtype=float)   # (K,)
    l_true = -np.log(sigma_true)
    auroc_true = auroc_from_l(l_true)

    rng = np.random.default_rng(task["seed"])
    # Fixed signal design: spread c over the informative probit range.
    c_grid = np.linspace(-2.5, 2.5, n_obs)
    biases = np.empty(K)
    sig_hat = np.empty(K)
    for k in range(K):
        p = _p_yes(c_grid, theta_true[k], sigma_true[k], lam_true)
        y = (rng.random(n_obs) < p).astype(float)
        res = minimize(
            _nll_assumed_lambda, np.array([0.5, np.log(0.5)]),
            args=(c_grid, y), method="L-BFGS-B",
            bounds=[(-2.0, 2.0), (np.log(0.02), np.log(5.0))],
        )
        if not res.success:
            sig_hat[k] = np.nan
            biases[k] = np.nan
            continue
        s_hat = float(np.exp(res.x[1]))
        sig_hat[k] = s_hat
        auroc_hat = float(auroc_from_l(np.array([-np.log(s_hat)]))[0])
        biases[k] = auroc_hat - auroc_true[k]
    ok = np.isfinite(biases)
    return {
        "rater_idx": task["rater_idx"],
        "lambda_true": lam_true,
        "auroc_bias_signed_mean": float(np.mean(biases[ok])),
        "auroc_bias_abs_mean": float(np.mean(np.abs(biases[ok]))),
        "auroc_bias_abs_max": float(np.max(np.abs(biases[ok]))),
        "n_domains_ok": int(ok.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-raters", type=int, default=100)
    p.add_argument("--n-obs", type=int, default=500,
                   help="Fixed observations per domain per fit (default 500).")
    p.add_argument("--seed-base", type=int, default=314)
    p.add_argument("--max-workers", type=int, default=None)
    args = p.parse_args()

    K = 6
    obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    print(f"\n=== F2.7 lapse misspecification bias (Wichmann-Hill) ===",
          flush=True)
    print(f"  λ_assumed (fit) = {LAMBDA_ASSUMED}; λ_true grid = "
          f"{LAMBDA_TRUE_GRID}", flush=True)
    print(f"  n_raters={args.n_raters} n_obs={args.n_obs}", flush=True)

    # Synthetic raters: skill ℓ from the Corr_l prior; θ ~ N(0, 0.5²).
    rng = np.random.default_rng(args.seed_base + 99)
    _, l_all = sample_prior_hier_K(N=args.n_raters, K=K, r=0.378, rng=rng,
                                   Sigma_l=Corr_l, Sigma_t=Corr_l)
    sigma_all = np.exp(-l_all)                       # σ = exp(-ℓ)
    theta_all = rng.normal(0.0, 0.5, size=(args.n_raters, K))

    tasks: List[Dict[str, Any]] = []
    for ri in range(args.n_raters):
        for lam in LAMBDA_TRUE_GRID:
            tasks.append({
                "rater_idx": ri, "K": K, "lambda_true": lam,
                "n_obs": args.n_obs,
                "sigma_true": sigma_all[ri].tolist(),
                "theta_true": theta_all[ri].tolist(),
                "seed": args.seed_base + ri,   # paired across λ
            })

    t0 = time.time()
    results = parallel_map(
        _worker, tasks, max_workers=args.max_workers,
        desc="lapse fits", ordered=True, progress_every=100,
    )
    rows = [r for r in results
            if isinstance(r, dict) and "__error__" not in r]

    print("\n=== Lapse misspecification summary ===", flush=True)
    summary = {}
    for lam in LAMBDA_TRUE_GRID:
        dd = [r for r in rows if r["lambda_true"] == lam]
        abs_mean = float(np.mean([r["auroc_bias_abs_mean"] for r in dd]))
        sgn_mean = float(np.mean([r["auroc_bias_signed_mean"] for r in dd]))
        summary[lam] = {"n": len(dd), "abs_bias": abs_mean,
                        "signed_bias": sgn_mean}
    floor = summary[LAMBDA_ASSUMED]["abs_bias"]
    worst_excess = 0.0
    for lam in LAMBDA_TRUE_GRID:
        excess = summary[lam]["abs_bias"] - floor
        summary[lam]["excess_over_wellspecified"] = excess
        worst_excess = max(worst_excess, excess if lam != LAMBDA_ASSUMED
                           else 0.0)
        tag = "  ← well-specified (noise floor)" \
            if lam == LAMBDA_ASSUMED else ""
        print(f"  λ_true={lam:<5}: |AUROC bias|={summary[lam]['abs_bias']:.4f}"
              f" (signed {summary[lam]['signed_bias']:+.4f})  "
              f"excess={excess:+.4f}{tag}", flush=True)
    verdict = "ROBUST" if worst_excess <= 0.02 else "SENSITIVE"
    print(f"\n  Noise floor (well-specified) = {floor:.4f}; worst EXCESS "
          f"bias from misspecification = {worst_excess:.4f} → {verdict} "
          f"(threshold 0.02)", flush=True)

    with open(os.path.join(OUT_DIR, "lapse_sensitivity.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "lapse_sensitivity.md"), "w") as f:
        f.write("# F2.7 Lapse-Rate Misspecification Bias (Wichmann-Hill)\n\n")
        f.write(f"Fit assumes λ={LAMBDA_ASSUMED}; data generated with "
                f"λ_true.  {args.n_raters} synthetic raters × 6 domains, "
                f"n_obs={args.n_obs} fixed design, MLE fit.\n\n")
        f.write("| λ_true | |AUROC bias| | signed bias | excess over "
                "well-specified |\n|---|---|---|---|\n")
        for lam in LAMBDA_TRUE_GRID:
            s = summary[lam]
            tag = " (control)" if lam == LAMBDA_ASSUMED else ""
            f.write(f"| {lam}{tag} | {s['abs_bias']:.4f} | "
                    f"{s['signed_bias']:+.4f} | "
                    f"{s['excess_over_wellspecified']:+.4f} |\n")
        f.write(f"\nNoise floor (well-specified MLE) = **{floor:.4f}**; "
                f"worst excess bias from λ misspecification = "
                f"**{worst_excess:.4f}** → **{verdict}** "
                f"(threshold 0.02).\n\n")
        f.write("Interpretation: the excess bias isolates the "
                "Wichmann-Hill effect (the well-specified cell's bias is "
                "the finite-sample MLE noise floor, not misspecification). "
                "A small worst-excess means assuming λ=0.025 when the true "
                "lapse is 4× higher (0.10) or 2.5× lower (0.01) does not "
                "materially distort the AUROC measurement — supporting the "
                "fixed-λ choice for Paper 1.\n")
    with open(os.path.join(OUT_DIR, "lapse_sensitivity.json"), "w") as f:
        json.dump({"summary": {str(k): v for k, v in summary.items()},
                   "noise_floor": floor, "worst_excess": worst_excess,
                   "verdict": verdict,
                   "total_seconds": time.time() - t0}, f, indent=2)
    print(f"\n  wrote lapse_sensitivity.{{csv,md,json}}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
