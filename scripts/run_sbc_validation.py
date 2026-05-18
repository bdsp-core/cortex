"""F2.1 (2026-05-15) — Simulation-based calibration (SBC) for Mode-A SMC.

Talts, Betancourt, Simpson, Vehtari, Gelman (2018) "Validating Bayesian
Inference Algorithms with Simulation-Based Calibration".  Verifies that
the SMC posterior produced by `run_session_mcmc_auroc` is calibrated
against its data-generating prior.

Protocol
--------
For each of N_DRAWS prior samples (t_star, l_star) ~ p(θ):
  1. Generate n_obs observations under the response model with θ = θ_star.
  2. Run the SMC engine to produce a posterior draw cloud over θ.
  3. For each scalar parameter θ_k, compute the rank of θ_star within the
     weighted SMC marginal:
        rank(θ_star) = sum_i w_i * 1[θ_i < θ_star]   (scaled to [0, 1])
  4. Pool rank values across N_DRAWS, bin into B bins.

Under correct calibration, the rank histogram should be uniform.  Test:
  - bin counts within Bonferroni-adjusted 95% binomial band per parameter.
  - per-parameter Kolmogorov-Smirnov against uniform.

Sample selection: uses uniform-domain random questions (not boundary-Fisher
adaptive selection) so the data-generating distribution matches the
inferential prior predictive.  Adaptive item selection changes the data
distribution conditional on the prior and would violate the SBC assumption.

Outputs:
  results/phase2_validation/sbc_results.json   (rank arrays per parameter)
  results/phase2_validation/sbc_summary.md     (markdown table)
  results/phase2_validation/sbc_rank_histograms.png  (12-panel grid)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# CRITICAL: single-thread BLAS before numpy import.
_THIS_DIR_BOOT = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR_BOOT not in sys.path:
    sys.path.insert(0, _THIS_DIR_BOOT)
from _parallel import configure_blas_single_thread, parallel_map  # noqa: E402
configure_blas_single_thread()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import binom, kstest  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

from core_mcmc import (
    load_fitted_Sigma, make_state_hier, sample_prior_hier_K,
    update, ess, resample_and_rejuvenate, simulate_response,
)

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── lazy per-worker shared read-only cache (spawn-safe) ───────────────
_SHARED: Dict[str, Any] = {}


def _get_corr_l() -> np.ndarray:
    if "Corr_l" not in _SHARED:
        obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
        _SHARED["Corr_l"] = np.asarray(obj["Corr_l"], dtype=float)
    return _SHARED["Corr_l"]


def _sbc_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level picklable worker: one SBC prior draw → rank dict."""
    Corr_l = _get_corr_l()
    return run_one_sbc_draw(
        K=task["K"], Sigma_l=Corr_l, Sigma_t=Corr_l,
        n_obs=task["n_obs"], N_particles=task["N_particles"],
        seed=task["seed"], n_mh_steps=task["n_mh_steps"],
        ess_threshold_frac=task["ess_threshold_frac"],
    )


def weighted_rank(value, samples, weights):
    """Weighted rank: fraction of samples (with weights) strictly below `value`.

    Returns a float in [0, 1].
    """
    w = weights / weights.sum()
    return float((w * (samples < value)).sum())


def run_one_sbc_draw(*, K, Sigma_l, Sigma_t, n_obs, N_particles, seed,
                      n_mh_steps, ess_threshold_frac):
    """One SBC draw: sample θ_star, generate data uniform-random, run SMC,
    return ranks of θ_star for each of 2K coordinates."""
    rng = np.random.default_rng(seed)
    # 1. draw θ_star from prior (single sample)
    t_star, l_star = sample_prior_hier_K(
        N=1, K=K, r=0.378, rng=rng, Sigma_l=Sigma_l, Sigma_t=Sigma_t,
    )
    t_star = t_star[0]
    l_star = l_star[0]
    # 2. generate n_obs observations under uniform-domain, uniform-signal design
    questions = []
    for _ in range(n_obs):
        k = int(rng.integers(0, K))
        s = float(rng.uniform(-2.5, 2.5))
        y = simulate_response(s, t_star[k], l_star[k], rng)
        questions.append((k, s, y))
    # 3. run SMC engine to convergence
    smc_rng = np.random.default_rng(seed + 10**6)
    state = make_state_hier(N_particles, K, r_assumed=0.378, rng=smc_rng,
                              Sigma_l=Sigma_l, Sigma_t=Sigma_t)
    proposal_scale = 2.38 / np.sqrt(2 * K)
    for (k, s, y) in questions:
        update(state, k, s, y)
        if ess(state["w"]) < ess_threshold_frac * N_particles:
            resample_and_rejuvenate(state, smc_rng, n_mh_steps, proposal_scale)
    # 4. compute ranks of θ_star per coordinate
    ranks_t = np.zeros(K)
    ranks_l = np.zeros(K)
    w = state["w"]
    for k in range(K):
        ranks_t[k] = weighted_rank(t_star[k], state["t"][:, k], w)
        ranks_l[k] = weighted_rank(l_star[k], state["l"][:, k], w)
    return {
        "ranks_t": ranks_t.tolist(),
        "ranks_l": ranks_l.tolist(),
        "t_star": t_star.tolist(),
        "l_star": l_star.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-draws", type=int, default=100,
                   help="Number of prior draws (default 100).")
    p.add_argument("--n-obs", type=int, default=120,
                   help="Observations per draw under uniform-random design (default 120).")
    p.add_argument("--N-particles", type=int, default=600,
                   help="Particles per session (default 600).")
    p.add_argument("--n-bins", type=int, default=20,
                   help="Rank histogram bins (default 20).")
    p.add_argument("--seed-base", type=int, default=7,
                   help="Starting RNG seed (default 7).")
    p.add_argument("--n-mh-steps", type=int, default=15,
                   help="MH steps per rejuvenation (default 15).")
    p.add_argument("--ess-threshold-frac", type=float, default=0.5,
                   help="ESS resample threshold (default 0.5).")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Parallel worker processes (default: 14 / cpu-2).")
    args = p.parse_args()

    K = 6

    print(f"\n=== F2.1 SBC validation ===", flush=True)
    print(f"  N_draws={args.n_draws} n_obs={args.n_obs} "
          f"N_particles={args.N_particles} B={args.n_bins}", flush=True)

    tasks: List[Dict[str, Any]] = [
        {
            "K": K,
            "n_obs": args.n_obs,
            "N_particles": args.N_particles,
            "seed": args.seed_base + di,
            "n_mh_steps": args.n_mh_steps,
            "ess_threshold_frac": args.ess_threshold_frac,
        }
        for di in range(args.n_draws)
    ]

    t0 = time.time()
    raw = parallel_map(
        _sbc_worker, tasks,
        max_workers=args.max_workers,
        desc="SBC draws",
        ordered=True,
        progress_every=10,
    )
    draws = []
    for di, d in enumerate(raw):
        if isinstance(d, dict) and "__error__" in d:
            print(f"  [SKIP] draw {di}: {d['__error__']}", flush=True)
            continue
        draws.append(d)

    # Bonferroni-adjusted band per parameter (2K=12 params)
    n_params = 2 * K
    alpha_adj = 0.05 / n_params
    expected_per_bin = args.n_draws / args.n_bins
    lo_band, hi_band = binom.interval(1 - alpha_adj, args.n_draws, 1.0 / args.n_bins)
    print(f"\n  per-bin expected count = {expected_per_bin:.1f}", flush=True)
    print(f"  Bonferroni-adjusted 95% band: [{lo_band}, {hi_band}]", flush=True)

    # Build rank arrays per param
    rank_arrs: Dict[str, np.ndarray] = {}
    for k in range(K):
        rank_arrs[f"t_{DOMAINS[k]}"] = np.array([d["ranks_t"][k] for d in draws])
        rank_arrs[f"l_{DOMAINS[k]}"] = np.array([d["ranks_l"][k] for d in draws])

    # Per-param diagnostic
    param_results = {}
    for name, ranks in rank_arrs.items():
        # KS test against Uniform(0, 1)
        ks_stat, ks_p = kstest(ranks, "uniform")
        # Bin counts
        counts, _ = np.histogram(ranks, bins=args.n_bins, range=(0.0, 1.0))
        out_of_band = int(((counts < lo_band) | (counts > hi_band)).sum())
        param_results[name] = {
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "n_bins_out_of_band": out_of_band,
            "bin_counts": counts.tolist(),
        }

    # Summary
    print("\n=== SBC summary ===", flush=True)
    n_pass = 0
    for name, res in param_results.items():
        status = "PASS" if res["n_bins_out_of_band"] <= 1 else "FAIL"
        n_pass += int(status == "PASS")
        print(f"  {name:8s}  KS p={res['ks_pvalue']:.3f}  "
              f"bins_out_of_band={res['n_bins_out_of_band']}/{args.n_bins}  {status}",
              flush=True)
    print(f"\n  {n_pass}/{n_params} parameters PASS Bonferroni-adjusted SBC.",
          flush=True)

    # Save JSON
    json_path = os.path.join(OUT_DIR, "sbc_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "n_draws": args.n_draws, "n_obs": args.n_obs,
                "N_particles": args.N_particles, "n_bins": args.n_bins,
                "K": K, "domains": DOMAINS,
                "bonferroni_band": [int(lo_band), int(hi_band)],
                "alpha_adjusted": alpha_adj,
            },
            "param_results": param_results,
            "n_draws_completed": len(draws),
            "n_pass": int(n_pass),
            "total_seconds": time.time() - t0,
        }, f, indent=2)
    print(f"  wrote {json_path}", flush=True)

    # Markdown summary
    md_path = os.path.join(OUT_DIR, "sbc_summary.md")
    with open(md_path, "w") as f:
        f.write("# Simulation-Based Calibration (SBC) — F2.1\n\n")
        f.write(f"Talts et al. 2018.  N_draws = {args.n_draws}, "
                f"n_obs = {args.n_obs} (uniform-random design), "
                f"N_particles = {args.N_particles}, B = {args.n_bins} bins.\n\n")
        f.write(f"Bonferroni-adjusted 95% band per bin: "
                f"[{int(lo_band)}, {int(hi_band)}] expected ≈ "
                f"{expected_per_bin:.1f}.\n\n")
        f.write("| Parameter | KS p | bins out of band | Status |\n")
        f.write("|---|---|---|---|\n")
        for name, res in param_results.items():
            status = "✅ PASS" if res["n_bins_out_of_band"] <= 1 else "⚠️ FAIL"
            f.write(f"| {name} | {res['ks_pvalue']:.3f} | "
                    f"{res['n_bins_out_of_band']}/{args.n_bins} | {status} |\n")
        f.write(f"\n**Pass count: {n_pass}/{n_params}.**\n")
        f.write(f"\nTotal compute: {(time.time() - t0)/60:.1f} min.\n")
    print(f"  wrote {md_path}", flush=True)

    # Rank histograms figure (12-panel grid)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, K, figsize=(2.0 * K, 4.2), sharey=True)
    for k in range(K):
        for row, prefix in enumerate(["t", "l"]):
            ax = axes[row, k]
            name = f"{prefix}_{DOMAINS[k]}"
            ranks = rank_arrs[name]
            counts, edges = np.histogram(ranks, bins=args.n_bins, range=(0.0, 1.0))
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.bar(centers, counts, width=1.0 / args.n_bins,
                   color="#0072B2", edgecolor="white", linewidth=0.4)
            ax.axhspan(lo_band, hi_band, color="#888888", alpha=0.18,
                       linewidth=0)
            ax.axhline(expected_per_bin, color="#444444", linestyle="--",
                       linewidth=0.6)
            ax.set_title(name, fontsize=8)
            ax.set_xlim(0, 1)
            if row == 1:
                ax.set_xlabel("rank")
            if k == 0:
                ax.set_ylabel("count")
    fig.suptitle("SBC rank histograms (uniform if calibrated)", fontsize=10)
    fig.tight_layout()
    fig_path_pdf = os.path.join(OUT_DIR, "sbc_rank_histograms.pdf")
    fig_path_png = os.path.join(OUT_DIR, "sbc_rank_histograms.png")
    fig.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fig_path_pdf}", flush=True)
    print(f"  wrote {fig_path_png}", flush=True)
    print(f"\nDone in {(time.time() - t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
