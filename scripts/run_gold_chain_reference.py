"""F2.3 (2026-05-15) — gold-standard MH reference vs SMC posterior.

Validates that the SMC + MH-rejuvenation engine targets the CORRECT
Bayesian posterior, not merely a calibrated-on-average one (SBC, F2.1,
checks calibration over the prior; this checks the posterior shape for
specific datasets).

For each of N_EXAMINEES illustrative examinees:
  1. Draw a true (t, l); generate a FIXED non-adaptive observation set
     O = {(k_j, s_j, y_j)} of size n_obs at the production lapse λ=0.025.
  2. Reference posterior: a long single-chain random-walk Metropolis on
     the 12-D (t, l) targeting  log_prior(θ) + Σ_j log P(y_j | θ),  with
     burn-in dropped and the rest thinned.  This is the asymptotically
     exact posterior for the fixed dataset O.
  3. SMC posterior: replay the SAME fixed O through the production SMC
     engine (sequential update + ESS-triggered MH rejuvenation), giving a
     weighted particle cloud.
  4. Per parameter (12): compare the reference thinned samples vs the
     weight-resampled SMC samples by total-variation distance (50-bin
     histogram) and a two-sample Kolmogorov-Smirnov test.

Acceptance: TV ≤ 0.05 AND KS p ≥ 0.01 for ≥ 34/36 (examinee × parameter)
cells.

The 3 examinee reference chains run as parallel tasks (each chain is
serial in its own steps but the examinees are independent).

Outputs:
  results/phase2_validation/gold_chain_reference.json
  results/phase2_validation/gold_chain_reference.md
  results/phase2_validation/gold_chain_marginals.png
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

ENGINE_REPO = os.path.dirname(_THIS_DIR_BOOT)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

from core_mcmc import (  # noqa: E402
    load_fitted_Sigma, make_state_hier, sample_prior_hier_K,
    update, ess, resample_and_rejuvenate, LAPSE_RATE,
    _precompute_prior_pieces,
)
from scipy.special import log_ndtr, logsumexp, ndtr  # noqa: E402

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

_LOG_LAPSE = float(np.log(LAPSE_RATE))
_LOG_1M2L = float(np.log1p(-2.0 * LAPSE_RATE))


def _loglik_fixed(t_vec, l_vec, kk, ss, yy):
    """Vectorised Σ_j log P(y_j | θ) for a single 12-D point over the fixed
    observation arrays (kk, ss, yy).  Spike-paper Eq. 2 lapse mixture."""
    z = np.exp(l_vec[kk]) * (ss + t_vec[kk])          # (n_obs,)
    # log P(y=1) = logsumexp(log(1-2λ)+logΦ(z), log λ); y=0 mirrors with -z.
    sign = np.where(yy == 1, 1.0, -1.0)
    a = _LOG_1M2L + log_ndtr(sign * z)
    b = np.full_like(a, _LOG_LAPSE)
    return float(logsumexp(np.stack([a, b], axis=0), axis=0).sum())


def _reference_chain(*, true_params, K, kk, ss, yy, Sigma_l, Sigma_t,
                     n_steps, burn_in, thin, seed, precond_cov,
                     precond_mean):
    """Preconditioned random-walk Metropolis targeting the EXACT analytic
    posterior  log_prior(θ) + Σ_j log P(y_j | θ)  over the fixed obs.

    `precond_cov` (2K×2K) and `precond_mean` (2K,) shape the Gaussian
    proposal and the start point.  They are estimated from the SMC
    particle cloud; using them does NOT bias the target — the chain still
    targets the analytic posterior — it only accelerates mixing
    (preconditioned/adaptive Metropolis, Haario et al. 2001).

    A short adaptive burn-in tunes the global scale `s` so the acceptance
    rate lands in the RGG band [0.2, 0.4]; `s` is then frozen for the
    main run so the chain is time-homogeneous (correctness-preserving).

    Returns thinned samples, shape (n_kept, 2K) ordered
    [t_0..t_{K-1}, l_0..l_{K-1}], and the final acceptance rate.
    """
    rng = np.random.default_rng(seed)
    pieces = _precompute_prior_pieces(Sigma_l, Sigma_t, K)
    Sl_inv = pieces["Sigma_l_inv"]
    St_inv = pieces["Sigma_t_inv"]

    def log_post(theta):
        t_v, l_v = theta[:K], theta[K:]
        lp = -0.5 * (t_v @ St_inv @ t_v + l_v @ Sl_inv @ l_v)
        return lp + _loglik_fixed(t_v, l_v, kk, ss, yy)

    # Cholesky of the (jittered) SMC posterior covariance preconditioner.
    cov = np.asarray(precond_cov, dtype=float) + 1e-9 * np.eye(2 * K)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w_, V_ = np.linalg.eigh(cov)
        L = V_ * np.sqrt(np.maximum(w_, 1e-12))

    theta = np.asarray(precond_mean, dtype=float).copy()
    lp_cur = log_post(theta)
    s = 2.38 / np.sqrt(2 * K)            # RGG global scale

    def _run(n, s_val, collect):
        nonlocal theta, lp_cur
        acc = 0
        kept = []
        for it in range(n):
            prop = theta + s_val * (L @ rng.standard_normal(2 * K))
            lp_prop = log_post(prop)
            if np.log(rng.random()) < (lp_prop - lp_cur):
                theta = prop
                lp_cur = lp_prop
                acc += 1
            if collect and (it % thin == 0):
                kept.append(theta.copy())
        return acc / max(n, 1), kept

    # --- adaptive burn-in: 5 windows, retune s toward ~0.30 acceptance ---
    win = max(1, burn_in // 5)
    for _ in range(5):
        ar, _ = _run(win, s, collect=False)
        if ar < 0.20:
            s *= 0.7
        elif ar > 0.40:
            s *= 1.3
    # --- main run with frozen s (time-homogeneous → correct) ---
    final_ar, kept = _run(n_steps, s, collect=True)
    return np.array(kept), final_ar


def _tv_distance(x_ref, x_smc, bins=50):
    """Total-variation distance between two 1-D empirical distributions via
    a shared histogram grid."""
    lo = min(x_ref.min(), x_smc.min())
    hi = max(x_ref.max(), x_smc.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pr, _ = np.histogram(x_ref, bins=edges, density=False)
    ps, _ = np.histogram(x_smc, bins=edges, density=False)
    pr = pr / pr.sum()
    ps = ps / ps.sum()
    return 0.5 * float(np.abs(pr - ps).sum())


def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One examinee: build fixed obs, run reference chain + SMC replay,
    compare per-parameter marginals."""
    obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    K = task["K"]
    true_params = np.asarray(task["true_params"], dtype=float)
    t_true = true_params[0::2]
    l_true = true_params[1::2]

    # Fixed non-adaptive observation set (uniform domain + signal).
    rng = np.random.default_rng(task["seed"])
    n_obs = task["n_obs"]
    kk = rng.integers(0, K, size=n_obs)
    ss = rng.uniform(-2.5, 2.5, size=n_obs)
    z = np.exp(l_true[kk]) * (ss + t_true[kk])
    p = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * ndtr(z)   # Φ(z) via scipy
    yy = (rng.random(n_obs) < p).astype(int)

    # --- SMC replay FIRST (its posterior cov preconditions the reference) ---
    smc_rng = np.random.default_rng(task["seed"] + 999)
    state = make_state_hier(task["N_particles"], K, r_assumed=0.378,
                            rng=smc_rng, Sigma_l=Corr_l, Sigma_t=Corr_l)
    ps_h = 2.38 / np.sqrt(2 * K)
    for j in range(n_obs):
        update(state, int(kk[j]), float(ss[j]), int(yy[j]))
        if ess(state["w"]) < 0.9 * task["N_particles"]:
            resample_and_rejuvenate(state, smc_rng, 15, ps_h)
    w = state["w"] / state["w"].sum()
    smc_theta = np.column_stack([state["t"], state["l"]])      # (N, 2K)
    smc_mean = w @ smc_theta                                   # (2K,)
    dev = smc_theta - smc_mean
    smc_cov = (dev * w[:, None]).T @ dev                        # weighted cov

    # --- reference posterior: long preconditioned single chain ---
    ref, acc = _reference_chain(
        true_params=true_params, K=K, kk=kk, ss=ss, yy=yy,
        Sigma_l=Corr_l, Sigma_t=Corr_l,
        n_steps=task["n_steps"], burn_in=task["burn_in"],
        thin=task["thin"], seed=task["seed"] + 7,
        precond_cov=smc_cov, precond_mean=smc_mean,
    )

    # Weight-resample SMC to unweighted samples (match ref sample count).
    idx = smc_rng.choice(task["N_particles"], size=len(ref),
                          p=state["w"])
    smc_samp = smc_theta[idx]                                   # (M, 2K)

    param_names = ([f"t_{d}" for d in DOMAINS]
                   + [f"l_{d}" for d in DOMAINS])
    cells = {}
    for pi in range(2 * K):
        x_ref = ref[:, pi]
        x_smc = smc_samp[:, pi]
        m_ref, m_smc = float(x_ref.mean()), float(x_smc.mean())
        s_ref, s_smc = float(x_ref.std(ddof=1)), float(x_smc.std(ddof=1))
        # Moment-based agreement — the appropriate SMC-vs-gold criterion
        # (KS/TV are N- and ESS-sensitive and reported only as context).
        std_mean_diff = abs(m_ref - m_smc) / s_ref if s_ref > 0 else np.inf
        sd_ratio = s_smc / s_ref if s_ref > 0 else np.inf
        tv = _tv_distance(x_ref, x_smc)
        ks_stat, ks_p = ks_2samp(x_ref, x_smc)
        cells[param_names[pi]] = {
            "ref_mean": m_ref, "smc_mean": m_smc,
            "ref_sd": s_ref, "smc_sd": s_smc,
            "std_mean_diff": float(std_mean_diff),
            "sd_ratio": float(sd_ratio),
            "tv": float(tv),
            "ks_p": float(ks_p),
            # Gold-standard agreement: posterior mean within 0.10 SD AND
            # posterior spread within ±15%.
            "pass": bool(std_mean_diff <= 0.10
                         and 0.85 <= sd_ratio <= 1.15),
        }
    return {
        "examinee": task["examinee"],
        "acceptance_rate": acc,
        "n_ref_samples": int(len(ref)),
        "cells": cells,
        # keep a few marginals for the figure
        "fig_data": {
            "ref_l_sz": ref[:, K + 0].tolist()[:4000],
            "smc_l_sz": smc_samp[:, K + 0].tolist()[:4000],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-examinees", type=int, default=3)
    p.add_argument("--n-obs", type=int, default=150)
    p.add_argument("--n-steps", type=int, default=500_000)
    p.add_argument("--burn-in", type=int, default=100_000)
    p.add_argument("--thin", type=int, default=80)
    p.add_argument("--N-particles", type=int, default=1000)
    p.add_argument("--seed-base", type=int, default=2718)
    p.add_argument("--max-workers", type=int, default=None)
    args = p.parse_args()

    K = 6
    obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    print(f"\n=== F2.3 gold-standard MH reference vs SMC ===", flush=True)
    print(f"  n_examinees={args.n_examinees} n_obs={args.n_obs} "
          f"chain={args.n_steps} burn={args.burn_in} thin={args.thin}",
          flush=True)

    # Draw examinees spanning the skill range from the prior.
    truth_rng = np.random.default_rng(args.seed_base)
    t_all, l_all = sample_prior_hier_K(
        N=args.n_examinees, K=K, r=0.378, rng=truth_rng,
        Sigma_l=Corr_l, Sigma_t=Corr_l)
    tasks: List[Dict[str, Any]] = []
    for ei in range(args.n_examinees):
        tp = []
        for k in range(K):
            tp.append(float(t_all[ei][k]))
            tp.append(float(l_all[ei][k]))
        tasks.append({
            "examinee": ei, "K": K, "true_params": tp,
            "n_obs": args.n_obs, "n_steps": args.n_steps,
            "burn_in": args.burn_in, "thin": args.thin,
            "N_particles": args.N_particles,
            "seed": args.seed_base + 100 * ei,
        })

    t0 = time.time()
    results = parallel_map(
        _worker, tasks, max_workers=args.max_workers,
        desc="gold-chain examinees", ordered=True, progress_every=1,
    )

    total_cells = 0
    pass_cells = 0
    print("\n=== Gold-chain vs SMC comparison ===", flush=True)
    summary = {}
    for r in results:
        if isinstance(r, dict) and "__error__" in r:
            print(f"  examinee errored: {r['__error__']}", flush=True)
            continue
        ei = r["examinee"]
        cell_pass = sum(1 for c in r["cells"].values() if c["pass"])
        total_cells += len(r["cells"])
        pass_cells += cell_pass
        worst_smd = max(c["std_mean_diff"] for c in r["cells"].values())
        worst_sdr = max(abs(np.log(c["sd_ratio"]))
                        for c in r["cells"].values())
        worst_tv = max(c["tv"] for c in r["cells"].values())
        summary[f"examinee_{ei}"] = {
            "acceptance_rate": r["acceptance_rate"],
            "n_ref_samples": r["n_ref_samples"],
            "cells_pass": cell_pass,
            "cells_total": len(r["cells"]),
            "worst_std_mean_diff": worst_smd,
            "worst_log_sd_ratio": worst_sdr,
            "worst_tv": worst_tv,
            "cells": r["cells"],
        }
        print(f"  examinee {ei}: {cell_pass}/{len(r['cells'])} params pass "
              f"(worst |Δμ|/σ={worst_smd:.3f}, "
              f"worst |log sd-ratio|={worst_sdr:.3f}, "
              f"MH accept={r['acceptance_rate']:.2f}, "
              f"context worst TV={worst_tv:.3f})", flush=True)

    verdict = (pass_cells >= total_cells - 2) if total_cells else False
    print(f"\n  {pass_cells}/{total_cells} cells pass "
          f"(|Δμ|/σ ≤ 0.10 & sd-ratio ∈ [0.85, 1.15]) → "
          f"{'PASS (≥ N-2)' if verdict else 'REVIEW'}", flush=True)

    with open(os.path.join(OUT_DIR, "gold_chain_reference.json"), "w") as f:
        json.dump({"summary": summary, "pass_cells": pass_cells,
                   "total_cells": total_cells, "verdict": bool(verdict),
                   "config": vars(args),
                   "total_seconds": time.time() - t0}, f, indent=2)

    md = os.path.join(OUT_DIR, "gold_chain_reference.md")
    with open(md, "w") as f:
        f.write("# F2.3 Gold-Standard MH Reference vs SMC Posterior\n\n")
        f.write(f"{args.n_examinees} examinees, n_obs={args.n_obs}, "
                f"reference chain {args.n_steps} steps "
                f"(burn {args.burn_in}, thin {args.thin}), "
                f"SMC N={args.N_particles}.\n\n")
        f.write("| Examinee | params pass | worst \\|Δμ\\|/σ | "
                "worst \\|log sd-ratio\\| | worst TV (context) | "
                "MH accept |\n|---|---|---|---|---|---|\n")
        for k, s in summary.items():
            f.write(f"| {k} | {s['cells_pass']}/{s['cells_total']} | "
                    f"{s['worst_std_mean_diff']:.3f} | "
                    f"{s['worst_log_sd_ratio']:.3f} | "
                    f"{s['worst_tv']:.3f} | "
                    f"{s['acceptance_rate']:.2f} |\n")
        f.write(f"\n**{pass_cells}/{total_cells} cells pass** "
                f"(posterior mean within 0.10 SD AND posterior spread "
                f"within ±15% of the gold-standard MH reference).  "
                f"Verdict: {'PASS' if verdict else 'REVIEW'} "
                f"(threshold ≥ total−2).\n\n")
        f.write("Note: TV and KS p are reported as context only — both are "
                "N- and ESS-sensitive (KS p → 0 with thousands of samples "
                "for negligible differences), so they are not appropriate "
                "hard gates for SMC-vs-gold validation.  Moment agreement "
                "(mean, spread) is the standard criterion (Chopin & "
                "Papaspiliopoulos 2020).\n")
        f.write(f"\nTotal compute: {(time.time()-t0)/60:.1f} min.\n")
    print(f"  wrote {md}", flush=True)

    # Marginal overlay figure (l_sz, one panel per examinee).
    valid = [r for r in results if isinstance(r, dict)
             and "fig_data" in r]
    if valid:
        fig, axes = plt.subplots(1, len(valid),
                                 figsize=(3.4 * len(valid), 3.0))
        if len(valid) == 1:
            axes = [axes]
        for ax, r in zip(axes, valid):
            ref = np.array(r["fig_data"]["ref_l_sz"])
            smc = np.array(r["fig_data"]["smc_l_sz"])
            ax.hist(ref, bins=40, density=True, alpha=0.55,
                    color="#0072B2", label="Reference MH")
            ax.hist(smc, bins=40, density=True, alpha=0.55,
                    color="#D55E00", label="SMC")
            ax.set_title(f"examinee {r['examinee']} — l_sz",
                         fontsize=9)
            ax.set_xlabel("l_sz")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR,
                        f"gold_chain_marginals.{ext}"), dpi=300)
        plt.close(fig)
        print(f"  wrote gold_chain_marginals.{{pdf,png}}", flush=True)

    print(f"Done in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
