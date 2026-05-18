"""T3.3 — Posterior credible-interval coverage skeleton.

For 10 synthetic raters at known ℓ_true, run the cert session and check that
the empirical coverage of 90% / 95% credible intervals on ℓ_k is approximately
nominal.  Skeleton-grade: with N=10 raters the SE on coverage is ~ √(0.9 * 0.1
/ 10) = 0.095, so we only assert a very loose bound (>0.7).

Marked @pytest.mark.slow.  Production-grade coverage requires N ≥ 200 raters.
"""
from __future__ import annotations

import numpy as np
import pytest

import core_mcmc


pytestmark = pytest.mark.slow


def _weighted_quantile(samples, weights, q):
    """Weighted quantile of a 1-D sample / weight pair."""
    order = np.argsort(samples)
    s = samples[order]
    w = weights[order]
    c = np.cumsum(w)
    c = c / c[-1]
    return float(np.interp(q, c, s))


def _run_and_check_coverage(ell_true, t_true, K, N=200, max_q=100, seed=0):
    """Run cert; return whether the 90% and 95% CI on each ℓ_k cover the truth."""
    true_params = []
    for k in range(K):
        true_params.append(float(t_true[k])); true_params.append(float(ell_true[k]))
    rng = np.random.default_rng(seed)

    state = core_mcmc.make_state_hier(N, K, r_assumed=0.378, rng=rng)
    # Simulate ~max_q observations, simple uniform-domain choice (faster than
    # EV item selection and adequate for credibility coverage).
    for q in range(max_q):
        k = q % K
        s = float(rng.uniform(-2.0, 2.0))
        y = core_mcmc.simulate_response(s, t_true[k], ell_true[k], rng)
        core_mcmc.update(state, k, s, y)
        if core_mcmc.ess(state["w"]) < 0.5 * N:
            core_mcmc.resample_and_rejuvenate(state, rng,
                                               n_mh_steps=10, proposal_scale=0.5)

    cov_90 = np.zeros(K, dtype=bool)
    cov_95 = np.zeros(K, dtype=bool)
    for k in range(K):
        samples = state["l"][:, k]
        weights = state["w"]
        lo90 = _weighted_quantile(samples, weights, 0.05)
        hi90 = _weighted_quantile(samples, weights, 0.95)
        lo95 = _weighted_quantile(samples, weights, 0.025)
        hi95 = _weighted_quantile(samples, weights, 0.975)
        cov_90[k] = (lo90 <= ell_true[k] <= hi90)
        cov_95[k] = (lo95 <= ell_true[k] <= hi95)
    return cov_90, cov_95


def test_posterior_coverage_skeleton():
    K = 3
    n_raters = 10
    rng_master = np.random.default_rng(2026)
    cover90 = []
    cover95 = []
    for i in range(n_raters):
        rng_local = np.random.default_rng(rng_master.integers(0, 2**31))
        # draw raters from a moderate-scale prior
        ell_true = rng_local.normal(0.2, 0.3, K)
        t_true = rng_local.normal(0.0, 0.5, K)
        c90, c95 = _run_and_check_coverage(
            ell_true, t_true, K, N=200, max_q=80,
            seed=int(rng_local.integers(0, 2**31)),
        )
        cover90.extend(c90.tolist())
        cover95.extend(c95.tolist())
    cov90 = np.mean(cover90)
    cov95 = np.mean(cover95)
    # Loose bound for skeleton with ~ 30 (rater × domain) observations
    assert cov90 > 0.7, f"90% CI empirical coverage {cov90:.2f} < 0.7"
    assert cov95 > 0.7, f"95% CI empirical coverage {cov95:.2f} < 0.7"
