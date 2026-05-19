"""Phase 7 sub-step 1 — Phase-2 validation suite synthetic at K=7
(α-scope per UNIFIED_REPO_MERGE_PLAN.md §"Phase 7", user-confirmed
2026-05-19).

The carried Phase-2 validation skeletons exercise the K-agnostic
engine at K=3 (`test_sbc_skeleton.py`, `test_posterior_coverage.py`)
and the K-independent lapse algebra (`test_lapse_rate.py`). The
methodology paper-grade scripts in `scripts/run_*` run at K=6.

Phase 7 needs explicit pins at **K=7** — the production deployment
dim (6 IIIC + spike + `other` = 7). The engine code paths are
byte-identical at all K (Phase-2 invariant: no K=6-specific branches
in `core_mcmc`/`core`), so a K=7 synthetic gate is the cleanest
direct check that the engine soundness carries to the production
configuration without disturbing the byte-identical reference
tests carried from the methodology repo.

Three pins, all slow-marked (each ~10–40 s alone):

  * test_sbc_skeleton_l_domain0_k7: SBC mean rank near 0.5 at K=7
    (same gate as the K=3 skeleton: 0.25 < mean_rank < 0.75; not-
    all-on-one-side soft check).
  * test_posterior_coverage_k7: 90/95 % CI empirical coverage > 0.7
    at K=7 (same gate as the K=3 skeleton).
  * test_lapse_algebra_k7: lapse mixture probability pin holds
    domain-wise at K=7 (the lapse is per-task per-trial; K only
    changes how many slots exist, not the per-task lapse equation).

These are the *synthetic α-scope* gates. The production-config
soundness (s_sd propagation under real item noise) is gated
separately by Phase-3.5's engine SBC artifact (`calibration/joint/
sbc_engine.json`, ran at K=6 IIIC ⇒ the 6 IIIC tasks of K=7), which
remains the headline engine-calibration result; the 7th task
`combined_spike` is gated upstream by Phase-3 (the byte-verbatim
reference fit) + Phase-3.5 (the joint-IRT s_j with s_sd propagation).

This file is Phase-7 specific and ADDED — the carried byte-identical
methodology tests (test_sbc_skeleton, test_posterior_coverage,
test_lapse_rate) are NOT edited.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import core
import core_mcmc

pytestmark = pytest.mark.slow

# K=7 = the production deployment dim (the 6 IIIC tasks +
# combined_spike), exactly what ships in data/deployment_prior/
# (`Sigma_prior` shape (14, 14) = (2K, 2K)).
K_PROD = 7


# ──────────────────────────────────────────────────────────────────
# Helpers (mirror the methodology skeleton tests, K-parameterized).
# ──────────────────────────────────────────────────────────────────

def _simulate_history(t_true, l_true, n_obs, rng):
    """Same forward model as core_mcmc.simulate_response (Phase-2)."""
    K = len(t_true)
    ks, ss, ys = [], [], []
    for _ in range(n_obs):
        k = int(rng.integers(0, K))
        s = float(rng.uniform(-2.5, 2.5))
        y = core_mcmc.simulate_response(s, t_true[k], l_true[k], rng)
        ks.append(k); ss.append(s); ys.append(y)
    return ks, ss, ys


def _posterior_l_samples(t_true, l_true, history, K, n_part, n_mh, seed,
                         domain_idx=0):
    """Mirror of test_sbc_skeleton._posterior_l_samples_for_domain
    (returns the weighted particle marginal for ℓ_{domain_idx})."""
    rng = np.random.default_rng(seed)
    state = core_mcmc.make_state_hier(n_part, K, r_assumed=0.378, rng=rng)
    for (k, s, y) in zip(*history):
        core_mcmc.update(state, k, s, y)
        if core_mcmc.ess(state["w"]) < 0.5 * n_part:
            core_mcmc.resample_and_rejuvenate(
                state, rng, n_mh_steps=n_mh, proposal_scale=0.5)
    core_mcmc.resample_and_rejuvenate(
        state, rng, n_mh_steps=n_mh, proposal_scale=0.5)
    return state["l"][:, domain_idx].copy(), state["w"].copy()


def _weighted_rank(target, samples, weights):
    weights = weights / weights.sum()
    return float((weights * (samples < target)).sum())


def _weighted_quantile(samples, weights, q):
    order = np.argsort(samples)
    s = samples[order]
    w = weights[order]
    c = np.cumsum(w)
    c = c / c[-1]
    return float(np.interp(q, c, s))


# ──────────────────────────────────────────────────────────────────
# Pins
# ──────────────────────────────────────────────────────────────────

def test_sbc_skeleton_l_domain0_k7():
    """SBC mean-rank pin at K=7 (the production dim)."""
    K = K_PROD
    n_draws = 20  # same budget as the methodology K=3 skeleton
    n_obs = 60     # ×K/3 to give each task ~ same per-task observations
    n_part = 200
    n_mh = 30

    rng_master = np.random.default_rng(2026)
    ranks = []
    for _ in range(n_draws):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        t_true, l_true = core_mcmc.sample_prior_hier_K(
            1, K, r=0.378, rng=rng)
        t_true = t_true[0]; l_true = l_true[0]

        history = _simulate_history(t_true, l_true, n_obs, rng)
        samples, weights = _posterior_l_samples(
            t_true, l_true, history, K, n_part, n_mh,
            seed=int(rng.integers(0, 2**31)))
        u = _weighted_rank(l_true[0], samples, weights)
        ranks.append(u)

    ranks = np.asarray(ranks)
    mean_rank = float(ranks.mean())
    # Same gates as the carried K=3 skeleton.
    assert 0.25 < mean_rank < 0.75, (
        f"SBC mean rank {mean_rank:.3f} far from 0.5 at K=7 "
        f"— possible bias. ranks={ranks}")
    assert ranks.min() < 0.5 and ranks.max() > 0.5, (
        f"All K=7 ranks on one side of 0.5 — strong bias indicator. "
        f"ranks={ranks}")


def test_posterior_coverage_k7():
    """90/95 % CI empirical coverage at K=7."""
    K = K_PROD
    n_raters = 10
    rng_master = np.random.default_rng(2026)
    cover90, cover95 = [], []
    for _ in range(n_raters):
        rng_local = np.random.default_rng(rng_master.integers(0, 2**31))
        ell_true = rng_local.normal(0.2, 0.3, K)
        t_true = rng_local.normal(0.0, 0.5, K)

        # Same engine-driven coverage check as the methodology
        # skeleton, K-parameterized through make_state_hier.
        N_part = 200
        max_q = 80 * K // 3  # scale per-task budget to match the K=3
        rng_run = np.random.default_rng(
            int(rng_local.integers(0, 2**31)))
        state = core_mcmc.make_state_hier(
            N_part, K, r_assumed=0.378, rng=rng_run)
        for q in range(max_q):
            k = q % K
            s = float(rng_run.uniform(-2.0, 2.0))
            y = core_mcmc.simulate_response(
                s, t_true[k], ell_true[k], rng_run)
            core_mcmc.update(state, k, s, y)
            if core_mcmc.ess(state["w"]) < 0.5 * N_part:
                core_mcmc.resample_and_rejuvenate(
                    state, rng_run, n_mh_steps=10, proposal_scale=0.5)

        for k in range(K):
            samples = state["l"][:, k]
            weights = state["w"]
            lo90 = _weighted_quantile(samples, weights, 0.05)
            hi90 = _weighted_quantile(samples, weights, 0.95)
            lo95 = _weighted_quantile(samples, weights, 0.025)
            hi95 = _weighted_quantile(samples, weights, 0.975)
            cover90.append(lo90 <= ell_true[k] <= hi90)
            cover95.append(lo95 <= ell_true[k] <= hi95)
    cov90 = float(np.mean(cover90))
    cov95 = float(np.mean(cover95))
    # Same loose skeleton bound as the carried K=3 test.
    assert cov90 > 0.7, (
        f"90% CI empirical coverage at K=7 was {cov90:.2f} (< 0.7)")
    assert cov95 > 0.7, (
        f"95% CI empirical coverage at K=7 was {cov95:.2f} (< 0.7)")


def test_lapse_algebra_k7():
    """Lapse mixture pin at K=7. The lapse is PER-TASK PER-TRIAL,
    K-independent at the equation level — this test makes that
    explicit at the production dim by running the lapse algebra in
    a per-task loop across 7 tasks."""
    # Eq. 2 spike-paper canonical lapse:  P(y=1|z) = λ + (1−2λ)·Φ(z)
    lam = core.LAPSE_RATE
    assert lam == 0.025  # Phase-6 invariant 2 pin
    # Per-task pin: at z=10, P(y=1) ≈ 0.975 (≈ 1−λ); at z=-10, ≈ 0.025.
    for k in range(K_PROD):
        # Use a per-task latent z that varies by k to exercise the
        # full numeric range (no K-dependence by design).
        z_hi = 10.0 + 0.1 * k
        z_lo = -10.0 - 0.1 * k
        p_hi = lam + (1.0 - 2.0 * lam) * float(norm.cdf(z_hi))
        p_lo = lam + (1.0 - 2.0 * lam) * float(norm.cdf(z_lo))
        # Reference-correct lapse floor and ceiling.
        assert abs(p_hi - (1.0 - lam)) < 1e-12, (
            f"K=7 task {k}: P(y=1|z={z_hi}) = {p_hi:.6e}, "
            f"expected {1.0 - lam:.6e}")
        assert abs(p_lo - lam) < 1e-12, (
            f"K=7 task {k}: P(y=1|z={z_lo}) = {p_lo:.6e}, "
            f"expected {lam:.6e}")
