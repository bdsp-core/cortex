"""T3.2 — Simulation-Based Calibration (SBC) skeleton.

Talts, Betancourt, Simpson, Vehtari, Gelman 2018:
   "Validating Bayesian Inference Algorithms with Simulation-Based Calibration".

For an algorithmically-correct posterior sampler P(theta | y), if you draw
theta_0 ~ prior, simulate y ~ likelihood(. | theta_0), and then sample
{theta_1, ..., theta_L} ~ P(. | y), the rank statistic of theta_0 within
{theta_0, theta_1, ..., theta_L} is uniformly distributed on {0, 1, ..., L}.

This is a SKELETON: N=20 prior draws is far too few for a paper-grade SBC
(Talts et al. recommend N ≥ 200, with bin counts >= ~ N/(L/(L+1)*B) per bin).
We assert only that the mean rank is not catastrophically biased.

Marked @pytest.mark.slow — keep the per-rep budget tight so the whole run
still fits well under 2 min.
"""
from __future__ import annotations

import numpy as np
import pytest

import core_mcmc


pytestmark = pytest.mark.slow


def _simulate_history(t_true, l_true, n_obs, rng):
    """Simulate `n_obs` (k, s, y) tuples with random domain & uniform signals.

    Uses the same lapse-mixture forward model as `core_mcmc.simulate_response`.
    Returns lists (ks, ss, ys).
    """
    K = len(t_true)
    ks, ss, ys = [], [], []
    for _ in range(n_obs):
        k = int(rng.integers(0, K))
        s = float(rng.uniform(-2.5, 2.5))
        y = core_mcmc.simulate_response(s, t_true[k], l_true[k], rng)
        ks.append(k); ss.append(s); ys.append(y)
    return ks, ss, ys


def _posterior_l_samples_for_domain(t_true, l_true, history, K, n_part, n_mh, seed):
    """Run a quick particle approximation of P(l_k | history) for k=0.

    Returns the weighted particle marginal as (l_samples, weights).
    """
    rng = np.random.default_rng(seed)
    state = core_mcmc.make_state_hier(n_part, K, r_assumed=0.378, rng=rng)
    for (k, s, y) in zip(*history):
        core_mcmc.update(state, k, s, y)
        if core_mcmc.ess(state["w"]) < 0.5 * n_part:
            core_mcmc.resample_and_rejuvenate(state, rng,
                                               n_mh_steps=n_mh, proposal_scale=0.5)
    # final rejuvenation pass for mixing
    core_mcmc.resample_and_rejuvenate(state, rng, n_mh_steps=n_mh, proposal_scale=0.5)
    return state["l"][:, 0].copy(), state["w"].copy()


def _weighted_rank(target, samples, weights):
    """Weighted rank: cumulative weight of samples < target.

    Returns a value in [0, 1] (a CDF-like quantile).  Under correct calibration
    this should be uniform on [0, 1].
    """
    weights = weights / weights.sum()
    return float((weights * (samples < target)).sum())


def test_sbc_skeleton_l_domain0():
    """Skeleton SBC for the marginal posterior of l[0]."""
    K = 3
    n_draws = 20
    n_obs = 50
    n_part = 200
    n_mh = 30

    rng_master = np.random.default_rng(2026)
    ranks = []
    for i in range(n_draws):
        # draw from the (compound-symmetry r=0.378) prior used for inference
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        t_true, l_true = core_mcmc.sample_prior_hier_K(1, K, r=0.378, rng=rng)
        t_true = t_true[0]; l_true = l_true[0]

        ks, ss, ys = _simulate_history(t_true, l_true, n_obs, rng)
        samples, weights = _posterior_l_samples_for_domain(
            t_true, l_true, (ks, ss, ys), K, n_part, n_mh,
            seed=int(rng.integers(0, 2**31)),
        )
        u = _weighted_rank(l_true[0], samples, weights)
        ranks.append(u)

    ranks = np.asarray(ranks)
    mean_rank = float(ranks.mean())

    # Expected mean under uniform: 0.5.  Allow ±0.25 (very loose for N=20).
    assert 0.25 < mean_rank < 0.75, (
        f"SBC mean rank {mean_rank:.3f} far from 0.5 — possible bias.  "
        f"Sample ranks: {ranks}"
    )

    # Soft check: not all ranks at 0 or all at 1
    assert ranks.min() < 0.5 and ranks.max() > 0.5, (
        f"All ranks on one side of 0.5 — strong bias indicator.  ranks={ranks}"
    )
