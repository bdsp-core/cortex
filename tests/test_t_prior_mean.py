"""Drift-guard for the Step-4 engine `t_prior_mean` hook (sandbox EB enabler).

t_prior_mean mirrors l_prior_mean: it shifts the θ (t) block prior mean. Default
None ⇒ BYTE-IDENTICAL to the shipped zero-mean prior. When set, it must shift the
sampled cloud + the prior density consistently (so MH rejuvenation stays correct).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _p in (os.path.join(_REPO, "engine"), os.path.join(_REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core_mcmc as cm  # noqa: E402


def _corr(K=3):
    A = np.eye(K) + 0.1
    return A / np.sqrt(np.outer(np.diag(A), np.diag(A)))


def test_t_prior_mean_none_is_byte_identical():
    """make_state_hier with t_prior_mean=None must reproduce the no-kwarg state
    bit-for-bit (same seed)."""
    K = 3
    Sl = _corr(K)
    s1 = cm.make_state_hier(500, K, 0.378, np.random.default_rng(0),
                            Sigma_l=Sl, Sigma_t=Sl)
    s2 = cm.make_state_hier(500, K, 0.378, np.random.default_rng(0),
                            Sigma_l=Sl, Sigma_t=Sl, t_prior_mean=None)
    np.testing.assert_array_equal(s1["t"], s2["t"])
    np.testing.assert_array_equal(s1["l"], s2["l"])
    np.testing.assert_array_equal(s1["log_prior"], s2["log_prior"])


def test_t_prior_mean_shifts_cloud_and_density():
    """A non-zero t_prior_mean shifts the sampled t cloud by ~the mean, and the
    log-prior density centers there (peak at t=mean for the t block)."""
    K = 3
    Sl = _corr(K)
    mu = np.array([0.0, 1.7, -0.5])
    s = cm.make_state_hier(20000, K, 0.378, np.random.default_rng(1),
                           Sigma_l=Sl, Sigma_t=Sl, t_prior_mean=mu)
    # sampled t-block mean ≈ mu
    np.testing.assert_allclose(s["t"].mean(axis=0), mu, atol=0.05)
    # l-block stays zero-mean
    np.testing.assert_allclose(s["l"].mean(axis=0), np.zeros(K), atol=0.05)
    # density: log_prior_hier centered at mu — a particle AT mu out-scores one far away
    at_mu = np.tile(mu, (1, 1))
    far = np.tile(mu + 2.0, (1, 1))
    zero_l = np.zeros((1, K))
    lp_mu = cm.log_prior_hier(at_mu, zero_l, 0.378, K, Sigma_l=Sl, Sigma_t=Sl,
                              t_prior_mean=mu)
    lp_far = cm.log_prior_hier(far, zero_l, 0.378, K, Sigma_l=Sl, Sigma_t=Sl,
                               t_prior_mean=mu)
    assert lp_mu[0] > lp_far[0]


def test_t_prior_mean_stored_for_rejuvenation():
    """The mean is stored on the state so MH rejuvenation evaluates the shifted
    prior (else the resample/MCMC would target the wrong density)."""
    K = 2
    Sl = _corr(K)
    mu = np.array([0.3, -0.4])
    s = cm.make_state_hier(100, K, 0.378, np.random.default_rng(2),
                           Sigma_l=Sl, Sigma_t=Sl, t_prior_mean=mu)
    assert "_t_prior_mean" in s
    np.testing.assert_array_equal(s["_t_prior_mean"], mu)
    # _log_prior_of uses the stored mean (consistent with make_state_hier's log_prior)
    lp = cm._log_prior_of(s, s["t"], s["l"])
    np.testing.assert_allclose(lp, s["log_prior"], rtol=1e-9)
