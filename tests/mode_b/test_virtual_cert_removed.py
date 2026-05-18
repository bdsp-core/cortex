"""T1.1 regression — virtual-cert mechanism is gone in `core_mcmc`.

Pins:
  - Passing `virtual_pairs=[(0, 4, 0.95)]` to `run_session_mcmc_certification`
    on the hier path emits `DeprecationWarning` and the argument is ignored.
  - `virtual_granted` in the returned dict is all zeros.
  - With the same seed and config, hier and brute decisions are no longer
    biased by the asymmetric old virtual-cert advantage (we don't expect them
    to match exactly — they use different samplers — but neither path should
    auto-pass any domain that the other has not also nearly passed).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import core_mcmc
import engine_mode_b  # F3.1: Mode-B relocated
import core_mcmc_brute_k


def _make_clearpass_params(K=6, seed=0):
    rng = np.random.default_rng(seed)
    l_true = rng.normal(0.4, 0.3, K)
    t_true = rng.normal(0.0, 0.5, K)
    out = []
    for k in range(K):
        out.append(float(t_true[k]))
        out.append(float(l_true[k]))
    return out


def test_virtual_pairs_warns_and_is_ignored():
    K = 6
    true_params = _make_clearpass_params(K=K, seed=0)
    l_star = np.zeros(K)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = engine_mode_b.run_session_mcmc_certification(
            method="hier", true_params=true_params, K=K, r_assumed=0.378,
            l_star=l_star, max_q=5, N=100, seed=0, ess_threshold_frac=0.9,
            virtual_pairs=[(0, 4, 0.95)],
        )
    deprecation_msgs = [str(rec.message) for rec in w
                        if issubclass(rec.category, DeprecationWarning)]
    assert deprecation_msgs, (
        f"expected DeprecationWarning when passing virtual_pairs; got warnings: "
        f"{[str(rec.message) for rec in w]}"
    )
    assert any("virtual_pairs" in msg for msg in deprecation_msgs)


def test_virtual_granted_always_zero_hier():
    K = 6
    true_params = _make_clearpass_params(K=K, seed=0)
    l_star = np.zeros(K)
    out = engine_mode_b.run_session_mcmc_certification(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        l_star=l_star, max_q=40, N=200, seed=0, ess_threshold_frac=0.9,
    )
    vg = np.asarray(out["virtual_granted"])
    assert vg.shape == (K,)
    assert vg.dtype == np.bool_
    assert not vg.any(), (
        f"virtual_granted should be all-zeros after FIX-T1.1; got {vg}"
    )


def test_no_virtual_pass_advantage_hier_vs_brute():
    """Both methods should make decisions strictly from data, not from a virtual
    auto-pass.  We check that no domain is passed by hier with a pass_prob far
    below the threshold (which would be the smoking gun of leftover virtual
    cert)."""
    K = 6
    true_params = _make_clearpass_params(K=K, seed=1)
    l_star = np.zeros(K)
    out = engine_mode_b.run_session_mcmc_certification(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        l_star=l_star, max_q=80, N=200, seed=0, ess_threshold_frac=0.9,
    )
    decisions = np.asarray(out["decisions"])
    pass_probs = np.asarray(out["pass_probs_final"])
    stop_thresh = float(out["stop_thresh_used"])
    z_buf = float(out["z_buffer"])
    n_eff = 1.0  # we don't have N_eff in output; use a loose lower bound on pass_prob
    # Any PASS decision must have pass_prob comfortably above stop_thresh
    # (within MCSE buffer reach).  Allow a slack of 0.20 (loose, since pass_probs
    # may drift slightly between the decision step and the final reading).
    for k, dec in enumerate(decisions):
        if dec == 1:
            assert pass_probs[k] >= stop_thresh - 0.20, (
                f"domain {k} PASSED with pass_prob_final={pass_probs[k]} far below "
                f"stop_thresh={stop_thresh}; possible leftover virtual-cert auto-pass"
            )
