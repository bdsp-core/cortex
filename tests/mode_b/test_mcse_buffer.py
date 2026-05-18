"""T1.3 regression — Monte-Carlo standard-error buffer on stopping rule.

The Wave-1 fix was to require
    PASS:  pass_probs[k] - Z_BUFFER * mcse[k] >= stop_thresh
    FAIL:  pass_probs[k] + Z_BUFFER * mcse[k] <= 1 - stop_thresh
where mcse[k] = sqrt(p (1-p) / N_eff).  Z_BUFFER = 2.0.

This test pins the constant and re-implements the buffered rule against
hand-computed values for two synthetic states:
  - Borderline: p=0.952, N_eff=100, mcse≈0.0214 → 0.952 - 0.0428 = 0.909 < 0.95.
                Must NOT stop PASS.
  - Solid:      p=0.97,  N_eff=1000, mcse≈0.0054 → 0.97 - 0.0107 = 0.959 > 0.95.
                Must stop PASS.
"""
from __future__ import annotations

import numpy as np

import core_mcmc
import engine_mode_b  # F3.1: Mode-B relocated
import core_mcmc_brute_k


# ---------------------------------------------------------------------------
# pin Z_BUFFER constant
# ---------------------------------------------------------------------------

def test_z_buffer_constant_consistent():
    assert core_mcmc.Z_BUFFER == 2.0
    assert core_mcmc_brute_k.Z_BUFFER == 2.0


# ---------------------------------------------------------------------------
# numerical case: borderline pass_prob with low N_eff must NOT trigger PASS
# ---------------------------------------------------------------------------

def _buffered_decision(p, n_eff, stop_thresh, z=2.0):
    mcse = np.sqrt(max(p * (1.0 - p), 0.0) / max(n_eff, 1.0))
    if p - z * mcse >= stop_thresh:
        return 1
    if p + z * mcse <= 1.0 - stop_thresh:
        return -1
    return 0


def test_borderline_does_not_stop_pass():
    p = 0.952
    n_eff = 100.0
    stop_thresh = 0.95
    mcse = np.sqrt(p * (1.0 - p) / n_eff)
    np.testing.assert_allclose(mcse, 0.02137, atol=1e-4)
    buffered = p - 2.0 * mcse
    np.testing.assert_allclose(buffered, 0.9092, atol=1e-3)
    assert buffered < stop_thresh, "buffered tail prob below stop_thresh — must not pass"
    decision = _buffered_decision(p, n_eff, stop_thresh)
    assert decision == 0, "borderline state must remain undecided"


def test_solid_state_stops_pass():
    p = 0.97
    n_eff = 1000.0
    stop_thresh = 0.95
    mcse = np.sqrt(p * (1.0 - p) / n_eff)
    np.testing.assert_allclose(mcse, 0.005403, atol=1e-5)
    buffered = p - 2.0 * mcse
    np.testing.assert_allclose(buffered, 0.9592, atol=1e-3)
    assert buffered >= stop_thresh, "buffered tail prob clears stop_thresh"
    decision = _buffered_decision(p, n_eff, stop_thresh)
    assert decision == 1, "solid state must PASS"


def test_low_pass_prob_stops_fail():
    """Symmetric FAIL test: p=0.03, N_eff=1000 → buffered upper = 0.0408 ≤ 0.05."""
    p = 0.03
    n_eff = 1000.0
    stop_thresh = 0.95
    mcse = np.sqrt(p * (1.0 - p) / n_eff)
    upper = p + 2.0 * mcse
    assert upper <= 1.0 - stop_thresh, "buffered upper below 1-thresh — must fail"
    decision = _buffered_decision(p, n_eff, stop_thresh)
    assert decision == -1


def test_session_returns_mcse_final():
    """Sanity check: real session returns mcse_final array of the right shape and
    sign."""
    K = 3
    rng = np.random.default_rng(0)
    l_true = rng.normal(0.4, 0.3, K)
    t_true = rng.normal(0.0, 0.5, K)
    true_params = []
    for k in range(K):
        true_params.append(float(t_true[k]))
        true_params.append(float(l_true[k]))
    out = engine_mode_b.run_session_mcmc_certification(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        l_star=np.zeros(K), max_q=40, N=200, seed=0, ess_threshold_frac=0.9,
    )
    mcse = np.asarray(out["mcse_final"])
    assert mcse.shape == (K,)
    assert np.all(mcse >= 0.0)
    assert float(out["z_buffer"]) == 2.0
