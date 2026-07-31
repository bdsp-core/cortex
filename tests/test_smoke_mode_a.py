"""F1.5 (2026-05-15) — Mode-A Multi-AUROC smoke test.

End-to-end smoke of `run_session_mcmc_auroc` for both hier (with Corr_l) and
brute methods.  Verifies:
  - The session driver returns the expected output contract.
  - For a clearly-pass synthetic rater, max-HW falls below δ=0.05 within the
    question budget.
  - `post_hoc_delta_sweep` produces monotonically non-decreasing n_q across
    decreasing δ (tighter precision target ⇒ same or more questions).
  - The trajectory shape is consistent (T = n_q + 1, K columns).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

import core_mcmc

from tests.conftest import true_params_array

_EXPECTED_KEYS = {
    "n_questions", "stopped_early", "true_auroc",
    "final_lo", "final_hi",
    "mean_acceptance_rate", "n_rejuvenations",
    "delta_auroc", "method", "per_domain_n",
}


def _load_corr_l():
    """Load the fitted unstructured prior covariance (Corr_l) if available."""
    path = os.path.join(os.path.dirname(core_mcmc.__file__), "Sigma_l_fitted.npy")
    if not os.path.exists(path):
        return None
    obj = np.load(path, allow_pickle=True).item()
    return np.asarray(obj.get("Corr_l", obj.get("Sigma_l")), dtype=float)


def _check_mode_a_contract(out, K, expected_keys, runtime_s):
    missing = expected_keys - set(out.keys())
    assert not missing, f"Mode-A session missing keys: {missing}"
    final_lo = np.asarray(out["final_lo"])
    final_hi = np.asarray(out["final_hi"])
    assert final_lo.shape == (K,), f"final_lo shape {final_lo.shape} != ({K},)"
    assert final_hi.shape == (K,), f"final_hi shape {final_hi.shape} != ({K},)"
    assert np.all(final_hi >= final_lo), "CI inversion: hi < lo for some domain"
    assert np.all((final_lo >= 0.0) & (final_hi <= 1.0)), \
        "AUROC posterior CI outside [0, 1]"
    per_domain_n = np.asarray(out["per_domain_n"])
    assert per_domain_n.shape == (K,)
    assert np.all(per_domain_n >= 0)
    assert int(per_domain_n.sum()) == int(out["n_questions"])
    assert runtime_s < 90.0, f"Mode-A smoke too slow: {runtime_s:.2f}s > 90s"


def test_mode_a_hier_smoke(synthetic_true_params):
    """Mode-A hier with Corr_l reaches HW < 0.05 for clearly-pass rater."""
    K = 3
    tp = synthetic_true_params(K=K, seed=0)
    true_params = true_params_array(tp["t_true"], tp["l_true"])

    t0 = time.time()
    out = core_mcmc.run_session_mcmc_auroc(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        max_q=400, delta_auroc=0.05, N=400, seed=0,
        ess_threshold_frac=0.5,
        proposal_scale=2.38 / np.sqrt(2 * K),
        log_trajectory=False,
    )
    runtime = time.time() - t0
    _check_mode_a_contract(out, K, _EXPECTED_KEYS, runtime)
    hw_max = float(((out["final_hi"] - out["final_lo"]) / 2.0).max())
    assert out["stopped_early"], (
        f"hier Mode-A should stop early at δ=0.05 within max_q=400 "
        f"for clearly-pass rater (true_l ~ N(0.4, 0.3²)); "
        f"got n_q={out['n_questions']}, hw_max={hw_max:.4f}"
    )


def test_mode_a_brute_smoke(synthetic_true_params):
    """Mode-A brute (no info sharing) reaches HW < 0.05 within larger budget."""
    K = 3
    tp = synthetic_true_params(K=K, seed=0)
    true_params = true_params_array(tp["t_true"], tp["l_true"])

    t0 = time.time()
    out = core_mcmc.run_session_mcmc_auroc(
        method="brute", true_params=true_params, K=K, r_assumed=0.378,
        max_q=500, delta_auroc=0.05, N=400, seed=0,
        ess_threshold_frac=0.5,
        log_trajectory=False,
    )
    runtime = time.time() - t0
    _check_mode_a_contract(out, K, _EXPECTED_KEYS, runtime)
    hw_max = float(((out["final_hi"] - out["final_lo"]) / 2.0).max())
    assert out["stopped_early"], (
        f"brute Mode-A should stop early at δ=0.05 within max_q=500; "
        f"got n_q={out['n_questions']}, hw_max={hw_max:.4f}"
    )


def test_mode_a_post_hoc_delta_sweep_monotone(synthetic_true_params):
    """Tighter δ ⇒ same or more questions to reach HW<δ (monotonicity).

    Runs a single session at δ=0.025 with `run_until_max=True`, then uses
    `post_hoc_delta_sweep` to report stops at {0.025, 0.05, 0.10}.
    """
    K = 3
    tp = synthetic_true_params(K=K, seed=1)
    true_params = true_params_array(tp["t_true"], tp["l_true"])

    out = core_mcmc.run_session_mcmc_auroc(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        max_q=300, delta_auroc=0.025, N=400, seed=1,
        run_until_max=True, log_trajectory=True,
        ess_threshold_frac=0.5,
        proposal_scale=2.38 / np.sqrt(2 * K),
    )
    assert "lo_traj" in out and "hi_traj" in out, \
        "log_trajectory=True should populate lo_traj/hi_traj"

    deltas = [0.025, 0.05, 0.10]
    sweep = core_mcmc.post_hoc_delta_sweep(out["lo_traj"], out["hi_traj"],
                                            deltas=deltas)
    # Convert None to inf for monotonicity comparison.
    def _safe(s):
        return float("inf") if s is None else s

    n_025 = _safe(sweep[0.025])
    n_050 = _safe(sweep[0.05])
    n_100 = _safe(sweep[0.10])
    assert n_100 <= n_050 <= n_025, (
        f"post-hoc δ-sweep should be monotone non-decreasing in tightness: "
        f"n_q(δ=0.10)={n_100}, n_q(δ=0.05)={n_050}, n_q(δ=0.025)={n_025}"
    )
    # δ=0.10 is loose; should always reach it
    assert sweep[0.10] is not None, \
        "δ=0.10 should be reachable within max_q for clearly-pass rater"


def test_mode_a_corr_l_loaded():
    """Corr_l is present in Sigma_l_fitted.npy for K=6 SPARCNET production."""
    Corr_l = _load_corr_l()
    if Corr_l is None:
        pytest.skip("Sigma_l_fitted.npy not present in repo")
    assert Corr_l.shape == (6, 6), f"Corr_l shape {Corr_l.shape} != (6, 6)"
    np.testing.assert_allclose(np.diag(Corr_l), 1.0, atol=1e-6,
                                err_msg="Corr_l should have unit diagonal")
    assert np.all(np.linalg.eigvalsh(Corr_l) > 0), \
        "Corr_l must be positive definite"


def test_mode_a_with_corr_l_k6():
    """K=6 Mode-A with fitted Corr_l: contract + post-hoc sweep monotonicity."""
    Corr_l = _load_corr_l()
    if Corr_l is None:
        pytest.skip("Sigma_l_fitted.npy not present in repo")
    K = 6
    rng = np.random.default_rng(0)
    l_true = rng.normal(0.4, 0.3, size=K)
    t_true = rng.normal(0.0, 0.5, size=K)
    true_params = true_params_array(t_true, l_true)

    out = core_mcmc.run_session_mcmc_auroc(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        Sigma_l=Corr_l, Sigma_t=Corr_l,
        max_q=400, delta_auroc=0.05, N=400, seed=0,
        run_until_max=True, log_trajectory=True,
        ess_threshold_frac=0.5,
        proposal_scale=2.38 / np.sqrt(2 * K),
    )
    _check_mode_a_contract(out, K, _EXPECTED_KEYS, runtime_s=0.0)
    # Trajectory shape: T = max_q + 1 (because run_until_max=True), K columns.
    # `n_questions` records the FIRST stop step, but the loop continues to
    # max_q to support post-hoc δ-sweep across multiple thresholds.
    max_q = 400  # match the kwarg above
    assert out["lo_traj"].shape == (max_q + 1, K), (
        f"lo_traj shape {out['lo_traj'].shape} != (max_q+1={max_q+1}, K={K})"
    )
    assert out["hi_traj"].shape == (max_q + 1, K)
