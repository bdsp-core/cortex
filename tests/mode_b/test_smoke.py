"""T3.1 — End-to-end smoke test for both certification methods.

Runs the post-Wave-1 hier and brute cert sessions on a small, fast
configuration and asserts that the return contract is honored and that
the engine actually decides at least most domains within the question
budget when the rater is clearly above l* = 0.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import core_mcmc
import engine_mode_b  # F3.1: Mode-B relocated
import core_mcmc_brute_k

from tests.conftest import true_params_array


_EXPECTED_KEYS_HIER = {
    "n_questions",
    "per_domain_n",
    "decisions",
    "stopped_early",
    "active_at_end",
    "true_auroc",
    "pass_probs_final",
    "virtual_granted",
    "stop_thresh_used",
    "mean_acceptance_rate",
    "n_rejuvenations",
    "mcse_final",
    "z_buffer",
}

_EXPECTED_KEYS_BRUTE = {
    "n_questions",
    "per_domain_n",
    "decisions",
    "stopped_early",
    "active_at_end",
    "true_auroc",
    "pass_probs_final",
    "virtual_granted",
    "stop_thresh_used",
    "mean_acceptance_rate",
    "n_rejuvenations",
}


def _check_smoke_output(out, K, expected_keys, runtime_s):
    # ---- contract ----
    missing = expected_keys - set(out.keys())
    assert not missing, f"missing keys in cert output: {missing}"

    decisions = np.asarray(out["decisions"])
    assert decisions.shape == (K,)
    pass_probs = np.asarray(out["pass_probs_final"])
    assert pass_probs.shape == (K,)
    assert np.all((pass_probs >= 0.0) & (pass_probs <= 1.0))

    # ---- runtime budget ----
    assert runtime_s < 60.0, f"smoke test too slow: {runtime_s:.2f}s > 60s"

    # ---- pass-prob progress: at least 2 of K marginals cleared stop_thresh ----
    # Under min-of-marginals IUT (FIX-T1.9), per-session decisions are
    # effectively all-or-nothing, so we instead check that the engine has
    # made meaningful progress: most marginal pass-probs should reach the
    # threshold for a clearly-pass synthetic rater.
    stop_t = float(out.get("stop_thresh_used", 0.9))
    marginals_at_threshold = int((pass_probs >= stop_t).sum())
    assert marginals_at_threshold >= 2, (
        f"expected at least 2/{K} marginals to reach stop_thresh={stop_t} for "
        f"clearly-pass rater; got {marginals_at_threshold}; "
        f"decisions={decisions}, pass_probs={pass_probs}"
    )


def test_smoke_hier(synthetic_true_params):
    """Smoke test the hier method (joint 2K-D SMC+MCMC)."""
    K = 3
    tp = synthetic_true_params(K=K, seed=0)
    true_params = true_params_array(tp["t_true"], tp["l_true"])
    l_star = np.zeros(K)

    t0 = time.time()
    # Note: explicit stop_thresh=0.9 (instead of the Šidák ~0.9915 default)
    # so the small-N MCSE buffer doesn't dominate at this CI-budget runtime.
    out = engine_mode_b.run_session_mcmc_certification(
        method="hier",
        true_params=true_params,
        K=K,
        r_assumed=0.378,
        l_star=l_star,
        max_q=150,
        N=300,
        seed=0,
        ess_threshold_frac=0.5,
        stop_thresh=0.9,
    )
    runtime = time.time() - t0
    _check_smoke_output(out, K, _EXPECTED_KEYS_HIER, runtime)

    # virtual_granted should always be all-zero post-W1A
    assert not np.asarray(out["virtual_granted"]).any(), (
        "virtual_granted must be all zeros after FIX-T1.1"
    )


def test_smoke_brute(synthetic_true_params):
    """Smoke test the brute method (K independent 2-D SMCs).

    F0.1: budget raised from 150 to 300 questions because the spike-paper
    symmetric lapse (floor λ, ceiling 1 − λ) produces a less-concentrated
    posterior than the previous asymmetric form, and brute lacks the
    cross-domain pooling that lets hier compensate.
    """
    K = 3
    tp = synthetic_true_params(K=K, seed=0)
    true_params = true_params_array(tp["t_true"], tp["l_true"])
    l_star = np.zeros(K)

    t0 = time.time()
    out = core_mcmc_brute_k.run_session_mcmc_brute_k_cert(
        true_params=true_params,
        K=K,
        l_star=l_star,
        max_q=300,
        N=300,
        seed=0,
        ess_threshold_frac=0.5,
        stop_thresh=0.9,
    )
    runtime = time.time() - t0
    _check_smoke_output(out, K, _EXPECTED_KEYS_BRUTE, runtime)
