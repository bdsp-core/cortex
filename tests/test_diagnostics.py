"""F2.5 (2026-05-15) — tests for diagnostics.py.

Pins:
  - weighted_ess: unit weights → N; degenerate weights → 1.
  - lag1_autocorrelation: identical before/after → 1.0; iid permutation → ~0.
  - integrated_autocorr_time: AR(1) with rho → analytic tau = (1+rho)/(1-rho).
  - split_rhat: well-mixed gaussian chain → ~1.0; pathological piecewise → > 1.5.
  - rejuvenation_event_diag: returns the expected schema fields and types.
  - The resample_and_rejuvenate diag_callback fires once per call.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import core_mcmc
from diagnostics import (
    weighted_ess,
    weighted_ess_per_parameter,
    lag1_autocorrelation,
    integrated_autocorr_time,
    split_rhat,
    rejuvenation_event_diag,
    write_diag_record,
)


# ── weighted_ess ──────────────────────────────────────────────────────

def test_weighted_ess_uniform_weights():
    w = np.full(100, 1.0 / 100)
    np.testing.assert_allclose(weighted_ess(w), 100.0)


def test_weighted_ess_degenerate():
    w = np.zeros(100)
    w[0] = 1.0
    np.testing.assert_allclose(weighted_ess(w), 1.0)


def test_weighted_ess_per_parameter_shapes():
    rng = np.random.default_rng(0)
    N, K = 200, 6
    t = rng.standard_normal((N, K))
    l = rng.standard_normal((N, K))
    w = rng.dirichlet(np.ones(N))
    out = weighted_ess_per_parameter(t, l, w)
    assert out["t"].shape == (K,)
    assert out["l"].shape == (K,)
    # All entries identical under multinomial reweighting
    np.testing.assert_allclose(out["t"], out["t"][0])


# ── lag1_autocorrelation ──────────────────────────────────────────────

def test_lag1_identical_arrays_is_one():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 6))
    out = lag1_autocorrelation(a, a)
    np.testing.assert_allclose(out, 1.0, atol=1e-12)


def test_lag1_independent_arrays_is_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1000, 6))
    b = rng.standard_normal((1000, 6))
    out = lag1_autocorrelation(a, b)
    # Each correlation should be near zero (within ~2/sqrt(N) ≈ 0.06)
    assert np.all(np.abs(out) < 0.10), f"out={out}"


def test_lag1_shape_mismatch_raises():
    a = np.zeros((10, 3))
    b = np.zeros((10, 4))
    with pytest.raises(ValueError, match="shape mismatch"):
        lag1_autocorrelation(a, b)


# ── integrated_autocorr_time ──────────────────────────────────────────

def test_tau_int_iid_chain_close_to_one():
    """IID chain should give τ_int ≈ 1."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    tau = integrated_autocorr_time(x, c=5.0)
    assert 0.5 < tau < 2.0, f"iid τ_int should be ~1, got {tau}"


def test_tau_int_ar1_chain_close_to_analytic():
    """AR(1) chain with rho=0.5: τ_int_analytic = (1+rho)/(1-rho) = 3."""
    rng = np.random.default_rng(1)
    n = 16384
    rho = 0.5
    x = np.zeros(n)
    eps = rng.standard_normal(n) * np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    tau = integrated_autocorr_time(x, c=5.0)
    analytic = (1 + rho) / (1 - rho)
    # Sokal estimator has finite-sample noise; ±25% tolerance is the spec.
    assert 0.75 * analytic < tau < 1.25 * analytic, (
        f"AR(1) rho={rho}: expected τ_int ≈ {analytic}, got {tau}"
    )


# ── split_rhat ────────────────────────────────────────────────────────

def test_split_rhat_well_mixed_close_to_one():
    rng = np.random.default_rng(0)
    chain = rng.standard_normal(4000)
    r = split_rhat(chain)
    assert 0.98 < r < 1.05, f"well-mixed chain should have R̂ ≈ 1, got {r}"


def test_split_rhat_pathological_chain_above_threshold():
    """Concatenate 4 distinct-mean sub-chains → high R̂."""
    rng = np.random.default_rng(0)
    parts = [
        rng.standard_normal(1000) + offset for offset in (-3.0, -1.0, 1.0, 3.0)
    ]
    chain = np.concatenate(parts)
    r = split_rhat(chain)
    assert r > 1.5, (
        f"pathological multi-mode chain should fail R̂ < 1.5; got {r}"
    )


def test_split_rhat_short_chain_raises():
    with pytest.raises(ValueError, match="too short"):
        split_rhat(np.array([1.0, 2.0]))


# ── rejuvenation_event_diag ───────────────────────────────────────────

def test_rejuvenation_event_diag_schema():
    rng = np.random.default_rng(0)
    N, K = 200, 6
    t_b = rng.standard_normal((N, K))
    t_a = t_b + 0.1 * rng.standard_normal((N, K))
    l_b = rng.standard_normal((N, K))
    l_a = l_b + 0.1 * rng.standard_normal((N, K))
    w = np.full(N, 1.0 / N)
    rec = rejuvenation_event_diag(
        event_id=3, q_index=42, n_mh_steps=15, accept_rate=0.27,
        t_before=t_b, t_after=t_a, l_before=l_b, l_after=l_a,
        weights_after=w,
        domain_names=["sz", "lpd", "gpd", "lrda", "grda", "iic"],
    )
    # Schema check
    expected = {
        "event_id", "q_index", "n_mh_steps", "accept_rate", "domains",
        "lag1_autocorr_t", "lag1_autocorr_l", "ess_t", "ess_l",
    }
    assert expected.issubset(rec.keys())
    assert rec["event_id"] == 3
    assert rec["q_index"] == 42
    assert rec["domains"] == ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
    assert len(rec["lag1_autocorr_t"]) == K
    assert len(rec["ess_t"]) == K
    # Lag-1 should be high (0.1 noise perturbation on identical clouds → ~1.0)
    assert all(0.9 < v < 1.0 for v in rec["lag1_autocorr_t"]), \
        f"lag1_t should be near 1 with small perturbation: {rec['lag1_autocorr_t']}"


def test_write_diag_record_jsonl_round_trip():
    """Verify the record can be re-loaded from JSONL without precision loss."""
    rng = np.random.default_rng(0)
    rec = rejuvenation_event_diag(
        event_id=0, q_index=10, n_mh_steps=15, accept_rate=0.30,
        t_before=rng.standard_normal((100, 3)),
        t_after=rng.standard_normal((100, 3)),
        l_before=rng.standard_normal((100, 3)),
        l_after=rng.standard_normal((100, 3)),
        weights_after=np.full(100, 1.0 / 100),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "diag.jsonl")
        write_diag_record(path, rec)
        with open(path) as f:
            line = f.readline()
        loaded = json.loads(line)
        assert loaded["event_id"] == 0
        assert loaded["q_index"] == 10
        assert loaded["accept_rate"] == 0.30


# ── resample_and_rejuvenate diag_callback integration ─────────────────

def test_resample_and_rejuvenate_diag_callback_fires():
    """The callback fires once per call and the captured record is well-formed."""
    rng = np.random.default_rng(0)
    K = 3
    state = core_mcmc.make_state_hier(N=200, K=K, r_assumed=0.378, rng=rng)
    # History format is (k, s, y) — domain index, signal, response
    state["history"] = [(0, 0.5, 1), (1, -0.3, 0)]
    from core_mcmc import _log_lik_history
    state["log_lik"] = _log_lik_history(state["t"], state["l"], state["history"])

    captured = []
    def cb(rec):
        captured.append(rec)

    core_mcmc.resample_and_rejuvenate(
        state, rng, n_mh_steps=3, proposal_scale=0.5,
        diag_callback=cb, q_index=2,
    )

    assert len(captured) == 1, f"expected exactly 1 diag record; got {len(captured)}"
    rec = captured[0]
    assert rec["event_id"] == 0
    assert rec["q_index"] == 2
    assert rec["n_mh_steps"] == 3
    assert 0.0 <= rec["accept_rate"] <= 1.0
    assert len(rec["lag1_autocorr_t"]) == K
    assert len(rec["ess_t"]) == K


def test_resample_and_rejuvenate_no_callback_unchanged():
    """Without a callback the function should behave exactly as before."""
    rng = np.random.default_rng(0)
    K = 3
    state = core_mcmc.make_state_hier(N=200, K=K, r_assumed=0.378, rng=rng)
    state["history"] = [(0, 0.5, 1), (1, -0.3, 0)]
    from core_mcmc import _log_lik_history
    state["log_lik"] = _log_lik_history(state["t"], state["l"], state["history"])
    accept_rate = core_mcmc.resample_and_rejuvenate(
        state, rng, n_mh_steps=3, proposal_scale=0.5,
    )
    assert 0.0 <= accept_rate <= 1.0
    # No "_diag_event_counter" key should be added when callback is None
    assert "_diag_event_counter" not in state
