"""Tests for the Step-3 decision-aligned item-selection objective.

`choose_item(..., ell_star=None)` must be BYTE-IDENTICAL to the shipped
A-optimal selector. When `ell_star` is provided it switches to
`_expected_decision_loss_vec` — minimise expected Σ_kk π_kk(1−π_kk) over
`decision_tasks`. See docs/ENGINE_IMPROVEMENT_RESULTS.md Step 3.
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


def _state(N=400, K=3, seed=0):
    rng = np.random.default_rng(seed)
    l = rng.normal(0.3, 0.5, size=(N, K))
    t = rng.normal(0.0, 0.4, size=(N, K))
    logw = rng.normal(0.0, 0.3, size=N)
    w = np.exp(logw - logw.max())
    w = w / w.sum()
    return {"t": t, "l": l, "w": w}


def _banks(K=3, seed=1):
    rng = np.random.default_rng(seed)
    sizes = [20, 25, 15][:K] + [18] * max(0, K - 3)
    return [np.sort(rng.uniform(-2.0, 2.0, size=n)) for n in sizes]


def test_choose_item_default_matches_bruteforce_variance_argmin():
    """ell_star=None ⇒ exact A-optimal selector: global argmin of
    _expected_loss_vec over the full bank (n_subsample=None = full grid)."""
    K = 3
    state = _state(K=K, seed=2)
    banks = _banks(K=K, seed=3)
    k_sel, s_sel = cm.choose_item(state, banks)

    best = (np.inf, None, None)
    for k in range(K):
        losses = cm._expected_loss_vec(state, k, banks[k])
        j = int(np.argmin(losses))
        if losses[j] < best[0]:
            best = (float(losses[j]), k, float(banks[k][j]))
    assert k_sel == best[1]
    assert s_sel == best[2]


def test_choose_item_default_is_unchanged_by_explicit_variance():
    state = _state(seed=5)
    banks = _banks(seed=6)
    assert cm.choose_item(state, banks) == cm.choose_item(
        state, banks, objective="variance")


def test_ell_variance_drops_theta_block():
    """ell_variance == the ℓ-only trace; differs from the full A-optimal trace
    (which adds Σ Var(θ)), and equals _expected_loss_vec(include_theta=False)."""
    K = 3
    state = _state(K=K, seed=9)
    banks = _banks(K=K, seed=10)
    # The per-signal objective must match include_theta=False exactly.
    k = 2
    got = cm._expected_loss_vec(state, k, banks[k], include_theta=False)
    # Manual ℓ-only trace.
    w = state["w"]
    el = np.exp(state["l"][:, k])[None, :]
    z = el * (banks[k][:, None] + state["t"][:, k][None, :])
    p = np.clip(cm._p_response_yes(z), 1e-9, 1 - 1e-9)
    p_yes = (p * w).sum(axis=1)
    w1 = p * w; w1 = w1 / w1.sum(axis=1, keepdims=True)
    w0 = (1 - p) * w; w0 = w0 / w0.sum(axis=1, keepdims=True)

    def var_vec(arr, weights):
        mu = (weights * arr).sum(axis=1)
        return (weights * (arr - mu[:, None]) ** 2).sum(axis=1)

    t1 = sum(var_vec(state["l"][:, kk], w1) for kk in range(K))
    t0 = sum(var_vec(state["l"][:, kk], w0) for kk in range(K))
    expected = p_yes * t1 + (1 - p_yes) * t0
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    # And the full trace is strictly larger (θ variance ≥ 0 added).
    full = cm._expected_loss_vec(state, k, banks[k], include_theta=True)
    assert np.all(full >= got - 1e-12)


def test_include_theta_true_is_byte_identical_default():
    """include_theta=True must reproduce the shipped trace bit-for-bit."""
    state = _state(seed=21)
    banks = _banks(seed=22)
    for k in range(3):
        a = cm._expected_loss_vec(state, k, banks[k])
        b = cm._expected_loss_vec(state, k, banks[k], include_theta=True)
        np.testing.assert_array_equal(a, b)


def test_decision_loss_matches_closed_form():
    """_expected_decision_loss_vec == p_yes·Σπ1(1−π1) + (1−p_yes)·Σπ0(1−π0)."""
    K = 3
    state = _state(K=K, seed=7)
    banks = _banks(K=K, seed=8)
    ell_star = np.array([0.0, 0.3, 0.5])
    dtasks = [0, 1, 2]
    k = 1
    sigs = banks[k]
    got = cm._expected_decision_loss_vec(state, k, sigs, ell_star, dtasks)

    # Manual replication of the reweighting + Bernoulli-variance objective.
    w = state["w"]
    el = np.exp(state["l"][:, k])[None, :]
    z = el * (sigs[:, None] + state["t"][:, k][None, :])
    p = cm._p_response_yes(z)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    p_yes = (p * w).sum(axis=1)
    w1 = p * w; w1 = w1 / w1.sum(axis=1, keepdims=True)
    w0 = (1 - p) * w; w0 = w0 / w0.sum(axis=1, keepdims=True)
    L1 = np.zeros(len(sigs)); L0 = np.zeros(len(sigs))
    for kk in dtasks:
        ind = (state["l"][:, kk] > ell_star[kk]).astype(float)
        pi1 = (w1 * ind[None, :]).sum(axis=1)
        pi0 = (w0 * ind[None, :]).sum(axis=1)
        L1 += pi1 * (1 - pi1)
        L0 += pi0 * (1 - pi0)
    expected = p_yes * L1 + (1 - p_yes) * L0
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_decision_objective_can_differ_from_variance():
    """On a cloud where one task is near-decided and another is at π≈0.5, the
    decision objective should be free to pick a different item than the
    variance objective (sanity: the two objectives are not identical)."""
    K = 3
    state = _state(K=K, seed=11)
    banks = _banks(K=K, seed=12)
    ell_star = np.array([0.3, 0.3, 0.3])
    k_var, s_var = cm.choose_item(state, banks)
    k_dec, s_dec = cm.choose_item(state, banks, objective="decision",
                                  ell_star=ell_star, decision_tasks=[0, 1, 2])
    # Both must be valid picks.
    for (k, s) in ((k_var, s_var), (k_dec, s_dec)):
        assert 0 <= k < K
        assert s in set(banks[k].tolist())
    # They need not differ on every state, but the decision objective must be
    # a genuine alternative: its chosen item's decision-loss is ≤ that of the
    # variance pick (it is the argmin of the decision objective).
    dec_at_dec = cm._expected_decision_loss_vec(
        state, k_dec, np.array([s_dec]), ell_star, [0, 1, 2])[0]
    dec_at_var = cm._expected_decision_loss_vec(
        state, k_var, np.array([s_var]), ell_star, [0, 1, 2])[0]
    assert dec_at_dec <= dec_at_var + 1e-12


def test_decision_loss_empty_tasks_is_zero():
    state = _state(seed=13)
    banks = _banks(seed=14)
    out = cm._expected_decision_loss_vec(
        state, 0, banks[0], np.array([0.0, 0.0, 0.0]), [])
    np.testing.assert_array_equal(out, np.zeros(len(banks[0])))
