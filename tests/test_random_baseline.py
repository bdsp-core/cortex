"""F-rand.1 — random item-selection null baseline.

Pins:
  - run_session_mcmc_auroc(method="random") dispatches to the independent-
    posterior brute_k engine with select="random".
  - The same seed reproduces the session bitwise (parallel-determinism
    guarantee — random selection draws from the session rng).
  - Different seeds give different question sequences (it IS random).
  - random and brute (ev) differ — random does NOT do adaptive selection.
  - Output contract matches the Mode-A return dict; method == "random".
  - random_item_brute_k respects bank bounds and active_domains.
"""
from __future__ import annotations

import numpy as np
import pytest

import core_mcmc
import core_mcmc_brute_k

from tests.conftest import true_params_array


def _tp(K, seed=0):
    rng = np.random.default_rng(seed)
    l = rng.normal(0.4, 0.3, size=K)
    t = rng.normal(0.0, 0.5, size=K)
    return true_params_array(t, l)


def test_random_selector_contract_and_bounds():
    rng = np.random.default_rng(0)
    K = 4
    states = core_mcmc_brute_k.make_state_brute_k(N=50, K=K, rng=rng)
    banks = [np.linspace(-2, 2, 7 + k) for k in range(K)]  # ragged banks
    sel_rng = np.random.default_rng(1)
    seen_k = set()
    for _ in range(200):
        k, s = core_mcmc_brute_k.random_item_brute_k(states, sel_rng, banks)
        assert 0 <= k < K
        assert s in banks[k], f"signal {s} not drawn from bank {k}"
        seen_k.add(k)
    assert seen_k == set(range(K)), "random selection must reach all domains"


def test_random_selector_respects_active_domains():
    rng = np.random.default_rng(0)
    K = 5
    states = core_mcmc_brute_k.make_state_brute_k(N=30, K=K, rng=rng)
    banks = [np.linspace(-2, 2, 9) for _ in range(K)]
    sel_rng = np.random.default_rng(2)
    active = [1, 3]
    for _ in range(150):
        k, _s = core_mcmc_brute_k.random_item_brute_k(
            states, sel_rng, banks, active_domains=active)
        assert k in active


def test_method_random_dispatch_and_determinism():
    K = 3
    tp = _tp(K, seed=7)
    common = dict(method="random", true_params=tp, K=K, r_assumed=0.378,
                  max_q=120, delta_auroc=0.05, N=200, seed=42,
                  ess_threshold_frac=0.5)
    out1 = core_mcmc.run_session_mcmc_auroc(**common)
    out2 = core_mcmc.run_session_mcmc_auroc(**common)
    assert out1["method"] == "random"
    # Bitwise determinism for the same seed.
    np.testing.assert_array_equal(out1["final_lo"], out2["final_lo"])
    np.testing.assert_array_equal(out1["final_hi"], out2["final_hi"])
    assert out1["n_questions"] == out2["n_questions"]
    # Output contract: same keys as the Mode-A AUROC dict.
    for key in ("n_questions", "stopped_early", "true_auroc",
                "final_lo", "final_hi", "delta_auroc", "method"):
        assert key in out1


def test_random_differs_from_ev_and_from_other_seed():
    K = 3
    tp = _tp(K, seed=11)
    base = dict(true_params=tp, K=K, r_assumed=0.378, max_q=200,
                delta_auroc=0.05, N=300, ess_threshold_frac=0.5)
    rnd_a = core_mcmc.run_session_mcmc_auroc(method="random", seed=1, **base)
    rnd_b = core_mcmc.run_session_mcmc_auroc(method="random", seed=2, **base)
    brute = core_mcmc.run_session_mcmc_auroc(method="brute", seed=1, **base)
    # Different seeds → different random sequences → different CI trajectory.
    assert not np.array_equal(rnd_a["final_lo"], rnd_b["final_lo"]), \
        "random with different seeds should differ"
    # random and brute (EV) use the same posterior model but different
    # item selection → final CIs should not be identical.
    assert not np.array_equal(rnd_a["final_lo"], brute["final_lo"]), \
        "random and EV selection should produce different posteriors"


@pytest.mark.parametrize("bad", ["adaptive", "kl", "", None])
def test_invalid_select_rejected(bad):
    tp = _tp(2, seed=0)
    with pytest.raises(ValueError, match="select must be"):
        core_mcmc_brute_k.run_session_mcmc_brute_k(
            tp, K=2, max_q=10, N=50, seed=0, select=bad)
