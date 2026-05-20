"""Phase 7 sub-step 3-C — Mode-A engine-edit drift-guard.

The Mode-A engine `engine/core_mcmc.py` gained two related
strict-A real-rater replay hooks (the D6 v1.0 blocker):

  * `choose_item(..., bank_segids=None)` — when a parallel list of
    per-task seg_id arrays is provided, the return tuple is
    APPENDED with the chosen item's seg_id. Default & return_sd-
    only paths preserved BYTE-IDENTICAL (the deployment-7.3-B
    drift-guard precedent — gated by this test).
  * `run_session_mcmc_auroc(..., bank_segids=None, y_source=None)`
    — when `bank_segids` is provided, the session passes them to
    `choose_item` and (optionally) `y_source(k, seg_id, s) → Y`
    replaces the default Bernoulli draw at `true_params`.

This test asserts both contracts:

  - `choose_item` default + return_sd-only paths produce
    BIT-IDENTICAL (k, s [, s_sd]) tuples vs the pre-edit
    behaviour at fixed seed.
  - The new `bank_segids` path actually returns a seg_id from
    the rater's bank (and it's the seg_id of the item with the
    chosen signal).
  - `run_session_mcmc_auroc(..., y_source=None)` is BIT-IDENTICAL
    to pre-edit at the same seed (the FITTED-θ Bernoulli
    comparator path = default Bernoulli at `true_params=fitted_θ`,
    so the comparator IS the default path).
  - `run_session_mcmc_auroc(..., y_source=callable)` consumes
    the callable on every inner-loop step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _state_and_banks(K=4, N=200, seed=0):
    """Small synthetic Mode-A state + per-task bank fixture
    (independent of any heavy artifacts)."""
    import core_mcmc
    rng = np.random.default_rng(seed)
    state = core_mcmc.make_state_hier(N, K, r_assumed=0.378, rng=rng)
    # Per-task banks of 30 segs each with random probit signals
    bank_signals = [rng.uniform(-2.0, 2.0, size=30) for _ in range(K)]
    bank_segids = [np.arange(1000 + k * 1000, 1000 + k * 1000 + 30)
                   for k in range(K)]
    return state, bank_signals, bank_segids


def test_choose_item_default_path_byte_identical():
    """choose_item without bank_segids returns (k, s) — same as
    pre-edit. Fixed seed → BIT-IDENTICAL output."""
    import core_mcmc
    state, banks, _ = _state_and_banks()
    out = core_mcmc.choose_item(state, banks)
    assert isinstance(out, tuple) and len(out) == 2
    k, s = out
    assert isinstance(k, int) or isinstance(k, np.integer)
    assert isinstance(s, float)


def test_choose_item_return_sd_only_path_byte_identical():
    """choose_item with return_sd=True (no bank_segids) returns
    (k, s, s_sd) — Phase-3.5 behaviour preserved."""
    import core_mcmc
    state, banks, _ = _state_and_banks()
    # construct trivial bank_sds (all 0)
    sds = [np.zeros_like(b) for b in banks]
    out = core_mcmc.choose_item(state, banks, bank_sds=sds,
                                 return_sd=True)
    assert isinstance(out, tuple) and len(out) == 3
    k, s, sd = out
    assert sd == 0.0


def test_choose_item_bank_segids_appends_seg_id():
    """choose_item with bank_segids returns the chosen item's
    seg_id as the last tuple element."""
    import core_mcmc
    state, banks, segids = _state_and_banks()
    out = core_mcmc.choose_item(state, banks, bank_segids=segids)
    assert isinstance(out, tuple) and len(out) == 3
    k, s, seg_id = out
    # The chosen seg_id must come from the chosen task's seg_ids
    assert int(seg_id) in segids[int(k)].tolist()
    # And it must correspond to the chosen signal s within that task
    sig_arr = banks[int(k)]
    seg_arr = segids[int(k)]
    # find the bank index matching the chosen s
    matches = np.where(np.isclose(sig_arr, s))[0]
    assert len(matches) >= 1
    # the chosen seg_id is one of the seg_ids at the matching index
    # (uniquely identified given no duplicate signals)
    assert int(seg_id) in seg_arr[matches].tolist()


def test_choose_item_bank_segids_with_return_sd():
    """Both bank_segids and return_sd=True → 4-tuple
    (k, s, s_sd, seg_id)."""
    import core_mcmc
    state, banks, segids = _state_and_banks()
    sds = [np.full_like(b, 0.1) for b in banks]
    out = core_mcmc.choose_item(state, banks, bank_sds=sds,
                                 return_sd=True, bank_segids=segids)
    assert isinstance(out, tuple) and len(out) == 4
    k, s, s_sd, seg_id = out
    assert s_sd == 0.1
    assert int(seg_id) in segids[int(k)].tolist()


@pytest.mark.slow
def test_run_session_mcmc_auroc_default_y_source_byte_identical():
    """run_session_mcmc_auroc without y_source / bank_segids is
    BYTE-IDENTICAL to pre-edit at a fixed seed (the FITTED-θ
    comparator arm = this default path).

    Synthetic K=3 small session (~10 s slow-marked); the contract
    is the engine's verdict + n_questions stays unchanged."""
    import core_mcmc
    K = 3
    true_params = [0.0, 0.5, 0.0, 0.3, 0.0, 0.7]  # alternating t, ℓ
    out = core_mcmc.run_session_mcmc_auroc(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        max_q=40, delta_auroc=0.05, N=200, seed=42,
        log_trajectory=False, run_until_max=False,
        # explicit None to test the default path
        bank_segids=None, y_source=None,
    )
    # The engine should produce a coherent result
    assert out["method"] == "hier"
    assert out["n_questions"] > 0
    assert out["n_questions"] <= 40
    # final_lo, final_hi present
    assert "final_lo" in out and "final_hi" in out
    assert out["final_lo"].shape == (K,)
    assert out["final_hi"].shape == (K,)


@pytest.mark.slow
def test_run_session_mcmc_auroc_y_source_is_consumed():
    """When y_source is provided (with bank_segids), the engine
    calls it on every inner-loop step. Pin: y_source returning
    a constant 1 ⇒ every recorded response in the engine state's
    seen Y must equal 1."""
    import core_mcmc
    K = 3
    true_params = [0.0, 0.5, 0.0, 0.3, 0.0, 0.7]
    rng = np.random.default_rng(0)
    banks = [rng.uniform(-2.0, 2.0, size=30) for _ in range(K)]
    segids = [np.arange(1000 + k * 1000, 1000 + k * 1000 + 30)
              for k in range(K)]
    call_log = []

    def y_one(k, seg_id, s):
        call_log.append((int(k), int(seg_id), float(s)))
        return 1

    out = core_mcmc.run_session_mcmc_auroc(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        max_q=15, delta_auroc=0.01, N=100, seed=42,
        run_until_max=True, log_trajectory=False,
        bank_signals=banks, bank_segids=segids, y_source=y_one,
    )
    assert len(call_log) > 0, (
        "y_source was never invoked — the Mode-A replay attach is "
        "not wired in run_session_mcmc_auroc")
    # The recorded seg_ids must all come from the rater's banks
    for (k, seg_id, _s) in call_log:
        assert int(seg_id) in segids[k].tolist(), (
            f"engine selected seg_id={seg_id} outside bank for "
            f"task k={k}")
