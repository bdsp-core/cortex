"""Drift-guard for the estimation-ladder engine kwargs (2026-06-08).

Pins, for the random/brute/hier skill/bias-recovery ablation consumed by
``paper_sims/drivers/sim_ladder.py`` (NEJM-AI estimation-accuracy thread), that
the opt-in kwargs added to ``run_session_mcmc_brute_k`` / ``run_session_mcmc_auroc``
are BYTE-IDENTICAL-BY-DEFAULT observers:

  * ``capture_posterior=True`` changes ONLY the output dict (adds ``*_traj`` keys);
    it does NOT alter ``n_questions`` / the AUROC CI trajectory / RNG consumption.
  * ``bank_segids`` threads the chosen ``seg_id`` through brute & random selection,
    consistent with the chosen signal value.
  * ``y_source`` overrides the response draw (parity with the hier path); a
    seg-keyed deterministic ``y_source`` returns the same response for a given
    ``(k, seg_id)`` regardless of which method asked it — the CRN exact-pairing
    contract the efficiency ratios rely on.

The unchanged DEFAULT code paths are additionally covered by the full regression
suite (test_random_baseline, test_smoke_mode_a, test_phase7_replay_engine_drift).
"""
from __future__ import annotations

import numpy as np

import core_mcmc
import core_mcmc_brute_k as bk

from tests.conftest import true_params_array

_TRAJ_KEYS = ("mean_l_traj", "sd_l_traj", "q025_l_traj", "q975_l_traj",
              "mean_t_traj", "sd_t_traj", "q025_t_traj", "q975_t_traj")


def _tp(K, seed=0):
    rng = np.random.default_rng(seed)
    l = rng.normal(0.4, 0.3, size=K)
    t = rng.normal(0.0, 0.5, size=K)
    return true_params_array(t, l)


def _banks(K, n=25):
    sigs = [np.linspace(-2.0, 2.0, n + k) for k in range(K)]   # ragged lengths
    segids = [np.arange(1000 * (k + 1), 1000 * (k + 1) + len(sigs[k]))
              for k in range(K)]
    return sigs, segids


def _assert_capture_is_observer(method):
    K = 3
    tp = _tp(K, seed=3)
    common = dict(method=method, true_params=tp, K=K, r_assumed=0.378,
                  max_q=60, delta_auroc=0.05, N=200, seed=11,
                  ess_threshold_frac=0.5, run_until_max=True, log_trajectory=True)
    base = core_mcmc.run_session_mcmc_auroc(**common)
    cap = core_mcmc.run_session_mcmc_auroc(capture_posterior=True, **common)
    # capture is a pure observer: behavior + AUROC trajectory unchanged.
    assert cap["n_questions"] == base["n_questions"]
    np.testing.assert_array_equal(cap["final_lo"], base["final_lo"])
    np.testing.assert_array_equal(cap["final_hi"], base["final_hi"])
    np.testing.assert_array_equal(cap["lo_traj"], base["lo_traj"])
    np.testing.assert_array_equal(cap["hi_traj"], base["hi_traj"])
    nrows = base["lo_traj"].shape[0]            # max_q + 1 under run_until_max
    for key in _TRAJ_KEYS:
        assert cap[key].shape == (nrows, K), (method, key, cap[key].shape)
    # interval ordering + non-negative SD.
    assert np.all(cap["q025_l_traj"] <= cap["mean_l_traj"] + 1e-9)
    assert np.all(cap["mean_l_traj"] <= cap["q975_l_traj"] + 1e-9)
    assert np.all(cap["q025_t_traj"] <= cap["q975_t_traj"] + 1e-9)
    assert np.all(cap["sd_l_traj"] >= 0) and np.all(cap["sd_t_traj"] >= 0)


def test_capture_observer_hier():
    _assert_capture_is_observer("hier")


def test_capture_observer_brute():
    _assert_capture_is_observer("brute")


def test_capture_observer_random():
    _assert_capture_is_observer("random")


def _assert_segid_threaded(select):
    K = 3
    tp = _tp(K, seed=5)
    sigs, segids = _banks(K)
    n = {"n": 0}

    def y_source(k, seg_id, s):
        # seg_id must be the id at the bank index whose signal == the chosen s.
        idx = int(np.argmin(np.abs(sigs[k] - s)))
        assert int(segids[k][idx]) == int(seg_id), (k, s, seg_id)
        n["n"] += 1
        return 1 if s > 0 else 0

    out = bk.run_session_mcmc_brute_k(
        tp, K=K, max_q=30, N=150, seed=7, run_until_max=True,
        bank_signals=sigs, bank_segids=segids, y_source=y_source, select=select)
    assert n["n"] == 30
    assert out["method"] == ("random" if select == "random" else "brute")


def test_bank_segids_threaded_brute():
    _assert_segid_threaded("ev")


def test_bank_segids_threaded_random():
    _assert_segid_threaded("random")


def test_y_source_response_is_method_invariant_per_segment():
    """A seg-keyed deterministic y_source returns the same y for a given
    (k, seg_id) whether brute or random asked it (CRN exact-pairing)."""
    K = 2
    tp = _tp(K, seed=9)
    sigs, segids = _banks(K)
    seen = {}

    def y_source(k, seg_id, s):
        u = ((1103515245 * (int(seg_id) + 7 * k) + 12345) % 2147483648) / 2147483648.0
        y = int(u < 0.5)
        if (k, int(seg_id)) in seen:
            assert seen[(k, int(seg_id))] == y
        seen[(k, int(seg_id))] = y
        return y

    for method in ("brute", "random"):
        core_mcmc.run_session_mcmc_auroc(
            method=method, true_params=tp, K=K, r_assumed=0.378,
            max_q=40, N=150, seed=3, run_until_max=True,
            bank_signals=sigs, bank_segids=segids, y_source=y_source)
    assert len(seen) > 0


def test_hier_y_source_capture_integration():
    """The hier path runs with bank_segids + y_source + capture_posterior
    together (the configuration the driver uses for method='hier')."""
    K = 2
    tp = _tp(K, seed=1)
    sigs, segids = _banks(K)

    def y_source(k, seg_id, s):
        return 1 if (int(seg_id) % 2 == 0) else 0

    out = core_mcmc.run_session_mcmc_auroc(
        method="hier", true_params=tp, K=K, r_assumed=0.378,
        max_q=25, N=200, seed=2, run_until_max=True,
        bank_signals=sigs, bank_segids=segids, y_source=y_source,
        capture_posterior=True)
    assert out["mean_l_traj"].shape == (26, K)
    assert np.all(np.isfinite(out["mean_l_traj"]))
