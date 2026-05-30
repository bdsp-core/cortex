"""Phase B — tests for scripts/session_controller.py.

Gates the adaptive-session controller: the pure CortexSession loop, the
y_source-equivalence contract, segment de-duplication, the native
delta-stop mechanism, and the queue-backed Qt EngineWorker thread.
Skipped wholesale when data/eeg_bank.h5 is absent (gitignored — D9).
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import cortex_engine_inputs as cei  # noqa: E402
import session_controller as sc  # noqa: E402
from core_mcmc import make_state_hier, choose_item, _expected_loss_vec  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(cei.BANK_PATH),
    reason=f"eeg_bank.h5 not present at {cei.BANK_PATH}")


@pytest.fixture(scope="module")
def inputs():
    return cei.build_iiic_engine_inputs()


def _const_y_source(value=1):
    return lambda k, seg_id, s: int(value)


def test_session_runs_and_contract(inputs):
    sess = sc.CortexSession(inputs, session_id="t-contract", n_particles=200,
                            max_questions=12, seed=1, delta_auroc=0.0)
    res = sess.run(_const_y_source(1))
    assert res.n_questions == 12
    assert len(res.trials) == 12
    assert res.selection == "adaptive"
    assert res.final_auroc_mean.shape == (6,)
    assert res.final_auroc_hw.shape == (6,)
    assert res.final_t_mean.shape == res.final_l_mean.shape == (6,)
    tel = res.trials[0]
    for key in ("trial_index", "task_k", "seg_id", "response_y", "select_ms",
                "expected_loss_chosen", "total_var", "max_hw", "ess", "rejuv"):
        assert key in tel, f"telemetry missing {key}"
    assert 0 <= tel["task_k"] < 6


def test_dedup_no_segment_served_twice(inputs):
    sess = sc.CortexSession(inputs, session_id="t-dedup", n_particles=200,
                            max_questions=40, seed=2, delta_auroc=0.0)
    res = sess.run(_const_y_source(1))
    assert len(res.served_seg_ids) == len(set(res.served_seg_ids))
    assert set(res.served_seg_ids).issubset(set(inputs.all_seg_ids))


def test_bank_exhaustion_is_capped(inputs):
    """The bank size caps the test — it cannot exceed bank size. Bank-
    size-agnostic so future expansions (v1.1.0 300 segs, future 500+)
    do not need a test bump."""
    sess = sc.CortexSession(inputs, session_id="t-cap", n_particles=200,
                            seed=3, delta_auroc=0.0)
    res = sess.run(_const_y_source(1))
    assert res.n_questions <= len(inputs.all_seg_ids)


def test_y_source_equivalence(inputs):
    """A queue-backed y_source fed a fixed response sequence must produce
    the exact same trajectory as a direct in-process y_source (plan §9)."""
    responses = [int(x) for x in
                 np.random.default_rng(7).integers(0, 2, size=30)]

    it = iter(responses)
    res_direct = sc.CortexSession(inputs, session_id="t-eq", n_particles=200,
                                  max_questions=20, seed=42,
                                  delta_auroc=0.0).run(
        lambda k, seg_id, s: next(it))

    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(r) for r in responses],
                     daemon=True).start()
    res_queue = sc.CortexSession(inputs, session_id="t-eq", n_particles=200,
                                 max_questions=20, seed=42,
                                 delta_auroc=0.0).run(
        lambda k, seg_id, s: q.get())

    assert ([t["seg_id"] for t in res_direct.trials]
            == [t["seg_id"] for t in res_queue.trials])
    assert ([t["max_hw"] for t in res_direct.trials]
            == [t["max_hw"] for t in res_queue.trials])
    assert ([t["total_var"] for t in res_direct.trials]
            == [t["total_var"] for t in res_queue.trials])


def test_delta_stop_mechanism(inputs):
    """A loose delta must fire the native AUROC half-width stop early."""
    sess = sc.CortexSession(inputs, session_id="t-stop", n_particles=200,
                            delta_auroc=0.5, seed=4)
    res = sess.run(_const_y_source(1))
    assert res.stop_reason == "delta_reached"
    assert res.n_questions < len(inputs.all_seg_ids)
    assert res.trials[-1]["max_hw"] < 0.5


def test_native_delta_bank_exhausts(inputs):
    """Phase-A finding (originally on the 100-segment bank): delta=0.05
    AUROC half-width is not reachable for a 0.6-skill rater within the
    bank. On the v1.1.0 300-segment bank this still bank-exhausts —
    a 0.6 rater is not strong enough to hit hw<0.05 even with 300
    questions across 6 tasks. The assertion is on the stop_reason +
    the n_questions equalling the bank size (whatever it is)."""
    K = len(inputs.task_codes)
    y_src = sc.make_simulated_y_source(np.zeros(K), np.full(K, 0.6), seed=0)
    sess = sc.CortexSession(inputs, session_id="t-exhaust",
                            n_particles=300, delta_auroc=0.05, seed=0)
    res = sess.run(y_src)
    assert res.stop_reason == "bank_exhausted"
    assert res.n_questions == len(inputs.all_seg_ids)


def test_random_selection_mode(inputs):
    sess = sc.CortexSession(inputs, session_id="t-rand", selection="random",
                            n_particles=200, max_questions=20, seed=5,
                            delta_auroc=0.0)
    res = sess.run(_const_y_source(1))
    assert res.selection == "random"
    assert res.n_questions == 20
    assert len(res.served_seg_ids) == len(set(res.served_seg_ids))


def test_first_question_varies_by_session(inputs):
    """Q1 must differ across examinees — drawn from the top-N, not a fixed
    argmin (Q2+ remain the strict adaptive argmin)."""
    first_qs = []
    for i in range(8):
        sess = sc.CortexSession(inputs, session_id=f"rater-{i}",
                                n_particles=300, max_questions=1,
                                delta_auroc=0.0)
        res = sess.run(_const_y_source(1))
        first_qs.append(res.served_seg_ids[0])
    assert len(set(first_qs)) > 1, f"Q1 identical across sessions: {first_qs}"


def test_choose_item_returns_global_variance_argmin(inputs):
    """The core 'does it correctly select' assertion: choose_item must
    return the (task, segment) that globally minimises expected posterior
    variance over the live candidate pool."""
    rng = np.random.default_rng(0)
    state = make_state_hier(300, 6, sc.R_ASSUMED, rng,
                            Sigma_l=inputs.Corr_l, Sigma_t=inputs.Corr_l)
    bs, bsd, bseg = inputs.as_engine_arrays()
    k, s, s_sd, seg = choose_item(state, bs, bank_sds=bsd,
                                  return_sd=True, bank_segids=bseg)
    global_min = min(
        float(_expected_loss_vec(state, kk, np.asarray(bs[kk]),
                                 signal_sds=np.asarray(bsd[kk])).min())
        for kk in range(6))
    chosen_loss = float(_expected_loss_vec(
        state, k, np.array([s]), signal_sds=np.array([s_sd]))[0])
    assert abs(chosen_loss - global_min) < 1e-9


def test_simulated_y_source_returns_binary(inputs):
    K = len(inputs.task_codes)
    y_src = sc.make_simulated_y_source(np.zeros(K), np.full(K, 0.5), seed=1)
    vals = {y_src(k % K, 0, 0.3) for k in range(40)}
    assert vals.issubset({0, 1})


def test_engine_worker_threaded_session(inputs):
    """The Qt EngineWorker runs a full session in a background QThread with
    a queue-backed y_source; SessionController drives it via submit_answer."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    ctrl = sc.SessionController(inputs, "t-thread", n_particles=200,
                                max_questions=15, seed=6, delta_auroc=0.0)
    done = {}
    ctrl.sessionComplete.connect(lambda res: done.update(result=res))
    ctrl.itemReady.connect(lambda item: ctrl.submit_answer(0))
    ctrl.start()

    deadline = time.time() + 30.0
    while "result" not in done and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert "result" in done, "threaded session did not complete in 30s"
    res = done["result"]
    assert res.n_questions == 15
    assert len(set(res.served_seg_ids)) == res.n_questions


# ─── v1.3.5: consecutive-same-domain cap (default OFF) ───────────────────
def test_v1_3_5_cap_default_off_never_excludes(inputs):
    """Default (no cap) must never drop the dominant domain — preserves the
    original active-domain behavior exactly."""
    sess = sc.CortexSession(inputs, session_id="t-cap-off", n_particles=50)
    assert sess._max_consec is None
    K = sess.K
    sess.policy.reset(K)
    bank = [np.array([0.1, 0.2]) for _ in range(K)]
    sess._last_task = 1
    sess._consec_count = 99            # way past any cap
    act = sess._compute_active_domains(bank)
    assert 1 in act                    # off => dominant stays selectable


def test_v1_3_5_cap_forces_switch_off_dominant(inputs):
    """With a cap, once the consecutive count reaches it the dominant task is
    excluded from the next selection (forcing a switch)."""
    sess = sc.CortexSession(inputs, session_id="t-cap3",
                            max_consecutive_same_domain=3, n_particles=50)
    assert sess._max_consec == 3
    K = sess.K
    sess.policy.reset(K)
    bank = [np.array([0.1, 0.2]) for _ in range(K)]
    sess._last_task = 1
    # below the cap: dominant stays in
    sess._consec_count = 2
    assert 1 in sess._compute_active_domains(bank)
    # at the cap: dominant forced out, but the set is non-empty
    sess._consec_count = 3
    act = sess._compute_active_domains(bank)
    assert 1 not in act and len(act) >= 1
