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


def _fast_ad6(inputs):
    """A deliberately-lenient AD6 policy that resolves a clearly-pass rater in
    a few trials per task — so the extended-collection tests run quickly."""
    import numpy as np
    from cortex_policy_k7 import load_ell_star_k7
    from cortex_policy import AD6Policy
    ell = load_ell_star_k7(inputs.task_codes)
    vp = list(np.diag(np.asarray(inputs.Corr_l, dtype=float)))
    return AD6Policy(ell, vp, n_min=3, R_star=0.05, alpha=0.30, Z=1.0)


def test_extended_exhausted_domain_continues_on_remaining(inputs):
    """Extended (post-decision) selection must keep going on the domains that
    still have un-served segs when one domain's bank is exhausted — it stops
    ONLY when EVERY domain is empty. (No premature data-collection ending.)"""
    sess = sc.CortexSession(inputs, session_id="t-exhaust", n_particles=50,
                            seed=1, extended_data_collection=True)
    sess._post_decision = True
    K = sess.K
    full = [np.array([0.1, 0.2]) for _ in range(K)]
    assert sess._compute_active_domains(full) == list(range(K))
    # domain 0 exhausted → the rest stay active
    one_empty = [np.array([])] + [np.array([0.1]) for _ in range(K - 1)]
    assert sess._compute_active_domains(one_empty) == list(range(1, K))
    # only the last domain has segs left → keep asking it (not None/empty)
    last_only = [np.array([]) for _ in range(K - 1)] + [np.array([0.1])]
    assert sess._compute_active_domains(last_only) == [K - 1]
    # even with the consecutive-same-domain cap maxed on that last domain, it
    # is NOT stranded — it falls through to the only domain with segs left.
    sess._max_consec = 5
    sess._last_task = K - 1
    sess._consec_count = 9
    assert sess._compute_active_domains(last_only) == [K - 1]
    # all domains exhausted → empty (the run loop then ends bank_exhausted)
    assert sess._compute_active_domains([np.array([]) for _ in range(K)]) == []


def test_extended_off_is_unchanged(inputs):
    """Default (extended OFF): the session stops at the AD6 decision point and
    no trial is flagged post_decision; the new fields mirror the official run."""
    sess = sc.CortexSession(inputs, session_id="t-ext-off", n_particles=200,
                            max_questions=120, seed=11,
                            policy=_fast_ad6(inputs))
    res = sess.run(sc.make_simulated_y_source(
        np.zeros(len(inputs.task_codes)),
        np.full(len(inputs.task_codes), 1.4), seed=11))
    assert res.stop_reason in ("all_resolved", "all_active_resolved")
    assert res.n_questions < 120                      # stopped early
    assert res.n_questions == len(res.trials)
    assert res.n_questions_total == res.n_questions    # no extension
    assert res.extended_stop_reason == res.stop_reason
    assert all(not t.get("post_decision") for t in res.trials)


def test_extended_on_freezes_snapshot_and_keeps_asking(inputs):
    """Extended ON: the OFFICIAL result is byte-frozen at the v1.4.0 decision
    point (same seed ⇒ identical pre-decision trajectory), while the session
    keeps asking — emitting post_decision-flagged trials — to max_questions."""
    K = len(inputs.task_codes)
    true_t, true_l = np.zeros(K), np.full(K, 1.4)

    off = sc.CortexSession(inputs, session_id="t-ext-cmp", n_particles=200,
                           max_questions=120, seed=11,
                           policy=_fast_ad6(inputs))
    res_off = off.run(sc.make_simulated_y_source(true_t, true_l, seed=11))

    emitted = []
    on = sc.CortexSession(inputs, session_id="t-ext-cmp", n_particles=200,
                          max_questions=120, seed=11,
                          policy=_fast_ad6(inputs),
                          extended_data_collection=True)
    res_on = on.run(sc.make_simulated_y_source(true_t, true_l, seed=11),
                    on_trial=emitted.append)

    # OFFICIAL result is frozen exactly at the would-have-stopped point.
    assert res_on.n_questions == res_off.n_questions
    assert res_on.stop_reason == res_off.stop_reason
    assert res_on.verdicts == res_off.verdicts
    assert np.allclose(res_on.final_auroc_mean, res_off.final_auroc_mean)
    assert np.allclose(res_on.final_l_mean, res_off.final_l_mean)
    assert len(res_on.trials) == res_on.n_questions
    assert all(not t.get("post_decision") for t in res_on.trials)

    # ...but the session kept asking past it, up to the ceiling / bank.
    assert res_on.n_questions_total > res_on.n_questions
    assert res_on.n_questions_total == min(120, len(inputs.all_seg_ids))
    assert len(emitted) == res_on.n_questions_total
    # Pre-decision trials match the official set; the rest are flagged.
    assert all(not t.get("post_decision")
               for t in emitted[:res_on.n_questions])
    assert all(t.get("post_decision")
               for t in emitted[res_on.n_questions:])


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
    """Bank-exhaustion path: when the delta-AUROC target is unreachable the
    session asks EVERY available question and stops with `bank_exhausted`.
    NOTE: the original Phase-A delta=0.05 / 0.6-rater scenario now RESOLVES
    (`delta_reached`) on the v1.3.6 600-segment IIIC bank (~100 q/task is enough
    to shrink hw<0.05). To keep this test exercising the exhaustion path
    independent of bank size, use an unreachable delta (the hw floor at finite
    N is ~0.05-0.1, far above 0.001)."""
    K = len(inputs.task_codes)
    y_src = sc.make_simulated_y_source(np.zeros(K), np.full(K, 0.6), seed=0)
    sess = sc.CortexSession(inputs, session_id="t-exhaust",
                            n_particles=300, delta_auroc=0.001, seed=0)
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
