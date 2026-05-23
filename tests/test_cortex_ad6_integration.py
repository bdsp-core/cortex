"""Phase B-AD6 — tests for the TerminationPolicy integration in
scripts/session_controller.py.

Gates the AD6 production surface added on top of the existing
delta-based methodology surface:

  * `default_policy_for` precedence ladder (None / 0.0 / >0 / explicit)
  * per-trial telemetry injection of `policy_diag` + `verdicts`
  * SessionResult.verdicts + SessionResult.policy_diagnostics
  * `policy.finalize_verdicts()` runs on the abort path too
  * n_per_task increment consistency (sum == n_questions)
  * `stop_reason="all_resolved"` is reachable with a relaxed-threshold
    AD6Policy (the production thresholds are deliberately strict; this
    test exercises the code path, not the calibration)

Skipped wholesale when data/eeg_bank.h5 is absent (gitignored — D9).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import cortex_engine_inputs as cei  # noqa: E402
import session_controller as sc  # noqa: E402
import cortex_policy as cp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(cei.BANK_PATH),
    reason=f"eeg_bank.h5 not present at {cei.BANK_PATH}")


@pytest.fixture(scope="module")
def inputs():
    return cei.build_iiic_engine_inputs()


def _const_y(value=1):
    return lambda k, seg_id, s: int(value)


# ── precedence ladder ────────────────────────────────────────────────────
def test_default_delta_auroc_resolves_to_ad6_policy(inputs):
    sess = sc.CortexSession(inputs, session_id="t-prec-ad6",
                            n_particles=120, max_questions=1, seed=10)
    assert isinstance(sess.policy, cp.AD6Policy)
    assert sess.delta_auroc is None


def test_zero_delta_resolves_to_no_stop_policy(inputs):
    sess = sc.CortexSession(inputs, session_id="t-prec-nostop",
                            n_particles=120, max_questions=1, seed=11,
                            delta_auroc=0.0)
    assert isinstance(sess.policy, cp.NoStopPolicy)


def test_positive_delta_resolves_to_delta_stop_policy(inputs):
    sess = sc.CortexSession(inputs, session_id="t-prec-delta",
                            n_particles=120, max_questions=1, seed=12,
                            delta_auroc=0.15)
    assert isinstance(sess.policy, cp.DeltaStopPolicy)
    assert sess.policy.delta_auroc == 0.15


def test_explicit_policy_takes_precedence(inputs):
    """policy=... wins over any delta_auroc value (incl. None)."""
    explicit = cp.NoStopPolicy()
    sess = sc.CortexSession(inputs, session_id="t-prec-explicit",
                            n_particles=120, max_questions=1, seed=13,
                            delta_auroc=0.15, policy=explicit)
    assert sess.policy is explicit


# ── AD6 telemetry injection ──────────────────────────────────────────────
def test_ad6_telemetry_includes_policy_diag_and_verdicts(inputs):
    """Each trial dict must carry the AD6 diagnostics + the running
    verdict list — the viewer / trials.jsonl rely on both for live UI."""
    sess = sc.CortexSession(inputs, session_id="t-ad6-tel",
                            n_particles=200, max_questions=4, seed=14)
    res = sess.run(_const_y(1))
    assert res.n_questions == 4
    for tel in res.trials:
        assert "policy_diag" in tel, "AD6 must inject policy_diag into tel"
        assert "verdicts" in tel, "AD6 must inject the running verdicts list"
        diag = tel["policy_diag"]
        for key in ("pi", "mcse", "R", "ess", "verdicts", "n_per_task"):
            assert key in diag, f"policy_diag missing {key}"
        assert len(diag["pi"]) == 6
        assert len(diag["mcse"]) == 6
        assert len(diag["R"]) == 6
        assert len(diag["n_per_task"]) == 6
        assert len(tel["verdicts"]) == 6


def test_ad6_session_result_carries_verdicts_and_diagnostics(inputs):
    sess = sc.CortexSession(inputs, session_id="t-ad6-result",
                            n_particles=200, max_questions=5, seed=15)
    res = sess.run(_const_y(1))
    assert isinstance(res.verdicts, list)
    assert len(res.verdicts) == 6
    # finalize_verdicts() must never leave a task in PENDING
    assert cp.PENDING not in res.verdicts
    assert isinstance(res.policy_diagnostics, dict)
    assert "pi" in res.policy_diagnostics and "R" in res.policy_diagnostics


# ── no-policy paths emit no policy_diag / verdicts ───────────────────────
def test_no_stop_path_omits_policy_telemetry(inputs):
    sess = sc.CortexSession(inputs, session_id="t-nostop-tel",
                            n_particles=200, max_questions=4, seed=16,
                            delta_auroc=0.0)
    res = sess.run(_const_y(1))
    for tel in res.trials:
        assert "policy_diag" not in tel
        assert "verdicts" not in tel
    assert res.verdicts is None
    assert res.policy_diagnostics is None


def test_delta_stop_path_omits_policy_telemetry(inputs):
    """DeltaStopPolicy must keep tel and SessionResult unpolluted by AD6
    keys — the legacy methodology contract."""
    sess = sc.CortexSession(inputs, session_id="t-delta-tel",
                            n_particles=200, max_questions=6, seed=17,
                            delta_auroc=0.15)
    res = sess.run(_const_y(1))
    for tel in res.trials:
        assert "policy_diag" not in tel
        assert "verdicts" not in tel
    assert res.verdicts is None
    assert res.policy_diagnostics is None


# ── n_per_task consistency ───────────────────────────────────────────────
def test_n_per_task_sums_to_n_questions(inputs):
    sess = sc.CortexSession(inputs, session_id="t-npt",
                            n_particles=200, max_questions=10, seed=18)
    res = sess.run(_const_y(1))
    # tel carries the post-increment count for EVERY policy path
    final_n_per_task = res.trials[-1]["n_per_task"]
    assert sum(final_n_per_task) == res.n_questions
    # and the same count surfaces in the AD6 diagnostics dict
    assert res.policy_diagnostics["n_per_task"] == final_n_per_task


# ── all_resolved stop reason — relaxed-threshold AD6 ─────────────────────
def test_all_resolved_stop_reason_reachable(inputs):
    """A maximally-permissive AD6Policy (n_min=1, R*=-inf, alpha=0.5, Z=0)
    must resolve every task within a few trials and stop with the
    'all_resolved' stop_reason. This exercises the production code path
    even though the field thresholds are deliberately strict."""
    ell_star = [0.0] * 6
    var_prior = list(np.diag(np.asarray(inputs.Corr_l, float)))
    permissive = cp.AD6Policy(ell_star, var_prior,
                              n_min=1, R_star=-float("inf"),
                              alpha=0.5, Z=0.0)
    sess = sc.CortexSession(inputs, session_id="t-allres",
                            n_particles=200, max_questions=40, seed=19,
                            policy=permissive)
    res = sess.run(_const_y(1))
    assert res.stop_reason == "all_resolved"
    assert res.n_questions < 40
    assert res.verdicts is not None and cp.PENDING not in res.verdicts


# ── abort path still finalizes verdicts ──────────────────────────────────
def test_abort_finalizes_verdicts(inputs):
    """When SessionAborted fires mid-session, finalize_verdicts() must
    still run so the on-disk certificate reflects partial conclusions."""
    abort_after = 3

    def y_src(k, seg_id, s):
        if y_src.calls >= abort_after:
            raise sc.SessionAborted()
        y_src.calls += 1
        return 1
    y_src.calls = 0

    sess = sc.CortexSession(inputs, session_id="t-abort",
                            n_particles=200, max_questions=20, seed=20)
    res = sess.run(y_src)
    assert res.aborted is True
    assert res.stop_reason == "aborted"
    assert res.n_questions == abort_after
    # AD6 still rendered verdicts — PENDING tasks become REFER_*
    assert isinstance(res.verdicts, list) and len(res.verdicts) == 6
    assert cp.PENDING not in res.verdicts
    valid = {cp.PASS, cp.FAIL,
             cp.REFER_BORDERLINE, cp.REFER_UNINFORMATIVE}
    assert set(res.verdicts).issubset(valid)


def test_abort_on_nostop_keeps_verdicts_none(inputs):
    """NoStopPolicy never resolves anything; on abort, finalize_verdicts()
    returns None — symmetric with the rest of the NoStop contract."""
    def y_src(k, seg_id, s):
        if y_src.calls >= 2:
            raise sc.SessionAborted()
        y_src.calls += 1
        return 1
    y_src.calls = 0
    sess = sc.CortexSession(inputs, session_id="t-abort-nostop",
                            n_particles=200, max_questions=20, seed=21,
                            delta_auroc=0.0)
    res = sess.run(y_src)
    assert res.aborted is True
    assert res.verdicts is None
    assert res.policy_diagnostics is None
