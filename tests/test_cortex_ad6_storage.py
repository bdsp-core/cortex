"""Phase D-AD6 — tests for the AD6 storage + UI integration.

Covers the cortex_storage + eeg_bank_viewer changes wired up to
SessionResult.verdicts and SessionResult.policy_diagnostics:

  * certificate.json — `per_task[i].verdict` is populated when verdicts
    are present and is None on the legacy NoStop / Delta paths.
  * trajectory.npz — encodes delta_auroc=None as NaN (AD6 sessions have
    no methodology delta; the legacy float values pass through unchanged).
  * ResultsScreen — renders the "all_resolved" stop-reason sentence
    instead of falling through to an empty string.

Bank-independent — stubs SessionResult with SimpleNamespace, the same
pattern test_cortex_storage.py uses.
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import cortex_storage as cs  # noqa: E402
import cortex_policy as cp  # noqa: E402

PARTICIPANT = {"name": "Jane Doe", "email": "jane@example.com",
               "expertise": "Fellow", "institution": "MGH"}
TASK_CODES = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]


def _telemetry(trial_index, k=0, seg_id=1000, pc="seizure", y=1):
    return {"trial_index": trial_index, "task_k": k, "task_code": TASK_CODES[k],
            "seg_id": seg_id, "pattern_class_true": pc, "s_mean": 0.2,
            "s_sd": 0.1, "response_y": y, "select_ms": 12.0,
            "expected_loss_chosen": 3.1, "total_var_t": 1.5, "total_var_l": 1.6,
            "total_var": 3.1, "auroc_mean": [0.8] * 6, "auroc_hw": [0.2] * 6,
            "max_hw": 0.2, "t_post_mean": [0.0] * 6, "l_post_mean": [0.4] * 6,
            "ess": 350.0, "rejuv": False}


def _gui(trial_index, label="Seizure"):
    return {"trial_index": trial_index, "seg_id": 1000, "task_k": 0,
            "response_raw": 0, "response_label": label,
            "reaction_time_ms": 820.5, "answer_changes": 0, "montage": "bipolar",
            "gain_uv": 100.0, "bandpass": "0.5-70 Hz", "notch": "60 Hz",
            "window_s": 10.0, "pan_t_start": 0.0,
            "interaction": []}


def _ad6_result(n=2, verdicts=None):
    """SessionResult stub for the AD6 path — delta_auroc=None,
    verdicts populated, policy_diagnostics carries the last call."""
    if verdicts is None:
        verdicts = [cp.PASS, cp.FAIL,
                    cp.REFER_BORDERLINE, cp.REFER_UNINFORMATIVE,
                    cp.PASS, cp.FAIL]
    return SimpleNamespace(
        session_id="sess-ad6", stop_reason="all_resolved", delta_auroc=None,
        n_questions=n, task_codes=list(TASK_CODES),
        final_auroc_mean=np.full(6, 0.82), final_auroc_hw=np.full(6, 0.14),
        final_l_mean=np.full(6, 0.45), final_t_mean=np.zeros(6),
        served_seg_ids=list(range(1000, 1000 + n)),
        t_traj=np.zeros((n, 50, 6)), l_traj=np.zeros((n, 50, 6)),
        w_traj=np.full((n, 50), 0.02), aborted=False,
        verdicts=list(verdicts),
        policy_diagnostics={"pi": [0.95] * 6, "mcse": [0.01] * 6,
                            "R": [0.4] * 6, "ess": 350.0,
                            "verdicts": list(verdicts),
                            "n_per_task": [1, 1, 0, 0, 0, 0]})


def _legacy_result(n=2):
    """SessionResult stub for the legacy Delta path — no verdicts."""
    return SimpleNamespace(
        session_id="sess-leg", stop_reason="delta_reached", delta_auroc=0.15,
        n_questions=n, task_codes=list(TASK_CODES),
        final_auroc_mean=np.full(6, 0.82), final_auroc_hw=np.full(6, 0.14),
        final_l_mean=np.full(6, 0.45), final_t_mean=np.zeros(6),
        served_seg_ids=list(range(1000, 1000 + n)),
        t_traj=np.zeros((n, 50, 6)), l_traj=np.zeros((n, 50, 6)),
        w_traj=np.full((n, 50), 0.02), aborted=False,
        verdicts=None, policy_diagnostics=None)


def _recorder(tmp_path, sid="sess-test"):
    return cs.SessionRecorder(
        sid, PARTICIPANT, {"delta_auroc": None, "n_particles": 600},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=False)         # see test_cortex_storage._recorder


# ── certificate.json verdict plumbing ────────────────────────────────────
def test_certificate_per_task_verdict_populates_for_ad6(tmp_path):
    rec = _recorder(tmp_path, sid="sess-ad6")
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1), _gui(1))
    rec.finalize(_ad6_result())
    cert = json.loads((rec.dir / "certificate.json").read_text())
    assert len(cert["per_task"]) == 6
    expected = [cp.PASS, cp.FAIL, cp.REFER_BORDERLINE,
                cp.REFER_UNINFORMATIVE, cp.PASS, cp.FAIL]
    for i, entry in enumerate(cert["per_task"]):
        assert entry["verdict"] == expected[i]
        # the original AUROC/ell/t fields must still be present
        for key in ("task_code", "auroc_mean", "auroc_halfwidth",
                    "ell_mean", "t_mean"):
            assert key in entry
    rec.close()


def test_certificate_per_task_verdict_is_none_for_legacy_delta(tmp_path):
    """The legacy Delta / NoStop paths set verdicts=None; the certificate
    must explicitly carry verdict=null for each task — not omit the key —
    so downstream parsers can rely on the schema."""
    rec = _recorder(tmp_path, sid="sess-leg")
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_legacy_result(n=1))
    cert = json.loads((rec.dir / "certificate.json").read_text())
    for entry in cert["per_task"]:
        assert "verdict" in entry
        assert entry["verdict"] is None
    rec.close()


# ── trajectory.npz delta_auroc=None guard ────────────────────────────────
def test_trajectory_npz_encodes_none_delta_as_nan(tmp_path):
    rec = _recorder(tmp_path, sid="sess-ad6-traj")
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_ad6_result(n=1))
    npz = np.load(rec.dir / "trajectory.npz")
    assert math.isnan(float(npz["delta_auroc"]))
    rec.close()


def test_trajectory_npz_preserves_float_delta(tmp_path):
    rec = _recorder(tmp_path, sid="sess-leg-traj")
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_legacy_result(n=1))
    npz = np.load(rec.dir / "trajectory.npz")
    assert float(npz["delta_auroc"]) == pytest.approx(0.15)
    rec.close()


# ── ResultsScreen "all_resolved" copy ────────────────────────────────────
def test_results_screen_renders_all_resolved():
    """The viewer's _STOP_TEXT must map 'all_resolved' to a non-empty
    user-facing sentence — otherwise the AD6 production path would land
    on a blank subtitle on the certification screen."""
    pytest.importorskip("PyQt6")
    pytest.importorskip("h5py")
    if not os.path.exists(os.path.join(_REPO, "data", "eeg_bank.h5")):
        pytest.skip("eeg_bank.h5 not present (viewer imports load it)")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel
    import eeg_bank_viewer as ev

    app = QApplication.instance() or QApplication([])  # noqa: F841
    screen = ev.ResultsScreen(_ad6_result(n=24), n_correct=20, n_answered=24)
    try:
        texts = [w.text() for w in screen.findChildren(QLabel)]
        assert any("Each category reached a final assessment." in t
                   for t in texts), (
            f"_STOP_TEXT['all_resolved'] not rendered; saw: {texts}")
        assert any("24 recordings reviewed" in t for t in texts)
    finally:
        screen.deleteLater()
        app.processEvents()
