"""Tests for scripts/cortex_render_videos.py and its finalize() integration.

Two layers:

  * Fast: import-cleanliness, render_videos=False opt-out actually suppresses
    rendering, and a renderer failure does not break SessionRecorder.finalize().
  * Slow (marked, excluded from the default fast lane): render the actual
    Andrew session at first-test-csv-results/ end-to-end and verify both
    MP4 files are produced and non-empty. ffmpeg is required for the slow
    case; it is not required for any of the fast cases.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import cortex_storage as cs  # noqa: E402

PARTICIPANT = {"name": "Test User", "email": "t@example.com",
               "expertise": "Other", "institution": "TEST"}
TASK_CODES = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
ANDREW_DIR = _REPO / "first-test-csv-results"


def _telemetry(trial_index):
    return {"trial_index": trial_index, "task_k": 0, "task_code": "sz",
            "seg_id": 1000 + trial_index, "pattern_class_true": "seizure",
            "s_mean": 0.2, "s_sd": 0.1, "response_y": 1, "select_ms": 12.0,
            "expected_loss_chosen": 3.1, "total_var_t": 1.5, "total_var_l": 1.6,
            "total_var": 3.1, "auroc_mean": [0.8] * 6, "auroc_hw": [0.2] * 6,
            "max_hw": 0.2, "t_post_mean": [0.0] * 6, "l_post_mean": [0.4] * 6,
            "ess": 350.0, "rejuv": False}


def _gui(trial_index):
    return {"trial_index": trial_index, "seg_id": 1000 + trial_index,
            "task_k": 0, "response_raw": 0, "response_label": "Seizure",
            "reaction_time_ms": 800.0, "answer_changes": 0,
            "montage": "bipolar", "gain_uv": 70.0, "bandpass": "0.5-70 Hz",
            "notch": "60 Hz", "window_s": 10.0, "pan_t_start": 0.0,
            "interaction": []}


def _result(n=2):
    """SessionResult stub with a real (zero-filled) trajectory so finalize()
    enters the render branch but the renderer (if it were to run) would have
    a well-shaped trajectory.npz to read."""
    return SimpleNamespace(
        session_id="sess-render", stop_reason="all_resolved",
        delta_auroc=None, n_questions=n, task_codes=list(TASK_CODES),
        final_auroc_mean=np.full(6, 0.82), final_auroc_hw=np.full(6, 0.14),
        final_l_mean=np.full(6, 0.45), final_t_mean=np.zeros(6),
        served_seg_ids=list(range(1000, 1000 + n)),
        t_traj=np.zeros((n, 50, 6)), l_traj=np.zeros((n, 50, 6)),
        w_traj=np.full((n, 50), 1.0 / 50), aborted=False,
        verdicts=["PASS"] * 6,
        policy_diagnostics={"pi": [0.99] * 6, "mcse": [0.01] * 6,
                            "R": [0.7] * 6, "ess": 500.0})


# ── module imports cleanly ──────────────────────────────────────────────
def test_renderer_module_imports():
    import cortex_render_videos as cv
    assert hasattr(cv, "render_all")
    assert hasattr(cv, "render_collapse")
    assert hasattr(cv, "render_passfail")
    assert hasattr(cv, "main")


# ── render_videos=False suppresses rendering ────────────────────────────
def test_render_videos_false_skips_render(tmp_path):
    rec = cs.SessionRecorder(
        "sess-no-render", PARTICIPANT,
        {"policy": "AD6Policy", "n_particles": 50},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=False)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1), _gui(1))
    rec.finalize(_result(n=2))
    rec.close()
    assert not (rec.dir / "collapse.mp4").exists()
    assert not (rec.dir / "passfail.mp4").exists()
    # The other finalize artifacts must still be written
    assert (rec.dir / "certificate.json").exists()
    assert (rec.dir / "trajectory.npz").exists()


# ── renderer failure does not break finalize() ──────────────────────────
def test_render_failure_does_not_break_finalize(tmp_path, monkeypatch,
                                                 capsys):
    """If the renderer raises (missing ffmpeg, bad data, anything), the
    rest of finalize() must still complete and emit a warning. The other
    artifacts must still be produced."""
    import cortex_render_videos as cv

    def _boom(session_dir):
        raise RuntimeError("simulated render failure")
    monkeypatch.setattr(cv, "render_all", _boom)

    rec = cs.SessionRecorder(
        "sess-boom", PARTICIPANT,
        {"policy": "AD6Policy", "n_particles": 50},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=True)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1), _gui(1))
    rec.finalize(_result(n=2))
    rec.close()
    captured = capsys.readouterr()
    assert "video render failed" in captured.out
    # finalize() continued past the render block — certificate + npz still on disk
    assert (rec.dir / "certificate.json").exists()
    assert (rec.dir / "trajectory.npz").exists()
    # And no mp4s, since the renderer was forced to raise
    assert not (rec.dir / "collapse.mp4").exists()
    assert not (rec.dir / "passfail.mp4").exists()


# ── aborted sessions skip rendering even with render_videos=True ────────
def test_aborted_session_skips_render(tmp_path):
    rec = cs.SessionRecorder(
        "sess-aborted", PARTICIPANT,
        {"policy": "AD6Policy", "n_particles": 50},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=True)        # explicitly on; aborted gate should win
    rec.write_trial(_telemetry(0), _gui(0))
    result = _result(n=1)
    result.aborted = True
    rec.finalize(result)
    rec.close()
    assert not (rec.dir / "collapse.mp4").exists()
    assert not (rec.dir / "passfail.mp4").exists()


# ── slow: real Andrew session renders both videos ────────────────────────
@pytest.mark.slow
@pytest.mark.skipif(
    not (ANDREW_DIR / "trajectory.npz").exists()
    or not (ANDREW_DIR / "certificate.json").exists(),
    reason="reference Andrew session not present at first-test-csv-results/")
def test_render_andrew_session_end_to_end(tmp_path):
    """Copy the real Andrew session to a tmp dir (so we never write into
    the reference fixture) and confirm both MP4s render with non-zero
    file size. Takes ~15s on a developer machine; ffmpeg required."""
    import cortex_render_videos as cv
    sd = tmp_path / "andrew"
    sd.mkdir()
    for fname in ("trajectory.npz", "trials.jsonl", "events.jsonl",
                  "certificate.json", "participant.json"):
        shutil.copy2(ANDREW_DIR / fname, sd / fname)
    out = cv.render_all(sd)
    for kind in ("collapse", "passfail"):
        p = out[kind]
        assert p.exists(), f"{kind}.mp4 not produced"
        assert p.stat().st_size > 50_000, (
            f"{kind}.mp4 suspiciously small: {p.stat().st_size} bytes")
