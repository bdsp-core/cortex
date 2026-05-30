"""Tests for scripts/render_engine_explainer.py — the v1.1.3 third MP4.

Three layers:

  * Fast unit tests on the pure-math helpers (weighted mean/cov, ellipse,
    KDE grid, frame plan).
  * Fast integration: end-to-end render on a tiny synthetic session
    using the bundled imageio_ffmpeg binary. Verifies the MP4 is valid
    via ffprobe — guards against silent encoder regressions.
  * Cortex-finalize wiring: SessionRecorder.finalize() must invoke the
    explainer alongside cortex_render_videos.render_all, with an
    independent failure budget (a crash in one MUST NOT suppress the
    other).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import render_engine_explainer as ree   # noqa: E402
import cortex_render_videos as cv       # noqa: E402  (ffmpeg path)
import cortex_storage as cs             # noqa: E402

TASK_CODES = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
PARTICIPANT = {"name": "Explainer Tester", "email": "e@example.com",
               "expertise": "Other", "institution": "TEST"}


def _make_synthetic_session(sd: Path, *, T=4, N=80, K=6):
    """Build the four artifacts the explainer reads. Particle clouds
    are randn-with-shrink so each trial shows posterior contraction
    (more visually interesting; also lets us assert contraction)."""
    rng = np.random.default_rng(42)
    t_traj = np.zeros((T, N, K), dtype=np.float64)
    l_traj = np.zeros((T, N, K), dtype=np.float64)
    w_traj = np.full((T, N), 1.0 / N, dtype=np.float64)
    for j in range(T):
        shrink = 0.8 ** j
        t_traj[j] = rng.standard_normal((N, K)) * 0.5 * shrink
        l_traj[j] = 0.4 + rng.standard_normal((N, K)) * 0.6 * shrink
    np.savez(sd / "trajectory.npz",
             t_traj=t_traj, l_traj=l_traj, w_traj=w_traj,
             seg_ids=np.arange(T, dtype=np.int64),
             task_codes=np.array(TASK_CODES))
    (sd / "events.jsonl").write_text("")
    (sd / "trials.jsonl").write_text(
        "\n".join(json.dumps({
            "trial_index": i, "task_k": i % K,
            "task_code": TASK_CODES[i % K],          # v1.2.8 bottom-panel color
            "s_mean": float(np.sin(i)), "s_sd": 0.1,  # v1.2.8 bottom-panel y
            "auroc_hw": [0.2] * K,
            "policy_diag": {"pi": [0.5] * K, "mcse": [0.01] * K,
                            "R": [0.4] * K, "ess": 400.0},
            "verdicts": ["PENDING"] * K,
        }) for i in range(T)))
    (sd / "certificate.json").write_text(json.dumps({
        "session_id": "explainer-e2e", "stop_reason": "all_resolved",
        "task_codes": list(TASK_CODES),
        "per_task": [{"task": c, "verdict": "PASS"} for c in TASK_CODES],
        "n_questions": T,
    }))
    (sd / "participant.json").write_text(json.dumps({
        "session_id": "explainer-e2e",
        "identity": {"name": "Explainer Tester"},
    }))
    return t_traj, l_traj, w_traj


# ─── pure-math helpers ───────────────────────────────────────────────
def test_weighted_mean_cov_2d_recovers_known_distribution():
    rng = np.random.default_rng(0)
    n = 5000
    t = rng.standard_normal(n) * 0.4 + 0.10
    l = rng.standard_normal(n) * 0.7 - 0.20
    w = np.ones(n)
    mu, cov = ree._weighted_mean_cov_2d(t, l, w)
    assert abs(mu[0] - 0.10) < 0.05
    assert abs(mu[1] - (-0.20)) < 0.05
    assert abs(cov[0, 0] - 0.16) < 0.03
    assert abs(cov[1, 1] - 0.49) < 0.05
    assert abs(cov[0, 1]) < 0.05      # uncorrelated


def test_ellipse_xy_closes_and_centers_on_mean():
    mu = np.array([0.3, -0.2])
    cov = np.array([[0.25, 0.05], [0.05, 0.16]])
    ex, ey = ree._ellipse_xy(mu, cov, n_std=1.96, n_pts=128)
    assert ex.shape == (128,)
    assert abs(float(ex.mean()) - mu[0]) < 0.05
    assert abs(float(ey.mean()) - mu[1]) < 0.05
    # Closed loop — last ≈ first
    assert abs(ex[0] - ex[-1]) < 1e-6
    assert abs(ey[0] - ey[-1]) < 1e-6


def test_weighted_mean_cov_2d_handles_zero_weight_sum():
    """Defensive: a zero-weight cloud must not divide-by-zero."""
    mu, cov = ree._weighted_mean_cov_2d(np.zeros(3), np.zeros(3),
                                         np.zeros(3))
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)
    # No NaN / Inf — fallback returned zeros + small-eye
    assert np.isfinite(mu).all() and np.isfinite(cov).all()


def test_v1_1_5_default_timing_hits_60_seconds():
    """v1.1.5 spec: defaults must produce a ~60 sec video across the
    six IIIC tasks for a typical T≥60 session. The combo is FPS=10,
    TRIALS_PER_TASK=60, HOLD_SECONDS_PER_TASK=4.0 → per task block
    = 60 + 40 = 100 frames; total = 600 frames; duration = 60.0 s."""
    assert ree.FPS == 10
    assert ree.TRIALS_PER_TASK == 60
    assert ree.HOLD_SECONDS_PER_TASK == 4.0
    for T in (60, 100, 300):
        fpt, total, _ = ree._frame_plan(
            n_trials=T, n_tasks=6, fps=ree.FPS,
            trials_per_task=ree.TRIALS_PER_TASK,
            hold_seconds=ree.HOLD_SECONDS_PER_TASK)
        assert fpt == 100, f"T={T}: fpt should be 100, got {fpt}"
        assert total == 600, f"T={T}: total frames should be 600"
        assert abs(total / ree.FPS - 60.0) < 0.01, \
            f"T={T}: duration {total / ree.FPS:.2f} s, want 60.0"


def test_v1_1_5_short_sessions_proportionally_shorter():
    """Sessions shorter than TRIALS_PER_TASK render proportionally
    shorter videos rather than padding to 60 s — each trial is shown
    1:1 and the per-task hold stays constant."""
    fpt, total, _ = ree._frame_plan(
        n_trials=20, n_tasks=6, fps=ree.FPS,
        trials_per_task=ree.TRIALS_PER_TASK,
        hold_seconds=ree.HOLD_SECONDS_PER_TASK)
    assert fpt == 20 + 40                # all 20 trials + hold
    assert total == fpt * 6
    duration_s = total / ree.FPS
    assert 35.0 < duration_s < 40.0, \
        f"short session should be ~36 s, got {duration_s:.1f}"


def test_frame_plan_cycles_through_all_tasks():
    fpt, total, at = ree._frame_plan(n_trials=30, n_tasks=6,
                                      fps=24, trials_per_task=20,
                                      hold_seconds=0.5)
    assert total == fpt * 6
    # First frame is task 0; first frame of task 1 is at index fpt.
    task0, _, _ = at(0)
    task1, _, _ = at(fpt)
    task_last, _, _ = at(total - 1)
    assert task0 == 0
    assert task1 == 1
    assert task_last == 5


# ─── end-to-end render ───────────────────────────────────────────────
def test_render_engine_explainer_produces_valid_mp4(tmp_path):
    """Tiny session render — exercises the full pipeline through
    matplotlib + the bundled imageio_ffmpeg binary."""
    _make_synthetic_session(tmp_path, T=3, N=60)
    # Fast settings keep test time under ~10s on a dev box.
    out = ree.render_engine_explainer(
        tmp_path, fps=12, trials_per_task=2, hold_seconds=0.2)
    assert out.exists()
    sz = out.stat().st_size
    assert sz > 5_000, f"engine_explainer.mp4 suspiciously small: {sz}"
    # Verify the MP4 is structurally valid via the bundled ffmpeg.
    if cv.FFMPEG_EXE:
        r = subprocess.run(
            [cv.FFMPEG_EXE, "-v", "error", "-i", str(out),
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=20)
        assert r.returncode == 0, \
            f"engine_explainer.mp4 invalid: {r.stderr[:300]}"


def test_render_engine_explainer_zero_trials_is_noop(tmp_path):
    """Session with no trials → log a warning, return cleanly, no
    crash, no MP4 produced (or empty placeholder is acceptable)."""
    _make_synthetic_session(tmp_path, T=0, N=50)
    out = ree.render_engine_explainer(tmp_path, fps=12,
                                       trials_per_task=1)
    # Either no file or a zero-bytes placeholder — both honest.
    if out.exists():
        assert out.stat().st_size < 1_000


# ─── cortex_storage finalize integration ─────────────────────────────
def _telemetry(i, k=0):
    return {"trial_index": i, "task_k": k, "task_code": TASK_CODES[k],
            "seg_id": 1000 + i, "pattern_class_true": "seizure",
            "s_mean": 0.2, "s_sd": 0.1, "response_y": 1,
            "select_ms": 12.0, "expected_loss_chosen": 3.1,
            "total_var_t": 1.5, "total_var_l": 1.6, "total_var": 3.1,
            "auroc_mean": [0.8] * 6, "auroc_hw": [0.2] * 6,
            "max_hw": 0.2, "t_post_mean": [0.0] * 6,
            "l_post_mean": [0.4] * 6, "ess": 350.0, "rejuv": False}


def _gui(i):
    return {"trial_index": i, "seg_id": 1000 + i, "task_k": 0,
            "response_raw": 0, "response_label": "Seizure",
            "reaction_time_ms": 800.0, "answer_changes": 0,
            "montage": "bipolar", "gain_uv": 70.0,
            "bandpass": "0.5-70 Hz", "notch": "60 Hz",
            "window_s": 10.0, "pan_t_start": 0.0, "interaction": []}


def _result_with_traj(T=3, N=50):
    rng = np.random.default_rng(7)
    return SimpleNamespace(
        session_id="finalize-test", stop_reason="all_resolved",
        delta_auroc=None, n_questions=T, task_codes=list(TASK_CODES),
        final_auroc_mean=np.full(6, 0.82), final_auroc_hw=np.full(6, 0.14),
        final_l_mean=np.full(6, 0.45), final_t_mean=np.zeros(6),
        served_seg_ids=list(range(1000, 1000 + T)),
        t_traj=rng.standard_normal((T, N, 6)) * 0.5,
        l_traj=0.4 + rng.standard_normal((T, N, 6)) * 0.6,
        w_traj=np.full((T, N), 1.0 / N),
        aborted=False,
        verdicts=["PASS"] * 6,
        policy_diagnostics={"pi": [0.99] * 6, "mcse": [0.01] * 6,
                            "R": [0.7] * 6, "ess": 500.0})


def test_finalize_runs_engine_explainer_alongside_render_all(
        tmp_path, monkeypatch):
    """SessionRecorder.finalize() must invoke render_engine_explainer
    after cortex_render_videos.render_all. Monkey-patched both renderers
    to record calls — the actual MP4 production is exercised by the
    end-to-end test above."""
    calls = []
    import cortex_render_videos as cv_mod
    import render_engine_explainer as ree_mod
    monkeypatch.setattr(cv_mod, "render_all",
                        lambda d, **kw: calls.append(("render_all", Path(d))))
    monkeypatch.setattr(
        ree_mod, "render_engine_explainer",
        lambda d, **kw: calls.append(("render_engine_explainer",
                                       Path(d))))
    rec = cs.SessionRecorder(
        "finalize-test", PARTICIPANT,
        {"policy": "AD6Policy", "n_particles": 50},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=True)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1), _gui(1))
    rec.finalize(_result_with_traj(T=2, N=40))
    rec.close()
    names = [c[0] for c in calls]
    assert "render_all" in names
    assert "render_engine_explainer" in names


def test_finalize_render_failures_are_independent(tmp_path, monkeypatch):
    """render_all crashing MUST NOT suppress render_engine_explainer
    (and vice versa). Each renderer gets its own try/except budget."""
    import cortex_render_videos as cv_mod
    import render_engine_explainer as ree_mod
    explainer_called = []
    monkeypatch.setattr(cv_mod, "render_all",
                        lambda d, **kw: (_ for _ in ()).throw(
                            RuntimeError("cv boom")))
    monkeypatch.setattr(
        ree_mod, "render_engine_explainer",
        lambda d, **kw: explainer_called.append(Path(d)))
    rec = cs.SessionRecorder(
        "indep-test", PARTICIPANT,
        {"policy": "AD6Policy", "n_particles": 50},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=True)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_result_with_traj(T=1, N=40))
    rec.close()
    assert explainer_called, (
        "render_engine_explainer was not called when render_all raised; "
        "the failure budgets are NOT independent.")


def test_finalize_skips_explainer_for_aborted_session(
        tmp_path, monkeypatch):
    """Aborted session → neither renderer runs (the partial trajectory
    isn't worth the render time)."""
    import render_engine_explainer as ree_mod
    explainer_called = []
    monkeypatch.setattr(
        ree_mod, "render_engine_explainer",
        lambda d, **kw: explainer_called.append(Path(d)))
    rec = cs.SessionRecorder(
        "aborted-test", PARTICIPANT,
        {"policy": "AD6Policy", "n_particles": 50},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=True)
    rec.write_trial(_telemetry(0), _gui(0))
    aborted = _result_with_traj(T=1, N=40)
    aborted.aborted = True
    rec.finalize(aborted)
    rec.close()
    assert not explainer_called, (
        "engine_explainer ran for an aborted session — should have "
        "been skipped (matches cortex_render_videos.render_all behavior).")


# ─── v1.2.8: bottom panel = questions-vs-signal-strength by domain ────
def test_v1_2_8_explainer_drops_expected_loss_curve():
    """The bottom panel no longer plots the expected-posterior-variance
    score curve (and no longer imports the engine to do so)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "render_engine_explainer.py").read_text()
    assert "_expected_loss_vec" not in src
    assert "expected posterior variance" not in src
    assert "signal strength by domain" in src


def test_v1_2_8_domain_palette_covers_seven_tasks():
    assert set(ree.DOMAIN_COLORS) == {
        "spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"}
    assert ree.DOMAIN_TITLES.get("spike") == "Spike"   # was missing pre-1.2.8


def test_v1_2_8_load_session_parses_questions(tmp_path):
    _make_synthetic_session(tmp_path, T=6, N=40, K=6)
    sess = ree._load_session(tmp_path)
    qs = sess["questions"]
    assert len(qs) == 6
    assert set(qs[0]) == {"idx", "s", "domain"}
    assert qs[0]["idx"] == 0
    assert qs[0]["domain"] in ree.DOMAIN_COLORS


def test_v1_2_8_explainer_title_drops_participant_name():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "render_engine_explainer.py").read_text()
    assert "sess['participant_name']" not in src


def test_render_engine_explainer_forwards_progress_callback(tmp_path):
    """v1.2.9: the renderer must forward matplotlib's per-frame
    progress_callback(current, total) so finalize can build a real ETA."""
    _make_synthetic_session(tmp_path, T=3, N=40)
    seen = []
    ree.render_engine_explainer(
        tmp_path, fps=12, trials_per_task=2, hold_seconds=0.2,
        progress_callback=lambda i, n: seen.append((i, n)))
    assert seen, "progress_callback was never called"
    i_last, n_last = seen[-1]
    assert n_last > 0 and 0 <= i_last <= n_last
