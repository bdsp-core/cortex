"""Phase F — offscreen end-to-end test of the full CORTEX pipeline.

Drives a complete scripted IIIC session through the REAL stack —
SessionController + EngineWorker QThread + queue-backed y_source +
BankViewer + SessionRecorder + ResultsScreen — and asserts every output
artifact. This is the first test exercising all the pieces together;
Phases A-E test them in isolation or with fakes. Skipped when
data/eeg_bank.h5 is absent (gitignored — D9).
"""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402
import cortex_engine_inputs as cei  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(cei.BANK_PATH),
    reason=f"eeg_bank.h5 not present at {cei.BANK_PATH}")

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
import eeg_bank_viewer as ev  # noqa: E402
from session_controller import SessionController  # noqa: E402
from cortex_storage import SessionRecorder  # noqa: E402

N_QUESTIONS = 12


def test_full_pipeline_end_to_end(tmp_path):
    app = QApplication.instance() or QApplication([])

    inputs = cei.build_iiic_engine_inputs()
    tutorial_sid = inputs.all_seg_ids[0]
    engine_inputs = inputs.without([tutorial_sid])

    # capped, small-N, delta=0 -> a deterministic N_QUESTIONS-trial run
    controller = SessionController(
        engine_inputs, "e2e-session", capture_clouds=True,
        max_questions=N_QUESTIONS, n_particles=200, delta_auroc=0.0)
    recorder = SessionRecorder(
        "e2e-session", {"name": "E2E Tester", "expertise": "Fellow"},
        {"delta_auroc": 0.0, "n_particles": 200},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced", dropbox_cfg=None,
        render_videos=False)         # ffmpeg adds 15s; e2e test stays fast
    win = ev.BankViewer(controller, "e2e-session", tutorial_sid, recorder)

    done = {}
    controller.sessionComplete.connect(lambda r: done.setdefault("result", r))

    def respond(item):
        # Defer the answer with our own 0-timer so the post-paint RT-clock
        # QTimer (scheduled by show_item) fires first — otherwise _rt_t0 is
        # still None and the interaction trace would be dropped.
        def _answer():
            if win._awaiting_answer:
                win._pan(+1)
                win._select_answer(0)
                win._confirm_answer()
        QTimer.singleShot(0, _answer)

    controller.itemReady.connect(respond)
    controller.start()

    deadline = time.time() + 60.0
    while "result" not in done and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)

    assert "result" in done, "session did not complete within 60 s"
    result = done["result"]
    assert result.n_questions == N_QUESTIONS
    assert not result.aborted

    app.processEvents()
    assert isinstance(win.centralWidget(), ev.ResultsScreen)

    # every artifact present + well-formed
    sdir = recorder.dir
    trials = (sdir / "trials.jsonl").read_text().strip().splitlines()
    assert len(trials) == N_QUESTIONS
    rec0 = json.loads(trials[0])
    assert "response_y" in rec0 and "max_hw" in rec0       # engine telemetry
    assert "reaction_time_ms" in rec0                       # GUI metadata
    assert "is_correct" in rec0                             # merged field

    events = (sdir / "events.jsonl").read_text().strip().splitlines()
    assert len(events) >= N_QUESTIONS                       # >=1 event/trial

    assert (sdir / "participant.json").exists()
    cert = json.loads((sdir / "certificate.json").read_text())
    assert cert["total_trials"] == N_QUESTIONS
    assert len(cert["per_task"]) == 6

    npz = np.load(sdir / "trajectory.npz")
    assert npz["t_traj"].shape[0] == N_QUESTIONS

    synced = tmp_path / "synced"
    assert len(list(synced.glob("*_trials.csv"))) == 1
    assert len(list(synced.glob("*_summary.csv"))) == 1

    win.close()
    win.deleteLater()
    app.processEvents()
