"""Phase C — offscreen tests for the engine-wired CORTEX viewer.

Exercises select-then-confirm, reaction-time capture, the interaction
trace, the _awaiting_answer guard, engine-driven show_item, and the
registration session_id. A FakeController stands in for the real engine
thread so the viewer is tested in isolation. Skipped when
data/eeg_bank.h5 is absent (gitignored — D9).
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import h5py  # noqa: E402
import cortex_engine_inputs as cei  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(cei.BANK_PATH),
    reason=f"eeg_bank.h5 not present at {cei.BANK_PATH}")

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
import eeg_bank_viewer as ev  # noqa: E402


class FakeController(QObject):
    """Stands in for SessionController — same signal surface, records calls."""

    itemReady = pyqtSignal(dict)
    trialDone = pyqtSignal(dict)
    sessionComplete = pyqtSignal(object)
    sessionFailed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.submitted = []
        self.started = False
        self.aborted = False

    def start(self):
        self.started = True

    def submit_answer(self, raw_choice):
        self.submitted.append(int(raw_choice))

    def abort(self):
        self.aborted = True


@pytest.fixture(scope="session")
def qapp():
    """One QApplication held for the whole test session (pytest-qt pattern)."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def iiic_segs():
    with h5py.File(cei.BANK_PATH, "r") as f:
        return sorted(int(x) for x in f["iiic"])


@pytest.fixture
def viewer(qapp, iiic_segs):
    fake = FakeController()
    win = ev.BankViewer(fake, "test-session", iiic_segs[0])
    yield qapp, fake, win, iiic_segs
    win.close()
    win.deleteLater()
    qapp.processEvents()
    qapp.processEvents()


def _show(app, win, seg, trial=0, k=2):
    win.show_item({"trial_index": trial, "task_k": k, "task_code": "gpd",
                   "seg_id": seg, "s_mean": 0.1, "s_sd": 0.1})
    app.processEvents()      # fire the post-paint RT-clock QTimer
    app.processEvents()


def test_show_item_arms_panel(viewer):
    app, fake, win, segs = viewer
    _show(app, win, segs[1])
    assert win._awaiting_answer is True
    assert win._cur_seg == segs[1]
    assert win._rt_t0 is not None
    assert win.confirm_btn.isEnabled() is False


def test_select_then_confirm(viewer):
    app, fake, win, segs = viewer
    _show(app, win, segs[1])
    win._select_answer(0)
    assert win._selected_choice == 0
    assert win.confirm_btn.isEnabled() is True
    win._select_answer(3)                       # change the choice
    assert win._selected_choice == 3
    assert win._answer_changes == 1
    win._confirm_answer()
    assert fake.submitted == [3]
    assert len(win.gui_trial_log) == 1
    rec = win.gui_trial_log[0]
    assert rec["response_raw"] == 3
    assert rec["response_label"] == "LRDA"
    assert rec["answer_changes"] == 1
    assert isinstance(rec["reaction_time_ms"], float)
    assert rec["reaction_time_ms"] >= 0.0
    assert rec["seg_id"] == segs[1]
    assert rec["trial_index"] == 0


def test_awaiting_answer_guard(viewer):
    app, fake, win, segs = viewer
    _show(app, win, segs[1])
    win._select_answer(2)
    win._confirm_answer()
    assert not win._awaiting_answer
    # input after confirm is ignored — exactly one record, one submit
    win._select_answer(5)
    win._confirm_answer()
    assert len(win.gui_trial_log) == 1
    assert fake.submitted == [2]


def test_interaction_trace_records_actions(viewer):
    app, fake, win, segs = viewer
    _show(app, win, segs[1])
    win._pan(+1)
    win._select_answer(1)
    win._confirm_answer()
    actions = [e["action"] for e in win.gui_trial_log[0]["interaction"]]
    assert "pan" in actions
    assert "select" in actions
    for ev_rec in win.gui_trial_log[0]["interaction"]:
        assert ev_rec["t_ms"] >= 0.0


def test_session_complete_sets_result(viewer):
    app, fake, win, segs = viewer
    _show(app, win, segs[1])
    sentinel = object()
    fake.sessionComplete.emit(sentinel)
    app.processEvents()
    assert win.session_result is sentinel
    assert not win._awaiting_answer


def test_registration_generates_session_id(qapp, monkeypatch):
    captured = {}
    monkeypatch.setattr(ev.RegistrationPage, "_save_row",
                        staticmethod(lambda row: captured.update(row)))
    reg = ev.RegistrationPage()
    try:
        reg.f_name.setText("Test User")
        reg.f_email.setText("test@example.com")
        assert reg.commit() is True
        assert isinstance(reg.session_id, str) and len(reg.session_id) >= 32
        assert captured["session_id"] == reg.session_id
    finally:
        reg.deleteLater()
        qapp.processEvents()
