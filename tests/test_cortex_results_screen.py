"""Phase E — offscreen tests for the CORTEX results screen.

ResultsScreen rendering and the BankViewer._on_session_complete swap.
A FakeController + a stub SessionResult stand in for the engine.
Skipped when data/eeg_bank.h5 is absent (gitignored — D9).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import cortex_engine_inputs as cei  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(cei.BANK_PATH),
    reason=f"eeg_bank.h5 not present at {cei.BANK_PATH}")

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402
import eeg_bank_viewer as ev  # noqa: E402

TASK_CODES = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]


class FakeController(QObject):
    itemReady = pyqtSignal(dict)
    trialDone = pyqtSignal(dict)
    sessionComplete = pyqtSignal(object)
    sessionFailed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.submitted = []

    def start(self):
        pass

    def submit_answer(self, c):
        self.submitted.append(int(c))

    def abort(self):
        pass


def _result(aborted=False, n=42, stop="delta_reached"):
    """v1.1.2 ResultsScreen reads verdicts + ℓ̂/θ̂/π rather than AUROC,
    so the stub carries the AD6-policy outputs too."""
    return SimpleNamespace(
        task_codes=list(TASK_CODES), n_questions=n, stop_reason=stop,
        delta_auroc=0.15, aborted=aborted,
        final_auroc_mean=np.array([0.91, 0.84, 0.78, 0.66, 0.72, 0.80]),
        final_auroc_hw=np.array([0.10, 0.12, 0.14, 0.13, 0.11, 0.12]),
        final_t_mean=np.zeros(6), final_l_mean=np.full(6, 0.4),
        verdicts=["PASS"] * 6,
        policy_diagnostics={"pi": [0.97] * 6, "mcse": [0.01] * 6,
                            "R": [0.7] * 6, "ess": 500.0})


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def iiic_segs():
    with h5py.File(cei.BANK_PATH, "r") as f:
        return sorted(int(x) for x in f["iiic"])


def _labels(widget):
    return [w.text() for w in widget.findChildren(QLabel)]


def test_results_screen_builds(qapp, monkeypatch):
    """v1.1.2 surface: per-task PASS/FAIL/REFER verdict with
    threshold-relative skill narrative; no agreement-with-reference
    line; no AUROC point estimates on the main panel (still in the
    summary CSV for analysis)."""
    # Stub the ℓ* loader so the test doesn't require a cert_config on disk.
    import cortex_policy as cp
    monkeypatch.setattr(cp, "load_ell_star_iiic",
                        lambda codes, **kw: [0.42] * len(codes))
    screen = ev.ResultsScreen(_result(), n_correct=30, n_answered=42)
    try:
        texts = _labels(screen)
        assert any("Assessment Complete" in t for t in texts)
        assert any("42 recordings reviewed" in t for t in texts)
        for lbl in ("Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"):
            assert lbl in texts
        # v1.1.2 contract: agreement / accuracy line is gone, AUROC
        # number is no longer on the main panel.
        assert not any("Agreement with the reference label" in t
                       for t in texts), (
            "v1.1.2 removed the agreement-with-reference-label line "
            "from the results screen (still in the summary CSV).")
        assert "0.91" not in texts, (
            "v1.1.2 removed the AUROC point-estimate column from the "
            "main panel — replaced with PASS/FAIL/REFER verdicts.")
        # v1.1.2 additions: clinician-friendly verdict label appears
        assert any("Pass" in t for t in texts)
        assert screen.close_btn.text() == "CLOSE"
        # Details panel is collapsed by default
        assert screen._details_panel.isHidden() is True
        assert "Show technical details" in screen.details_btn.text()
    finally:
        screen.deleteLater()
        qapp.processEvents()


def test_results_screen_handles_no_answers(qapp):
    # n_answered=0 must not divide-by-zero
    screen = ev.ResultsScreen(_result(n=0, stop="bank_exhausted"),
                              n_correct=0, n_answered=0)
    screen.deleteLater()
    qapp.processEvents()


def test_session_complete_swaps_to_results(qapp, iiic_segs):
    fake = FakeController()
    win = ev.BankViewer(fake, "test-results", iiic_segs[0])
    try:
        fake.sessionComplete.emit(_result(aborted=False))
        qapp.processEvents()
        assert win._session_over is True
        assert isinstance(win.centralWidget(), ev.ResultsScreen)
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_aborted_session_no_results_screen(qapp, iiic_segs):
    fake = FakeController()
    win = ev.BankViewer(fake, "test-abort", iiic_segs[0])
    try:
        central_before = win.centralWidget()
        fake.sessionComplete.emit(_result(aborted=True))
        qapp.processEvents()
        assert win._session_over is True
        assert win.centralWidget() is central_before   # NOT swapped
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_on_trial_done_tallies_accuracy(qapp, iiic_segs):
    fake = FakeController()
    win = ev.BankViewer(fake, "test-acc", iiic_segs[0])
    try:
        win.gui_trial_log = [
            {"trial_index": 0, "response_label": "Seizure"},
            {"trial_index": 1, "response_label": "LPD"},
        ]
        win._on_trial_done({"trial_index": 0, "pattern_class_true": "seizure"})
        win._on_trial_done({"trial_index": 1, "pattern_class_true": "gpd"})
        assert win._n_answered == 2
        assert win._n_correct == 1          # Seizure==seizure; LPD!=gpd
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()
