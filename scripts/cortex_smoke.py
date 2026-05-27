"""CORTEX self-driving smoke test.

Launches the real CORTEX GUI on your display and auto-drives a short
session end to end — Landing -> Consent -> Registration -> tutorial ->
a capped engine-driven question run -> the results screen — so you can
watch the whole flow once before distributing the test.

Run:
    .venv/bin/python scripts/cortex_smoke.py

It answers on its own; just watch. Close the window when it finishes.
This is a developer tool, not part of the test suite — it opens a real
window and writes a real session under results/sessions/.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "scripts"))

from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QFont  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import eeg_bank_viewer as ev  # noqa: E402
from cortex_engine_inputs import build_iiic_engine_inputs  # noqa: E402
from session_controller import SessionController  # noqa: E402
from cortex_storage import SessionRecorder  # noqa: E402

STEP_MS = 1600          # pause between auto-advanced intro / tutorial screens
N_QUESTIONS = 8         # capped session length for the smoke


def _banner(msg):
    print(f"  [smoke] {msg}", flush=True)


def main():
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_MacDontSwapCtrlAndMeta, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Palatino"))
    state = {}

    def goto(widget):
        prev = state.get("page")
        if prev is not None:
            widget.setGeometry(prev.geometry())
        state["page"] = widget
        widget.show()
        if prev is not None and prev is not widget:
            prev.close()

    def drive_tutorial(win):
        ov = win._overlay
        if ov is None or not win._tutorial_active:
            return                       # tutorial finished — engine takes over
        ov._advance()
        QTimer.singleShot(STEP_MS, lambda: drive_tutorial(win))

    def open_viewer(reg):
        _banner("building engine + viewer")
        inputs = build_iiic_engine_inputs()
        tutorial_sid = inputs.all_seg_ids[0]
        engine_inputs = inputs.without([tutorial_sid])
        controller = SessionController(
            engine_inputs, reg.session_id, capture_clouds=True,
            max_questions=N_QUESTIONS, n_particles=300)
        recorder = SessionRecorder(
            reg.session_id, reg.registration,
            {"n_iiic_segments": len(engine_inputs.all_seg_ids), "smoke": True})
        win = ev.BankViewer(controller, reg.session_id, tutorial_sid, recorder)
        app.installEventFilter(win)
        win.show()
        reg.close()
        state["page"] = win

        def respond(item):
            def _answer():
                if win._awaiting_answer:
                    win._select_answer(item["task_k"])
                    QTimer.singleShot(450, win._confirm_answer)
            QTimer.singleShot(650, _answer)

        controller.itemReady.connect(respond)
        controller.sessionComplete.connect(
            lambda r: _banner(f"session complete — {r.n_questions} questions, "
                              f"stop={r.stop_reason}"))
        QTimer.singleShot(STEP_MS, lambda: drive_tutorial(win))
        app.processEvents()
        win.start_tutorial(tutorial_sid)
        _banner("tutorial running — engine session follows")

    def open_registration():
        reg = ev.RegistrationPage()
        # Page 1 — identity + eligibility (all required to advance).
        reg.f_name.setText("Smoke Tester")
        reg.f_email.setText("smoke@example.com")
        reg.f_eligibility.setChecked(True)
        # Page 2 — clinical background (4 dropdowns required to advance).
        reg.f_expertise.setCurrentIndex(4)            # "Fellow"
        reg.f_practice.setCurrentIndex(1)             # Academic medical center
        reg.f_years_eeg.setCurrentIndex(1)            # "0–4"
        reg.f_eeg_volume.setCurrentIndex(2)           # "5–20"

        def go():
            if reg.commit():
                open_viewer(reg)

        QTimer.singleShot(STEP_MS, go)
        goto(reg)

    def open_consent():
        consent = ev.ConsentPage()
        consent.accept_btn.clicked.connect(open_registration)
        QTimer.singleShot(STEP_MS, consent.accept_btn.click)
        goto(consent)

    landing = ev.LandingPage()
    landing.begin_btn.clicked.connect(open_consent)
    QTimer.singleShot(STEP_MS, landing.begin_btn.click)
    goto(landing)
    _banner("smoke started — watch the window")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
