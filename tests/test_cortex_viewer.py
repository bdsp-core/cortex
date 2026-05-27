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


def _fill_required(reg):
    """Set the minimum fields required to pass the v1.1.1 wizard
    validation. Page 3 (demographics) is fully optional."""
    reg.f_name.setText("Test User")
    reg.f_email.setText("test@example.com")
    reg.f_eligibility.setChecked(True)
    reg.f_expertise.setCurrentIndex(4)      # "Fellow"
    reg.f_practice.setCurrentIndex(1)       # "Academic medical center"
    reg.f_years_eeg.setCurrentIndex(1)      # "0–4"
    reg.f_eeg_volume.setCurrentIndex(2)     # "5–20"


def test_registration_generates_session_id(qapp, monkeypatch):
    captured = {}
    monkeypatch.setattr(ev.RegistrationPage, "_save_row",
                        staticmethod(lambda row: captured.update(row)))
    reg = ev.RegistrationPage()
    try:
        _fill_required(reg)
        assert reg.commit() is True
        assert isinstance(reg.session_id, str) and len(reg.session_id) >= 32
        assert captured["session_id"] == reg.session_id
        # All v1.1.3 schema columns must be present in the persisted row.
        for col in ev.REGISTRATION_FIELDS:
            assert col in captured, f"missing {col} in registration row"
        # gender_identity was removed in v1.1.3 — must NOT appear.
        assert "gender_identity" not in captured
    finally:
        reg.deleteLater()
        qapp.processEvents()


def test_registration_blocks_without_eligibility(qapp, monkeypatch):
    """The eligibility checkbox is a hard gate on page 1 — commit must
    fail (and surface a page-1 error) until the user opts in."""
    monkeypatch.setattr(ev.RegistrationPage, "_save_row",
                        staticmethod(lambda row: None))
    reg = ev.RegistrationPage()
    try:
        _fill_required(reg)
        reg.f_eligibility.setChecked(False)
        assert reg.commit() is False
        # commit should snap the wizard back to page 1 with an error
        assert reg._stack.currentIndex() == 0
        assert "healthcare professional" in reg._msg1.text().lower()
    finally:
        reg.deleteLater()
        qapp.processEvents()


def test_registration_requires_sex_dropdowns(qapp, monkeypatch):
    """Page-2 required dropdowns (sex, years EEG, EEG volume, practice
    setting, expertise) must be set or commit fails."""
    monkeypatch.setattr(ev.RegistrationPage, "_save_row",
                        staticmethod(lambda row: None))
    for missing_field in ("f_expertise", "f_practice",
                          "f_years_eeg", "f_eeg_volume"):
        reg = ev.RegistrationPage()
        try:
            _fill_required(reg)
            getattr(reg, missing_field).setCurrentIndex(0)
            assert reg.commit() is False, \
                f"commit should fail when {missing_field} is unset"
            assert reg._stack.currentIndex() == 1
        finally:
            reg.deleteLater()
            qapp.processEvents()


def test_registration_csv_schema_v3(qapp, tmp_path, monkeypatch):
    """The persisted CSV header must equal REGISTRATION_FIELDS (v3) exactly
    (downstream analysis joins on these column names). v1.1.3 dropped
    gender_identity — confirm it's absent."""
    import cortex_storage as cs
    monkeypatch.setattr(cs, "user_data_root", lambda: tmp_path)

    reg = ev.RegistrationPage()
    try:
        _fill_required(reg)
        # Pick non-default page-3 answers so we exercise the optional cols.
        reg.f_sex.setCurrentIndex(1)            # "Male"
        reg.f_country.setCurrentIndex(1)        # "United States"
        reg.f_race.setCurrentIndex(7)           # "White"
        # v1.1.3: f_gender no longer exists on the page.
        assert not hasattr(reg, "f_gender"), \
            "f_gender widget should be gone in v1.1.3"
        assert reg.commit() is True
    finally:
        reg.deleteLater()
        qapp.processEvents()

    import csv as _csv
    with open(tmp_path / "registrations.csv", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
        fh.seek(0)
        header = next(_csv.reader(fh))
    assert header == ev.REGISTRATION_FIELDS
    assert "gender_identity" not in header
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "Test User"
    assert r["sex"] == "Male"
    assert r["race_ethnicity"] == "White"
    assert r["consent_version"] == ev.CONSENT_VERSION
    assert r["eligibility_confirmed"] == "yes"


def test_registration_schema_migration_rotates_v1_csv(
        qapp, tmp_path, monkeypatch):
    """A v1 (pre-v1.1.1) header on disk must be rotated to
    registrations.v1.csv.bak — not have v3 rows appended to it."""
    import cortex_storage as cs
    monkeypatch.setattr(cs, "user_data_root", lambda: tmp_path)

    legacy_path = tmp_path / "registrations.csv"
    with open(legacy_path, "w", encoding="utf-8", newline="") as fh:
        import csv as _csv
        w = _csv.writer(fh)
        w.writerow(ev._REGISTRATION_FIELDS_V1)
        w.writerow(["old-sid", "2020-01-01T00:00:00+00:00",
                    "Old User", "30", "M", "Old Inst",
                    "old@example.com", "Fellow", "ABPN"])

    reg = ev.RegistrationPage()
    try:
        _fill_required(reg)
        assert reg.commit() is True
    finally:
        reg.deleteLater()
        qapp.processEvents()

    import csv as _csv
    with open(legacy_path, encoding="utf-8") as fh:
        header = next(_csv.reader(fh))
        rows = list(_csv.DictReader(open(legacy_path, encoding="utf-8")))
    assert header == ev.REGISTRATION_FIELDS
    assert len(rows) == 1
    assert rows[0]["name"] == "Test User"
    bak = tmp_path / "registrations.v1.csv.bak"
    assert bak.exists()
    with open(bak, encoding="utf-8") as fh:
        assert next(_csv.reader(fh)) == ev._REGISTRATION_FIELDS_V1


def test_registration_schema_migration_rotates_v2_csv(
        qapp, tmp_path, monkeypatch):
    """A v2 (v1.1.1 / v1.1.2) header on disk must be rotated to
    registrations.v2.csv.bak — new for v1.1.3 since the upgrade path
    now spans more than one schema bump."""
    import cortex_storage as cs
    monkeypatch.setattr(cs, "user_data_root", lambda: tmp_path)

    legacy_path = tmp_path / "registrations.csv"
    with open(legacy_path, "w", encoding="utf-8", newline="") as fh:
        import csv as _csv
        w = _csv.writer(fh)
        w.writerow(ev._REGISTRATION_FIELDS_V2)
        w.writerow(["old-sid", "2026-05-26T00:00:00+00:00",
                    "v1.1.1-placeholder", "", "yes",
                    "Old User", "33", "old@example.com", "MGH",
                    "Fellow", "Academic medical center", "10–14", "21–50",
                    "5", "Normal color vision", "No",
                    "Female", "Woman", "United States", "White"])

    reg = ev.RegistrationPage()
    try:
        _fill_required(reg)
        assert reg.commit() is True
    finally:
        reg.deleteLater()
        qapp.processEvents()

    import csv as _csv
    with open(legacy_path, encoding="utf-8") as fh:
        header = next(_csv.reader(fh))
    assert header == ev.REGISTRATION_FIELDS
    bak = tmp_path / "registrations.v2.csv.bak"
    assert bak.exists(), \
        "v2-header CSV must rotate to .v2.csv.bak (not .v1 or .legacy)"
    with open(bak, encoding="utf-8") as fh:
        assert next(_csv.reader(fh)) == ev._REGISTRATION_FIELDS_V2


# ─────── v1.1.2: ResultsScreen — verdicts, narrative, details panel ───────

def _synthetic_result(verdicts=None, l_mean=None, t_mean=None, pi=None):
    """SessionResult stub with K=6 IIIC tasks + optional verdicts + π."""
    from types import SimpleNamespace
    import numpy as np
    codes = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
    return SimpleNamespace(
        task_codes=codes,
        n_questions=42,
        stop_reason="all_resolved",
        final_auroc_mean=np.full(6, 0.82),
        final_auroc_hw=np.full(6, 0.10),
        final_l_mean=np.array(l_mean if l_mean is not None
                              else [0.85, 0.45, 0.30, 0.60, 0.42, 0.70]),
        final_t_mean=np.array(t_mean if t_mean is not None
                              else [0.05, -0.20, 0.35, 0.00, -0.05, 0.15]),
        verdicts=verdicts if verdicts is not None else [
            "PASS", "REFER_BORDERLINE", "FAIL",
            "PASS", "REFER_UNINFORMATIVE", "PASS"],
        policy_diagnostics={"pi": pi if pi is not None
                            else [0.98, 0.55, 0.02, 0.96, 0.40, 0.97],
                            "mcse": [0.01] * 6, "R": [0.7] * 6,
                            "ess": 500.0},
        delta_auroc=None,
        seed=42, selection="adaptive", n_particles=200, trials=[],
        served_seg_ids=[], aborted=False)


def test_results_screen_skill_narrative_function():
    """Pure-function check — the skill narrative branches on (ℓ̂, ℓ*)
    with a 0.05 'near threshold' band."""
    f = ev.ResultsScreen._skill_narrative
    text, _ = f(0.85, 0.42)
    assert "Above threshold by 0.43" == text
    text, _ = f(0.21, 0.42)
    assert "Below threshold by 0.21" == text
    text, _ = f(0.44, 0.42)
    assert "Near the passing threshold" == text
    text, _ = f(None, 0.42)
    assert "not estimated" in text.lower()
    text, _ = f(0.5, None)            # threshold load failed
    assert "ℓ̂ = +0.50" in text


def test_results_screen_bias_narrative_function():
    """Pure-function check — bias narrative branches on |θ̂| with
    direction inverted (negative θ̂ = liberal/over-calls)."""
    f = ev.ResultsScreen._bias_narrative
    text, _ = f(0.05)
    assert "near neutral" in text.lower()
    text, _ = f(0.15)
    assert "slight" in text.lower() and "conservative" in text.lower()
    text, _ = f(0.40)
    assert "strong" in text.lower() and "conservative" in text.lower()
    text, _ = f(-0.15)
    assert "slight" in text.lower() and "liberal" in text.lower()
    text, _ = f(-0.40)
    assert "strong" in text.lower() and "liberal" in text.lower()


def test_results_screen_renders_verdicts(qapp, monkeypatch):
    """ResultsScreen builds the per-task verdict table with clinician-
    friendly labels; the details panel starts hidden."""
    # Stub load_ell_star_iiic so the test doesn't need cert_config on disk.
    import cortex_policy as cp
    monkeypatch.setattr(cp, "load_ell_star_iiic",
                        lambda codes, **kw: [0.42] * len(codes))
    result = _synthetic_result()
    screen = ev.ResultsScreen(result, n_correct=20, n_answered=42)
    try:
        # Verdict labels appear somewhere in the screen's child QLabels
        all_text = " | ".join(
            w.text() for w in screen.findChildren(__import__(
                "PyQt6.QtWidgets", fromlist=["QLabel"]).QLabel))
        assert "Pass" in all_text
        assert "Did not pass" in all_text
        assert "Refer (borderline)" in all_text
        assert "Refer (need more data)" in all_text
        # Threshold-relative skill narratives
        assert "Above threshold by 0.43" in all_text   # sz: 0.85 vs 0.42
        assert "Below threshold by 0.12" in all_text   # gpd: 0.30 vs 0.42
        # Bias narratives
        assert "Strong conservative" in all_text       # gpd: θ=+0.35
        # The agreement-with-reference-label line is GONE
        assert "Agreement with" not in all_text
        # Details panel starts hidden, toggles on, toggles off. Use
        # isHidden() which reflects the explicit setVisible() state
        # regardless of whether the parent window has been shown
        # (Qt's isVisible() returns False until the top-level window
        # is on screen — which is why _toggle_details internally uses
        # isHidden() too, otherwise a second click would never hide).
        assert screen._details_panel.isHidden() is True
        assert "Show" in screen.details_btn.text()
        screen._toggle_details()
        assert screen._details_panel.isHidden() is False
        assert "Hide" in screen.details_btn.text()
        screen._toggle_details()
        assert screen._details_panel.isHidden() is True
        assert "Show" in screen.details_btn.text()
        # Raw posterior numbers are present once shown
        all_text_now = " | ".join(
            w.text() for w in screen.findChildren(__import__(
                "PyQt6.QtWidgets", fromlist=["QLabel"]).QLabel))
        assert "0.850" in all_text_now                  # ℓ̂ for sz
        assert "0.420" in all_text_now                  # ℓ* for any row
        assert "+0.350" in all_text_now or "+0.35" in all_text_now  # θ̂ for gpd
        assert "0.980" in all_text_now                  # π for sz
    finally:
        screen.deleteLater()
        qapp.processEvents()


def test_results_screen_handles_missing_threshold(qapp, monkeypatch):
    """If cert_config.yaml is missing / malformed, load_ell_star_iiic
    raises — the screen must still render with skill ℓ̂ shown but no
    threshold-relative narrative."""
    import cortex_policy as cp

    def _boom(codes, **kw):
        raise FileNotFoundError("cert_config.yaml not found")
    monkeypatch.setattr(cp, "load_ell_star_iiic", _boom)
    result = _synthetic_result()
    screen = ev.ResultsScreen(result, n_correct=20, n_answered=42)
    try:
        all_text = " | ".join(
            w.text() for w in screen.findChildren(__import__(
                "PyQt6.QtWidgets", fromlist=["QLabel"]).QLabel))
        # Verdicts still shown
        assert "Pass" in all_text
        # Fallback narrative shows raw ℓ̂ without 'Above/Below threshold'
        assert "Above threshold" not in all_text
        assert "Below threshold" not in all_text
        # Raw ℓ̂ surfaced via the fallback
        assert "ℓ̂ = +0.85" in all_text
    finally:
        screen.deleteLater()
        qapp.processEvents()


def test_registration_wizard_advances_through_pages(qapp, monkeypatch):
    """Clicking NEXT on page 1 advances to page 2; NEXT on page 2
    advances to page 3 — but only if validators pass."""
    monkeypatch.setattr(ev.RegistrationPage, "_save_row",
                        staticmethod(lambda row: None))
    reg = ev.RegistrationPage()
    try:
        assert reg._stack.currentIndex() == 0
        # Page 1 unfilled — NEXT should keep us on page 1
        reg._advance_from_page1()
        assert reg._stack.currentIndex() == 0
        # Fill page 1 fields + eligibility, NEXT should advance
        reg.f_name.setText("Test User")
        reg.f_email.setText("test@example.com")
        reg.f_eligibility.setChecked(True)
        reg._advance_from_page1()
        assert reg._stack.currentIndex() == 1
        # Page 2 unfilled — NEXT should keep us on page 2
        reg._advance_from_page2()
        assert reg._stack.currentIndex() == 1
        # Fill page 2 dropdowns, NEXT should advance to page 3
        reg.f_expertise.setCurrentIndex(4)
        reg.f_practice.setCurrentIndex(1)
        reg.f_years_eeg.setCurrentIndex(1)
        reg.f_eeg_volume.setCurrentIndex(2)
        reg._advance_from_page2()
        assert reg._stack.currentIndex() == 2
    finally:
        reg.deleteLater()
        qapp.processEvents()
