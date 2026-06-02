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
    # Phase-9 K=7 contract: IIIC button index i → engine raw = i + 1
    # (spike=0 occupies engine task slot 0; IIIC tasks shift to k ∈ {1..6}).
    # Button 3 (LRDA) → raw 4 (matches engine task k=4=lrda).
    assert fake.submitted == [4]
    assert len(win.gui_trial_log) == 1
    rec = win.gui_trial_log[0]
    assert rec["response_raw"] == 4
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
    # Phase-9 K=7 contract: IIIC button 2 (GPD) → raw 3 (engine task k=3=gpd).
    assert fake.submitted == [3]


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


# ── v1.2.1 regression test for cortex-v1.2.0 spike-not-tested bug ──────────

def test_main_open_viewer_uses_k7_engine_inputs():
    """v1.2.1 regression: open_viewer() inside main() must import the K=7
    engine-inputs builder (build_k7_engine_inputs from cortex_engine_inputs_k7),
    NOT the K=6 builder (build_iiic_engine_inputs from cortex_engine_inputs).

    The v1.2.0 ship had `from cortex_engine_inputs import build_iiic_engine_inputs`
    here, which returned K=6 inputs (300 IIIC seg_ids only). The 50 spike segs
    in the bundled K=7 bank were never accessible to the engine; the session
    completed with 6 task verdicts and no spike verdict (see
    docs/SIM_V1_2_0_REPORT.md +
    /home/exx/.local/share/CORTEX/sessions/f3da305d-*/certificate.json
    for the diagnostic that surfaced this).

    session_controller.py:67 has `build_iiic_engine_inputs = build_k7_engine_inputs`
    as a back-compat alias, but that only applies to consumers importing FROM
    session_controller. open_viewer() imports directly from cortex_engine_inputs,
    bypassing the alias.

    This test is AST-based — it parses scripts/eeg_bank_viewer.py and inspects
    the import statements inside the open_viewer nested function. Any future
    refactor that swaps the import back to the K=6 module (or removes the K=7
    import) fails this test."""
    import ast
    from pathlib import Path
    viewer_src = (Path(__file__).resolve().parents[1] / "scripts"
                  / "eeg_bank_viewer.py").read_text()
    tree = ast.parse(viewer_src)
    open_viewer_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "open_viewer":
            open_viewer_node = node
            break
    assert open_viewer_node is not None, (
        "open_viewer nested function not found in eeg_bank_viewer.py — has "
        "main() been refactored? Update this test if so.")
    uses_k7_builder = False
    uses_k6_builder = False
    for sub in ast.walk(open_viewer_node):
        if isinstance(sub, ast.ImportFrom):
            if sub.module == "cortex_engine_inputs_k7":
                if any(a.name == "build_k7_engine_inputs" for a in sub.names):
                    uses_k7_builder = True
            elif sub.module == "cortex_engine_inputs":
                if any(a.name == "build_iiic_engine_inputs"
                       for a in sub.names):
                    uses_k6_builder = True
    assert uses_k7_builder, (
        "open_viewer must import build_k7_engine_inputs from "
        "cortex_engine_inputs_k7 (the K=7 production path). Without this, "
        "spike segments in the bundled bank are never asked.")
    assert not uses_k6_builder, (
        "open_viewer must NOT import build_iiic_engine_inputs (the K=6 path) "
        "— this was the cortex-v1.2.0 spike-not-tested bug. Use the K=7 "
        "builder from cortex_engine_inputs_k7 instead.")


def test_k7_engine_inputs_actually_returns_7_tasks_with_spike():
    """Companion smoke test for the above: build_k7_engine_inputs() must
    actually return 7 task codes with 'spike' as the first one + a `family`
    method on the result so eeg_bank_viewer can dispatch spike-vs-IIIC UI."""
    import cortex_engine_inputs_k7 as c7
    inputs = c7.build_k7_engine_inputs()
    assert list(inputs.task_codes) == [
        "spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"], (
        f"K=7 task_codes ordering changed; got {inputs.task_codes}")
    assert callable(getattr(inputs, "family", None)), (
        "K=7 EngineInputs must expose a family(seg_id) method for the "
        "family-aware UI dispatch in eeg_bank_viewer.show_item.")
    n_segs = len(inputs.all_seg_ids)
    assert n_segs > 6, f"K=7 bank has only {n_segs} segs — too small"


def test_default_policy_for_k7_uses_v14_block_by_default():
    """v1.2.1 regression: default_policy_for_k7's default block_name must
    match load_ell_star_k7's default. Both should target the K=7 production
    ship (ell_star_unified_v14). v1.2.0 had a mismatch — default_policy_for_k7
    defaulted to v13 while load_ell_star_k7 defaulted to v14 — which (after
    fixing the K=6/K=7 import bug above) would cause the engine to compute
    verdicts against v13 ℓ\\* while ResultsScreen rendered narratives against
    v14 ℓ\\*. Numbers on the same screen would disagree."""
    import inspect
    import cortex_policy_k7 as cp_k7
    sig = inspect.signature(cp_k7.default_policy_for_k7)
    assert sig.parameters["block_name"].default == "ell_star_unified_v14", (
        f"default_policy_for_k7 default block_name should be 'v14' to match "
        f"load_ell_star_k7; got {sig.parameters['block_name'].default!r}")
    # And confirm load_ell_star_k7's default matches:
    sig_load = inspect.signature(cp_k7.load_ell_star_k7)
    assert sig_load.parameters["block_name"].default == "ell_star_unified_v14"


# ── v1.2.4 regression tests for the 4-issue fix ───────────────────────────

def test_main_open_viewer_picks_iiic_tutorial_seg_under_k7():
    """v1.2.4 Issue 1 regression: open_viewer() must pick an IIIC seg as
    tutorial_sid, NOT the first item of all_seg_ids (which under K=7 is a
    spike seg — the K=7 inputs manifest has spike segments at indices 0..49).
    The tutorial UI is hard-coded for IIIC (eeg30s renderer + 6-button panel),
    so a spike tutorial_sid would silently fail to load (`bank.[/iiic/<spike_seg_id>]`
    doesn't exist). Surfaced by Eli's v1.2.3 self-test."""
    import ast
    from pathlib import Path
    viewer_src = (Path(__file__).resolve().parents[1] / "scripts"
                  / "eeg_bank_viewer.py").read_text()
    tree = ast.parse(viewer_src)
    open_viewer_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "open_viewer":
            open_viewer_node = node
            break
    assert open_viewer_node is not None
    # Look for a call to inputs.family(...) inside a generator-expression
    # filter on inputs.all_seg_ids — this is the "pick first IIIC seg" pattern.
    src_slice = ast.unparse(open_viewer_node)
    assert "inputs.family" in src_slice and "iiic" in src_slice, (
        "open_viewer must use inputs.family(...) to filter all_seg_ids "
        "for the tutorial seg (Issue 1 regression).")


def test_redraw_skips_spectrogram_for_spike_family():
    """v1.2.4 Issue 2 regression: BankViewer._redraw must SKIP
    _draw_spectrogram() when self._cur_family == 'spike'. Spike segments
    don't carry sdata/sfreqs/stimes, and the spike-paper methodology uses
    the EEG signal only (no spectrogram panel). Surfaced by Eli's v1.2.3
    self-test."""
    import ast
    from pathlib import Path
    viewer_src = (Path(__file__).resolve().parents[1] / "scripts"
                  / "eeg_bank_viewer.py").read_text()
    tree = ast.parse(viewer_src)
    redraw_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_redraw":
            redraw_node = node
            break
    assert redraw_node is not None
    src_slice = ast.unparse(redraw_node)
    # Family-aware spectrogram gate
    assert "_cur_family" in src_slice, "_redraw must read _cur_family"
    assert "spike" in src_slice, "_redraw must mention spike in its gate"
    # Confirm _draw_spectrogram is conditional on something not just spec_cb
    assert "_draw_spectrogram" in src_slice


def test_compute_active_domains_k7_spike_first_sectioning():
    """v1.2.4 Issue 3 regression: K=7 sessions must ask spike (k=0) first
    UNTIL the spike verdict locks (PASS/FAIL) OR the spike pool exhausts,
    then transition to IIIC (k=1..6). Implements `_compute_active_domains`
    on CortexSession."""
    import sys as _sys
    _SCRIPTS = str(_REPO + "/scripts")
    if _SCRIPTS not in _sys.path:
        _sys.path.insert(0, _SCRIPTS)
    import numpy as np
    from session_controller import CortexSession
    from cortex_engine_inputs_k7 import build_k7_engine_inputs
    from cortex_policy import PASS, PENDING

    inputs = build_k7_engine_inputs()
    inp = inputs.without([inputs.all_seg_ids[0]])
    sess = CortexSession(inp, session_id="t-active", n_particles=200)
    sess.policy.reset(7)

    # Phase A: spike pool non-empty + PENDING → only k=0
    bs, _, _ = inp.as_engine_arrays(inp.all_seg_ids)
    assert sess._compute_active_domains(bs) == [0]

    # Phase A→B trigger 1: spike resolved → IIIC tasks active
    sess.policy._verdicts = [PASS] + [PENDING] * 6
    assert sess._compute_active_domains(bs) == [1, 2, 3, 4, 5, 6]

    # Phase A→B trigger 2: spike pool empty + spike PENDING → still skip k=0
    sess.policy._verdicts = [PENDING] * 7
    iiic_only = [s for s in inp.all_seg_ids if inp.family(s) != "spike"]
    bs_empty_spike, _, _ = inp.as_engine_arrays(iiic_only)
    assert bs_empty_spike[0].size == 0
    assert sess._compute_active_domains(bs_empty_spike) == [1, 2, 3, 4, 5, 6]

    # All resolved: empty list → run loop will stop the session
    sess.policy._verdicts = [PASS] * 7
    assert sess._compute_active_domains(bs) == []


def test_redraw_hides_spec_container_for_spike_and_disables_checkbox():
    """v1.2.5 Issue 1 regression: _redraw must HIDE spec_container (not just
    skip drawing) when family is spike, so the tutorial's IIIC spectrogram
    does not bleed through. Also must disable the spec_cb checkbox so the
    user cannot toggle on a panel that has no data. The EEG plot has
    stretch=1 in plots_row while spec_container has stretch=0, so hiding
    the container makes the EEG auto-expand to fill the freed space.
    AST-checked so future refactors of _redraw cannot reintroduce the
    silent-skip-but-do-not-hide bug from v1.2.4."""
    import ast
    from pathlib import Path
    viewer_src = (Path(__file__).resolve().parents[1] / "scripts"
                  / "eeg_bank_viewer.py").read_text()
    tree = ast.parse(viewer_src)
    redraw_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_redraw":
            redraw_node = node
            break
    assert redraw_node is not None
    src = ast.unparse(redraw_node)
    # Must hide the container outright + disable the checkbox for spike
    assert "spec_container.setVisible(False)" in src
    assert "spec_cb.setEnabled(False)" in src
    # And must re-enable both when the family is not spike
    assert "spec_cb.setEnabled(True)" in src


def test_tutorial_spectrogram_step_notes_spike_has_no_spectrogram():
    """v1.2.6: the spectrogram coach-mark in start_tutorial must tell the
    user the spectrogram is only shown for the pattern-classification
    (IIIC) recordings and that spike questions show the EEG alone. The
    tutorial renders an IIIC example so the panel IS visible during the
    walkthrough; without this note a tester would not know the panel
    disappears for the spike block (paired with the v1.2.5 _redraw hide).
    AST-checked against the start_tutorial step list so the note cannot
    silently drop out of a future tutorial-copy edit."""
    import ast
    from pathlib import Path
    viewer_src = (Path(__file__).resolve().parents[1] / "scripts"
                  / "eeg_bank_viewer.py").read_text()
    tree = ast.parse(viewer_src)
    start_tutorial_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "start_tutorial":
            start_tutorial_node = node
            break
    assert start_tutorial_node is not None
    src = ast.unparse(start_tutorial_node)
    # The spectrogram step ("The spectrogram" title) must carry the spike caveat.
    assert "The spectrogram" in src
    assert "no spectrogram panel" in src, (
        "start_tutorial spectrogram step must note that spike-present "
        "questions show the EEG alone with no spectrogram panel (v1.2.6).")


def test_render_collapse_passfail_k7_grid_layout():
    """v1.2.5 Issue 2 regression: cortex_render_videos.render_collapse and
    render_passfail must use a K-aware grid (was hardcoded
    GridSpecFromSubplotSpec(2, 3) = 6 cells; K=7 crashes at inner[2, 0]).
    Both functions must size the grid to fit K panels."""
    import ast
    from pathlib import Path
    rv_src = (Path(__file__).resolve().parents[1] / "scripts"
              / "cortex_render_videos.py").read_text()
    tree = ast.parse(rv_src)
    for fn_name in ("render_collapse", "render_passfail"):
        node = None
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == fn_name:
                node = n
                break
        assert node is not None, f"{fn_name} not found"
        src = ast.unparse(node)
        # K-aware grid sizing
        assert "n_cols" in src and "n_rows" in src, (
            f"{fn_name} must compute n_cols + n_rows from K")
        # The hardcoded GridSpecFromSubplotSpec(2, 3, ...) must be GONE
        assert "GridSpecFromSubplotSpec(2, 3" not in src, (
            f"{fn_name} still has the hardcoded K=6 2x3 grid (Issue 2)")
        # The hardcoded `r == 1` (last-row marker for 2-row grids) must be
        # replaced with n_rows-aware logic.
        assert "r == 1" not in src, (
            f"{fn_name} still has the hardcoded `r == 1` last-row check")


def test_render_collapse_passfail_actually_render_k7_session():
    """v1.2.5 Issue 2 end-to-end regression: render both MP4s for both K=6
    and K=7 synthetic sessions; both must succeed without IndexError."""
    import sys as _sys
    import tempfile
    from pathlib import Path
    _SCRIPTS = str(_REPO + "/scripts")
    if _SCRIPTS not in _sys.path:
        _sys.path.insert(0, _SCRIPTS)
    import numpy as np
    from cortex_render_videos import render_collapse, render_passfail
    for K_n, codes in [(7, ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]),
                        (6, ["sz", "lpd", "gpd", "lrda", "grda", "iic"])]:
        T, N = 12, 50      # tiny; just exercises the grid layout
        session = {
            "participant_name": f"K{K_n}",
            "task_codes": codes,
            "t_traj": np.random.randn(T, N, K_n).astype(np.float32) * 0.5,
            "l_traj": np.random.randn(T, N, K_n).astype(np.float32) * 0.5,
            "w_traj": np.ones((T, N), dtype=np.float32) / N,
            "trials": [{"auroc_hw": [0.1] * K_n,
                         "policy_diag": {"pi": [0.5] * K_n,
                                          "verdicts": ["PENDING"] * K_n,
                                          "mcse": [0.05] * K_n},
                         "verdicts": ["PENDING"] * K_n,
                         "task_k": i % K_n} for i in range(T)],
            "n_questions": T,
            "final_verdicts": ["PASS"] * K_n,
            "stop_reason": "all_resolved",
        }
        with tempfile.TemporaryDirectory() as td:
            for fn, fn_label in [(render_collapse, "collapse"),
                                  (render_passfail, "passfail")]:
                out = Path(td) / f"{fn_label}_k{K_n}.mp4"
                fn(session, out)
                assert out.exists() and out.stat().st_size > 1024


def test_session_skips_empty_spike_bank_without_indexerror():
    """v1.2.4 Issue 4 regression: when the spike pool exhausts mid-session
    (random rater + N_MIN=20 + 50 spike bank: ~trial 270 in Eli's frozen
    v1.2.3 self-test), the engine's `choose_item` used to crash with
    `IndexError: index 0 is out of bounds for axis 0 with size 0` because
    `bank_signals[0]` had become `np.array([])`. The phase-aware
    active_domains in _compute_active_domains prevents the empty-bank pick.
    """
    import sys as _sys
    _SCRIPTS = str(_REPO + "/scripts")
    if _SCRIPTS not in _sys.path:
        _sys.path.insert(0, _SCRIPTS)
    import numpy as np
    from session_controller import CortexSession
    from cortex_engine_inputs_k7 import build_k7_engine_inputs

    inputs = build_k7_engine_inputs()
    inp = inputs.without([inputs.all_seg_ids[0]])
    sess = CortexSession(inp, session_id="t-empty-spike",
                         seed=99, max_questions=100, n_particles=200)
    rng = np.random.default_rng(99)
    def random_y(k, seg_id, s):
        return int(rng.integers(0, 2)) if k == 0 else \
               int(rng.choice([0, 0, 0, 0, 0, 1]))
    # If the bug were present, this run would crash at some trial in [50, 100].
    result = sess.run(random_y)
    assert result.n_questions > 0
    assert result.stop_reason in (
        "all_resolved", "bank_exhausted", "all_active_resolved", "max_reached")


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


def test_registration_csv_schema_v4(qapp, tmp_path, monkeypatch):
    """The persisted CSV header must equal REGISTRATION_FIELDS (v4) exactly
    (downstream analysis joins on these column names). v1.1.4 added
    wants_visualizations as a session preference; gender_identity
    (dropped in v1.1.3) stays absent."""
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
        # v1.1.4: opt INTO visualizations so we exercise the non-default
        # value in the persisted row.
        reg.f_visualizations.setCurrentIndex(1)  # "Yes — generate ..."
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
    assert "wants_visualizations" in header
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "Test User"
    assert r["sex"] == "Male"
    assert r["race_ethnicity"] == "White"
    assert r["consent_version"] == ev.CONSENT_VERSION
    assert r["eligibility_confirmed"] == "yes"
    assert r["wants_visualizations"].lower().startswith("yes")
    assert ev.is_opt_in_for_visualizations(r) is True


def test_registration_default_opts_out_of_visualizations(
        qapp, tmp_path, monkeypatch):
    """Silent participant ⇒ wants_visualizations defaults to 'No'.
    This is the v1.1.4 fast path that fixes the 3-min wait."""
    import cortex_storage as cs
    monkeypatch.setattr(cs, "user_data_root", lambda: tmp_path)
    reg = ev.RegistrationPage()
    try:
        _fill_required(reg)
        # Do NOT touch f_visualizations — it should default to index 0
        # ("No — faster results").
        assert reg.commit() is True
    finally:
        reg.deleteLater()
        qapp.processEvents()
    import csv as _csv
    with open(tmp_path / "registrations.csv", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows[0]["wants_visualizations"].lower().startswith("no")
    assert ev.is_opt_in_for_visualizations(rows[0]) is False


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


def test_registration_schema_migration_rotates_v3_csv(
        qapp, tmp_path, monkeypatch):
    """A v3 (v1.1.3) header on disk must be rotated to
    registrations.v3.csv.bak — new for v1.1.4 since the upgrade path
    now spans three schema bumps. The v1→v3 and v2→v3 paths from
    v1.1.3 are exercised separately above."""
    import cortex_storage as cs
    monkeypatch.setattr(cs, "user_data_root", lambda: tmp_path)

    legacy_path = tmp_path / "registrations.csv"
    with open(legacy_path, "w", encoding="utf-8", newline="") as fh:
        import csv as _csv
        w = _csv.writer(fh)
        w.writerow(ev._REGISTRATION_FIELDS_V3)
        w.writerow(["old-sid", "2026-05-27T00:00:00+00:00",
                    "v1.1.3-placeholder", "", "yes",
                    "Old User", "33", "old@example.com", "MGH",
                    "Fellow", "Academic medical center",
                    "10–14", "21–50",
                    "5", "Normal color vision", "No",
                    "Female", "United States", "White"])

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
    bak = tmp_path / "registrations.v3.csv.bak"
    assert bak.exists(), \
        "v3-header CSV must rotate to .v3.csv.bak (not .v2 or .legacy)"
    with open(bak, encoding="utf-8") as fh:
        assert next(_csv.reader(fh)) == ev._REGISTRATION_FIELDS_V3


# ─────── v1.1.4: ComputingResultsPage + background finalize ───────

def test_computing_results_page_opt_out_text(qapp):
    """Opt-out path shows the fast-results copy."""
    page = ev.ComputingResultsPage(opt_in=False)
    try:
        from PyQt6.QtWidgets import QLabel
        texts = [w.text() for w in page.findChildren(QLabel)]
        joined = " ".join(texts)
        assert any("Thank you for participating" in t for t in texts)
        assert "few seconds" in joined
        # opt-in copy must NOT appear
        assert "2 to 3 minutes" not in joined
        assert page.opt_in is False
        # Spinner is indeterminate (min == max == 0)
        assert page.spinner.minimum() == 0
        assert page.spinner.maximum() == 0
    finally:
        page.deleteLater()
        qapp.processEvents()


def test_computing_results_page_opt_in_text(qapp):
    """Opt-in path shows the visualizations advisory, with no time estimate
    (v1.3.1: the '2 to 3 minutes' line was removed)."""
    page = ev.ComputingResultsPage(opt_in=True)
    try:
        from PyQt6.QtWidgets import QLabel
        texts = [w.text() for w in page.findChildren(QLabel)]
        joined = " ".join(texts)
        assert "personalized visualizations" in joined
        assert "2 to 3 minutes" not in joined
        assert "few seconds" not in joined        # opt-out copy only
        assert page.opt_in is True
    finally:
        page.deleteLater()
        qapp.processEvents()


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
    # Phase-9 K=7: ResultsScreen now imports load_ell_star_k7 from
    # cortex_policy_k7 (Layer 6a change). Patch both sources so the test
    # works regardless of how ResultsScreen resolves the loader at call time.
    import cortex_policy as cp
    import cortex_policy_k7 as cp_k7
    monkeypatch.setattr(cp, "load_ell_star_iiic",
                        lambda codes, **kw: [0.42] * len(codes))
    monkeypatch.setattr(cp_k7, "load_ell_star_k7",
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


def test_results_screen_show_folder_button_hidden_when_no_session_dir(
        qapp, monkeypatch):
    """Pre-v1.1.5 callers (and test fixtures) construct ResultsScreen
    without session_dir → the new 'Show results folder' button must
    not appear on the screen."""
    # Phase-9 K=7: ResultsScreen now imports load_ell_star_k7 from
    # cortex_policy_k7 (Layer 6a change). Patch both sources so the test
    # works regardless of how ResultsScreen resolves the loader at call time.
    import cortex_policy as cp
    import cortex_policy_k7 as cp_k7
    monkeypatch.setattr(cp, "load_ell_star_iiic",
                        lambda codes, **kw: [0.42] * len(codes))
    monkeypatch.setattr(cp_k7, "load_ell_star_k7",
                        lambda codes, **kw: [0.42] * len(codes))
    result = _synthetic_result()
    screen = ev.ResultsScreen(result, n_correct=20, n_answered=42)
    try:
        assert hasattr(screen, "show_folder_btn")
        assert screen.show_folder_btn.isHidden() is True
        # CLOSE button still present + visible
        assert hasattr(screen, "close_btn")
    finally:
        screen.deleteLater()
        qapp.processEvents()


def test_results_screen_show_folder_button_dispatches_to_os(
        qapp, tmp_path, monkeypatch):
    """When session_dir is supplied, clicking the button must
    invoke the OS file manager via subprocess.Popen with the right
    command for sys.platform. Mock subprocess.Popen to record the
    call without actually spawning."""
    # Phase-9 K=7: ResultsScreen now imports load_ell_star_k7 from
    # cortex_policy_k7 (Layer 6a change). Patch both sources so the test
    # works regardless of how ResultsScreen resolves the loader at call time.
    import cortex_policy as cp
    import cortex_policy_k7 as cp_k7
    monkeypatch.setattr(cp, "load_ell_star_iiic",
                        lambda codes, **kw: [0.42] * len(codes))
    monkeypatch.setattr(cp_k7, "load_ell_star_k7",
                        lambda codes, **kw: [0.42] * len(codes))
    # Create a real dir so the existence check passes.
    sd = tmp_path / "session-uuid"
    sd.mkdir()

    calls = []

    class _FakePopen:
        def __init__(self, cmd, **kw):
            calls.append((list(cmd), kw))

    monkeypatch.setattr(ev.subprocess, "Popen", _FakePopen)

    result = _synthetic_result()
    screen = ev.ResultsScreen(result, n_correct=20, n_answered=42,
                               session_dir=sd)
    try:
        # show_folder_btn now visible (isHidden() reflects
        # setVisible(True); the True/False state is reliable even
        # when the parent window isn't shown).
        assert hasattr(screen, "show_folder_btn")
        assert screen.show_folder_btn.isHidden() is False
        screen.show_folder_btn.click()
        assert calls, "subprocess.Popen was not called"
        cmd, _ = calls[0]
        # Path must be the session dir — the heart of the v1.1.5
        # 'this session, not the parent sessions/ root' decision.
        assert cmd[-1] == str(sd)
        # Per-platform dispatch
        import sys as _sys
        if _sys.platform == "darwin":
            assert cmd[0] == "open"
        elif _sys.platform == "win32":
            assert cmd[0] == "explorer"
        else:
            assert cmd[0] == "xdg-open"
    finally:
        screen.deleteLater()
        qapp.processEvents()


def test_results_screen_show_folder_button_no_op_when_dir_missing(
        qapp, tmp_path, monkeypatch):
    """Defensive: if the session_dir was deleted between session-end
    and button click (e.g., participant cleared the folder manually),
    the click should log + no-op rather than crash."""
    # Phase-9 K=7: ResultsScreen now imports load_ell_star_k7 from
    # cortex_policy_k7 (Layer 6a change). Patch both sources so the test
    # works regardless of how ResultsScreen resolves the loader at call time.
    import cortex_policy as cp
    import cortex_policy_k7 as cp_k7
    monkeypatch.setattr(cp, "load_ell_star_iiic",
                        lambda codes, **kw: [0.42] * len(codes))
    monkeypatch.setattr(cp_k7, "load_ell_star_k7",
                        lambda codes, **kw: [0.42] * len(codes))
    # session_dir that does NOT exist
    sd = tmp_path / "deleted-session-uuid"
    calls = []
    monkeypatch.setattr(
        ev.subprocess, "Popen",
        lambda *a, **kw: calls.append(a))

    result = _synthetic_result()
    screen = ev.ResultsScreen(result, n_correct=20, n_answered=42,
                               session_dir=sd)
    try:
        # Button is still visible (we constructed with session_dir).
        # Click ⇒ early-return because dir doesn't exist; no Popen.
        screen.show_folder_btn.click()
        assert not calls, ("Popen should NOT be called when the "
                           "session dir is missing")
    finally:
        screen.deleteLater()
        qapp.processEvents()


def test_results_screen_handles_missing_threshold(qapp, monkeypatch):
    """If cert_config.yaml is missing / malformed, load_ell_star_k7
    raises — the screen must still render with skill ℓ̂ shown but no
    threshold-relative narrative."""
    # Phase-9 K=7: patch BOTH loaders so the test covers both paths.
    import cortex_policy as cp
    import cortex_policy_k7 as cp_k7

    def _boom(codes, **kw):
        raise FileNotFoundError("cert_config.yaml not found")
    monkeypatch.setattr(cp, "load_ell_star_iiic", _boom)
    monkeypatch.setattr(cp_k7, "load_ell_star_k7", _boom)
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


def test_v1_3_5_live_session_wires_consec_cap():
    """The live test wires the consecutive-same-domain cap into the production
    session. v1.3.6 lowered the default 12 -> 5 (Eli's call)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "scripts"
    viewer = (root / "eeg_bank_viewer.py").read_text()
    assert "max_consecutive_same_domain=MAX_CONSEC_SAME_DOMAIN_DEFAULT" in viewer
    sc = (root / "session_controller.py").read_text()
    assert "MAX_CONSEC_SAME_DOMAIN_DEFAULT = 5" in sc
