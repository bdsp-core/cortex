"""Phase D — tests for scripts/cortex_storage.py.

The per-session SessionRecorder: crash-safe trials.jsonl / events.jsonl,
participant.json, certificate.json, trajectory.npz, and the synced CSVs.
Bank-independent — uses synthetic trial dicts and a stub result object.
"""
from __future__ import annotations

import csv
import json
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


def _gui(trial_index, label="Seizure", rt=820.5, changes=1):
    return {"trial_index": trial_index, "seg_id": 1000, "task_k": 0,
            "response_raw": 0, "response_label": label, "reaction_time_ms": rt,
            "answer_changes": changes, "montage": "bipolar", "gain_uv": 100.0,
            "bandpass": "0.5-70 Hz", "notch": "60 Hz", "window_s": 10.0,
            "pan_t_start": 0.0,
            "interaction": [{"t_ms": 100.0, "action": "pan", "detail": 1},
                            {"t_ms": 300.0, "action": "select",
                             "detail": label}]}


def _result(aborted=False, n=2):
    return SimpleNamespace(
        session_id="sess-test", stop_reason="delta_reached", delta_auroc=0.15,
        n_questions=n, task_codes=list(TASK_CODES),
        final_auroc_mean=np.full(6, 0.82), final_auroc_hw=np.full(6, 0.14),
        final_l_mean=np.full(6, 0.45), final_t_mean=np.zeros(6),
        served_seg_ids=list(range(1000, 1000 + n)),
        t_traj=np.zeros((n, 50, 6)), l_traj=np.zeros((n, 50, 6)),
        w_traj=np.full((n, 50), 0.02), aborted=aborted)


def _recorder(tmp_path, synced=True):
    return cs.SessionRecorder(
        "sess-test", PARTICIPANT,
        {"delta_auroc": 0.15, "n_particles": 600},
        sessions_root=tmp_path / "sessions",
        synced_dir=(tmp_path / "synced") if synced else None,
        dropbox_cfg=None,            # tests must never hit the real Dropbox
        render_videos=False)         # ffmpeg per test would explode runtime


def test_participant_json_written(tmp_path):
    rec = _recorder(tmp_path)
    doc = json.loads((rec.dir / "participant.json").read_text())
    assert doc["session_id"] == "sess-test"
    assert doc["identity"]["name"] == "Jane Doe"
    assert doc["session_config"]["n_particles"] == 600
    rec.close()


def test_trials_jsonl_crash_safe(tmp_path):
    rec = _recorder(tmp_path)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1, y=0), _gui(1, label="LPD"))
    # readable on disk BEFORE close() — flush + fsync per trial
    lines = (rec.dir / "trials.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["trial_index"] == 0
    assert rec0["reaction_time_ms"] == 820.5
    assert rec0["answer_changes"] == 1
    assert rec0["is_correct"] is True            # Seizure == seizure
    assert "interaction" not in rec0             # trace -> events.jsonl
    rec.close()


def test_events_jsonl(tmp_path):
    rec = _recorder(tmp_path)
    rec.write_trial(_telemetry(0), _gui(0))
    evlines = (rec.dir / "events.jsonl").read_text().strip().splitlines()
    assert len(evlines) == 2
    ev0 = json.loads(evlines[0])
    assert ev0["trial_index"] == 0
    assert ev0["action"] == "pan"
    assert ev0["session_id"] == "sess-test"
    rec.close()


def test_is_correct_merge(tmp_path):
    rec = _recorder(tmp_path)
    rec.write_trial(_telemetry(0, pc="lpd"), _gui(0, label="Seizure"))
    line = json.loads((rec.dir / "trials.jsonl").read_text().splitlines()[0])
    assert line["is_correct"] is False           # Seizure != lpd
    rec.close()


def test_spike_is_correct_uses_sign_of_s_mean():
    """v1.3.6: the BINARY spike task is graded against sign(s_mean) (spike
    present = s_mean>0), NOT the IIIC label==pattern_class_true match — which is
    always False for spike (pattern_class_true is the placeholder "spike") and
    produced spurious 0% spike accuracy."""
    # unit — the helper
    assert cs._spike_correct(1, 1.5) is True      # said Yes, s_mean>0 -> correct
    assert cs._spike_correct(0, -1.2) is True     # said No, s_mean<0 -> correct
    assert cs._spike_correct(1, -0.8) is False    # said Yes, s_mean<0 -> wrong
    assert cs._spike_correct(0, 0.9) is False     # said No, s_mean>0 -> wrong
    assert cs._spike_correct(None, 1.0) is False  # defensive
    # integration — via _merge, a binary spike trial now grades correctly
    tel = {"trial_index": 0, "task_k": 0, "task_code": "spike", "seg_id": 5,
           "pattern_class_true": "spike", "s_mean": 1.3, "response_y": 1}
    gui = {"trial_index": 0, "response_label": "Yes (spike present)"}
    assert cs.SessionRecorder._merge(tel, gui)["is_correct"] is True   # was always False
    assert cs.SessionRecorder._merge(dict(tel, response_y=0), gui)["is_correct"] is False


def test_finalize_writes_certificate_and_trajectory(tmp_path):
    rec = _recorder(tmp_path)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1), _gui(1))
    rec.finalize(_result())
    cert = json.loads((rec.dir / "certificate.json").read_text())
    assert cert["total_trials"] == 2
    assert len(cert["per_task"]) == 6
    assert cert["per_task"][0]["task_code"] == "sz"
    npz = np.load(rec.dir / "trajectory.npz")
    assert npz["t_traj"].shape == (2, 50, 6)
    rec.close()


def test_synced_csvs_on_clean_completion(tmp_path):
    rec = _recorder(tmp_path, synced=True)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1), _gui(1))
    rec.finalize(_result(aborted=False))
    rec.close()
    synced = tmp_path / "synced"
    trials = list(synced.glob("*_trials.csv"))
    summary = list(synced.glob("*_summary.csv"))
    assert len(trials) == 1 and len(summary) == 1
    rows = list(csv.DictReader(open(trials[0])))
    assert len(rows) == 2
    assert rows[0]["participant_name"] == "Jane Doe"   # v1.3.9: name included
    assert "target_present" in rows[0]                  # v1.3.9: asked-task truth
    assert rows[0]["is_correct"] == "True"
    srow = list(csv.DictReader(open(summary[0])))[0]
    assert srow["n_questions"] == "2"
    assert "auroc_sz" in srow


def test_synced_csvs_include_name_protect_email_and_institution(tmp_path):
    """v1.3.9 (controlled internal test): the synced/uploaded CSVs now INCLUDE the
    participant NAME (needed to link sessions to experience level for the
    PASS/FAIL/REFER calibration), but email is still excluded and the institution
    NAME is still coded to a site_id. The full identity stays LOCAL in
    participant.json. (Supersedes the v1.3.6 full de-identification.)"""
    rec = _recorder(tmp_path)                    # PARTICIPANT = Jane Doe / MGH / Fellow
    rec.write_trial(_telemetry(0), _gui(0))
    rec.write_trial(_telemetry(1, y=0), _gui(1, label="LPD"))
    rec.finalize(_result(n=2))
    rec.close()
    synced = tmp_path / "synced"
    files = sorted(synced.glob("*.csv"))
    assert files, "no synced CSVs written"
    # (a) filenames stay session_id-keyed (no name embedded)
    for f in files:
        assert f.name.startswith("sess-test_")
        assert "Jane" not in f.name and "Doe" not in f.name
    # (b) email + institution NAME still never appear in the synced bytes
    blob = "\n".join(f.read_text() for f in files)
    for leak in ("jane@example.com", "MGH"):
        assert leak not in blob, f"identifier leaked into synced CSV: {leak!r}"
    # (c) summary: name IS included; institution still coded; study vars retained
    srow = list(csv.DictReader((synced / "sess-test_summary.csv").open()))[0]
    assert srow["participant_name"] == "Jane Doe"
    assert "institution" not in srow
    assert srow["session_id"] == "sess-test"
    assert srow["expertise"] == "Fellow"                 # study variable retained
    assert srow["site_id"].startswith("site_") and srow["site_id"] != "MGH"
    # (d) trials CSV carries the name + the asked-task truth column
    trow = list(csv.DictReader((synced / "sess-test_trials.csv").open()))[0]
    assert trow["participant_name"] == "Jane Doe"
    assert "target_present" in trow
    # (e) full identity still LOCAL in participant.json
    doc = json.loads((rec.dir / "participant.json").read_text())
    assert doc["identity"]["name"] == "Jane Doe"
    assert doc["identity"]["institution"] == "MGH"


def test_site_id_coding():
    """institution NAME -> stable coded site_id; never the plaintext name."""
    assert cs._site_id("MGH") == cs._site_id("  mgh ")   # normalized + stable
    assert cs._site_id("MGH") != cs._site_id("BWH")       # distinct sites
    assert cs._site_id("") == "" and cs._site_id(None) == ""
    assert "MGH" not in cs._site_id("MGH")                # not the plaintext name


def test_consent_provenance_travels_with_deid_data(tmp_path):
    """C2: consent provenance (consent_version + irb_protocol_id) must travel
    WITH the de-identified synced summary, so every shared record is auditably
    linked to the IRB-approved consent it was collected under (NEJM AI / ICMJE)."""
    p = dict(PARTICIPANT)
    p.update({"consent_version": "v1.2.3-irb-approved",
              "irb_protocol_id": "IRB-2026-0142", "expertise": "Epileptologist"})
    rec = cs.SessionRecorder(
        "sess-test", p, {"delta_auroc": 0.15, "n_particles": 600},
        sessions_root=tmp_path / "sessions", synced_dir=tmp_path / "synced",
        dropbox_cfg=None, render_videos=False)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_result(n=1))
    rec.close()
    srow = list(csv.DictReader(
        (tmp_path / "synced" / "sess-test_summary.csv").open()))[0]
    assert srow["consent_version"] == "v1.2.3-irb-approved"
    assert srow["irb_protocol_id"] == "IRB-2026-0142"
    assert srow["expertise"] == "Epileptologist"          # B1: tier travels too


def test_summary_csv_includes_v1_1_1_demographic_fields(tmp_path):
    """v1.1.1 expanded the summary row with demographic + clinical-
    background columns from the RegistrationPage so the uploaded data
    is mineable. Verify they're present and populated when supplied."""
    rich_participant = dict(PARTICIPANT)
    rich_participant.update({
        "practice_setting": "Academic medical center",
        "years_reading_eeg": "10–14", "eeg_volume_per_month": "21–50",
        "self_rated_confidence": "5", "color_vision": "Normal color vision",
        "prior_test_taken": "No", "sex": "Female",
        # v1.1.3: gender_identity dropped — confirm absence in summary too.
        "country": "United States",
        "race_ethnicity": "White", "consent_version": "v1.1.4-placeholder",
        "irb_protocol_id": "",
    })
    rec = cs.SessionRecorder(
        "sess-test", rich_participant,
        {"delta_auroc": 0.15, "n_particles": 600},
        sessions_root=tmp_path / "sessions",
        synced_dir=tmp_path / "synced",
        dropbox_cfg=None, render_videos=False)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_result(aborted=False, n=1))
    rec.close()
    summary = list((tmp_path / "synced").glob("*_summary.csv"))
    srow = list(csv.DictReader(open(summary[0])))[0]
    for col, want in (
        ("practice_setting", "Academic medical center"),
        ("years_reading_eeg", "10–14"),
        ("eeg_volume_per_month", "21–50"),
        ("sex", "Female"),
        ("country", "United States"), ("race_ethnicity", "White"),
        ("consent_version", "v1.1.4-placeholder"),
    ):
        assert srow[col] == want, f"{col}: got {srow[col]!r}, want {want!r}"
    # v1.1.3: gender_identity column should be ABSENT from the summary.
    assert "gender_identity" not in srow


def test_summary_csv_back_compat_missing_demographics(tmp_path):
    """When the participant dict lacks the v1.1.1 fields (pre-upgrade
    callers), the summary row must still write — empty strings, not
    KeyError or crash."""
    rec = _recorder(tmp_path, synced=True)            # legacy PARTICIPANT
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_result(aborted=False, n=1))
    rec.close()
    summary = list((tmp_path / "synced").glob("*_summary.csv"))
    srow = list(csv.DictReader(open(summary[0])))[0]
    assert srow["sex"] == "" and srow["country"] == ""
    assert srow["consent_version"] == ""


def test_aborted_session_no_synced_csv(tmp_path):
    rec = _recorder(tmp_path, synced=True)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_result(aborted=True, n=1))
    rec.close()
    # local crash-safe files exist; the synced CSVs do NOT
    assert (rec.dir / "trials.jsonl").read_text().strip()
    assert not list((tmp_path / "synced").glob("*.csv"))


def test_load_dropbox_config_parsing(tmp_path, monkeypatch):
    """Schema-level coverage of all three auth-mode permutations the bundle
    supports: refresh-token (preferred), legacy access-token, and the
    empty-config case."""
    cfg = tmp_path / "c.yaml"
    monkeypatch.setattr(cs, "CONFIG_PATH", cfg)

    # (a) empty config — no upload configured
    cfg.write_text('dropbox:\n  access_token: ""\n  folder: "/r"\n')
    assert cs.load_dropbox_config() is None
    cfg.write_text('synced_results_dir: ""\n')           # no dropbox section
    assert cs.load_dropbox_config() is None

    # (b) legacy access-token-only mode — still supported for one-off use
    cfg.write_text('dropbox:\n  access_token: "tok"\n  folder: "/r"\n')
    assert cs.load_dropbox_config() == {
        "refresh_token": "", "app_key": "", "app_secret": "",
        "access_token": "tok", "folder": "/r"}

    # (c) refresh-token mode with app_secret (Confidential apps — the
    #     default Dropbox app type). app_secret is required at every token
    #     refresh, not just the one-time exchange.
    cfg.write_text(
        'dropbox:\n  app_key: "AK"\n  app_secret: "AS"\n'
        '  refresh_token: "RT"\n  folder: "/r"\n')
    assert cs.load_dropbox_config() == {
        "refresh_token": "RT", "app_key": "AK", "app_secret": "AS",
        "access_token": "", "folder": "/r"}

    # (d) refresh-token mode without app_secret — valid only for Public
    #     (PKCE) apps. Loader still returns the config; the SDK construction
    #     decides whether app_secret is needed.
    cfg.write_text('dropbox:\n  app_key: "AK"\n  refresh_token: "RT"\n'
                   '  folder: "/r"\n')
    assert cs.load_dropbox_config() == {
        "refresh_token": "RT", "app_key": "AK", "app_secret": "",
        "access_token": "", "folder": "/r"}

    # (e) partial refresh-token credentials must NOT activate; falls through
    #     to the access-token check. With neither, returns None.
    cfg.write_text('dropbox:\n  app_key: "AK"\n  folder: "/r"\n')
    assert cs.load_dropbox_config() is None
    cfg.write_text('dropbox:\n  refresh_token: "RT"\n  folder: "/r"\n')
    assert cs.load_dropbox_config() is None


def test_dropbox_upload_calls_client(tmp_path):
    """Smoke: the upload loop hits files_upload(data, path) per CSV — the
    actual SDK construction is bypassed via the _client test seam."""
    f1 = tmp_path / "sess_a_trials.csv"
    f1.write_text("trial,data\n1,x\n")
    f2 = tmp_path / "sess_a_summary.csv"
    f2.write_text("n\n1\n")

    class FakeClient:
        def __init__(self):
            self.uploads = []

        def files_upload(self, data, path):
            self.uploads.append((path, data))

    fake = FakeClient()
    cfg = {"refresh_token": "RT", "app_key": "AK", "access_token": "",
           "folder": "/results"}
    cs._dropbox_upload(cfg, [f1, f2], _client=fake)
    assert len(fake.uploads) == 2
    assert sorted(u[0] for u in fake.uploads) == [
        "/results/sess_a_summary.csv", "/results/sess_a_trials.csv"]
    assert fake.uploads[0][1] == f1.read_bytes()


def test_build_dropbox_client_picks_refresh_path_with_secret(monkeypatch):
    """Confidential-app path (the Dropbox default): app_secret is passed to
    the SDK constructor so the runtime refresh call to /oauth2/token can
    authenticate. This is the regression that prevents the
    `No auth function available for given request` error reported on the
    Judas session."""
    captured = {}

    class FakeDropbox:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    import sys as _sys
    fake_module = type(_sys)("dropbox")
    fake_module.Dropbox = FakeDropbox
    monkeypatch.setitem(_sys.modules, "dropbox", fake_module)

    cfg = {"refresh_token": "RT", "app_key": "AK", "app_secret": "AS",
           "access_token": "", "folder": "/results"}
    client, mode = cs._build_dropbox_client(cfg)
    assert mode == "refresh-token"
    assert captured["args"] == ()
    assert captured["kwargs"] == {
        "oauth2_refresh_token": "RT", "app_key": "AK", "app_secret": "AS"}


def test_build_dropbox_client_refresh_path_without_secret(monkeypatch):
    """Public-app path (PKCE): app_secret is empty and is NOT passed to the
    SDK. The previous regression that prevents the
    `Unable to refresh access token without refresh token and app key`
    error reported on the Joseph session — refresh-token mode activates
    even when app_secret is unset."""
    captured = {}

    class FakeDropbox:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    import sys as _sys
    fake_module = type(_sys)("dropbox")
    fake_module.Dropbox = FakeDropbox
    monkeypatch.setitem(_sys.modules, "dropbox", fake_module)

    cfg = {"refresh_token": "RT", "app_key": "AK", "app_secret": "",
           "access_token": "", "folder": "/results"}
    client, mode = cs._build_dropbox_client(cfg)
    assert mode == "refresh-token"
    assert captured["kwargs"] == {
        "oauth2_refresh_token": "RT", "app_key": "AK"}
    assert "app_secret" not in captured["kwargs"]


def test_build_dropbox_client_falls_back_to_legacy(monkeypatch):
    """When only access_token is configured (legacy mode), the SDK must be
    constructed positionally with the access token."""
    captured = {}

    class FakeDropbox:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    import sys as _sys
    fake_module = type(_sys)("dropbox")
    fake_module.Dropbox = FakeDropbox
    monkeypatch.setitem(_sys.modules, "dropbox", fake_module)

    cfg = {"refresh_token": "", "app_key": "", "access_token": "sl.u.AG",
           "folder": "/results"}
    client, mode = cs._build_dropbox_client(cfg)
    assert mode == "access-token (legacy)"
    assert captured["args"] == ("sl.u.AG",)
    assert captured["kwargs"] == {}


def test_finalize_invokes_dropbox_upload(tmp_path, monkeypatch):
    """finalize() must hand the whole config dict to _dropbox_upload — the
    new signature is (cfg, csv_paths) so the upload picks refresh-token vs
    legacy access-token construction itself."""
    calls = []
    monkeypatch.setattr(cs, "_dropbox_upload",
                        lambda cfg, paths, **kw:
                        calls.append((cfg, list(paths))))
    cfg = {"refresh_token": "RT", "app_key": "AK", "access_token": "",
           "folder": "/r"}
    rec = cs.SessionRecorder(
        "sess-dbx", PARTICIPANT, {"n_particles": 600},
        sessions_root=tmp_path / "sessions", synced_dir=None,
        dropbox_cfg=cfg, render_videos=False)
    rec.write_trial(_telemetry(0), _gui(0))
    rec.finalize(_result(n=1))
    rec.close()
    assert len(calls) == 1
    forwarded_cfg, paths = calls[0]
    assert forwarded_cfg["refresh_token"] == "RT"
    assert forwarded_cfg["app_key"] == "AK"
    assert forwarded_cfg["folder"] == "/r"
    assert len(paths) == 2 and all(p.exists() for p in paths)


def test_close_guards_further_writes(tmp_path):
    rec = _recorder(tmp_path)
    rec.close()
    rec.close()                                  # idempotent — no error
    rec.write_trial(_telemetry(0), _gui(0))      # guarded — no-op
    assert not (rec.dir / "trials.jsonl").read_text().strip()


def test_load_synced_dir_parsing(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    monkeypatch.setattr(cs, "CONFIG_PATH", cfg)
    cfg.write_text('synced_results_dir: ""\n')
    assert cs.load_synced_dir() is None
    cfg.write_text('synced_results_dir: "/tmp/cortex"\n')
    assert str(cs.load_synced_dir()) == "/tmp/cortex"


# ─── v1.2.9: staged progress + ETA from finalize ─────────────────────
def test_finalize_emits_progress_stages(tmp_path):
    """finalize(progress=cb) must drive a monotonic 0..1 fraction through
    named stages and end at ('Done', 1.0). render_videos=False so the slow
    MP4 stages are skipped (the fraction still completes)."""
    rec = _recorder(tmp_path)
    rec.write_trial(_telemetry(0), _gui(0))
    events = []
    rec.finalize(_result(n=1),
                 progress=lambda s, f, e: events.append((s, f, e)))
    rec.close()
    assert events, "no progress events emitted"
    stages = [s for s, _, _ in events]
    fracs = [f for _, f, _ in events]
    assert stages[-1] == "Done" and fracs[-1] == 1.0
    assert all(b >= a for a, b in zip(fracs, fracs[1:])), "frac not monotonic"
    assert "Calculating results" in stages
    assert "Saving results" in stages
    # ETA is None or a non-negative number of seconds
    assert all(e is None or e >= 0 for _, _, e in events)
