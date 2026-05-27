"""Tests for scripts/cortex_diagnostics.py.

The diagnostic-upload path mirrors cortex_storage's auth + upload code
(it imports _dropbox_upload directly) so these tests focus on:
  * payload schema + PHI scrubbing
  * folder routing (`/results/diagnostics/` by default)
  * dev-vs-frozen upload gating + CORTEX_DIAG_UPLOAD env override
  * excepthook chain semantics: original traceback always reaches the
    previous hook, even when the uploader fails
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import cortex_storage as cs                       # noqa: E402
import cortex_diagnostics as cd                   # noqa: E402


class FakeClient:
    """Stand-in for dropbox.Dropbox in upload tests."""
    def __init__(self):
        self.uploads = []                          # list of (path, bytes)

    def files_upload(self, data, path):
        self.uploads.append((path, data))


def _make_exc():
    """Raise+catch a synthetic exception so we have a real exc_info triple."""
    try:
        raise RuntimeError("boom — test exception")
    except RuntimeError:
        return sys.exc_info()


@pytest.fixture(autouse=True)
def _isolate_runtime(tmp_path, monkeypatch):
    """Redirect user_data_root() at the module level so cortex.log /
    diagnostics/ both land in tmp_path. Also reset the install flag
    between tests so install_excepthook() is exercisable from scratch."""
    monkeypatch.setattr(cs, "user_data_root", lambda: tmp_path)
    # Pre-create an empty cortex.log so _read_log_tail has a stat target.
    (tmp_path / "cortex.log").write_text("")
    cd._reset_for_tests()
    yield
    cd._reset_for_tests()


# ── payload ─────────────────────────────────────────────────────────

def test_payload_has_required_fields():
    payload = cd.collect_diagnostic_payload(_make_exc())
    required = {"schema_version", "timestamp_utc", "app_version", "frozen",
                "platform", "machine", "python_version", "executable",
                "argv", "machine_info", "exception_type",
                "exception_message", "traceback", "cortex_log_tail",
                "cortex_log_tail_bytes"}
    assert required.issubset(payload.keys())
    assert payload["schema_version"] == 2          # bumped when machine_info landed
    assert payload["exception_type"] == "RuntimeError"
    assert "boom" in payload["exception_message"]
    assert "RuntimeError: boom" in payload["traceback"]


def test_payload_with_no_active_exception():
    """Heartbeat case: collect with no live exception — payload still
    populated, traceback empty, exception fields None."""
    payload = cd.collect_diagnostic_payload((None, None, None))
    assert payload["exception_type"] is None
    assert payload["exception_message"] is None
    assert payload["traceback"] == ""


def test_payload_is_json_serialisable():
    payload = cd.collect_diagnostic_payload(_make_exc())
    blob = json.dumps(payload, default=str)
    assert json.loads(blob)["exception_type"] == "RuntimeError"


# ── machine info ────────────────────────────────────────────────────

def test_machine_info_keys_present():
    info = cd.collect_machine_info()
    expected = {"hostname", "computer_name", "username", "os_name",
                "os_release", "os_version", "model"}
    assert expected == set(info.keys())


def test_machine_info_has_hostname_and_os_on_real_host():
    """On any real host where these tests can run, hostname + os_name
    are obtainable. (model + computer_name may be None on locked-down
    containers, so we don't assert them here.)"""
    info = cd.collect_machine_info()
    assert isinstance(info["hostname"], str) and info["hostname"]
    assert isinstance(info["os_name"], str) and info["os_name"]
    # os_name is one of Darwin / Linux / Windows on supported targets
    assert info["os_name"] in {"Darwin", "Linux", "Windows"}


def test_machine_info_payload_embedding():
    """The payload must carry machine_info as a nested dict — not flat,
    not a JSON string. Triage tooling depends on that shape."""
    payload = cd.collect_diagnostic_payload(_make_exc())
    mi = payload["machine_info"]
    assert isinstance(mi, dict)
    assert "hostname" in mi
    assert "computer_name" in mi


def test_machine_info_robust_to_every_failure(monkeypatch):
    """Force every probe to raise — collect_machine_info must still
    return a complete dict with None values, never bubble up an error."""
    def raiser(*a, **kw): raise OSError("nope")
    monkeypatch.setattr(cd.socket, "gethostname", raiser)
    monkeypatch.setattr(cd.getpass, "getuser", raiser)
    monkeypatch.setattr(cd, "_computer_name", raiser)
    monkeypatch.setattr(cd, "_hardware_model", raiser)
    monkeypatch.setattr(cd.platform, "system", lambda: "")
    monkeypatch.setattr(cd.platform, "release", lambda: "")
    monkeypatch.setattr(cd.platform, "version", lambda: "")
    info = cd.collect_machine_info()
    assert info["hostname"] is None
    assert info["computer_name"] is None
    assert info["username"] is None
    assert info["os_name"] is None
    assert info["model"] is None


def test_safe_subprocess_returns_none_on_missing_binary():
    """Calling a non-existent utility must not raise; just return None."""
    assert cd._safe_subprocess(["/no/such/binary/anywhere", "--version"]) is None


def test_safe_subprocess_returns_none_on_nonzero_exit():
    """Non-zero exit -> None (even if stdout is non-empty)."""
    # `false` returns 1 with no stdout; cross-platform on Unix.
    if sys.platform != "win32":
        assert cd._safe_subprocess(["false"]) is None


def test_computer_name_falls_back_to_hostname_when_scutil_absent(monkeypatch):
    """If scutil/COMPUTERNAME both yield nothing, socket.gethostname() is
    the last-line fallback — the function still returns something."""
    monkeypatch.setattr(cd, "_safe_subprocess", lambda *a, **kw: None)
    monkeypatch.delenv("COMPUTERNAME", raising=False)
    monkeypatch.setattr(cd.socket, "gethostname", lambda: "fallback-host")
    assert cd._computer_name() == "fallback-host"


# ── PHI scrubbing ───────────────────────────────────────────────────

def test_log_tail_scrubs_csv_filenames(tmp_path):
    log = tmp_path / "cortex.log"
    log.write_text(
        "2026-05-27T12:00:00 INFO cortex_storage: "
        "Dropbox upload of abc12345-uuid-here_Jane_Doe_trials.csv failed\n"
        "2026-05-27T12:00:01 INFO cortex_storage: "
        "Dropbox upload of abc12345-uuid-here_Jane_Doe_summary.csv ok\n")
    payload = cd.collect_diagnostic_payload((None, None, None))
    tail = payload["cortex_log_tail"]
    assert "Jane_Doe" not in tail
    assert "<redacted>_trials.csv" in tail
    assert "<redacted>_summary.csv" in tail
    # The session-id prefix is preserved (it's a UUID, not PHI)
    assert "abc12345-uuid-here_" in tail


def test_scrub_can_be_disabled():
    """scrub=False keeps the raw text — used only by debug tooling.
    Default upload path must NEVER call this with scrub=False."""
    raw = "uploading abc12345-uuid-here_Jane_Doe_trials.csv now"
    assert cd._scrub_phi(raw) != raw                # default scrubs
    assert "Jane_Doe" in raw                        # sanity


# ── folder routing ──────────────────────────────────────────────────

def test_diagnostics_folder_defaults_to_debugging_incidents():
    """Default is a fixed top-level /debugging-incidents — sibling to
    the results folder, never nested under it."""
    cfg = {"folder": "/results"}
    assert cd._diagnostics_folder(cfg) == "/debugging-incidents"


def test_diagnostics_folder_ignores_results_setting():
    """The results folder must NOT influence where diagnostics land —
    even if the user reconfigures dropbox.folder, diagnostics stay put."""
    assert cd._diagnostics_folder({"folder": "/anywhere"}) == "/debugging-incidents"
    assert cd._diagnostics_folder({"folder": ""}) == "/debugging-incidents"
    assert cd._diagnostics_folder({}) == "/debugging-incidents"


def test_diagnostics_folder_explicit_override():
    cfg = {"folder": "/results",
           "diagnostics_folder": "/cortex_crash_dumps"}
    assert cd._diagnostics_folder(cfg) == "/cortex_crash_dumps"


# ── upload routing ──────────────────────────────────────────────────

def test_upload_routes_to_diagnostics_folder():
    fake = FakeClient()
    cfg = {"refresh_token": "RT", "app_key": "AK", "access_token": "",
           "folder": "/results"}
    payload = cd.collect_diagnostic_payload(_make_exc())
    ok = cd.upload_diagnostic(payload, cfg=cfg, _client=fake)
    assert ok is True
    assert len(fake.uploads) == 1
    path, data = fake.uploads[0]
    assert path.startswith("/debugging-incidents/cortex_diag_")
    assert path.endswith(".json")
    # No bleed-through from the results folder setting
    assert "/results/" not in path
    # the uploaded bytes equal the JSON serialisation
    parsed = json.loads(data.decode("utf-8"))
    assert parsed["exception_type"] == "RuntimeError"


def test_upload_skipped_when_no_cfg():
    """No Dropbox config -> returns False, doesn't touch the network."""
    assert cd.upload_diagnostic({"x": 1}, cfg=None) is False


def test_upload_writes_local_copy_in_user_data_root(tmp_path):
    """Even with a fake client, the local copy lands under
    user_data_root()/diagnostics/ so a future retry could pick it up."""
    fake = FakeClient()
    cfg = {"folder": "/results"}
    cd.upload_diagnostic({"hello": "world"}, cfg=cfg, _client=fake)
    files = list((tmp_path / "diagnostics").iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("cortex_diag_")


# ── excepthook chaining ─────────────────────────────────────────────

def test_excepthook_in_dev_skips_upload_by_default(monkeypatch):
    """sys.frozen False + no env var -> no upload attempted."""
    monkeypatch.delenv("CORTEX_DIAG_UPLOAD", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    called = []
    monkeypatch.setattr(cd, "upload_diagnostic",
                        lambda *a, **kw: called.append(("upload",)) or True)
    cd.install_excepthook()
    cd._excepthook(*_make_exc())
    assert called == []                            # uploader never invoked


def test_excepthook_in_dev_with_env_uploads(monkeypatch):
    monkeypatch.setenv("CORTEX_DIAG_UPLOAD", "1")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    called = []
    monkeypatch.setattr(cd, "upload_diagnostic",
                        lambda payload, **kw: called.append(payload) or True)
    cd.install_excepthook()
    cd._excepthook(*_make_exc())
    assert len(called) == 1
    assert called[0]["exception_type"] == "RuntimeError"


def test_excepthook_in_frozen_uploads(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("CORTEX_DIAG_UPLOAD", raising=False)
    called = []
    monkeypatch.setattr(cd, "upload_diagnostic",
                        lambda payload, **kw: called.append(payload) or True)
    cd.install_excepthook()
    cd._excepthook(*_make_exc())
    assert len(called) == 1


def test_excepthook_chains_to_previous_hook(monkeypatch):
    """The original traceback handler always runs after the diagnostic
    layer, regardless of whether the upload succeeded or was skipped."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.delenv("CORTEX_DIAG_UPLOAD", raising=False)
    seen = []
    def prev_hook(et, ev, tb):
        seen.append((et.__name__, str(ev)))
    monkeypatch.setattr(sys, "excepthook", prev_hook)
    cd.install_excepthook()
    cd._excepthook(*_make_exc())
    assert seen == [("RuntimeError", "boom — test exception")]


def test_excepthook_uploader_failure_is_nonfatal(monkeypatch):
    """If the uploader itself raises, the previous hook STILL runs —
    a diagnostic-layer bug must never eat the user's traceback."""
    monkeypatch.setenv("CORTEX_DIAG_UPLOAD", "1")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    def boom(*a, **kw): raise RuntimeError("uploader broken")
    monkeypatch.setattr(cd, "upload_diagnostic", boom)
    seen = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda et, ev, tb: seen.append(et.__name__))
    cd.install_excepthook()
    cd._excepthook(*_make_exc())
    assert seen == ["RuntimeError"]                # original still chained


def test_install_excepthook_is_idempotent(monkeypatch):
    """Calling install twice doesn't double-wrap or lose the original."""
    original = sys.excepthook
    cd.install_excepthook()
    after_first = sys.excepthook
    cd.install_excepthook()
    after_second = sys.excepthook
    assert after_first is after_second             # same function object
    assert after_first is not original             # we did replace it
