"""CORTEX internal test — diagnostic-log upload on uncaught exception.

When a CORTEX bundle crashes on a test-taker's machine — bank load
failure, config parse error, Qt init failure, mid-session traceback —
the operator has no console (PyInstaller ``console=False`` on macOS
silences stdout) and no straightforward way to ask the user for
``cortex.log``. This module installs a ``sys.excepthook`` that, on any
uncaught exception, builds a small diagnostic payload and uploads it to
Dropbox using **the exact same auth path** as ``cortex_storage`` — the
refresh-token client constructed by
``cortex_storage._build_dropbox_client`` and the upload loop in
``cortex_storage._dropbox_upload``.

Privacy contract: no PHI ships. The payload carries the exception
type/message/traceback, machine info (OS, Python, app version, frozen
flag), and the tail of ``cortex.log``. CSV-filename log lines that
contain the safe-name-sanitised participant name (e.g.
``<uuid>_<safe_name>_trials.csv``) are redacted before upload.

Failure semantics: the uploader is best-effort. Anything it does is
wrapped — a Dropbox outage, an expired refresh token, or a missing
``dropbox`` package must never replace the user's original traceback.
The original exception is always re-raised through the previously-
installed excepthook so the dev / debug experience is preserved.

Dev-mode gate: in source runs (``sys.frozen`` is False) the upload is
skipped unless ``CORTEX_DIAG_UPLOAD=1`` is exported — otherwise every
local crash during development would spam the shared Dropbox folder.
"""
from __future__ import annotations

import datetime
import getpass
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Optional

import cortex_storage as _cs

logger = logging.getLogger(__name__)

# Diagnostics live in their OWN top-level folder inside the Dropbox App
# folder — sibling to ``/results``, not nested under it — so debugging
# uploads never get mixed in with the test-taker result CSVs. The same
# refresh token reaches both folders (App-folder scope covers the whole
# app subtree). Overridable via cortex_config.yaml
# ``dropbox.diagnostics_folder``.
DEFAULT_DIAGNOSTICS_FOLDER = "/debugging-incidents"

# Cap on bytes of cortex.log to ship. ~200 KB is enough for several
# sessions' worth of warnings while staying small enough that even a
# slow connection completes the upload before the OS kills the app.
LOG_TAIL_BYTES = 200_000

# Regex matches the CSV filenames written by SessionRecorder
# (``<session_id>_<safe_name>_trials.csv`` /
# ``<session_id>_<safe_name>_summary.csv``). The middle group is the
# only PHI surface in cortex.log; scrub it to ``<redacted>``.
#
# Session-id class is alphanumeric+hyphen (NOT underscore) so the
# trailing literal ``_`` of group 1 cleanly separates session-id from
# the safe_name. Real session_ids are uuid4() — pure hex+hyphens — but
# the broader class keeps us robust if the id scheme ever changes.
_CSV_NAME_RE = re.compile(
    r"([A-Za-z0-9-]{6,}_)([A-Za-z0-9_]{1,40})(_trials\.csv|_summary\.csv)"
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _safe_subprocess(args, *, timeout: float = 2.0) -> Optional[str]:
    """Run ``args`` and return stripped stdout, or None on any failure.

    Used to query OS-specific introspection utilities (``scutil``,
    ``sysctl``) that may be missing, restricted, or slow on a given host.
    A short timeout prevents a hung utility from delaying the crash
    upload — diagnostics should always complete quickly.
    """
    try:
        result = subprocess.run(
            args, capture_output=True, text=True,
            timeout=timeout, check=False)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


def _computer_name() -> Optional[str]:
    """The 'pretty' computer name shown in OS settings, if obtainable.

    macOS: ``scutil --get ComputerName`` returns the Sharing-tab name
    (e.g. "Eli's MacBook Pro") rather than the .local hostname.
    Windows: ``%COMPUTERNAME%`` env var. Linux + fallback: socket
    hostname. Returns None only if every fallback fails.
    """
    if sys.platform == "darwin":
        name = _safe_subprocess(["scutil", "--get", "ComputerName"])
        if name:
            return name
    if sys.platform == "win32":
        name = os.environ.get("COMPUTERNAME") or ""
        if name.strip():
            return name.strip()
    try:
        return socket.gethostname() or None
    except OSError:
        return None


def _hardware_model() -> Optional[str]:
    """Best-effort hardware-model string. Returns None if not obtainable
    on the current platform.

    macOS: ``sysctl -n hw.model`` -> e.g. "MacBookPro18,3".
    Linux: ``/sys/devices/virtual/dmi/id/product_name`` -> e.g. "XPS 13 9310"
    (DMI table; readable on most distros, may be absent in containers).
    Windows: skipped — WMI/Powershell probes are slow and can prompt UAC;
    ``platform.platform()`` already embeds enough Windows info.
    """
    if sys.platform == "darwin":
        return _safe_subprocess(["sysctl", "-n", "hw.model"])
    if sys.platform.startswith("linux"):
        try:
            p = Path("/sys/devices/virtual/dmi/id/product_name")
            if p.exists():
                model = p.read_text(
                    encoding="utf-8", errors="replace").strip()
                return model or None
        except OSError:
            pass
    return None


def collect_machine_info() -> dict:
    """Best-effort host-identifying info, robust to any single failure.

    Every value may be None. The function never raises — it is called
    from inside the excepthook path, where a secondary exception would
    swallow the primary traceback.
    """
    try:
        hostname = socket.gethostname() or None
    except OSError:
        hostname = None
    try:
        username = getpass.getuser() or None
    except (OSError, KeyError, ImportError):
        # getpass.getuser() falls back to env vars then raises if none
        # are set (common on CI / locked-down service accounts).
        username = None
    try:
        computer_name = _computer_name()
    except Exception:                              # pragma: no cover
        computer_name = None
    try:
        model = _hardware_model()
    except Exception:                              # pragma: no cover
        model = None
    return {
        "hostname": hostname,
        "computer_name": computer_name,
        "username": username,
        "os_name": platform.system() or None,
        "os_release": platform.release() or None,
        "os_version": platform.version() or None,
        "model": model,
    }


def _read_log_tail(log_path: Path, n_bytes: int = LOG_TAIL_BYTES) -> str:
    """Return the last ``n_bytes`` of ``log_path``, or '' if unreadable."""
    try:
        size = log_path.stat().st_size
    except OSError:
        return ""
    try:
        with open(log_path, "rb") as fh:
            if size > n_bytes:
                fh.seek(size - n_bytes)
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _scrub_phi(text: str) -> str:
    """Replace participant-name substrings in CSV filenames with <redacted>."""
    return _CSV_NAME_RE.sub(r"\1<redacted>\3", text)


def collect_diagnostic_payload(exc_info=None, *, scrub: bool = True) -> dict:
    """Build a JSON-serialisable diagnostic dict from an exception.

    ``exc_info`` is a ``(type, value, tb)`` triple in the
    ``sys.exc_info()`` shape. If None, the current exception (if any) is
    used; if there's none, a payload with ``traceback=""`` is returned
    (still useful as a heartbeat).
    """
    if exc_info is None:
        exc_info = sys.exc_info()
    exc_type, exc_val, exc_tb = exc_info
    tb_str = ""
    if exc_type is not None:
        tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    log_tail = _read_log_tail(_cs.user_data_root() / "cortex.log")
    if scrub:
        tb_str = _scrub_phi(tb_str)
        log_tail = _scrub_phi(log_tail)
    return {
        "schema_version": 2,
        "timestamp_utc": _utc_now_iso(),
        "app_version": _cs.APP_VERSION,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
        "argv": list(sys.argv),
        # Host-identifying fields under one nested key so triage can grep
        # "machine_info" and see hostname / computer name / model at once.
        # Added in schema v2; older v1 payloads won't carry this block.
        "machine_info": collect_machine_info(),
        "exception_type": exc_type.__name__ if exc_type else None,
        "exception_message": str(exc_val) if exc_val is not None else None,
        "traceback": tb_str,
        "cortex_log_tail": log_tail,
        "cortex_log_tail_bytes": len(log_tail.encode("utf-8")),
    }


def _diagnostics_folder(cfg: dict) -> str:
    """Resolve the Dropbox path the diagnostic file should land in.

    Default: a fixed top-level ``/debugging-incidents`` inside the
    Dropbox App folder, deliberately disjoint from the results folder so
    crash dumps and test-taker CSVs never share a directory listing. If
    ``cortex_config.yaml`` sets ``dropbox.diagnostics_folder`` that
    overrides; the results folder (``dropbox.folder``) is ignored here.
    """
    explicit = str(cfg.get("diagnostics_folder") or "").strip()
    if explicit:
        return "/" + explicit.strip("/")
    return DEFAULT_DIAGNOSTICS_FOLDER


def _payload_filename() -> str:
    """Per-crash unique name so concurrent failures don't collide."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"cortex_diag_{ts}_{short}.json"


def upload_diagnostic(payload: dict, *,
                      cfg: Optional[dict] = "__auto__",
                      _client=None) -> bool:
    """Write ``payload`` to a temp file and upload it via Dropbox.

    Returns True iff the upload was attempted with a constructed client.
    Returns False (silently) when:
      * no Dropbox config is present in cortex_config.yaml, or
      * the ``dropbox`` package is missing, or
      * the file write or client construction itself fails.

    Reuses ``cortex_storage._dropbox_upload`` for the actual SDK calls;
    that path is the regression-tested one (refresh-token vs legacy
    access-token, per-file failure handling, identical folder semantics
    as the results upload).
    """
    if cfg == "__auto__":
        cfg = _cs.load_dropbox_config()
    if not cfg:
        logger.info("diagnostic upload skipped: no Dropbox config")
        return False
    # Use Application Support / app-data dir, not a process-local /tmp,
    # so the file survives if the upload fails and a future session can
    # potentially retry. (Best-effort: we do not currently retry.)
    out_dir = _cs.user_data_root() / "diagnostics"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("diagnostic dir create failed: %s", e)
        return False
    out_path = out_dir / _payload_filename()
    try:
        out_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except OSError as e:
        logger.warning("diagnostic file write failed: %s", e)
        return False
    # Re-use the canonical uploader, but with a per-call folder
    # override. _dropbox_upload reads cfg["folder"], so shadow that key.
    upload_cfg = dict(cfg)
    upload_cfg["folder"] = _diagnostics_folder(cfg)
    try:
        _cs._dropbox_upload(upload_cfg, [out_path], _client=_client)
    except Exception as e:                        # pragma: no cover
        # _dropbox_upload swallows per-file failures already; this is
        # only reachable if _build_dropbox_client itself raised. Still
        # non-fatal — the local copy under user_data_root/diagnostics
        # remains for offline retrieval.
        logger.warning("diagnostic upload failed at SDK layer: %s", e)
        return False
    return True


_PREV_EXCEPTHOOK = None
_INSTALLED = False


def _should_upload_in_dev() -> bool:
    return os.environ.get("CORTEX_DIAG_UPLOAD", "").strip().lower() in {
        "1", "true", "yes"}


def _excepthook(exc_type, exc_val, exc_tb) -> None:
    """sys.excepthook replacement: upload diagnostic, then chain.

    Everything here is wrapped: under no circumstance does the diagnostic
    path mask the original exception. The previously-installed hook
    (typically the stdlib default, which prints to stderr) is always
    invoked at the end so the user/operator still sees the traceback
    locally.
    """
    try:
        is_frozen = bool(getattr(sys, "frozen", False))
        if is_frozen or _should_upload_in_dev():
            payload = collect_diagnostic_payload((exc_type, exc_val, exc_tb))
            upload_diagnostic(payload)
        else:
            logger.info(
                "diagnostic upload skipped: dev mode without "
                "CORTEX_DIAG_UPLOAD=1")
    except Exception as e:                        # pragma: no cover
        # Even our own diagnostic layer crashing must not eat the
        # original traceback. Log and fall through.
        logger.warning("diagnostic excepthook itself raised: %s", e)
    # Chain to the previously-installed hook (or the default).
    hook = _PREV_EXCEPTHOOK or sys.__excepthook__
    try:
        hook(exc_type, exc_val, exc_tb)
    except Exception:                             # pragma: no cover
        # Last-ditch: print directly so the user still sees something.
        traceback.print_exception(exc_type, exc_val, exc_tb)


def install_excepthook() -> None:
    """Install the diagnostic-uploading sys.excepthook. Idempotent."""
    global _PREV_EXCEPTHOOK, _INSTALLED
    if _INSTALLED:
        return
    _PREV_EXCEPTHOOK = sys.excepthook
    sys.excepthook = _excepthook
    _INSTALLED = True
    logger.info(
        "diagnostic excepthook installed (frozen=%s, dev-upload=%s)",
        bool(getattr(sys, "frozen", False)), _should_upload_in_dev())


def _reset_for_tests() -> None:
    """Test helper: undo install_excepthook() so each test starts clean."""
    global _PREV_EXCEPTHOOK, _INSTALLED
    if _INSTALLED and _PREV_EXCEPTHOOK is not None:
        sys.excepthook = _PREV_EXCEPTHOOK
    _PREV_EXCEPTHOOK = None
    _INSTALLED = False
