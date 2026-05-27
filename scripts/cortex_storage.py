"""CORTEX internal test — per-session storage + synced CSV delivery (Phase D).

A SessionRecorder persists one test-taker's session:

  results/sessions/<session_id>/
    participant.json   identity + session metadata (PHI; stays local)
    trials.jsonl       one JSON line per question, flush+fsync (crash-safe)
    events.jsonl       one JSON line per interaction-trace event
    certificate.json   per-task AUROC estimates, written at session end
    trajectory.npz     the SMC particle-cloud trajectory

On normal completion it also writes two CSVs into the cloud-synced folder
(`synced_results_dir` in cortex_config.yaml): a detailed one-row-per-
question file and a one-row session summary. Each session writes its own
files — no shared file, so multi-machine cloud sync never collides.
Aborted sessions write only the local crash-safe files.
"""
from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Frozen PyInstaller bundles: __file__ for a PYZ-loaded module does not
# resolve to a real filesystem path whose parent.parent is the data unpack
# root. sys._MEIPASS is set by the bootloader to that root; the bundled
# cortex_config.yaml is dropped at the unpack root by cortex.spec.
if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
else:
    _REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = _REPO / "cortex_config.yaml"


def user_data_root() -> Path:
    """Return the directory CORTEX should write per-session output to.

    Dev runs (`python scripts/...`) write under <repo>/results/ — handy
    for debugging from the working tree.

    Frozen runs (PyInstaller-bundled .app/.exe) write under the user's
    platform-appropriate app-data dir. The .app bundle itself is
    read-only — both as a general principle and specifically under
    macOS App Translocation, where downloaded unsigned bundles are run
    from a temporary read-only path — so writing inside the bundle
    aborts the first registration attempt.
    """
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "CORTEX"
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA") or str(
                Path.home() / "AppData" / "Local")
            return Path(base) / "CORTEX"
        # Linux / other Unix — XDG_DATA_HOME if set, else ~/.local/share
        base = os.environ.get("XDG_DATA_HOME") or str(
            Path.home() / ".local" / "share")
        return Path(base) / "CORTEX"
    return _REPO / "results"


SESSIONS_ROOT = user_data_root() / "sessions"
SCHEMA_VERSION = 1
APP_VERSION = "cortex-internal-0.1"

_LOGGING_CONFIGURED = False


def setup_logging():
    """Configure CORTEX logging. Idempotent.

    Always writes to ``user_data_root() / "cortex.log"`` so the bundled
    .app/.exe has a recoverable record of Dropbox upload status, video
    render outcome, and warnings — PyInstaller ``console=False`` Mac
    bundles silence stdout, so without this the operator has no
    visibility post-session.

    In dev mode (``sys.frozen`` is False), also attaches a stderr
    StreamHandler so the same messages still appear in the terminal.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    log_path = user_data_root() / "cortex.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    if not getattr(sys, "frozen", False):
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    _LOGGING_CONFIGURED = True
    logger.info("logging initialised; log_path=%s frozen=%s",
                log_path, getattr(sys, "frozen", False))

_TASK_CODES = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]

# trials.jsonl/CSV scalar columns — the per-task [6]-arrays stay in the
# JSONL only; the CSV is kept flat for spreadsheet use.
_DETAIL_COLS = ["session_id", "participant_name", "trial_index", "seg_id",
                "task_code", "pattern_class_true", "response_label",
                "response_y", "is_correct", "reaction_time_ms",
                "answer_changes", "n_interactions", "select_ms", "montage",
                "gain_uv", "bandpass", "notch", "window_s", "max_hw", "ess",
                "total_var", "expected_loss_chosen", "rejuv"]


def load_synced_dir():
    """Return the configured cloud-synced results dir as a Path, or None."""
    if not CONFIG_PATH.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    d = str(cfg.get("synced_results_dir") or "").strip()
    return Path(d).expanduser() if d else None


def load_dropbox_config():
    """Return the Dropbox upload config if one is set in cortex_config.yaml,
    else None.

    Two auth modes are supported. **Refresh-token mode (preferred)** keeps
    the bundle working indefinitely — the SDK swaps the access token via
    Dropbox's OAuth endpoint when it expires:

        dropbox:
          app_key:       "<your app key>"
          app_secret:    "<your app secret — required for Confidential apps>"
          refresh_token: "<from scripts/cortex_dropbox_oauth.py>"
          folder:        "/results"

    ``app_secret`` is required for Confidential clients (Dropbox's default
    app type) — without it, the SDK's refresh call hits "No auth function
    available for given request" at /oauth2/token. For Public (PKCE-only)
    apps it can be empty.

    **Legacy mode** uses a short-lived access token (the `sl.u.AG…` prefix
    Dropbox issues from the app console "Generated access token" button).
    These expire in ~4 hours and the bundle then stops uploading until the
    token is regenerated. Retained only for one-off dev sessions:

        dropbox:
          access_token:  "sl.u.AG…"
          folder:        "/results"

    Returned dict carries all keys (refresh_token, app_key, app_secret,
    access_token, folder); missing fields are empty strings. The upload
    path inspects which credentials are present and constructs the SDK
    client accordingly. Returns None if neither auth mode is configured.
    """
    if not CONFIG_PATH.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    dbx = cfg.get("dropbox") or {}
    refresh = str(dbx.get("refresh_token") or "").strip()
    app_key = str(dbx.get("app_key") or "").strip()
    app_secret = str(dbx.get("app_secret") or "").strip()
    access = str(dbx.get("access_token") or "").strip()
    has_refresh = bool(refresh and app_key)
    if not has_refresh and not access:
        return None
    return {
        "refresh_token": refresh,
        "app_key": app_key,
        "app_secret": app_secret,
        "access_token": access,
        "folder": str(dbx.get("folder") or "/results").strip(),
    }


def _build_dropbox_client(cfg):
    """Construct a real dropbox.Dropbox client from a load_dropbox_config()
    dict. Refresh-token credentials take precedence; falls back to the
    legacy access-token-only path if only that's configured. Returns
    ``(client, mode_label)`` or raises on failure.

    For Confidential apps (Dropbox's default app type), `app_secret` is
    required at every token refresh — not just the initial exchange. Without
    it, the SDK's refresh call to /oauth2/token returns
    "No auth function available for given request". For Public (PKCE-only)
    apps, `app_secret` can be empty.
    """
    import dropbox
    if cfg.get("refresh_token") and cfg.get("app_key"):
        # Refresh-token mode: the SDK keeps the access token alive by
        # exchanging the refresh token at Dropbox's /oauth2/token endpoint
        # when the short-lived access token expires.
        kwargs = {
            "oauth2_refresh_token": cfg["refresh_token"],
            "app_key": cfg["app_key"],
        }
        if cfg.get("app_secret"):
            kwargs["app_secret"] = cfg["app_secret"]
        client = dropbox.Dropbox(**kwargs)
        return client, "refresh-token"
    # Legacy short-lived access-token path. Expires in ~4 hours.
    client = dropbox.Dropbox(cfg["access_token"])
    return client, "access-token (legacy)"


def _dropbox_upload(cfg, csv_paths, _client=None):
    """Upload result CSVs to Dropbox. All failures are non-fatal — the local
    copies under results/sessions/ remain the source of truth.

    `cfg` is a dict from load_dropbox_config() with keys ``folder`` plus
    either (``refresh_token``, ``app_key``) for the preferred long-lived
    path, or ``access_token`` for the legacy short-lived path. `_client`
    is a test-injection point standing in for a real Dropbox client; when
    provided it bypasses the auth-construction logic entirely.
    """
    client = _client
    if client is None:
        try:
            import dropbox  # noqa: F401  (presence check before _build)
        except ImportError:
            logger.warning(
                "'dropbox' package not installed — upload skipped "
                "(local CSVs retained). Run: pip install dropbox")
            return
        try:
            client, mode = _build_dropbox_client(cfg)
            logger.info("Dropbox client constructed (%s mode)", mode)
        except Exception as e:                       # pragma: no cover
            logger.warning("Dropbox client init failed (%s)", e)
            return
    base = cfg.get("folder", "/results").strip("/")
    base = ("/" + base) if base else ""
    ok = 0
    for p in csv_paths:
        try:
            client.files_upload(p.read_bytes(), f"{base}/{p.name}")
            ok += 1
        except Exception as e:
            logger.warning("Dropbox upload of %s failed (%s)", p.name, e)
    if ok:
        logger.info("uploaded %d result CSV(s) to Dropbox %s/", ok, base)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))[:40] or "anon"


def _as_list(v):
    """None -> []; an array/sequence -> list (avoids numpy truthiness)."""
    return [] if v is None else list(v)


class SessionRecorder:
    """Writes one session's results. Created at session start, fed each
    trial via write_trial(), closed out with finalize() then close()."""

    def __init__(self, session_id, participant, session_config,
                 sessions_root=None, synced_dir="__auto__",
                 dropbox_cfg="__auto__", render_videos=True):
        self.session_id = str(session_id)
        self.participant = dict(participant or {})
        self.session_config = dict(session_config or {})
        root = Path(sessions_root) if sessions_root else SESSIONS_ROOT
        self.dir = root / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        if synced_dir == "__auto__":
            self.synced_dir = load_synced_dir()
        else:
            self.synced_dir = Path(synced_dir) if synced_dir else None
        self.dropbox_cfg = (load_dropbox_config()
                            if dropbox_cfg == "__auto__" else dropbox_cfg)
        # Whether finalize() auto-renders the per-test-taker MP4 visualizations
        # (collapse.mp4 + passfail.mp4 into self.dir). Tests opt out to keep
        # the suite fast and avoid an ffmpeg subprocess per recorder.
        self.render_videos = bool(render_videos)
        self._closed = False
        self._trials = []
        self._started_utc = _utc_now()
        self._trials_fh = open(self.dir / "trials.jsonl", "a", encoding="utf-8")
        self._events_fh = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._write_participant()

    # ── identity ────────────────────────────────────────────────────────
    def _write_participant(self):
        doc = {
            "session_id": self.session_id,
            "schema_version": SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "started_utc": self._started_utc,
            "identity": self.participant,            # PHI — stays local
            "session_config": self.session_config,
        }
        (self.dir / "participant.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8")

    # ── per-trial ───────────────────────────────────────────────────────
    @staticmethod
    def _merge(telemetry, gui):
        """Merge engine telemetry + GUI metadata into one flat trial record.
        `gui` may be None (defensive)."""
        gui = gui or {}
        rec = dict(telemetry)
        for k in ("response_raw", "response_label", "reaction_time_ms",
                  "answer_changes", "montage", "gain_uv", "bandpass",
                  "notch", "window_s", "pan_t_start"):
            if k in gui:
                rec[k] = gui[k]
        rec["n_interactions"] = len(gui.get("interaction", []))
        pc = rec.get("pattern_class_true")
        label = str(gui.get("response_label", "")).lower()
        rec["is_correct"] = bool(label and pc and label == pc)
        return rec

    def write_trial(self, telemetry, gui):
        """Persist one answered question — crash-safe (flush + fsync)."""
        if self._closed:
            return
        rec = self._merge(telemetry, gui)
        self._trials.append(rec)
        line = {k: v for k, v in rec.items() if k != "interaction"}
        self._trials_fh.write(json.dumps(line, default=float) + "\n")
        self._trials_fh.flush()
        os.fsync(self._trials_fh.fileno())
        for ev in (gui or {}).get("interaction", []):
            self._events_fh.write(json.dumps(
                {"session_id": self.session_id,
                 "trial_index": rec.get("trial_index"), **ev}) + "\n")
        self._events_fh.flush()
        os.fsync(self._events_fh.fileno())

    # ── end of session ──────────────────────────────────────────────────
    def finalize(self, result):
        """Write certificate.json + trajectory.npz; on a clean (non-aborted)
        completion also the result CSVs — kept locally and delivered to the
        synced folder and/or Dropbox if either is configured."""
        if self._closed:
            return
        finished = _utc_now()
        self._write_certificate(result, finished)
        self._write_trajectory(result)
        # Per-test-taker MP4 visualizations — wow-factor delivery alongside
        # the certificate. Wrapped in try/except so a missing ffmpeg or
        # matplotlib failure can never block the rest of finalize().
        # Renderer reads trajectory.npz + trials.jsonl + certificate.json +
        # participant.json from self.dir, so it MUST run after the writes
        # above. Aborted sessions skip rendering — the partial trajectory
        # is not worth a 15s render.
        if (self.render_videos
                and result is not None
                and not getattr(result, "aborted", False)
                and getattr(result, "t_traj", None) is not None):
            try:
                from cortex_render_videos import render_all
                render_all(self.dir)
            except Exception as e:
                logger.warning("per-test-taker video render failed: %s", e)
            # v1.1.3: engine-explainer MP4 — independent failure budget
            # so a problem in one render path can't suppress the other.
            try:
                from render_engine_explainer import render_engine_explainer
                render_engine_explainer(self.dir)
            except Exception as e:
                logger.warning("engine_explainer render failed: %s", e)
        if result is None or getattr(result, "aborted", False):
            return
        try:
            csv_paths = self._write_result_csvs(result, finished)
        except Exception as e:
            logger.warning("could not write result CSVs: %s", e)
            return
        # (1) copy into a local cloud-synced folder, if configured
        if self.synced_dir is not None:
            try:
                self.synced_dir.mkdir(parents=True, exist_ok=True)
                for p in csv_paths:
                    (self.synced_dir / p.name).write_bytes(p.read_bytes())
            except Exception as e:
                logger.warning("synced-folder copy failed: %s", e)
        # (2) upload directly to Dropbox, if configured
        if self.dropbox_cfg is not None:
            _dropbox_upload(self.dropbox_cfg, csv_paths)

    def _write_certificate(self, result, finished_utc):
        per_task = []
        codes = list(getattr(result, "task_codes", _TASK_CODES) or _TASK_CODES)
        am = _as_list(getattr(result, "final_auroc_mean", None))
        ah = _as_list(getattr(result, "final_auroc_hw", None))
        lm = _as_list(getattr(result, "final_l_mean", None))
        tm = _as_list(getattr(result, "final_t_mean", None))
        verdicts = getattr(result, "verdicts", None) or []
        for i, code in enumerate(codes):
            per_task.append({
                "task_code": code,
                "auroc_mean": float(am[i]) if i < len(am) else None,
                "auroc_halfwidth": float(ah[i]) if i < len(ah) else None,
                "ell_mean": float(lm[i]) if i < len(lm) else None,
                "t_mean": float(tm[i]) if i < len(tm) else None,
                "verdict": verdicts[i] if i < len(verdicts) else None,
            })
        n = len(self._trials)
        n_correct = sum(1 for t in self._trials if t.get("is_correct"))
        doc = {
            "session_id": self.session_id,
            "finished_utc": finished_utc,
            "total_trials": n,
            "n_correct": n_correct,
            "accuracy": (n_correct / n) if n else None,
            "stop_reason": getattr(result, "stop_reason", None),
            "delta_auroc": getattr(result, "delta_auroc", None),
            "aborted": bool(getattr(result, "aborted", False)),
            "per_task": per_task,
        }
        (self.dir / "certificate.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8")

    def _write_trajectory(self, result):
        if result is None or getattr(result, "t_traj", None) is None:
            return
        try:
            # AD6 production sessions instantiate with delta_auroc=None; the
            # npz writer encodes that as NaN so a fixed dtype is preserved.
            delta = result.delta_auroc
            delta_val = float(delta) if delta is not None else float("nan")
            np.savez_compressed(
                self.dir / "trajectory.npz",
                t_traj=result.t_traj, l_traj=result.l_traj,
                w_traj=result.w_traj,
                task_codes=np.array(list(result.task_codes)),
                seg_ids=np.array(list(result.served_seg_ids)),
                delta_auroc=delta_val,
                n_questions=int(result.n_questions))
        except Exception as e:
            logger.warning("could not write trajectory.npz: %s", e)

    def _write_result_csvs(self, result, finished_utc):
        """Write the detail + summary CSVs into the session directory; return
        their paths (the canonical local copies, distributed by finalize)."""
        name = self.participant.get("name", "anon")
        stem = f"{self.session_id}_{_safe_name(name)}"
        trials_path = self.dir / f"{stem}_trials.csv"
        summary_path = self.dir / f"{stem}_summary.csv"
        # detailed — one row per question
        with open(trials_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_DETAIL_COLS,
                               extrasaction="ignore")
            w.writeheader()
            for t in self._trials:
                row = dict(t)
                row["session_id"] = self.session_id
                row["participant_name"] = name
                w.writerow(row)
        # summary — one row per session
        summ = self._summary_row(result, finished_utc, name)
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summ.keys()))
            w.writeheader()
            w.writerow(summ)
        return [trials_path, summary_path]

    def _summary_row(self, result, finished_utc, name):
        n = len(self._trials)
        n_correct = sum(1 for t in self._trials if t.get("is_correct"))
        rts = [t["reaction_time_ms"] for t in self._trials
               if t.get("reaction_time_ms") is not None]
        # Surface demographic + clinical-background fields from the
        # RegistrationPage row into the uploaded summary so the data is
        # mineable globally (registrations.csv stays local-only).
        # Missing keys default to "" — back-compatible with pre-v1.1.1
        # participant dicts that don't carry the new fields.
        p = self.participant
        row = {
            "session_id": self.session_id,
            "participant_name": name,
            "expertise": p.get("expertise", ""),
            "institution": p.get("institution", ""),
            "started_utc": self._started_utc,
            "finished_utc": finished_utc,
            "n_questions": n,
            "stop_reason": getattr(result, "stop_reason", None),
            "accuracy": round(n_correct / n, 4) if n else None,
            "mean_rt_ms": round(float(np.mean(rts)), 1) if rts else None,
            "delta_auroc": getattr(result, "delta_auroc", None),
            # v1.1.1 mineable demographic fields
            "practice_setting": p.get("practice_setting", ""),
            "years_reading_eeg": p.get("years_reading_eeg", ""),
            "eeg_volume_per_month": p.get("eeg_volume_per_month", ""),
            "self_rated_confidence": p.get("self_rated_confidence", ""),
            "color_vision": p.get("color_vision", ""),
            "prior_test_taken": p.get("prior_test_taken", ""),
            "sex": p.get("sex", ""),
            "country": p.get("country", ""),
            "race_ethnicity": p.get("race_ethnicity", ""),
            "consent_version": p.get("consent_version", ""),
            "irb_protocol_id": p.get("irb_protocol_id", ""),
        }
        codes = list(getattr(result, "task_codes", _TASK_CODES) or _TASK_CODES)
        am = _as_list(getattr(result, "final_auroc_mean", None))
        ah = _as_list(getattr(result, "final_auroc_hw", None))
        lm = _as_list(getattr(result, "final_l_mean", None))
        tm = _as_list(getattr(result, "final_t_mean", None))
        for i, code in enumerate(codes):
            row[f"auroc_{code}"] = round(float(am[i]), 4) if i < len(am) else None
            row[f"hw_{code}"] = round(float(ah[i]), 4) if i < len(ah) else None
            row[f"ell_{code}"] = round(float(lm[i]), 4) if i < len(lm) else None
            row[f"t_{code}"] = round(float(tm[i]), 4) if i < len(tm) else None
        return row

    def close(self):
        """Close the append-only logs. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for fh in (self._trials_fh, self._events_fh):
            try:
                fh.close()
            except Exception:
                pass
