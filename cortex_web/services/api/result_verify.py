"""Server-side deterministic replay verification of ingested results.

The certification verdicts are computed in the participant's browser; result
ingest validates the profile STAMP but not the MATH, so a tampered client
could post fabricated verdicts under a valid stamp. This module closes that
gap: every complete precision sitting is re-derived server-side from its
stored picks — the engine is deterministic given (sampleSeed, exclusion,
stamp, picks) — and the replayed verdicts/intervals/trajectory are compared
against what the client submitted (scripts/replay_precision_session.ts,
bundled for plain node by scripts/build_replay_verifier.sh, the same
rolldown discipline as the n-way precision sidecar).

Verdicts recorded on the results row (persistence/migrations.py RESULTS):
  pass         — replay meaningfully equivalent to the submitted result
  divergent    — the engine does NOT reproduce the submitted result (flag
                 for review; the stored result is never mutated)
  unreplayable — the sitting's stamp is one the current engine no longer
                 serves (e.g. the retired 2026-07 unfloored profile);
                 terminal, never re-swept
  error        — the replay could not run (missing manifest, timeout,
                 crash); retried by the next sweep

Execution model: one verification at a time (module lock), fired on a
daemon thread after result ingest (CORTEX_RESULT_VERIFY=off disables the
hook), plus an explicit sweep entry point for backfill/restart recovery:

    /opt/cortex/.venv/bin/python -m api.result_verify --sweep 20
    /opt/cortex/.venv/bin/python -m api.result_verify --session <id>

A full replay re-runs the particle engine (~1–5 min per sitting on the
production host); the lock plus the sparse completion rate keeps that
negligible next to the API's normal load.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from . import config

_WEB_DIR = config.CORTEX_WEB / "apps" / "web"
_BUNDLE_SCRIPT = _WEB_DIR / "scripts" / "build_replay_verifier.sh"
_BUNDLE = _WEB_DIR / ".replay-verifier-dist" / "replay_precision_session.mjs"

# A tight-loop replay repays ALL the selection compute the browser hides in
# participant think-time (full-bank scoring per trial), so a ~200-trial
# sitting runs tens of minutes single-threaded. Fine for an async verifier.
REPLAY_TIMEOUT_S = int(os.environ.get("CORTEX_VERIFY_TIMEOUT_S", "3600"))
_DETAIL_CAP = 4000

_verify_lock = threading.Lock()


class VerifyError(Exception):
    """The replay could not be executed (environment, not divergence)."""


def _log(msg: str) -> None:
    print(f"[cortex.verify] {msg}", file=sys.stderr, flush=True)


def bundle_root() -> Path:
    """Where versioned EEG bundles live (manifest.json per version)."""
    env = os.environ.get("CORTEX_VERIFY_BUNDLE_ROOT")
    if env:
        return Path(env)
    prod = Path("/opt/cortex/bundle")
    if prod.is_dir():
        return prod
    return _WEB_DIR / "public" / "bundle"


def manifest_path(bundle_version: str) -> Path:
    path = bundle_root() / bundle_version / "manifest.json"
    if not path.is_file():
        raise VerifyError(f"manifest not found: {path}")
    return path


def ensure_bundle() -> Path:
    """Build the node replay bundle lazily (atomic publish; cheap when fresh)."""
    source = _WEB_DIR / "scripts" / "replay_precision_session.ts"
    if _BUNDLE.is_file() and _BUNDLE.stat().st_mtime >= source.stat().st_mtime:
        return _BUNDLE
    completed = subprocess.run(
        ["bash", str(_BUNDLE_SCRIPT)], capture_output=True, text=True,
        timeout=300,
    )
    if completed.returncode != 0 or not _BUNDLE.is_file():
        raise VerifyError(
            f"replay bundle build failed: {completed.stderr.strip()[-500:]}")
    return _BUNDLE


def export_fixture(db, session_id: str) -> tuple[dict, str]:
    """Rebuild the sanitized replay fixture for one complete sitting."""
    session = db.get_session(session_id)
    if session is None:
        raise VerifyError(f"unknown session {session_id}")
    if session.get("termination_policy") != "precision_v1":
        raise VerifyError("only precision_v1 sittings are replayable")
    result = db.get_result(session_id)
    if result is None:
        raise VerifyError("no stored result to verify against")
    trials = db.session_trials(session_id)
    if not trials:
        raise VerifyError("no stored trials to replay")
    fixture = {
        "sessionId": session_id,
        "sampleSeed": int(session.get("sample_seed") or 0),
        "exclusion": json.loads(session.get("candidate_exclusion") or "[]"),
        "trials": [
            {
                "pick": trial["pick"],
                "diag": json.loads(trial["diag"]) if trial.get("diag") else {},
            }
            for trial in trials
        ],
        "result": result,
        "stopReason": session.get("stop_reason"),
    }
    if session.get("nway_profile"):
        fixture["nwayProfile"] = json.loads(session["nway_profile"])
    return fixture, str(session.get("bundle_version") or "")


def _run_replay(fixture: dict, manifest: Path) -> dict:
    """Execute the node replay; return its summary JSON."""
    bundle = ensure_bundle()
    completed = subprocess.run(
        ["node", str(bundle), str(manifest)],
        input=json.dumps(fixture), capture_output=True, text=True,
        timeout=REPLAY_TIMEOUT_S, cwd=str(_WEB_DIR),
        # Lowest scheduling priority: verification must never compete with
        # the serving API for CPU on the production host.
        preexec_fn=lambda: os.nice(19),
    )
    if not completed.stdout.strip():
        raise VerifyError(
            f"replay produced no summary (rc={completed.returncode}): "
            f"{completed.stderr.strip()[-500:]}")
    try:
        return json.loads(completed.stdout)
    except ValueError:
        raise VerifyError(
            f"replay summary is not JSON (rc={completed.returncode}): "
            f"{completed.stdout.strip()[:300]}")


def _detail(summary: dict) -> str:
    trimmed = {
        "meaningfullyEquivalent": summary.get("meaningfullyEquivalent"),
        "expectedQuestions": summary.get("expectedQuestions"),
        "replayedQuestions": summary.get("replayedQuestions"),
        "exactSelectedItems": summary.get("exactSelectedItems"),
        "numericBitDifferences": summary.get("numericBitDifferences"),
        "maxAbsoluteDifference": summary.get("maxAbsoluteDifference"),
        "maxAbsoluteDifferencePath": summary.get("maxAbsoluteDifferencePath"),
        "trajectorySha256": summary.get("trajectorySha256"),
        "meaningfulDifferences": summary.get("meaningfulDifferences", [])[:10],
    }
    return json.dumps(trimmed)[:_DETAIL_CAP]


def verify_session(db, session_id: str) -> str:
    """Replay one sitting and record the verdict. Returns the status."""
    with _verify_lock:
        try:
            fixture, bundle_version = export_fixture(db, session_id)
            summary = _run_replay(fixture, manifest_path(bundle_version))
            status = "pass" if summary.get("meaningfullyEquivalent") else "divergent"
            db.set_result_verification(session_id, status, _detail(summary))
        except (VerifyError, subprocess.TimeoutExpired, OSError) as e:
            # A stamp the current engine refuses to serve (the retired
            # unfloored profile on pre-promotion sittings) can never replay
            # on this build: terminal, not retried.
            status = ("unreplayable"
                      if "profile mismatch" in str(e) else "error")
            db.set_result_verification(session_id, status,
                                       json.dumps({"error": str(e)[:1000]}))
        _log(f"{session_id}: {status}")
        return status


def schedule_verification(db, session_id: str) -> None:
    """Post-ingest fire-and-forget hook (serialized by the module lock)."""
    if os.environ.get("CORTEX_RESULT_VERIFY", "").strip().lower() == "off":
        return
    threading.Thread(target=verify_session, args=(db, session_id),
                     name=f"result-verify-{session_id[:8]}",
                     daemon=True).start()


def sweep(db, limit: int = 20) -> dict:
    """Verify every pending (or previously errored) complete sitting."""
    statuses: dict[str, str] = {}
    for session_id in db.unverified_result_sessions(limit):
        statuses[session_id] = verify_session(db, session_id)
    return statuses


def main() -> None:
    import argparse

    from .db import Database

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session")
    parser.add_argument("--sweep", type=int, metavar="LIMIT")
    parser.add_argument("--db", help="override CORTEX_DB")
    args = parser.parse_args()
    if not args.session and args.sweep is None:
        parser.error("pass --session <id> and/or --sweep <limit>")
    db = Database(args.db or os.environ.get("CORTEX_DB") or None)
    if args.session:
        print(json.dumps({args.session: verify_session(db, args.session)}))
    if args.sweep is not None:
        print(json.dumps(sweep(db, args.sweep)))


if __name__ == "__main__":
    main()
