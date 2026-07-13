"""The certification-test surface: bundle manifest, tutorial example, the
server-side per-session draw, per-question progress, and final results."""
from __future__ import annotations

import calendar
import json
import random
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import awards, dashboard_logic
from ..deps import require_auth
from ..models import ProgressIn, ResultsIn, SessionIn

router = APIRouter(prefix="/api")


@router.get("/manifest")
def manifest(req: Request, _code: str = Depends(require_auth)):
    # Real bundle identity. The SPA uses bundleUrl as the base for lazily-
    # fetched EEG/spec blobs; the per-session question SET comes from
    # POST /api/session (the server-side draw), so sessionSample is advisory.
    cfg = req.app.state.cfg
    bank = req.app.state.get_bank()
    return {
        "bundleUrl": cfg["bundle_url"],
        "version": bank.version if bank else None,
        "sessionSample": cfg["session_sample"],
    }


@router.get("/tutorial-example")
def tutorial_example(req: Request, _code: str = Depends(require_auth)):
    # One IIIC example segment for the tutorial walkthrough (goal 3: no
    # full-manifest fetch to the browser). 503 if the bank isn't configured.
    bank = req.app.state.get_bank()
    if bank is None:
        raise HTTPException(503, "question bank unavailable")
    return bank.example()


# One-directional test/train washout: STARTING an exam is blocked for
# WASHOUT_HOURS after a completed training sitting, so a certificate measures
# stable skill rather than a same-day practice boost. The other direction
# (training after an exam) stays open: a finished exam cannot be biased
# retroactively, and the product loop (test -> regimen -> train) depends on
# it. Item-level leakage is separately handled by the exposure exclusion in
# new_session. A rolling window (not a calendar day) closes the
# train-at-23:59, test-at-00:05 hole.
WASHOUT_HOURS = 12


def _iso_plus_hours(iso: str, hours: int) -> str:
    t = calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t + hours * 3600))


def _washout_reopens(db, code: str) -> str | None:
    """When the caller may start an exam again (ISO), or None if unblocked."""
    last = db.latest_training_finished(code)
    if not last:
        return None
    try:
        reopens = _iso_plus_hours(last, WASHOUT_HOURS)
    except ValueError:
        return None   # malformed legacy timestamp: never block on it
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return reopens if now < reopens else None


@router.post("/session")
def new_session(body: SessionIn, req: Request, code: str = Depends(require_auth)):
    # Server-side balanced draw (goal 3): pick this sitting's ~session_sample
    # questions from the full bank, excluding the participant's recently-seen
    # segments (goal 5 spacing), and stamp the bundle version (provenance O3).
    # The browser SMC engine runs over `bank.segments` exactly as before.
    db, cfg = req.app.state.db, req.app.state.cfg
    bank = req.app.state.get_bank()
    if bank is None:
        raise HTTPException(503, "question bank unavailable")
    reopens = _washout_reopens(db, code)
    if reopens is not None:
        # Machine-readable: the SPA turns this into the washout banner.
        return JSONResponse(status_code=409, content={
            "error": "training_washout", "reopensAtUtc": reopens})
    seed = (body.sampleSeed if body.sampleSeed is not None
            else random.randint(0, 2**31 - 1))
    exclude = db.get_exposure_exclusion(code, cfg["spacing_days"], cfg["spacing_sessions"])
    drawn = bank.draw(seed, cfg["session_sample"], exclude)
    session_id = uuid.uuid4().hex
    # A fresh sitting retires any still-open one (at most one resumable
    # session per account; see GET /api/session/active).
    db.supersede_open_sessions(code)
    # Persist the drawn candidate pool itself (a few KB of seg_ids): the
    # seed alone does NOT reproduce it later, because the exposure
    # exclusion is temporal — replaying the same seed after more sittings
    # yields a different pool. This makes every session exactly replayable.
    db.create_session(session_id, code, body.participant, seed,
                      bundle_version=bank.version,
                      drawn_seg_ids=json.dumps(
                          [s["segId"] for s in drawn["segments"]]))
    return {"sessionId": session_id, "sampleSeed": seed, "bank": drawn}


# Sessions older than this aren't offered for resume — the participant's
# context is gone, and a fresh spacing-aware draw serves them better.
RESUME_WINDOW_HOURS = 24


def _replay_prefix(trials: list[dict]) -> list[dict]:
    """The contiguous, fully-picked prefix of a sitting's trial log — the part
    the engine can deterministically replay. A gap or a pick-less row (a
    pre-answer crash artifact) ends what can be reconstructed."""
    replay: list[dict] = []
    for t in trials:
        if t.get("pick") is None or t["trial_index"] != len(replay):
            break
        replay.append({"trialIndex": t["trial_index"], "segId": t["seg_id"],
                       "pick": t["pick"]})
    return replay


def session_status(db, bank, code: str) -> dict:
    """Light resume/washout status for the dashboard bootstrap: the CTA labels
    need two facts, not the drawn-pool payload GET /api/session/active carries
    (hundreds of KB). Mirrors that endpoint's checks EXCEPT the bank-subset
    rebuild — deliberately optimistic there: the pre-flight click still fetches
    the full payload and falls back to a fresh start if the subset fails."""
    reopens = _washout_reopens(db, code)
    washout = {"reopensAtUtc": reopens} if reopens else None
    resumable = False
    row = db.get_active_session(code, max_age_hours=RESUME_WINDOW_HOURS)
    if (row is not None and bank is not None
            and row.get("bundle_version") == bank.version
            and row.get("drawn_seg_ids")):
        resumable = bool(_replay_prefix(db.session_trials(row["session_id"])))
    return {"examResumable": resumable, "washout": washout}


@router.get("/session/active")
def active_session(req: Request, code: str = Depends(require_auth)):
    """The participant's most recent resumable sitting: its exact drawn pool
    plus the logged trials — powers mid-test resume after a refresh/crash.
    Replay contract: the pool is returned verbatim in original draw order
    (sessions.drawn_seg_ids), the engine seed re-derives from the stored
    sample_seed (client uses `web-${sampleSeed}`), so feeding the logged
    picks back through the engine reconstructs its exact state. Returns
    {"active": null} when there is nothing (or nothing safe) to resume:
    no open sitting, sitting too old, bundle version changed, no trials yet,
    or a hole in the trial log."""
    db = req.app.state.db
    bank = req.app.state.get_bank()
    reopens = _washout_reopens(db, code)
    washout = {"reopensAtUtc": reopens} if reopens else None
    row = db.get_active_session(code, max_age_hours=RESUME_WINDOW_HOURS)
    if row is None or bank is None:
        return {"active": None, "washout": washout}
    if row.get("bundle_version") != bank.version or not row.get("drawn_seg_ids"):
        return {"active": None, "washout": washout}
    payload = bank.subset(json.loads(row["drawn_seg_ids"]))
    if payload is None:
        return {"active": None, "washout": washout}
    replay = _replay_prefix(db.session_trials(row["session_id"]))
    if not replay:
        # nothing checkpointed: a fresh start is equal
        return {"active": None, "washout": washout}
    payload["sampleSeed"] = row.get("sample_seed")
    return {"active": {
        "sessionId": row["session_id"],
        "startedUtc": row["started_utc"],
        "bank": payload,
        "trials": replay,
    }, "washout": washout}


@router.post("/progress")
def progress(body: ProgressIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    sess = db.get_session(body.sessionId)
    if sess is None or sess["code"] != code:
        raise HTTPException(404, "unknown session")
    db.upsert_trial(body.sessionId, body.trial)
    return {"ok": True}


@router.post("/results")
def results(body: ResultsIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    sess = db.get_session(body.sessionId)
    if sess is None or sess["code"] != code:
        raise HTTPException(404, "unknown session")
    # Derive the EVAL operating points (ℓ/θ/σ + median RT per task) BEFORE
    # writing, then store-result + finalize-session + trajectory-replace land
    # as ONE transaction — no half-finalized session can survive a crash
    # between statements.
    eval_pts = dashboard_logic.eval_points_from_result(
        body.result, db.session_trials(body.sessionId))
    db.store_result_finalized(body.sessionId, code, body.result,
                              body.stopReason, body.nQuestions, eval_pts)
    # Recognition AFTER the write lands: badge award/revoke + cert milestones
    # (awards.py). Internally best-effort; can never fail the ingest.
    awards.evaluate_certification(db, code, body.result)
    return {"ok": True}
