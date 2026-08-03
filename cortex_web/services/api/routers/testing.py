"""The certification-test surface: tutorial example, the server-side
per-session draw, per-question progress, and final results. (The bundle
identity travels inside the POST /api/session draw payload.)"""
from __future__ import annotations

import json
import random
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import awards, dashboard_logic, result_verify, timeutil
from ..compute_rollout import compute_mode_for
from ..policy_rollout import (
    AD6_POLICY,
    PRECISION_POLICY,
    precision_recalibration_for,
    termination_policy_for,
)
from ..deps import require_auth
from ..models import ProgressBatchIn, ProgressIn, ResultsIn, SessionIn
from ..nway_profile import allowed_nway_profiles, nway_profile_for_new_session
from ..percentile_rollout import profile_for as percentile_profile_for
from ..percentile_runtime import validate_percentile_report

router = APIRouter(prefix="/api")


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
    return timeutil.iso_at(timeutil.parse_iso(iso) + hours * 3600)


def _washout_reopens(db, code: str) -> str | None:
    """When the caller may start an exam again (ISO), or None if unblocked."""
    last = db.latest_training_finished(code)
    if not last:
        return None
    try:
        reopens = _iso_plus_hours(last, WASHOUT_HOURS)
    except ValueError:
        return None   # malformed legacy timestamp: never block on it
    now = timeutil.utc_now()
    return reopens if now < reopens else None


@router.post("/session")
def new_session(body: SessionIn, req: Request, code: str = Depends(require_auth)):
    # AD6 keeps the server-side balanced ~session_sample draw. Precision
    # receives the full validated 35k candidate bank.
    # Both exclude recently-seen segments and stamp exact provenance.
    db, cfg = req.app.state.db, req.app.state.cfg
    termination_policy = termination_policy_for(db, cfg, code)
    compute_mode = compute_mode_for(db, cfg, code, termination_policy)
    participant_record = db.get_participant(code)
    percentile_profile = percentile_profile_for(
        req.app.state.percentile_runtime, cfg, code, participant_record)
    bank = (req.app.state.get_precision_bank()
            if termination_policy == PRECISION_POLICY
            else req.app.state.get_bank())
    if bank is None:
        label = ("precision_v1 bank"
                 if termination_policy == PRECISION_POLICY
                 else "question bank")
        raise HTTPException(503, f"{label} unavailable")
    reopens = _washout_reopens(db, code)
    if reopens is not None:
        # Machine-readable: the SPA turns this into the washout banner.
        return JSONResponse(status_code=409, content={
            "error": "training_washout", "reopensAtUtc": reopens})
    seed = (body.sampleSeed if body.sampleSeed is not None
            else random.randint(0, 2**31 - 1))
    exclude = db.get_exposure_exclusion(
        code, cfg["spacing_days"], cfg["spacing_sessions"])
    drawn = (
        (bank.lean(seed, exclude) if body.leanBank else bank.full(seed, exclude))
        if termination_policy == PRECISION_POLICY
        else bank.draw(seed, cfg["session_sample"], exclude))
    if termination_policy == PRECISION_POLICY:
        required = ("corrT", "nParticles", "perDomainCap", "precisionBandEdges")
        missing = [key for key in required if key not in drawn]
        if missing:
            raise HTTPException(
                503, f"precision_v1 bundle profile incomplete: {','.join(missing)}")
        if drawn["nParticles"] != 1200 or drawn["perDomainCap"] != 60:
            raise HTTPException(503, "precision_v1 bundle profile mismatch")
        # Refuse-by-default: the fail-closed response-model rollout decides
        # between the qualified draw-latent stamp and the floor015 mixture
        # (see nway_profile.py — off/unknown stamp the mixture; the research
        # escape is unreachable on production deployments regardless of env).
        nway_profile = nway_profile_for_new_session(
            bank.manifest_sha256, db=db, cfg=cfg, code=code)
        drawn["nwayProfile"] = nway_profile
    else:
        nway_profile = None
    drawn["terminationPolicy"] = termination_policy
    # Server-authoritative c1 stopping-recalibration stamp (n_min 0 /
    # persistence 3). Persisted with the sitting so resume and server-side
    # replay verification always run the sitting's own stopping constants;
    # absent => the shipped 20/2 configuration.
    precision_recalibration = precision_recalibration_for(
        db, cfg, code, termination_policy)
    if precision_recalibration:
        drawn["precisionRecalibration"] = precision_recalibration
    session_id = uuid.uuid4().hex
    # A fresh sitting retires any still-open one (at most one resumable
    # session per account; see GET /api/session/active).
    db.supersede_open_sessions(code)
    # AD6 persists the sampled ids. Precision persists the much smaller
    # temporal exclusion list plus exact manifest hash; together they recreate
    # the same full manifest-ordered candidate pool without storing 35k ids.
    db.create_session(session_id, code, body.participant, seed,
                      bundle_version=bank.version,
                      drawn_seg_ids=(
                          None if termination_policy == PRECISION_POLICY
                          else json.dumps([s["segId"] for s in drawn["segments"]])),
                      termination_policy=termination_policy,
                      compute_mode=compute_mode,
                      candidate_exclusion=(
                          json.dumps(sorted(exclude))
                          if termination_policy == PRECISION_POLICY else None),
                      candidate_bank_sha256=(
                          bank.manifest_sha256
                          if termination_policy == PRECISION_POLICY else None),
                      nway_profile=nway_profile,
                      precision_recalibration=precision_recalibration,
                      norm_profile=percentile_profile)
    return {"sessionId": session_id, "sampleSeed": seed,
            "terminationPolicy": termination_policy,
            "computeMode": compute_mode, "bank": drawn,
            "percentileProfile": percentile_profile}


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


def session_status(db, bank, code: str, precision_bank_getter=None) -> dict:
    """Light resume/washout status for the dashboard bootstrap: the CTA labels
    need two facts, not the drawn-pool payload GET /api/session/active carries
    (hundreds of KB). Mirrors that endpoint's checks EXCEPT the bank-subset
    rebuild — deliberately optimistic there: the pre-flight click still fetches
    the full payload and falls back to a fresh start if the subset fails."""
    reopens = _washout_reopens(db, code)
    washout = {"reopensAtUtc": reopens} if reopens else None
    resumable = False
    row = db.get_active_session(code, max_age_hours=RESUME_WINDOW_HOURS)
    if row and row.get("termination_policy") == PRECISION_POLICY:
        bank = precision_bank_getter() if precision_bank_getter else None
    candidate_replayable = bool(row and (
        (row.get("termination_policy") == PRECISION_POLICY
         and row.get("candidate_exclusion") is not None
         and row.get("candidate_bank_sha256") == bank.manifest_sha256
         and row.get("nway_profile") is not None)
        or row.get("drawn_seg_ids"))) if bank is not None else False
    if (row is not None and bank is not None
            and row.get("bundle_version") == bank.version
            and candidate_replayable):
        resumable = bool(_replay_prefix(db.session_trials(row["session_id"])))
    return {"examResumable": resumable, "washout": washout}


@router.get("/session/active")
def active_session(req: Request, code: str = Depends(require_auth),
                   lean: int = 0):
    """The participant's most recent resumable sitting: its exact drawn pool
    plus the logged trials — powers mid-test resume after a refresh/crash.
    Replay contract: AD6 returns sessions.drawn_seg_ids verbatim; Precision
    reconstructs manifest order from candidate_exclusion after a hash match.
    The engine seed re-derives from stored sample_seed, so feeding the logged
    picks back through the engine reconstructs its exact state. Returns
    {"active": null} when there is nothing (or nothing safe) to resume:
    no open sitting, sitting too old, bundle version changed, no trials yet,
    or a hole in the trial log."""
    db = req.app.state.db
    reopens = _washout_reopens(db, code)
    washout = {"reopensAtUtc": reopens} if reopens else None
    row = db.get_active_session(code, max_age_hours=RESUME_WINDOW_HOURS)
    bank = (req.app.state.get_precision_bank()
            if row and row.get("termination_policy") == PRECISION_POLICY
            else req.app.state.get_bank())
    if row is None or bank is None:
        return {"active": None, "washout": washout}
    if row.get("bundle_version") != bank.version:
        return {"active": None, "washout": washout}
    if row.get("termination_policy") == PRECISION_POLICY:
        if (row.get("candidate_exclusion") is None
                or row.get("candidate_bank_sha256") != bank.manifest_sha256
                or row.get("nway_profile") is None):
            return {"active": None, "washout": washout}
        exclusion = set(json.loads(row["candidate_exclusion"]))
        payload = (bank.lean(int(row.get("sample_seed") or 0), exclusion)
                   if lean
                   else bank.full(int(row.get("sample_seed") or 0), exclusion))
    else:
        if not row.get("drawn_seg_ids"):
            return {"active": None, "washout": washout}
        payload = bank.subset(json.loads(row["drawn_seg_ids"]))
    if payload is None:
        return {"active": None, "washout": washout}
    replay = _replay_prefix(db.session_trials(row["session_id"]))
    if not replay:
        # nothing checkpointed: a fresh start is equal
        return {"active": None, "washout": washout}
    payload["sampleSeed"] = row.get("sample_seed")
    payload["terminationPolicy"] = row.get("termination_policy") or AD6_POLICY
    if row.get("nway_profile"):
        try:
            stored_nway_profile = json.loads(row["nway_profile"])
        except (TypeError, ValueError):
            return {"active": None, "washout": washout}
        # Resume only under a stamp the current configuration could serve: a
        # research draw-latent sitting is refused once the escape is inactive
        # (never replayed under a different response model — fail closed).
        if stored_nway_profile not in allowed_nway_profiles(bank.manifest_sha256):
            return {"active": None, "washout": washout}
        payload["nwayProfile"] = stored_nway_profile
    if row.get("precision_recalibration"):
        # Resume replays under the sitting's stored stopping recalibration,
        # independent of the current rollout configuration.
        payload["precisionRecalibration"] = row["precision_recalibration"]
    stored_percentile_profile = None
    if row.get("norm_profile"):
        try:
            stored_percentile_profile = json.loads(row["norm_profile"])
        except (TypeError, ValueError):
            return {"active": None, "washout": washout}
        if not req.app.state.percentile_runtime.profile_matches(
            stored_percentile_profile
        ):
            return {"active": None, "washout": washout}
    return {"active": {
        "sessionId": row["session_id"],
        "startedUtc": row["started_utc"],
        "computeMode": row.get("compute_mode") or "serial",
        "bank": payload,
        "trials": replay,
        "percentileProfile": stored_percentile_profile,
    }, "washout": washout}


@router.post("/progress")
def progress(body: ProgressIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    sess = db.get_session(body.sessionId)
    if sess is None or sess["code"] != code:
        raise HTTPException(404, "unknown session")
    db.upsert_trial(body.sessionId, body.trial)
    return {"ok": True}


# The SPA's outbox coalesces checkpoints into small batches (and fires a
# keepalive batch on tab-hide); each batch lands in ONE transaction. The
# single-trial endpoint above stays for pre-batch tabs still running an older
# SPA build out of the append-only /assets path.
PROGRESS_BATCH_MAX = 100


@router.post("/progress/batch")
def progress_batch(body: ProgressBatchIn, req: Request,
                   code: str = Depends(require_auth)):
    if len(body.trials) > PROGRESS_BATCH_MAX:
        raise HTTPException(400, f"batch exceeds {PROGRESS_BATCH_MAX} trials")
    db = req.app.state.db
    sess = db.get_session(body.sessionId)
    if sess is None or sess["code"] != code:
        raise HTTPException(404, "unknown session")
    db.upsert_trials(body.sessionId, body.trials)
    return {"ok": True, "count": len(body.trials)}


@router.post("/results")
def results(body: ResultsIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    sess = db.get_session(body.sessionId)
    if sess is None or sess["code"] != code:
        raise HTTPException(404, "unknown session")
    expected_policy = sess.get("termination_policy") or AD6_POLICY
    reported_policy = body.result.get("terminationPolicy") or AD6_POLICY
    if reported_policy != expected_policy:
        raise HTTPException(409, "termination policy does not match session stamp")
    try:
        stored_nway_profile = (
            json.loads(sess["nway_profile"]) if sess.get("nway_profile") else None)
    except (TypeError, ValueError):
        raise HTTPException(409, "stored n-way profile is invalid")
    if body.result.get("nwayProfile") != stored_nway_profile:
        raise HTTPException(409, "n-way profile does not match session stamp")
    try:
        stored_percentile_profile = (
            json.loads(sess["norm_profile"]) if sess.get("norm_profile") else None)
    except (TypeError, ValueError):
        raise HTTPException(409, "stored percentile profile is invalid")
    if stored_percentile_profile is not None \
            and body.result.get("percentile") is None:
        # A browser tab loaded before the percentile-capable SPA was deployed
        # cannot calculate the new report. Preserve the core assessment result
        # and persist the absence explicitly; malformed reports still fail.
        body.result["percentile"] = {
            "status": "unavailable_legacy_client",
            "profile": stored_percentile_profile,
        }
    percentile_errors = validate_percentile_report(
        stored_percentile_profile, body.result.get("percentile"))
    if percentile_errors:
        raise HTTPException(409, "; ".join(percentile_errors))
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
    # Server-side deterministic replay of the sitting (fire-and-forget,
    # serialized worker): the client computed these verdicts, so the server
    # re-derives them from the stored picks and records agreement on the
    # results row (api.result_verify). Precision sittings only.
    if expected_policy == PRECISION_POLICY:
        result_verify.schedule_verification(db, body.sessionId)
    return {"ok": True}
