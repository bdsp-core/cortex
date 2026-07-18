"""Dashboard, history, and learning-protocol surfaces.

Real certification data only: per-task ℓ/θ/ℓ*/AUROC + verdict from the
latest result, plus cert-summary KPIs. Learning-protocol surfaces (regimen,
trajectories, training sessions) carry NO data until the L1 trainer ships;
`sample` stays in the contracts (always False) so the client keeps a single
response shape.
"""
from __future__ import annotations

import random
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import awards, dashboard_logic, engine_trainer
from ..deps import require_auth
from ..models import (
    SessionIn, TrainingFinalizeIn, TrainingProgressIn, TrainingStartIn,
    TrajectoryIn)

router = APIRouter(prefix="/api")

# A sitting is ≤ a few hundred questions; anything past this is a client bug
# or abuse, not data (the outbox posts small batches).
MAX_POINTS_PER_POST = 1000


def _validate_points(points: list[dict]) -> None:
    """Reject a malformed batch up front: every point needs an integer taskK
    (and an integer segId when present). Without this a bad point raised
    KeyError mid-transaction → a 500 that discarded the whole batch."""
    if len(points) > MAX_POINTS_PER_POST:
        raise HTTPException(413, f"too many points (max {MAX_POINTS_PER_POST})")
    for i, p in enumerate(points):
        try:
            int(p["taskK"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(422, f"point {i}: integer 'taskK' required")
        if p.get("segId") is not None:
            try:
                int(p["segId"])
            except (TypeError, ValueError):
                raise HTTPException(422, f"point {i}: 'segId' must be an integer")


def _training_enabled(req: Request, code: str) -> bool:
    """Resolve the training-exposure flag for this participant. Only hits the DB
    for a participant lookup in cohort mode (all/off need no row)."""
    cfg = req.app.state.cfg
    participant = (req.app.state.db.get_participant(code)
                   if cfg.get("training_mode") == "cohort" else None)
    return dashboard_logic.training_enabled(cfg, code, participant)


# ── payload builders ──
# Module-level so GET /api/bootstrap composes the SAME payloads the standalone
# endpoints return (one implementation, two routes). Referenced through the
# module attribute in bootstrap.py so test monkeypatching reaches them.

def dashboard_payload(req: Request, code: str) -> dict:
    db = req.app.state.db
    # LIMIT-1 fetch: the dashboard reads only the newest attempt, so it must
    # not pull + JSON-parse every historical ~370 KB result blob.
    latest = db.latest_result_for_code(code)
    training_enabled = _training_enabled(req, code)
    training_resumable = db.has_training_history(code)
    if latest is None:
        return {"result": None, "hasResult": False, "tasks": [],
                "kpis": None, "sample": False,
                "trainingEnabled": training_enabled,
                "trainingResumable": training_resumable}
    result = latest["result"]
    # Latest real measurement per domain (rows come ordered by task_k, ts
    # ascending, so the last one seen per task is the most recent).
    latest_traj: dict[int, dict] = {}
    for r in db.get_trajectories(code):
        latest_traj[int(r["task_k"])] = r
    tasks = dashboard_logic.dashboard_tasks(result, latest_traj)
    return {
        # Slimmed: the raw per-trial bulk (~95% of a ~370 KB blob) never
        # reaches the dashboard — the Shell reads only hasResult/tasks/kpis.
        "result": dashboard_logic.slim_result(result),
        "hasResult": True,
        "tasks": tasks,
        "kpis": dashboard_logic.dashboard_kpis(tasks, latest.get("finished_utc")),
        "sample": False,
        "trainingEnabled": training_enabled,
        "trainingResumable": training_resumable,
    }


def activity_payload(db, code: str, tz: int) -> dict:
    # Per-day activity levels for the consistency heatmap (sign-in / cert /
    # training), keyed by the user's LOCAL date. `tz` = getTimezoneOffset().
    return {"days": db.activity_levels(code, tz)}


def regimen_payload(db, code: str) -> dict:
    reg = db.get_active_regimen(code)
    return {"regimen": reg["plan"] if reg is not None else None, "sample": False}


def trajectories_payload(db, code: str) -> dict:
    rows = db.get_trajectories(code)
    pts = [{"taskK": r["task_k"], "phase": r["phase"], "ell": r["ell"],
            "theta": r["theta"], "sd": r["sd"], "rt": r["rt"], "ts": r["ts"]}
           for r in rows]
    return {"trajectories": pts, "sample": False}


@router.get("/dashboard")
def dashboard(req: Request, code: str = Depends(require_auth)):
    return dashboard_payload(req, code)


@router.get("/activity")
def activity(req: Request, tz: int = 0, code: str = Depends(require_auth)):
    return activity_payload(req.app.state.db, code, tz)


@router.get("/history")
def history(req: Request, code: str = Depends(require_auth)):
    # Completed certification attempts for this participant, newest first.
    # Real data only (no sample); scoped strictly by the authed code. Each
    # result is slimmed (per-trial bulk stripped): the history UI reads only
    # verdicts/AUROC per attempt, and expanding an attempt fetches the
    # per-question breakdown lazily from its own endpoint. Without this a
    # returning participant re-downloaded every attempt's full ~370 KB blob
    # on every history view.
    sessions = req.app.state.db.list_results_for_code(code)
    for s in sessions:
        s["result"] = dashboard_logic.slim_result(s["result"])
    return {"sessions": sessions}


@router.get("/history/{session_id}/questions")
def history_questions(session_id: str, req: Request, code: str = Depends(require_auth)):
    # Per-question breakdown for one of THIS participant's tests. Lazy —
    # the history UI fetches it only when a test's dropdown is expanded.
    db = req.app.state.db
    sess = db.get_session(session_id)
    if sess is None or sess["code"] != code:
        raise HTTPException(404, "unknown session")
    # Resolve the per-question correct answers against the bundle that
    # PRODUCED this session (its provenance stamp), not a global default.
    # The loaded bank short-circuits the manifest re-read when it matches.
    truth = dashboard_logic._truth_map_for(sess.get("bundle_version"),
                                           req.app.state.get_bank())
    questions = dashboard_logic.question_breakdown(db.session_trials(session_id), truth)
    return {"sessionId": session_id, "nQuestions": len(questions),
            "questions": questions}


@router.get("/regimen")
def regimen(req: Request, code: str = Depends(require_auth)):
    return regimen_payload(req.app.state.db, code)


@router.get("/trajectories")
def trajectories(req: Request, code: str = Depends(require_auth)):
    return trajectories_payload(req.app.state.db, code)


@router.post("/trajectories")
def trajectories_append(body: TrajectoryIn, req: Request, code: str = Depends(require_auth)):
    _validate_points(body.points)
    req.app.state.db.append_trajectory_points(code, body.points)
    return {"ok": True}


@router.get("/training-sessions")
def training_list(req: Request, code: str = Depends(require_auth)):
    return {"sessions": req.app.state.db.list_training_sessions(code)}


@router.post("/training-sessions")
def training_start(body: TrainingStartIn, req: Request, code: str = Depends(require_auth)):
    if not _training_enabled(req, code):
        raise HTTPException(403, "training is not enabled for this account")
    db = req.app.state.db
    training_id = uuid.uuid4().hex
    reg = db.get_active_regimen(code)   # link the sitting to its regimen (Phase O2)
    db.create_training_session(
        training_id, code, body.taskFocus,
        regimen_id=(reg["regimen_id"] if reg else None),
        source_session_id=(reg.get("source_session_id") if reg else None))
    # Phase L3: tell the client which trainer drives this sitting — the
    # server-side learning engine (POST /api/training-engine/*) or the
    # incumbent client-side trainer. Same reversible-flag pattern as
    # training_mode; default off.
    cfg = req.app.state.cfg
    participant = (db.get_participant(code)
                   if cfg.get("trainer_engine") == "cohort" else None)
    return {"trainingId": training_id,
            "engineMode": engine_trainer.enabled(cfg, code, participant)}


@router.post("/training-bank")
def training_bank(body: SessionIn, req: Request, code: str = Depends(require_auth)):
    """Per-session candidate pool for the immersive TRAINER (not the exam).

    Deliberately draws through a DIFFERENT gate than POST /api/session, because
    the exam draw applies two rules that are wrong for training:

      * No post-training exam washout. That washout (routers/testing.py) is
        one-directional by design — it blocks starting an *exam* soon after
        training so a certificate can't ride a same-day practice boost. Blocking
        *training* after training was an unintended side effect of the two
        surfaces sharing the exam draw; it stranded "Resume training" on a
        `training_washout` error for 12h after every sitting.

      * No test/train exposure exclusion. The trainer is built on spaced
        *repetition* (its own within-session `served` set + the retention
        scheduler handle spacing), so the exam's 30-day test+training exclusion
        would starve exactly the weak-domain pools the learner is training —
        the empty-pool "Resume training" blank screen. Item-level exam leakage
        stays handled the other direction: the exam draw already excludes
        training-seen segments.

    It also does NOT create or supersede a `sessions` row — a training draw must
    never disturb the participant's resumable exam sitting."""
    if not _training_enabled(req, code):
        raise HTTPException(403, "training is not enabled for this account")
    cfg = req.app.state.cfg
    bank = req.app.state.get_bank()
    if bank is None:
        raise HTTPException(503, "question bank unavailable")
    seed = (body.sampleSeed if body.sampleSeed is not None
            else random.randint(0, 2**31 - 1))
    drawn = bank.draw(seed, cfg["session_sample"])   # exclude=None → full pool
    return {"sampleSeed": seed, "bank": drawn}


@router.post("/training-sessions/finalize")
def training_finalize(body: TrainingFinalizeIn, req: Request, code: str = Depends(require_auth)):
    ok = req.app.state.db.finalize_training_session(
        body.trainingId, code, body.nItems, body.summary)
    if not ok:
        raise HTTPException(404, "unknown training session")
    # Recognition AFTER the sitting lands: session-count / training-day
    # milestones (awards.py). Internally best-effort; never fails the finalize.
    awards.evaluate_training(req.app.state.db, code)
    return {"ok": True}


@router.post("/regimen")
def regimen_create(req: Request, code: str = Depends(require_auth)):
    """Build (and activate) a training regimen from this participant's latest
    certification result: one track per NON-PASSED task, carrying its mastery
    bar ℓ*. The client trainer trains these tasks; the fresh retest re-certifies."""
    if not _training_enabled(req, code):
        raise HTTPException(403, "training is not enabled for this account")
    db = req.app.state.db
    latest = db.latest_result_for_code(code)
    if latest is None:
        raise HTTPException(400, "no certification result to build a regimen from")
    src = latest.get("session_id")
    tasks = dashboard_logic.dashboard_tasks(latest["result"])
    # One deck row per NON-PASSED task. Shape MUST match the client RegimenPlan /
    # RegimenDeckEntry the dashboard renders (deck[].ell/ellStar are .toFixed'd,
    # so coerce nullable ℓ/ℓ* to floats). New/learning/due start at 0 for a fresh
    # regimen; they light up as the trainer's exposure/SRS state accrues.
    deck = [
        {
            "taskK": t["taskK"],
            "code": t["code"],
            "label": t.get("label") or t["code"],
            "ell": float(t["ell"]) if t.get("ell") is not None else 0.0,
            "ellStar": float(t["ellStar"]) if t.get("ellStar") is not None else 0.0,
            "new": 0, "learning": 0, "due": 0,
        }
        for t in tasks
        if not dashboard_logic.is_certified_verdict(t.get("verdict"))
    ]
    # Real-skill handoff: the learner's MEASURED per-task posterior (ALL 7 tasks,
    # engine coords ℓ/θ + the ℓ posterior SD from the cert `perTask` block, which
    # dashboard_tasks doesn't carry). The client seeds the trainer's belief clouds
    # from this instead of a generic prior, so training starts where the learner
    # actually is.
    per = latest["result"].get("perTask") or []
    sd_by_k = {
        int(p["taskK"]): p.get("sd")
        for p in per
        if isinstance(p, dict) and p.get("taskK") is not None
    }
    prior = [
        {
            "taskK": t["taskK"],
            "ell": float(t["ell"]) if t.get("ell") is not None else 0.0,
            "theta": float(t["theta"]) if t.get("theta") is not None else 0.0,
            "sd": (float(sd_by_k[t["taskK"]])
                   if sd_by_k.get(t["taskK"]) is not None else None),
        }
        for t in tasks
    ]
    # Raw test stream (learning handoff contract v1.1 §2a): the source cert
    # sitting's EXACT question sequence — the contiguous fully-picked prefix
    # in served order, full n-way pick included (never binarized: measured to
    # corrupt criterion estimates). A replay-capable trainer seeds its belief
    # by re-observing this stream through its own observation model, which is
    # preferred over the `prior` summary; a client must seed from ONE of the
    # two, never both (double counting). taskK rides along so the asked task
    # is known without a bank join.
    stream: list[dict] = []
    if src:
        for t in db.session_trials(src):
            if t.get("pick") is None or t["trial_index"] != len(stream):
                break
            stream.append({"trialIndex": t["trial_index"],
                           "segId": t["seg_id"], "taskK": t["task_k"],
                           "pick": t["pick"]})
    plan = {"weeks": 8, "weekOf": 1, "deck": deck, "prior": prior,
            "sourceSessionId": src, "testStream": stream}
    regimen_id = uuid.uuid4().hex
    db.create_regimen(regimen_id, code, src, plan)
    return {"regimenId": regimen_id, "regimen": plan}


@router.post("/training-progress")
def training_progress(body: TrainingProgressIn, req: Request,
                      code: str = Depends(require_auth)):
    """Persist real per-trial trainer output for an OWNED training session:
    exposure rows (`training_trials`, so retests exclude trained segments) +
    real trajectory points (`is_real=1`, set server-side). Anti-tamper: the
    training session must belong to the authenticated participant."""
    _validate_points(body.points)
    db = req.app.state.db
    if db.training_session_owner(body.trainingId) != code:
        raise HTTPException(404, "unknown training session")
    db.record_training_progress(code, body.trainingId, body.points)
    return {"ok": True, "n": len(body.points)}
