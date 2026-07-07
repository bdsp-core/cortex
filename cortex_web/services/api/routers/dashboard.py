"""Dashboard, history, and learning-protocol surfaces.

Real certification data only: per-task ℓ/θ/ℓ*/AUROC + verdict from the
latest result, plus cert-summary KPIs. Learning-protocol surfaces (regimen,
trajectories, training sessions) carry NO data until the L1 trainer ships;
`sample` stays in the contracts (always False) so the client keeps a single
response shape.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import dashboard_logic
from ..deps import require_auth
from ..models import (
    TrainingFinalizeIn, TrainingProgressIn, TrainingStartIn, TrajectoryIn)

router = APIRouter(prefix="/api")


@router.get("/dashboard")
def dashboard(req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    sessions = db.list_results_for_code(code)
    if not sessions:
        return {"result": None, "hasResult": False, "tasks": [],
                "kpis": None, "sample": False}
    latest = sessions[0]
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
    }


@router.get("/activity")
def activity(req: Request, tz: int = 0, code: str = Depends(require_auth)):
    # Per-day activity levels for the consistency heatmap (sign-in / cert /
    # training), keyed by the user's LOCAL date. `tz` = getTimezoneOffset().
    return {"days": req.app.state.db.activity_levels(code, tz)}


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
    reg = req.app.state.db.get_active_regimen(code)
    return {"regimen": reg["plan"] if reg is not None else None, "sample": False}


@router.get("/trajectories")
def trajectories(req: Request, code: str = Depends(require_auth)):
    rows = req.app.state.db.get_trajectories(code)
    pts = [{"taskK": r["task_k"], "phase": r["phase"], "ell": r["ell"],
            "theta": r["theta"], "sd": r["sd"], "rt": r["rt"], "ts": r["ts"]}
           for r in rows]
    return {"trajectories": pts, "sample": False}


@router.post("/trajectories")
def trajectories_append(body: TrajectoryIn, req: Request, code: str = Depends(require_auth)):
    req.app.state.db.append_trajectory_points(code, body.points)
    return {"ok": True}


@router.get("/training-sessions")
def training_list(req: Request, code: str = Depends(require_auth)):
    return {"sessions": req.app.state.db.list_training_sessions(code)}


@router.post("/training-sessions")
def training_start(body: TrainingStartIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    training_id = uuid.uuid4().hex
    reg = db.get_active_regimen(code)   # link the sitting to its regimen (Phase O2)
    db.create_training_session(
        training_id, code, body.taskFocus,
        regimen_id=(reg["regimen_id"] if reg else None),
        source_session_id=(reg.get("source_session_id") if reg else None))
    return {"trainingId": training_id}


@router.post("/training-sessions/finalize")
def training_finalize(body: TrainingFinalizeIn, req: Request, code: str = Depends(require_auth)):
    ok = req.app.state.db.finalize_training_session(
        body.trainingId, code, body.nItems, body.summary)
    if not ok:
        raise HTTPException(404, "unknown training session")
    return {"ok": True}


@router.post("/regimen")
def regimen_create(req: Request, code: str = Depends(require_auth)):
    """Build (and activate) a training regimen from this participant's latest
    certification result: one track per NON-PASSED task, carrying its mastery
    bar ℓ*. The client trainer trains these tasks; the fresh retest re-certifies."""
    db = req.app.state.db
    sessions = db.list_results_for_code(code)
    if not sessions:
        raise HTTPException(400, "no certification result to build a regimen from")
    latest = sessions[0]
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
        if str(t.get("verdict") or "").upper() != "PASS"
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
    plan = {"weeks": 8, "weekOf": 1, "deck": deck, "prior": prior, "sourceSessionId": src}
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
    db = req.app.state.db
    if db.training_session_owner(body.trainingId) != code:
        raise HTTPException(404, "unknown training session")
    db.record_training_progress(code, body.trainingId, body.points)
    return {"ok": True, "n": len(body.points)}
