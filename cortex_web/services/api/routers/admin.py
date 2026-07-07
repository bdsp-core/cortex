"""X-Admin-Token-gated endpoints (participant seeding + exports)."""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import security
from ..deps import require_admin
from ..models import AdminGenIn

router = APIRouter(prefix="/api/admin")


@router.get("/participants")
def admin_list(req: Request, _: bool = Depends(require_admin)):
    return [dict(r) for r in req.app.state.db.list_participants()]


@router.get("/training-monitor")
def admin_training_monitor(req: Request, _: bool = Depends(require_admin)):
    """Pilot safety + volume telemetry for the deployed trainer (SAP §Monitoring):
    learners/sessions/trials/exposure, per-domain trial counts, graduation count,
    and the confirmed false-graduation rate."""
    cfg = req.app.state.cfg
    return {
        "trainingMode": cfg.get("training_mode", "all"),
        "cohortSize": len(cfg.get("training_allowlist") or ()),
        **req.app.state.db.training_monitor(),
    }


@router.post("/participants")
def admin_gen(body: AdminGenIn, req: Request, _: bool = Depends(require_admin)):
    """Seed test accounts. Since accounts are email+password since the
    public-signup change, this auto-generates synthetic emails so the
    seeded accounts can still authenticate via /api/auth — distinguishable
    from real signups by the @cortex.seed suffix."""
    db = req.app.state.db
    created = []
    for _i in range(body.count):
        code = f"{body.prefix}-{secrets.token_hex(4)}"
        password = secrets.token_urlsafe(9)
        email = f"{code}@cortex.seed"
        db.register_participant(
            code=code, password_hash=security.hash_password(password),
            email=email, display_name=code, signup_ip="admin",
        )
        db.mark_email_verified(code)   # seeded accounts skip email verification
        created.append({"code": code, "email": email, "password": password})
    return created


@router.get("/sessions")
def admin_sessions(req: Request, _: bool = Depends(require_admin)):
    return [dict(r) for r in req.app.state.db.all_sessions()]


@router.get("/results/{session_id}")
def admin_result(session_id: str, req: Request, _: bool = Depends(require_admin)):
    res = req.app.state.db.get_result(session_id)
    if res is None:
        raise HTTPException(404, "no result for session")
    return res
