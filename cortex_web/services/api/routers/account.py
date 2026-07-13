"""Account self-service (Settings page) + the consent ledger (Phase O1)."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import helpers, security
from ..deps import client_ip, require_auth
from ..models import (ConsentIn, ConsentWithdrawIn, EmailChangeIn,
                      PasswordChangeIn, ProfileIn)

router = APIRouter(prefix="/api")


@router.get("/profile")
def get_profile(req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    row = db.get_participant(code)
    if row is None:
        raise HTTPException(404, "account not found")
    prof: dict[str, Any] = {}
    if row.get("profile"):
        try:
            prof = json.loads(row["profile"])
        except Exception:
            prof = {}
    # Every account gets one at creation/boot-backfill; ensure_public_id is
    # the lazy repair for any row that slipped past both (e.g. written by an
    # old process mid-deploy).
    public_id = row.get("public_id") or db.ensure_public_id(code) or ""
    return {
        "email": row.get("email") or "",
        "displayName": row.get("display_name") or "",
        "expertise": row.get("signup_expertise") or "",
        "authProvider": row.get("auth_provider") or "local",
        "publicId": str(public_id),
        "profile": prof,
        # Training-reminder digest (digest.py). NULL/0 = on (the default).
        "trainingReminders": not (row.get("digest_opt_out") or 0),
    }


@router.put("/profile")
def put_profile(body: ProfileIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    dn = body.displayName.strip()[:helpers.MAX_FIELD_LEN] if body.displayName is not None else None
    if dn is not None and not dn:
        raise HTTPException(400, "display name cannot be empty")
    prof = helpers.clean_profile(body.profile) if body.profile is not None else None
    # expertise lives both in its own column and in the profile blob; keep them aligned
    exp = body.expertise.strip()[:helpers.MAX_FIELD_LEN] if body.expertise is not None else (
        prof.get("expertise") if prof else None)
    db.update_profile(code, display_name=dn,
                      profile=(json.dumps(prof) if prof is not None else None),
                      signup_expertise=exp)
    if body.trainingReminders is not None:
        db.set_digest_opt_out(code, not body.trainingReminders)
    return {"ok": True}


@router.post("/account/password")
def change_password(body: PasswordChangeIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    row = db.get_participant(code)
    if row is None or not security.verify_password(body.currentPassword, row["password_hash"]):
        raise HTTPException(403, "current password is incorrect")
    if len(body.newPassword) < helpers.MIN_PASSWORD_LEN:
        raise HTTPException(400, f"password must be at least {helpers.MIN_PASSWORD_LEN} characters")
    db.set_password_hash(code, security.hash_password(body.newPassword))
    return {"ok": True}


@router.post("/account/email")
def change_email(body: EmailChangeIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    row = db.get_participant(code)
    if row is None or not security.verify_password(body.password, row["password_hash"]):
        raise HTTPException(403, "password is incorrect")
    new_email = helpers.norm_email(body.newEmail)
    if not helpers.is_email(new_email):
        raise HTTPException(400, "invalid email")
    existing = db.get_participant_by_email(new_email)
    if existing is not None and existing["code"] != code:
        raise HTTPException(409, "an account with this email already exists")
    try:
        db.update_email(code, new_email)
    except Exception as e:
        if helpers.is_unique_violation(e):
            raise HTTPException(409, "an account with this email already exists")
        raise
    return {"ok": True, "email": new_email}


# ── consent ledger (Phase O1) ───────────────────────────────────
@router.post("/consent")
def consent_record(body: ConsentIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    cver = (body.consentVersion or "").strip()[:helpers.MAX_FIELD_LEN]
    if not cver:
        raise HTTPException(400, "consentVersion required")
    ctype = (body.consentType or "").strip()[:helpers.MAX_FIELD_LEN] or "research_irb"
    irb = (body.irbProtocolId or "").strip()[:helpers.MAX_FIELD_LEN] or None
    db.record_consent(code, ctype, cver, irb_protocol_id=irb, consent_ip=client_ip(req))
    return {"ok": True}


@router.post("/consent/withdraw")
def consent_withdraw(body: ConsentWithdrawIn, req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    n = db.withdraw_consent(code, (body.consentType or "").strip() or None)
    return {"ok": True, "withdrawn": n}


@router.get("/consent")
def consent_list(req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    return {"events": db.get_consent_events(code)}
