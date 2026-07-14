"""Public auth flows: signup, email verification, login (password + Google),
password reset."""
from __future__ import annotations

import json
import os
import secrets
import sys
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from .. import awards, helpers, security
from ..config import TOKEN_TTL
from ..deps import client_ip
from ..models import AuthGoogleIn, AuthIn, EmailIn, RegisterIn, ResetIn, VerifyIn

router = APIRouter(prefix="/api")


@router.post("/register")
def register(body: RegisterIn, req: Request):
    db, limiter = req.app.state.db, req.app.state.limiter
    ip = client_ip(req)
    # Honeypot: silently 200 a bot that filled the trap field. The
    # response is intentionally indistinguishable from success so it
    # doesn't tip off scanners; nothing is actually written.
    if body.honeypot:
        return {"ok": True}
    if not limiter.hit("register", ip):
        raise HTTPException(429, "too many signups from this IP; try again later")
    email = helpers.norm_email(body.email)
    if not helpers.is_email(email):
        raise HTTPException(400, "invalid email")
    if len(body.password) < helpers.MIN_PASSWORD_LEN:
        raise HTTPException(400, f"password must be at least {helpers.MIN_PASSWORD_LEN} characters")
    display = body.displayName.strip()[:helpers.MAX_FIELD_LEN]
    if not display:
        raise HTTPException(400, "displayName required")
    existing = db.get_participant_by_email(email)
    if existing is not None and (existing.get("email_verified_utc")
                                 or existing.get("google_sub")):
        raise HTTPException(409, "an account with this email already exists")
    # Reject domains that can never receive the verification code (no DNS
    # MX/A records — typically a typo'd domain). Fails open on DNS trouble.
    if not helpers.email_domain_deliverable(email.rsplit("@", 1)[1]):
        raise HTTPException(
            400, "this email domain doesn't appear to accept mail; double-check it for typos")
    prof = helpers.clean_profile(body.profile)
    expertise = (body.expertise.strip() or prof.get("expertise", "")).strip()[:helpers.MAX_FIELD_LEN] or None
    if existing is not None:
        # Unverified pending signup: mailbox ownership was never proven, so
        # the new claimant may retake it — refresh credentials/profile in
        # place (stable code + public_id) and reissue the code below. Turns
        # the old 409 dead-end (typo'd password, abandoned flow) into a
        # clean retry; the true mailbox owner still wins, since verifying
        # requires reading the freshly-mailed code.
        code = existing["code"]
        db.update_pending_registration(
            code=code,
            password_hash=security.hash_password(body.password),
            display_name=display,
            signup_ip=ip,
            signup_expertise=expertise,
            profile=(json.dumps(prof) if prof else None),
        )
    else:
        code = "u-" + secrets.token_urlsafe(12)
        try:
            db.register_participant(
                code=code,
                password_hash=security.hash_password(body.password),
                email=email,
                display_name=display,
                signup_ip=ip,
                signup_expertise=expertise,
                profile=(json.dumps(prof) if prof else None),
            )
        except Exception as e:
            # UNIQUE-index violation race (two concurrent signups, same email).
            # Translate to 409 instead of a 500.
            if helpers.is_unique_violation(e):
                raise HTTPException(409, "an account with this email already exists")
            raise
    # "Account created" milestone (idempotent; also covers a pending-signup
    # retry whose account predates this feature). Best-effort.
    awards.evaluate_account_created(db, code)
    # No session is issued at signup: the account must verify its email
    # before it can sign in. Email a 6-digit code; the SPA advances to the
    # verify screen.
    try:
        dev_code = helpers.issue_code(db, code, email, "verify")
    except Exception as e:
        # The account row already exists, so a 500 here would strand it:
        # the user's retry hits 409 "already exists" with no way forward.
        # Land them on the verify screen instead — "resend code" covers
        # the transient email outage.
        print(f"[cortex.register] verification email failed for {email}: {e}",
              file=sys.stderr, flush=True)
        dev_code = None
    resp = {"needsVerification": True, "email": email, "displayName": display}
    if helpers.expose_codes() and dev_code is not None:
        resp["devCode"] = dev_code
    return resp


@router.post("/verify/confirm")
def verify_confirm(body: VerifyIn, req: Request):
    db, limiter = req.app.state.db, req.app.state.limiter
    if not limiter.hit("verify", client_ip(req)):
        raise HTTPException(429, "too many attempts; try again later")
    email = helpers.norm_email(body.email)
    row = db.get_participant_by_email(email)
    if row is None:
        raise HTTPException(400, "invalid or expired code")
    if row.get("email_verified_utc"):
        return {"ok": True, "alreadyVerified": True}
    if not helpers.check_code(db, row["code"], "verify", body.code):
        raise HTTPException(400, "invalid or expired code")
    db.consume_auth_code(row["code"], "verify")
    db.mark_email_verified(row["code"])
    db.clear_email_undeliverable(row["code"])  # code arrived → mailbox works
    # Cohort invites emailed to this address before it had an account become
    # normal pending invites now (the user still accepts in-app).
    db.attach_email_invites(email, row["code"])
    return {"ok": True}


@router.post("/verify/status")
def verify_status(body: EmailIn, req: Request):
    """Polled by the verify screen while the user waits for their code: did
    the email we just sent hard-bounce (routers/ses_events.py)? Anti-oracle:
    always 200, and `undeliverable` is only ever true for an account that is
    still pending verification — unknown emails and verified accounts are
    indistinguishable."""
    db, limiter = req.app.state.db, req.app.state.limiter
    if not limiter.hit("verify_status", client_ip(req)):
        raise HTTPException(429, "too many attempts; try again later")
    row = db.get_participant_by_email(helpers.norm_email(body.email))
    undeliverable = bool(
        row is not None
        and not row.get("email_verified_utc")
        and row.get("email_undeliverable_utc"))
    return {"undeliverable": undeliverable}


@router.post("/verify/resend")
def verify_resend(body: EmailIn, req: Request):
    db, limiter = req.app.state.db, req.app.state.limiter
    if not limiter.hit("resend", client_ip(req)):
        raise HTTPException(429, "too many requests; try again later")
    email = helpers.norm_email(body.email)
    row = db.get_participant_by_email(email)
    resp: dict[str, Any] = {"ok": True}
    # Only (re)issue for an existing, still-unverified account; respond 200
    # either way so the endpoint doesn't reveal which emails exist.
    if row is not None and not row.get("email_verified_utc"):
        try:
            dev_code = helpers.issue_code(db, row["code"], email, "verify")
        except Exception as e:
            # An SMTP outage must not 500: a 500-for-real-accounts vs
            # 200-for-unknown-emails split would leak which emails exist,
            # exactly what the 200-regardless contract above prevents.
            print(f"[cortex.resend] verification email failed for {email}: {e}",
                  file=sys.stderr, flush=True)
            dev_code = None
        if helpers.expose_codes() and dev_code is not None:
            resp["devCode"] = dev_code
    return resp


@router.post("/auth")
def auth(body: AuthIn, req: Request):
    db, limiter = req.app.state.db, req.app.state.limiter
    if not limiter.hit("auth", client_ip(req)):
        raise HTTPException(429, "too many login attempts; try again later")
    email = helpers.norm_email(body.email)
    if not helpers.is_email(email):
        raise HTTPException(401, "invalid credentials")
    row = db.get_participant_by_email(email)
    if row is None or not row["active"]:
        raise HTTPException(401, "invalid credentials")
    if not security.verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "invalid credentials")
    # Credentials are valid but the email isn't verified yet: signal the SPA
    # to route to the verify screen (detail is a stable machine code).
    if not row.get("email_verified_utc"):
        raise HTTPException(403, "email_not_verified")
    code = row["code"]
    db.record_login_day(code, body.tzOffset or 0)
    token = security.issue_token(code, ttl_seconds=TOKEN_TTL,
                                 extra={"email": email})
    return {"token": token, "expiresIn": TOKEN_TTL, "code": code,
            "email": email, "displayName": row.get("display_name") or ""}


@router.post("/auth/google")
def auth_google(body: AuthGoogleIn, req: Request):
    db, limiter = req.app.state.db, req.app.state.limiter
    ip = client_ip(req)
    if not limiter.hit("auth_google", ip):
        raise HTTPException(429, "too many sign-in attempts; try again later")
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    if not client_id:
        raise HTTPException(503, "Google sign-in is not configured")
    try:
        info = helpers.verify_google_credential(body.credential, client_id)
    except Exception:
        raise HTTPException(401, "invalid Google credential")
    if not info.get("email_verified"):
        raise HTTPException(401, "Google account email is not verified")
    sub = str(info.get("sub") or "")
    email = helpers.norm_email(info.get("email") or "")
    name = (info.get("name") or "").strip()[:helpers.MAX_FIELD_LEN]
    if not sub or not helpers.is_email(email):
        raise HTTPException(401, "incomplete Google profile")
    # Find by Google id → else link to an existing same-email account
    # (Google asserts the email) → else create a fresh OAuth account.
    row = db.get_participant_by_google_sub(sub)
    if row is None:
        existing = db.get_participant_by_email(email)
        if existing is not None:
            db.link_google_sub(existing["code"], sub)
            if not existing.get("email_verified_utc"):
                db.mark_email_verified(existing["code"])
            row = db.get_participant_by_email(email)
        else:
            code = "u-" + secrets.token_urlsafe(12)
            try:
                db.register_oauth_participant(
                    code=code, email=email, display_name=name or email,
                    google_sub=sub, signup_ip=ip)
            except Exception as e:
                # UNIQUE race (concurrent first sign-in): fall back to lookup.
                if helpers.is_unique_violation(e):
                    row = db.get_participant_by_google_sub(sub) or db.get_participant_by_email(email)
                else:
                    raise
            row = row or db.get_participant_by_email(email)
    if row is None or not row.get("active"):
        raise HTTPException(403, "account is disabled")
    # "Account created" milestone (idempotent; fires once on first sign-in).
    awards.evaluate_account_created(db, row["code"])
    # Google accounts are born verified: attach any cohort invites emailed to
    # this address before it had an account (no-op on later sign-ins).
    db.attach_email_invites(email, row["code"])
    db.record_login_day(row["code"], body.tzOffset or 0)
    token = security.issue_token(row["code"], ttl_seconds=TOKEN_TTL,
                                 extra={"email": email})
    return {"token": token, "expiresIn": TOKEN_TTL, "code": row["code"],
            "email": email, "displayName": row.get("display_name") or ""}


@router.post("/forgot")
def forgot(body: EmailIn, req: Request):
    db, limiter = req.app.state.db, req.app.state.limiter
    if not limiter.hit("forgot", client_ip(req)):
        raise HTTPException(429, "too many requests; try again later")
    email = helpers.norm_email(body.email)
    row = db.get_participant_by_email(email)
    resp: dict[str, Any] = {"ok": True}
    # Respond 200 regardless so the endpoint doesn't reveal which emails
    # have accounts; only actually issue a code for a real account.
    if row is not None and row["active"]:
        try:
            dev_code = helpers.issue_code(db, row["code"], email, "reset")
        except Exception as e:
            # Same anti-oracle contract as /verify/resend: an email-send
            # failure logs and still 200s instead of leaking account
            # existence through a 500.
            print(f"[cortex.forgot] reset email failed for {email}: {e}",
                  file=sys.stderr, flush=True)
            dev_code = None
        if helpers.expose_codes() and dev_code is not None:
            resp["devCode"] = dev_code
    return resp


@router.post("/reset")
def reset(body: ResetIn, req: Request):
    db, limiter = req.app.state.db, req.app.state.limiter
    if not limiter.hit("reset", client_ip(req)):
        raise HTTPException(429, "too many attempts; try again later")
    email = helpers.norm_email(body.email)
    if len(body.newPassword) < helpers.MIN_PASSWORD_LEN:
        raise HTTPException(400, f"password must be at least {helpers.MIN_PASSWORD_LEN} characters")
    row = db.get_participant_by_email(email)
    if row is None or not helpers.check_code(db, row["code"], "reset", body.code):
        raise HTTPException(400, "invalid or expired code")
    db.consume_auth_code(row["code"], "reset")
    db.set_password_hash(row["code"], security.hash_password(body.newPassword))
    # A successful reset also confirms control of the email address.
    if not row.get("email_verified_utc"):
        db.mark_email_verified(row["code"])
    db.clear_email_undeliverable(row["code"])  # code arrived → mailbox works
    return {"ok": True}
