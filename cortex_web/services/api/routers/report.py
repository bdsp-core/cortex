"""Public support/feedback reports → email to the team (see /report page)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .. import helpers
from .. import mailer
from ..config import REPORT_TO
from ..deps import client_ip
from ..models import ReportIn

router = APIRouter(prefix="/api")


@router.post("/report")
def report(body: ReportIn, req: Request):
    limiter = req.app.state.limiter
    ip = client_ip(req)
    if not limiter.hit("report", ip):
        raise HTTPException(429, "too many reports from this IP — try again later")
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "please describe the issue or suggestion")
    message = message[:5000]
    username = body.username.strip()[:helpers.MAX_FIELD_LEN]
    email = helpers.norm_email(body.email)
    reply_to = email if helpers.is_email(email) else None
    subject, text = helpers.build_report_email(username, email, message, body.client or {}, ip)
    mailer.send_email(REPORT_TO, subject, text, reply_to=reply_to)
    return {"ok": True}
