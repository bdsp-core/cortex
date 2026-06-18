"""Transactional email for auth codes (verification + password reset).

Three backends, selected by CORTEX_EMAIL_BACKEND (ses | smtp | dev) or
auto-detected: SMTP if CORTEX_SMTP_HOST is set, else SES if CORTEX_EMAIL_FROM +
boto3 are present, else a dev stub that logs the code to stderr (no real account
needed for local/CI).

  CORTEX_EMAIL_BACKEND   ses | smtp | dev    (optional; otherwise auto)
  CORTEX_EMAIL_FROM      "CORTEX <no-reply@your-domain>"   (sender; ses/smtp)

  SMTP (works with Gmail/Workspace app passwords, a Stanford relay, Postmark/
  SendGrid/Resend, etc.):
    CORTEX_SMTP_HOST       smtp host (presence selects the smtp backend)
    CORTEX_SMTP_PORT       default 587
    CORTEX_SMTP_USER       login user   (optional for open relays)
    CORTEX_SMTP_PASSWORD   login password / app password
    CORTEX_SMTP_SECURITY   starttls (default) | ssl | none

  SES:
    AWS_REGION / AWS_DEFAULT_REGION       (SES region; boto3 resolves creds)

boto3/smtplib are imported lazily so SQLite-only dev installs need neither.
"""
from __future__ import annotations

import os
import sys

_SUBJECTS = {
    "verify": "Your CORTEX verification code",
    "reset": "Your CORTEX password reset code",
}
_INTROS = {
    "verify": "Use this code to verify your email and finish creating your CORTEX account:",
    "reset": "Use this code to reset your CORTEX password:",
}


def _body(code: str, purpose: str) -> str:
    intro = _INTROS.get(purpose, "Your CORTEX code:")
    return (
        f"{intro}\n\n"
        f"    {code}\n\n"
        f"This code expires in 15 minutes. If you did not request it, ignore this email.\n"
    )


def _backend() -> str:
    forced = os.environ.get("CORTEX_EMAIL_BACKEND", "").strip().lower()
    if forced in ("ses", "smtp", "dev"):
        return forced
    # auto: SMTP if a host is configured; else SES if boto3 + sender; else dev.
    if os.environ.get("CORTEX_SMTP_HOST"):
        return "smtp"
    if os.environ.get("CORTEX_EMAIL_FROM"):
        try:
            import boto3  # noqa: F401
            return "ses"
        except Exception:
            return "dev"
    return "dev"


def _send_smtp(to_email: str, subject: str, body: str, reply_to: str | None = None) -> None:
    import smtplib
    import ssl
    from email.message import EmailMessage

    host = os.environ["CORTEX_SMTP_HOST"]
    port = int(os.environ.get("CORTEX_SMTP_PORT", "587"))
    user = os.environ.get("CORTEX_SMTP_USER")
    password = os.environ.get("CORTEX_SMTP_PASSWORD")
    sender = os.environ.get("CORTEX_EMAIL_FROM") or user
    security = os.environ.get("CORTEX_SMTP_SECURITY", "starttls").strip().lower()

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_email
    if reply_to:
        msg["Reply-To"] = reply_to
    msg["Subject"] = subject
    msg.set_content(body)

    if security == "ssl":
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as s:
            if user:
                s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.ehlo()
            if security == "starttls":
                s.starttls(context=ssl.create_default_context())
                s.ehlo()
            if user:
                s.login(user, password)
            s.send_message(msg)


def _send_ses(to_email: str, subject: str, body: str, reply_to: str | None = None) -> None:
    import boto3

    sender = os.environ["CORTEX_EMAIL_FROM"]
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("ses", region_name=region) if region else boto3.client("ses")
    client.send_email(
        Source=sender,
        Destination={"ToAddresses": [to_email]},
        ReplyToAddresses=[reply_to] if reply_to else [],
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
        },
    )


def send_auth_code(to_email: str, code: str, purpose: str) -> None:
    """Send a verification/reset code. SMTP/SES errors propagate so the endpoint
    can surface a 5xx; the dev stub never raises."""
    subject = _SUBJECTS.get(purpose, "Your CORTEX code")
    body = _body(code, purpose)
    backend = _backend()
    if backend == "smtp":
        _send_smtp(to_email, subject, body)
        return
    if backend == "ses":
        _send_ses(to_email, subject, body)
        return
    # dev stub: log so local/CI flows can read the code from server output.
    print(f"[cortex.email:dev] to={to_email} purpose={purpose} code={code}", file=sys.stderr, flush=True)


def send_email(to_email: str, subject: str, body: str, reply_to: str | None = None) -> None:
    """Send an arbitrary transactional email (e.g. a user report). Same backend
    dispatch as send_auth_code; SMTP/SES errors propagate so the caller can
    surface a 5xx. The dev stub never raises (logs to stderr)."""
    backend = _backend()
    if backend == "smtp":
        _send_smtp(to_email, subject, body, reply_to)
        return
    if backend == "ses":
        _send_ses(to_email, subject, body, reply_to)
        return
    print(f"[cortex.email:dev] to={to_email} reply_to={reply_to} subject={subject!r}\n{body}",
          file=sys.stderr, flush=True)
