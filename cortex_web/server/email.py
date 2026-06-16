"""Transactional email for auth codes (verification + password reset).

Production backend is AWS SES (the deploy box already lives in Stanford AWS).
SES is used when boto3 is importable AND a sender is configured; otherwise the
send is a dev stub that logs the code to stdout so local + CI smoke tests need
no real email account. Selection is automatic but can be forced:

    CORTEX_EMAIL_BACKEND = ses | dev      (default: ses if available else dev)
    CORTEX_EMAIL_FROM    = "CORTEX <no-reply@your-domain>"   (required for ses)
    AWS_REGION / AWS_DEFAULT_REGION       (SES region; boto3 resolves creds)

Keep this module dependency-light: boto3 is imported lazily so SQLite-only dev
installs don't need it.
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
    if forced in ("ses", "dev"):
        return forced
    # auto: SES only if boto3 + a sender are present, else dev stub.
    if os.environ.get("CORTEX_EMAIL_FROM"):
        try:
            import boto3  # noqa: F401
            return "ses"
        except Exception:
            return "dev"
    return "dev"


def _send_ses(to_email: str, subject: str, body: str) -> None:
    import boto3

    sender = os.environ["CORTEX_EMAIL_FROM"]
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("ses", region_name=region) if region else boto3.client("ses")
    client.send_email(
        Source=sender,
        Destination={"ToAddresses": [to_email]},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
        },
    )


def send_auth_code(to_email: str, code: str, purpose: str) -> None:
    """Send a verification/reset code. Never raises into the request path on a
    dev stub; SES errors propagate so the endpoint can surface a 5xx."""
    subject = _SUBJECTS.get(purpose, "Your CORTEX code")
    body = _body(code, purpose)
    if _backend() == "ses":
        _send_ses(to_email, subject, body)
        return
    # dev stub: log so local/CI flows can read the code from server output.
    print(f"[cortex.email:dev] to={to_email} purpose={purpose} code={code}", file=sys.stderr, flush=True)
