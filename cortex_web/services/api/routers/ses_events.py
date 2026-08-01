"""SES delivery-event webhook (SNS HTTPS subscription).

The SES identity's default configuration set (cortex-transactional, see
deploy/README.md "Email delivery monitoring") publishes bounce/complaint/
reject events to the SNS topic cortex-ses-events. This endpoint is that
topic's HTTPS subscriber: when a verification/reset email HARD-bounces
(recipient mailbox doesn't exist), the participant row is flagged
email_undeliverable, and the verify screen — which polls
POST /api/verify/status while the user waits — tells them to fix the
address instead of letting them wait forever.

Auth model: three independent gates. (1) a capability token in the
subscription URL (CORTEX_SNS_WEBHOOK_TOKEN, known only to us and AWS);
(2) a TopicArn allowlist (CORTEX_SNS_TOPIC_ARN); (3) the SNS message
signature itself (sns_verify.py — SigV1/SigV2 RSA over the canonical
string, cert pinned to https://sns.<region>.amazonaws.com/). Signature
verification is ON by default and fails closed (403 → SNS retries with
backoff, so a transient cert-fetch failure heals); the emergency escape is
CORTEX_SNS_VERIFY=off, which restores the previous two-gate behavior.

SNS delivery notes: posts arrive with Content-Type text/plain, so the
envelope is parsed from the raw body, not via pydantic. Non-2xx responses
make SNS retry with backoff; malformed/ignored events return 200 so they
are not redelivered.
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import urllib.request

import anyio
from fastapi import APIRouter, HTTPException, Request

from .. import helpers, sns_verify

router = APIRouter(prefix="/api")

# Bounce subtypes that mean "this mailbox will never receive our mail".
# (Suppressed = SES's account-level suppression list short-circuited the
# send — the address already hard-bounced once before.)
_PERMANENT_BOUNCE = "Permanent"


def _webhook_token() -> str:
    """Empty string = feature off (endpoint 404s). Read per-request, like
    helpers.expose_codes(), so tests can toggle via env."""
    return os.environ.get("CORTEX_SNS_WEBHOOK_TOKEN", "").strip()


def _expected_topic_arn() -> str:
    return os.environ.get("CORTEX_SNS_TOPIC_ARN", "").strip()


def _http_get(url: str) -> None:
    """Confirm-subscription fetch. Module-level so tests can monkeypatch."""
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (https enforced by caller)
        resp.read()


def _log(msg: str) -> None:
    print(f"[cortex.ses] {msg}", file=sys.stderr, flush=True)


@router.post("/ses/events")
async def ses_events(req: Request):
    token = _webhook_token()
    if not token:
        raise HTTPException(404, "not found")
    if not hmac.compare_digest(req.query_params.get("token", ""), token):
        raise HTTPException(403, "bad token")

    try:
        envelope = json.loads((await req.body()).decode("utf-8"))
    except Exception:
        raise HTTPException(400, "not JSON")

    if os.environ.get("CORTEX_SNS_VERIFY", "").strip().lower() != "off":
        try:
            # Cert fetch + RSA verify off the event loop (network + CPU).
            await anyio.to_thread.run_sync(sns_verify.verify_envelope, envelope)
        except sns_verify.SnsVerifyError as e:
            _log(f"rejected unverifiable envelope: {e}")
            raise HTTPException(403, "bad signature")

    expected_arn = _expected_topic_arn()
    if expected_arn and envelope.get("TopicArn") != expected_arn:
        _log(f"dropped message for unexpected TopicArn={envelope.get('TopicArn')!r}")
        return {"ok": True}

    kind = envelope.get("Type")
    if kind == "SubscriptionConfirmation":
        url = envelope.get("SubscribeURL", "")
        # Only ever call back into SNS itself.
        if not (url.startswith("https://sns.") and ".amazonaws.com/" in url):
            raise HTTPException(400, "bad SubscribeURL")
        await anyio.to_thread.run_sync(_http_get, url)
        _log(f"confirmed SNS subscription for {envelope.get('TopicArn')}")
        return {"ok": True}

    if kind != "Notification":
        return {"ok": True}   # UnsubscribeConfirmation etc. — nothing to do

    try:
        event = json.loads(envelope.get("Message", ""))
    except Exception:
        return {"ok": True}   # not an SES event payload — ignore, don't retry

    if event.get("eventType") != "Bounce":
        return {"ok": True}   # complaints/delays reach the operator via email
    bounce = event.get("bounce") or {}
    if bounce.get("bounceType") != _PERMANENT_BOUNCE:
        return {"ok": True}   # transient — SES is still retrying

    db = req.app.state.db
    recipients = bounce.get("bouncedRecipients") or []
    sub_type = bounce.get("bounceSubType")

    def _flag_bounced() -> list[tuple[str, str]]:
        # All DB work for this webhook in ONE worker thread. This endpoint is
        # async (it awaits the request body and the SNS subscribe callback),
        # so blocking queries inline would stall the single-worker event loop
        # — and therefore every other participant's request — for as long as
        # a bounce batch takes.
        marked: list[tuple[str, str]] = []
        for rcpt in recipients:
            email = helpers.norm_email(str(rcpt.get("emailAddress", "")))
            if not email:
                continue
            row = db.get_participant_by_email(email)
            if row is None:
                continue
            db.mark_email_undeliverable(row["code"])
            marked.append((email, row["code"]))
        return marked

    flagged = await anyio.to_thread.run_sync(_flag_bounced)
    for email, participant in flagged:
        _log(f"permanent bounce for {email} → flagged {participant} "
             f"(subType={sub_type})")
    return {"ok": True, "flagged": len(flagged)}
