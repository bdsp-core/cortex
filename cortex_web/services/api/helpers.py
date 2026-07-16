"""Endpoint-shared helpers: validation, auth-code lifecycle, Google token
verification, and the support-report email template.

Routers call these as module attributes (`helpers._check_code(...)`) so tests
can monkeypatch them here and every call site sees the patch.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from . import mailer
from . import security
from .db import Database, utc_now

# ───────────────────────── validation ──────────────────────────
# RFC 5322 is overkill; this matches what every real email service accepts
# and rejects obvious garbage. Deliverability of the DOMAIN is checked
# separately at signup (email_domain_deliverable below, 2026-07-09); the
# mailbox itself is still trusted at face value until the verify code.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
MIN_PASSWORD_LEN = 8
MAX_FIELD_LEN = 200       # tame oversized payloads (display_name etc.)

# Whitelisted demographic/clinical profile keys (collected at signup, editable
# in Settings). Anything else in a submitted profile dict is dropped.
PROFILE_FIELDS = (
    "expertise", "institution", "practice_setting", "years_reading_eeg",
    "eeg_volume_per_month", "self_rated_confidence", "color_vision",
    "prior_test_taken", "sex", "age", "location", "country",
)


def norm_email(s: str) -> str:
    return s.strip().lower()


def is_email(s: str) -> bool:
    return bool(_EMAIL_RE.match(s)) and len(s) <= MAX_FIELD_LEN


def email_domain_deliverable(domain: str) -> bool:
    """Best-effort DNS check that an email domain can actually receive mail:
    MX records, else A/AAAA (the RFC 5321 fallback). Catches typo'd domains
    that can never deliver (e.g. stanfodhealthcare.org, 2026-07-09 incident)
    BEFORE the account is created and stranded on the verify screen.

    Fails OPEN: resolver trouble, timeouts, or a missing dnspython must never
    block signups — only a definitive no-such-domain / no-mail-host answer
    returns False. dnspython is imported lazily (same contract as google-auth
    in app.py: dev installs without it skip the check)."""
    try:
        import dns.resolver
    except Exception:
        return True
    res = dns.resolver.Resolver()
    res.timeout = res.lifetime = 3.0
    try:
        answers = res.resolve(domain, "MX")
        # RFC 7505 null MX ("0 .") = the domain explicitly refuses mail.
        return not all(str(r.exchange) == "." for r in answers)
    except dns.resolver.NXDOMAIN:
        return False
    except dns.resolver.NoAnswer:
        pass  # domain exists but has no MX — fall through to A/AAAA
    except Exception:
        return True
    for rtype in ("A", "AAAA"):
        try:
            res.resolve(domain, rtype)
            return True
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            continue
        except Exception:
            return True
    return False


def clean_profile(d: dict[str, Any]) -> dict[str, str]:
    """Keep only whitelisted profile keys, string-coerced and length-capped."""
    out: dict[str, str] = {}
    for k in PROFILE_FIELDS:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()[:MAX_FIELD_LEN]
        if s:
            out[k] = s
    return out


def is_unique_violation(e: Exception) -> bool:
    """True when an insert/update failed on a UNIQUE constraint — the
    concurrent-signup / email-collision race. Matches both backends' message
    text (sqlite3.IntegrityError / psycopg UniqueViolation) so endpoints can
    translate it to a clean 409 instead of a 500."""
    s = str(e)
    return "UNIQUE" in s or "unique" in s or "duplicate" in s


# ─────────────── auth-code lifecycle (verify/reset) ─────────────

def future_utc(seconds: int) -> str:
    """An ISO-Z timestamp `seconds` in the future. Same fixed format as
    db.utc_now() so lexicographic compare == chronological compare."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + seconds))


def expose_codes() -> bool:
    """Dev/CI only: echo issued codes in API responses so headless smoke tests
    can complete the verify/reset flow. Never set in production (SES sends the
    real email). Off unless CORTEX_EMAIL_EXPOSE_CODE=1."""
    return os.environ.get("CORTEX_EMAIL_EXPOSE_CODE") == "1"


def issue_code(db: Database, participant_code: str, email: str, purpose: str) -> str:
    """Generate + store + email a fresh 6-digit code for (account, purpose)."""
    code = security.gen_numeric_code()
    db.put_auth_code(participant_code, purpose, security.hash_code(code),
                     future_utc(security.CODE_TTL_SECONDS))
    mailer.send_auth_code(email, code, purpose)
    return code


def check_code(db: Database, participant_code: str, purpose: str, presented: str) -> bool:
    """Validate a presented code: not consumed, not expired, under the attempt
    cap, and matching. Bumps the attempt counter on a mismatch. The caller is
    responsible for consuming the code on success."""
    row = db.get_auth_code(participant_code, purpose)
    if row is None or row.get("consumed_utc"):
        return False
    if utc_now() >= row["expires_utc"]:
        return False
    if int(row.get("attempts") or 0) >= security.CODE_MAX_ATTEMPTS:
        return False
    if not security.verify_code(presented.strip(), row["code_hash"]):
        db.increment_auth_attempts(participant_code, purpose)
        return False
    return True


# ───────────────────── Google Sign-In ──────────────────────────

def verify_google_credential(credential: str, client_id: str) -> dict:
    """Verify a Google ID token against `client_id` (the OAuth audience) and
    return its claims (sub, email, email_verified, name, …), or raise. google-
    auth is imported lazily so dev/test installs without it still load; tests
    monkeypatch this function. Verification checks issuer, audience, signature,
    and expiry."""
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    return google_id_token.verify_oauth2_token(
        credential, google_requests.Request(), client_id)


# ───────────────────── support-report email ────────────────────

def build_report_email(username: str, email: str, message: str,
                       client: dict[str, Any], ip: str) -> tuple[str, str]:
    """Render the support-report email (plain text) sent to REPORT_TO. The
    receiver template: reporter identity, the message, submission time, and the
    browser-collected diagnostics for troubleshooting."""
    def g(k: str) -> str:
        v = client.get(k)
        s = str(v).strip() if v is not None else ""
        return s[:500] if s else "(not provided)"
    who = username or email or "anonymous"
    subject = f"[CORTEX Report] {who}"
    lines = [
        "A new report was submitted through CORTEX (app.cortexeeg.org).",
        "",
        "── Reporter ──",
        f"Username: {username or '(not provided)'}",
        f"Email:    {email or '(not provided)'}",
        "",
        "── Message ──",
        message,
        "",
        "── Submitted ──",
        f"Server time (UTC):   {utc_now()}",
        f"Reporter local time: {g('clientTime')}",
        f"App language:        {g('appLang')}",
        "",
        "── Diagnostics (for troubleshooting) ──",
        f"Operating system: {g('os')}",
        f"Browser:          {g('browser')}",
        f"Device platform:  {g('platform')}",
        f"Screen:           {g('screen')}",
        f"Viewport:         {g('viewport')}",
        f"Timezone:         {g('timezone')}",
        f"Locale:           {g('language')}",
        f"Page URL:         {g('url')}",
        f"Referrer:         {g('referrer')}",
        f"Online:           {g('online')}",
        f"User agent:       {g('userAgent')}",
        f"IP address:       {ip}  (approximate location can be looked up from this)",
        "",
        "Sent automatically by CORTEX. Reply to this email to reach the reporter.",
    ]
    return subject, "\n".join(lines)
