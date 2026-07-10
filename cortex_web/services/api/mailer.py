"""Transactional email (verification + password-reset codes, cohort
invitations, support reports).

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

Every user-facing message is a LETTER: multipart/alternative with a plain-text
part (deliverability + text-only clients) and an HTML part styled as a
bordered card with the CORTEX logo header (assets/cortex_logo_email.png,
embedded inline via CID so it renders without remote-image loading) and a
"Thank you, The CORTEX Team" sign-off. Content is expressed as BLOCKS —
("p", text) paragraphs, ("code", "123456") code boxes, ("button", label, url)
call-to-action buttons — rendered by _render_text/_render_html so the two
parts can never drift apart. House copy style: no em dashes.

boto3/smtplib are imported lazily so SQLite-only dev installs need neither.
"""
from __future__ import annotations

import html as html_mod
import os
import re
import sys
from email.message import EmailMessage
from pathlib import Path

_SUBJECTS = {
    "verify": "Your CORTEX verification code",
    "reset": "Your CORTEX password reset code",
}
_INTROS = {
    "verify": "Use this code to verify your email and finish creating your CORTEX account:",
    "reset": "Use this code to reset your CORTEX password:",
}
_TITLES = {
    "verify": "Verify your email",
    "reset": "Reset your password",
}
_LINK_LABELS = {
    "verify": "Verify my email",
    "reset": "Open the reset form",
}

# ── brand (mirrors the SPA's light-theme tokens; emails render on white) ──
_TEAL = "#2f8f83"
_TEAL_DEEP = "#235e57"
_INK = "#25332f"
_LOGO_PATH = Path(__file__).with_name("assets") / "cortex_logo_email.png"
_LOGO_CID = "cortex-logo"

_logo_cache: list[bytes | None] = []


def _logo_bytes() -> bytes | None:
    """The vendored 440x81 logo PNG, or None (letter renders without a header
    image; nothing user-facing breaks on an install missing the asset)."""
    if not _logo_cache:
        try:
            _logo_cache.append(_LOGO_PATH.read_bytes())
        except OSError:
            _logo_cache.append(None)
    return _logo_cache[0]


def _one_click_link(code: str, purpose: str, to_email: str) -> str | None:
    """One-click deep link into the SPA (root-path query params; parsed and
    stripped by apps/web/src/deepLink.ts). Only when CORTEX_PUBLIC_ORIGIN is
    set (e.g. https://app.cortexeeg.org); dev/CI emails stay link-free. The
    link carries the same short-lived code as the email body, so it grants
    nothing the email itself doesn't."""
    origin = os.environ.get("CORTEX_PUBLIC_ORIGIN", "").strip().rstrip("/")
    if not origin:
        return None
    from urllib.parse import quote
    q = quote(to_email, safe="")
    if purpose == "verify":
        return f"{origin}/?verifyEmail={q}&verifyCode={code}"
    if purpose == "reset":
        return f"{origin}/?resetEmail={q}&resetCode={code}"
    return None


# ─────────────────────── letter rendering ───────────────────────
# Blocks: ("p", text) | ("code", code) | ("button", label, url)

def _render_text(blocks: list[tuple]) -> str:
    """Plain-text part. Paragraphs separated by blank lines, codes indented,
    buttons become "label: url" lines, plus the team sign-off."""
    out: list[str] = []
    for b in blocks:
        if b[0] == "p":
            out.append(b[1])
        elif b[0] == "code":
            out.append(f"    {b[1]}")
        elif b[0] == "button":
            out.append(f"{b[1]}: {b[2]}")
    out.append("Thank you,\nThe CORTEX Team")
    return "\n\n".join(out) + "\n"


_URL_RE = re.compile(r"(https?://[^\s<]+)")


def _p_html(text: str) -> str:
    """Escape, then turn bare URLs into links (escaping first keeps
    user-controlled text inert; URLs we emit are our own origin)."""
    escaped = html_mod.escape(text)
    linked = _URL_RE.sub(
        rf'<a href="\1" style="color:{_TEAL};">\1</a>', escaped)
    return f'<p style="margin:0 0 14px;">{linked}</p>'


def _render_html(title: str, blocks: list[tuple]) -> str:
    """The letter: a bordered white card on a neutral page, logo header,
    content blocks, team sign-off, small footer. Table layout + inline styles
    only (email-client-safe); the logo is referenced by CID."""
    body_bits: list[str] = []
    for b in blocks:
        if b[0] == "p":
            body_bits.append(_p_html(b[1]))
        elif b[0] == "code":
            body_bits.append(
                f'<div style="margin:18px 0;padding:14px 0;text-align:center;'
                f'background:#f0f6f5;border:1px solid #cfe3e0;border-radius:6px;'
                f"font-family:'Courier New',monospace;font-size:26px;"
                f'letter-spacing:8px;font-weight:bold;color:{_TEAL_DEEP};">'
                f"{html_mod.escape(b[1])}</div>")
        elif b[0] == "button":
            label, url = html_mod.escape(b[1]), html_mod.escape(b[2], quote=True)
            body_bits.append(
                '<table role="presentation" cellpadding="0" cellspacing="0" '
                'style="margin:6px 0 18px;"><tr>'
                f'<td style="background:{_TEAL};border-radius:6px;">'
                f'<a href="{url}" style="display:inline-block;padding:11px 22px;'
                'font-family:Arial,Helvetica,sans-serif;font-size:14px;'
                'color:#ffffff;text-decoration:none;font-weight:bold;">'
                f"{label}</a></td></tr></table>")
    header = (
        f'<img src="cid:{_LOGO_CID}" width="220" alt="CORTEX" '
        'style="display:block;border:0;outline:none;max-width:220px;height:auto;">'
        if _logo_bytes() else
        f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:20px;'
        f'font-weight:bold;color:{_TEAL_DEEP};">CORTEX</div>')
    font = "font-family:Arial,Helvetica,sans-serif;"
    return f"""<!doctype html>
<html><body style="margin:0;padding:0;background:#eef1f1;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef1f1;">
<tr><td align="center" style="padding:28px 12px;">
  <table role="presentation" width="560" cellpadding="0" cellspacing="0"
    style="width:560px;max-width:100%;background:#ffffff;border:1px solid #d7dedd;border-top:3px solid {_TEAL};border-radius:6px;">
    <tr><td align="left" style="padding:26px 36px 20px;border-bottom:1px solid #e6ecea;">{header}</td></tr>
    <tr><td align="left" style="padding:26px 36px 4px;{font}color:{_INK};font-size:15px;line-height:1.6;">
      <h1 style="margin:0 0 16px;font-size:18px;color:{_TEAL_DEEP};">{html_mod.escape(title)}</h1>
      {"".join(body_bits)}
      <p style="margin:18px 0 24px;">Thank you,<br>
        <b style="color:{_TEAL_DEEP};">The CORTEX Team</b></p>
    </td></tr>
    <tr><td align="left" style="padding:14px 36px;background:#f6f8f8;border-top:1px solid #e6ecea;border-radius:0 0 6px 6px;{font}font-size:12px;line-height:1.5;color:#7d8a86;">
      CORTEX, the EEG skill certification platform · app.cortexeeg.org<br>
      This is an automated message; replies to this address are not monitored.
    </td></tr>
  </table>
</td></tr>
</table>
</body></html>
"""


# ───────────────────────── send machinery ─────────────────────────

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


def _compose(to_email: str, subject: str, text: str, html: str | None,
             reply_to: str | None) -> EmailMessage:
    sender = (os.environ.get("CORTEX_EMAIL_FROM")
              or os.environ.get("CORTEX_SMTP_USER") or "no-reply@localhost")
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_email
    if reply_to:
        msg["Reply-To"] = reply_to
    msg["Subject"] = subject
    msg.set_content(text)
    if html is not None:
        msg.add_alternative(html, subtype="html")
        logo = _logo_bytes()
        if logo:
            # Attach the logo INSIDE the html alternative (multipart/related)
            # so text-only clients never see an attachment.
            msg.get_payload()[-1].add_related(
                logo, maintype="image", subtype="png", cid=f"<{_LOGO_CID}>")
    return msg


def _send_smtp(msg: EmailMessage) -> None:
    import smtplib
    import ssl

    host = os.environ["CORTEX_SMTP_HOST"]
    port = int(os.environ.get("CORTEX_SMTP_PORT", "587"))
    user = os.environ.get("CORTEX_SMTP_USER")
    password = os.environ.get("CORTEX_SMTP_PASSWORD")
    security = os.environ.get("CORTEX_SMTP_SECURITY", "starttls").strip().lower()

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


def _send_ses(msg: EmailMessage, to_email: str) -> None:
    # send_raw_email (not send_email): the v1 Simple API cannot carry the
    # inline CID logo attachment.
    import boto3

    sender = os.environ["CORTEX_EMAIL_FROM"]
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("ses", region_name=region) if region else boto3.client("ses")
    client.send_raw_email(Source=sender, Destinations=[to_email],
                          RawMessage={"Data": msg.as_bytes()})


def _dispatch(to_email: str, subject: str, text: str, html: str | None,
              reply_to: str | None, dev_line: str) -> None:
    backend = _backend()
    if backend == "dev":
        # dev stub: log so local/CI flows can read codes from server output.
        print(dev_line, file=sys.stderr, flush=True)
        return
    msg = _compose(to_email, subject, text, html, reply_to)
    if backend == "smtp":
        _send_smtp(msg)
    else:
        _send_ses(msg, to_email)


# ───────────────────────── public API ─────────────────────────

def _auth_blocks(code: str, purpose: str, to_email: str) -> list[tuple]:
    blocks: list[tuple] = [
        ("p", _INTROS.get(purpose, "Your CORTEX code:")),
        ("code", code),
    ]
    link = _one_click_link(code, purpose, to_email)
    if link:
        blocks.append(("button", _LINK_LABELS.get(purpose, "Open CORTEX"), link))
    blocks.append(("p", "This code expires in 15 minutes. If you did not "
                        "request it, you can safely ignore this email."))
    return blocks


def _body(code: str, purpose: str, to_email: str) -> str:
    """Plain-text part of an auth-code letter (kept as a named function: the
    tests pin the code + one-click link into this exact part)."""
    return _render_text(_auth_blocks(code, purpose, to_email))


def _html_body(code: str, purpose: str, to_email: str) -> str:
    return _render_html(_TITLES.get(purpose, "Your CORTEX code"),
                        _auth_blocks(code, purpose, to_email))


def send_auth_code(to_email: str, code: str, purpose: str) -> None:
    """Send a verification/reset code letter. SMTP/SES errors propagate so the
    endpoint can react; the dev stub never raises."""
    subject = _SUBJECTS.get(purpose, "Your CORTEX code")
    _dispatch(to_email, subject,
              _body(code, purpose, to_email), _html_body(code, purpose, to_email),
              reply_to=None,
              dev_line=f"[cortex.email:dev] to={to_email} purpose={purpose} code={code}")


def send_letter(to_email: str, subject: str, title: str,
                paragraphs: list[str], button: tuple[str, str] | None = None) -> None:
    """A branded letter from arbitrary paragraphs (+ optional CTA button) —
    the cohort-invitation path. SMTP/SES errors propagate; dev stub logs."""
    blocks: list[tuple] = [("p", p) for p in paragraphs]
    if button is not None:
        blocks.append(("button", button[0], button[1]))
    _dispatch(to_email, subject,
              _render_text(blocks), _render_html(title, blocks),
              reply_to=None,
              dev_line=f"[cortex.email:dev] to={to_email} subject={subject!r}")


def send_email(to_email: str, subject: str, body: str, reply_to: str | None = None) -> None:
    """Plain-text-only transactional email (the support-report path, where the
    body is a preformatted diagnostic block, not a letter). Same backend
    dispatch; SMTP/SES errors propagate; the dev stub never raises."""
    _dispatch(to_email, subject, body, None, reply_to,
              dev_line=f"[cortex.email:dev] to={to_email} reply_to={reply_to} "
                       f"subject={subject!r}\n{body}")
