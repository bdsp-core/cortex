"""Backend tests — security primitives + full API round-trip on a temp DB.

Run from cortex_web/services/ (that dir must be on sys.path for the package-
relative imports):
    python -m pytest api/test_server.py -q
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from . import helpers, security
from .app import create_app
from .db import Database


# ───────────────────────── security unit ─────────────────────────

def test_password_roundtrip():
    h = security.hash_password("correct horse")
    assert security.verify_password("correct horse", h)
    assert not security.verify_password("wrong", h)
    assert not security.verify_password("correct horse", "garbage")


def test_password_hashes_are_salted():
    a = security.hash_password("same")
    b = security.hash_password("same")
    assert a != b  # distinct salts
    assert security.verify_password("same", a)
    assert security.verify_password("same", b)


def test_jwt_roundtrip_and_expiry():
    now = 1_000_000
    tok = security.issue_token("cortex-x", ttl_seconds=100, now=now)
    claims = security.decode_token(tok, now=now + 10)
    assert claims["sub"] == "cortex-x"
    with pytest.raises(security.TokenError):
        security.decode_token(tok, now=now + 200)  # expired


def test_jwt_tamper_rejected():
    tok = security.issue_token("cortex-x", ttl_seconds=100)
    h, p, s = tok.split(".")
    forged = f"{h}.{p}.{'A' * len(s)}"
    with pytest.raises(security.TokenError):
        security.decode_token(forged)


# ───────────────────────── API round-trip ─────────────────────────

def _write_test_bank(bundle_dir, version="test-bank", per_class=8):
    """Write a minimal web bundle manifest (7 classes × per_class segs) so the
    server-side draw (POST /api/session) has something to sample. No EEG blobs —
    server tests never run the browser engine. CI-safe (no real bundle needed)."""
    words = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]
    segs, sid = [], 0
    for k, w in enumerate(words):
        for i in range(per_class):
            sm = [0.0] * 7
            sm[k] = 1.0 + 0.05 * i           # informative on the true task
            segs.append({
                "segId": sid, "patternClass": w,
                "testClass": "spike" if w == "spike" else "iiic",
                "sMean": sm, "sSd": [0.3] * 7,
                "applicableTaskIdx": [0] if w == "spike" else [1, 2, 3, 4, 5, 6],
                "fsHz": 200, "nCh": 20, "nSamp": 100, "channelNames": [],
                "specShape": None, "eeg": f"seg/{sid}.eeg", "spec": "",
            })
            sid += 1
    manifest = {
        "version": version, "eegScale": 4.0, "specDbRange": [-10.0, 25.0],
        "taskCodes": ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
        "taskLabels": ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
        "taskPatternWords": words,
        "taskClasses": ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
        "certBlock": "test", "ellStar": [0.0] * 7,
        "corrL": [[1.0 if i == j else 0.0 for j in range(7)] for i in range(7)],
        "corrT": [[1.0 if i == j else 0.0 for j in range(7)] for i in range(7)],
        "nParticles": 1200, "nSegments": len(segs), "segments": segs,
    }
    d = Path(bundle_dir) / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("CORTEX_JWT_SECRET", "test-secret")
    monkeypatch.setenv("CORTEX_EMAIL_BACKEND", "dev")        # no real SES in tests
    monkeypatch.setenv("CORTEX_EMAIL_EXPOSE_CODE", "1")      # echo codes for the flow
    # A tiny fixture question bank so POST /api/session can draw (goal 3).
    _write_test_bank(tmp_path / "bundle", "test-bank", per_class=8)
    monkeypatch.setenv("CORTEX_BUNDLE_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("CORTEX_BUNDLE_URL", "/bundle/test-bank")
    monkeypatch.setenv("CORTEX_SESSION_SAMPLE", "21")        # draw 21 of 56 → 3/class
    app = create_app(db_path=tmp_path / "t.db")
    return TestClient(app)


# The real deliverability check, captured before the autouse bypass below so
# its own unit tests (test_domain_check_*) can still exercise it.
_REAL_DOMAIN_CHECK = helpers.email_domain_deliverable


@pytest.fixture(autouse=True)
def _dns_check_open(monkeypatch):
    """Bypass the register-time DNS deliverability check: tests sign up with
    @example.test addresses (NXDOMAIN in real DNS), and the suite must stay
    deterministic offline. The dedicated tests re-patch or call the captured
    original."""
    monkeypatch.setattr(helpers, "email_domain_deliverable", lambda domain: True)


_REG_COUNTER = [0]


def _register(client) -> tuple[str, str, str]:
    """Register a fresh account (still unverified). Return (email, pw, devCode)."""
    _REG_COUNTER[0] += 1
    email = f"t{_REG_COUNTER[0]}@example.test"
    pw = "test-pw-1234567890"
    r = client.post("/api/register",
                    json={"email": email, "password": pw,
                          "displayName": f"Test User {_REG_COUNTER[0]}"})
    assert r.status_code == 200, r.text
    assert r.json().get("needsVerification") is True
    return email, pw, r.json()["devCode"]


def _make_participant(client) -> tuple[str, str]:
    """Register AND verify an account; return (email, password) ready to auth."""
    email, pw, code = _register(client)
    r = client.post("/api/verify/confirm", json={"email": email, "code": code})
    assert r.status_code == 200, r.text
    return email, pw


def _auth_header(client, email, password) -> dict:
    r = client.post("/api/auth", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True


def test_auth_rejects_bad_password(client):
    email, _pw = _make_participant(client)
    r = client.post("/api/auth", json={"email": email, "password": "nope"})
    assert r.status_code == 401


def test_register_rejects_short_password(client):
    r = client.post("/api/register",
                    json={"email": "x@y.com", "password": "short",
                          "displayName": "X"})
    assert r.status_code == 400


def test_register_rejects_bad_email(client):
    r = client.post("/api/register",
                    json={"email": "not-an-email", "password": "test-pw-1234567890",
                          "displayName": "X"})
    assert r.status_code == 400


def test_register_rejects_undeliverable_email_domain(client, monkeypatch):
    monkeypatch.setattr(helpers, "email_domain_deliverable", lambda domain: False)
    body = {"email": "x@stanfodhealthcare.org", "password": "test-pw-1234567890",
            "displayName": "Typo"}
    r = client.post("/api/register", json=body)
    assert r.status_code == 400
    assert "typo" in r.json()["error"]
    # Rejected BEFORE the row was written — the same email registers cleanly
    # (not 409) once the domain resolves.
    monkeypatch.setattr(helpers, "email_domain_deliverable", lambda domain: True)
    r2 = client.post("/api/register", json=body)
    assert r2.status_code == 200, r2.text


# ─────────────── email-domain deliverability (unit) ───────────────

def _stub_dns(monkeypatch, mx="answer", a="answer", aaaa="answer"):
    """Install a fake dnspython in sys.modules. Each of mx/a/aaaa is 'answer',
    'null-mx' (RFC 7505 '0 .' exchange), or an error mode: 'nxdomain',
    'noanswer', 'timeout'."""
    import types
    resolver = types.ModuleType("dns.resolver")

    class NXDOMAIN(Exception):
        pass

    class NoAnswer(Exception):
        pass

    class Timeout(Exception):
        pass

    resolver.NXDOMAIN, resolver.NoAnswer = NXDOMAIN, NoAnswer
    raises = {"nxdomain": NXDOMAIN, "noanswer": NoAnswer, "timeout": Timeout}
    plan = {"MX": mx, "A": a, "AAAA": aaaa}

    class Resolver:
        timeout = lifetime = None

        def resolve(self, domain, rtype):
            v = plan[rtype]
            if v in raises:
                raise raises[v]()
            exch = "." if v == "null-mx" else "mail.x.test."
            return [types.SimpleNamespace(exchange=exch)]

    resolver.Resolver = Resolver
    dns_mod = types.ModuleType("dns")
    dns_mod.resolver = resolver
    monkeypatch.setitem(sys.modules, "dns", dns_mod)
    monkeypatch.setitem(sys.modules, "dns.resolver", resolver)


def test_domain_check_mx_present(monkeypatch):
    _stub_dns(monkeypatch)
    assert _REAL_DOMAIN_CHECK("ok.test") is True


def test_domain_check_nxdomain(monkeypatch):
    _stub_dns(monkeypatch, mx="nxdomain")
    assert _REAL_DOMAIN_CHECK("stanfodhealthcare.org") is False


def test_domain_check_a_record_fallback(monkeypatch):
    _stub_dns(monkeypatch, mx="noanswer", a="answer")
    assert _REAL_DOMAIN_CHECK("amx.test") is True


def test_domain_check_no_mx_no_a(monkeypatch):
    _stub_dns(monkeypatch, mx="noanswer", a="noanswer", aaaa="noanswer")
    assert _REAL_DOMAIN_CHECK("dead.test") is False


def test_domain_check_null_mx_refuses_mail(monkeypatch):
    _stub_dns(monkeypatch, mx="null-mx")
    assert _REAL_DOMAIN_CHECK("nullmx.test") is False


def test_domain_check_fails_open_on_timeout(monkeypatch):
    _stub_dns(monkeypatch, mx="timeout")
    assert _REAL_DOMAIN_CHECK("slow.test") is True


def test_domain_check_fails_open_without_dnspython(monkeypatch):
    monkeypatch.setitem(sys.modules, "dns", None)          # import → ImportError
    monkeypatch.setitem(sys.modules, "dns.resolver", None)
    assert _REAL_DOMAIN_CHECK("whatever.test") is True


def test_register_dup_email_is_409(client):
    email, _pw = _make_participant(client)
    r = client.post("/api/register",
                    json={"email": email,
                          "password": "anotherpw1234567890",
                          "displayName": "Y"})
    assert r.status_code == 409


def test_register_retakes_unverified_account(client):
    email = "retake@example.test"
    r1 = client.post("/api/register",
                     json={"email": email, "password": "first-pw-123456789",
                           "displayName": "First Try"})
    assert r1.status_code == 200
    old_code = r1.json()["devCode"]
    # Same email again before verifying: retake in place, not a 409 dead-end.
    r2 = client.post("/api/register",
                     json={"email": email, "password": "second-pw-12345678",
                           "displayName": "Second Try"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["needsVerification"] is True
    new_code = r2.json()["devCode"]
    # the retake replaced the pending code (skip on the 1e-6 code collision)
    if old_code != new_code:
        r = client.post("/api/verify/confirm", json={"email": email, "code": old_code})
        assert r.status_code == 400
    assert client.post("/api/verify/confirm",
                       json={"email": email, "code": new_code}).status_code == 200
    # the SECOND registration's credentials are the live ones
    assert client.post("/api/auth", json={
        "email": email, "password": "second-pw-12345678"}).status_code == 200
    assert client.post("/api/auth", json={
        "email": email, "password": "first-pw-123456789"}).status_code == 401


def test_register_never_retakes_google_account(client):
    client.app.state.db.register_oauth_participant(
        code="u-goog-retake-test", email="goog-retake@example.test",
        display_name="Goog", google_sub="sub-retake-123")
    r = client.post("/api/register",
                    json={"email": "goog-retake@example.test",
                          "password": "test-pw-1234567890", "displayName": "X"})
    assert r.status_code == 409


def test_register_honeypot_silently_drops_bot(client):
    r = client.post("/api/register",
                    json={"email": "bot@example.test",
                          "password": "test-pw-1234567890",
                          "displayName": "Bot",
                          "honeypot": "https://spam.example.com"})
    assert r.status_code == 200
    # but the bot's "account" was NOT actually created. (Checked directly:
    # a same-email re-register succeeding no longer proves absence now that
    # unverified accounts can be retaken in place.)
    assert client.app.state.db.get_participant_by_email("bot@example.test") is None
    r2 = client.post("/api/register",
                     json={"email": "bot@example.test",
                           "password": "test-pw-1234567890",
                           "displayName": "Real human"})
    assert r2.status_code == 200, r2.text


def test_client_error_telemetry_logged_and_rate_limited(client, capfd):
    body = {"message": "TypeError: x is undefined", "stack": "at a\nat b",
            "url": "/", "surface": "mobile", "ua": "test-agent"}
    assert client.post("/api/client-error", json=body).status_code == 200
    out = capfd.readouterr().err
    assert "[cortex.clienterr]" in out and "TypeError: x is undefined" in out
    assert "at a | at b" in out           # newlines collapse to one line
    # over-limit posts 429 (bucket: 10/h/IP; one already spent above)
    for _ in range(9):
        assert client.post("/api/client-error", json=body).status_code == 200
    assert client.post("/api/client-error", json=body).status_code == 429
    # hostile payloads are rejected by the model's length caps
    assert client.post("/api/client-error",
                       json={"message": "x" * 501}).status_code == 422


def test_no_em_dash_in_user_facing_error_strings():
    """House messaging style: no em dashes in anything a user reads.
    HTTPException detail strings surface directly in the SPA's error UI, so
    scan every router (+ shared modules with user-facing strings) at the
    source level. Rephrase with a semicolon/period/comma instead."""
    api_dir = Path(__file__).parent
    py_files = list((api_dir / "routers").glob("*.py"))
    py_files += [api_dir / "helpers.py", api_dir / "mailer.py"]
    offenders = []
    for f in py_files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "HTTPException" in line and "—" in line:
                offenders.append(f"{f.name}:{i}")
    assert offenders == [], f"em dash in user-facing error strings: {offenders}"


def test_mailer_body_one_click_link(monkeypatch):
    from . import mailer
    monkeypatch.setenv("CORTEX_PUBLIC_ORIGIN", "https://app.example.test/")
    body = mailer._body("123456", "verify", "user+tag@example.test")
    assert ("https://app.example.test/?verifyEmail=user%2Btag%40example.test"
            "&verifyCode=123456") in body
    assert "123456" in body  # the typed-code fallback stays
    body = mailer._body("654321", "reset", "u@example.test")
    assert "resetEmail=u%40example.test&resetCode=654321" in body
    # without a public origin (dev/CI), emails stay link-free
    monkeypatch.delenv("CORTEX_PUBLIC_ORIGIN")
    assert "verifyCode" not in mailer._body("123456", "verify", "u@example.test")


def test_mailer_letter_mime(monkeypatch):
    """The letter is multipart: plain text (code + sign-off) + HTML (branded
    card, escaped content, CID logo inside the html alternative so text-only
    clients never see an attachment)."""
    from . import mailer
    monkeypatch.setenv("CORTEX_PUBLIC_ORIGIN", "https://app.example.test")
    msg = mailer._compose(
        "u@example.test", "Your CORTEX verification code",
        mailer._body("123456", "verify", "u@example.test"),
        mailer._html_body("123456", "verify", "u@example.test"), None)
    text = msg.get_body(("plain",)).get_content()
    assert "123456" in text and "The CORTEX Team" in text
    html_part = msg.get_body(("html",))
    html = html_part.get_content()
    assert "Verify your email" in html and "The CORTEX Team" in html
    assert 'src="cid:cortex-logo"' in html and "verifyCode=123456" in html
    # the logo rides inside the html alternative (multipart/related)
    related = msg.get_payload()[-1]
    assert related.get_content_type() == "multipart/related"
    assert any(p.get_content_type() == "image/png" for p in related.iter_parts())
    # user-controlled text is escaped in the HTML part
    evil = mailer._render_html("T", [("p", 'x <script>alert("y")</script>')])
    assert "<script>" not in evil and "&lt;script&gt;" in evil


# ───────────── SES bounce webhook + verify/status ─────────────

_SES_TOKEN = "test-sns-token"
_SES_TOPIC = "arn:aws:sns:us-west-2:000000000000:cortex-ses-events"


def _sns_env(monkeypatch):
    monkeypatch.setenv("CORTEX_SNS_WEBHOOK_TOKEN", _SES_TOKEN)
    monkeypatch.setenv("CORTEX_SNS_TOPIC_ARN", _SES_TOPIC)


def _bounce_envelope(email, bounce_type="Permanent", topic=_SES_TOPIC):
    return {
        "Type": "Notification",
        "TopicArn": topic,
        "Message": json.dumps({
            "eventType": "Bounce",
            "bounce": {
                "bounceType": bounce_type,
                "bounceSubType": "General",
                "bouncedRecipients": [{"emailAddress": email}],
            },
        }),
    }


def _post_sns(client, envelope, token=_SES_TOKEN):
    # SNS posts text/plain — send raw content, not FastAPI-parsed JSON.
    return client.post(f"/api/ses/events?token={token}",
                       content=json.dumps(envelope),
                       headers={"Content-Type": "text/plain"})


def test_ses_webhook_404_when_disabled(client):
    r = _post_sns(client, _bounce_envelope("x@example.test"))
    assert r.status_code == 404


def test_ses_webhook_403_on_bad_token(client, monkeypatch):
    _sns_env(monkeypatch)
    r = _post_sns(client, _bounce_envelope("x@example.test"), token="wrong")
    assert r.status_code == 403


def test_ses_webhook_confirms_subscription(client, monkeypatch):
    _sns_env(monkeypatch)
    from .routers import ses_events
    fetched = []
    monkeypatch.setattr(ses_events, "_http_get", lambda url: fetched.append(url))
    url = "https://sns.us-west-2.amazonaws.com/?Action=ConfirmSubscription&Token=abc"
    r = _post_sns(client, {"Type": "SubscriptionConfirmation",
                           "TopicArn": _SES_TOPIC, "SubscribeURL": url})
    assert r.status_code == 200 and fetched == [url]
    # ...but never a non-SNS callback URL
    r = _post_sns(client, {"Type": "SubscriptionConfirmation",
                           "TopicArn": _SES_TOPIC,
                           "SubscribeURL": "https://evil.example.com/x"})
    assert r.status_code == 400 and len(fetched) == 1


def test_ses_bounce_flags_pending_signup(client, monkeypatch):
    _sns_env(monkeypatch)
    email, _pw, code = _register(client)
    # before the bounce: nothing to report
    r = client.post("/api/verify/status", json={"email": email})
    assert r.json() == {"undeliverable": False}
    assert _post_sns(client, _bounce_envelope(email)).json()["flagged"] == 1
    r = client.post("/api/verify/status", json={"email": email})
    assert r.json() == {"undeliverable": True}
    # confirming the code proves delivery → flag clears, status goes quiet
    assert client.post("/api/verify/confirm",
                       json={"email": email, "code": code}).status_code == 200
    r = client.post("/api/verify/status", json={"email": email})
    assert r.json() == {"undeliverable": False}


def test_ses_bounce_ignores_transient_and_foreign_topic(client, monkeypatch):
    _sns_env(monkeypatch)
    email, _pw, _code = _register(client)
    assert _post_sns(client, _bounce_envelope(email, bounce_type="Transient")
                     ).json() == {"ok": True}
    assert _post_sns(client, _bounce_envelope(
        email, topic="arn:aws:sns:us-west-2:000000000000:other")
                     ).json() == {"ok": True}
    r = client.post("/api/verify/status", json={"email": email})
    assert r.json() == {"undeliverable": False}


def test_verify_status_anti_oracle_on_unknown_email(client, monkeypatch):
    _sns_env(monkeypatch)
    r = client.post("/api/verify/status", json={"email": "ghost@example.test"})
    assert r.status_code == 200 and r.json() == {"undeliverable": False}
    # a bounce for an email with no account is a no-op, not an error
    assert _post_sns(client, _bounce_envelope("ghost@example.test")
                     ).json()["flagged"] == 0


def test_gated_requires_token(client):
    assert client.get("/api/manifest").status_code == 401
    assert client.post("/api/session", json={"participant": {}}).status_code == 401


def test_full_session_flow(client):
    code, pw = _make_participant(client)
    hdr = _auth_header(client, code, pw)

    man = client.get("/api/manifest", headers=hdr).json()
    assert man["sessionSample"] == 21
    assert man["version"] == "test-bank"        # real bundle id, not "v1.1-local"

    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {"expertise": "attending",
                                            "institution": "MGH"},
                            "sampleSeed": 12345}).json()["sessionId"]
    assert sid

    # per-trial checkpoints
    for i in range(3):
        r = client.post("/api/progress", headers=hdr,
                        json={"sessionId": sid,
                              "trial": {"trialIndex": i, "segId": 100 + i,
                                        "taskK": i % 6, "pick": 0,
                                        "isCorrect": True, "reactionMs": 1500.0}})
        assert r.status_code == 200

    # finalize
    r = client.post("/api/results", headers=hdr,
                    json={"sessionId": sid,
                          "result": {"verdicts": ["PASS"] * 6},
                          "stopReason": "all_resolved", "nQuestions": 42})
    assert r.status_code == 200

    # admin can read it back
    res = client.get(f"/api/admin/results/{sid}",
                     headers={"X-Admin-Token": "test-admin"}).json()
    assert res["verdicts"] == ["PASS"] * 6

    sessions = client.get("/api/admin/sessions",
                          headers={"X-Admin-Token": "test-admin"}).json()
    row = next(s for s in sessions if s["session_id"] == sid)
    assert row["status"] == "complete"
    assert row["n_questions"] == 42


def test_progress_rejects_foreign_session(client):
    code_a, pw_a = _make_participant(client)
    code_b, pw_b = _make_participant(client)
    hdr_a = _auth_header(client, code_a, pw_a)
    hdr_b = _auth_header(client, code_b, pw_b)
    sid = client.post("/api/session", headers=hdr_a,
                      json={"participant": {}}).json()["sessionId"]
    # B must not be able to write to A's session
    r = client.post("/api/progress", headers=hdr_b,
                    json={"sessionId": sid, "trial": {"trialIndex": 0}})
    assert r.status_code == 404


def test_session_resume_roundtrip(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    # nothing to resume yet
    assert client.get("/api/session/active", headers=hdr).json() == {"active": None}
    start = client.post("/api/session", headers=hdr,
                        json={"participant": {}, "sampleSeed": 99}).json()
    sid = start["sessionId"]
    drawn_ids = [s["segId"] for s in start["bank"]["segments"]]
    # a sitting with no checkpointed trials is not worth resuming
    assert client.get("/api/session/active", headers=hdr).json() == {"active": None}
    for i in range(3):
        client.post("/api/progress", headers=hdr,
                    json={"sessionId": sid,
                          "trial": {"trialIndex": i, "segId": drawn_ids[i],
                                    "taskK": 1, "pick": i % 2,
                                    "isCorrect": True, "reactionMs": 900.0}})
    active = client.get("/api/session/active", headers=hdr).json()["active"]
    assert active["sessionId"] == sid
    assert active["bank"]["sampleSeed"] == 99
    # the pool replays verbatim, in the original draw order
    assert [s["segId"] for s in active["bank"]["segments"]] == drawn_ids
    assert active["trials"] == [
        {"trialIndex": i, "segId": drawn_ids[i], "pick": i % 2} for i in range(3)]
    # finalizing ends resumability
    client.post("/api/results", headers=hdr,
                json={"sessionId": sid, "result": {"verdicts": ["PASS"] * 6},
                      "stopReason": "all_resolved", "nQuestions": 3})
    assert client.get("/api/session/active", headers=hdr).json() == {"active": None}


def test_session_resume_replay_stops_at_log_gap(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    start = client.post("/api/session", headers=hdr, json={"participant": {}}).json()
    sid = start["sessionId"]
    ids = [s["segId"] for s in start["bank"]["segments"]]
    for i in (0, 2):   # hole at index 1
        client.post("/api/progress", headers=hdr,
                    json={"sessionId": sid,
                          "trial": {"trialIndex": i, "segId": ids[i], "pick": 0}})
    active = client.get("/api/session/active", headers=hdr).json()["active"]
    assert [t["trialIndex"] for t in active["trials"]] == [0]


def test_new_session_supersedes_open_one(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    first = client.post("/api/session", headers=hdr, json={"participant": {}}).json()
    client.post("/api/progress", headers=hdr,
                json={"sessionId": first["sessionId"],
                      "trial": {"trialIndex": 0, "pick": 1,
                                "segId": first["bank"]["segments"][0]["segId"]}})
    second = client.post("/api/session", headers=hdr, json={"participant": {}}).json()
    client.post("/api/progress", headers=hdr,
                json={"sessionId": second["sessionId"],
                      "trial": {"trialIndex": 0, "pick": 1,
                                "segId": second["bank"]["segments"][0]["segId"]}})
    active = client.get("/api/session/active", headers=hdr).json()["active"]
    assert active["sessionId"] == second["sessionId"]


def test_session_returns_balanced_server_drawn_bank(client):
    code, pw = _make_participant(client)
    hdr = _auth_header(client, code, pw)
    body = client.post("/api/session", headers=hdr,
                       json={"participant": {}, "sampleSeed": 7}).json()
    assert body["sessionId"] and body["sampleSeed"] == 7
    bank = body["bank"]
    assert bank["bundleUrl"] == "/bundle/test-bank"
    assert bank["version"] == "test-bank"
    assert bank["nParticles"] == 1200                 # v15 engine inputs flow through
    segs = bank["segments"]
    assert len(segs) == 21                            # CORTEX_SESSION_SAMPLE
    # balanced: every one of the 7 classes represented (3 each)
    from collections import Counter
    classes = Counter(s["patternClass"] for s in segs)
    assert len(classes) == 7 and all(c == 3 for c in classes.values())
    # the session row is stamped with the bundle version (provenance O3)
    row = next(s for s in client.get("/api/admin/sessions",
                                     headers={"X-Admin-Token": "test-admin"}).json()
               if s["session_id"] == body["sessionId"])
    assert row.get("bundle_version") == "test-bank"


def test_session_draw_excludes_recently_seen_segments(client):
    code, pw = _make_participant(client)
    hdr = _auth_header(client, code, pw)
    # First sitting: record every served seg as the SPA would (via /api/progress).
    first = client.post("/api/session", headers=hdr,
                        json={"participant": {}, "sampleSeed": 1}).json()
    seen = [s["segId"] for s in first["bank"]["segments"]]
    for i, sid in enumerate(seen):
        client.post("/api/progress", headers=hdr, json={
            "sessionId": first["sessionId"],
            "trial": {"trialIndex": i, "segId": sid, "taskK": 0}})
    # Second sitting (same participant): the draw must exclude every seen seg.
    second = client.post("/api/session", headers=hdr,
                         json={"participant": {}, "sampleSeed": 2}).json()
    got = {s["segId"] for s in second["bank"]["segments"]}
    assert got.isdisjoint(set(seen))


def test_tutorial_example_returns_one_iiic_segment(client):
    code, pw = _make_participant(client)
    hdr = _auth_header(client, code, pw)
    ex = client.get("/api/tutorial-example", headers=hdr).json()
    assert ex["bundleUrl"] == "/bundle/test-bank"
    assert len(ex["segments"]) == 1
    assert ex["segments"][0]["testClass"] == "iiic"


def test_exposure_exclusion_windows(tmp_path):
    db = Database(tmp_path / "x.db")
    with db._connection() as conn:                   # FK: sessions.code → participants
        conn.execute("INSERT INTO participants(code, password_hash, created_utc) "
                     "VALUES (?,?,?)", ("C", "x", "2000-01-01T00:00:00Z"))
    db.create_session("sA", "C", {}, 1)
    db.upsert_trial("sA", {"trialIndex": 0, "segId": 10, "taskK": 0})
    db.create_session("sB", "C", {}, 2)
    db.upsert_trial("sB", {"trialIndex": 0, "segId": 20, "taskK": 0})
    with db._connection() as conn:                   # backdate sA far into the past
        conn.execute("UPDATE sessions SET started_utc=? WHERE session_id=?",
                     ("2000-01-01T00:00:00Z", "sA"))
    # narrow windows → only the recent sB (last 1 session AND within 30 days)
    excl = db.get_exposure_exclusion("C", days_window=30, session_window=1)
    assert 20 in excl and 10 not in excl
    # wide day window → sA re-enters via the time clause despite session_window=1
    excl2 = db.get_exposure_exclusion("C", days_window=99999, session_window=1)
    assert {10, 20} <= excl2


def test_admin_requires_token(client):
    assert client.get("/api/admin/sessions").status_code == 403
    assert client.get("/api/admin/sessions",
                      headers={"X-Admin-Token": "wrong"}).status_code == 403


def test_disabled_participant_cannot_auth(client, tmp_path):
    email, pw = _make_participant(client)
    # works while active
    assert client.post("/api/auth", json={"email": email, "password": pw}).status_code == 200
    # disable it directly in the DB the app is using, then auth must fail
    row = client.app.state.db.get_participant_by_email(email)
    client.app.state.db.set_participant_active(row["code"], False)
    assert client.post("/api/auth", json={"email": email, "password": pw}).status_code == 401


def test_expired_token_rejected_by_api(client, monkeypatch):
    code, pw = _make_participant(client)
    # mint a token that is already expired
    from . import security
    tok = security.issue_token(code, ttl_seconds=-1)
    r = client.get("/api/manifest", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401


# ───────────────── email verification + password reset ─────────────────

def test_unverified_account_cannot_auth(client):
    email, pw, _code = _register(client)
    r = client.post("/api/auth", json={"email": email, "password": pw})
    assert r.status_code == 403
    assert r.json()["error"] == "email_not_verified"


def test_verify_then_auth_succeeds(client):
    email, pw, code = _register(client)
    assert client.post("/api/verify/confirm",
                       json={"email": email, "code": code}).status_code == 200
    r = client.post("/api/auth", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    assert r.json()["token"]


def test_verify_rejects_wrong_code(client):
    email, _pw, code = _register(client)
    wrong = "000000" if code != "000000" else "111111"
    assert client.post("/api/verify/confirm",
                       json={"email": email, "code": wrong}).status_code == 400


def test_verify_attempt_cap(client):
    email, _pw, code = _register(client)
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(security.CODE_MAX_ATTEMPTS):
        assert client.post("/api/verify/confirm",
                           json={"email": email, "code": wrong}).status_code == 400
    # even the CORRECT code is now refused — the code is locked out
    assert client.post("/api/verify/confirm",
                       json={"email": email, "code": code}).status_code == 400


def test_resend_issues_new_code_and_verifies(client):
    email, _pw, _old = _register(client)
    r = client.post("/api/verify/resend", json={"email": email})
    assert r.status_code == 200
    new_code = r.json()["devCode"]
    assert client.post("/api/verify/confirm",
                       json={"email": email, "code": new_code}).status_code == 200


def test_resend_unknown_email_is_200_no_code(client):
    r = client.post("/api/verify/resend", json={"email": "nobody@example.test"})
    assert r.status_code == 200
    assert "devCode" not in r.json()


def test_forgot_reset_flow(client):
    email, pw = _make_participant(client)
    r = client.post("/api/forgot", json={"email": email})
    assert r.status_code == 200
    code = r.json()["devCode"]
    newpw = "brand-new-pw-9876543210"
    assert client.post("/api/reset",
                       json={"email": email, "code": code,
                             "newPassword": newpw}).status_code == 200
    # old password no longer works, new one does
    assert client.post("/api/auth", json={"email": email, "password": pw}).status_code == 401
    assert client.post("/api/auth", json={"email": email, "password": newpw}).status_code == 200


def test_reset_rejects_wrong_code(client):
    email, _pw = _make_participant(client)
    client.post("/api/forgot", json={"email": email})
    r = client.post("/api/reset",
                    json={"email": email, "code": "000000",
                          "newPassword": "another-good-pw-123"})
    assert r.status_code == 400


def test_forgot_unknown_email_is_200(client):
    r = client.post("/api/forgot", json={"email": "ghost@example.test"})
    assert r.status_code == 200
    assert "devCode" not in r.json()


# ───────────────── dashboard / learning-protocol (Phase 2) ─────────────────

def test_dashboard_requires_auth(client):
    assert client.get("/api/dashboard").status_code == 401
    assert client.get("/api/trajectories").status_code == 401
    assert client.get("/api/regimen").status_code == 401


def test_dashboard_empty_has_no_data(client):
    # No certification result yet → no tasks, no KPIs, never sample data.
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    d = client.get("/api/dashboard", headers=hdr).json()
    assert d["hasResult"] is False and d["result"] is None
    assert d["tasks"] == [] and d["kpis"] is None
    assert d["sample"] is False


_SEVEN = [(0, "spike", "Spike"), (1, "sz", "Seizure"), (2, "lpd", "LPD"),
          (3, "gpd", "GPD"), (4, "lrda", "LRDA"), (5, "grda", "GRDA"),
          (6, "iic", "Other")]


def test_dashboard_reflects_real_per_task(client):
    # A Step-1 result carries `perTask` with real ℓ/θ/ℓ*/AUROC; the dashboard
    # surfaces those verbatim and derives real cert-summary KPIs.
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    per = [{"taskK": k, "code": c, "label": lab, "ell": 1.0 + k, "theta": 0.1,
            "ellStar": 0.3, "auroc": 0.8, "aurocHw": 0.05,
            "verdict": "PASS" if k < 5 else "FAIL"} for k, c, lab in _SEVEN]
    client.post("/api/results", headers=hdr,
                json={"sessionId": sid,
                      "result": {"verdicts": ["PASS"] * 5 + ["FAIL"] * 2,
                                 "perTask": per},
                      "stopReason": "all_resolved", "nQuestions": 30})
    d = client.get("/api/dashboard", headers=hdr).json()
    assert d["hasResult"] is True and d["sample"] is False
    assert len(d["tasks"]) == 7
    t0 = d["tasks"][0]
    assert t0["ell"] == 1.0 and t0["ellStar"] == 0.3 and t0["auroc"] == 0.8
    assert t0["theta"] == 0.1 and t0["verdict"] == "PASS"
    k = d["kpis"]
    assert k["tasksCertified"] == 5 and k["tasksTotal"] == 7
    assert k["meanAuroc"] == 0.8 and k["lastAssessed"]


def test_dashboard_legacy_result_verdicts_only(client):
    # A pre-Step-1 result has only `verdicts` (no perTask) → verdict shows, but
    # ℓ/ℓ*/AUROC come back None and meanAuroc is None (nothing fabricated).
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    client.post("/api/results", headers=hdr,
                json={"sessionId": sid, "result": {"verdicts": ["PASS"] * 7},
                      "stopReason": "all_resolved", "nQuestions": 30})
    d = client.get("/api/dashboard", headers=hdr).json()
    assert d["hasResult"] is True and len(d["tasks"]) == 7
    assert all(t["verdict"] == "PASS" for t in d["tasks"])
    assert all(t["ell"] is None and t["theta"] is None and t["auroc"] is None
               for t in d["tasks"])
    assert d["kpis"]["tasksCertified"] == 7 and d["kpis"]["meanAuroc"] is None


def test_dashboard_ring_uses_latest_trajectory(client):
    # The mastery ring's ℓ tracks the LATEST real measurement: a training point
    # added after the cert eval (newer ts) overrides the cert ℓ; ℓ* is unchanged.
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    per = [{"taskK": k, "code": c, "label": lab, "ell": 0.2, "theta": 0.0,
            "sd": 0.2, "ellStar": 0.3, "auroc": 0.8, "verdict": "FAIL"}
           for k, c, lab in _SEVEN]
    client.post("/api/results", headers=hdr, json={"sessionId": sid,
        "result": {"verdicts": ["FAIL"] * 7, "perTask": per},
        "stopReason": "x", "nQuestions": 30})
    # before training: ℓ is the cert eval value
    assert client.get("/api/dashboard", headers=hdr).json()["tasks"][0]["ell"] == 0.2
    # a later real training point for domain 0 raises ℓ (newer ts wins)
    code = client.app.state.db.get_participant_by_email(email)["code"]
    client.app.state.db.append_trajectory_points(code, [
        {"taskK": 0, "phase": "train", "ell": 0.9, "theta": 0.15, "sd": 0.1,
         "rt": 1000, "ts": "2099-01-01T00:00:00Z", "isReal": True}])
    d = client.get("/api/dashboard", headers=hdr).json()
    t0 = next(t for t in d["tasks"] if t["taskK"] == 0)
    assert t0["ell"] == 0.9 and t0["theta"] == 0.15   # latest measurement
    assert t0["ellStar"] == 0.3                         # threshold unchanged
    assert next(t for t in d["tasks"] if t["taskK"] == 1)["ell"] == 0.2  # untouched


def test_history_empty_then_after_result_with_isolation(client):
    e1, p1 = _make_participant(client)
    e2, p2 = _make_participant(client)
    h1 = _auth_header(client, e1, p1)
    h2 = _auth_header(client, e2, p2)

    # empty to start
    assert client.get("/api/history", headers=h1).json()["sessions"] == []

    # participant 1 posts a completed result
    sid = client.post("/api/session", headers=h1,
                      json={"participant": {}}).json()["sessionId"]
    client.post("/api/results", headers=h1,
                json={"sessionId": sid,
                      "result": {"verdicts": ["PASS"] * 7,
                                 "roc": [{"auroc": 0.9}] * 7},
                      "stopReason": "all_resolved", "nQuestions": 33})

    h = client.get("/api/history", headers=h1).json()["sessions"]
    assert len(h) == 1
    assert h[0]["session_id"] == sid
    assert h[0]["n_questions"] == 33
    assert h[0]["stop_reason"] == "all_resolved"
    assert h[0]["result"]["verdicts"] == ["PASS"] * 7

    # isolation: participant 2 does not see participant 1's attempt
    assert client.get("/api/history", headers=h2).json()["sessions"] == []


def test_history_requires_auth(client):
    assert client.get("/api/history").status_code == 401


_TRUTH = {"seg": {100: "lpd", 200: "gpd"},
          "words": ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
          "classes": ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
          "labels": ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]}


def _diag(k, y, s, R, pi=0.5, ell=0.3, theta=-0.1):
    d = {"taskK": k, "y": y, "s": s, "R": [0.0] * 7, "pi": [0.0] * 7,
         "lMean": [0.0] * 7, "tMean": [0.0] * 7}
    d["R"][k] = R; d["pi"][k] = pi; d["lMean"][k] = ell; d["tMean"][k] = theta
    return d


def test_question_breakdown_pure():
    from .dashboard_logic import question_breakdown as _question_breakdown
    trials = [
        # spike, said YES, s>0 → correct; ΔR from 0 → 0.2
        {"trial_index": 0, "seg_id": 50, "task_k": 0, "pick": 0, "reaction_ms": 1500,
         "diag": _diag(0, 1, 0.8, 0.2, pi=0.6, ell=0.5, theta=0.1)},
        # IIIC: chose LPD (pick=2) on seg100 (truth lpd) → correct; ΔR 0→0.3
        {"trial_index": 1, "seg_id": 100, "task_k": 2, "pick": 2, "reaction_ms": 2000,
         "diag": _diag(2, 1, 0.0, 0.3)},
        # IIIC: chose LPD (pick=2) on seg200 (truth GPD) → INCORRECT; the user's
        # example: answer "LPD", correct "GPD". ΔR 0.3→0.5 = 0.2 (same domain).
        {"trial_index": 2, "seg_id": 200, "task_k": 2, "pick": 2, "reaction_ms": 1800,
         "diag": _diag(2, 0, 0.0, 0.5)},
    ]
    rows = _question_breakdown(trials, _TRUTH)
    # domain reads the PHASE (Spike / IIIC), not the probed sub-domain
    assert rows[0]["q"] == 1 and rows[0]["domain"] == "Spike"
    assert rows[1]["domain"] == "IIIC" and rows[2]["domain"] == "IIIC"
    assert rows[0]["answer"] == "Yes" and rows[0]["correct"] == "Yes" and rows[0]["isCorrect"] is True
    assert rows[0]["deltaR"] == 0.2 and rows[0]["pi"] == 0.6 and rows[0]["ell"] == 0.5
    assert rows[1]["answer"] == "LPD" and rows[1]["correct"] == "LPD" and rows[1]["isCorrect"] is True
    assert rows[1]["deltaR"] == 0.3
    assert rows[2]["answer"] == "LPD" and rows[2]["correct"] == "GPD" and rows[2]["isCorrect"] is False
    assert abs(rows[2]["deltaR"] - 0.2) < 1e-9   # cumulative-R delta on same domain


def test_history_questions_endpoint(client, monkeypatch):
    from . import dashboard_logic
    monkeypatch.setattr(dashboard_logic, "_truth_map_for", lambda *a, **k: _TRUTH)
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    # (taskK, y, s, R, pick, segId) — diag stored as a JSON string by the DB
    rows = [(0, 1, 0.9, 0.2, 0, 50),     # spike: said Yes, s>0 → correct
            (0, 0, -0.5, 0.35, 7, 51),   # spike: said No (sentinel pick), s<0 → correct
            (2, 0, 0.0, 0.5, 2, 200)]    # IIIC: chose LPD on a GPD segment → incorrect
    for i, (k, y, s, R, pick, seg) in enumerate(rows):
        client.post("/api/progress", headers=hdr, json={"sessionId": sid,
            "trial": {"trialIndex": i, "segId": seg, "taskK": k,
                      "pick": pick, "reactionMs": 1500 + i, "diag": _diag(k, y, s, R)}})
    r = client.get(f"/api/history/{sid}/questions", headers=hdr).json()
    assert r["nQuestions"] == 3
    q0, q1, q2 = r["questions"]
    assert q0["domain"] == "Spike" and q0["answer"] == "Yes" and q0["isCorrect"] is True and q0["rt"] == 1500
    assert q1["answer"] == "No" and q1["correct"] == "No" and q1["isCorrect"] is True
    assert q2["domain"] == "IIIC" and q2["answer"] == "LPD" and q2["correct"] == "GPD" and q2["isCorrect"] is False

    # auth-scoped: another participant cannot read this session's questions
    e2, p2 = _make_participant(client)
    assert client.get(f"/api/history/{sid}/questions",
                      headers=_auth_header(client, e2, p2)).status_code == 404


def test_truth_map_resolves_per_bundle_version(tmp_path, monkeypatch):
    # The label bug: a session's per-question correct answer must be resolved
    # against the bundle that PRODUCED it (its segId space), not a global "first
    # manifest". Two bundles assign the SAME segId to DIFFERENT classes; the
    # truth map must follow bundle_version.
    from . import config, dashboard_logic
    bdir = tmp_path / "bundle"
    _write_test_bank(bdir, "aaa-first", per_class=4)    # sorts first → legacy fallback
    _write_test_bank(bdir, "zzz-second", per_class=4)
    # Repaint seg 5's class differently in each so the maps must diverge.
    import json as _json
    for ver, cls in (("aaa-first", "lpd"), ("zzz-second", "gpd")):
        mp = bdir / ver / "manifest.json"
        m = _json.loads(mp.read_text())
        for s in m["segments"]:
            if s["segId"] == 5:
                s["patternClass"] = cls
        mp.write_text(_json.dumps(m))
    monkeypatch.setattr(config, "BUNDLE_DIR", bdir)
    monkeypatch.setattr(dashboard_logic, "_TRUTH_CACHE", {})
    # version routes to its OWN bundle (the fix)
    assert dashboard_logic._truth_map_for("aaa-first")["seg"][5] == "lpd"
    assert dashboard_logic._truth_map_for("zzz-second")["seg"][5] == "gpd"
    # unknown/legacy (NULL bundle_version) → alphabetically-first bundle
    assert dashboard_logic._truth_map_for(None)["seg"][5] == "lpd"
    assert dashboard_logic._truth_map_for("nonexistent")["seg"][5] == "lpd"


def test_history_questions_requires_auth(client):
    assert client.get("/api/history/whatever/questions").status_code == 401


def test_activity_heatmap_levels(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)   # /api/auth records today as a login day
    today = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())[:10]
    assert client.get("/api/activity", headers=hdr).json()["days"].get(today) == 1

    # a completed cert test bumps today to level 2
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    client.post("/api/results", headers=hdr, json={"sessionId": sid,
                "result": {"verdicts": ["PASS"] * 7}, "stopReason": "x", "nQuestions": 1})
    assert client.get("/api/activity", headers=hdr).json()["days"].get(today) == 2

    # a completed training session bumps today to level 3 (highest wins)
    db = client.app.state.db
    code = db.get_participant_by_email(email)["code"]
    db.create_training_session("tr-act-1", code, "gpd")
    db.finalize_training_session("tr-act-1", code, 5, {"mode": "practice"})
    assert client.get("/api/activity", headers=hdr).json()["days"].get(today) == 3


def test_activity_requires_auth_and_login_day_idempotent(client):
    assert client.get("/api/activity").status_code == 401
    email, pw = _make_participant(client)
    db = client.app.state.db
    code = db.get_participant_by_email(email)["code"]
    db.record_login_day(code)
    db.record_login_day(code)   # same day again — idempotent
    days = db.activity_levels(code)
    assert sum(1 for v in days.values()) == len(days)  # no duplicate-day blowup
    assert len(days) == 1 and set(days.values()) == {1}


def test_local_day_shift():
    from .db import _local_day
    # PDT (offset 420 = UTC ahead 7h): a 04:00 UTC event is still the prior day
    assert _local_day("2026-06-23T04:00:00Z", 420) == "2026-06-22"
    assert _local_day("2026-06-23T20:00:00Z", 420) == "2026-06-23"
    # UTC+2 (offset -120): a 23:00 UTC event is already the next local day
    assert _local_day("2026-06-22T23:00:00Z", -120) == "2026-06-23"
    assert _local_day("2026-06-23T04:00:00Z", 0) == "2026-06-23"   # offset 0 = UTC


def test_activity_endpoint_honors_tz(client):
    email, pw = _make_participant(client)
    db = client.app.state.db
    code = db.get_participant_by_email(email)["code"]
    # a cert session finished at 2026-01-15 04:00 UTC (past, won't collide w/ login)
    with db._connection() as conn:
        conn.execute("INSERT INTO sessions(session_id,code,participant,started_utc,"
                     "finished_utc,status) VALUES(?,?,?,?,?, 'complete')",
                     ("act-tz", code, "{}", "2026-01-15T03:00:00Z", "2026-01-15T04:00:00Z"))
    db.store_result("act-tz", {"verdicts": []})
    hdr = _auth_header(client, email, pw)
    # PDT (tz=420): the 04:00 UTC cert lands on the LOCAL Jan 14, not UTC Jan 15
    a = client.get("/api/activity?tz=420", headers=hdr).json()["days"]
    assert a.get("2026-01-14") == 2 and "2026-01-15" not in a
    # UTC (tz=0): lands on Jan 15
    assert client.get("/api/activity?tz=0", headers=hdr).json()["days"].get("2026-01-15") == 2


def test_trajectories_empty_then_real(client):
    # Empty (no fabricated curve) until a REAL (is_real=1) point exists. A point
    # posted without isReal is quarantined (is_real=0) and stays hidden.
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    t0 = client.get("/api/trajectories", headers=hdr).json()
    assert t0["sample"] is False and t0["trajectories"] == []
    # quarantined point — still hidden
    client.post("/api/trajectories", headers=hdr,
                json={"points": [{"taskK": 0, "phase": "train", "ell": 0.9,
                                  "theta": 0.1, "sd": 0.12, "rt": 2000}]})
    assert client.get("/api/trajectories", headers=hdr).json()["trajectories"] == []
    # real trainer point — surfaced
    client.post("/api/trajectories", headers=hdr,
                json={"points": [{"taskK": 1, "phase": "train", "ell": 0.8,
                                  "theta": 0.0, "sd": 0.10, "rt": 1900,
                                  "isReal": True}]})
    t1 = client.get("/api/trajectories", headers=hdr).json()
    assert t1["sample"] is False and len(t1["trajectories"]) == 1
    assert t1["trajectories"][0]["taskK"] == 1


def test_results_writes_eval_trajectory(client):
    # Finishing a cert test seeds one real EVAL trajectory point per task —
    # ℓ/θ/σ from perTask + the per-task MEDIAN reaction time from the trials.
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    # per-trial reaction times: task 0 has two (median 2050), task 1 has one (2200)
    for i, (tk, rt) in enumerate([(0, 2000), (0, 2100), (1, 2200)]):
        client.post("/api/progress", headers=hdr, json={"sessionId": sid,
            "trial": {"trialIndex": i, "taskK": tk, "reactionMs": rt}})
    per = [{"taskK": k, "code": c, "label": lab, "ell": 1.0 + k, "theta": 0.1,
            "sd": 0.2, "ellStar": 0.3, "auroc": 0.8, "verdict": "PASS"}
           for k, c, lab in _SEVEN]
    body = {"sessionId": sid, "result": {"verdicts": ["PASS"] * 7, "perTask": per},
            "stopReason": "all_resolved", "nQuestions": 30}
    client.post("/api/results", headers=hdr, json=body)

    pts = client.get("/api/trajectories", headers=hdr).json()["trajectories"]
    assert len(pts) == 7 and all(p["phase"] == "eval" for p in pts)
    p0 = next(p for p in pts if p["taskK"] == 0)
    assert p0["ell"] == 1.0 and p0["sd"] == 0.2 and p0["rt"] == 2050  # median(2000,2100)
    assert next(p for p in pts if p["taskK"] == 1)["rt"] == 2200       # single trial
    # tasks with no trials still get an eval point, with rt=None
    assert next(p for p in pts if p["taskK"] == 6)["rt"] is None

    # idempotent on re-post: still 7 eval points, not 14
    client.post("/api/results", headers=hdr, json=body)
    pts2 = client.get("/api/trajectories", headers=hdr).json()["trajectories"]
    assert len([p for p in pts2 if p["phase"] == "eval"]) == 7


def test_backfill_legacy_eval_trajectory(client):
    # A legacy result (verdicts + trials blob, NO perTask) gets its eval point
    # reconstructed from the final trial's lMean/tMean + trials reaction times.
    from . import backfill_eval_trajectories as bf
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    # trials-table rows with reaction times (median RT source): task 0 → 2050
    for i, (tk, rt) in enumerate([(0, 2000), (0, 2100), (3, 3000)]):
        client.post("/api/progress", headers=hdr, json={"sessionId": sid,
            "trial": {"trialIndex": i, "taskK": tk, "reactionMs": rt}})
    lmean = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    tmean = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    legacy = {"verdicts": ["PASS"] * 5 + ["FAIL"] * 2,
              "trials": [{"trialIndex": 0, "lMean": [0] * 7, "tMean": [0] * 7},
                         {"trialIndex": 1, "lMean": lmean, "tMean": tmean}]}
    client.post("/api/results", headers=hdr, json={"sessionId": sid, "result": legacy,
                "stopReason": "all_resolved", "nQuestions": 2})

    # before: legacy (ℓ null), no trajectory points
    d0 = client.get("/api/dashboard", headers=hdr).json()
    assert d0["tasks"][0]["ell"] is None
    assert client.get("/api/trajectories", headers=hdr).json()["trajectories"] == []

    ell_star = [0.30, 0.25, 0.50, 0.33, 0.48, 0.49, 0.44]
    db = client.app.state.db
    summary = bf.run(db, ell_star, apply=True)
    assert any(s.get("session_id") == sid and s["status"] == "reconstructed"
               for s in summary["sessions"])

    # after: dashboard ℓ/ℓ* real, 7 eval trajectory points with median RT
    d1 = client.get("/api/dashboard", headers=hdr).json()
    t0 = next(t for t in d1["tasks"] if t["taskK"] == 0)
    assert t0["ell"] == 1.0 and t0["ellStar"] == 0.30 and t0["auroc"] is None
    pts = client.get("/api/trajectories", headers=hdr).json()["trajectories"]
    assert len(pts) == 7 and all(p["phase"] == "eval" for p in pts)
    p0 = next(p for p in pts if p["taskK"] == 0)
    assert p0["ell"] == 1.0 and p0["theta"] == 0.1 and p0["rt"] == 2050  # median(2000,2100)
    assert next(p for p in pts if p["taskK"] == 1)["rt"] is None  # no trials for task 1

    # idempotent: re-run does not reconstruct this session again
    summary2 = bf.run(db, ell_star, apply=True)
    assert not any(s.get("session_id") == sid and s["status"] == "reconstructed"
                   for s in summary2["sessions"])


def test_backfill_unrecoverable_without_trials(client):
    # A legacy result with no trials array can't be reconstructed → skipped.
    from . import backfill_eval_trajectories as bf
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    client.post("/api/results", headers=hdr, json={"sessionId": sid,
                "result": {"verdicts": ["PASS"] * 7}, "stopReason": "x", "nQuestions": 7})
    summary = bf.run(client.app.state.db, [0.3] * 7, apply=True)
    assert any(s.get("session_id") == sid and s["status"] == "unrecoverable"
               for s in summary["sessions"])
    assert client.get("/api/trajectories", headers=hdr).json()["trajectories"] == []


def test_regimen_empty_then_real(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    r0 = client.get("/api/regimen", headers=hdr).json()
    assert r0["sample"] is False and r0["regimen"] is None
    row = client.app.state.db.get_participant_by_email(email)
    client.app.state.db.create_regimen("rg-1", row["code"], None,
                                        {"weeks": 6, "deck": []})
    r = client.get("/api/regimen", headers=hdr).json()
    assert r["sample"] is False and r["regimen"]["weeks"] == 6


def test_training_session_lifecycle(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    tid = client.post("/api/training-sessions", headers=hdr,
                      json={"taskFocus": "gpd"}).json()["trainingId"]
    lst = client.get("/api/training-sessions", headers=hdr).json()["sessions"]
    assert len(lst) == 1 and lst[0]["status"] == "in_progress"
    assert client.post("/api/training-sessions/finalize", headers=hdr,
                       json={"trainingId": tid, "nItems": 40,
                             "summary": {"correct": 33}}).status_code == 200
    lst2 = client.get("/api/training-sessions", headers=hdr).json()["sessions"]
    assert lst2[0]["status"] == "complete" and lst2[0]["n_items"] == 40
    # finalizing an unknown id is a 404
    assert client.post("/api/training-sessions/finalize", headers=hdr,
                       json={"trainingId": "nope"}).status_code == 404


def test_training_session_isolation(client):
    """A participant cannot finalize another participant's training session."""
    e1, p1 = _make_participant(client)
    e2, p2 = _make_participant(client)
    h1 = _auth_header(client, e1, p1)
    h2 = _auth_header(client, e2, p2)
    tid = client.post("/api/training-sessions", headers=h1, json={}).json()["trainingId"]
    assert client.post("/api/training-sessions/finalize", headers=h2,
                       json={"trainingId": tid}).status_code == 404


def test_smtp_backend_sends_code(monkeypatch):
    """The SMTP backend builds a STARTTLS-authenticated message carrying the code
    to the right recipient. smtplib is faked so no real server is needed."""
    import smtplib
    from . import mailer as email_mod

    sent: dict = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=0):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def ehlo(self):
            pass

        def starttls(self, context=None):
            sent["starttls"] = True

        def login(self, u, p):
            sent["login"] = (u, p)

        def send_message(self, msg):
            sent["to"] = msg["To"]
            sent["from"] = msg["From"]
            sent["subject"] = msg["Subject"]
            # letters are multipart (text + branded html); the code must be
            # in the PLAIN part for text-only clients
            sent["body"] = msg.get_body(("plain",)).get_content()
            sent["html"] = (msg.get_body(("html",)) or msg).get_content()

    monkeypatch.setenv("CORTEX_EMAIL_BACKEND", "smtp")
    monkeypatch.setenv("CORTEX_SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("CORTEX_SMTP_USER", "apikey")
    monkeypatch.setenv("CORTEX_SMTP_PASSWORD", "secret")
    monkeypatch.setenv("CORTEX_EMAIL_FROM", "CORTEX <no-reply@example.test>")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    email_mod.send_auth_code("alice@example.test", "123456", "verify")
    assert sent["to"] == "alice@example.test"
    assert sent["from"] == "CORTEX <no-reply@example.test>"
    assert "123456" in sent["body"]
    assert "123456" in sent["html"] and "The CORTEX Team" in sent["html"]
    assert sent["login"] == ("apikey", "secret")
    assert sent.get("starttls") is True


def test_db_direct_participant():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        db = Database(os.path.join(d, "x.db"))
        db.add_participant("c1", security.hash_password("p"), "lbl")
        assert db.get_participant("c1")["label"] == "lbl"
        db.set_participant_active("c1", False)
        assert db.get_participant("c1")["active"] == 0
        db.close()


def test_report_emails_receiver_with_diagnostics(client, monkeypatch):
    """A submitted report is emailed to the receiver with the message + the
    browser diagnostics, and Reply-To set to the reporter's (normalized) email."""
    from . import mailer as email_mod
    sent: dict = {}

    def fake_send(to, subject, body, reply_to=None):
        sent.update(to=to, subject=subject, body=body, reply_to=reply_to)

    monkeypatch.setattr(email_mod, "send_email", fake_send)
    r = client.post("/api/report", json={
        "username": "dr_test",
        "email": "Reporter@Example.test",
        "message": "Spectrogram looks off on Safari.",
        "client": {"os": "macOS", "timezone": "America/Los_Angeles", "userAgent": "UA/1.0"},
    })
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert sent["to"] == "elikeldsen@icloud.com"
    assert sent["reply_to"] == "reporter@example.test"
    assert "dr_test" in sent["subject"]
    assert "Spectrogram looks off on Safari." in sent["body"]
    assert "macOS" in sent["body"] and "America/Los_Angeles" in sent["body"]


def test_report_requires_message(client):
    r = client.post("/api/report",
                    json={"username": "x", "email": "x@y.com", "message": "   "})
    assert r.status_code == 400


# ── Sign in with Google ──────────────────────────────────────────────────────
def _google(monkeypatch, claims):
    """Configure GOOGLE_CLIENT_ID + stub the ID-token verifier to return claims."""
    from . import helpers
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setattr(helpers, "verify_google_credential", lambda cred, cid: claims)


def test_auth_google_creates_then_signs_in_returning_user(client, monkeypatch):
    _google(monkeypatch, {"sub": "g-123", "email": "NewUser@Gmail.com",
                          "email_verified": True, "name": "New User"})
    r = client.post("/api/auth/google", json={"credential": "tok"})
    assert r.status_code == 200
    b = r.json()
    assert b["email"] == "newuser@gmail.com" and b["displayName"] == "New User" and b["token"]
    # Same Google sub → same account, not a duplicate.
    r2 = client.post("/api/auth/google", json={"credential": "tok"})
    assert r2.status_code == 200 and r2.json()["code"] == b["code"]


def test_auth_google_links_existing_email_account(client, monkeypatch):
    email, pw = _make_participant(client)            # existing verified password account
    _google(monkeypatch, {"sub": "g-link", "email": email.upper(),
                          "email_verified": True, "name": "Linked"})
    r = client.post("/api/auth/google", json={"credential": "tok"})
    assert r.status_code == 200
    # The account is linked, not duplicated: password sign-in still works.
    assert client.post("/api/auth", json={"email": email, "password": pw}).status_code == 200


def test_auth_google_rejects_unverified_email(client, monkeypatch):
    _google(monkeypatch, {"sub": "g-x", "email": "x@y.com", "email_verified": False})
    assert client.post("/api/auth/google", json={"credential": "tok"}).status_code == 401


def test_auth_google_invalid_token_is_401(client, monkeypatch):
    from . import helpers
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    def boom(cred, cid):
        raise ValueError("bad audience / signature")
    monkeypatch.setattr(helpers, "verify_google_credential", boom)
    assert client.post("/api/auth/google", json={"credential": "tok"}).status_code == 401


def test_auth_google_unconfigured_is_503(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    assert client.post("/api/auth/google", json={"credential": "tok"}).status_code == 503


# ───────────── Phase O1: signup-expertise persistence + consent ledger ─────────────

def test_register_persists_signup_expertise(client):
    email = "expert@example.test"
    r = client.post("/api/register", json={
        "email": email, "password": "test-pw-1234567890",
        "displayName": "Dr Expert", "expertise": "epileptologist"})
    assert r.status_code == 200, r.text
    row = client.app.state.db.get_participant_by_email(email)
    assert row["signup_expertise"] == "epileptologist"


def test_register_without_expertise_is_backward_compatible(client):
    email, _pw, _code = _register(client)        # no expertise sent
    row = client.app.state.db.get_participant_by_email(email)
    assert row["signup_expertise"] is None


def test_consent_record_list_and_withdraw(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    r = client.post("/api/consent", headers=hdr,
                    json={"consentType": "research_irb", "consentVersion": "v1.0",
                          "irbProtocolId": "2016P000058"})
    assert r.status_code == 200 and r.json()["ok"] is True
    events = client.get("/api/consent", headers=hdr).json()["events"]
    assert len(events) == 1
    e = events[0]
    assert e["consent_version"] == "v1.0" and e["irb_protocol_id"] == "2016P000058"
    assert e["accepted_utc"] and e["withdrawn_utc"] is None
    w = client.post("/api/consent/withdraw", headers=hdr, json={})
    assert w.status_code == 200 and w.json()["withdrawn"] == 1
    assert client.get("/api/consent", headers=hdr).json()["events"][0]["withdrawn_utc"] is not None


def test_consent_requires_version_and_auth(client):
    assert client.post("/api/consent", json={"consentVersion": "v1.0"}).status_code == 401
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    assert client.post("/api/consent", headers=hdr, json={"consentVersion": "  "}).status_code == 400


def test_consent_reaffirm_clears_withdrawal(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    client.post("/api/consent", headers=hdr, json={"consentVersion": "v1.0"})
    client.post("/api/consent/withdraw", headers=hdr, json={})
    client.post("/api/consent", headers=hdr, json={"consentVersion": "v1.0"})   # re-affirm
    assert client.get("/api/consent", headers=hdr).json()["events"][0]["withdrawn_utc"] is None


# ───────────── Phase O2: learning-linkage FKs + is_real quarantine ─────────────

def test_trajectory_linkage_and_is_real_default_and_explicit(client):
    email, _pw = _make_participant(client)
    db = client.app.state.db
    code = db.get_participant_by_email(email)["code"]
    db.append_trajectory_points(code, [
        {"taskK": 0, "phase": "train", "ell": 0.9},                          # defaults
        {"taskK": 1, "phase": "train", "ell": 0.8, "isReal": True,
         "trainingId": "tr-1", "sourceSessionId": "s-1", "seqInSession": 5},
    ])
    rows = db._fetchall(
        "SELECT task_k, is_real, training_id, source_session_id, seq_in_session "
        "FROM param_trajectories WHERE code=? ORDER BY task_k", (code,))
    # default row is quarantined (is_real=0) with no linkage
    assert rows[0]["is_real"] == 0 and rows[0]["training_id"] is None
    # explicit real row carries the FKs + ordering
    assert rows[1]["is_real"] == 1 and rows[1]["training_id"] == "tr-1"
    assert rows[1]["source_session_id"] == "s-1" and rows[1]["seq_in_session"] == 5


def test_training_session_links_active_regimen(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    db = client.app.state.db
    code = db.get_participant_by_email(email)["code"]
    db.create_regimen("rg-9", code, "src-sess-1", {"weeks": 6, "deck": []})
    tid = client.post("/api/training-sessions", headers=hdr,
                      json={"taskFocus": "gpd"}).json()["trainingId"]
    row = db._fetchall("SELECT regimen_id, source_session_id FROM training_sessions "
                       "WHERE training_id=?", (tid,))[0]
    assert row["regimen_id"] == "rg-9" and row["source_session_id"] == "src-sess-1"


def test_pg_branch_drives_the_pool(monkeypatch):
    """The Postgres branch is one pooled-connection unit of work per
    operation (checkout → execute → return-to-pool). The resilience the old
    shared-single-connection code hand-rolled — rollback-on-error (the
    2026-06-21 outage), commit-after-read so nothing lingers idle-in-
    transaction (06-22), reconnect after a severed connection (06-24) — is
    psycopg_pool's contract now: connection() commits/rolls back on exit and
    `check` replaces dead connections at checkout. CI has no Postgres, so
    drive the branch with a fake pool and assert the unit-of-work shape."""
    from contextlib import contextmanager
    from . import db as db_module

    class _Cur:
        rowcount = 1
        def fetchone(self): return None
        def fetchall(self): return []

    class _Conn:
        def __init__(self, log): self._log = log
        def execute(self, sql, params=()):
            self._log.append(sql)
            return _Cur()

    class _Pool:
        def __init__(self):
            self.checkouts, self.closed, self.sql = 0, False, []
            self._conn = _Conn(self.sql)
        @contextmanager
        def connection(self):
            self.checkouts += 1
            yield self._conn
        def close(self): self.closed = True

    pool = _Pool()
    monkeypatch.setattr(db_module.Database, "_make_pool", lambda self: pool)
    db = Database("postgresql://u:pw@localhost/x")
    boot = pool.checkouts
    assert boot >= 1                                  # schema+migrations ran via the pool
    assert any("CREATE TABLE" in s for s in pool.sql)
    assert db.get_participant("nobody") is None       # read = one checkout
    assert pool.checkouts == boot + 1
    assert pool.sql[-1] == "SELECT * FROM participants WHERE code=%s"  # ?→%s translated
    db.ping()                                         # deep-health probe = one checkout
    assert pool.checkouts == boot + 2
    db.close()
    assert pool.closed


def test_failed_statement_does_not_poison_later_ones(tmp_path):
    """The incident class behind the old hand-rolled recovery machinery,
    asserted at the new boundary: a statement that errors must leave the
    Database usable (its unit of work rolled back), never poison later
    calls. On Postgres this is psycopg_pool's rollback-on-exception; the
    SQLite branch mirrors it in _connection()."""
    db = Database(tmp_path / "poison.db")
    db.add_participant("dup", "hash-a")
    with pytest.raises(Exception):
        db.add_participant("dup", "hash-b")   # PK violation → rolled back
    db.add_participant("ok", "hash-c")        # must still work afterwards
    assert db.get_participant("ok") is not None
    assert db.get_participant("dup")["password_hash"] == "hash-a"
    db.close()


# ───────────────────── signup profile + Settings ─────────────────────

def test_signup_stores_profile_and_get_profile(client):
    """Demographics collected at signup persist on the account and read back
    via GET /api/profile (no pre-tutorial re-ask)."""
    _REG_COUNTER[0] += 1
    email = f"p{_REG_COUNTER[0]}@example.test"
    pw = "test-pw-1234567890"
    prof = {"institution": "Stanford", "years_reading_eeg": "3-5",
            "sex": "Female", "age": "34", "location": "Palo Alto, CA",
            "bogus_field": "should be dropped"}
    r = client.post("/api/register", json={
        "email": email, "password": pw, "displayName": "Prof Test",
        "expertise": "Neurology resident", "profile": prof})
    assert r.status_code == 200, r.text
    code = r.json()["devCode"]
    client.post("/api/verify/confirm", json={"email": email, "code": code})
    hdr = _auth_header(client, email, pw)
    got = client.get("/api/profile", headers=hdr).json()
    assert got["expertise"] == "Neurology resident"
    assert got["displayName"] == "Prof Test"
    assert got["profile"]["institution"] == "Stanford"
    assert got["profile"]["age"] == "34"
    assert got["profile"]["location"] == "Palo Alto, CA"
    assert "bogus_field" not in got["profile"]   # whitelist drops unknown keys


def test_update_profile(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    r = client.put("/api/profile", headers=hdr, json={
        "displayName": "New Name", "expertise": "EEG technologist",
        "profile": {"institution": "MGH", "age": "40"}})
    assert r.status_code == 200, r.text
    got = client.get("/api/profile", headers=hdr).json()
    assert got["displayName"] == "New Name"
    assert got["expertise"] == "EEG technologist"
    assert got["profile"]["institution"] == "MGH"


def test_change_password(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    # wrong current password → 403
    assert client.post("/api/account/password", headers=hdr,
                       json={"currentPassword": "nope", "newPassword": "brand-new-pw-123"}).status_code == 403
    # correct → 200, old pw stops working, new pw works
    assert client.post("/api/account/password", headers=hdr,
                       json={"currentPassword": pw, "newPassword": "brand-new-pw-123"}).status_code == 200
    assert client.post("/api/auth", json={"email": email, "password": pw}).status_code == 401
    assert client.post("/api/auth", json={"email": email, "password": "brand-new-pw-123"}).status_code == 200


def test_change_email(client):
    email, pw = _make_participant(client)
    other_email, _ = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    # wrong password → 403
    assert client.post("/api/account/email", headers=hdr,
                       json={"newEmail": "fresh@example.test", "password": "nope"}).status_code == 403
    # collision with an existing account → 409
    assert client.post("/api/account/email", headers=hdr,
                       json={"newEmail": other_email, "password": pw}).status_code == 409
    # valid change → 200; can auth with the new email
    assert client.post("/api/account/email", headers=hdr,
                       json={"newEmail": "fresh@example.test", "password": pw}).status_code == 200
    assert client.post("/api/auth", json={"email": "fresh@example.test", "password": pw}).status_code == 200


# ─────────────── public 9-digit user ids ───────────────

_PID_RE = r"[1-9]\d{8}"   # exactly 9 digits, no leading zero


def test_public_id_assigned_at_signup(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    got = client.get("/api/profile", headers=hdr).json()
    assert re.fullmatch(_PID_RE, got["publicId"]), got
    # stable across reads
    assert client.get("/api/profile", headers=hdr).json()["publicId"] == got["publicId"]


def test_public_ids_unique_across_accounts(client):
    pids = set()
    for _ in range(5):
        email, pw = _make_participant(client)
        hdr = _auth_header(client, email, pw)
        pids.add(client.get("/api/profile", headers=hdr).json()["publicId"])
    assert len(pids) == 5


def test_public_id_backfilled_at_boot(client, tmp_path):
    """Grandfathering: an account created before the public_id column gets an
    id assigned by the boot backfill (simulated by clearing the column and
    re-opening the database, as a deploy restart would)."""
    email, _pw = _make_participant(client)
    db = client.app.state.db
    db._write("UPDATE participants SET public_id=NULL WHERE email=?", (email,))
    assert db.get_participant_by_email(email)["public_id"] is None
    db2 = Database(tmp_path / "t.db")
    try:
        pid = db2.get_participant_by_email(email)["public_id"]
        assert pid and re.fullmatch(_PID_RE, pid)
    finally:
        db2.close()


def test_public_id_lazy_assign_on_profile_read(client):
    """A row that somehow missed both creation-time assignment and the boot
    backfill is repaired on the next GET /api/profile."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    db = client.app.state.db
    db._write("UPDATE participants SET public_id=NULL WHERE email=?", (email,))
    pid = client.get("/api/profile", headers=hdr).json()["publicId"]
    assert re.fullmatch(_PID_RE, pid)
    assert client.get("/api/profile", headers=hdr).json()["publicId"] == pid


def test_public_id_collision_retries(client, monkeypatch):
    """If the random draw repeats an existing id, the unique index rejects it
    and the allocator retries with a fresh draw (email 409s still propagate)."""
    from . import db as db_module
    email1, _pw = _make_participant(client)
    db = client.app.state.db
    taken = db.get_participant_by_email(email1)["public_id"]
    draws = iter([taken, taken, "314159265"])
    monkeypatch.setattr(db_module, "_random_public_id", lambda: next(draws))
    email2, _pw2 = _make_participant(client)
    assert db.get_participant_by_email(email2)["public_id"] == "314159265"


def test_admin_seeded_accounts_get_public_ids(client):
    r = client.post("/api/admin/participants", json={"prefix": "seed", "count": 2},
                    headers={"X-Admin-Token": "test-admin"})
    assert r.status_code == 200, r.text
    db = client.app.state.db
    pids = {db.get_participant(a["code"])["public_id"] for a in r.json()}
    assert len(pids) == 2
    assert all(re.fullmatch(_PID_RE, p) for p in pids)


# ─────────────── cohorts ───────────────

def _pid_of(client, hdr) -> str:
    return client.get("/api/profile", headers=hdr).json()["publicId"]


def _make_cohort(client, name="Unit A"):
    """A manager account + their cohort. Return (cohort_id, manager_hdr)."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    r = client.post("/api/cohorts", headers=hdr, json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["cohortId"], hdr


def _join_cohort(client, cid, manager_hdr):
    """A fresh account invited by the manager + accepted. Return (hdr, pid)."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    pid = _pid_of(client, hdr)
    r = client.post(f"/api/cohorts/{cid}/invite", headers=manager_hdr,
                    json={"publicId": pid})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/cohorts/{cid}/accept", headers=hdr).status_code == 200
    return hdr, pid


def test_cohort_create_and_list(client):
    cid, mh = _make_cohort(client, name="  ICU Team ")
    lst = client.get("/api/cohorts", headers=mh).json()["cohorts"]
    assert len(lst) == 1
    c = lst[0]
    assert c["cohortId"] == cid and c["name"] == "ICU Team"
    assert c["role"] == "manager" and c["status"] == "active"
    assert c["memberCount"] == 1   # the manager is an active member


def test_cohort_invite_accept_flow(client):
    cid, mh = _make_cohort(client)
    email, pw = _make_participant(client)
    uh = _auth_header(client, email, pw)
    pid = _pid_of(client, uh)
    assert client.post(f"/api/cohorts/{cid}/invite", headers=mh,
                       json={"publicId": pid}).status_code == 200
    # Pending invitee: sees the cohort (name only) but no members/performance.
    mine = client.get("/api/cohorts", headers=uh).json()["cohorts"]
    assert mine[0]["status"] == "invited" and mine[0]["role"] == "member"
    det = client.get(f"/api/cohorts/{cid}", headers=uh).json()
    assert det["name"] == "Unit A" and "members" not in det
    assert client.get(f"/api/cohorts/{cid}/performance", headers=uh).status_code == 403
    # Accept → active member with full (public-id-keyed) visibility.
    assert client.post(f"/api/cohorts/{cid}/accept", headers=uh).status_code == 200
    det = client.get(f"/api/cohorts/{cid}", headers=uh).json()
    assert {m["publicId"] for m in det["members"]} >= {pid}
    assert client.get(f"/api/cohorts/{cid}/performance", headers=uh).status_code == 200
    # Double-accept is a clean 409, not a state change.
    assert client.post(f"/api/cohorts/{cid}/accept", headers=uh).status_code == 409


def test_cohort_id_invite_sends_notification(client, monkeypatch):
    from . import mailer
    sent = []
    monkeypatch.setattr(mailer, "send_letter",
                        lambda to, subject, title, paragraphs, button=None:
                        sent.append((to, subject, button)))
    cid, mh = _make_cohort(client)
    email, pw = _make_participant(client)
    uh = _auth_header(client, email, pw)
    pid = _pid_of(client, uh)
    assert client.post(f"/api/cohorts/{cid}/invite", headers=mh,
                       json={"publicId": pid}).status_code == 200
    assert sent and sent[0][0] == email
    assert "cohort invitation" in sent[0][1]
    # the button deep-links to the invite, not the front door
    assert sent[0][2] is not None and f"/?cohort={cid}" in sent[0][2][1]


def test_cohort_invites_expire_after_30_days(client):
    cid, mh = _make_cohort(client)
    member, mpw = _make_participant(client)
    uh = _auth_header(client, member, mpw)
    pid = _pid_of(client, uh)
    client.post(f"/api/cohorts/{cid}/invite", headers=mh, json={"publicId": pid})
    client.post(f"/api/cohorts/{cid}/invite-email", headers=mh,
                json={"email": "old@example.test"})
    db = client.app.state.db
    db._write("UPDATE cohort_members SET invited_utc='2026-05-01T00:00:00Z' "
              "WHERE status='invited'", ())
    db._write("UPDATE cohort_email_invites SET invited_utc='2026-05-01T00:00:00Z'", ())
    # stale invites vanish from every view; ACTIVE memberships (here the
    # manager's own row) are untouched
    det = client.get(f"/api/cohorts/{cid}", headers=mh).json()
    assert det["emailInvites"] == []
    assert [m["status"] for m in det.get("members", [])] == ["active"]
    assert client.get("/api/cohorts", headers=uh).json()["cohorts"] == []
    # and an expired email invite never attaches at signup
    r = client.post("/api/register", json={"email": "old@example.test",
                    "password": "old-pw-123456789012", "displayName": "O"})
    client.post("/api/verify/confirm",
                json={"email": "old@example.test", "code": r.json()["devCode"]})
    oh = _auth_header(client, "old@example.test", "old-pw-123456789012")
    assert client.get("/api/cohorts", headers=oh).json()["cohorts"] == []


def test_cohort_invite_daily_account_quota(client, monkeypatch):
    from . import deps
    monkeypatch.setitem(deps.RATE_LIMITS, "cohort_invite_account", (2, 86400))
    cid, mh = _make_cohort(client)
    member, mpw = _make_participant(client)
    pid = _pid_of(client, _auth_header(client, member, mpw))
    # the id and email paths share ONE per-account daily bucket
    assert client.post(f"/api/cohorts/{cid}/invite", headers=mh,
                       json={"publicId": pid}).status_code == 200
    assert client.post(f"/api/cohorts/{cid}/invite-email", headers=mh,
                       json={"email": "a@example.test"}).status_code == 200
    r = client.post(f"/api/cohorts/{cid}/invite-email", headers=mh,
                    json={"email": "b@example.test"})
    assert r.status_code == 429
    assert "daily invitation limit" in r.json()["error"]


def test_cohort_email_invite_existing_account(client):
    cid, mh = _make_cohort(client)
    invitee, pw = _make_participant(client)
    assert client.post(f"/api/cohorts/{cid}/invite-email", headers=mh,
                       json={"email": invitee}).json() == {"ok": True}
    # the invitee got a NORMAL pending invite: same consent gate as the id flow
    uh = _auth_header(client, invitee, pw)
    mine = client.get("/api/cohorts", headers=uh).json()["cohorts"]
    assert mine[0]["status"] == "invited"
    assert client.post(f"/api/cohorts/{cid}/accept", headers=uh).status_code == 200
    # re-inviting a member reveals nothing (anti-oracle: identical response)
    assert client.post(f"/api/cohorts/{cid}/invite-email", headers=mh,
                       json={"email": invitee}).json() == {"ok": True}


def test_cohort_email_invite_unknown_address_attaches_at_signup(client):
    cid, mh = _make_cohort(client)
    email = "future-member@example.test"
    # identical ok for an address with no account (anti-oracle)…
    assert client.post(f"/api/cohorts/{cid}/invite-email", headers=mh,
                       json={"email": email}).json() == {"ok": True}
    # …tracked as the manager's email-invite bookkeeping
    det = client.get(f"/api/cohorts/{cid}", headers=mh).json()
    assert [e["email"] for e in det["emailInvites"]] == [email]
    # the address signs up + verifies → converts to a pending membership
    r = client.post("/api/register", json={
        "email": email, "password": "future-pw-123456789", "displayName": "F"})
    assert client.post("/api/verify/confirm", json={
        "email": email, "code": r.json()["devCode"]}).status_code == 200
    det = client.get(f"/api/cohorts/{cid}", headers=mh).json()
    assert det["emailInvites"] == []
    uh = _auth_header(client, email, "future-pw-123456789")
    mine = client.get("/api/cohorts", headers=uh).json()["cohorts"]
    assert mine and mine[0]["status"] == "invited"   # accept still required
    assert client.post(f"/api/cohorts/{cid}/accept", headers=uh).status_code == 200


def test_cohort_email_invite_cancel_and_manager_gate(client):
    cid, mh = _make_cohort(client)
    email = "withdrawn@example.test"
    client.post(f"/api/cohorts/{cid}/invite-email", headers=mh, json={"email": email})
    assert client.post(f"/api/cohorts/{cid}/invite-email/cancel", headers=mh,
                       json={"email": email}).json() == {"ok": True}
    assert client.get(f"/api/cohorts/{cid}", headers=mh).json()["emailInvites"] == []
    # a canceled invite must NOT attach at signup
    r = client.post("/api/register", json={
        "email": email, "password": "withdrawn-pw-1234567", "displayName": "W"})
    client.post("/api/verify/confirm", json={"email": email, "code": r.json()["devCode"]})
    uh = _auth_header(client, email, "withdrawn-pw-1234567")
    assert client.get("/api/cohorts", headers=uh).json()["cohorts"] == []
    # non-managers can't reach the email-invite surface (404, not 403:
    # cohort existence stays unconfirmed for outsiders)
    assert client.post(f"/api/cohorts/{cid}/invite-email", headers=uh,
                       json={"email": "x@example.test"}).status_code == 404


def test_cohort_names_manager_only_and_no_internal_ids(client):
    cid, mh = _make_cohort(client)
    uh, _pid = _join_cohort(client, cid, mh)
    # Manager sees display names; a member never does.
    mgr_det = client.get(f"/api/cohorts/{cid}", headers=mh)
    assert all("displayName" in m for m in mgr_det.json()["members"])
    mem_det = client.get(f"/api/cohorts/{cid}", headers=uh)
    assert all("displayName" not in m for m in mem_det.json()["members"])
    mem_perf = client.get(f"/api/cohorts/{cid}/performance", headers=uh)
    assert all("displayName" not in m for m in mem_perf.json()["members"])
    mgr_perf = client.get(f"/api/cohorts/{cid}/performance", headers=mh)
    assert all("displayName" in m for m in mgr_perf.json()["members"])
    # The internal participant code / email must never cross the cohort API.
    for r in (mgr_det, mem_det, mem_perf, mgr_perf):
        assert '"u-' not in r.text and "@example.test" not in r.text


def test_cohort_outsider_cannot_probe(client):
    cid, _mh = _make_cohort(client)
    email, pw = _make_participant(client)
    oh = _auth_header(client, email, pw)
    # Existence is not confirmed to outsiders: 404 (not 403) everywhere.
    assert client.get(f"/api/cohorts/{cid}", headers=oh).status_code == 404
    assert client.get(f"/api/cohorts/{cid}/performance", headers=oh).status_code == 404
    assert client.post(f"/api/cohorts/{cid}/invite", headers=oh,
                       json={"publicId": "123456789"}).status_code == 404
    assert client.delete(f"/api/cohorts/{cid}", headers=oh).status_code == 404
    assert client.get("/api/cohorts/ch-nonexistent", headers=oh).status_code == 404


def test_cohort_member_cannot_manage(client):
    cid, mh = _make_cohort(client)
    uh, _pid = _join_cohort(client, cid, mh)
    other_email, other_pw = _make_participant(client)
    oh = _auth_header(client, other_email, other_pw)
    opid = _pid_of(client, oh)
    # A plain member is known to the cohort → 403 (not 404) on manager ops.
    assert client.post(f"/api/cohorts/{cid}/invite", headers=uh,
                       json={"publicId": opid}).status_code == 403
    assert client.post(f"/api/cohorts/{cid}/remove", headers=uh,
                       json={"publicId": opid}).status_code == 403
    assert client.delete(f"/api/cohorts/{cid}", headers=uh).status_code == 403


def test_cohort_invite_errors(client):
    cid, mh = _make_cohort(client)
    uh, pid = _join_cohort(client, cid, mh)
    assert client.post(f"/api/cohorts/{cid}/invite", headers=mh,
                       json={"publicId": "12345"}).status_code == 400
    assert client.post(f"/api/cohorts/{cid}/invite", headers=mh,
                       json={"publicId": "987654321"}).status_code == 404
    # Re-inviting an existing member → 409 (also covers inviting yourself).
    assert client.post(f"/api/cohorts/{cid}/invite", headers=mh,
                       json={"publicId": pid}).status_code == 409


def test_cohort_decline_leave_remove_delete(client):
    cid, mh = _make_cohort(client)
    # decline: invited user says no → row gone.
    email, pw = _make_participant(client)
    dh = _auth_header(client, email, pw)
    dpid = _pid_of(client, dh)
    client.post(f"/api/cohorts/{cid}/invite", headers=mh, json={"publicId": dpid})
    assert client.post(f"/api/cohorts/{cid}/decline", headers=dh).status_code == 200
    assert client.get("/api/cohorts", headers=dh).json()["cohorts"] == []
    # leave: active member leaves → loses access.
    lh, _lpid = _join_cohort(client, cid, mh)
    assert client.post(f"/api/cohorts/{cid}/leave", headers=lh).status_code == 200
    assert client.get(f"/api/cohorts/{cid}", headers=lh).status_code == 404
    # the manager can neither leave nor be removed
    mpid = _pid_of(client, mh)
    assert client.post(f"/api/cohorts/{cid}/leave", headers=mh).status_code == 400
    assert client.post(f"/api/cohorts/{cid}/remove", headers=mh,
                       json={"publicId": mpid}).status_code == 400
    # remove: manager removes an active member by public id.
    rh, rpid = _join_cohort(client, cid, mh)
    assert client.post(f"/api/cohorts/{cid}/remove", headers=mh,
                       json={"publicId": rpid}).status_code == 200
    assert client.get(f"/api/cohorts/{cid}", headers=rh).status_code == 404
    # delete: the pod disappears for everyone.
    xh, _xpid = _join_cohort(client, cid, mh)
    assert client.delete(f"/api/cohorts/{cid}", headers=mh).status_code == 200
    assert client.get(f"/api/cohorts/{cid}", headers=mh).status_code == 404
    assert client.get("/api/cohorts", headers=xh).json()["cohorts"] == []


def test_cohort_multi_membership(client):
    cid1, mh1 = _make_cohort(client, name="Pod One")
    cid2, mh2 = _make_cohort(client, name="Pod Two")
    email, pw = _make_participant(client)
    uh = _auth_header(client, email, pw)
    pid = _pid_of(client, uh)
    for cid, mh in ((cid1, mh1), (cid2, mh2)):
        client.post(f"/api/cohorts/{cid}/invite", headers=mh, json={"publicId": pid})
        client.post(f"/api/cohorts/{cid}/accept", headers=uh)
    mine = client.get("/api/cohorts", headers=uh).json()["cohorts"]
    assert {c["name"] for c in mine} == {"Pod One", "Pod Two"}
    assert all(c["status"] == "active" for c in mine)


def test_cohort_performance_window_and_quarantine(client):
    """The series contains ONLY real (is_real=1) points inside the 6-month
    window: recent eval points appear with skill=ℓ / bias=t / sd; stale and
    synthetic points never reach cohort peers; a data-less member still shows
    up (empty series) so the pod legend is complete."""
    cid, mh = _make_cohort(client)
    uh, upid = _join_cohort(client, cid, mh)
    db = client.app.state.db
    ucode = None
    for m in db.cohort_member_rows(cid):
        if m["public_id"] == upid:
            ucode = m["code"]
    # recent real eval point (what a finished certification test writes)
    db.write_eval_trajectory(ucode, "sess-recent", [
        {"taskK": 0, "ell": 1.25, "theta": -0.4, "sd": 0.3, "rt": 900.0},
        {"taskK": 3, "ell": 0.75, "theta": 0.2, "sd": 0.5, "rt": 1100.0},
    ])
    # stale real point (outside the 183-day lookback) + recent synthetic point
    db.append_trajectory_points(ucode, [
        {"taskK": 0, "phase": "eval", "ell": 9.0, "theta": 9.0, "sd": 0.1,
         "ts": "2025-01-01T00:00:00Z", "isReal": 1},
        {"taskK": 0, "phase": "train", "ell": 8.0, "theta": 8.0, "sd": 0.1,
         "isReal": 0},
    ])
    perf = client.get(f"/api/cohorts/{cid}/performance", headers=uh).json()
    by_pid = {m["publicId"]: m for m in perf["members"]}
    me = by_pid[upid]
    assert me["isYou"] is True
    series = {s["taskK"]: s["points"] for s in me["series"]}
    assert set(series) == {0, 3}
    assert len(series[0]) == 1   # stale + synthetic points filtered out
    assert series[0][0]["skill"] == 1.25 and series[0][0]["bias"] == -0.4
    assert series[0][0]["sd"] == 0.3 and series[0][0]["phase"] == "eval"
    assert series[3][0]["skill"] == 0.75
    # the manager has no data yet but still appears, with an empty series
    mgr = next(m for m in perf["members"] if m["isManager"])
    assert mgr["series"] == []
    assert perf["from"] < perf["to"]


def test_cohort_performance_timeframe_and_daily_dedup(client):
    """`days` selects the lookback window; only the LATEST point per task per
    UTC day is returned (a busy day — e.g. a training session's many trials —
    collapses to one point); days<=0 = all time (axis `from` = earliest point)."""
    cid, mh = _make_cohort(client)
    uh, upid = _join_cohort(client, cid, mh)
    db = client.app.state.db
    ucode = next(m["code"] for m in db.cohort_member_rows(cid)
                 if m["public_id"] == upid)
    # three real points on task 0 the SAME UTC day (latest ts must win) + one
    # on the next day — all ~5 weeks old so a 1-week window excludes them.
    db.append_trajectory_points(ucode, [
        {"taskK": 0, "phase": "eval",  "ell": 1.0, "theta": 0.1, "sd": 0.3,
         "ts": "2026-06-01T08:00:00Z", "isReal": 1},
        {"taskK": 0, "phase": "train", "ell": 1.1, "theta": 0.2, "sd": 0.2,
         "ts": "2026-06-01T09:00:00Z", "isReal": 1},
        {"taskK": 0, "phase": "train", "ell": 1.4, "theta": 0.3, "sd": 0.1,
         "ts": "2026-06-01T18:30:00Z", "isReal": 1},   # latest that day → wins
        {"taskK": 0, "phase": "eval",  "ell": 0.7, "theta": 0.0, "sd": 0.3,
         "ts": "2026-06-02T10:00:00Z", "isReal": 1},   # separate day
    ])
    # all time: one point per calendar day, latest-of-day survives
    allp = client.get(f"/api/cohorts/{cid}/performance?days=0", headers=uh).json()
    me = next(m for m in allp["members"] if m["isYou"])
    pts = {s["taskK"]: s["points"] for s in me["series"]}[0]
    assert len(pts) == 2
    assert pts[0]["skill"] == 1.4 and pts[0]["phase"] == "train"   # 06-01 latest
    assert pts[1]["skill"] == 0.7                                  # 06-02
    assert allp["from"] <= "2026-06-01T08:00:00Z"                  # axis fits data
    # a 1-week window excludes the ~5-week-old points entirely
    wk = client.get(f"/api/cohorts/{cid}/performance?days=7", headers=uh).json()
    assert next(m for m in wk["members"] if m["isYou"])["series"] == []


# ─────────────── group-1 hardening (2026-07-01 audit) ───────────────

def test_health_deep_checks_db_and_reports_bank(client, monkeypatch):
    # Shallow probe: static liveness (never touches the DB).
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert "release" in r.json()
    # Deep probe: DB round-trip + bank state.
    r = client.get("/api/health", params={"deep": 1})
    assert r.status_code == 200
    assert r.json()["db"] == "ok"
    assert r.json()["bank"] in ("not_loaded_yet", "loaded", "missing")
    # DB down → deep probe 503s (this is what an uptime monitor must see),
    # while the shallow probe stays 200.
    db = client.app.state.db

    def _dead():
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "ping", _dead)
    r = client.get("/api/health", params={"deep": 1})
    assert r.status_code == 503 and r.json()["ok"] is False
    assert r.json()["db"].startswith("error:")
    assert client.get("/api/health").status_code == 200


def test_put_profile_partial_update_preserves_profile(client):
    """Omitting `profile` in a PUT must NOT wipe the stored profile (the old
    `{}` default silently did)."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    assert client.put("/api/profile", headers=hdr,
                      json={"displayName": "Name One",
                            "profile": {"institution": "Stanford"}}).status_code == 200
    # Partial update: displayName only, `profile` omitted entirely.
    assert client.put("/api/profile", headers=hdr,
                      json={"displayName": "Name Two"}).status_code == 200
    prof = client.get("/api/profile", headers=hdr).json()
    assert prof["displayName"] == "Name Two"
    assert prof["profile"].get("institution") == "Stanford"


def test_register_survives_email_send_failure(client, monkeypatch):
    """An SMTP/SES outage during signup must not strand the account: the
    account row exists, so a 500 would make every retry 409. Instead the
    signup succeeds (no devCode) and the resend path recovers."""
    from . import mailer as email_mod
    real_send = email_mod.send_auth_code

    def boom(*a, **k):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(email_mod, "send_auth_code", boom)
    r = client.post("/api/register",
                    json={"email": "mailout@example.test",
                          "password": "test-pw-1234567890",
                          "displayName": "Outage User"})
    assert r.status_code == 200, r.text
    assert r.json()["needsVerification"] is True
    assert "devCode" not in r.json()
    # Email service back up → resend issues a fresh code; verification works.
    monkeypatch.setattr(email_mod, "send_auth_code", real_send)
    r = client.post("/api/verify/resend", json={"email": "mailout@example.test"})
    assert r.status_code == 200 and r.json().get("devCode")
    r = client.post("/api/verify/confirm",
                    json={"email": "mailout@example.test", "code": r.json()["devCode"]})
    assert r.status_code == 200


def test_session_persists_drawn_seg_ids(client):
    """The server-drawn candidate pool is stamped onto the session row —
    the seed alone can't reproduce it later (exposure exclusion is temporal)."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    r = client.post("/api/session", json={"participant": {}}, headers=hdr)
    assert r.status_code == 200, r.text
    served = [s["segId"] for s in r.json()["bank"]["segments"]]
    sess = client.app.state.db.get_session(r.json()["sessionId"])
    assert sess["bundle_version"] == "test-bank"
    stored = json.loads(sess["drawn_seg_ids"])
    assert stored == served and len(stored) > 0


def test_truth_map_reuses_loaded_bank(tmp_path, monkeypatch):
    """When the session's bundle IS the loaded bank, the truth map comes from
    the in-memory segments — no manifest re-read from disk."""
    from . import config, dashboard_logic
    from .session_bank import SessionBank
    _write_test_bank(tmp_path / "b", "bank-x")
    bank = SessionBank(tmp_path / "b" / "bank-x" / "manifest.json", "/bundle/bank-x")
    # Point the disk fallback somewhere empty: only the bank can answer.
    monkeypatch.setattr(config, "BUNDLE_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(dashboard_logic, "_TRUTH_CACHE", {})
    truth = dashboard_logic._truth_map_for("bank-x", bank)
    assert truth["seg"][0] == "spike"
    assert truth["words"] and truth["labels"] and truth["classes"]
    # A different version must NOT be served from this bank.
    assert dashboard_logic._truth_map_for("other-version", bank)["seg"] == {}


def test_videos_rejects_oversized_shape_and_mismatched_blobs(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    files = {"t": ("t.bin", b"\0" * 16), "l": ("l.bin", b"\0" * 16),
             "w": ("w.bin", b"\0" * 16)}
    # Forged huge shape → 413 before any allocation.
    meta = json.dumps({"shape": [99999, 99999, 99], "taskCodes": [], "segIds": [],
                       "trials": [], "certificate": {}})
    r = client.post("/api/videos", headers=hdr, data={"meta": meta}, files=files)
    assert r.status_code == 413
    # Plausible shape but blobs that don't match it → 400.
    meta = json.dumps({"shape": [2, 3, 4], "taskCodes": [], "segIds": [],
                       "trials": [], "certificate": {}})
    r = client.post("/api/videos", headers=hdr, data={"meta": meta}, files=files)
    assert r.status_code == 400


def test_rate_limiter_sweeps_stale_ip_buckets(monkeypatch):
    from . import deps
    lim = deps.RateLimiter()
    t = [1_000_000.0]
    monkeypatch.setattr(deps.time, "time", lambda: t[0])
    assert lim.hit("register", "1.2.3.4")
    assert ("register", "1.2.3.4") in lim._hits
    t[0] += 3601.0                       # register window fully expired
    lim._n = lim._SWEEP_EVERY - 1        # next hit triggers the sweep
    assert lim.hit("register", "5.6.7.8")
    assert ("register", "1.2.3.4") not in lim._hits
    assert ("register", "5.6.7.8") in lim._hits


# ─────────────── videos job queue (2026-07-01) ───────────────

def _viz_files(T=2, N=3, K=4):
    """Correctly-sized dummy blobs + meta for a (T,N,K) trajectory."""
    files = {"t": ("t.bin", b"\0" * (T * N * K * 4)),
             "l": ("l.bin", b"\0" * (T * N * K * 4)),
             "w": ("w.bin", b"\0" * (T * N * 4))}
    meta = {"shape": [T, N, K], "taskCodes": ["spike"], "segIds": [1, 2],
            "trials": [{"q": 1}], "certificate": {"v": 1}}
    return files, json.dumps(meta)


def _poll_job(client, hdr, jid, tries=200, wait=0.05):
    for _ in range(tries):
        st = client.get(f"/api/videos/{jid}", headers=hdr).json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(wait)
    return st


def test_videos_job_flow(client, monkeypatch):
    """Submit → 202 {jobId}; poll to done; download the zip; ownership and
    unknown-job lookups 404. The heavy renderer is stubbed — the job plumbing
    (thread, statuses, registry, files) is what's under test."""
    import zipfile
    from .routers import videos as videos_router

    def fake_render(sd):
        z = sd / "cortex_visualizations.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("summary.mp4", b"fake-mp4-bytes")
        return z

    monkeypatch.setattr(videos_router, "_execute_render", fake_render)
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    files, meta = _viz_files()
    r = client.post("/api/videos", headers=hdr, data={"meta": meta}, files=files)
    assert r.status_code == 202, r.text
    jid = r.json()["jobId"]

    st = _poll_job(client, hdr, jid)
    assert st["status"] == "done", st

    dl = client.get(f"/api/videos/{jid}/download", headers=hdr)
    assert dl.status_code == 200
    assert dl.content[:2] == b"PK"          # a real zip
    # Re-download works (artifacts persist until the TTL sweep).
    assert client.get(f"/api/videos/{jid}/download", headers=hdr).status_code == 200

    # Another participant must not see the job at all.
    email2, pw2 = _make_participant(client)
    hdr2 = _auth_header(client, email2, pw2)
    assert client.get(f"/api/videos/{jid}", headers=hdr2).status_code == 404
    assert client.get(f"/api/videos/{jid}/download", headers=hdr2).status_code == 404
    assert client.get("/api/videos/no-such-job", headers=hdr).status_code == 404


def test_videos_job_error_surfaces(client, monkeypatch):
    from .routers import videos as videos_router

    def broken_render(sd):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(videos_router, "_execute_render", broken_render)
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    files, meta = _viz_files()
    r = client.post("/api/videos", headers=hdr, data={"meta": meta}, files=files)
    assert r.status_code == 202
    st = _poll_job(client, hdr, r.json()["jobId"])
    assert st["status"] == "error"
    assert "ffmpeg exploded" in st.get("error", "")
    # Download before/without success is a clean 409, not a 500.
    assert client.get(f"/api/videos/{r.json()['jobId']}/download", headers=hdr).status_code == 409


def test_videos_queue_cap(client, monkeypatch):
    """One render runs, one may queue; a third submission gets an honest 503.
    The render is gated on an Event so the test is deterministic."""
    import threading as _threading
    import zipfile
    from .routers import videos as videos_router

    gate = _threading.Event()

    def slow_render(sd):
        assert gate.wait(timeout=15), "test gate never opened"
        z = sd / "cortex_visualizations.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("a.mp4", b"x")
        return z

    monkeypatch.setattr(videos_router, "_execute_render", slow_render)
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    files, meta = _viz_files()

    r1 = client.post("/api/videos", headers=hdr, data={"meta": meta}, files=files)
    r2 = client.post("/api/videos", headers=hdr, data={"meta": meta}, files=files)
    assert r1.status_code == 202 and r2.status_code == 202
    r3 = client.post("/api/videos", headers=hdr, data={"meta": meta}, files=files)
    assert r3.status_code == 503                     # queue full

    gate.set()                                       # let both renders finish
    assert _poll_job(client, hdr, r1.json()["jobId"])["status"] == "done"
    assert _poll_job(client, hdr, r2.json()["jobId"])["status"] == "done"


# ─────────────── listing-payload slimming (2026-07-01) ───────────────

def _finish_session_with_fat_result(client, hdr):
    """Start a session and post a result carrying the heavy raw keys."""
    r = client.post("/api/session", json={"participant": {}}, headers=hdr)
    assert r.status_code == 200, r.text
    sid = r.json()["sessionId"]
    fat_result = {
        "verdicts": ["PASS"] * 7,
        "roc": [{"auroc": 0.9, "hw": 0.05}] * 7,
        "perTask": [{"taskK": k, "ell": 0.1, "theta": 0.0, "sd": 0.2,
                     "ellStar": 0.0, "auroc": 0.9, "verdict": "PASS"} for k in range(7)],
        "trials": [{"trialIndex": i, "segId": i, "diag": {"pi": [0.5] * 7}}
                   for i in range(50)],                      # the heavy key
        "servedSegIds": list(range(700)),                    # the other heavy key
        "participant": {"expertise": "attending"},
        "sampleSeed": 42,
    }
    r = client.post("/api/results", headers=hdr,
                    json={"sessionId": sid, "result": fat_result,
                          "stopReason": "resolved", "nQuestions": 50})
    assert r.status_code == 200, r.text
    return sid


def test_history_and_dashboard_strip_heavy_result_keys(client):
    """Listing surfaces must NOT ship the per-trial bulk (~95% of a stored
    blob): trials + servedSegIds are stripped, everything the UI reads
    (verdicts/roc/perTask + metadata) survives. The admin endpoint keeps the
    FULL blob — it is the export/debug path."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = _finish_session_with_fat_result(client, hdr)

    hist = client.get("/api/history", headers=hdr).json()["sessions"]
    assert len(hist) == 1 and hist[0]["session_id"] == sid
    res = hist[0]["result"]
    assert "trials" not in res and "servedSegIds" not in res
    assert res["verdicts"] == ["PASS"] * 7
    assert res["roc"][0]["auroc"] == 0.9
    assert len(res["perTask"]) == 7
    assert hist[0]["n_questions"] == 50

    dash = client.get("/api/dashboard", headers=hdr).json()
    assert dash["hasResult"] is True
    assert "trials" not in dash["result"] and "servedSegIds" not in dash["result"]
    assert dash["kpis"]["tasksCertified"] == 7
    assert len(dash["tasks"]) == 7 and dash["tasks"][0]["verdict"] == "PASS"

    # Admin path: the full blob, untouched.
    full = client.get(f"/api/admin/results/{sid}",
                      headers={"X-Admin-Token": "test-admin"}).json()
    assert len(full["trials"]) == 50
    assert len(full["servedSegIds"]) == 700


# ─────────────── API defense-in-depth headers (2026-07-02) ───────────────

def test_api_security_headers_present(client):
    """The API sets its own hardening headers so it is self-protecting if ever
    reached without Caddy in front. JSON-only API → a default-src 'none' CSP
    is safe, and authed JSON must not be cached."""
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert r.headers["cross-origin-resource-policy"] == "same-origin"


def test_api_headers_on_authed_and_error_responses(client):
    """Headers ride on authed 200s AND on error responses (a 401 body is still
    JSON that shouldn't be cached)."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    ok = client.get("/api/dashboard", headers=hdr)
    assert ok.status_code == 200 and ok.headers["cache-control"] == "no-store"
    unauth = client.get("/api/dashboard")   # no token → 401
    assert unauth.status_code == 401
    assert unauth.headers["cache-control"] == "no-store"
    assert unauth.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"


# ───────────────── G4: test → train → retest loop (staging) ─────────────────

def test_g4_train_retest_loop_staging(client):
    """G4 staging e2e: one simulated participant through cert → regimen → train →
    fresh retest, three rounds. Verifies the regimen weak set, that real
    (is_real=1) training exposure + trajectories are persisted, that EVERY trained
    segment is excluded from every subsequent retest draw (D-INT-4/7 spacing), and
    that the is_real quarantine holds against a raw synthetic post."""
    email, pw = _make_participant(client)
    h = _auth_header(client, email, pw)

    def exam_draw():
        r = client.post("/api/session", json={}, headers=h)
        assert r.status_code == 200, r.text
        d = r.json()
        return d["sessionId"], [s["segId"] for s in d["bank"]["segments"]]

    verdicts = ["PASS", "PASS", "REFER_BORDERLINE", "PASS", "FAIL", "PASS", "PASS"]
    result = {"verdicts": verdicts,
              "perTask": [{"taskK": k, "ellStar": 0.3, "verdict": v,
                           "ell": 0.4, "theta": 0.0} for k, v in enumerate(verdicts)]}

    # E0 → store a result with two non-PASSED tasks (lpd=2, lrda=4)
    sid0, drawn0 = exam_draw()
    r = client.post("/api/results",
                    json={"sessionId": sid0, "result": result, "nQuestions": len(drawn0)},
                    headers=h)
    assert r.status_code == 200, r.text

    # regimen = the weak set
    r = client.post("/api/regimen", json={}, headers=h)
    assert r.status_code == 200, r.text
    reg = r.json()["regimen"]
    assert {t["taskK"] for t in reg["deck"]} == {2, 4}
    # real-skill handoff: a measured per-task posterior for ALL 7 tasks
    assert {p["taskK"] for p in reg["prior"]} == set(range(7))
    assert all("ell" in p and "theta" in p and "sd" in p for p in reg["prior"])

    all_trained: set[int] = set()
    prev_drawn = drawn0
    for round_i in range(3):
        r = client.post("/api/training-sessions", json={"taskFocus": "2"}, headers=h)
        tid = r.json()["trainingId"]
        train_segs = prev_drawn[:6]                     # train on fresh (unexcluded) segs
        points = [{"taskK": 2 if i % 2 == 0 else 4, "segId": s, "ell": 0.5 + 0.02 * i,
                   "theta": 0.0, "sd": 0.3, "rt": 1500, "seqInSession": i}
                  for i, s in enumerate(train_segs)]
        r = client.post("/api/training-progress",
                        json={"trainingId": tid, "points": points}, headers=h)
        assert r.status_code == 200 and r.json()["n"] == len(points), r.text
        assert client.post("/api/training-sessions/finalize",
                           json={"trainingId": tid, "nItems": len(points)},
                           headers=h).status_code == 200
        all_trained.update(train_segs)

        # real trajectories surfaced (is_real=1, phase='train')
        traj = client.get("/api/trajectories", headers=h).json()["trajectories"]
        assert any(p["phase"] == "train" for p in traj)

        # fresh RETEST excludes every trained segment
        sid, drawn = exam_draw()
        leaked = all_trained.intersection(drawn)
        assert not leaked, f"round {round_i}: trained segs leaked into retest {leaked}"
        client.post("/api/results", json={"sessionId": sid, "result": result}, headers=h)
        prev_drawn = drawn

    # is_real quarantine: a raw client trajectory post (no isReal) must NOT surface
    client.post("/api/trajectories",
                json={"points": [{"taskK": 0, "ell": 9.9, "phase": "train"}]}, headers=h)
    traj = client.get("/api/trajectories", headers=h).json()["trajectories"]
    assert not any(p.get("ell") == 9.9 for p in traj), "quarantined synthetic point leaked"


def test_training_flag_modes(client):
    """The training-exposure flag gates /dashboard trainingEnabled + 403s the
    training start endpoints. Default 'all' preserves the live-for-everyone
    posture; 'off' is a kill-switch; 'cohort' restricts to the allowlist."""
    email, pw = _make_participant(client)
    h = _auth_header(client, email, pw)
    result = {"verdicts": ["FAIL"] * 7,
              "perTask": [{"taskK": k, "ellStar": 0.3, "verdict": "FAIL",
                           "ell": 0.0, "theta": 0.0} for k in range(7)]}
    sid = client.post("/api/session", json={}, headers=h).json()["sessionId"]
    client.post("/api/results",
                json={"sessionId": sid, "result": result, "nQuestions": 5}, headers=h)
    cfg = client.app.state.cfg

    cfg["training_mode"] = "all"
    assert client.get("/api/dashboard", headers=h).json()["trainingEnabled"] is True
    assert client.post("/api/regimen", json={}, headers=h).status_code == 200
    assert client.post("/api/training-sessions", json={}, headers=h).status_code == 200

    cfg["training_mode"] = "off"
    assert client.get("/api/dashboard", headers=h).json()["trainingEnabled"] is False
    assert client.post("/api/regimen", json={}, headers=h).status_code == 403
    assert client.post("/api/training-sessions", json={}, headers=h).status_code == 403

    cfg["training_mode"] = "cohort"
    cfg["training_allowlist"] = frozenset()
    assert client.get("/api/dashboard", headers=h).json()["trainingEnabled"] is False
    cfg["training_allowlist"] = frozenset({email.lower()})
    assert client.get("/api/dashboard", headers=h).json()["trainingEnabled"] is True
    assert client.post("/api/regimen", json={}, headers=h).status_code == 200


def test_training_monitor_false_graduation(client):
    """The admin monitor computes the SAP safety endpoint: a domain the trainer
    graduated (final trained ℓ ≥ ℓ*) whose next fresh retest did NOT pass is a
    CONFIRMED false graduation. Built deterministically via controlled timestamps
    (utc_now is second-resolution, so the API can't order these reliably)."""
    db = client.app.state.db
    email, _pw = _make_participant(client)         # real participant (sessions.code FK)
    code = db.get_participant_by_email(email)["code"]
    ELLSTAR = 0.3

    def result(v3, v5):
        vmap = {3: v3, 5: v5}
        return {"verdicts": ["PASS"] * 7,
                "perTask": [{"taskK": k, "ellStar": ELLSTAR,
                             "verdict": vmap.get(k, "PASS"),
                             "ell": 0.0, "theta": 0.0} for k in range(7)]}

    def insert_result(sid, when, res):
        with db._connection() as conn:
            conn.execute(db._q(
                "INSERT INTO sessions(session_id, code, participant, started_utc, "
                "finished_utc, status, n_questions) VALUES (?,?,'{}',?,?, 'complete', 5)"),
                (sid, code, when, when))
            conn.execute(db._q(
                "INSERT INTO results(session_id, result, received_utc) VALUES (?,?,?)"),
                (sid, json.dumps(res), when))

    def insert_training(tid, when, grad_tasks):
        with db._connection() as conn:
            conn.execute(db._q(
                "INSERT INTO training_sessions(training_id, code, started_utc, "
                "finished_utc, status, n_items) VALUES (?,?,?,?, 'complete', 4)"),
                (tid, code, when, when))
            for task in grad_tasks:
                conn.execute(db._q(
                    "INSERT INTO param_trajectories(code, task_k, phase, ell, ts, "
                    "training_id, is_real) VALUES (?,?,'train',?,?,?, 1)"),
                    (code, task, 0.5, when, tid))       # 0.5 ≥ ℓ*=0.3 ⇒ graduated

    # cert(pre, both FAIL) → train (graduate 3 & 5) → retest: task3 FAIL, task5 PASS
    insert_result("s-pre", "2026-01-01T00:00:00Z", result("FAIL", "FAIL"))
    insert_training("tr-1", "2026-01-02T00:00:00Z", grad_tasks=[3, 5])
    insert_result("s-post", "2026-01-03T00:00:00Z", result("FAIL", "PASS"))

    m = db.training_monitor()
    assert m["trainingTrials"] == 2
    assert m["learners"] == 1
    assert m["graduatedDomains"] == 2       # both 3 and 5 cleared ℓ*
    assert m["confirmedRetests"] == 2       # both had a later retest
    assert m["falseGraduations"] == 1       # only task 3's retest FAILed
    assert abs(m["falseGraduationRate"] - 0.5) < 1e-9

    # endpoint: admin-gated + echoes the flag mode
    r = client.get("/api/admin/training-monitor",
                   headers={"X-Admin-Token": "test-admin"})
    assert r.status_code == 200, r.text
    assert r.json()["falseGraduations"] == 1 and r.json()["trainingMode"] == "all"
    assert client.get("/api/admin/training-monitor").status_code == 403


def test_training_monitor_empty(client):
    """Monitor on an empty DB returns zeros + a null rate (no divide-by-zero)."""
    m = client.app.state.db.training_monitor()
    assert m["learners"] == 0 and m["trainingTrials"] == 0
    assert m["graduatedDomains"] == 0 and m["falseGraduationRate"] is None

# ─────────────── G1 robustness hardening (2026-07-07) ───────────────

def test_forgot_survives_email_send_failure_no_account_oracle(client, monkeypatch):
    """An SMTP outage must not turn /forgot into an account-existence oracle:
    a real account and an unknown email must BOTH still answer 200."""
    from . import mailer as email_mod
    email, pw = _make_participant(client)

    def boom(*a, **k):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(email_mod, "send_auth_code", boom)
    r_real = client.post("/api/forgot", json={"email": email})
    r_fake = client.post("/api/forgot", json={"email": "nobody@example.test"})
    assert r_real.status_code == 200 and r_fake.status_code == 200
    assert "devCode" not in r_real.json()      # send failed → no code echoed


def test_resend_survives_email_send_failure(client, monkeypatch):
    """Same anti-oracle contract for /verify/resend."""
    from . import mailer as email_mod
    email, _pw, _code = _register(client)      # unverified account

    def boom(*a, **k):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(email_mod, "send_auth_code", boom)
    r = client.post("/api/verify/resend", json={"email": email})
    assert r.status_code == 200
    assert "devCode" not in r.json()


def test_videos_malformed_meta_is_400(client):
    """Bad meta (not JSON / missing shape / non-3-int shape) is a clean 400,
    never an unhandled 500."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    files, _ = _viz_files()
    for bad_meta in ("not-json{", json.dumps({"noShape": 1}),
                     json.dumps({"shape": [2, 3]}),
                     json.dumps({"shape": ["a", "b", "c"]}),
                     json.dumps([1, 2, 3])):
        r = client.post("/api/videos", headers=hdr,
                        data={"meta": bad_meta}, files=files)
        assert r.status_code == 400, (bad_meta, r.status_code, r.text)


def test_trajectory_points_missing_taskK_is_422(client):
    """A malformed point must 4xx up front, not KeyError mid-transaction."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    r = client.post("/api/trajectories", headers=hdr,
                    json={"points": [{"ell": 0.5}]})
    assert r.status_code == 422
    r = client.post("/api/trajectories", headers=hdr,
                    json={"points": [{"taskK": "not-an-int"}]})
    assert r.status_code == 422
    # A valid batch still lands.
    r = client.post("/api/trajectories", headers=hdr,
                    json={"points": [{"taskK": 1, "ell": 0.5, "isReal": False}]})
    assert r.status_code == 200


def test_training_progress_bad_points_rejected(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    tid = client.post("/api/training-sessions", json={}, headers=hdr).json()["trainingId"]
    r = client.post("/api/training-progress", headers=hdr,
                    json={"trainingId": tid, "points": [{"segId": 5}]})
    assert r.status_code == 422                     # no taskK
    r = client.post("/api/training-progress", headers=hdr,
                    json={"trainingId": tid,
                          "points": [{"taskK": 2, "segId": "abc"}]})
    assert r.status_code == 422                     # non-int segId
    r = client.post("/api/training-progress", headers=hdr,
                    json={"trainingId": tid,
                          "points": [{"taskK": 2, "segId": 5, "ell": 0.1}]})
    assert r.status_code == 200, r.text             # valid batch still lands


def test_points_batch_cap_is_413(client):
    from .routers import dashboard as dash_router
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    too_many = [{"taskK": 0}] * (dash_router.MAX_POINTS_PER_POST + 1)
    r = client.post("/api/trajectories", headers=hdr, json={"points": too_many})
    assert r.status_code == 413


def test_api_body_cap_413(client):
    """A declared body over the 4 MB API cap is rejected before parsing —
    the API-side mirror of the Caddy edge limit."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    big = "x" * (4 * 1024 * 1024 + 100)
    r = client.post("/api/trajectories", headers=hdr,
                    json={"points": [], "pad": big})
    assert r.status_code == 413


def test_orphan_viz_dirs_swept(tmp_path, monkeypatch):
    """cortex_viz_* dirs from a dead process are reclaimed once per boot;
    fresh dirs (a live job's) are left alone."""
    import tempfile as _tempfile
    from .routers import videos as videos_router
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_path))
    old = tmp_path / "cortex_viz_dead"
    old.mkdir()
    import os as _os
    stale = time.time() - videos_router._JOB_TTL_SECONDS - 60
    _os.utime(old, (stale, stale))
    fresh = tmp_path / "cortex_viz_live"
    fresh.mkdir()
    videos_router._sweep_orphan_dirs()
    assert not old.exists()
    assert fresh.exists()

# ─────────────── G2 efficiency (2026-07-07) ───────────────

def test_dashboard_uses_latest_of_multiple_attempts(client):
    """With several completed attempts, /api/dashboard reflects the NEWEST one
    via the LIMIT-1 fetch (latest_result_for_code) — and the helper's row
    matches list_results_for_code[0]."""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    db = client.app.state.db

    def _finish(verdict):
        r = client.post("/api/session", json={"participant": {}}, headers=hdr)
        sid = r.json()["sessionId"]
        res = {"perTask": [{"taskK": 0, "code": "spike", "verdict": verdict,
                            "ell": 0.5, "ellStar": 0.0, "theta": 0.0,
                            "auroc": 0.9, "sd": 0.1}]}
        r = client.post("/api/results", headers=hdr,
                        json={"sessionId": sid, "result": res,
                              "stopReason": "test", "nQuestions": 1})
        assert r.status_code == 200, r.text
        return sid

    _finish("FAIL")
    time.sleep(1.1)          # finished_utc has 1 s resolution; force ordering
    sid2 = _finish("PASS")

    latest = db.latest_result_for_code(_code_of(client, email))
    assert latest["session_id"] == sid2
    assert latest == db.list_results_for_code(_code_of(client, email))[0]

    d = client.get("/api/dashboard", headers=hdr).json()
    assert d["hasResult"] is True
    verdicts = {t["taskK"]: t["verdict"] for t in d["tasks"]}
    assert verdicts.get(0) == "PASS"


def _code_of(client, email) -> str:
    return client.app.state.db.get_participant_by_email(email)["code"]


def test_pg_pool_max_size_env(monkeypatch):
    """CORTEX_PG_POOL_MAX drives the pool ceiling (default 10). psycopg isn't
    installed in CI, so fake the modules _make_pool lazily imports."""
    import sys
    import types
    captured = {}

    class _FakePool:
        def __init__(self, conninfo, **kw):
            captured.update(kw)

        @staticmethod
        def check_connection(conn):
            return None

        def connection(self):
            raise RuntimeError("not needed")

        def close(self):
            pass

    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.rows = fake_rows
    fake_pool_mod = types.ModuleType("psycopg_pool")
    fake_pool_mod.ConnectionPool = _FakePool
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)
    monkeypatch.setitem(sys.modules, "psycopg_pool", fake_pool_mod)
    monkeypatch.setenv("CORTEX_PG_POOL_MAX", "7")
    try:
        Database("postgresql://u:pw@localhost/x")
    except Exception:
        pass    # boot migrations fail on the fake pool — sizing already captured
    assert captured.get("max_size") == 7


def test_cohort_list_counts_via_grouped_query(client):
    """memberCount in the cohort list comes from active_member_counts (one
    grouped query) and counts only ACTIVE members."""
    email_m, pw_m = _make_participant(client)
    hdr_m = _auth_header(client, email_m, pw_m)
    r = client.post("/api/cohorts", json={"name": "Pod"}, headers=hdr_m)
    cid = r.json()["cohortId"]
    # Invite a second user; while the invite is pending only the manager counts.
    email_b, pw_b = _make_participant(client)
    hdr_b = _auth_header(client, email_b, pw_b)
    pid_b = client.get("/api/profile", headers=hdr_b).json()["publicId"]
    assert client.post(f"/api/cohorts/{cid}/invite", headers=hdr_m,
                       json={"publicId": pid_b}).status_code == 200
    lst = client.get("/api/cohorts", headers=hdr_m).json()["cohorts"]
    assert [c["memberCount"] for c in lst if c["cohortId"] == cid] == [1]
    assert client.post(f"/api/cohorts/{cid}/accept", headers=hdr_b).status_code == 200
    lst = client.get("/api/cohorts", headers=hdr_m).json()["cohorts"]
    assert [c["memberCount"] for c in lst if c["cohortId"] == cid] == [2]
    # Direct helper: grouped counts + empty input.
    db = client.app.state.db
    assert db.active_member_counts([cid])[cid] == 2
    assert db.active_member_counts([]) == {}


# ─────────────── G3 transactional /results (2026-07-07) ───────────────

def test_results_writes_are_atomic(client):
    """A failure inside store_result_finalized rolls back ALL of it: no
    results row, session still in_progress, no eval trajectory rows. (The old
    three-separate-writes shape could leave a stored result on a session that
    was never finalized — which then shadowed the real latest attempt on
    Postgres, where NULL finished_utc sorts first under DESC.)"""
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    db = client.app.state.db
    r = client.post("/api/session", json={"participant": {}}, headers=hdr)
    sid = r.json()["sessionId"]
    code = _code_of(client, email)

    with pytest.raises(Exception):
        # A malformed eval point (no taskK) blows up on the LAST leg of the
        # transaction, after result + finalize already executed.
        db.store_result_finalized(sid, code, {"perTask": []}, "test", 1,
                                  [{"ell": 0.5}])

    assert db.get_result(sid) is None                       # rolled back
    sess = db.get_session(sid)
    assert sess["status"] == "in_progress" and sess["finished_utc"] is None
    assert db.get_trajectories(code) == []

    # The clean path still lands everything.
    db.store_result_finalized(sid, code, {"perTask": []}, "test", 1,
                              [{"taskK": 0, "ell": 0.5}])
    assert db.get_result(sid) is not None
    assert db.get_session(sid)["status"] == "complete"
    assert len(db.get_trajectories(code)) == 1
