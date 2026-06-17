"""Backend tests — security primitives + full API round-trip on a temp DB.

Run from cortex_web/:
    python -m pytest server/test_server.py -q
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from . import security
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

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("CORTEX_JWT_SECRET", "test-secret")
    monkeypatch.setenv("CORTEX_EMAIL_BACKEND", "dev")        # no real SES in tests
    monkeypatch.setenv("CORTEX_EMAIL_EXPOSE_CODE", "1")      # echo codes for the flow
    app = create_app(db_path=tmp_path / "t.db")
    return TestClient(app)


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


def test_register_dup_email_is_409(client):
    email, _pw = _make_participant(client)
    r = client.post("/api/register",
                    json={"email": email,
                          "password": "anotherpw1234567890",
                          "displayName": "Y"})
    assert r.status_code == 409


def test_register_honeypot_silently_drops_bot(client):
    r = client.post("/api/register",
                    json={"email": "bot@example.test",
                          "password": "test-pw-1234567890",
                          "displayName": "Bot",
                          "honeypot": "https://spam.example.com"})
    assert r.status_code == 200
    # but the bot's "account" was NOT actually created — proves it by
    # registering with the same email succeeding normally.
    r2 = client.post("/api/register",
                     json={"email": "bot@example.test",
                           "password": "test-pw-1234567890",
                           "displayName": "Real human"})
    assert r2.status_code == 200, r2.text


def test_gated_requires_token(client):
    assert client.get("/api/manifest").status_code == 401
    assert client.post("/api/session", json={"participant": {}}).status_code == 401


def test_full_session_flow(client):
    code, pw = _make_participant(client)
    hdr = _auth_header(client, code, pw)

    man = client.get("/api/manifest", headers=hdr).json()
    assert man["sessionSample"] == 500

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


def test_dashboard_empty_returns_sample(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    d = client.get("/api/dashboard", headers=hdr).json()
    assert d["hasResult"] is False and d["result"] is None
    assert len(d["tasks"]) == 7 and d["sample"] is True
    assert d["kpis"]["streak"] >= 0


def test_dashboard_reflects_latest_result(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    client.post("/api/results", headers=hdr,
                json={"sessionId": sid, "result": {"verdicts": ["PASS"] * 7},
                      "stopReason": "all_resolved", "nQuestions": 30})
    d = client.get("/api/dashboard", headers=hdr).json()
    assert d["hasResult"] is True
    assert d["result"]["verdicts"] == ["PASS"] * 7


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


def test_trajectories_sample_then_real(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    t0 = client.get("/api/trajectories", headers=hdr).json()
    assert t0["sample"] is True and len(t0["trajectories"]) == 7 * 9
    client.post("/api/trajectories", headers=hdr,
                json={"points": [{"taskK": 0, "phase": "train", "ell": 0.9,
                                  "theta": 0.1, "sd": 0.12, "rt": 2000}]})
    t1 = client.get("/api/trajectories", headers=hdr).json()
    assert t1["sample"] is False and len(t1["trajectories"]) == 1


def test_regimen_sample_then_real(client):
    email, pw = _make_participant(client)
    hdr = _auth_header(client, email, pw)
    assert client.get("/api/regimen", headers=hdr).json()["sample"] is True
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
    from . import email as email_mod

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
            sent["body"] = msg.get_content()

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
