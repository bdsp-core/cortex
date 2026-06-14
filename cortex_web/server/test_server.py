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
    app = create_app(db_path=tmp_path / "t.db")
    return TestClient(app)


_REG_COUNTER = [0]


def _make_participant(client) -> tuple[str, str]:
    """Create an account via the public /api/register endpoint; return
    (email, password) so the caller can also authenticate later."""
    _REG_COUNTER[0] += 1
    email = f"t{_REG_COUNTER[0]}@example.test"
    pw = "test-pw-1234567890"
    r = client.post("/api/register",
                    json={"email": email, "password": pw,
                          "displayName": f"Test User {_REG_COUNTER[0]}"})
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


def test_db_direct_participant():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        db = Database(os.path.join(d, "x.db"))
        db.add_participant("c1", security.hash_password("p"), "lbl")
        assert db.get_participant("c1")["label"] == "lbl"
        db.set_participant_active("c1", False)
        assert db.get_participant("c1")["active"] == 0
        db.close()
