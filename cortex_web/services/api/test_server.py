"""Backend tests — security primitives + full API round-trip on a temp DB.

Run from cortex_web/:
    python -m pytest server/test_server.py -q
"""
from __future__ import annotations

import json
import time
from pathlib import Path

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
    with db._lock:                                   # FK: sessions.code → participants
        db._exec("INSERT INTO participants(code, password_hash, created_utc) "
                 "VALUES (?,?,?)", ("C", "x", "2000-01-01T00:00:00Z"))
        db._conn.commit()
    db.create_session("sA", "C", {}, 1)
    db.upsert_trial("sA", {"trialIndex": 0, "segId": 10, "taskK": 0})
    db.create_session("sB", "C", {}, 2)
    db.upsert_trial("sB", {"trialIndex": 0, "segId": 20, "taskK": 0})
    with db._lock:                                   # backdate sA far into the past
        db._exec("UPDATE sessions SET started_utc=? WHERE session_id=?",
                 ("2000-01-01T00:00:00Z", "sA"))
        db._conn.commit()
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
    from .app import _question_breakdown
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
    from . import app as app_module
    monkeypatch.setattr(app_module, "_TRUTH_CACHE", _TRUTH)
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
    with db._lock:
        db._exec("INSERT INTO sessions(session_id,code,participant,started_utc,"
                 "finished_utc,status) VALUES(?,?,?,?,?, 'complete')",
                 ("act-tz", code, "{}", "2026-01-15T03:00:00Z", "2026-01-15T04:00:00Z"))
        db._conn.commit()
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


def test_report_emails_receiver_with_diagnostics(client, monkeypatch):
    """A submitted report is emailed to the receiver with the message + the
    browser diagnostics, and Reply-To set to the reporter's (normalized) email."""
    from . import email as email_mod
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
    from . import app as app_module
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setattr(app_module, "_verify_google_credential", lambda cred, cid: claims)


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
    from . import app as app_module
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    def boom(cred, cid):
        raise ValueError("bad audience / signature")
    monkeypatch.setattr(app_module, "_verify_google_credential", boom)
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


def test_pg_exec_rolls_back_on_error(tmp_path):
    """Regression for the 2026-06-21 prod auth outage: on Postgres the shared
    autocommit=False connection must roll back when a statement errors, or the
    aborted transaction poisons every later query (InFailedSqlTransaction) until
    restart — a single deadlock/timeout becomes a total login outage. CI has no
    Postgres, so we drive the _pg branch with a fake connection and assert
    _exec rolls back (and re-raises) on failure."""
    db = Database(tmp_path / "rb.db")   # real sqlite instance; we override _pg

    class _FakeCursor:
        def execute(self, sql, params):
            raise RuntimeError("simulated deadlock")

    class _FakeConn:
        closed = False        # a live connection with only an aborted txn —
        broken = False        # _exec must roll back (not reconnect) this case
        def __init__(self):
            self.rolled_back = 0
        def cursor(self):
            return _FakeCursor()
        def rollback(self):
            self.rolled_back += 1

    fake = _FakeConn()
    db._pg = True
    db._conn = fake
    with pytest.raises(RuntimeError):
        db._exec("SELECT 1")
    assert fake.rolled_back == 1, "a failed statement must roll back the txn"


def test_pg_fetch_commits_read_txn(tmp_path):
    """Regression for the 2026-06-22 nightly-backup hang: on Postgres the shared
    autocommit=False connection must COMMIT after a read, or a bare SELECT leaves
    it `idle in transaction` indefinitely, holding AccessShare on the table. That
    AccessShare blocks the AccessExclusive a second Database()'s boot migration
    needs (the backup's `export-sessions` ALTER hung exactly this way for hours)
    and pins the VACUUM xmin. CI has no Postgres, so drive the _pg branch with a
    fake connection and assert _fetchone/_fetchall each commit the read txn."""
    db = Database(tmp_path / "rc.db")   # real sqlite instance; we override _pg

    class _FakeCursor:
        def execute(self, sql, params):
            pass
        def fetchone(self):
            return {"x": 1}
        def fetchall(self):
            return [{"x": 1}, {"x": 2}]
        def close(self):
            pass

    class _FakeConn:
        def __init__(self):
            self.committed = 0
        def cursor(self):
            return _FakeCursor()
        def commit(self):
            self.committed += 1
        def rollback(self):
            pass

    fake = _FakeConn()
    db._pg = True
    db._conn = fake
    db._fetchone("SELECT 1")
    db._fetchall("SELECT 1")
    assert fake.committed == 2, "each read must commit so the shared conn never lingers idle-in-transaction"


class _Info:
    """Stand-in for psycopg's conn.info — only transaction_status is read by
    _exec (0 == PQTRANS_IDLE, 2 == PQTRANS_INTRANS). Kept psycopg-free so these
    run in the SQLite-only dev/CI venv, like the other _pg tests above."""
    def __init__(self, status): self.transaction_status = status


def test_pg_reconnects_when_connection_severed(tmp_path, monkeypatch):
    """Regression for the 2026-06-24 prod auth outage: on Postgres the single
    long-lived connection can be SEVERED out from under the app — a Postgres
    restart (an unattended-upgrade of a libpq dependency bounces the service), a
    terminated backend, or a dropped socket. With no recovery the dead connection
    makes every later request 500 until a manual restart (the connection killed
    by the Jun-23 06:21 Postgres restart surfaced as AdminShutdown on the next
    login and stayed down). _exec must detect the dead connection, reconnect, and
    retry the statement at a clean transaction boundary so the request self-heals.
    Driven with fakes (the _exec discriminator is the conn's broken flag, not the
    exception type), so it needs no Postgres."""
    db = Database(tmp_path / "rx.db")   # real sqlite instance; we override _pg

    class _DeadCursor:
        def execute(self, sql, params):
            raise RuntimeError("the connection is lost")

    class _DeadConn:                 # server killed it: next op fails, conn broken
        closed, broken = False, True
        info = _Info(0)              # was idle when killed → clean boundary
        def cursor(self): return _DeadCursor()
        def rollback(self): raise RuntimeError("the connection is lost")
        def close(self): pass

    class _LiveCursor:
        def __init__(self, conn): self._conn = conn
        def execute(self, sql, params): self._conn.executed += 1
        def fetchone(self): return {"ok": 1}
        def close(self): pass

    class _FreshConn:                # what _reconnect() hands back
        closed, broken = False, False
        info = _Info(0)
        def __init__(self): self.executed = 0; self.committed = 0
        def cursor(self): return _LiveCursor(self)
        def commit(self): self.committed += 1
        def close(self): pass

    fresh = _FreshConn()
    db._pg = True
    db._conn = _DeadConn()
    monkeypatch.setattr(db, "_pg_connect", lambda: fresh)

    # A read on the severed connection must transparently reconnect + retry,
    # returning the row instead of bubbling a 500.
    row = db._fetchone("SELECT * FROM participants WHERE email=?", ("x@y.z",))
    assert db._conn is fresh, "must reconnect to a fresh connection"
    assert fresh.executed == 1, "must retry the statement on the fresh connection"
    assert row == {"ok": 1}


def test_pg_severed_mid_transaction_does_not_retry(tmp_path, monkeypatch):
    """Safety guard for the reconnect path: if the connection dies mid-write (an
    earlier statement in the transaction was already lost with it), _exec must
    reconnect for the NEXT request but must NOT retry this statement — replaying
    half of a multi-statement write on the fresh connection would persist a
    partial write. So it reconnects and re-raises; the next request starts
    clean."""
    db = Database(tmp_path / "rx2.db")

    class _DeadCursor:
        def execute(self, sql, params):
            raise RuntimeError("the connection is lost")

    class _DeadConn:
        closed, broken = False, True
        info = _Info(2)             # 2 == PQTRANS_INTRANS → mid-transaction
        def cursor(self): return _DeadCursor()
        def rollback(self): pass
        def close(self): pass

    reconnected = {"n": 0}
    def fake_connect():
        reconnected["n"] += 1
        return _DeadConn()
    db._pg = True
    db._conn = _DeadConn()
    monkeypatch.setattr(db, "_pg_connect", fake_connect)

    with pytest.raises(RuntimeError):
        db._exec("INSERT INTO participants(code,password_hash,created_utc) VALUES (?,?,?)",
                 ("c", "h", "t"))
    assert reconnected["n"] == 1, "must reconnect so the next request is clean"


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
