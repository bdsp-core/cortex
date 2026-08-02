"""Server-side replay-verification plumbing (api.result_verify).

The node replay itself is exercised end-to-end against the local stack (a
real sitting replayed by the real engine); here the runner is stubbed and
the tests pin the plumbing: fixture export fidelity, verdict bookkeeping,
sweep/retry semantics, and the ingest hook's kill-switch.
"""
from __future__ import annotations

import json

import pytest

from . import result_verify
from .test_server import (  # reuse the live-app fixtures
    _auth_header, _make_participant, client,  # noqa: F401
)


def _complete_precision_session(client):
    client.app.state.cfg["precision_policy_rollout"] = "all"
    code, pw = _make_participant(client)
    hdr = _auth_header(client, code, pw)
    body = client.post("/api/session", headers=hdr,
                       json={"participant": {}, "sampleSeed": 77}).json()
    sid = body["sessionId"]
    profile = body["bank"]["nwayProfile"]
    for i in range(3):
        assert client.post("/api/progress", headers=hdr, json={
            "sessionId": sid,
            "trial": {"trialIndex": i, "segId": 10 + i, "taskK": 0, "pick": 0,
                      "diag": {"segId": 10 + i}},
        }).status_code == 200
    accepted = client.post("/api/results", headers=hdr, json={
        "sessionId": sid,
        "result": {
            "terminationPolicy": "precision_v1",
            "nwayProfile": profile,
            "verdicts": ["ABOVE_CUT"] * 7,
            "determinations": ["DETERMINED"] * 7,
        },
        "stopReason": "all_estimated_or_undeterminable", "nQuestions": 3,
    })
    assert accepted.status_code == 200, accepted.text
    return sid, profile


def test_export_fixture_carries_the_stamp_and_ordered_picks(client):
    sid, profile = _complete_precision_session(client)
    fixture, bundle_version = result_verify.export_fixture(
        client.app.state.db, sid)
    assert fixture["sessionId"] == sid
    assert fixture["sampleSeed"] == 77
    assert fixture["exclusion"] == []
    assert [t["diag"]["segId"] for t in fixture["trials"]] == [10, 11, 12]
    assert fixture["nwayProfile"] == profile
    assert fixture["stopReason"] == "all_estimated_or_undeterminable"
    assert fixture["result"]["verdicts"] == ["ABOVE_CUT"] * 7
    assert bundle_version  # the test bank stamps its version


def test_verify_session_records_pass_divergent_and_error(client, monkeypatch):
    db = client.app.state.db
    sid, _profile = _complete_precision_session(client)

    monkeypatch.setattr(result_verify, "manifest_path", lambda v: "unused")
    monkeypatch.setattr(result_verify, "_run_replay", lambda fixture, m: {
        "meaningfullyEquivalent": True, "expectedQuestions": 3,
        "replayedQuestions": 3, "trajectorySha256": "t" * 64,
    })
    assert result_verify.verify_session(db, sid) == "pass"
    row = db.result_verification(sid)
    assert row["verify_status"] == "pass"
    assert json.loads(row["verify_detail"])["replayedQuestions"] == 3
    assert row["verified_utc"]

    monkeypatch.setattr(result_verify, "_run_replay", lambda fixture, m: {
        "meaningfullyEquivalent": False,
        "meaningfulDifferences": ["result.verdicts[1]: A != B"],
    })
    assert result_verify.verify_session(db, sid) == "divergent"
    detail = json.loads(db.result_verification(sid)["verify_detail"])
    assert detail["meaningfulDifferences"] == ["result.verdicts[1]: A != B"]

    def _boom(fixture, m):
        raise result_verify.VerifyError("no manifest on this host")
    monkeypatch.setattr(result_verify, "_run_replay", _boom)
    assert result_verify.verify_session(db, sid) == "error"
    assert "no manifest" in db.result_verification(sid)["verify_detail"]

    # A retired stamp the engine refuses is terminal, never re-swept.
    def _refused(fixture, m):
        raise result_verify.VerifyError(
            "replay produced no summary: n-way profile mismatch: engineProfileId")
    monkeypatch.setattr(result_verify, "_run_replay", _refused)
    assert result_verify.verify_session(db, sid) == "unreplayable"
    assert db.unverified_result_sessions() == []


def test_sweep_retries_errors_but_never_reopens_verdicts(client, monkeypatch):
    db = client.app.state.db
    sid_a, _ = _complete_precision_session(client)
    sid_b, _ = _complete_precision_session(client)
    monkeypatch.setattr(result_verify, "manifest_path", lambda v: "unused")
    monkeypatch.setattr(result_verify, "_run_replay", lambda fixture, m: {
        "meaningfullyEquivalent": True,
    })
    # Both pending → both verified by the sweep.
    statuses = result_verify.sweep(db, limit=10)
    assert statuses == {sid_a: "pass", sid_b: "pass"}
    # A pass (or a divergent) is final: nothing pending remains.
    assert result_verify.sweep(db, limit=10) == {}
    # An error re-enters the sweep until it heals.
    db.set_result_verification(sid_a, "error", None)
    assert db.unverified_result_sessions() == [sid_a]
    assert result_verify.sweep(db, limit=10) == {sid_a: "pass"}
    # Divergent does NOT re-enter (a real divergence must stay visible).
    db.set_result_verification(sid_b, "divergent", None)
    assert db.unverified_result_sessions() == []


def test_ingest_hook_respects_the_kill_switch(client, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(result_verify, "verify_session",
                        lambda db, sid: calls.append(sid))
    # conftest sets CORTEX_RESULT_VERIFY=off: the hook must not fire.
    sid, _ = _complete_precision_session(client)
    assert calls == []
    # With the switch back on, ingest schedules a verification.
    monkeypatch.delenv("CORTEX_RESULT_VERIFY")
    threads: list = []
    real_thread = result_verify.threading.Thread

    class _Immediate(real_thread):  # run synchronously for the assertion
        def start(self):
            threads.append(self)
            self.run()
    monkeypatch.setattr(result_verify.threading, "Thread", _Immediate)
    sid2, _ = _complete_precision_session(client)
    assert calls == [sid2] and len(threads) == 1


def test_export_refuses_ad6_and_missing_result(client):
    db = client.app.state.db
    code, pw = _make_participant(client)
    hdr = _auth_header(client, code, pw)
    client.app.state.cfg["precision_policy_rollout"] = "off"
    sid = client.post("/api/session", headers=hdr,
                      json={"participant": {}}).json()["sessionId"]
    with pytest.raises(result_verify.VerifyError):
        result_verify.export_fixture(db, sid)   # ad6 → not replayable
    with pytest.raises(result_verify.VerifyError):
        result_verify.export_fixture(db, "nonexistent-session")
