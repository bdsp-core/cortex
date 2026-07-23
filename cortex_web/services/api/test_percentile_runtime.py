from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient

from .app import create_app
from .percentile_rollout import profile_for
from .percentile_runtime import (
    DOMAINS, PercentileRuntime, validate_percentile_report,
)
from .test_server import _auth_header, _make_participant, _write_test_bank

HERE = Path(__file__).resolve()
METADATA = HERE.parents[2] / "apps" / "web" / "public" / "norms" \
    / "historical-calibration-k7-provisional-v1.json"


def test_runtime_verifies_and_scores_all_domains():
    runtime = PercentileRuntime(METADATA)
    assert runtime.ready, runtime.error
    profile = runtime.profile(display=True)
    assert profile["normId"] == "historical-calibration-k7-provisional-v1"
    scores = runtime.score_domains({
        domain: ([0.0, 0.2], [0.6, 0.4]) for domain in DOMAINS
    })
    assert set(scores["domains"]) == set(DOMAINS)
    for score in scores["domains"].values():
        assert 0 <= score["estimate"] <= 100
        assert 0 <= score["lower"] <= score["upper"] <= 100
        assert score["referenceReplicates"] == 500


def test_runtime_fails_closed_on_changed_binary(tmp_path):
    metadata = json.loads(METADATA.read_text())
    meta_path = tmp_path / METADATA.name
    data_path = tmp_path / Path(metadata["runtimeDataUrl"]).name
    meta_path.write_text(METADATA.read_text())
    raw = METADATA.with_name(data_path.name).read_bytes()
    data_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    runtime = PercentileRuntime(meta_path)
    assert not runtime.ready
    assert "SHA-256 mismatch" in (runtime.error or "")


def test_public_rollout_requires_exact_operator_acknowledgement():
    runtime = PercentileRuntime(METADATA)
    participant = {"email": "doctor@example.org", "public_id": "123456789"}
    assert profile_for(runtime, {"percentile_mode": "all"}, "p", participant) is None
    cfg = {
        "percentile_mode": "all",
        "percentile_release_sha256": runtime.metadata["runtimeDataSha256"],
    }
    assert profile_for(runtime, cfg, "p", participant)["display"] is True
    shadow = profile_for(
        runtime, {"percentile_mode": "shadow"}, "p", participant)
    assert shadow["display"] is False
    assert profile_for(runtime, {
        **cfg, "percentile_mode": "cohort", "percentile_allowlist": frozenset(),
    }, "p", participant) is None
    cohort = {
        **cfg,
        "percentile_mode": "cohort",
        "percentile_allowlist": frozenset({"elikeldsen@icloud.com"}),
    }
    assert profile_for(
        runtime, cohort, "p", {"email": "ELIKELDSEN@icloud.com"}
    )["display"] is True
    assert profile_for(runtime, cohort, "p", participant) is None


def test_report_validator_enforces_exact_stamp_and_shape():
    runtime = PercentileRuntime(METADATA)
    profile = runtime.profile(display=True)
    domains = {
        domain: runtime.score_domain(domain, [0.0], [1.0])
        for domain in DOMAINS
    }
    report = {"status": "available", "profile": profile, "domains": domains}
    assert validate_percentile_report(profile, report) == []
    changed = deepcopy(report)
    changed["profile"]["normId"] = "other"
    assert "percentile profile does not match session stamp" in \
        validate_percentile_report(profile, changed)
    assert validate_percentile_report(None, report) == [
        "unstamped session reported percentile data"
    ]
    legacy = {
        "status": "unavailable_legacy_client",
        "profile": profile,
    }
    assert validate_percentile_report(profile, legacy) == []


def _percentile_client(tmp_path, monkeypatch, mode="all", release=True):
    monkeypatch.setenv("CORTEX_ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("CORTEX_JWT_SECRET", "test-secret")
    monkeypatch.setenv("CORTEX_EMAIL_BACKEND", "dev")
    monkeypatch.setenv("CORTEX_EMAIL_EXPOSE_CODE", "1")
    monkeypatch.setenv("CORTEX_PRECISION_POLICY_ROLLOUT", "email_allowlist")
    _write_test_bank(tmp_path / "bundle", "test-bank", per_class=8)
    monkeypatch.setenv("CORTEX_BUNDLE_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("CORTEX_BUNDLE_URL", "/bundle/test-bank")
    monkeypatch.setenv("CORTEX_PERCENTILE_MODE", mode)
    monkeypatch.setenv("CORTEX_PERCENTILE_NORM_METADATA", str(METADATA))
    runtime = PercentileRuntime(METADATA)
    monkeypatch.setenv(
        "CORTEX_PERCENTILE_RELEASE_SHA256",
        runtime.metadata["runtimeDataSha256"] if release else "wrong",
    )
    return TestClient(create_app(db_path=tmp_path / "percentile.db"))


def test_session_stamp_and_result_ingest_contract(tmp_path, monkeypatch):
    client = _percentile_client(tmp_path, monkeypatch)
    email, password = _make_participant(client)
    headers = _auth_header(client, email, password)
    started = client.post("/api/session", json={}, headers=headers)
    assert started.status_code == 200, started.text
    started = started.json()
    profile = started["percentileProfile"]
    assert profile["display"] is True
    runtime = client.app.state.percentile_runtime
    report = {
        "status": "available",
        "profile": profile,
        "domains": {
            domain: runtime.score_domain(domain, [0.0], [1.0])
            for domain in DOMAINS
        },
    }
    accepted = client.post("/api/results", headers=headers, json={
        "sessionId": started["sessionId"],
        "result": {"percentile": report},
        "nQuestions": 0,
    })
    assert accepted.status_code == 200, accepted.text
    row = client.app.state.db.get_session(started["sessionId"])
    assert row["norm_id"] == profile["normId"]
    assert row["norm_sha256"] == profile["normSha256"]
    dashboard = client.get("/api/dashboard", headers=headers).json()
    assert len(dashboard["tasks"]) == 7
    assert all(task["percentile"] is not None for task in dashboard["tasks"])
    assert all(
        task["percentileProfile"]["normSha256"] == profile["normSha256"]
        for task in dashboard["tasks"]
    )
    history = client.get("/api/history", headers=headers).json()
    assert history["sessions"][0]["result"]["percentile"]["status"] == "available"


def test_stamped_session_preserves_result_from_legacy_client(
        tmp_path, monkeypatch):
    client = _percentile_client(tmp_path, monkeypatch)
    email, password = _make_participant(client)
    headers = _auth_header(client, email, password)
    started = client.post("/api/session", json={}, headers=headers).json()
    accepted = client.post("/api/results", headers=headers, json={
        "sessionId": started["sessionId"],
        "result": {},
        "nQuestions": 0,
    })
    assert accepted.status_code == 200, accepted.text
    history = client.get("/api/history", headers=headers).json()
    report = history["sessions"][0]["result"]["percentile"]
    assert report == {
        "status": "unavailable_legacy_client",
        "profile": started["percentileProfile"],
    }


def test_bad_release_ack_leaves_new_sessions_unstamped(tmp_path, monkeypatch):
    client = _percentile_client(tmp_path, monkeypatch, release=False)
    health = client.get("/api/health?deep=1")
    assert health.status_code == 503
    assert health.json()["percentile"]["releaseAcknowledged"] is False
    email, password = _make_participant(client)
    started = client.post(
        "/api/session", json={},
        headers=_auth_header(client, email, password),
    )
    assert started.status_code == 200
    assert started.json()["percentileProfile"] is None


def test_training_summary_is_server_authoritative(tmp_path, monkeypatch):
    client = _percentile_client(tmp_path, monkeypatch)
    email, password = _make_participant(client)
    headers = _auth_header(client, email, password)
    training = client.post(
        "/api/training-sessions", json={}, headers=headers)
    assert training.status_code == 200, training.text
    training = training.json()
    assert training["percentileProfile"]["display"] is True
    pool = client.post("/api/training-bank", json={}, headers=headers).json()
    seg_ids = [row["segId"] for row in pool["bank"]["segments"]]
    started = client.post("/api/training-engine/start", headers=headers, json={
        "trainingId": training["trainingId"], "segIds": seg_ids,
    })
    assert started.status_code == 200, started.text
    started = started.json()
    assert all(s["percentile"] is not None for s in started["snapshot"])
    item = started["item"]
    recorded = client.post("/api/training-engine/record", headers=headers, json={
        "trainingId": training["trainingId"],
        "segId": item["segId"],
        "pick": item["yStar"],
    })
    assert recorded.status_code == 200, recorded.text
    finalized = client.post(
        "/api/training-sessions/finalize", headers=headers, json={
            "trainingId": training["trainingId"],
            "nItems": 1,
            "summary": {"percentile": {"forged": True}},
        })
    assert finalized.status_code == 200, finalized.text
    row = client.app.state.db.get_training_session(training["trainingId"])
    summary = json.loads(row["summary"])
    assert summary["percentile"]["status"] == "available"
    assert set(summary["percentile"]["domains"]) == set(DOMAINS)
    assert all(
        "forged" not in entry
        for entry in summary["percentile"]["domains"].values()
    )
