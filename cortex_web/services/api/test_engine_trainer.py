"""Phase L3 engine-trainer closed loop over HTTP (skipped in minimal CI —
numpy is an engine dep, not a web-API dep).

Covers: gating (default off), replay seeding from the cert sitting,
engine-served items answered by a simulated learner through the real
endpoints, no-repeat serving, snapshot sanity, ledger-driven rebuild
(restart recovery), and the engineMode flag on training-session start.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from fastapi.testclient import TestClient  # noqa: E402

from .app import create_app  # noqa: E402
from .test_server import (_auth_header, _make_participant,  # noqa: E402
                          _write_test_bank)


def _app(tmp_path, monkeypatch, engine_mode):
    monkeypatch.setenv("CORTEX_ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("CORTEX_JWT_SECRET", "test-secret")
    monkeypatch.setenv("CORTEX_EMAIL_BACKEND", "dev")
    monkeypatch.setenv("CORTEX_EMAIL_EXPOSE_CODE", "1")
    _write_test_bank(tmp_path / "bundle", "test-bank", per_class=8)
    monkeypatch.setenv("CORTEX_BUNDLE_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("CORTEX_BUNDLE_URL", "/bundle/test-bank")
    monkeypatch.setenv("CORTEX_SESSION_SAMPLE", "21")
    if engine_mode is None:
        monkeypatch.delenv("CORTEX_TRAINER_ENGINE", raising=False)
    else:
        monkeypatch.setenv("CORTEX_TRAINER_ENGINE", engine_mode)
    return TestClient(create_app(db_path=tmp_path / "t.db"))


@pytest.fixture()
def eclient(tmp_path, monkeypatch):
    """The standard test app with the engine trainer switched ON."""
    return _app(tmp_path, monkeypatch, "all")


@pytest.fixture()
def offclient(tmp_path, monkeypatch):
    """The kill switch: CORTEX_TRAINER_ENGINE=off (the archived incumbent
    fallback; since the 2026-07-17 integration the DEFAULT is 'all')."""
    return _app(tmp_path, monkeypatch, "off")


def _cert_with_stream(client, hdr, n_trials=8):
    d = client.post("/api/session", json={}, headers=hdr).json()
    sid = d["sessionId"]
    segs = [s["segId"] for s in d["bank"]["segments"]]
    rng = np.random.default_rng(3)
    for i in range(n_trials):
        # Real pick coding: 0-based TASK-axis index; n-way answers 1..6;
        # the binary task 0 answers pick=0 ("yes") / pick=7 (sentinel "no").
        task = 0 if i == 0 else 1 + (i % 6)
        pick = (0 if rng.random() < 0.5 else 7) if task == 0 \
            else int(rng.integers(1, 7))
        client.post("/api/progress", headers=hdr, json={
            "sessionId": sid,
            "trial": {"trialIndex": i, "segId": segs[i],
                      "taskK": task, "pick": pick,
                      "isCorrect": bool(i % 2), "reactionMs": 900}})
    verdicts = ["PASS"] * 7
    verdicts[2] = "FAIL"
    result = {"verdicts": verdicts,
              "perTask": [{"taskK": k, "ellStar": 0.0, "verdict": v,
                           "ell": 0.2, "theta": 0.0}
                          for k, v in enumerate(verdicts)]}
    client.post("/api/results", headers=hdr,
                json={"sessionId": sid, "result": result,
                      "nQuestions": n_trials})
    return sid


def test_engine_gated_off_by_killswitch(offclient):
    """CORTEX_TRAINER_ENGINE=off must 403 and report engineMode=False —
    the zero-deploy reversion path to the archived incumbent trainer."""
    email, pw = _make_participant(offclient)
    hdr = _auth_header(offclient, email, pw)
    r = offclient.post("/api/training-sessions", headers=hdr,
                       json={"taskFocus": None}).json()
    assert r["engineMode"] is False
    assert offclient.post("/api/training-engine/start", headers=hdr,
                          json={"trainingId": r["trainingId"],
                                "segIds": []}).status_code == 403


def test_engine_closed_loop_and_rebuild(eclient):
    email, pw = _make_participant(eclient)
    hdr = _auth_header(eclient, email, pw)
    _cert_with_stream(eclient, hdr)

    r = eclient.post("/api/training-sessions", headers=hdr,
                     json={"taskFocus": None}).json()
    tid = r["trainingId"]
    assert r["engineMode"] is True
    pool = eclient.post("/api/training-bank", json={}, headers=hdr).json()
    seg_ids = [s["segId"] for s in pool["bank"]["segments"]]

    st = eclient.post("/api/training-engine/start", headers=hdr,
                      json={"trainingId": tid, "segIds": seg_ids})
    assert st.status_code == 200, st.text
    st = st.json()
    assert st["seeded"] == 8            # the cert stream replayed
    assert len(st["snapshot"]) == 7
    # attainability is a REPORT (7 tasks, probabilities); practice mode
    # serves regardless of it
    assert set(st["attainability"].keys()) == {
        "spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"}
    assert all(0.0 <= v <= 1.0 for v in st["attainability"].values())
    # depletion guard: the expanded seeding cloud must keep real ancestor
    # diversity after contraction (raw 400-particle replay collapsed to a
    # handful — the measured live failure)
    assert st["seedUnique"] > 40
    item = st["item"]
    assert item is not None and item["segId"] in set(seg_ids)

    rng = np.random.default_rng(9)
    served, links, golds = [], [], []
    for i in range(25):
        if item is None:
            break
        golds.append(item["yStar"])
        # simulated learner: 80% agreement with the item's gold label,
        # respecting the link's pick coding (binary 0/1; n-way = 0-based
        # TASK-axis index in 1..6, the exam's coding)
        if item.get("link") == "nway":
            pick = int(item["yStar"] if rng.random() < 0.8
                       else (item["yStar"] % 6) + 1)
        else:
            pick = int(item["yStar"] if rng.random() < 0.8
                       else 1 - item["yStar"])
        served.append(item["segId"])
        links.append(item.get("link", "binary"))
        # the client-side ledger write (the real client's outbox path)
        assert eclient.post("/api/training-progress", headers=hdr, json={
            "trainingId": tid, "points": [{
                "taskK": item["task"], "segId": item["segId"],
                "ell": 0.0, "theta": 0.0, "sd": 0.3, "rt": 800,
                "seqInSession": i, "pick": pick, "yStar": item["yStar"],
                "isCorrect": pick == item["yStar"],
                "mode": item["mode"], "link": item.get("link", "binary"),
            }]}).status_code == 200
        rec = eclient.post("/api/training-engine/record", headers=hdr,
                           json={"trainingId": tid, "segId": item["segId"],
                                 "taskK": item["task"], "pick": pick})
        assert rec.status_code == 200, rec.text
        rec = rec.json()
        for s in rec["snapshot"]:
            assert 0.0 <= s["passMass"] <= 1.0
            assert np.isfinite([s["skill"], s["theta"], s["sd"]]).all()
        item = rec["item"]
    n_answered = len(served)
    assert n_answered >= 10
    assert len(set(served)) == n_answered      # no repeats within sitting
    assert "nway" in links                     # native n-way serving live
    # D58 anti-exploit gate: within served n-way items the GOLD class must
    # vary (true-class-only pools made the correct button constant within
    # a domain run — exploited live at 36/36 @ 368 ms)
    nway_golds = [g for g, l in zip(golds, links) if l == "nway"]
    if len(nway_golds) >= 6:
        assert len(set(nway_golds)) > 1, \
            f"n-way golds constant — exploitable: {nway_golds}"
    # ledger metadata landed (L4): mode/link per row, engine_meta per sitting
    db = eclient.app.state.db
    rows = db._fetchall("SELECT mode, link FROM training_trials "
                        "WHERE training_id=?", (tid,))
    assert all(r["mode"] in ("skill", "bias", "review") for r in rows)
    assert {r["link"] for r in rows} <= {"binary", "nway"}
    meta = db._fetchone("SELECT engine_meta FROM training_sessions "
                        "WHERE training_id=?", (tid,))
    import json as _json
    meta = _json.loads(meta["engine_meta"])
    assert "attainability" in meta and meta["nway"] is True

    # restart recovery: re-start rebuilds belief + served set from the
    # ledger; the next item never repeats an already-served segment
    st2 = eclient.post("/api/training-engine/start", headers=hdr,
                       json={"trainingId": tid, "segIds": seg_ids}).json()
    assert st2["rebuiltSeq"] == n_answered
    if st2["item"] is not None:
        assert st2["item"]["segId"] not in set(served)

    # recording a never-served segment is rejected (no pending item)
    r = eclient.post("/api/training-engine/record", headers=hdr,
                     json={"trainingId": tid, "segId": 999_999,
                           "taskK": 1, "pick": 1})
    assert r.status_code == 409


def test_retention_and_interleave(eclient, monkeypatch):
    """L4 serving wrappers: with an immediate retention interval, review
    items fire on the REVIEW_EVERY cadence; with a small MAX_CONSEC, no
    domain run exceeds the cap while alternatives exist."""
    from . import engine_trainer as et
    monkeypatch.setattr(et, "RETENTION_FIRST_S", 0.0)
    monkeypatch.setattr(et, "REVIEW_EVERY", 5)
    monkeypatch.setattr(et, "REVIEW_MAX", 3)
    monkeypatch.setattr(et, "MAX_CONSEC", 3)
    email, pw = _make_participant(eclient)
    hdr = _auth_header(eclient, email, pw)
    _cert_with_stream(eclient, hdr)
    tid = eclient.post("/api/training-sessions", headers=hdr,
                       json={"taskFocus": None}).json()["trainingId"]
    pool = eclient.post("/api/training-bank", json={}, headers=hdr).json()
    seg_ids = [s["segId"] for s in pool["bank"]["segments"]]
    item = eclient.post("/api/training-engine/start", headers=hdr,
                        json={"trainingId": tid,
                              "segIds": seg_ids}).json()["item"]
    rng = np.random.default_rng(2)
    tasks, modes = [], []
    for _ in range(20):
        if item is None:
            break
        pick = (int(rng.integers(1, 7)) if item.get("link") == "nway"
                else int(rng.integers(0, 2)))
        tasks.append(item["task"])
        modes.append(item["mode"])
        rec = eclient.post("/api/training-engine/record", headers=hdr,
                           json={"trainingId": tid, "segId": item["segId"],
                                 "taskK": item["task"], "pick": pick}).json()
        item = rec["item"]
    assert "review" in modes, f"no review fired: {modes}"
    run, worst = 1, 1
    for a, b in zip(tasks, tasks[1:]):
        run = run + 1 if a == b else 1
        worst = max(worst, run)
    assert worst <= 3, f"domain run {worst} exceeds MAX_CONSEC: {tasks}"
    # cross-sitting retention state persisted
    db = eclient.app.state.db
    assert db.get_retention(db.training_session_owner(tid))


def test_burst_flagging_at_ingest(eclient):
    """L4 data hygiene: a response and its predecessor both sub-500 ms
    -> 'burst'; isolated fast or normal responses stay unflagged."""
    email, pw = _make_participant(eclient)
    hdr = _auth_header(eclient, email, pw)
    tid = eclient.post("/api/training-sessions", headers=hdr,
                       json={"taskFocus": None}).json()["trainingId"]
    rts = [900, 300, 280, 250, 800, 300]
    pts = [{"taskK": 1, "segId": 10 + i, "ell": 0, "theta": 0, "sd": 0.3,
            "rt": rt, "rtMs": rt, "seqInSession": i, "pick": 1, "yStar": 1}
           for i, rt in enumerate(rts)]
    assert eclient.post("/api/training-progress", headers=hdr,
                        json={"trainingId": tid,
                              "points": pts}).status_code == 200
    rows = eclient.app.state.db._fetchall(
        "SELECT seq_in_session, quality_flag FROM training_trials "
        "WHERE training_id=? ORDER BY seq_in_session", (tid,))
    flags = [r["quality_flag"] for r in rows]
    assert flags == [None, None, "burst", "burst", None, None]
