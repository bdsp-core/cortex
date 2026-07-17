"""V4 shadow-tool test on a FABRICATED dev DB (new Phase-L2 schema).

Skipped in the minimal-CI environment (numpy/scipy are engine deps, not web
API deps); runs under the repo venv. Validates: schema round-trip through
the real Database class, §2a stream extraction, replay seeding, training
step-through with gate-weight readout, readiness output shape, and
determinism of the whole shadow run.
"""
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "services")))
sys.path.insert(0, _HERE)

from api.db import Database  # noqa: E402
import le_shadow  # noqa: E402

WORDS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]


def _manifest(n_per=6):
    segs, sid = [], 0
    for k, w in enumerate(WORDS):
        for i in range(n_per):
            sm = [-1.0] * 7
            sm[k] = 0.5 + 0.1 * i
            segs.append(dict(segId=sid, patternClass=w,
                             sMean=sm, sSd=[0.3] * 7))
            sid += 1
    return dict(taskCodes=["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
                taskPatternWords=WORDS, ellStar=[0.1] * 7, segments=segs)


def _fab_db(path):
    db = Database(path)
    db._write("INSERT INTO participants(code, password_hash, created_utc) "
              "VALUES (?,?,?)", ("p1", "x", "2026-07-16T08:00:00Z"))
    db._write("INSERT INTO sessions(session_id, code, participant, "
              "started_utc, finished_utc, status) VALUES (?,?,?,?,?,?)",
              ("s1", "p1", "", "2026-07-16T08:10:00Z",
               "2026-07-16T08:40:00Z", "complete"))
    rng = np.random.default_rng(5)
    for i in range(10):
        # Real pick coding: TASK-axis indices (n-way answers 1..6; the
        # binary task uses 0="yes" / 7=sentinel "no").
        k = 0 if i == 9 else int(rng.integers(1, 7))
        pick = (0 if i % 2 else 7) if k == 0 else int(rng.integers(1, 7))
        db.upsert_trial("s1", dict(
            trialIndex=i, segId=(k * 6) + (i % 6), taskK=k,
            pick=pick, isCorrect=bool(i % 2),
            reactionMs=900 + i,
            shownClientUtc=f"2026-07-16T08:1{i % 6}:00Z",
            answeredClientUtc=f"2026-07-16T08:1{i % 6}:02Z"))
    db._write("INSERT INTO training_sessions(training_id, code, started_utc, "
              "status) VALUES (?,?,?,?)",
              ("t1", "p1", "2026-07-16T09:00:00Z", "complete"))
    pts = []
    for i in range(8):
        k = 2 + (i % 2)                       # train two n-way tasks
        # unique segIds: training_trials is keyed (training_id, seg_id) —
        # one exposure row per segment per sitting
        pts.append(dict(taskK=k, segId=6 + i, ell=0.2, theta=0.0,
                        sd=0.3, rt=1200, seqInSession=i,
                        pick=i % 2, yStar=(i + 1) % 2, isCorrect=i % 2 == 1,
                        feedbackShown="Correct — This is X." if i % 2
                        else "Incorrect — This is not X.",
                        shownClientUtc=f"2026-07-16T09:0{i}:00Z",
                        answeredClientUtc=f"2026-07-16T09:0{i}:01Z"))
    db.record_training_progress("p1", "t1", pts)
    return db


def test_shadow_end_to_end(tmp_path):
    man_path = tmp_path / "manifest.json"
    json.dump(_manifest(), open(man_path, "w"))
    _fab_db(tmp_path / "web.db")

    rows = le_shadow.run(str(tmp_path / "web.db"), str(man_path),
                         le_shadow.DEFAULT_ARTIFACT,
                         str(tmp_path / "out.jsonl"))
    assert len(rows) == 1
    r = rows[0]
    assert r["code"] == "p1" and r["session_id"] == "s1"
    assert r["n_replayed"] == 10           # full contiguous picked prefix
    assert r["n_train"] == 8
    assert r["n_gate_weights"] == 8
    assert 0.0 <= r["mean_gate_weight"] <= 1.2   # Wilson gate range
    for key in ("readiness_at_seed", "readiness_after_training"):
        pm = r[key]
        assert set(pm) == {"sz", "lpd", "gpd", "lrda", "grda", "iic"}
        assert all(0.0 <= v <= 1.0 for v in pm.values())
    out = [json.loads(l) for l in open(tmp_path / "out.jsonl")]
    assert out == rows

    # determinism: the shadow is a pure function of DB + manifest + artifact
    rows2 = le_shadow.run(str(tmp_path / "web.db"), str(man_path),
                          le_shadow.DEFAULT_ARTIFACT, None)
    assert rows2 == rows


def test_stream_prefix_stops_at_gap(tmp_path):
    _fab_db(tmp_path / "w2.db")
    db = Database(tmp_path / "w2.db")
    # a pick-less row at index 10 ends the §2a-replayable prefix
    db.upsert_trial("s1", dict(trialIndex=10, segId=1, taskK=1, pick=None))
    db.upsert_trial("s1", dict(trialIndex=11, segId=2, taskK=1, pick=3))
    conn = le_shadow._connect_ro(str(tmp_path / "w2.db"))
    rows = conn.execute("SELECT * FROM trials WHERE session_id='s1' "
                        "ORDER BY trial_index").fetchall()
    stream = le_shadow.stream_from_trials(rows)
    assert len(stream) == 10
