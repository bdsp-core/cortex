"""R0 regression test — builds gold over a synthetic operational SQLite DB.

No PHI: a fabricated participant with a name + free-text institution lets us
assert the allowlist de-identification (those values must NEVER reach gold) and
the gold==blob + reconciliation gates, entirely offline / CI-safe.
"""
import json
import sqlite3

from .build_gold import (
    K, TASK_CODES, build_gold, read_operational, verify, _connect_ro,
)

NAME = "Jane Q. Testperson"
INSTITUTION = "Fictional Memorial Hospital"


def _diag(trial_index, task_k):
    """A per-question diag with the real K=7 array shape (incl. lSd)."""
    return {
        "trialIndex": trial_index, "taskK": task_k, "segId": 100 + trial_index,
        "s": 0.5, "sSd": 0.1, "y": trial_index % 2, "ess": 420.0, "rejuv": bool(trial_index % 2),
        "pi": [0.1 * k for k in range(K)], "mcse": [0.01] * K, "R": [0.3] * K,
        "lMean": [0.4 + 0.01 * k for k in range(K)], "tMean": [0.0] * K,
        "lSd": [0.12] * K, "nPerTask": [trial_index + 1] * K,
        "verdicts": ["PENDING"] * K,
    }


def _make_op_db(path):
    c = sqlite3.connect(path)
    c.executescript("""
      CREATE TABLE participants (code TEXT, password_hash TEXT, label TEXT,
        active INTEGER, created_utc TEXT, email TEXT, display_name TEXT,
        signup_ip TEXT, email_verified_utc TEXT, auth_provider TEXT, google_sub TEXT);
      CREATE TABLE sessions (session_id TEXT, code TEXT, participant TEXT,
        sample_seed BIGINT, started_utc TEXT, finished_utc TEXT, status TEXT,
        stop_reason TEXT, n_questions INTEGER);
      CREATE TABLE results (session_id TEXT, result TEXT, received_utc TEXT);
      CREATE TABLE trials (session_id TEXT, trial_index INTEGER, seg_id INTEGER,
        task_k INTEGER, pick INTEGER, is_correct INTEGER, reaction_ms REAL,
        diag TEXT, received_utc TEXT);
      CREATE TABLE training_sessions (training_id TEXT, code TEXT, task_focus TEXT,
        started_utc TEXT, finished_utc TEXT, status TEXT, n_items INTEGER, summary TEXT);
      CREATE TABLE param_trajectories (code TEXT, task_k INTEGER, phase TEXT,
        ell REAL, theta REAL, sd REAL, rt REAL, ts TEXT);
    """)
    participant_blob = {
        "name": NAME, "institution": INSTITUTION,
        "expertise": "epileptologist", "practice_setting": "academic",
        "race_ethnicity": "prefer_not", "eeg_volume_per_month": "50-100",
        "years_reading_eeg": "10+", "self_rated_confidence": 4, "country": "US",
        "sex": "female", "color_vision": "normal", "prior_test_taken": "no",
        "consent_version": "v1.0", "irb_protocol_id": "2016P000058",
    }
    c.execute("INSERT INTO participants(code,display_name,email,created_utc,auth_provider) "
              "VALUES (?,?,?,?,?)", ("u-test", NAME, "jane@example.org", "2026-06-10T00:00:00Z", "local"))
    n_trials = 4
    trials = [_diag(i, i % K) for i in range(n_trials)]
    result_blob = {
        "verdicts": ["PASS", "FAIL", "REFER_BORDERLINE", "PASS", "REFER_BORDERLINE",
                     "REFER_BORDERLINE", "REFER_BORDERLINE"],
        "sampleSeed": 123, "participant": participant_blob,
        "servedSegIds": [100 + i for i in range(n_trials)],
        # blob trials omit lSd (mirrors prod); table diag keeps it
        "trials": [{kk: vv for kk, vv in d.items() if kk != "lSd"} for d in trials],
    }
    c.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
              ("sess-1", "u-test", json.dumps(participant_blob), 123,
               "2026-06-11T00:00:00Z", "2026-06-11T00:30:00Z", "complete", "all_resolved", n_trials))
    c.execute("INSERT INTO results VALUES (?,?,?)", ("sess-1", json.dumps(result_blob), "2026-06-11T00:30:01Z"))
    for i, d in enumerate(trials):
        c.execute("INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?)",
                  ("sess-1", i, 100 + i, i % K, (i % K), i % 2, 1500.0 + i, json.dumps(d), "x"))
    c.commit()
    c.close()


def test_gold_build_and_gates(tmp_path):
    op = str(tmp_path / "op.sqlite")
    _make_op_db(op)
    handle = _connect_ro(op)
    raw = read_operational(handle)
    handle[1].close()

    salt = b"test-salt"
    gold = build_gold(raw, salt)
    report = verify(raw, gold, salt)

    # all R0 gates pass
    assert report["all_pass"], json.dumps(report, indent=2, default=str)
    assert report["checks"]["gold_verdicts_match_blob"]["pass"]
    assert report["checks"]["trial_blob_reconciliation"]["pass"]
    assert report["checks"]["anti_leak"]["pass"]

    # grains have the expected shape
    assert len(gold["dim_participant"]) == 1
    assert len(gold["fact_test_session"]) == 1
    assert len(gold["fact_test_task_outcome"]) == K          # one per task
    assert len(gold["fact_trial"]) == 4                      # one per trial
    assert len(gold["fact_trial_task"]) == 4 * K             # trial × task long

    # headline verdicts flattened faithfully
    verdict_by_k = {o["task_k"]: o["verdict"] for o in gold["fact_test_task_outcome"]}
    assert verdict_by_k[0] == "PASS" and verdict_by_k[1] == "FAIL"

    # lSd survives into gold from the TABLE diag (the blob omits it)
    assert any(r["ell_sd"] == 0.12 for r in gold["fact_trial_task"])

    # de-identification: the free-text identifiers appear NOWHERE in gold
    blob_text = json.dumps(gold, default=str).lower()
    assert NAME.lower() not in blob_text
    assert INSTITUTION.lower() not in blob_text
    # but the allowlisted demographic DID come through
    assert gold["dim_participant"][0]["expertise"] == "epileptologist"


def test_anti_leak_catches_injected_identifier(tmp_path):
    """Sanity: if an identifier were projected, the anti-leak assertion fails."""
    op = str(tmp_path / "op.sqlite")
    _make_op_db(op)
    handle = _connect_ro(op)
    raw = read_operational(handle)
    handle[1].close()
    salt = b"test-salt"
    gold = build_gold(raw, salt)
    gold["dim_participant"][0]["expertise"] = NAME   # inject a leak
    report = verify(raw, gold, salt)
    assert not report["checks"]["anti_leak"]["pass"]
