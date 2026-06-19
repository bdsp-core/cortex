"""R1 regression tests — the de-identified export, entirely offline / CI-safe.

Verifies the R1 gate: reproducible checksums across two runs, cohort-aware
disclosure control (aggregate-only for small n, k-anon for larger n), codebook
coverage, and that no free-text identifier reaches any output file.
"""
import json
import sqlite3
from pathlib import Path

from research.gold.build_gold import K
from research.export.export_cli import export
from research.gold.build_gold import read_operational, _connect_ro

NAME_TPL = "Identifiable Person {i}"
INST_TPL = "Secret Clinic {i}"
EXPERTISE = ["epileptologist", "fellow", "neurologist"]


def _diag(ti):
    return {"trialIndex": ti, "taskK": ti % K, "segId": 100 + ti, "s": 0.5, "sSd": 0.1,
            "y": ti % 2, "ess": 400.0, "rejuv": False,
            "pi": [0.1] * K, "mcse": [0.01] * K, "R": [0.3] * K,
            "lMean": [0.4] * K, "tMean": [0.0] * K, "lSd": [0.12] * K,
            "nPerTask": [ti + 1] * K, "verdicts": ["PENDING"] * K}


def _make_op_db(path, n_participants, trials_per=3):
    c = sqlite3.connect(path)
    c.executescript("""
      CREATE TABLE participants (code TEXT, password_hash TEXT, label TEXT, active INTEGER,
        created_utc TEXT, email TEXT, display_name TEXT, signup_ip TEXT,
        email_verified_utc TEXT, auth_provider TEXT, google_sub TEXT);
      CREATE TABLE sessions (session_id TEXT, code TEXT, participant TEXT, sample_seed BIGINT,
        started_utc TEXT, finished_utc TEXT, status TEXT, stop_reason TEXT, n_questions INTEGER);
      CREATE TABLE results (session_id TEXT, result TEXT, received_utc TEXT);
      CREATE TABLE trials (session_id TEXT, trial_index INTEGER, seg_id INTEGER, task_k INTEGER,
        pick INTEGER, is_correct INTEGER, reaction_ms REAL, diag TEXT, received_utc TEXT);
      CREATE TABLE training_sessions (training_id TEXT, code TEXT, task_focus TEXT,
        started_utc TEXT, finished_utc TEXT, status TEXT, n_items INTEGER, summary TEXT);
      CREATE TABLE param_trajectories (code TEXT, task_k INTEGER, phase TEXT, ell REAL,
        theta REAL, sd REAL, rt REAL, ts TEXT);
    """)
    for i in range(n_participants):
        code, sid = f"u-{i}", f"sess-{i}"
        pblob = {"name": NAME_TPL.format(i=i), "institution": INST_TPL.format(i=i),
                 "expertise": EXPERTISE[i % len(EXPERTISE)], "practice_setting": "academic",
                 "race_ethnicity": "prefer_not", "eeg_volume_per_month": "50-100",
                 "years_reading_eeg": "10+", "self_rated_confidence": 4, "country": "US",
                 "sex": "female" if i % 2 else "male", "color_vision": "normal",
                 "prior_test_taken": "no", "consent_version": "v1.0", "irb_protocol_id": "2016P000058"}
        trials = [_diag(t) for t in range(trials_per)]
        rblob = {"verdicts": ["PASS", "FAIL", "REFER_BORDERLINE", "PASS", "REFER_BORDERLINE",
                              "REFER_BORDERLINE", "REFER_BORDERLINE"], "sampleSeed": i,
                 "participant": pblob, "servedSegIds": [100 + t for t in range(trials_per)],
                 "trials": [{kk: vv for kk, vv in d.items() if kk != "lSd"} for d in trials]}
        c.execute("INSERT INTO participants(code,display_name,email,created_utc,auth_provider) VALUES (?,?,?,?,?)",
                  (code, NAME_TPL.format(i=i), f"p{i}@ex.org", "2026-06-10T00:00:00Z", "local"))
        c.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
                  (sid, code, json.dumps(pblob), i, "2026-06-11T00:00:00Z", "2026-06-11T00:30:00Z",
                   "complete", "all_resolved", trials_per))
        c.execute("INSERT INTO results VALUES (?,?,?)", (sid, json.dumps(rblob), "x"))
        for t, d in enumerate(trials):
            c.execute("INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?)",
                      (sid, t, 100 + t, t % K, t % K, t % 2, 1500.0, json.dumps(d), "x"))
    c.commit()
    c.close()


def _run_export(tmp, n, outname):
    op = str(tmp / f"op-{outname}.sqlite")
    _make_op_db(op, n)
    handle = _connect_ro(op)
    raw = read_operational(handle)
    handle[1].close()
    out = tmp / outname
    manifest = export(raw, b"salt", out, "2026-06-19T00:00:00Z", aggregate_below=20, k=5)
    return out, manifest


def _all_text(outdir: Path) -> str:
    return "\n".join(p.read_text() for p in outdir.rglob("*") if p.is_file()).lower()


def test_small_cohort_is_aggregate_only_and_reproducible(tmp_path):
    out1, m1 = _run_export(tmp_path, 3, "exp1")
    out2, m2 = _run_export(tmp_path, 3, "exp2")

    # cohort-aware: small n -> aggregate only, NO individual quasi-identifiers
    assert m1["disclosure"]["mode"] == "aggregate_only"
    assert (out1 / "tables" / "demographic_marginals.csv").exists()
    assert not (out1 / "tables" / "dim_participant.csv").exists()
    # but per-participant performance buckets ARE published (pseudonymous)
    assert m1["per_participant_folders"] == 3
    folders = list((out1 / "per_participant").iterdir())
    assert (folders[0] / "testing.json").exists()
    assert not (folders[0] / "demographic.json").exists()

    # reproducible: identical data-file checksums across two runs
    assert m1["files"] == m2["files"]

    # codebook present + covers everything (export would have raised otherwise)
    assert (out1 / "codebook.json").exists()

    # anti-leak: no name / institution anywhere in the export
    text = _all_text(out1)
    for i in range(3):
        assert NAME_TPL.format(i=i).lower() not in text
        assert INST_TPL.format(i=i).lower() not in text
    # the marginal counts ARE present (expertise distribution)
    assert "epileptologist" in text


def test_large_cohort_is_k_anonymized(tmp_path):
    out, m = _run_export(tmp_path, 25, "exp")
    assert m["disclosure"]["mode"] == "k_anonymized"
    assert (out / "tables" / "dim_participant.csv").exists()
    assert not (out / "tables" / "demographic_marginals.csv").exists()
    # still no identifiers leaked
    text = _all_text(out)
    assert NAME_TPL.format(i=0).lower() not in text
