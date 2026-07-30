"""CORTEX gold analytics ETL — Phase R0 (DESIGN-driven, build-only).

READ-ONLY over the operational database (CORTEX_DB). Flattens the opaque JSON
blobs — `results.result`, `trials.diag`, `sessions.participant` — into tidy,
per-grain "gold" tables written to a SEPARATE, rebuildable SQLite file. It never
touches the operational write path (SELECTs only) so it cannot destabilize the
live service.

De-identification is by ALLOWLIST: free-text identifiers (`name`, `institution`,
`display_name`, `email`, IPs) are NEVER selected into gold. A hard anti-leak
assertion additionally fails the build if any identifier value appears in any
gold cell.

Grounded in the REAL prod blob shapes (verified 2026-06-19), which differ from
the engine source in two ways the plan got wrong:
  * `trials.diag` (the table) carries `lSd` per question — it is NOT lost; only
    the `results.result` blob omits it. The table is the richer source.
  * No AUROC is stored anywhere; `final_auroc` requires an engine re-run and is
    left NULL here (Phase O3/reproducibility round-trip enriches it).

Usage:
    CORTEX_DB=postgresql://... GOLD_HMAC_SALT=<secret> \
        python -m research.gold.build_gold --out /tmp/gold.sqlite
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sqlite3
from typing import Any, Optional

# ── task metadata (K=7 Full-7, D5 order) ──────────────────────────────
TASK_CODES = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
TASK_LABELS = ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
K = len(TASK_CODES)

# ── de-identification policy (verified against real participant blobs) ─
# Quasi-identifier / non-identifying fields projected into gold:
DEMOGRAPHIC_ALLOWLIST = [
    "expertise", "practice_setting", "eeg_volume_per_month",
    "years_reading_eeg", "self_rated_confidence", "country", "sex",
    "color_vision", "prior_test_taken",
]
CONSENT_FIELDS = ["consent_version", "irb_protocol_id"]
# NEVER projected; used only to source the anti-leak assertion:
IDENTIFIER_FIELDS = ["name", "institution"]


# ── read-only operational DB access (backend-agnostic) ────────────────
def _connect_ro(cortex_db: str):
    """Open the operational DB read-only. Postgres URL → psycopg; else SQLite."""
    if cortex_db.startswith(("postgres://", "postgresql://")):
        import psycopg
        from psycopg.rows import dict_row
        return ("pg", psycopg.connect(cortex_db, autocommit=True, row_factory=dict_row))
    conn = sqlite3.connect(cortex_db)
    conn.row_factory = sqlite3.Row
    return ("sqlite", conn)


def _rows(handle, sql: str) -> list[dict]:
    kind, conn = handle
    cur = conn.cursor()
    cur.execute(sql)
    out = [dict(r) for r in cur.fetchall()]
    cur.close()
    return out


def _loads(blob: Any) -> Any:
    if blob is None:
        return None
    if isinstance(blob, (dict, list)):
        return blob              # psycopg may already decode jsonb
    return json.loads(blob)


# ── pseudonymisation ──────────────────────────────────────────────────
def participant_sk(code: str, salt: bytes) -> str:
    """Deterministic, un-invertible pseudonym (stable across releases)."""
    return hmac.new(salt, code.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


# ── extract ───────────────────────────────────────────────────────────
def read_operational(handle) -> dict[str, Any]:
    participants = _rows(handle, "SELECT * FROM participants")
    sessions = _rows(handle, "SELECT * FROM sessions")
    results = {r["session_id"]: _loads(r["result"])
               for r in _rows(handle, "SELECT session_id, result FROM results")}
    trials = _rows(handle, "SELECT * FROM trials ORDER BY session_id, trial_index")
    training = _rows(handle, "SELECT * FROM training_sessions")
    traj = _rows(handle, "SELECT * FROM param_trajectories")
    trials_by_session: dict[str, list[dict]] = {}
    for t in trials:
        trials_by_session.setdefault(t["session_id"], []).append(t)
    return {
        "participants": participants,
        "sessions": sessions,
        "results": results,
        "trials_by_session": trials_by_session,
        "training": training,
        "traj": traj,
    }


# ── transform ─────────────────────────────────────────────────────────
def _latest_participant_blob(code: str, raw: dict) -> dict:
    """Most-recent completed session's result.participant for this code, {} if none."""
    sess = sorted(
        (s for s in raw["sessions"]
         if s["code"] == code and s.get("status") == "complete"),
        key=lambda s: s.get("finished_utc") or s.get("started_utc") or "",
        reverse=True,
    )
    for s in sess:
        res = raw["results"].get(s["session_id"]) or {}
        p = res.get("participant") or {}
        if p:
            return p
    return {}


def build_gold(raw: dict, salt: bytes) -> dict[str, list[dict]]:
    sk = {p["code"]: participant_sk(p["code"], salt) for p in raw["participants"]}
    complete = [s for s in raw["sessions"] if s.get("status") == "complete"]

    # dim_participant + dim_consent (allowlist projection only)
    dim_participant, dim_consent = [], []
    for p in raw["participants"]:
        code = p["code"]
        blob = _latest_participant_blob(code, raw)
        n_test = sum(1 for s in complete if s["code"] == code)
        dim_participant.append({
            "participant_sk": sk[code],
            "created_month": (p.get("created_utc") or "")[:7],   # generalise to month
            "auth_provider": p.get("auth_provider") or "local",
            "n_test_sessions": n_test,
            **{f: blob.get(f) for f in DEMOGRAPHIC_ALLOWLIST},
        })
        dim_consent.append({
            "participant_sk": sk[code],
            **{f: blob.get(f) for f in CONSENT_FIELDS},
            "consent_utc": None,    # not captured operationally yet (Phase O1)
        })

    # fact_test_session
    fact_session = []
    for s in complete:
        fact_session.append({
            "session_sk": s["session_id"],
            "participant_sk": sk.get(s["code"]),
            "started_utc": s.get("started_utc"),
            "finished_utc": s.get("finished_utc"),
            "stop_reason": s.get("stop_reason"),
            "n_questions": s.get("n_questions"),
            "sample_seed": s.get("sample_seed"),
            "provenance_inferred": True,   # no version stamping yet (Phase O3)
            "percentile_norm_id": s.get("norm_id"),
            "percentile_norm_sha256": s.get("norm_sha256"),
            "percentile_score_schema_version": s.get("score_schema_version"),
        })

    # fact_trial (trial grain) + fact_trial_task (trial×task long, SMC evolution)
    #   + fact_test_task_outcome (one row per session×task — the headline grain)
    fact_trial, fact_trial_task, fact_outcome = [], [], []
    for s in complete:
        sid = s["session_id"]
        res = raw["results"].get(sid) or {}
        verdicts = res.get("verdicts") or []
        percentile_report = res.get("percentile") or {}
        percentile_profile = (
            percentile_report.get("profile")
            if isinstance(percentile_report, dict) else {}
        ) or {}
        percentile_domains = (
            percentile_report.get("domains")
            if isinstance(percentile_report, dict)
            and percentile_report.get("status") == "available"
            else {}
        ) or {}
        tbl = raw["trials_by_session"].get(sid, [])
        # per-task accumulators for binarized correctness
        correct_n = [0] * K
        correct_d = [0] * K
        last_diag: Optional[dict] = None
        for t in tbl:
            diag = _loads(t.get("diag")) or {}
            if diag:
                last_diag = diag
            tk = t.get("task_k")
            is_corr = t.get("is_correct")
            if tk is not None and 0 <= tk < K and is_corr is not None:
                correct_d[tk] += 1
                correct_n[tk] += int(is_corr)
            fact_trial.append({
                "session_sk": sid,
                "trial_index": t.get("trial_index"),
                "asked_task_k": tk,
                "seg_id": t.get("seg_id"),
                "pick": t.get("pick"),
                "response_y": diag.get("y"),
                "binarized_correct": (int(is_corr) if is_corr is not None else None),
                "reaction_ms": t.get("reaction_ms"),
                "ess": diag.get("ess"),
                "rejuvenated": (1 if diag.get("rejuv") else 0) if "rejuv" in diag else None,
            })
            # explode per-task SMC posteriors to long
            for k in range(K):
                def at(name):
                    arr = diag.get(name)
                    return arr[k] if isinstance(arr, list) and k < len(arr) else None
                fact_trial_task.append({
                    "session_sk": sid,
                    "trial_index": t.get("trial_index"),
                    "task_k": k,
                    "task_code": TASK_CODES[k],
                    "pass_mass": at("pi"),
                    "mcse": at("mcse"),
                    "info_gate_R": at("R"),
                    "ell_mean": at("lMean"),
                    "theta_mean": at("tMean"),
                    "ell_sd": at("lSd"),       # present in the TABLE diag (not the blob)
                    "n_items_task": at("nPerTask"),
                    "verdict": at("verdicts"),
                })
        # headline per-task outcome from the final cloud (last trial's arrays)
        for k in range(K):
            def final(name):
                arr = (last_diag or {}).get(name)
                return arr[k] if isinstance(arr, list) and k < len(arr) else None
            fact_outcome.append({
                "session_sk": sid,
                "participant_sk": sk.get(s["code"]),
                "task_k": k,
                "task_code": TASK_CODES[k],
                "verdict": verdicts[k] if k < len(verdicts) else None,
                "final_pass_mass": final("pi"),
                "final_ell_mean": final("lMean"),
                "final_theta_mean": final("tMean"),
                "final_ell_sd": final("lSd"),
                "n_items_task": final("nPerTask"),
                "binarized_correct_rate": (correct_n[k] / correct_d[k]) if correct_d[k] else None,
                "final_auroc": None,   # not stored anywhere — needs engine re-run (Phase O3)
                "ell_star": None,      # from the bundle manifest — deferred enrichment
                "percentile_status": (
                    percentile_report.get("status")
                    if isinstance(percentile_report, dict) else None
                ),
                "percentile_estimate": (
                    percentile_domains.get(TASK_CODES[k], {}).get("estimate")
                ),
                "percentile_lower_95": (
                    percentile_domains.get(TASK_CODES[k], {}).get("lower")
                ),
                "percentile_upper_95": (
                    percentile_domains.get(TASK_CODES[k], {}).get("upper")
                ),
                "percentile_norm_id": percentile_profile.get("normId"),
                "percentile_norm_sha256": percentile_profile.get("normSha256"),
                "percentile_score_schema_version": percentile_profile.get(
                    "scoreSchemaVersion"),
            })

    # fact_training_session + fact_trajectory_point (empty today; is_real guard)
    fact_training = []
    for tr in raw["training"]:
        summary = _loads(tr.get("summary")) or {}
        percentile = summary.get("percentile") or {}
        fact_training.append({
            "training_sk": tr["training_id"],
            "participant_sk": sk.get(tr["code"]),
            "started_utc": tr.get("started_utc"),
            "finished_utc": tr.get("finished_utc"),
            "n_items": tr.get("n_items"),
            "percentile_status": (
                percentile.get("status")
                if isinstance(percentile, dict) else None
            ),
            "percentile_norm_id": tr.get("norm_id"),
            "percentile_norm_sha256": tr.get("norm_sha256"),
            "percentile_score_schema_version": tr.get(
                "score_schema_version"),
        })
    fact_trajectory = [{
        "participant_sk": sk.get(p["code"]), "task_k": p.get("task_k"),
        "phase": p.get("phase"), "ell": p.get("ell"), "theta": p.get("theta"),
        "sd": p.get("sd"), "rt": p.get("rt"), "ts": p.get("ts"),
        "is_real": int(p.get("is_real") or 0),   # realness derived ONLY from the column (Phase O2)
    } for p in raw["traj"]]

    # bridge_participant_journey (one row per participant)
    bridge = []
    for p in raw["participants"]:
        code = p["code"]
        psk = sk[code]
        my_sessions = [f for f in fact_session if f["participant_sk"] == psk]
        bridge.append({
            "participant_sk": psk,
            "n_test_sessions": len(my_sessions),
            "n_training_sessions": sum(1 for f in fact_training if f["participant_sk"] == psk),
            "n_trajectory_points": sum(1 for f in fact_trajectory if f["participant_sk"] == psk),
            "first_test_utc": min((f["started_utc"] for f in my_sessions), default=None),
            "last_test_utc": max((f["finished_utc"] or f["started_utc"] for f in my_sessions), default=None),
        })

    return {
        "dim_participant": dim_participant,
        "dim_consent": dim_consent,
        "fact_test_session": fact_session,
        "fact_test_task_outcome": fact_outcome,
        "fact_trial": fact_trial,
        "fact_trial_task": fact_trial_task,
        "fact_training_session": fact_training,
        "fact_trajectory_point": fact_trajectory,
        "bridge_participant_journey": bridge,
    }


# ── load (rebuildable SQLite, isolated from prod) ─────────────────────
def write_sqlite(gold: dict[str, list[dict]], path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        for table, rows in gold.items():
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            if not rows:
                continue
            cols = list(rows[0].keys())
            conn.execute(f"CREATE TABLE {table} ({', '.join(c + ' ' for c in cols)})")
            ph = ", ".join("?" for _ in cols)
            conn.executemany(
                f"INSERT INTO {table} VALUES ({ph})",
                [tuple(r.get(c) for c in cols) for r in rows],
            )
        conn.commit()
    finally:
        conn.close()


# ── verify (the R0 gate) ──────────────────────────────────────────────
def verify(raw: dict, gold: dict, salt: bytes) -> dict[str, Any]:
    report: dict[str, Any] = {"checks": {}, "row_counts": {t: len(r) for t, r in gold.items()}}

    # 1) gold == blob: per-(latest session, task) verdict matches result.verdicts
    sk = {p["code"]: participant_sk(p["code"], salt) for p in raw["participants"]}
    mism = 0
    out_by_session: dict[str, dict[int, str]] = {}
    for o in gold["fact_test_task_outcome"]:
        out_by_session.setdefault(o["session_sk"], {})[o["task_k"]] = o["verdict"]
    for sid, res in raw["results"].items():
        v = (res or {}).get("verdicts") or []
        for k, want in enumerate(v):
            got = out_by_session.get(sid, {}).get(k)
            if got != want:
                mism += 1
    report["checks"]["gold_verdicts_match_blob"] = {"mismatches": mism, "pass": mism == 0}

    # 2) trial-vs-blob reconciliation: table trial count == blob trial count
    recon = []
    for s in raw["sessions"]:
        if s.get("status") != "complete":
            continue
        sid = s["session_id"]
        tbl = len(raw["trials_by_session"].get(sid, []))
        blob = res = raw["results"].get(sid) or {}
        blob_n = len(res.get("trials") or [])
        recon.append({"session": sid, "table": tbl, "blob": blob_n, "match": tbl == blob_n})
    report["checks"]["trial_blob_reconciliation"] = {
        "per_session": recon,
        "pass": all(r["match"] or (r["table"] == 0 and r["blob"] == 0) for r in recon),
    }

    # 3) anti-leak: no identifier value appears in any gold cell
    identifiers: set[str] = set()
    for s in raw["sessions"]:
        p = _loads(s.get("participant")) or {}
        for f in IDENTIFIER_FIELDS:
            if p.get(f):
                identifiers.add(str(p[f]).strip().lower())
    for pp in raw["participants"]:
        if pp.get("display_name"):
            identifiers.add(str(pp["display_name"]).strip().lower())
        if pp.get("email"):
            identifiers.add(str(pp["email"]).split("@")[0].strip().lower())
    identifiers.discard("")
    leaks = []
    for table, rows in gold.items():
        for r in rows:
            for col, val in r.items():
                if isinstance(val, str) and val.strip().lower() in identifiers:
                    leaks.append({"table": table, "col": col})
    report["checks"]["anti_leak"] = {
        "n_identifier_values": len(identifiers), "leaks": leaks[:10], "pass": not leaks,
    }

    report["all_pass"] = all(c["pass"] for c in report["checks"].values())
    return report


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CORTEX gold ETL (R0)")
    ap.add_argument("--db", default=os.environ.get("CORTEX_DB", ""),
                    help="operational DB (CORTEX_DB): postgres URL or sqlite path")
    ap.add_argument("--out", default=None, help="gold SQLite output path")
    ap.add_argument("--salt", default=os.environ.get("GOLD_HMAC_SALT", "DEV-INSECURE-SALT"),
                    help="HMAC salt for participant pseudonyms (seal in prod)")
    args = ap.parse_args(argv)
    if not args.db:
        ap.error("no DB: set CORTEX_DB or pass --db")

    handle = _connect_ro(args.db)
    try:
        raw = read_operational(handle)
    finally:
        handle[1].close()

    salt = args.salt.encode("utf-8")
    gold = build_gold(raw, salt)
    if args.out:
        write_sqlite(gold, args.out)
    report = verify(raw, gold, salt)
    if args.out:
        report["written_to"] = args.out
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
