"""Admin reporting/analytics derived from stored data.

Split out of db.py: these are read-only aggregations over several tables
that interpret results (a verdict means "certified"), not persistence. Living
on Database forced a function-local `from .dashboard_logic import ...` inside
the query layer purely to dodge an import cycle — the giveaway that the
dependency pointed the wrong way. As module functions taking a db handle they
sit at the same layer as awards.py and digest.py, and the import is top-level.
"""
from __future__ import annotations

import json

from .dashboard_logic import is_certified_verdict


def training_monitor(db) -> dict:
    """Aggregate safety + volume telemetry for the deployed trainer. Volume
    metrics are SQL; the SAFETY metric — confirmed false-graduation (the
    trainer declared a domain mastered, but the participant's next fresh
    retest did NOT pass it) — is cross-referenced in Python by linking each
    training session's final per-task ℓ against ℓ* and the subsequent cert
    verdict. See docs/LIVE_TRAINER.md for the active trainer architecture."""
    def _n(sql: str) -> int:
        row = db._fetchone(sql)
        return int((row["n"] if row else 0) or 0)

    learners = _n("SELECT COUNT(DISTINCT code) AS n FROM param_trajectories "
                  "WHERE is_real=1 AND phase='train'")
    started = _n("SELECT COUNT(*) AS n FROM training_sessions")
    completed = _n("SELECT COUNT(*) AS n FROM training_sessions "
                   "WHERE status='complete'")
    trials = _n("SELECT COUNT(*) AS n FROM param_trajectories "
                "WHERE is_real=1 AND phase='train'")
    exposure = _n("SELECT COUNT(*) AS n FROM training_trials")
    per_domain = [
        {"taskK": int(r["task_k"]), "trials": int(r["n"])}
        for r in db._fetchall(
            "SELECT task_k, COUNT(*) AS n FROM param_trajectories "
            "WHERE is_real=1 AND phase='train' GROUP BY task_k ORDER BY task_k")]
    return {
        "learners": learners,
        "sessionsStarted": started,
        "sessionsCompleted": completed,
        "trainingTrials": trials,
        "exposureRows": exposure,
        "perDomainTrials": per_domain,
        **_graduation_safety(db),
    }

def _graduation_safety(db) -> dict:
    """Graduation count + CONFIRMED false-graduation rate (the SAP primary
    safety endpoint). A (participant, domain) 'graduated' if its final
    trained ℓ ≥ ℓ*; it's a 'confirmed retest' if a fresh cert finished after
    that training; a 'false graduation' if that retest's verdict ≠ PASS."""
    def _per_task(result: dict, task: int, field: str):
        for p in (result or {}).get("perTask") or []:
            if isinstance(p, dict) and p.get("taskK") == task:
                return p.get(field)
        return None

    rows = db._fetchall(
        "SELECT code, training_id, task_k, ell FROM param_trajectories "
        "WHERE is_real=1 AND phase='train' AND training_id IS NOT NULL "
        "AND ell IS NOT NULL ORDER BY ts")
    final_ell: dict = {}   # (code, training_id, task) -> final ℓ (ts-ordered, last wins)
    for r in rows:
        final_ell[(r["code"], r["training_id"], int(r["task_k"]))] = float(r["ell"])
    empty = {"graduatedDomains": 0, "confirmedRetests": 0,
             "falseGraduations": 0, "falseGraduationRate": None}
    if not final_ell:
        return empty
    tfin = {t["training_id"]: t.get("finished_utc")
            for t in db._fetchall(
                "SELECT training_id, finished_utc FROM training_sessions")}
    # Cert results per participant, ascending by finish time (ISO ⇒
    # sortable) — ONE query across all learner codes, not one per code
    # (the old per-code list_results_for_code loop was an N+1 that also
    # re-parsed every learner's full result blobs once per learner).
    codes = sorted({c for (c, _t, _k) in final_ell})
    ph = ",".join("?" * len(codes))
    results: dict = {}
    for r in db._fetchall(
            f"SELECT s.code AS code, s.finished_utc AS finished_utc, "
            f"r.result AS result FROM results r "
            f"JOIN sessions s ON s.session_id = r.session_id "
            f"WHERE s.code IN ({ph}) AND s.finished_utc IS NOT NULL "
            f"ORDER BY s.code, s.finished_utc",
            tuple(codes)):
        results.setdefault(r["code"], []).append(
            {"finished_utc": r["finished_utc"],
             "result": json.loads(r["result"])})
    graduated = confirmed = false_grad = 0
    for (code, tid, task), ell in final_ell.items():
        rs = results.get(code, [])
        # ℓ* is bundle-wide: take it from any of the participant's results
        ellstar = next((v for r in rs
                        if (v := _per_task(r["result"], task, "ellStar")) is not None), None)
        if ellstar is None or ell < float(ellstar):
            continue
        graduated += 1
        tf = tfin.get(tid)
        if not tf:
            continue
        nxt = next((r for r in rs if r["finished_utc"] > tf), None)
        if nxt is None:
            continue
        verdict = _per_task(nxt["result"], task, "verdict")
        if verdict is None:
            continue
        confirmed += 1
        if not is_certified_verdict(verdict):
            false_grad += 1
    return {
        "graduatedDomains": graduated,
        "confirmedRetests": confirmed,
        "falseGraduations": false_grad,
        "falseGraduationRate": (false_grad / confirmed) if confirmed else None,
    }
