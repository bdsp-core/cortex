"""Recognition layer: per-domain certification badges + unannounced milestones.

Badges are credentials with teeth. A certification attempt that PASSes a
domain earns (or re-earns) its badge; a later attempt that FAILs a held
domain loses it (revoked_utc). PENDING/inconclusive verdicts never move a
badge in either direction. History is append-only: earned, lost, re-earned
each leave a row, and Settings shows the whole ledger.

Milestones are once-ever recognition moments awarded at irregular,
deliberately unannounced thresholds (the ladder is not shown anywhere in
the UI; the surprise IS the design, see the engagement notes in the repo
discussion). They surface exactly once as a dashboard banner (pending until
acknowledged), then live permanently in Settings.

Both evaluators run AFTER their triggering event has fully landed (result
ingest / training finalize), reference only work already done, and never
raise: recognition bookkeeping must not be able to fail a data write. No
evaluator runs on any exam-entry path; the exam surface stays free of
reward mechanics by design.
"""
from __future__ import annotations

import logging

from . import dashboard_logic

log = logging.getLogger("cortex.awards")

# Irregular ladders (roughly geometric): recognition should arrive on a
# heavy-tailed, hard-to-anticipate schedule, not every round number.
MILESTONE_SESSIONS = (1, 3, 7, 12, 19, 29, 41, 59, 83)
MILESTONE_DAYS = (5, 12, 30, 60)


def evaluate_certification(db, code: str, result: dict) -> None:
    """Badge award/revoke + certification milestones for one finalized
    attempt. Logs and swallows all failures."""
    try:
        tasks = dashboard_logic.dashboard_tasks(result)
        passes = 0
        for t in tasks:
            verdict = str(t.get("verdict") or "").upper()
            key = str(t.get("code") or t.get("taskK"))
            label = str(t.get("label") or key)
            if verdict == "PASS":
                passes += 1
                auroc = t.get("auroc")
                detail = (f"AUROC {auroc:.2f}"
                          if isinstance(auroc, (int, float)) else None)
                db.award_badge(code, key, label, detail)
            elif verdict == "FAIL":
                db.revoke_badge(code, key)
        if db.count_completed_results(code) == 1:
            db.award_milestone(code, "first-certification",
                               "First certification test",
                               "Your baseline measurement is on the books.")
        if tasks and passes == len(tasks):
            db.award_milestone(code, "full-seven", "Full seven-domain pass",
                               "Every domain cleared its bar in a single attempt.")
    except Exception:
        log.exception("[cortex.awards] certification evaluation failed for %s", code)


def evaluate_training(db, code: str) -> None:
    """Session-count + distinct-training-day milestones after a sitting
    finalizes. Logs and swallows all failures."""
    try:
        n = db.count_completed_trainings(code)
        if n in MILESTONE_SESSIONS:
            db.award_milestone(
                code, f"sessions-{n}",
                "First training session" if n == 1 else f"{n} training sessions",
                "The protocol begins." if n == 1
                else "Counted across your whole program.")
        d = db.count_training_days(code)
        if d in MILESTONE_DAYS:
            db.award_milestone(code, f"days-{d}", f"{d} training days",
                               f"You have now trained on {d} different days.")
    except Exception:
        log.exception("[cortex.awards] training evaluation failed for %s", code)
