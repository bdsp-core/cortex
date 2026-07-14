"""Recognition layer: per-domain certification badges + milestones.

Badges are credentials with teeth. A certification attempt that PASSes a
domain earns (or re-earns) its badge; a later attempt that FAILs a held
domain loses it (revoked_utc). PENDING/inconclusive verdicts never move a
badge in either direction. History is append-only: earned, lost, re-earned
each leave a row, and Settings shows the whole ledger. The seven domain
badges ARE the "domain passed" recognitions.

Milestones are once-ever, permanent recognition moments from a fixed
official set (below). They surface exactly once as a dashboard banner
(pending until acknowledged), then live permanently in Settings. The set is
deliberately small so recognition stays meaningful rather than constant:

  * account            Account created
  * first-certification / cert-{5,10,50,100}     certification-count
  * first-training / sessions-{10,50,100,314,500,1000,10000}  session-count
  * streak-{week,month,year}   consecutive train-or-test day streaks (7/30/365)

Evaluators run AFTER their triggering event has fully landed (account
create / result ingest / training finalize), reference only work already
done, and never raise: recognition bookkeeping must not be able to fail a
data write. No evaluator runs on any exam-entry path; the exam surface stays
free of reward mechanics by design.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from . import dashboard_logic

log = logging.getLogger("cortex.awards")

# Milestone copy is deliberately dry and a little funny: certification is a long
# haul, and a reason to smirk under a surgical mask now and then is good for
# morale. Keep it short, one sentence, no em dashes (house style).

# Certification-count milestones (count -> (key, label, detail)); count 1 is
# the separate first-certification below.
MILESTONE_CERTS = {
    5:   ("cert-5",   "5th certification test",   "Five tests in, and that twitch is either mastery or muscle artifact."),
    10:  ("cert-10",  "10th certification test",  "Double digits. You now dream in the 10-20 system."),
    50:  ("cert-50",  "50th certification test",  "Fifty tests. You've seen more spikes than a hedgehog convention."),
    100: ("cert-100", "100th certification test", "One hundred tests. The EEG machine now asks you for a second opinion."),
}

# Training-session-count milestones (count -> (key, label, detail)); count 1 is
# the separate first-training below.
MILESTONE_SESSIONS = {
    10:    ("sessions-10",    "10th training session",    "Ten sessions in, and the squiggles are starting to make sense. Suspicious."),
    50:    ("sessions-50",    "50th training session",    "Fifty sessions. You could read these in your sleep, and probably have."),
    100:   ("sessions-100",   "100th training session",   "One hundred sessions: generalized, rhythmic, and relentless."),
    314:   ("sessions-314",   "314th training session",   "Session 314, as irrational as it is delicious. Have a slice."),
    500:   ("sessions-500",   "500th training session",   "Five hundred sessions. Your baseline is everyone else's ceiling."),
    1000:  ("sessions-1000",  "1,000th training session", "A thousand sessions. Even the artifact has stopped trying to fool you."),
    10000: ("sessions-10000", "10,000th training session","Ten thousand sessions. Officially an expert, per that book everyone cites."),
}

# Consecutive train-or-test day streak milestones (threshold days, high→low).
STREAK_MILESTONES = (
    (365, "streak-year",  "First year streak",  "A full year, no gaps. A truly continuous recording."),
    (30,  "streak-month", "First month streak", "Thirty days unbroken, steadier than the hospital Wi-Fi."),
    (7,   "streak-week",  "First week streak",  "Seven days straight, more regular than most rhythms you read."),
)


def evaluate_account_created(db, code: str) -> None:
    """The "Account created" milestone, at signup (local + OAuth). Idempotent;
    logs and swallows all failures so it can never break registration."""
    try:
        db.award_milestone(code, "account", "Account created",
                           "Welcome aboard. The squiggles have been expecting you.")
    except Exception:
        log.exception("[cortex.awards] account milestone failed for %s", code)


def evaluate_certification(db, code: str, result: dict) -> None:
    """Badge award/revoke + certification-count + streak milestones for one
    finalized attempt. Logs and swallows all failures."""
    try:
        for t in dashboard_logic.dashboard_tasks(result):
            verdict = str(t.get("verdict") or "").upper()
            key = str(t.get("code") or t.get("taskK"))
            label = str(t.get("label") or key)
            if verdict == "PASS":
                auroc = t.get("auroc")
                detail = (f"AUROC {auroc:.2f}"
                          if isinstance(auroc, (int, float)) else None)
                db.award_badge(code, key, label, detail)
            elif verdict == "FAIL":
                db.revoke_badge(code, key)
        n = db.count_completed_results(code)
        if n == 1:
            db.award_milestone(code, "first-certification", "First certification test",
                               "A baseline at last. Even attendings once mistook a blink for a spike.")
        elif n in MILESTONE_CERTS:
            mkey, mlabel, mdetail = MILESTONE_CERTS[n]
            db.award_milestone(code, mkey, mlabel, mdetail)
        _evaluate_streak(db, code)
    except Exception:
        log.exception("[cortex.awards] certification evaluation failed for %s", code)


def evaluate_training(db, code: str) -> None:
    """Training-session-count + streak milestones after a sitting finalizes.
    Logs and swallows all failures."""
    try:
        n = db.count_completed_trainings(code)
        if n == 1:
            db.award_milestone(code, "first-training", "First training session",
                               "First rep down. Somewhere, your hippocampus is already forgetting this.")
        elif n in MILESTONE_SESSIONS:
            mkey, mlabel, mdetail = MILESTONE_SESSIONS[n]
            db.award_milestone(code, mkey, mlabel, mdetail)
        _evaluate_streak(db, code)
    except Exception:
        log.exception("[cortex.awards] training evaluation failed for %s", code)


# ── streak milestones ─────────────────────────────────────────────
# A streak day is one the participant TRAINED or TESTED (activity level >= 2),
# not a plain sign-in — matching the dashboard streak bar (streak.ts). The run
# ends today or (if today isn't done yet) yesterday; awarded the moment it
# first reaches a threshold, which can only happen on a train/test event, i.e.
# exactly when these evaluators fire.

def _streak_len(levels: dict, today: date) -> int:
    def q(d: date) -> bool:
        return levels.get(d.isoformat(), 0) >= 2
    start = today if q(today) else today - timedelta(days=1)
    if not q(start):
        return 0
    n, d = 0, start
    while q(d):
        n += 1
        d -= timedelta(days=1)
    return n


def _evaluate_streak(db, code: str) -> None:
    row = db.get_participant(code)
    tz = int((row or {}).get("tz_offset_min") or 0)
    levels = db.activity_levels(code, tz)
    lt = time.gmtime(time.time() - tz * 60)
    streak = _streak_len(levels, date(lt.tm_year, lt.tm_mon, lt.tm_mday))
    for threshold, mkey, mlabel, detail in STREAK_MILESTONES:
        if streak >= threshold:
            db.award_milestone(code, mkey, mlabel, detail)
