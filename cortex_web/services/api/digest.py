"""Ripeness-triggered training digest (the return-trigger email).

The app only speaks when visited; this letter closes the loop when the
learner is away. Guard rails, in order:

  * TRAINING ONLY by design. No exam nudges, ever: the exam/washout surface
    stays a clean psychometric measure, and its austerity is deliberate.
  * Heavy-tailed cadence, not daily nagging: a letter goes out only when the
    deck has been ripening for exactly 1, 2, 3, 5, 8, 13 or 21 whole days
    since the last completed training sitting (or since the regimen was
    created, for a learner who has not trained yet). After 21 the account is
    dormant and we go quiet. Training again resets the cycle.
  * At most one letter per LOCAL day (digest_log ledger; the row is claimed
    BEFORE sending so a crash costs one letter, never a double-send), aimed
    at the learner's local morning via the tz offset stored at bootstrap.
  * Content is the live protocol state, nothing invented: domains still
    below their certification bar per the latest real trajectory point, plus
    a near-bar line when an estimate sits within one SD of the bar.
  * Only verified, deliverable, opted-in addresses (Settings toggle;
    digest_recipients does the filtering).

Scheduling: an in-process asyncio loop started from the app lifespan — the
same single-uvicorn-worker assumption as the rate limiter and SessionBank.
The initial delay keeps short-lived test clients from ever running a pass;
tests call run_digest_pass() directly with a frozen `now`. Kill switch:
CORTEX_DIGEST_DISABLED=1.
"""
from __future__ import annotations

import asyncio
import calendar
import logging
import os
import time
from typing import Optional

from . import mailer

log = logging.getLogger("cortex.digest")

BACKOFF_DAYS = (1, 2, 3, 5, 8, 13, 21)
SEND_HOUR_FROM, SEND_HOUR_TO = 7, 12   # learner-local send window
TICK_S = 900
INITIAL_DELAY_S = 120

SUBJECT = "Your CORTEX training deck is due"
TITLE = "Time for a training session"


def _parse_utc(ts: Optional[str]) -> Optional[float]:
    if not ts or len(ts) < 19:
        return None
    try:
        return float(calendar.timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")))
    except Exception:
        return None


def _local_hour(now_s: float, tz_offset_min: int) -> int:
    return time.gmtime(now_s - tz_offset_min * 60).tm_hour


def _local_day(now_s: float, tz_offset_min: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now_s - tz_offset_min * 60))


def build_letter(db, code: str, now_s: float) -> Optional[dict]:
    """The letter content for `code`, or None when nothing is due today:
    no active regimen/deck, wrong backoff day, or no domain below its bar."""
    reg = db.get_active_regimen(code)
    if not reg:
        return None
    deck = (reg.get("plan") or {}).get("deck") or []
    if not deck:
        return None
    last_trained = db.latest_training_finished(code)
    anchor_s = _parse_utc(last_trained) or _parse_utc(reg.get("created_utc"))
    if anchor_s is None:
        return None
    days = int((now_s - anchor_s) // 86400)
    if days not in BACKOFF_DAYS:
        return None
    # Live protocol state: the latest real trajectory point per task wins over
    # the deck's frozen at-creation estimate.
    latest: dict[int, dict] = {}
    for r in db.get_trajectories(code):
        latest[int(r["task_k"])] = r
    below: list[str] = []
    near: list[str] = []
    for row in deck:
        k = row.get("taskK")
        label = row.get("label") or row.get("code") or f"task {k}"
        ell_star = float(row.get("ellStar") or 0.0)
        tr = latest.get(int(k)) if k is not None else None
        ell = (float(tr["ell"]) if tr and tr.get("ell") is not None
               else float(row.get("ell") or 0.0))
        sd = float(tr["sd"]) if tr and tr.get("sd") is not None else None
        if ell >= ell_star:
            continue
        below.append(str(label))
        if sd is not None and ell + sd >= ell_star:
            near.append(str(label))
    if not below:
        return None
    n = len(below)
    doms = ", ".join(below)
    paras: list[str] = []
    if last_trained is not None:
        paras.append(f"It has been {days} day{'s' if days != 1 else ''} "
                     "since your last training session.")
    else:
        paras.append("Your training protocol is ready and waiting for its first session.")
    paras.append(
        f"{n} domains in your protocol are still below their certification bar: {doms}."
        if n > 1 else
        f"1 domain in your protocol is still below its certification bar: {doms}.")
    if near:
        paras.append(f"You are close on {near[0]}: your latest estimate is within "
                     "one standard deviation of the bar. One strong session could clear it.")
    paras.append("You can turn these reminders off any time in Settings.")
    origin = os.environ.get("CORTEX_PUBLIC_ORIGIN", "").strip().rstrip("/")
    button = ("Resume training", origin) if origin else None
    return {"subject": SUBJECT, "title": TITLE, "paragraphs": paras, "button": button}


def run_digest_pass(db, now_s: Optional[float] = None) -> int:
    """One scheduler tick: send every letter due right now. Returns the number
    sent. Per-recipient failures are logged and skipped; the pass never raises."""
    if os.environ.get("CORTEX_DIGEST_DISABLED"):
        return 0
    if now_s is None:
        now_s = time.time()
    try:
        recipients = db.digest_recipients()
    except Exception:
        log.exception("[cortex.digest] recipient query failed")
        return 0
    sent = 0
    for r in recipients:
        try:
            tz = int(r.get("tz_offset_min") or 0)
            if not (SEND_HOUR_FROM <= _local_hour(now_s, tz) < SEND_HOUR_TO):
                continue
            letter = build_letter(db, r["code"], now_s)
            if letter is None:
                continue
            if not db.claim_digest_day(r["code"], _local_day(now_s, tz)):
                continue
            mailer.send_letter(r["email"], letter["subject"], letter["title"],
                               letter["paragraphs"], letter["button"])
            sent += 1
        except Exception:
            log.exception("[cortex.digest] send failed for %s", r.get("code"))
    return sent


async def scheduler_loop(db, tick_s: float = TICK_S,
                         initial_delay_s: float = INITIAL_DELAY_S) -> None:
    """In-process scheduler (app lifespan owns the task; single worker)."""
    await asyncio.sleep(initial_delay_s)
    while True:
        try:
            n = run_digest_pass(db)
            if n:
                log.info("[cortex.digest] sent %d letter(s)", n)
        except Exception:
            log.exception("[cortex.digest] pass failed")
        await asyncio.sleep(tick_s)
