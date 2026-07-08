"""Cohorts: manager-run peer groups (isolated pods).

Access model, enforced server-side on every endpoint:
  * A cohort's data is visible ONLY to its manager and its ACTIVE members —
    an outsider gets 404 (existence is not confirmed), a pending invitee
    gets the cohort's name but no member/performance data.
  * Every payload identifies people by 9-digit public id. Display names
    resolve for the MANAGER alone; the internal participant `code` (the JWT
    subject / DB key) never appears in any cohort response.
  * Invites address users by public id (invite + accept: performance is
    shared only after the invitee consents). Decline / leave / remove all
    delete the membership row; the manager deletes the cohort itself.
Performance series come from param_trajectories with the same is_real=1
contract as the dashboard: certification evals today, real training points
the day the trainer ships — synthetic rows can never reach a cohort peer.
"""
from __future__ import annotations

import re
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import helpers
from ..deps import client_ip, require_auth
from ..models import CohortCreateIn, CohortMemberIn

router = APIRouter(prefix="/api")

MAX_COHORT_NAME_LEN = 60
MAX_COHORTS_MANAGED = 10     # pods one account can run
MAX_COHORT_MEMBERS = 100     # rows (active + pending) per pod
LOOKBACK_DAYS = 183          # default performance window when none is requested
MAX_LOOKBACK_DAYS = 4000      # ~11y clamp on any client-requested window
EPOCH_CUTOFF = "1970-01-01T00:00:00Z"   # "all time" sentinel (no lower bound)

_PUBLIC_ID_RE = re.compile(r"^[1-9]\d{8}$")


def _lookback_cutoff(days: int = LOOKBACK_DAYS) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(time.time() - days * 86400))


def _cohort_and_role(db, cohort_id: str, code: str, *,
                     need_manager: bool = False,
                     need_active: bool = False):
    """Resolve (cohort, my member row, am-i-manager) and enforce access.

    Outsiders get 404 — a cohort id must not be probeable. A known member
    who lacks the needed role gets 403 with a human reason."""
    cohort = db.get_cohort(cohort_id)
    membership = db.cohort_membership(cohort_id, code) if cohort else None
    is_manager = bool(cohort) and cohort["manager_code"] == code
    if cohort is None or (membership is None and not is_manager):
        raise HTTPException(404, "cohort not found")
    if need_manager and not is_manager:
        raise HTTPException(403, "only the cohort manager can do this")
    if need_active and not is_manager and membership["status"] != "active":
        raise HTTPException(403, "accept the invitation first")
    return cohort, membership, is_manager


def _member_entry(db, row: dict, requester_code: str,
                  manager_code: str, include_names: bool) -> dict:
    """One member row → its API shape. public_id is guaranteed by signup/boot
    backfill; ensure_public_id is the belt-and-suspenders repair."""
    pid = row.get("public_id") or db.ensure_public_id(row["code"]) or ""
    entry = {
        "publicId": str(pid),
        "status": row["status"],
        "isManager": row["code"] == manager_code,
        "isYou": row["code"] == requester_code,
        "joinedUtc": row.get("joined_utc"),
    }
    if include_names:
        entry["displayName"] = row.get("display_name") or ""
    return entry


@router.get("/cohorts")
def list_cohorts(req: Request, code: str = Depends(require_auth)):
    db = req.app.state.db
    mine = db.cohorts_for(code)
    # One grouped COUNT for all the caller's cohorts — the list view needs
    # only counts, not full member rows fetched cohort-by-cohort.
    counts = db.active_member_counts([r["cohort_id"] for r in mine])
    out = []
    for r in mine:
        out.append({
            "cohortId": r["cohort_id"],
            "name": r["name"],
            "role": "manager" if r["manager_code"] == code else "member",
            "status": r["status"],
            "memberCount": counts.get(r["cohort_id"], 0),
            "createdUtc": r["created_utc"],
        })
    return {"cohorts": out}


@router.post("/cohorts")
def create_cohort(body: CohortCreateIn, req: Request,
                  code: str = Depends(require_auth)):
    db, limiter = req.app.state.db, req.app.state.limiter
    if not limiter.hit("cohort_create", client_ip(req)):
        raise HTTPException(429, "too many cohorts created — try again later")
    name = body.name.strip()[:MAX_COHORT_NAME_LEN]
    if not name:
        raise HTTPException(400, "cohort name required")
    if db.count_cohorts_managed(code) >= MAX_COHORTS_MANAGED:
        raise HTTPException(409, f"you already manage {MAX_COHORTS_MANAGED} cohorts")
    cohort_id = "ch-" + secrets.token_urlsafe(9)
    db.create_cohort(cohort_id, name, code)
    return {"cohortId": cohort_id, "name": name}


@router.get("/cohorts/{cohort_id}")
def cohort_detail(cohort_id: str, req: Request,
                  code: str = Depends(require_auth)):
    db = req.app.state.db
    cohort, membership, is_manager = _cohort_and_role(db, cohort_id, code)
    resp = {
        "cohortId": cohort["cohort_id"],
        "name": cohort["name"],
        "role": "manager" if is_manager else "member",
        "status": membership["status"] if membership else "active",
        "createdUtc": cohort["created_utc"],
    }
    # A pending invitee sees only the cohort's name (enough to decide);
    # the member list is for active members + the manager.
    if is_manager or (membership and membership["status"] == "active"):
        rows = db.cohort_member_rows(cohort_id)
        resp["members"] = [
            _member_entry(db, m, code, cohort["manager_code"], is_manager)
            for m in rows
            # pending invites are the manager's bookkeeping, not peers' info
            if m["status"] == "active" or is_manager
        ]
    return resp


@router.post("/cohorts/{cohort_id}/invite")
def cohort_invite(cohort_id: str, body: CohortMemberIn, req: Request,
                  code: str = Depends(require_auth)):
    db, limiter = req.app.state.db, req.app.state.limiter
    _cohort_and_role(db, cohort_id, code, need_manager=True)
    if not limiter.hit("cohort_invite", client_ip(req)):
        raise HTTPException(429, "too many invites — try again later")
    pid = body.publicId.strip().replace(" ", "")
    if not _PUBLIC_ID_RE.fullmatch(pid):
        raise HTTPException(400, "a user ID is 9 digits")
    target = db.get_participant_by_public_id(pid)
    if target is None or not target.get("active"):
        raise HTTPException(404, "no account with that user ID")
    if db.count_cohort_members(cohort_id) >= MAX_COHORT_MEMBERS:
        raise HTTPException(409, f"cohort is at its {MAX_COHORT_MEMBERS}-member limit")
    try:
        db.invite_cohort_member(cohort_id, target["code"])
    except Exception as e:
        if helpers.is_unique_violation(e):
            raise HTTPException(409, "that user is already in this cohort or has a pending invitation")
        raise
    return {"ok": True, "publicId": pid}


@router.post("/cohorts/{cohort_id}/accept")
def cohort_accept(cohort_id: str, req: Request,
                  code: str = Depends(require_auth)):
    db = req.app.state.db
    _cohort_and_role(db, cohort_id, code)
    if not db.accept_cohort_invite(cohort_id, code):
        raise HTTPException(409, "no pending invitation")
    return {"ok": True}


@router.post("/cohorts/{cohort_id}/decline")
def cohort_decline(cohort_id: str, req: Request,
                   code: str = Depends(require_auth)):
    db = req.app.state.db
    _, membership, is_manager = _cohort_and_role(db, cohort_id, code)
    if is_manager or membership is None or membership["status"] != "invited":
        raise HTTPException(409, "no pending invitation")
    db.remove_cohort_member(cohort_id, code)
    return {"ok": True}


@router.post("/cohorts/{cohort_id}/leave")
def cohort_leave(cohort_id: str, req: Request,
                 code: str = Depends(require_auth)):
    db = req.app.state.db
    _, membership, is_manager = _cohort_and_role(db, cohort_id, code)
    if is_manager:
        raise HTTPException(400, "the manager cannot leave — delete the cohort instead")
    db.remove_cohort_member(cohort_id, code)
    return {"ok": True}


@router.post("/cohorts/{cohort_id}/remove")
def cohort_remove(cohort_id: str, body: CohortMemberIn, req: Request,
                  code: str = Depends(require_auth)):
    db = req.app.state.db
    cohort, _, _ = _cohort_and_role(db, cohort_id, code, need_manager=True)
    pid = body.publicId.strip().replace(" ", "")
    target = db.get_participant_by_public_id(pid)
    if target is None or db.cohort_membership(cohort_id, target["code"]) is None:
        raise HTTPException(404, "that user is not in this cohort")
    if target["code"] == cohort["manager_code"]:
        raise HTTPException(400, "the manager cannot be removed — delete the cohort instead")
    db.remove_cohort_member(cohort_id, target["code"])
    return {"ok": True}


@router.delete("/cohorts/{cohort_id}")
def cohort_delete(cohort_id: str, req: Request,
                  code: str = Depends(require_auth)):
    db = req.app.state.db
    _cohort_and_role(db, cohort_id, code, need_manager=True)
    db.delete_cohort(cohort_id)
    return {"ok": True}


@router.get("/cohorts/{cohort_id}/performance")
def cohort_performance(cohort_id: str, req: Request,
                       days: int | None = None,
                       code: str = Depends(require_auth)):
    """The graph payload: per active member, per task (0..6), the REAL
    trajectory points inside the requested window — skill (ℓ) + bias (t) with
    sd. Members with no points still appear (the legend shows the whole
    pod). Stable member order = invite time, so client color slots hold.

    `days` selects the lookback window (7 / 30 / 90 / 365 …). Omitted →
    the default LOOKBACK_DAYS window (the historical behavior). `days <= 0`
    → "all time" (no lower bound; the returned `from` is the earliest point
    so the client axis fits the data). Only the latest operating point per
    member/task/calendar-day is returned, so a busy day (e.g. a training
    session's many trials) contributes one point rather than a vertical
    stack."""
    db = req.app.state.db
    cohort, _, is_manager = _cohort_and_role(db, cohort_id, code,
                                             need_active=True)
    rows = [m for m in db.cohort_member_rows(cohort_id)
            if m["status"] == "active"]
    all_time = days is not None and days <= 0
    if all_time:
        cutoff = EPOCH_CUTOFF
    else:
        win = LOOKBACK_DAYS if days is None else min(days, MAX_LOOKBACK_DAYS)
        cutoff = _lookback_cutoff(win)
    points = db.eval_trajectories_for_codes_since(
        [m["code"] for m in rows], cutoff)
    # Collapse to the latest point per (member, task, UTC day). The query
    # returns rows ordered by ts ascending, so the last write for a day wins.
    # (ts stored at second granularity as "YYYY-MM-DDT..Z", so ts[:10] = day.)
    by_code: dict[str, dict[int, dict[str, dict]]] = {}
    for p in points:
        if p.get("task_k") is None:
            continue
        day = (p["ts"] or "")[:10]
        by_code.setdefault(p["code"], {}).setdefault(int(p["task_k"]), {})[day] = {
            "ts": p["ts"],
            "skill": p["ell"],
            "bias": p["theta"],
            "sd": p["sd"],
            "phase": p.get("phase") or "eval",
        }
    members = []
    for m in rows:
        entry = _member_entry(db, m, code, cohort["manager_code"], is_manager)
        tasks = by_code.get(m["code"], {})
        entry["series"] = [
            {"taskK": k, "points": [tasks[k][day] for day in sorted(tasks[k])]}
            for k in sorted(tasks)]
        members.append(entry)
    now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if all_time:
        # start the axis at the earliest real point (fall back to a short
        # window when the pod has no data yet, so the axis is never degenerate)
        earliest = min((p["ts"] for p in points if p.get("ts")), default=None)
        frm = earliest or _lookback_cutoff(30)
    else:
        frm = cutoff
    # Per-task cut scores (ℓ*) so the client can draw the goal line on the skill
    # chart. Sourced from the live bank manifest (the authoritative current cut).
    bank = req.app.state.get_bank()
    ell_star = bank.engine.get("ellStar") if bank else None
    return {
        "cohortId": cohort["cohort_id"],
        "name": cohort["name"],
        "from": frm,
        "to": now_ts,
        "ellStar": ell_star,
        "members": members,
    }
