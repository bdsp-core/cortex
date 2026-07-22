"""GET /api/bootstrap — the dashboard-entry payload in one round-trip.

Landing on the dashboard used to fan out six small authed GETs (dashboard,
trajectories, regimen, activity, session status, cohorts). This endpoint
returns them as named sections of a single response, with two invariants:

  * Each section is built by the SAME module function its standalone
    endpoint returns (one implementation, two routes) — the standalone
    endpoints stay for surfaces that refresh independently (the invite
    banner's poll, post-training refreshes, the Cohorts tab).
  * Sections are failure-isolated: one broken section becomes null plus an
    entry in `errors` instead of a 500 for the whole payload. The SPA
    treats a null section exactly like a failed standalone call today
    (that surface renders its empty state, the rest of the dashboard
    lives).

The `session` section is deliberately LIGHT — {examResumable, washout} —
not the multi-hundred-KB drawn pool of GET /api/session/active; the CTA
labels need two facts, and the full payload is fetched only when the
participant actually clicks into the test pre-flight.

`include` (optional, comma-separated section names) lets a future surface
ask for a subset — e.g. the phone home screen — without a new endpoint.
Builders are registered in _SECTION_BUILDERS; adding a section is one
entry + one line in the SPA's BootstrapData type.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ..deps import require_auth
from . import awards, cohorts, dashboard, testing

log = logging.getLogger("cortex.bootstrap")

router = APIRouter(prefix="/api")

# name → builder(req, code, tz). Helper calls go through the module attribute
# (dashboard.dashboard_payload etc.) so test monkeypatching reaches them —
# the same convention as the rest of routers/ (see routers/__init__.py).
# `traj` is a per-request memo (see _trajectory_memo): the dashboard and
# trajectories sections both need param_trajectories, and without it a single
# bootstrap ran that query twice. It stays LAZY so a section that doesn't need
# the rows never triggers the query, and so a failing query still surfaces
# inside the requesting section's own try/except (failure isolation intact).
_SECTION_BUILDERS = {
    "dashboard": lambda req, code, tz, traj: dashboard.dashboard_payload(
        req, code, trajectories=traj),
    "trajectories": lambda req, code, tz, traj: dashboard.trajectories_payload(
        req.app.state.db, code, trajectories=traj),
    "regimen": lambda req, code, tz, traj: dashboard.regimen_payload(
        req.app.state.db, code),
    "activity": lambda req, code, tz, traj: dashboard.activity_payload(
        req.app.state.db, code, tz),
    "session": lambda req, code, tz, traj: testing.session_status(
        req.app.state.db, req.app.state.get_bank(), code,
        req.app.state.get_precision_bank),
    "cohorts": lambda req, code, tz, traj: cohorts.cohorts_payload(
        req.app.state.db, code),
    "awards": lambda req, code, tz, traj: awards.pending_payload(
        req.app.state.db, code),
}


def _trajectory_memo(db, code: str):
    """One lazy param_trajectories read shared by the sections that need it."""
    cache: dict[str, list] = {}

    def get() -> list:
        if "rows" not in cache:
            cache["rows"] = db.get_trajectories(code)
        return cache["rows"]

    return get


@router.get("/bootstrap")
def bootstrap(req: Request, tz: int = 0, include: str = "",
              code: str = Depends(require_auth)):
    if include.strip():
        wanted = [s.strip() for s in include.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in _SECTION_BUILDERS]
        if unknown:
            raise HTTPException(400, f"unknown bootstrap section: {unknown[0]}")
    else:
        wanted = list(_SECTION_BUILDERS)
    # Opportunistically refresh the device's tz offset: the digest scheduler
    # (digest.py) uses it to aim letters at the learner's local morning.
    # Best-effort; a failure must never break the dashboard payload.
    if -900 <= tz <= 900:
        try:
            req.app.state.db.set_tz_offset(code, tz)
        except Exception:
            log.exception("[cortex.bootstrap] tz persist failed for %s", code)
    out: dict = {}
    errors: dict[str, str] = {}
    traj = _trajectory_memo(req.app.state.db, code)
    for name in wanted:
        try:
            out[name] = _SECTION_BUILDERS[name](req, code, tz, traj)
        except Exception:
            log.exception("[cortex.bootstrap] section %r failed for %s", name, code)
            out[name] = None
            errors[name] = "internal"
    if errors:
        out["errors"] = errors
    return out
