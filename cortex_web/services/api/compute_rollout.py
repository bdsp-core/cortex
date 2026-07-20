"""Server-authoritative rollout for browser compute execution.

This controls only *where* the already-frozen Precision calculations run.
It never changes the policy, particle profile, selection algorithm, or result.
Unknown configuration values fail closed to the serial path.
"""
from __future__ import annotations

SERIAL = "serial"
DUAL_BRANCH_AUTO = "dual_branch_auto"
PRECISION_POLICY = "precision_v1"


def compute_mode_for(db, cfg: dict, code: str,
                     termination_policy: str) -> str:
    """Choose a mode for a new sitting; clients cannot request one."""
    if termination_policy != PRECISION_POLICY:
        return SERIAL
    rollout = cfg.get("precision_compute_rollout")
    if rollout == "all":
        return DUAL_BRANCH_AUTO
    if rollout != "email_allowlist":
        return SERIAL
    participant = db.get_participant(code)
    email = str((participant or {}).get("email") or "").strip().lower()
    allowlist = cfg.get("precision_compute_emails") or frozenset()
    return DUAL_BRANCH_AUTO if email in allowlist else SERIAL
