"""Server-owned termination-policy rollout.

Requests never select their stopping policy. Unknown configuration values fail
closed to AD6, which remains the rollback implementation.
"""

AD6_POLICY = "ad6"
PRECISION_POLICY = "precision_v1"


def termination_policy_for(db, cfg: dict, code: str) -> str:
    mode = cfg.get("precision_policy_rollout")
    if mode == "all":
        return PRECISION_POLICY
    if mode != "email_allowlist":
        return AD6_POLICY
    participant = db.get_participant(code)
    email = str((participant or {}).get("email") or "").strip().lower()
    allowlist = cfg.get("precision_policy_emails") or frozenset()
    return PRECISION_POLICY if email in allowlist else AD6_POLICY
