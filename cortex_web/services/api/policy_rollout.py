"""Server-owned termination-policy rollout.

Requests never select their stopping policy. Unknown configuration values fail
closed to AD6, which remains the rollback implementation.
"""

from .rollout import email_allowlisted

AD6_POLICY = "ad6"
PRECISION_POLICY = "precision_v1"


def termination_policy_for(db, cfg: dict, code: str) -> str:
    mode = cfg.get("precision_policy_rollout")
    if mode == "all":
        return PRECISION_POLICY
    if mode != "email_allowlist":
        return AD6_POLICY
    return (PRECISION_POLICY
            if email_allowlisted(db, cfg, code, "precision_policy_emails")
            else AD6_POLICY)
