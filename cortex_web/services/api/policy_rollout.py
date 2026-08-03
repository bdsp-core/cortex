"""Server-owned termination-policy rollout.

Requests never select their stopping policy. Unknown configuration values fail
closed to AD6, which remains the rollback implementation.
"""

from .rollout import email_allowlisted

AD6_POLICY = "ad6"
PRECISION_POLICY = "precision_v1"
PRECISION_C1 = "c1"


def termination_policy_for(db, cfg: dict, code: str) -> str:
    mode = cfg.get("precision_policy_rollout")
    if mode == "all":
        return PRECISION_POLICY
    if mode != "email_allowlist":
        return AD6_POLICY
    return (PRECISION_POLICY
            if email_allowlisted(db, cfg, code, "precision_policy_emails")
            else AD6_POLICY)


def precision_recalibration_for(
        db, cfg: dict, code: str, termination_policy: str) -> str | None:
    """The c1 stopping-recalibration stamp for a NEW sitting, or None.

    c1 = evidence floor n_min 20->0 + declaration persistence 2->3 (the
    2026-08 nmin-stopping-study promotion). Applies only to precision_v1
    sittings; unknown modes fail closed to the shipped 20/2 configuration.
    Existing sittings always retain their persisted stamp.
    """
    if termination_policy != PRECISION_POLICY:
        return None
    mode = cfg.get("precision_c1_rollout")
    if mode == "all":
        return PRECISION_C1
    if mode != "email_allowlist":
        return None
    return (PRECISION_C1
            if email_allowlisted(db, cfg, code, "precision_c1_emails")
            else None)
