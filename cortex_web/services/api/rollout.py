"""Shared matching for the server-owned rollout gates.

The gates themselves stay in their own modules — stopping policy
(policy_rollout) and compute execution (compute_rollout) are independent
stamps, and nothing here couples them. What lives here is the matching each
performs, so "who is in this rollout" and, crucially, "what happens when the
configuration is unrecognised" have ONE implementation rather than a copy per
gate. Fail-closed logic that exists twice is exactly the kind that drifts.

The full decision surface of every caller is pinned in test_rollout_matrix.py.
"""
from __future__ import annotations

from typing import Optional


def email_allowlisted(db, cfg: dict, code: str, emails_key: str) -> bool:
    """True when this participant's email is in the named allowlist.

    The email is normalised the way the allowlists are built (stripped,
    lowercased). A missing participant row matches nothing.
    """
    participant = db.get_participant(code)
    email = str((participant or {}).get("email") or "").strip().lower()
    return email in (cfg.get(emails_key) or frozenset())


def identity_allowlisted(allowlist, code: str,
                         participant: Optional[dict]) -> bool:
    """True when a participant's code, 9-digit public id, or email is listed.

    Candidates are stripped and lowercased. Note that
    dashboard_logic.training_enabled deliberately does NOT use this helper: it
    matches without stripping, and changing that would alter a live exposure
    gate rather than refactor it.
    """
    if not allowlist:
        return False
    candidates = {str(code).strip().lower()}
    for key in ("public_id", "email"):
        value = (participant or {}).get(key)
        if value:
            candidates.add(str(value).strip().lower())
    return bool(candidates & allowlist)
