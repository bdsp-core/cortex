"""Fail-closed exposure policy for the provisional percentile preview."""
from __future__ import annotations

from typing import Any

from .rollout import identity_allowlisted

VALID_MODES = frozenset({"off", "shadow", "cohort", "all"})


def profile_for(
    runtime, cfg: dict[str, Any], code: str, participant: dict | None,
) -> dict[str, Any] | None:
    """Return the exact session stamp, or None when calculation is disabled.

    ``shadow`` calculates/persists but never renders. Public modes additionally
    require an operator to acknowledge the exact binary SHA in the environment;
    copying code/artifacts alone can therefore never activate this preview.
    """
    mode = str((cfg or {}).get("percentile_mode", "off")).strip().lower()
    if mode not in VALID_MODES or mode == "off" or not runtime.ready:
        return None
    if mode == "shadow":
        return runtime.profile(display=False)
    if mode == "cohort" and not identity_allowlisted(
        (cfg or {}).get("percentile_allowlist"), code, participant
    ):
        return None
    metadata = runtime.metadata or {}
    if (cfg or {}).get("percentile_release_sha256") != metadata.get(
        "runtimeDataSha256"
    ):
        return None
    return runtime.profile(display=True)
