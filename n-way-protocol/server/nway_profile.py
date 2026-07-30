"""Production-facing profile rollout contract, intentionally not wired to API.

The real API can adopt this module after green-light review.  Until then it is
an executable specification with fail-closed behavior and migration fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

BINARY_PROFILE_ID = "precision_binary_ovr_v1"
NWAY_PROFILE_ID = "precision_nway_f1_ensemble9_floor015_fisher_v1"
BINARY_RESPONSE_MODEL = "binary_ovr_v1"
NWAY_RESPONSE_MODEL = "iiic_conditional_f1_v1"
NWAY_ARTIFACT_MODEL = "iiic_conditional_f1_v1_artifact_ensemble"
NWAY_SELECTOR_VERSION = "categorical_fisher_totalvar_v1"
NWAY_ALGORITHM_VERSION = "nway_protocol_0.2.0-rd"


@dataclass(frozen=True)
class SessionProfileStamp:
    engine_profile_id: str
    response_model: str
    response_artifact_id: str | None
    response_artifact_sha256: str | None
    selector_version: str
    engine_algorithm_version: str
    candidate_bank_sha256: str

    def payload(self) -> dict:
        return asdict(self)


def binary_stamp(candidate_bank_sha256: str) -> SessionProfileStamp:
    return SessionProfileStamp(
        engine_profile_id=BINARY_PROFILE_ID,
        response_model=BINARY_RESPONSE_MODEL,
        response_artifact_id=None,
        response_artifact_sha256=None,
        selector_version="binary_totalvar_v1",
        engine_algorithm_version="cortex_web_binary_current",
        candidate_bank_sha256=candidate_bank_sha256,
    )


def nway_stamp(candidate_bank_sha256: str, artifact: dict) -> SessionProfileStamp:
    if artifact.get("qualification") != "qualified":
        raise ValueError("only a qualified response artifact can stamp an n-way session")
    if artifact.get("provenance", {}).get("dr07") != "qualified":
        raise ValueError("DR07-qualified provenance is required")
    if artifact.get("model") != NWAY_ARTIFACT_MODEL or not artifact.get("draws"):
        raise ValueError("the standard n-way profile requires an artifact ensemble")
    return SessionProfileStamp(
        engine_profile_id=NWAY_PROFILE_ID,
        response_model=NWAY_RESPONSE_MODEL,
        response_artifact_id=str(artifact["artifactId"]),
        response_artifact_sha256=str(artifact["sha256"]),
        selector_version=NWAY_SELECTOR_VERSION,
        engine_algorithm_version=NWAY_ALGORITHM_VERSION,
        candidate_bank_sha256=candidate_bank_sha256,
    )


def response_profile_for(
    db,
    config: dict,
    participant_code: str,
    termination_policy: str,
    candidate_bank_sha256: str,
    artifact: dict | None,
) -> SessionProfileStamp:
    """Server-owned rollout; unknown/missing/unqualified always selects binary."""
    fallback = binary_stamp(candidate_bank_sha256)
    if termination_policy != "precision_v1" or artifact is None:
        return fallback
    mode = config.get("nway_response_rollout", "off")
    enabled = mode == "all"
    if mode == "email_allowlist":
        participant = db.get_participant(participant_code) or {}
        email = str(participant.get("email") or "").strip().lower()
        enabled = email in (config.get("nway_response_emails") or frozenset())
    elif mode != "all":
        enabled = False
    if not enabled:
        return fallback
    try:
        return nway_stamp(candidate_bank_sha256, artifact)
    except (KeyError, TypeError, ValueError):
        return fallback


def assert_result_stamp(session: SessionProfileStamp, result: dict) -> None:
    for key, expected in session.payload().items():
        # API JSON uses camelCase; DB/session objects retain snake_case.
        pieces = key.split("_")
        json_key = pieces[0] + "".join(piece.title() for piece in pieces[1:])
        if result.get(json_key) != expected:
            raise ValueError(f"result_profile_mismatch:{json_key}")


def assert_resume_compatible(stored: SessionProfileStamp, offered: SessionProfileStamp) -> None:
    for key, expected in stored.payload().items():
        if offered.payload()[key] != expected:
            raise ValueError(f"resume_incompatible:{key}")
