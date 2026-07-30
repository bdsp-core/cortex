"""Immutable production stamp for the owner-approved native IIIC profile."""
from __future__ import annotations

RESPONSE_MODEL = "iiic_conditional_f1_v1"
RESPONSE_ARTIFACT_ID = "iiic-f1-crossfit-ensemble9-rd-20260720-floor015"
RESPONSE_ARTIFACT_SHA256 = (
    "2e35c400739d74778c33c6a9b2f2b8590a9c6f96f3b83841fc3fc6091db8457c"
)
SELECTOR_VERSION = "categorical_fisher_totalvar_v1"
PARTICLE_PROFILE_VERSION = "production_1200p_ess050_30mh_rd"
ENGINE_ALGORITHM_VERSION = "nway_protocol_0.2.0-rd"


def production_nway_profile(candidate_bank_sha256: str) -> dict:
    """Return the exact browser/server replay contract for a new sitting."""
    return {
        "engineProfileId": "precision_nway_f1_ensemble9_floor015_fisher_v1",
        "responseModel": RESPONSE_MODEL,
        "responseArtifactId": RESPONSE_ARTIFACT_ID,
        "responseArtifactSha256": RESPONSE_ARTIFACT_SHA256,
        "selectorVersion": SELECTOR_VERSION,
        "particleProfileVersion": PARTICLE_PROFILE_VERSION,
        "engineAlgorithmVersion": ENGINE_ALGORITHM_VERSION,
        "candidateBankSha256": candidate_bank_sha256,
    }
