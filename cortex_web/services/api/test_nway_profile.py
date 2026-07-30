from api.nway_profile import (
    ENGINE_ALGORITHM_VERSION,
    PARTICLE_PROFILE_VERSION,
    RESPONSE_ARTIFACT_SHA256,
    production_nway_profile,
)


def test_production_nway_profile_is_immutable_and_bank_stamped():
    bank_sha = "a" * 64
    profile = production_nway_profile(bank_sha)
    assert profile == {
        "engineProfileId": "precision_nway_f1_ensemble9_floor015_fisher_v1",
        "responseModel": "iiic_conditional_f1_v1",
        "responseArtifactId": "iiic-f1-crossfit-ensemble9-rd-20260720-floor015",
        "responseArtifactSha256": RESPONSE_ARTIFACT_SHA256,
        "selectorVersion": "categorical_fisher_totalvar_v1",
        "particleProfileVersion": PARTICLE_PROFILE_VERSION,
        "engineAlgorithmVersion": ENGINE_ALGORITHM_VERSION,
        "candidateBankSha256": bank_sha,
    }
