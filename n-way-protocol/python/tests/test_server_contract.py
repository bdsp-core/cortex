from __future__ import annotations

import pytest

from server.nway_profile import (
    NWAY_PROFILE_ID,
    NWAY_SELECTOR_VERSION,
    assert_result_stamp,
    assert_resume_compatible,
    response_profile_for,
)


class DB:
    def get_participant(self, _code):
        return {"email": "reader@example.org"}


QUALIFIED = {
    "artifactId": "f1-qualified-v1",
    "sha256": "a" * 64,
    "qualification": "qualified",
    "model": "iiic_conditional_f1_v1_artifact_ensemble",
    "draws": [{"beta": 1.0, "distractorLapse": 0.0, "weight": 1.0}],
    "provenance": {"dr07": "qualified"},
}


def test_rollout_is_server_owned_and_fails_closed():
    db = DB()
    bank = "b" * 64
    assert response_profile_for(db, {}, "reader", "precision_v1", bank, QUALIFIED).response_model == "binary_ovr_v1"
    assert response_profile_for(
        db, {"nway_response_rollout": "unexpected"}, "reader", "precision_v1", bank, QUALIFIED,
    ).response_model == "binary_ovr_v1"
    assert response_profile_for(
        db, {"nway_response_rollout": "all"}, "reader", "ad6", bank, QUALIFIED,
    ).response_model == "binary_ovr_v1"
    selected = response_profile_for(
        db, {"nway_response_rollout": "all"}, "reader", "precision_v1", bank, QUALIFIED,
    )
    assert selected.engine_profile_id == NWAY_PROFILE_ID
    assert selected.selector_version == NWAY_SELECTOR_VERSION


def test_unqualified_artifact_cannot_enter_rollout():
    selected = response_profile_for(
        DB(), {"nway_response_rollout": "all"}, "reader", "precision_v1", "b" * 64,
        {**QUALIFIED, "qualification": "exploratory_unqualified"},
    )
    assert selected.response_model == "binary_ovr_v1"


def test_qualified_scalar_artifact_cannot_enter_new_standard_rollout():
    selected = response_profile_for(
        DB(), {"nway_response_rollout": "all"}, "reader", "precision_v1", "b" * 64,
        {**QUALIFIED, "model": "iiic_conditional_f1_v1", "draws": None},
    )
    assert selected.response_model == "binary_ovr_v1"


def test_result_and_resume_require_exact_profile_stamp():
    stamp = response_profile_for(
        DB(), {"nway_response_rollout": "all"}, "reader", "precision_v1", "b" * 64,
        QUALIFIED,
    )
    result = {
        "engineProfileId": stamp.engine_profile_id,
        "responseModel": stamp.response_model,
        "responseArtifactId": stamp.response_artifact_id,
        "responseArtifactSha256": stamp.response_artifact_sha256,
        "selectorVersion": stamp.selector_version,
        "engineAlgorithmVersion": stamp.engine_algorithm_version,
        "candidateBankSha256": stamp.candidate_bank_sha256,
    }
    assert_result_stamp(stamp, result)
    assert_resume_compatible(stamp, stamp)
    with pytest.raises(ValueError, match="result_profile_mismatch"):
        assert_result_stamp(stamp, {**result, "responseModel": "binary_ovr_v1"})
    with pytest.raises(ValueError, match="resume_incompatible"):
        assert_resume_compatible(stamp, type(stamp)(
            **{**stamp.payload(), "engine_algorithm_version": "drifted"}
        ))
