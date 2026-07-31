import hashlib
import json
from pathlib import Path

from api.nway_profile import (
    ENGINE_ALGORITHM_VERSION,
    PARTICLE_PROFILE_VERSION,
    RESEARCH_DRAW_LATENT_ARTIFACT_ID,
    RESEARCH_DRAW_LATENT_ARTIFACT_SHA256,
    RESEARCH_UNQUALIFIED_ENV,
    RESPONSE_ARTIFACT_SHA256,
    allowed_nway_profiles,
    nway_profile_for_new_session,
    production_nway_profile,
    research_draw_latent_nway_profile,
)

BANK_SHA = "a" * 64
ESCAPE_ON = {RESEARCH_UNQUALIFIED_ENV: "1"}


def test_production_nway_profile_is_immutable_and_bank_stamped():
    profile = production_nway_profile(BANK_SHA)
    assert profile == {
        "engineProfileId": "precision_nway_f1_ensemble9_floor015_fisher_v1",
        "responseModel": "iiic_conditional_f1_v1",
        "responseArtifactId": "iiic-f1-crossfit-ensemble9-rd-20260720-floor015",
        "responseArtifactSha256": RESPONSE_ARTIFACT_SHA256,
        "selectorVersion": "categorical_fisher_totalvar_v1",
        "particleProfileVersion": PARTICLE_PROFILE_VERSION,
        "engineAlgorithmVersion": ENGINE_ALGORITHM_VERSION,
        "candidateBankSha256": BANK_SHA,
    }


def test_research_draw_latent_profile_shape():
    profile = research_draw_latent_nway_profile(BANK_SHA)
    assert profile == {
        "engineProfileId":
            "precision_nway_f1_engine_frame_atoms17_draw_latent_rd_v1",
        "responseModel": "iiic_conditional_f1_v1",
        "responseArtifactId": RESEARCH_DRAW_LATENT_ARTIFACT_ID,
        "responseArtifactSha256": RESEARCH_DRAW_LATENT_ARTIFACT_SHA256,
        "selectorVersion": "categorical_fisher_totalvar_v1",
        "particleProfileVersion": PARTICLE_PROFILE_VERSION,
        "engineAlgorithmVersion": ENGINE_ALGORITHM_VERSION,
        "candidateBankSha256": BANK_SHA,
        "responseAggregation": "draw_latent",
    }


def test_research_artifact_constants_bind_to_the_canonical_artifact():
    # Canonical-draws discipline (n-way-protocol artifact_floor.py): SHA-256
    # of the compact sorted-key JSON of the camelCase draw table.
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "n-way-protocol/artifacts/iiic_conditional_f1_engine_frame_atoms17_rd.json"
    )
    payload = json.loads(artifact_path.read_text())
    assert payload["status"] == "research_only_not_promoted"
    assert payload["promotionForbidden"] is True
    draws = [
        {"beta": float(d["beta"]),
         "distractorLapse": float(d["distractor_lapse"]),
         "weight": float(d["weight"])}
        for d in payload["bootstrap"]["draws"]
    ]
    assert len(draws) == 17
    assert min(d["distractorLapse"] for d in draws) >= payload["robustnessFloor"]
    canonical = json.dumps(draws, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == RESEARCH_DRAW_LATENT_ARTIFACT_SHA256


def test_new_sessions_stamp_production_by_default():
    assert nway_profile_for_new_session(
        BANK_SHA, environ={}, release=None,
    ) == production_nway_profile(BANK_SHA)
    # Non-"1" env values fail closed too.
    assert nway_profile_for_new_session(
        BANK_SHA, environ={RESEARCH_UNQUALIFIED_ENV: "true"}, release=None,
    ) == production_nway_profile(BANK_SHA)


def test_production_deployment_refuses_the_research_escape():
    # A deployed box always carries a RELEASE stamp: the env var alone must
    # NEVER stamp the unqualified draw-latent profile there.
    assert nway_profile_for_new_session(
        BANK_SHA, environ=ESCAPE_ON, release="0d86d05deadbeef",
    ) == production_nway_profile(BANK_SHA)
    assert allowed_nway_profiles(
        BANK_SHA, environ=ESCAPE_ON, release="0d86d05deadbeef",
    ) == [production_nway_profile(BANK_SHA)]


def test_local_escape_stamps_and_resumes_the_research_profile():
    assert nway_profile_for_new_session(
        BANK_SHA, environ=ESCAPE_ON, release=None,
    ) == research_draw_latent_nway_profile(BANK_SHA)
    assert allowed_nway_profiles(BANK_SHA, environ=ESCAPE_ON, release=None) == [
        production_nway_profile(BANK_SHA),
        research_draw_latent_nway_profile(BANK_SHA),
    ]
    # Once the escape is off, a stored research sitting is not resumable.
    assert allowed_nway_profiles(BANK_SHA, environ={}, release=None) == [
        production_nway_profile(BANK_SHA),
    ]
