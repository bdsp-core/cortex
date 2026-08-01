import hashlib
import json
from pathlib import Path

from api.nway_profile import (
    ENGINE_ALGORITHM_VERSION,
    PARTICLE_PROFILE_VERSION,
    QUALIFIED_DRAW_LATENT_ARTIFACT_ID,
    QUALIFIED_DRAW_LATENT_ARTIFACT_SHA256,
    RESPONSE_ARTIFACT_SHA256,
    allowed_nway_profiles,
    nway_profile_for_new_session,
    production_nway_profile,
    qualified_draw_latent_nway_profile,
    response_model_rollout,
)

BANK_SHA = "a" * 64


class _Db:
    """get_participant stub for the email-allowlist helper."""

    def __init__(self, email):
        self._email = email

    def get_participant(self, code):
        return {"email": self._email} if self._email is not None else None


ALLOWLISTED = _Db("owner@example.org")
OTHER = _Db("someone@example.org")


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


def test_qualified_draw_latent_profile_shape():
    assert qualified_draw_latent_nway_profile(BANK_SHA) == {
        "engineProfileId": "precision_nway_f1_nesting34_draw_latent_v1",
        "responseModel": "iiic_conditional_f1_v1",
        "responseArtifactId": QUALIFIED_DRAW_LATENT_ARTIFACT_ID,
        "responseArtifactSha256": QUALIFIED_DRAW_LATENT_ARTIFACT_SHA256,
        "selectorVersion": "categorical_fisher_totalvar_v1",
        "particleProfileVersion": PARTICLE_PROFILE_VERSION,
        "engineAlgorithmVersion": ENGINE_ALGORITHM_VERSION,
        "candidateBankSha256": BANK_SHA,
        "responseAggregation": "draw_latent",
    }


def test_qualified_artifact_constants_bind_to_the_canonical_artifact():
    # Canonical-draws discipline (n-way-protocol artifact_floor.py): SHA-256
    # of the compact sorted-key JSON of the camelCase draw table, bound to
    # the QUALIFIED Phase-2 re-emission (campaign + owner-ratified waiver in
    # its provenance chain).
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "n-way-protocol/artifacts"
        / "iiic_conditional_f1_engine_frame_nesting34_qualified.json"
    )
    payload = json.loads(artifact_path.read_text())
    assert payload["qualification"] == "qualified"
    assert payload["promotionForbidden"] is False
    waiver = payload["provenance"]["qualified_by"]["owner_waiver"]
    assert waiver["clause"] == "absolute_bias_coverage"
    assert waiver["absolute_bias_coverage_cell1"] == 0.94575
    assert waiver["absolute_bias_coverage_cell2"] == 0.94475
    draws = [
        {"beta": float(d["beta"]),
         "distractorLapse": float(d["distractor_lapse"]),
         "weight": float(d["weight"])}
        for d in payload["bootstrap"]["draws"]
    ]
    assert len(draws) == 34
    # The nesting atom is the only draw above the floor grid: lambda_d = 1.
    assert draws[-1]["distractorLapse"] == 1.0
    assert min(d["distractorLapse"] for d in draws) >= payload["robustnessFloor"]
    canonical = json.dumps(draws, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == (
        QUALIFIED_DRAW_LATENT_ARTIFACT_SHA256
    )


def test_response_model_rollout_fails_closed():
    cfg = {"nway_response_emails": frozenset({"owner@example.org"})}
    # Absent, off, and unknown all mean the mixture.
    assert response_model_rollout(ALLOWLISTED, {}, "c") == "mixture"
    assert response_model_rollout(
        ALLOWLISTED, cfg | {"nway_response_rollout": "off"}, "c") == "mixture"
    assert response_model_rollout(
        ALLOWLISTED, cfg | {"nway_response_rollout": "banana"}, "c") == "mixture"
    assert response_model_rollout(
        ALLOWLISTED, cfg | {"nway_response_rollout": "all"}, "c",
    ) == "qualified_draw_latent"
    assert response_model_rollout(
        ALLOWLISTED, cfg | {"nway_response_rollout": "email_allowlist"}, "c",
    ) == "qualified_draw_latent"
    assert response_model_rollout(
        OTHER, cfg | {"nway_response_rollout": "email_allowlist"}, "c",
    ) == "mixture"
    assert response_model_rollout(
        _Db(None), cfg | {"nway_response_rollout": "email_allowlist"}, "c",
    ) == "mixture"


def test_new_session_stamp_follows_the_rollout():
    cfg_off = {"nway_response_rollout": "off"}
    cfg_all = {"nway_response_rollout": "all"}
    cfg_list = {"nway_response_rollout": "email_allowlist",
                "nway_response_emails": frozenset({"owner@example.org"})}
    assert nway_profile_for_new_session(
        BANK_SHA, db=ALLOWLISTED, cfg=cfg_off, code="c",
    ) == production_nway_profile(BANK_SHA)
    assert nway_profile_for_new_session(
        BANK_SHA, db=ALLOWLISTED, cfg=cfg_all, code="c",
    ) == qualified_draw_latent_nway_profile(BANK_SHA)
    assert nway_profile_for_new_session(
        BANK_SHA, db=ALLOWLISTED, cfg=cfg_list, code="c",
    ) == qualified_draw_latent_nway_profile(BANK_SHA)
    assert nway_profile_for_new_session(
        BANK_SHA, db=OTHER, cfg=cfg_list, code="c",
    ) == production_nway_profile(BANK_SHA)
    # Callers without rollout context always get the fail-closed mixture.
    assert nway_profile_for_new_session(BANK_SHA) == (
        production_nway_profile(BANK_SHA))


def test_both_production_stamps_resume_and_nothing_else_does():
    assert allowed_nway_profiles(BANK_SHA) == [
        production_nway_profile(BANK_SHA),
        qualified_draw_latent_nway_profile(BANK_SHA),
    ]


def test_retired_stamps_are_not_resumable():
    """Retired sittings end SOFTLY, by absence from the allowed list.

    GET /api/session/active treats an unlisted stamp as nothing-to-resume
    ({"active": null}) and a new sitting supersedes the stored one — the
    owner-accepted clean restart, never an opaque error.
    """
    allowed = allowed_nway_profiles(BANK_SHA)
    legacy = production_nway_profile(BANK_SHA) | {
        "engineProfileId": "precision_nway_f1_ensemble9_fisher_v1",
        "responseArtifactId": "iiic-f1-crossfit-ensemble9-rd-20260720",
    }
    assert legacy not in allowed
    # The pre-qualification research escape stamp is retired with the escape:
    # the qualified draw-latent path (rollout) replaced it everywhere.
    research = qualified_draw_latent_nway_profile(BANK_SHA) | {
        "engineProfileId":
            "precision_nway_f1_engine_frame_atoms17_draw_latent_rd_v1",
        "responseArtifactId": "iiic-f1-engine-frame-atoms17-rd-20260730",
        "responseArtifactSha256":
            "b25bd1c5e680b77df394eeccab0961bd1f67c1593ff3d4a144d0b92eb8f90fb8",
    }
    assert research not in allowed
