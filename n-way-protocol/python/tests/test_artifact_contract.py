from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent


def test_exploratory_artifact_is_schema_valid_hash_bound_and_nonpromotable():
    artifact = json.loads((ROOT / "artifacts/iiic_conditional_f1_stage2_exploratory.json").read_text())
    schema = json.loads((ROOT / "schemas/response-artifact.schema.json").read_text())
    jsonschema.validate(artifact, schema)
    source = REPO / artifact["provenance"]["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact["sha256"]
    assert artifact["qualification"] == "exploratory_unqualified"
    assert artifact["provenance"]["dr07"] == "not_qualified"
    assert artifact["promotionForbidden"] is True


def test_integrated_ensemble_is_schema_valid_hash_bound_and_nonpromotable():
    artifact = json.loads(
        (ROOT / "artifacts/iiic_conditional_f1_integrated_ensemble9_rd.json").read_text()
    )
    schema = json.loads((ROOT / "schemas/response-artifact.schema.json").read_text())
    jsonschema.validate(artifact, schema)
    canonical_draws = json.dumps(
        artifact["draws"], sort_keys=True, separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(canonical_draws).hexdigest() == artifact["sha256"]
    assert sum(draw["weight"] for draw in artifact["draws"]) == 1
    assert artifact["qualification"] == "exploratory_unqualified"
    assert artifact["provenance"]["dr07"] == "not_qualified"
    assert artifact["promotionForbidden"] is True

    profile = json.loads(
        (ROOT / "artifacts/engine_profile_integrated_ensemble9_fisher_rd.json").read_text()
    )
    profile_schema = json.loads((ROOT / "schemas/engine-profile.schema.json").read_text())
    jsonschema.validate(profile, profile_schema)
    assert profile["responseArtifactId"] == artifact["artifactId"]
    assert profile["responseArtifactSha256"] == artifact["sha256"]
    assert profile["selectorVersion"] == "categorical_fisher_totalvar_v1"

    standard = json.loads((ROOT / "artifacts/nway_standard_rd.json").read_text())
    standard_schema = json.loads((ROOT / "schemas/nway-standard.schema.json").read_text())
    jsonschema.validate(standard, standard_schema)
    assert standard["engineProfileId"] == profile["engineProfileId"]
    assert standard["responseArtifactId"] == artifact["artifactId"]
    for key in ("engineProfilePath", "responseArtifactPath", "frozenRollbackArchive",
                "evidenceReport"):
        assert (ROOT / standard[key]).is_file()
    assert standard["productionAuthorized"] is False


def test_migration_is_additive_and_preserves_raw_pick_column():
    migration = (ROOT / "server/migration.sql").read_text().lower()
    assert "alter table sessions add column" in migration
    assert "drop table" not in migration
    assert "delete from" not in migration
    assert "trials" in migration and "pick" in migration
