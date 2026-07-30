from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from nway_protocol.artifact_floor import (
    assert_floored,
    derive_floored_crossfit,
    derive_floored_ensemble,
    floor_draws,
)

ROOT = Path(__file__).resolve().parents[2]
FLOORS = (0.15, 0.20)


def _floored_path(base: str, floor: float) -> Path:
    stem, suffix = base.rsplit(".", 1)
    return ROOT / f"artifacts/{stem}_floor{int(round(floor * 100)):03d}.{suffix}"


def test_floor_draws_raises_only_subfloor_lapses():
    draws = [
        {"beta": 1.0, "distractor_lapse": 0.0, "weight": 0.5},
        {"beta": 0.9, "distractor_lapse": 0.15, "weight": 0.25},
        {"beta": 1.1, "distractor_lapse": 0.30, "weight": 0.25},
    ]
    floored = floor_draws(draws, 0.15)
    assert [row["distractor_lapse"] for row in floored] == [0.15, 0.15, 0.30]
    assert [row["beta"] for row in floored] == [1.0, 0.9, 1.1]
    assert [row["weight"] for row in floored] == [0.5, 0.25, 0.25]
    assert draws[0]["distractor_lapse"] == 0.0


def test_floor_must_be_inside_unit_interval():
    with pytest.raises(ValueError):
        floor_draws([{"beta": 1.0, "distractor_lapse": 0.0, "weight": 1.0}], 0.0)
    with pytest.raises(ValueError):
        floor_draws([{"beta": 1.0, "distractor_lapse": 0.0, "weight": 1.0}], 1.0)


def test_guard_rejects_subfloor_and_promotable_artifacts():
    artifact = {
        "robustnessFloor": 0.15,
        "draws": [{"beta": 1.0, "distractorLapse": 0.1, "weight": 1.0}],
        "promotionForbidden": True,
    }
    with pytest.raises(AssertionError):
        assert_floored(artifact)
    artifact["draws"][0]["distractorLapse"] = 0.15
    artifact["promotionForbidden"] = False
    with pytest.raises(AssertionError):
        assert_floored(artifact)


@pytest.mark.parametrize("floor", FLOORS)
def test_floored_ensemble_is_schema_valid_hash_bound_and_reproducible(floor):
    base = json.loads(
        (ROOT / "artifacts/iiic_conditional_f1_integrated_ensemble9_rd.json").read_text()
    )
    committed = json.loads(
        _floored_path("iiic_conditional_f1_integrated_ensemble9_rd.json", floor).read_text()
    )
    schema = json.loads((ROOT / "schemas/response-artifact.schema.json").read_text())
    jsonschema.validate(committed, schema)
    assert_floored(committed)

    rederived = derive_floored_ensemble(
        base, floor, "iiic_conditional_f1_integrated_ensemble9_rd.json",
    )
    assert committed == rederived

    canonical = json.dumps(
        committed["draws"], sort_keys=True, separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == committed["sha256"]
    assert sum(draw["weight"] for draw in committed["draws"]) == 1
    for floored_draw, base_draw in zip(committed["draws"], base["draws"], strict=True):
        assert floored_draw["beta"] == base_draw["beta"]
        assert floored_draw["weight"] == base_draw["weight"]
        assert floored_draw["distractorLapse"] == max(
            base_draw["distractorLapse"], floor,
        )
    assert committed["qualification"] == "exploratory_unqualified"
    assert committed["promotionForbidden"] is True


@pytest.mark.parametrize("floor", FLOORS)
def test_floored_crossfit_floors_deployable_draws_and_preserves_fit(floor):
    base = json.loads(
        (ROOT / "artifacts/iiic_conditional_f1_crossfit_rd.json").read_text()
    )
    committed = json.loads(
        _floored_path("iiic_conditional_f1_crossfit_rd.json", floor).read_text()
    )
    assert_floored(committed)
    rederived = derive_floored_crossfit(
        base, floor, "iiic_conditional_f1_crossfit_rd.json",
    )
    assert committed == rederived
    # The recorded fit is evidence, not a deployable: it must stay untouched.
    assert committed["crossfit"] == base["crossfit"]
    assert committed["nWrongPicks"] == base["nWrongPicks"]
    for floored_draw, base_draw in zip(
        committed["bootstrap"]["draws"], base["bootstrap"]["draws"], strict=True,
    ):
        assert floored_draw["beta"] == base_draw["beta"]
        assert floored_draw["distractor_lapse"] == max(
            base_draw["distractor_lapse"], floor,
        )
