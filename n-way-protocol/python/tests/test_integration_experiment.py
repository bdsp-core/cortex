from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nway_protocol.integration_experiment import (
    FROZEN_ARM,
    INTEGRATED_ARM,
    IntegrationExperimentConfig,
    run,
)

ROOT = Path(__file__).resolve().parents[2]


def test_matched_integrated_experiment_smoke():
    artifact = json.loads(
        (ROOT / "artifacts/iiic_conditional_f1_integrated_ensemble9_rd.json").read_text()
    )
    draws = tuple(
        (draw["beta"], draw["distractorLapse"], draw["weight"])
        for draw in artifact["draws"]
    )
    config = IntegrationExperimentConfig(
        replicates=2,
        particles=48,
        mh_steps=1,
        own_cap=1,
        bank_segments=12,
        selector_per_domain=1,
        fisher_per_domain=1,
        integrated_draws=draws,
        seed_base=63_890_000,
    )
    rows, result = run(config, workers=1)
    assert len(rows) == 4
    assert set(result["arms"]) == {FROZEN_ARM, INTEGRATED_ARM}
    assert sum(result["truth_draw_counts"].values()) == 2
    for parameter in ("skill", "bias"):
        paired = result["paired_integrated_minus_frozen"]
        assert np.isfinite(paired[f"{parameter}_width_ratio"]["mean"])
        assert np.isfinite(paired[f"{parameter}_coverage_difference"]["mean"])
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    assert all(
        len({row["truth_artifact_draw_index"] for row in pair}) == 1
        for pair in by_seed.values()
    )
