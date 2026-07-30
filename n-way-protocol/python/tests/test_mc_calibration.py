from __future__ import annotations

import numpy as np

from nway_protocol.mc_calibration import (
    MCCalibrationConfig,
    ReplayProfile,
    _bootstrap_mcse,
    run,
)
from nway_protocol.reference import make_cloud


def test_particle_bootstrap_mcse_is_finite():
    cloud = make_cloud(64, np.eye(7), np.eye(7), np.random.default_rng(11))
    mean_mcse, radius_mcse = _bootstrap_mcse(
        cloud, cloud.l[:, 2], np.random.default_rng(12), 12,
    )
    assert np.isfinite(mean_mcse) and mean_mcse > 0
    assert np.isfinite(radius_mcse) and radius_mcse > 0


def test_fixed_history_mc_calibration_smoke():
    config = MCCalibrationConfig(
        histories=2,
        own_cap=1,
        bank_segments=12,
        selector_per_domain=1,
        history_particles=48,
        history_mh_steps=1,
        candidate=ReplayProfile("candidate", 48, 1, 2),
        reference=ReplayProfile("reference", 72, 2, 2),
        bootstrap_replicates=6,
        seed_base=63_190_000,
    )
    rows, result = run(config, workers=1)
    assert rows
    assert result["status"] == "research_only_not_promoted"
    assert set(result["parameters"]) == {"skill", "bias"}
    for metrics in result["parameters"].values():
        assert np.isfinite(metrics["selected_radius_inflation_q95"])
        assert 0 <= metrics["validation_radius_upper_coverage"] <= 1
