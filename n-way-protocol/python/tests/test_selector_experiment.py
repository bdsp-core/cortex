from __future__ import annotations

import numpy as np

from nway_protocol.selector_experiment import SelectorExperimentConfig, run


def test_selector_experiment_smoke_reports_gated_paired_comparisons():
    config = SelectorExperimentConfig(
        replicates=2,
        particles=48,
        mh_steps=1,
        own_cap=1,
        bank_segments=12,
        selector_per_domain=1,
        fisher_per_domain=1,
        seed_base=63_590_000,
    )
    rows, result = run(config, workers=1)
    assert len(rows) == config.replicates * len(config.selectors)
    assert set(result["selectors"]) == set(config.selectors)
    assert set(result["paired_vs_frozen"]) == set(config.selectors[1:])
    for comparison in result["paired_vs_frozen"].values():
        assert np.isfinite(comparison["skill_width_ratio"])
        assert np.isfinite(comparison["bias_width_ratio"])
