from __future__ import annotations

from nway_protocol.smc_experiment import SMCExperimentConfig, run


def test_smc_experiment_smoke_reports_every_research_profile():
    config = SMCExperimentConfig(
        histories=2,
        repeats=1,
        particles=48,
        mh_steps=1,
        own_cap=1,
        bank_segments=12,
        selector_per_domain=1,
        seed_base=63_490_000,
    )
    rows, result = run(config, workers=1)
    assert len(rows) == config.histories * len(config.profiles)
    assert set(result["profiles"]) == {profile.name for profile in config.profiles}
    assert result["status"] == "research_only_not_promoted"
