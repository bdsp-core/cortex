from dataclasses import replace

import numpy as np

from nway_protocol.qualification import QualificationConfig, run_replicate

TINY = QualificationConfig(
    particles=48, own_cap=2, bank_segments=24, selector_per_domain=3,
    mh_steps=1, formal_sbc=True, truth_t_mean=0.0, truth_l_mean=0.0,
    truth_sd=1.0,
)


def test_sigma_zero_lognormal_equals_fixed_truth_beta() -> None:
    mu = 0.25
    fixed = run_replicate(9_000_001, replace(TINY, truth_beta=float(np.exp(mu))))
    drawn = run_replicate(9_000_001, replace(TINY, truth_beta_lognormal=(mu, 0.0)))
    assert [row.__dict__ for row in fixed] == [row.__dict__ for row in drawn]


def test_continuum_replicate_is_deterministic() -> None:
    config = replace(TINY, truth_beta_lognormal=(0.0, 0.3))
    first = run_replicate(9_000_002, config)
    second = run_replicate(9_000_002, config)
    assert [row.__dict__ for row in first] == [row.__dict__ for row in second]
