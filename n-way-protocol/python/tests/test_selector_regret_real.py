from __future__ import annotations

import numpy as np

from nway_protocol.bank_audit import BankSegment
from nway_protocol.selector_regret_real import RealRegretConfig, run


def test_empirical_axis_regret_smoke():
    bank = [
        BankSegment(
            index, 20_000 + index,
            tuple(np.linspace(-1.2, 1.2, 7) + index * 0.05),
            tuple(np.full(7, 0.04 + (index % 3) * 0.02)),
        )
        for index in range(16)
    ]
    config = RealRegretConfig(
        states=2, particles=48, mh_steps=1, bank_sample=12,
        coarse_per_task=3, entropy_per_task=1, fisher_per_task=1,
        seed_base=63_790_000,
    )
    result = run(bank, config, workers=1)
    assert result["states"] == 2
    assert result["mean_challenger_regret"] <= result["mean_baseline_regret"] + 1e-12
    assert result["mean_challenger_shortlist"] >= result["mean_baseline_shortlist"]
