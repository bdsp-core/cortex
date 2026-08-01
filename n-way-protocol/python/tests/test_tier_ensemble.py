from __future__ import annotations

import numpy as np

from nway_protocol.tier_ensemble import build
from test_artifact_tiers import _records


def test_build_spans_tiers_floors_lapses_and_is_deterministic():
    rng = np.random.default_rng(9)
    records = (
        _records(rng, "expert", 6, 60, 1.6) + _records(rng, "novice", 10, 60, 0.8)
    )
    payload = build(records, bootstrap_draws=12)
    draws = payload["bootstrap"]["draws"]
    assert len(draws) == 9
    betas = [draw["beta"] for draw in draws]
    assert betas == sorted(betas)
    # Draws must reach both tiers rather than huddling around the pooled fit.
    assert min(betas) < 1.1
    assert max(betas) > 1.3
    assert all(draw["distractor_lapse"] >= 0.15 for draw in draws)
    assert sum(draw["weight"] for draw in draws) == 1.0
    assert payload["promotionForbidden"] is True

    again = build(records, bootstrap_draws=12)
    assert again["bootstrap"]["draws"] == draws
