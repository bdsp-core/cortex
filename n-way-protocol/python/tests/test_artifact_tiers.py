from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from nway_protocol.artifact_rd import DistractorRecord
from nway_protocol.artifact_tiers import stratify, tier_records

GROUP = (1, 2, 3, 4, 5, 6)


def _records(rng, tier: str, readers: int, per_reader: int, beta: float):
    rows = []
    for reader in range(readers):
        for index in range(per_reader):
            z = tuple(float(v) for v in np.concatenate([[0.0], rng.normal(0, 1, 6)]))
            asked = int(rng.integers(1, 7))
            distractors = [k for k in GROUP if k != asked]
            logits = beta * np.asarray([z[k] for k in distractors])
            probabilities = np.exp(logits - logsumexp(logits))
            pick = int(rng.choice(distractors, p=probabilities))
            rows.append(DistractorRecord(
                reader_id=f"{tier}-{reader}",
                source_id=f"case-{tier}-{reader}-{index}",
                asked_k=asked,
                raw_pick=pick,
                group=GROUP,
                z=z,
            ))
    return rows


def test_tier_records_filters_by_reader_prefix():
    rng = np.random.default_rng(3)
    records = _records(rng, "expert", 2, 5, 1.0) + _records(rng, "novice", 3, 5, 1.0)
    assert len(tier_records(records, "pooled")) == 25
    assert len(tier_records(records, "expert")) == 10
    assert all(
        record.reader_id.startswith("novice-")
        for record in tier_records(records, "novice")
    )


def test_stratify_recovers_a_steeper_expert_beta():
    rng = np.random.default_rng(5)
    records = (
        _records(rng, "expert", 6, 80, 1.6) + _records(rng, "novice", 10, 80, 0.8)
    )
    result = stratify(
        records, ensemble_betas=[0.85, 1.05], bootstrap_draws=8, folds=3,
    )
    tiers = result["tiers"]
    assert tiers["expert"]["full_fit"]["beta"] > tiers["novice"]["full_fit"]["beta"]
    assert (
        tiers["expert"]["bootstrap"]["beta_mass_above_ensemble_span"]
        > tiers["novice"]["bootstrap"]["beta_mass_above_ensemble_span"]
    )
    assert tiers["pooled"]["n_wrong_picks"] == len(records)
    assert result["promotionForbidden"] is True
    crossfit = tiers["expert"]["reader_held_out_crossfit"]
    assert "error" in crossfit or len(crossfit["folds"]) >= 2
    assert tiers["expert"]["full_fit"]["beta"] == pytest.approx(1.6, abs=0.35)
