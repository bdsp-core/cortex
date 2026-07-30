from __future__ import annotations

import numpy as np

from nway_protocol.artifact_rd import (
    ArtifactDraw,
    DistractorRecord,
    bootstrap_artifact,
    crossfit_artifact,
    dual_crossfit_artifact,
    f1_probabilities_mixture,
    fold_assignments,
)
from nway_protocol.reference import binary_p_yes, signal_z

GROUP = (1, 2, 3, 4, 5, 6)


def _records(seed: int = 44) -> list[DistractorRecord]:
    rng = np.random.default_rng(seed)
    records = []
    for reader in range(24):
        source = reader // 2
        for _ in range(8):
            asked = int(rng.choice(GROUP))
            z = rng.normal(size=7)
            distractors = [k for k in GROUP if k != asked]
            logits = 1.25 * z[distractors]
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()
            pick = int(rng.choice(distractors, p=probabilities))
            records.append(DistractorRecord(
                reader_id=f"reader-{reader}",
                source_id=f"source-{source}",
                asked_k=asked,
                raw_pick=pick,
                group=GROUP,
                z=tuple(z),
            ))
    return records


def test_reader_and_source_components_never_cross_folds():
    records = _records()
    assignments = fold_assignments(records, 4, "test")
    for attribute in ("reader_id", "source_id"):
        observed = {}
        for record, fold in zip(records, assignments, strict=True):
            observed.setdefault(getattr(record, attribute), set()).add(int(fold))
        assert all(len(folds) == 1 for folds in observed.values())


def test_crossfit_and_component_bootstrap_are_finite():
    records = _records()
    result = crossfit_artifact(records, folds=4, salt="test")
    assert 0.05 <= result["full_fit"]["beta"] <= 5
    assert np.isfinite(result["crossfit_mean_log_score"])
    draws = bootstrap_artifact(records, draws=4, seed=55)
    assert len(draws) == 4
    assert np.isclose(sum(draw.weight for draw in draws), 1)
    dual = dual_crossfit_artifact(records, folds=4, salt="test-dual")
    assert set(dual) >= {"reader_held_out", "source_held_out"}


def test_artifact_mixture_preserves_focal_binary_marginal():
    t = np.zeros((3, 7))
    l = np.zeros((3, 7))
    s = np.linspace(-1, 1, 7)
    sd = np.full(7, 0.05)
    probabilities = f1_probabilities_mixture(
        t, l, s, sd, 2, GROUP,
        [ArtifactDraw(0.8, 0.02, 0.4), ArtifactDraw(1.4, 0.05, 0.6)],
    )
    assert np.allclose(probabilities.sum(axis=1), 1)
    expected = binary_p_yes(signal_z(l[:, 2], t[:, 2], s[2], sd[2]))
    assert np.allclose(probabilities[:, GROUP.index(2)], expected)
