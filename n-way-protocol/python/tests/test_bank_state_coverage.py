from __future__ import annotations

import numpy as np

from nway_protocol.bank_state_coverage import allocation, coverage


def test_allocation_rows_are_distributions_over_the_five_distractors():
    axes = np.asarray([
        [0.0, 2.0, -1.0, 0.5, -0.5, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    distractors, q = allocation(axes, 3, beta=1.0, lapse=0.0)
    assert distractors == [1, 2, 4, 5, 6]
    np.testing.assert_allclose(q.sum(axis=1), 1.0, rtol=1e-12)
    # Flat axes give exactly the uniform allocation.
    np.testing.assert_allclose(q[1], 0.2, rtol=1e-12)
    # The strongest distractor axis is modal.
    assert distractors[int(np.argmax(q[0]))] == 1


def test_flooring_shrinks_identity_information_monotonically():
    rng = np.random.default_rng(11)
    axes = rng.normal(0, 1, size=(200, 7))
    result = coverage(axes, floors=(0.0, 0.15, 0.20))
    medians = [
        result["by_floor"][key]["by_asked_class"]["sz"]["median_kl_vs_uniform_nats"]
        for key in ("0.0", "0.15", "0.2")
    ]
    assert medians == sorted(medians, reverse=True)
    assert all(value >= 0 for value in medians)


def test_coverage_flags_empty_modal_pairs():
    # One dominant axis everywhere: class 1 is modal for every asked class,
    # so most ordered pairs have no modal support and must be flagged.
    axes = np.zeros((50, 7))
    axes[:, 1] = 3.0
    result = coverage(axes, floors=(0.0,))
    unfloored = result["by_floor"]["0.0"]
    assert "lpd->gpd" in unfloored["empty_modal_pairs"]
    assert not any(pair.endswith("->sz") for pair in unfloored["empty_modal_pairs"])
