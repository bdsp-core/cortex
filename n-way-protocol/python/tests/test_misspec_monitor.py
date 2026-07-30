from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from nway_protocol.misspec_monitor import (
    CusumMonitor,
    IIIC_GROUP,
    increment,
    mixture_distractor_probabilities,
    operating_characteristics,
)

Z = np.asarray([0.0, 0.9, -0.3, 0.4, -1.1, 0.6, -0.2])
DRAWS = ((0.9, 0.15, 0.5), (1.1, 0.20, 0.5))


def test_mixture_matches_hand_computed_conditional_allocation():
    asked = 3
    distractors = [k for k in IIIC_GROUP if k != asked]
    expected = np.zeros(5)
    for beta, lapse, weight in DRAWS:
        logits = beta * Z[distractors]
        softmax = np.exp(logits - logsumexp(logits))
        expected += (weight / sum(d[2] for d in DRAWS)) * (
            lapse / 5 + (1 - lapse) * softmax
        )
    computed = mixture_distractor_probabilities(asked, Z, DRAWS)
    np.testing.assert_allclose(computed, expected, rtol=1e-12)
    assert computed.sum() == pytest.approx(1.0, rel=1e-12)


def test_increment_sign_separates_model_from_uniform_world():
    asked = 1
    distractors = [k for k in IIIC_GROUP if k != asked]
    probabilities = mixture_distractor_probabilities(asked, Z, DRAWS)
    model_expectation = sum(
        probabilities[i] * increment(asked, pick, Z, DRAWS)
        for i, pick in enumerate(distractors)
    )
    uniform_expectation = sum(
        0.2 * increment(asked, pick, Z, DRAWS) for pick in distractors
    )
    # KL(model||uniform) > 0 and -KL(uniform||model) < 0.
    assert model_expectation > 0
    assert uniform_expectation < 0


def test_increment_rejects_correct_picks():
    with pytest.raises(ValueError):
        increment(2, 2, Z, DRAWS)


def test_floored_increment_is_bounded_below_by_log_floor():
    floor = 0.15
    draws = ((1.0, floor, 1.0),)
    worst = min(
        increment(asked, pick, 3.0 * Z, draws)
        for asked in IIIC_GROUP
        for pick in IIIC_GROUP
        if pick != asked
    )
    assert worst >= np.log(floor)


def test_cusum_trips_one_way_and_records_latency():
    monitor = CusumMonitor(threshold=1.0)
    assert not monitor.observe(0.8)
    assert monitor.statistic == 0.0  # positive evidence never builds credit
    assert not monitor.observe(-0.7)
    assert not monitor.tripped
    assert monitor.observe(-0.7)
    assert monitor.tripped and monitor.tripped_at == 3
    # Tripped state is absorbing regardless of later evidence.
    assert monitor.observe(5.0)
    assert monitor.tripped_at == 3


def test_operating_characteristics_calibrates_and_detects():
    rng = np.random.default_rng(7)
    axes = rng.normal(0, 1.0, size=(60, 7))
    result = operating_characteristics(
        axes, DRAWS, sessions=300, wrong_picks=40, false_trip_target=0.02,
    )
    assert result["null_false_trip_rate"] <= 0.05
    assert result["null_mean_increment"] > 0
    uniform = result["alternatives"]["uniform_n3b"]
    assert uniform["trip_rate"] > 0.9
    assert uniform["median_wrong_picks_to_trip"] < 40
