"""Production-faithful ACTIVE-domain eligibility for precision-mode selection.

The research harness historically offered every under-cap domain to the
selector until the safety cap; production restricts selection to domains the
Precision policy reports ACTIVE (advance.ts:283,399) with the variety
semantics of session.ts:71. `_precision_eligible` mirrors that; these tests
pin the mirrored semantics and the own-cap no-op.
"""
from __future__ import annotations

import numpy as np

from nway_protocol.qualification import (
    IIIC_GROUP,
    MAX_CONSECUTIVE_SAME_DOMAIN,
    QualificationConfig,
    _precision_eligible,
    _select,
    run_replicate,
)


def _result(states: list[str]) -> dict:
    return {"selectionStates": states}


CONFIG = QualificationConfig(own_cap=60, stopping="precision")


def test_first_question_offers_every_iiic_domain():
    counts = np.zeros(7, dtype=int)
    assert _precision_eligible(None, counts, CONFIG, -1, 0) == list(IIIC_GROUP)


def test_only_active_domains_are_offered():
    counts = np.zeros(7, dtype=int)
    states = ["UNDETERMINABLE_BANK", "ACTIVE", "ESTIMATE_COMPLETE", "ACTIVE",
              "UNDETERMINABLE_CAP", "ESTIMATE_COMPLETE", "ACTIVE"]
    assert _precision_eligible(_result(states), counts, CONFIG, 1, 1) == [1, 3, 6]


def test_domains_at_cap_are_never_offered():
    counts = np.zeros(7, dtype=int)
    counts[3] = CONFIG.own_cap
    states = ["UNDETERMINABLE_BANK"] + ["ACTIVE"] * 6
    assert 3 not in _precision_eligible(_result(states), counts, CONFIG, 1, 1)


def test_variety_prefers_other_active_domains_after_streak():
    counts = np.zeros(7, dtype=int)
    states = ["UNDETERMINABLE_BANK"] + ["ACTIVE"] * 6
    eligible = _precision_eligible(
        _result(states), counts, CONFIG, 2, MAX_CONSECUTIVE_SAME_DOMAIN,
    )
    assert eligible == [1, 3, 4, 5, 6]


def test_variety_reopens_estimate_complete_when_streak_domain_is_only_active():
    counts = np.zeros(7, dtype=int)
    states = ["UNDETERMINABLE_BANK", "ESTIMATE_COMPLETE", "ACTIVE",
              "ESTIMATE_COMPLETE", "UNDETERMINABLE_CAP", "UNDETERMINABLE_CAP",
              "ESTIMATE_COMPLETE"]
    eligible = _precision_eligible(
        _result(states), counts, CONFIG, 2, MAX_CONSECUTIVE_SAME_DOMAIN,
    )
    # The streak domain (2) is the only ACTIVE one: an ESTIMATE_COMPLETE
    # domain absorbs the variety question (and may consequently reopen).
    assert eligible == [1, 3, 6]


def test_terminal_domains_are_never_revived_by_variety():
    counts = np.zeros(7, dtype=int)
    states = ["UNDETERMINABLE_BANK", "UNDETERMINABLE_CAP", "ACTIVE",
              "UNDETERMINABLE_CAP", "UNDETERMINABLE_CAP", "UNDETERMINABLE_CAP",
              "UNDETERMINABLE_CAP"]
    eligible = _precision_eligible(
        _result(states), counts, CONFIG, 2, MAX_CONSECUTIVE_SAME_DOMAIN,
    )
    # No other ACTIVE domain and no ESTIMATE_COMPLETE variety target: the
    # streak domain remains the only offer, exactly as advance.ts leaves
    # `allowed` at the active list.
    assert eligible == [2]


def test_no_eligible_domain_when_everything_is_terminal():
    counts = np.zeros(7, dtype=int)
    states = ["UNDETERMINABLE_BANK"] + ["UNDETERMINABLE_CAP"] * 6
    assert _precision_eligible(_result(states), counts, CONFIG, 2, 1) == []


def test_select_honors_the_eligibility_restriction():
    rng = np.random.default_rng(1)

    class Cloud:
        t = rng.normal(size=(16, 7))
        l = rng.normal(size=(16, 7))
        w = np.full(16, 1 / 16)

    config = QualificationConfig(own_cap=4, selector_per_domain=2)
    s_mean = rng.normal(size=(8, 7))
    s_sd = np.full((8, 7), 0.1)
    counts = np.zeros(7, dtype=int)
    asked, _ = _select(
        Cloud(), "categorical_f1", set(range(8)), counts, s_mean, s_sd,
        config, eligible=[4],
    )
    assert asked == 4


def test_own_cap_rows_are_unchanged_by_the_eligibility_fix():
    # The historical own-cap path must stay byte-identical: eligibility is
    # only derived under precision stopping.
    config = QualificationConfig(replicates=1, particles=32, own_cap=2,
                                 bank_segments=24, selector_per_domain=2,
                                 mh_steps=1)
    rows = run_replicate(62_100_555, config)
    assert all(row.end_statuses is None for row in rows)
    assert all(row.questions == 12 for row in rows)
