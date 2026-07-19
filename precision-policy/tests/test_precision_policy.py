from __future__ import annotations

import numpy as np
import pytest

from cortex_termination_policy import TerminationPolicy
from precision_policy import (
    ACTIVE,
    DETERMINED,
    ESTIMATE_COMPLETE,
    FROZEN_PRECISION_PROFILE,
    PRECISION_RADIUS_MCSE_INFLATION_30MH,
    PRECISION_RADIUS_MCSE_Z,
    UNDETERMINABLE_BANK,
    UNDETERMINABLE_CAP,
    PrecisionPolicy,
    build_frozen_precision_policy,
    classify_determined_interval_against_cut,
)


def cloud(width: float, shift: float = 0.0, n: int = 200) -> dict:
    return {
        "t": np.zeros((n, 1)),
        "l": (np.linspace(-width, width, n) + shift)[:, None],
        "w": np.full(n, 1.0 / n),
    }


def telemetry(*, remaining=20, administered=None, band_remaining=None):
    return {
        "remaining_bank_counts": [remaining],
        "band_administered": [administered or [0, 0, 0]],
        "band_remaining": [band_remaining or [7, 7, 6]],
    }


def small_policy(**overrides) -> PrecisionPolicy:
    kwargs = {
        "var_prior": [1.0],
        "r_l": [0.10],
        "n_min": 2,
        "per_domain_cap": 4,
        "persistence": 2,
        "band_min": 0,
        "ess_floor_fraction": 0.5,
        "min_rejuvenation_acceptance": None,
        "min_distinct_ancestor_fraction": None,
    }
    kwargs.update(overrides)
    return PrecisionPolicy(**kwargs)


def test_policy_is_standalone_reversible_and_cap_ordered() -> None:
    policy = small_policy()
    assert isinstance(policy, TerminationPolicy)
    policy.reset(1)
    assert policy(cloud(0.04), telemetry(), [2], 1).domain_statuses == [ACTIVE]
    assert policy(cloud(0.04), telemetry(), [2], 1).domain_statuses == [
        ESTIMATE_COMPLETE
    ]
    assert policy(cloud(0.8), telemetry(), [3], 1).domain_statuses == [ACTIVE]
    assert policy(cloud(0.8), telemetry(), [4], 1).domain_statuses == [
        UNDETERMINABLE_CAP
    ]


def test_trial_zero_bank_feasibility_is_fail_closed() -> None:
    policy = small_policy(n_min=5, per_domain_cap=10)
    decision = policy.observe_bank_feasibility(telemetry(remaining=4), [0], 1)
    assert decision.stop
    assert decision.domain_statuses == [UNDETERMINABLE_BANK]


def test_interval_and_stopping_statistic_are_distinct() -> None:
    policy = small_policy(r_l=[0.5])
    policy.reset(1)
    diag = policy(cloud(0.15, shift=0.2), telemetry(), [2], 1).diagnostics
    low, high = diag["skill_intervals"][0]
    mean = diag["skill_posterior_mean"][0]
    assert diag["skill_point_centered_radius"][0] == pytest.approx(
        max(mean - low, high - mean)
    )
    assert diag["precision_statistic"] == "point_centered_radius"


def test_determined_only_reporting_boundary_rejects_terminal_statuses() -> None:
    assert classify_determined_interval_against_cut(
        DETERMINED, [0.4, 0.8], 0.3
    ) == "ABOVE_CUT"
    with pytest.raises(ValueError, match="DETERMINED"):
        classify_determined_interval_against_cut(
            UNDETERMINABLE_CAP, [0.4, 0.8], 0.3
        )


class FrozenInputs:
    task_codes = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    Corr_l = np.eye(7)
    Corr_t = np.eye(7)

    def as_engine_arrays(self):
        signals = [np.linspace(-2.0, 2.0, 30) for _ in range(7)]
        return signals, [np.zeros(30) for _ in range(7)], [
            np.arange(30) for _ in range(7)
        ]


def test_frozen_factory_pins_policy_and_controller_profile_without_g() -> None:
    policy = build_frozen_precision_policy(FrozenInputs())
    profile = FROZEN_PRECISION_PROFILE
    assert profile.maximum_k7_questions == 420
    assert profile.session_kwargs() == {
        "n_particles": 1200,
        "max_questions": None,
        "selection": "adaptive",
        "objective": "total_var",
        "n_subsample": 128,
        "uncertainty_aware_subsample": True,
        "first_item_topn": 10,
        "max_consecutive_same_domain": 5,
        "floor_progress_deadline": True,
        "per_domain_cap": 60,
        "ess_threshold_frac": 0.5,
        "n_mh_steps": 30,
        "extended_data_collection": False,
    }
    assert policy.reliability_mode == "quantile_mcse"
    assert policy.precision_statistic == "point_centered_radius"
    assert policy.persistence == 2
    assert policy.band_min == 3
    assert policy.radius_mcse_z == PRECISION_RADIUS_MCSE_Z == 1.645
    # Shipped default is the mc-guard v2 (30-MH) guard inflation since
    # 2026-07-18; the legacy 15-MH value is pinned in
    # test_precision_guard_default.py.
    assert (
        policy.radius_mcse_inflation
        == PRECISION_RADIUS_MCSE_INFLATION_30MH
        == 1.3090533918867642
    )
    assert policy.interval_radius_scale_by_domain is None
    assert policy.interval_radius_ramp_max_by_domain is None
    assert policy.bias_radius_scale_by_domain is None
    assert not hasattr(policy, "ell_star")
