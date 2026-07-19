"""Drift guards for the shipped PrecisionPolicy guard calibration.

Since 2026-07-18 the DEFAULT is mc-guard v2 (30 MH rejuvenation steps +
requalified inflation 1.3090533918867642), promoted after the
mc_guard_requal mini-OC passed all five gates. The pre-2026-07-18 15-MH
calibration (inflation 1.5962415320776275) is retained as
LEGACY_PRECISION_15MH_PROFILE for rollback only. These guards pin BOTH so a
future edit that silently reverts the promotion, or drifts the legacy
rollback, fails loudly. Guards compare policy-layer values on fixed inputs
only (the anaconda/MKL stack is not bit-reproducible end-to-end; see
calibration/mc_guard_requal/DECISION.md).
"""
from __future__ import annotations

import numpy as np

from precision_policy import (
    FROZEN_PRECISION_PROFILE,
    LEGACY_PRECISION_15MH_PROFILE,
    PRECISION_RADIUS_MCSE_INFLATION,
    PRECISION_RADIUS_MCSE_INFLATION_30MH,
    PrecisionPolicy,
    build_frozen_precision_policy,
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


def test_shipped_default_is_mc_guard_v2_30mh() -> None:
    profile = FROZEN_PRECISION_PROFILE
    # Policy IDENTITY / selection key stays stable across the recalibration.
    assert profile.name == "precision_v1"
    assert profile.n_mh_steps == 30
    assert profile.session_kwargs()["n_mh_steps"] == 30
    assert profile.radius_mcse_inflation == PRECISION_RADIUS_MCSE_INFLATION_30MH
    assert profile.radius_mcse_inflation == 1.3090533918867642
    policy = build_frozen_precision_policy(FrozenInputs())
    assert policy.radius_mcse_inflation == 1.3090533918867642
    assert policy.radius_mcse_z == 1.645
    assert (
        policy.radius_mcse_z * policy.radius_mcse_inflation
        == 2.1533928296537272
    )


def test_legacy_15mh_profile_retained_for_rollback() -> None:
    legacy = LEGACY_PRECISION_15MH_PROFILE
    assert legacy.name == "precision_v1_legacy_15mh"
    assert legacy.n_mh_steps == 15
    assert legacy.session_kwargs()["n_mh_steps"] == 15
    assert legacy.radius_mcse_inflation == PRECISION_RADIUS_MCSE_INFLATION
    assert legacy.radius_mcse_inflation == 1.5962415320776275
    policy = build_frozen_precision_policy(FrozenInputs(), profile=legacy)
    assert policy.radius_mcse_inflation == 1.5962415320776275


def test_default_and_legacy_differ_only_in_mh_and_inflation() -> None:
    default = FROZEN_PRECISION_PROFILE.session_kwargs()
    legacy = LEGACY_PRECISION_15MH_PROFILE.session_kwargs()
    assert default.pop("n_mh_steps") == 30
    assert legacy.pop("n_mh_steps") == 15
    assert default == legacy
    p_def = build_frozen_precision_policy(FrozenInputs())
    p_leg = build_frozen_precision_policy(
        FrozenInputs(), profile=LEGACY_PRECISION_15MH_PROFILE)
    for attr in (
        "reliability_mode",
        "precision_statistic",
        "persistence",
        "band_min",
        "radius_mcse_z",
        "n_min",
        "per_domain_cap",
    ):
        assert getattr(p_def, attr) == getattr(p_leg, attr), attr
    assert np.array_equal(p_def.r_l, p_leg.r_l)
    assert p_def.interval_radius_scale_by_domain is None
    assert p_def.interval_radius_ramp_max_by_domain is None
    assert p_def.bias_radius_scale_by_domain is None


def test_guard_margin_scales_exactly_with_inflation_on_fixed_cloud() -> None:
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
    rng = np.random.default_rng(61_999_997)
    values = rng.standard_normal(400) * 0.05
    weights = np.abs(rng.standard_normal(400))
    weights = weights / weights.sum()
    state = {"t": np.zeros((400, 1)), "l": values[:, None], "w": weights}
    telemetry = {
        "remaining_bank_counts": [20],
        "band_administered": [[0, 0, 0]],
        "band_remaining": [[7, 7, 6]],
    }
    diags = {}
    for label, inflation in (
        ("default_30mh", PRECISION_RADIUS_MCSE_INFLATION_30MH),
        ("legacy_15mh", PRECISION_RADIUS_MCSE_INFLATION),
    ):
        policy = PrecisionPolicy(
            **kwargs, radius_mcse_inflation=inflation)
        policy.reset(1)
        policy(state, telemetry, [3], 1)
        diags[label] = policy.last_diagnostics
    d30, d15 = diags["default_30mh"], diags["legacy_15mh"]
    # identical cloud -> identical raw statistic and MCSE; only the recorded
    # inflation (hence the guard margin) differs, and exactly so.
    assert (d30["skill_point_centered_radius"]
            == d15["skill_point_centered_radius"])
    assert (d30["skill_point_centered_radius_mcse"]
            == d15["skill_point_centered_radius_mcse"])
    assert d30["precision_statistic_mcse_inflation"] == 1.3090533918867642
    assert d15["precision_statistic_mcse_inflation"] == 1.5962415320776275
