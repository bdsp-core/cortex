"""Drift guards for the c1 stopping recalibration profile (2026-08-02).

c1 = evidence floor n_min 20->0 with declaration persistence 2->3 as the
compensating tightening (band floor, tolerances, guard, and cap unchanged).
Qualified by the nmin-stopping-study CRN campaigns plus the locked
reserved-seed confirmation. Served per sitting via the server-side c1
rollout; FROZEN_PRECISION_PROFILE stays the shipped default (20/2) until the
rollout is promoted to "all". These guards pin BOTH configurations so a
silent flip of either fails loudly.
"""
from __future__ import annotations

import numpy as np

from precision_policy import (
    FROZEN_PRECISION_PROFILE,
    PRECISION_C1_PROFILE,
    PRECISION_RADIUS_MCSE_INFLATION_30MH,
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


def test_shipped_default_remains_nmin20_persistence2() -> None:
    assert FROZEN_PRECISION_PROFILE.n_min == 20
    assert FROZEN_PRECISION_PROFILE.persistence == 2
    policy = build_frozen_precision_policy(FrozenInputs())
    assert policy.n_min == 20
    assert policy.persistence == 2


def test_c1_profile_recalibrates_only_nmin_and_persistence() -> None:
    profile = PRECISION_C1_PROFILE
    # Policy IDENTITY / selection key stays stable across the recalibration.
    assert profile.name == "precision_v1"
    assert profile.n_min == 0
    assert profile.persistence == 3
    # Everything else is byte-identical to the shipped profile.
    assert profile.per_domain_cap == FROZEN_PRECISION_PROFILE.per_domain_cap
    assert profile.n_particles == FROZEN_PRECISION_PROFILE.n_particles
    assert profile.n_mh_steps == FROZEN_PRECISION_PROFILE.n_mh_steps
    assert (profile.radius_mcse_inflation
            == PRECISION_RADIUS_MCSE_INFLATION_30MH)
    assert profile.session_kwargs() == FROZEN_PRECISION_PROFILE.session_kwargs()


def test_c1_policy_builds_with_recalibrated_floors() -> None:
    policy = build_frozen_precision_policy(
        FrozenInputs(), profile=PRECISION_C1_PROFILE)
    assert policy.n_min == 0
    assert policy.persistence == 3
    assert policy.per_domain_cap == 60
    assert policy.band_min == 3
    assert policy.radius_mcse_inflation == 1.3090533918867642
    assert policy.radius_mcse_z == 1.645
