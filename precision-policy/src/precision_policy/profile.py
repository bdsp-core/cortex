from __future__ import annotations

from dataclasses import dataclass

from .policy import (
    PRECISION_BAND_MIN,
    PRECISION_ESS_FLOOR_FRACTION,
    PRECISION_RADIUS_MCSE_INFLATION,
    PRECISION_RADIUS_MCSE_INFLATION_30MH,
    PRECISION_RADIUS_MCSE_Z,
    PRECISION_SURROGATE_ACCEPTANCE_FLOOR,
    PRECISION_SURROGATE_ANCESTRY_FLOOR,
    PRECISION_TASK_CODES,
    PrecisionPolicy,
)


# Shipped guard calibration (mc-guard v2, promoted to default 2026-07-18 after
# the mc_guard_requal mini-OC passed all gates): 30 MH rejuvenation steps with
# the requalified MCSE inflation. The 15-MH calibration is retained below as
# LEGACY_PRECISION_15MH_PROFILE for rollback/provenance only. "precision_v1"
# stays the POLICY IDENTITY / selection key — the guard recalibration is
# internal to that policy, not a new stopping rule.
DEFAULT_N_MH_STEPS = 30
DEFAULT_RADIUS_MCSE_INFLATION = PRECISION_RADIUS_MCSE_INFLATION_30MH


@dataclass(frozen=True)
class FrozenPrecisionProfile:
    name: str = "precision_v1"
    task_codes: tuple[str, ...] = PRECISION_TASK_CODES
    n_particles: int = 1200
    skill_correlation: str = "Corr_l"
    bias_correlation: str = "Corr_t"
    objective: str = "total_var"
    n_subsample: int = 128
    uncertainty_aware_subsample: bool = True
    first_item_topn: int = 10
    max_consecutive_same_domain: int = 5
    floor_progress_deadline: bool = True
    per_domain_cap: int = 60
    reliability_mode: str = "quantile_mcse"
    precision_statistic: str = "point_centered_radius"
    n_mh_steps: int = DEFAULT_N_MH_STEPS
    radius_mcse_inflation: float = DEFAULT_RADIUS_MCSE_INFLATION
    # Evidence floor + declaration persistence. Defaults are the shipped
    # values; the c1 recalibration (n_min=0, persistence=3) is served only
    # when the server stamps a sitting with recalibration "c1"
    # (CORTEX_PRECISION_C1_ROLLOUT). "precision_v1" stays the policy
    # identity — c1 recalibrates two criteria constants inside it.
    n_min: int = 20
    persistence: int = 2

    @property
    def maximum_k7_questions(self) -> int:
        return len(self.task_codes) * self.per_domain_cap

    def session_kwargs(self) -> dict:
        """Exact controller settings that must accompany the frozen policy."""

        return {
            "n_particles": self.n_particles,
            "max_questions": None,
            "selection": "adaptive",
            "objective": self.objective,
            "n_subsample": self.n_subsample,
            "uncertainty_aware_subsample": self.uncertainty_aware_subsample,
            "first_item_topn": self.first_item_topn,
            "max_consecutive_same_domain": self.max_consecutive_same_domain,
            "floor_progress_deadline": self.floor_progress_deadline,
            "per_domain_cap": self.per_domain_cap,
            "ess_threshold_frac": 0.5,
            "n_mh_steps": self.n_mh_steps,
            "extended_data_collection": False,
        }


FROZEN_PRECISION_PROFILE = FrozenPrecisionProfile()

# Legacy 15-MH guard calibration (inflation 1.5962415320776275), the shipped
# default before 2026-07-18. Retained for rollback and provenance ONLY; it is
# no longer the default. Its own drift history lives in
# cortex_web_python_reference/calibration/precision_frontier/
# radius_mcse_qualification.json.
LEGACY_PRECISION_15MH_PROFILE = FrozenPrecisionProfile(
    name="precision_v1_legacy_15mh",
    n_mh_steps=15,
    radius_mcse_inflation=PRECISION_RADIUS_MCSE_INFLATION,
)

# The c1 stopping recalibration (2026-08-02): evidence floor n_min 20->0 with
# declaration persistence 2->3 as the compensating tightening. Qualified by
# the nmin-stopping-study (stage-2/3 CRN campaigns + locked reserved-seed
# confirmation); band floor, tolerances, guard, and cap are unchanged. Served
# per sitting via the server-side c1 rollout; FROZEN_PRECISION_PROFILE stays
# the default until the rollout is promoted to "all".
PRECISION_C1_PROFILE = FrozenPrecisionProfile(
    n_min=0,
    persistence=3,
)


def build_frozen_precision_policy(
    inputs, profile: FrozenPrecisionProfile = FROZEN_PRECISION_PROFILE
) -> PrecisionPolicy:
    """Build the immutable, uncalibrated `frontier_p90guard_m3` policy.

    Defaults to the shipped guard calibration (mc-guard v2: 30 MH steps +
    requalified inflation). Pass `profile=LEGACY_PRECISION_15MH_PROFILE` to
    reconstruct the pre-2026-07-18 15-MH guard for rollback.
    """

    task_codes = tuple(str(code) for code in inputs.task_codes)
    if task_codes != profile.task_codes:
        raise ValueError(
            f"{profile.name} requires task order {profile.task_codes}, got {task_codes}"
        )
    if not hasattr(inputs, "Corr_t"):
        raise ValueError(f"{profile.name} requires the frozen Corr_t bias correlation")
    policy = PrecisionPolicy.from_inputs(
        inputs,
        confidence=0.95,
        n_min=profile.n_min,
        per_domain_cap=profile.per_domain_cap,
        persistence=profile.persistence,
        band_min=PRECISION_BAND_MIN,
        reliability_mode=profile.reliability_mode,
        ess_floor_fraction=PRECISION_ESS_FLOOR_FRACTION,
        min_rejuvenation_acceptance=PRECISION_SURROGATE_ACCEPTANCE_FLOOR,
        min_distinct_ancestor_fraction=PRECISION_SURROGATE_ANCESTRY_FLOOR,
        radius_mcse_z=PRECISION_RADIUS_MCSE_Z,
        radius_mcse_inflation=profile.radius_mcse_inflation,
        precision_statistic=profile.precision_statistic,
    )
    if any(
        value is not None
        for value in (
            policy.interval_radius_scale_by_domain,
            policy.interval_radius_ramp_max_by_domain,
            policy.bias_radius_scale_by_domain,
        )
    ):
        raise AssertionError("the frozen PrecisionPolicy must be uncalibrated")
    return policy
