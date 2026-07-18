from __future__ import annotations

from dataclasses import dataclass

from .policy import (
    PRECISION_BAND_MIN,
    PRECISION_ESS_FLOOR_FRACTION,
    PRECISION_RADIUS_MCSE_INFLATION,
    PRECISION_RADIUS_MCSE_Z,
    PRECISION_SURROGATE_ACCEPTANCE_FLOOR,
    PRECISION_SURROGATE_ANCESTRY_FLOOR,
    PRECISION_TASK_CODES,
    PrecisionPolicy,
)


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
            "n_mh_steps": 15,
            "extended_data_collection": False,
        }


FROZEN_PRECISION_PROFILE = FrozenPrecisionProfile()


def build_frozen_precision_policy(inputs) -> PrecisionPolicy:
    """Build the immutable, uncalibrated `frontier_p90guard_m3` policy."""

    profile = FROZEN_PRECISION_PROFILE
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
        n_min=20,
        per_domain_cap=profile.per_domain_cap,
        persistence=2,
        band_min=PRECISION_BAND_MIN,
        reliability_mode=profile.reliability_mode,
        ess_floor_fraction=PRECISION_ESS_FLOOR_FRACTION,
        min_rejuvenation_acceptance=PRECISION_SURROGATE_ACCEPTANCE_FLOOR,
        min_distinct_ancestor_fraction=PRECISION_SURROGATE_ANCESTRY_FLOOR,
        radius_mcse_z=PRECISION_RADIUS_MCSE_Z,
        radius_mcse_inflation=PRECISION_RADIUS_MCSE_INFLATION,
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
