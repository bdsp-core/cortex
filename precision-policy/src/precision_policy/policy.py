from __future__ import annotations

from statistics import NormalDist

import numpy as np

from cortex_termination_policy import StopDecision, TerminationPolicy

ACTIVE = "ACTIVE"
ESTIMATE_COMPLETE = "ESTIMATE_COMPLETE"
UNDETERMINABLE_CAP = "UNDETERMINABLE_CAP"
UNDETERMINABLE_BANK = "UNDETERMINABLE_BANK"
DETERMINED = "DETERMINED"

PRECISION_CONTRACTION_FRACTION = 1.40
PRECISION_TASK_CODES = ("spike", "sz", "lpd", "gpd", "lrda", "grda", "iic")
PRECISION_CONTRACTION_BY_DOMAIN = (1.50, 1.30, 1.35, 1.30, 1.30, 1.30, 1.25)
PRECISION_BAND_MIN = 3
PRECISION_ESS_FLOOR_FRACTION = 0.50
PRECISION_SURROGATE_ACCEPTANCE_FLOOR = 0.20
PRECISION_SURROGATE_ANCESTRY_FLOOR = 0.35
PRECISION_RADIUS_MCSE_Z = 1.645
# Legacy 15-MH guard inflation. Remains the PrecisionPolicy.__init__ default
# (raw-class contract; reproduces the frozen composed-sweep evidence, which is
# produced with 15 MH steps) and backs LEGACY_PRECISION_15MH_PROFILE. The
# SHIPPED profile default is the 30-MH value below.
PRECISION_RADIUS_MCSE_INFLATION = 1.5962415320776275
# Shipped guard inflation (mc-guard v2, default since 2026-07-18): the
# FrozenPrecisionProfile / live precision path uses this with n_mh_steps=30.
# Same 12-prior-histories x 24-replicate protocol as the 1.5962 qualification
# (LOO upper coverage 0.9795 vs 0.9805; mixed uniform+tail robustness panel
# q95 1.3010 did not escalate); mini-OC promotion gate passed all 5 gates.
# Valid ONLY with n_mh_steps=30. Provenance:
# cortex_web_python_reference/calibration/mc_guard_requal/
# radius_mcse_qualification_1200p30mh.json + MINI_OC_REPORT.md
PRECISION_RADIUS_MCSE_INFLATION_30MH = 1.3090533918867642
DEFAULT_N_MIN = 20


def _weighted_quantile(values, weights, q: float) -> float:
    """First weighted-CDF crossing, matching the browser convention."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cdf = np.cumsum(weights[order])
    index = int(np.searchsorted(cdf, float(q), side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _quantile_density(values, weights, q: float, ess_value: float) -> float:
    bandwidth = min(0.02, q / 2.0, (1.0 - q) / 2.0)
    bandwidth = max(bandwidth, min(0.01, 1.0 / np.sqrt(max(ess_value, 1.0))))
    q0 = max(0.0, q - bandwidth)
    q1 = min(1.0, q + bandwidth)
    x0 = _weighted_quantile(values, weights, q0)
    x1 = _weighted_quantile(values, weights, q1)
    span = x1 - x0
    if not np.isfinite(span) or span <= 0.0:
        return float("nan")
    return float((q1 - q0) / span)


def _halfwidth_mcse(
    values, weights, q_low: float, q_high: float, ess_value: float
) -> float:
    f_low = _quantile_density(values, weights, q_low, ess_value)
    f_high = _quantile_density(values, weights, q_high, ess_value)
    if (
        not np.isfinite(f_low)
        or not np.isfinite(f_high)
        or f_low <= 0.0
        or f_high <= 0.0
    ):
        return float("nan")
    n_eff = max(float(ess_value), 1.0)
    var_low = q_low * (1.0 - q_low) / (n_eff * f_low * f_low)
    var_high = q_high * (1.0 - q_high) / (n_eff * f_high * f_high)
    covariance = (
        (min(q_low, q_high) - q_low * q_high) / (n_eff * f_low * f_high)
    )
    return float(np.sqrt(max((var_low + var_high - 2.0 * covariance) / 4.0, 0.0)))


def _point_centered_radius(values, weights, low: float, high: float):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = float(np.sum(weights * values))
    left = float(mean - low)
    right = float(high - mean)
    return mean, max(left, right), left, right


def _point_centered_radius_mcse(
    values, weights, q_low: float, q_high: float, ess_value: float
) -> float:
    """Covariance-aware influence-function MCSE for the stopping radius."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    low = _weighted_quantile(values, weights, q_low)
    high = _weighted_quantile(values, weights, q_high)
    f_low = _quantile_density(values, weights, q_low, ess_value)
    f_high = _quantile_density(values, weights, q_high, ess_value)
    if (
        not np.isfinite(f_low)
        or not np.isfinite(f_high)
        or f_low <= 0.0
        or f_high <= 0.0
    ):
        return float("nan")
    mean = float(np.sum(weights * values))
    if_mean = values - mean
    if_low = (q_low - (values <= low).astype(float)) / f_low
    if_high = (q_high - (values <= high).astype(float)) / f_high
    n_eff = max(float(ess_value), 1.0)

    def standard_error(influence) -> float:
        centered = influence - float(np.sum(weights * influence))
        variance = float(np.sum(weights * centered * centered)) / n_eff
        return float(np.sqrt(max(variance, 0.0)))

    return max(standard_error(if_mean - if_low), standard_error(if_high - if_mean))


class PrecisionPolicy(TerminationPolicy):
    """Cut-independent, reversible stopping on achieved skill precision."""

    uses_domain_status = True

    def __init__(
        self,
        var_prior,
        *,
        contraction_fraction: float = PRECISION_CONTRACTION_FRACTION,
        r_l=None,
        confidence: float = 0.95,
        n_min: int = DEFAULT_N_MIN,
        per_domain_cap: int = 60,
        persistence: int = 2,
        band_edges=None,
        band_min: int = PRECISION_BAND_MIN,
        reliability_mode: str = "quantile_mcse",
        ess_floor_fraction: float = PRECISION_ESS_FLOOR_FRACTION,
        min_rejuvenation_acceptance: float | None = PRECISION_SURROGATE_ACCEPTANCE_FLOOR,
        min_distinct_ancestor_fraction: float | None = PRECISION_SURROGATE_ANCESTRY_FLOOR,
        radius_mcse_z: float = PRECISION_RADIUS_MCSE_Z,
        radius_mcse_inflation: float = PRECISION_RADIUS_MCSE_INFLATION,
        precision_statistic: str = "point_centered_radius",
        interval_radius_scale_by_domain=None,
        interval_radius_ramp_max_by_domain=None,
        interval_radius_ramp_abs_estimate_max: float = 2.0,
        bias_radius_scale_by_domain=None,
    ):
        self.var_prior = np.asarray(var_prior, dtype=float)
        if (
            self.var_prior.ndim != 1
            or len(self.var_prior) == 0
            or np.any(~np.isfinite(self.var_prior))
            or np.any(self.var_prior <= 0.0)
        ):
            raise ValueError("var_prior must be a non-empty positive vector")
        self.prior_sd = np.sqrt(self.var_prior)
        self.confidence = float(confidence)
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.q_low = (1.0 - self.confidence) / 2.0
        self.q_high = 1.0 - self.q_low
        self.prior_halfwidth_multiplier = NormalDist().inv_cdf(
            (1.0 + self.confidence) / 2.0
        )
        self.contraction_fraction = float(contraction_fraction)
        if not 0.0 < self.contraction_fraction < self.prior_halfwidth_multiplier:
            raise ValueError(
                "contraction_fraction must be positive and smaller than the "
                "prior central-interval halfwidth multiplier"
            )
        self.r_l = (
            self.contraction_fraction * self.prior_sd
            if r_l is None
            else np.asarray(r_l, dtype=float)
        )
        if self.r_l.shape != self.var_prior.shape:
            raise ValueError("r_l and var_prior must align by domain")
        prior_halfwidth = self.prior_halfwidth_multiplier * self.prior_sd
        if (
            np.any(~np.isfinite(self.r_l))
            or np.any(self.r_l <= 0.0)
            or np.any(self.r_l >= prior_halfwidth)
        ):
            raise ValueError(
                "each r_l must be positive and strictly narrower than the "
                "prior central credible interval"
            )
        self.contraction_fraction_by_domain = (self.r_l / self.prior_sd).astype(float)
        self.n_min = int(n_min)
        self.per_domain_cap = int(per_domain_cap)
        self.persistence = int(persistence)
        self.band_min = int(band_min)
        if self.n_min < 0 or self.per_domain_cap <= 0 or self.persistence <= 0:
            raise ValueError("n_min, per_domain_cap, and persistence are invalid")
        if self.band_min < 0:
            raise ValueError("band_min must be non-negative")

        if band_edges is None:
            self.band_edges = None
        else:
            self.band_edges = np.asarray(band_edges, dtype=float)
            expected_shape = (len(self.var_prior), 2)
            if self.band_edges.shape != expected_shape:
                raise ValueError(f"band_edges must have shape {expected_shape}")
            if np.any(~np.isfinite(self.band_edges)) or np.any(
                self.band_edges[:, 0] >= self.band_edges[:, 1]
            ):
                raise ValueError("each domain needs two increasing band edges")
        if self.band_min and self.band_edges is None:
            raise ValueError("band_edges are required when band_min > 0")

        self.reliability_mode = str(reliability_mode)
        if self.reliability_mode not in ("surrogate", "quantile_mcse"):
            raise ValueError("reliability_mode must be surrogate|quantile_mcse")
        self.ess_floor_fraction = float(ess_floor_fraction)
        if not 0.0 < self.ess_floor_fraction <= 1.0:
            raise ValueError("ess_floor_fraction must be in (0, 1]")
        self.min_rejuvenation_acceptance = (
            None
            if min_rejuvenation_acceptance is None
            else float(min_rejuvenation_acceptance)
        )
        self.min_distinct_ancestor_fraction = (
            None
            if min_distinct_ancestor_fraction is None
            else float(min_distinct_ancestor_fraction)
        )
        for name, value in (
            ("min_rejuvenation_acceptance", self.min_rejuvenation_acceptance),
            ("min_distinct_ancestor_fraction", self.min_distinct_ancestor_fraction),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.radius_mcse_z = float(radius_mcse_z)
        if self.radius_mcse_z < 0.0:
            raise ValueError("radius_mcse_z must be non-negative")
        self.radius_mcse_inflation = float(radius_mcse_inflation)
        if self.radius_mcse_inflation < 1.0:
            raise ValueError("radius_mcse_inflation must be at least 1")
        self.precision_statistic = str(precision_statistic)
        if self.precision_statistic not in (
            "point_centered_radius",
            "interval_halfwidth",
        ):
            raise ValueError(
                "precision_statistic must be point_centered_radius|interval_halfwidth"
            )

        self.interval_radius_scale_by_domain = self._validated_scale(
            interval_radius_scale_by_domain, "interval_radius_scale_by_domain"
        )
        self.interval_radius_ramp_max_by_domain = self._validated_scale(
            interval_radius_ramp_max_by_domain,
            "interval_radius_ramp_max_by_domain",
        )
        if (
            self.interval_radius_scale_by_domain is not None
            and self.interval_radius_ramp_max_by_domain is not None
        ):
            raise ValueError("constant and ramp interval-radius calibration are exclusive")
        self.interval_radius_ramp_abs_estimate_max = float(
            interval_radius_ramp_abs_estimate_max
        )
        if (
            not np.isfinite(self.interval_radius_ramp_abs_estimate_max)
            or self.interval_radius_ramp_abs_estimate_max <= 0.0
        ):
            raise ValueError("interval_radius_ramp_abs_estimate_max must be positive")
        self.bias_radius_scale_by_domain = self._validated_scale(
            bias_radius_scale_by_domain, "bias_radius_scale_by_domain"
        )
        if (
            self.interval_radius_scale_by_domain is not None
            or self.interval_radius_ramp_max_by_domain is not None
        ) and self.precision_statistic != "point_centered_radius":
            raise ValueError("interval-radius calibration requires point_centered_radius")

        self._statuses: list | None = None
        self._streaks: list | None = None
        self._terminal_reasons: list | None = None
        self._last_diag: dict | None = None

    def _validated_scale(self, values, name: str):
        if values is None:
            return None
        scale = np.asarray(values, dtype=float)
        if scale.shape != self.var_prior.shape:
            raise ValueError(f"{name} and var_prior must align by domain")
        if np.any(~np.isfinite(scale)) or np.any(scale < 1.0):
            raise ValueError(f"{name} must contain finite values at least 1")
        return scale

    @classmethod
    def from_inputs(cls, inputs, **kwargs):
        var_prior = np.diag(np.asarray(inputs.Corr_l, dtype=float))
        task_codes = tuple(str(code) for code in inputs.task_codes)
        if (
            "r_l" not in kwargs
            and "contraction_fraction" not in kwargs
            and task_codes == PRECISION_TASK_CODES
        ):
            kwargs["r_l"] = np.asarray(
                PRECISION_CONTRACTION_BY_DOMAIN, dtype=float
            ) * np.sqrt(var_prior)
        if "band_edges" not in kwargs and int(
            kwargs.get("band_min", PRECISION_BAND_MIN)
        ) > 0:
            bank_signals, _, _ = inputs.as_engine_arrays()
            edges = []
            for k, signals in enumerate(bank_signals):
                values = np.asarray(signals, dtype=float)
                if len(values) < 3 or np.any(~np.isfinite(values)):
                    raise ValueError(f"domain {k} cannot define signal terciles")
                q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
                if not float(q1) < float(q2):
                    raise ValueError(f"domain {k} has degenerate signal terciles")
                edges.append([float(q1), float(q2)])
            kwargs["band_edges"] = edges
        return cls(var_prior, **kwargs)

    @property
    def domain_statuses(self) -> list | None:
        return list(self._statuses) if self._statuses is not None else None

    @property
    def active_domains(self) -> list | None:
        if self._statuses is None:
            return None
        return [k for k, status in enumerate(self._statuses) if status == ACTIVE]

    @property
    def terminal_reasons(self) -> list | None:
        return (
            list(self._terminal_reasons)
            if self._terminal_reasons is not None
            else None
        )

    @property
    def last_diagnostics(self) -> dict | None:
        return dict(self._last_diag) if self._last_diag is not None else None

    @property
    def floor_only_blocked_domains(self) -> list:
        if self._last_diag is None or self._statuses is None:
            return []
        return [
            k
            for k, status in enumerate(self._statuses)
            if status == ACTIVE
            and self._last_diag["precision_persistent"][k]
            and self._last_diag["evidence_floor_met"][k]
            and not self._last_diag["content_floor_met"][k]
        ]

    def band_index(self, k: int, signal: float) -> int:
        if self.band_edges is None:
            return 0
        return int(
            np.searchsorted(self.band_edges[int(k)], float(signal), side="right")
        )

    def reset(self, K: int) -> None:
        if K != len(self.var_prior):
            raise ValueError(f"K={K} != configured domains={len(self.var_prior)}")
        self._statuses = [ACTIVE] * K
        self._streaks = [0] * K
        self._terminal_reasons = [None] * K
        self._last_diag = None

    def _content_met(self, telemetry, K: int) -> tuple[list, list]:
        if self.band_min == 0:
            return [True] * K, [[0, 0, 0] for _ in range(K)]
        administered = telemetry.get("band_administered")
        if administered is None:
            raise ValueError("PrecisionPolicy requires band_administered telemetry")
        deficits = []
        content_met = []
        for k in range(K):
            row = [int(value) for value in administered[k]]
            deficit_row = [max(0, self.band_min - value) for value in row]
            deficits.append(deficit_row)
            content_met.append(all(value == 0 for value in deficit_row))
        return content_met, deficits

    def _apply_bank_termination(
        self, telemetry, n_per_task, content_deficits, *, only_active=True
    ) -> None:
        remaining = telemetry.get("remaining_bank_counts")
        band_remaining = telemetry.get("band_remaining")
        if remaining is None:
            return
        for k, status in enumerate(self._statuses):
            if status.startswith("UNDETERMINABLE_"):
                continue
            if only_active and status != ACTIVE:
                continue
            impossible = int(n_per_task[k]) + int(remaining[k]) < self.n_min
            exhausted = int(remaining[k]) <= 0
            if self.band_min and band_remaining is not None:
                impossible = impossible or any(
                    int(supply) < int(deficit)
                    for supply, deficit in zip(
                        band_remaining[k], content_deficits[k]
                    )
                )
            if exhausted or impossible:
                self._statuses[k] = UNDETERMINABLE_BANK
                self._terminal_reasons[k] = UNDETERMINABLE_BANK

    def observe_bank_feasibility(self, telemetry, n_per_task, K) -> StopDecision:
        """Apply trial-0 bank checks without advancing precision streaks."""

        if self._statuses is None:
            self.reset(K)
        _, deficits = self._content_met(telemetry, K)
        self._apply_bank_termination(telemetry, n_per_task, deficits)
        return self.current_decision()

    def _reliability(self, state, telemetry, statistic_values, statistic_mcses):
        w = np.asarray(state["w"], dtype=float)
        ess_value = float(1.0 / float((w * w).sum()))
        ess_pass = ess_value >= self.ess_floor_fraction * len(w)
        event = telemetry.get("last_rejuvenation") or state.get("_last_rejuvenation")
        acceptance = None if event is None else float(event["acceptance_rate"])
        ancestry = (
            None if event is None else float(event["distinct_ancestor_fraction"])
        )
        acceptance_pass = (
            event is None
            or self.min_rejuvenation_acceptance is None
            or acceptance >= self.min_rejuvenation_acceptance
        )
        ancestry_pass = (
            event is None
            or self.min_distinct_ancestor_fraction is None
            or ancestry >= self.min_distinct_ancestor_fraction
        )
        surrogate_diagnostics_pass = bool(acceptance_pass and ancestry_pass)
        base_pass = bool(
            ess_pass
            and (
                self.reliability_mode != "surrogate" or surrogate_diagnostics_pass
            )
        )
        mcse_multiplier = self.radius_mcse_z * self.radius_mcse_inflation
        passes = []
        guarded_values = []
        for statistic_value, mcse_value, tolerance in zip(
            statistic_values, statistic_mcses, self.r_l
        ):
            guarded = (
                float(statistic_value)
                if self.reliability_mode == "surrogate"
                else float(statistic_value + mcse_multiplier * mcse_value)
            )
            guarded_values.append(guarded)
            mcse_pass = self.reliability_mode == "surrogate" or (
                np.isfinite(guarded) and guarded <= tolerance
            )
            passes.append(bool(base_pass and mcse_pass))
        return {
            "ess": ess_value,
            "ess_pass": bool(ess_pass),
            "rejuvenation_acceptance": acceptance,
            "acceptance_pass": bool(acceptance_pass),
            "distinct_ancestor_fraction": ancestry,
            "ancestry_pass": bool(ancestry_pass),
            "surrogate_diagnostics_pass": surrogate_diagnostics_pass,
            "precision_statistic_mcse_z": self.radius_mcse_z,
            "precision_statistic_mcse_inflation": self.radius_mcse_inflation,
            "precision_statistic_mcse_multiplier": mcse_multiplier,
            "guard_pass": passes,
            "guarded_precision_statistic": guarded_values,
        }

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        if self._statuses is None:
            self.reset(K)
        w = np.asarray(state["w"], dtype=float)
        w_sum = float(w.sum())
        if (
            not np.isfinite(w_sum)
            or w_sum <= 0.0
            or np.any(~np.isfinite(w))
            or np.any(w < 0.0)
        ):
            raise ValueError("PrecisionPolicy received invalid particle weights")
        w = w / w_sum

        intervals = []
        raw_intervals = []
        bias_intervals = []
        raw_bias_intervals = []
        halfwidths = []
        posterior_means = []
        bias_posterior_means = []
        point_centered_radii = []
        raw_point_centered_radii = []
        point_centered_radius_mcses = []
        raw_point_centered_radius_mcses = []
        statistic_values = []
        statistic_mcses = []
        ess_value = float(1.0 / float((w * w).sum()))

        for k in range(K):
            values = np.asarray(state["l"][:, k], dtype=float)
            low = _weighted_quantile(values, w, self.q_low)
            high = _weighted_quantile(values, w, self.q_high)
            raw_intervals.append([low, high])
            mean, radius, _, _ = _point_centered_radius(values, w, low, high)
            posterior_means.append(mean)
            raw_point_centered_radii.append(radius)
            raw_radius_mcse = (
                _point_centered_radius_mcse(
                    values, w, self.q_low, self.q_high, ess_value
                )
                if self.reliability_mode == "quantile_mcse"
                else float("nan")
            )
            if self.interval_radius_scale_by_domain is not None:
                scale = float(self.interval_radius_scale_by_domain[k])
            elif self.interval_radius_ramp_max_by_domain is not None:
                ramp_weight = min(
                    abs(mean) / self.interval_radius_ramp_abs_estimate_max, 1.0
                )
                scale = 1.0 + (
                    float(self.interval_radius_ramp_max_by_domain[k]) - 1.0
                ) * ramp_weight
            else:
                scale = 1.0
            calibrated_radius = scale * radius
            radius_mcse = scale * raw_radius_mcse
            interval = (
                [low, high]
                if self.interval_radius_scale_by_domain is None
                and self.interval_radius_ramp_max_by_domain is None
                else [mean - calibrated_radius, mean + calibrated_radius]
            )
            intervals.append(interval)
            halfwidths.append((interval[1] - interval[0]) / 2.0)
            point_centered_radii.append(calibrated_radius)
            point_centered_radius_mcses.append(radius_mcse)
            raw_point_centered_radius_mcses.append(raw_radius_mcse)
            if self.precision_statistic == "point_centered_radius":
                statistic_values.append(calibrated_radius)
                statistic_mcses.append(radius_mcse)
            else:
                statistic_values.append((high - low) / 2.0)
                statistic_mcses.append(
                    _halfwidth_mcse(
                        values, w, self.q_low, self.q_high, ess_value
                    )
                    if self.reliability_mode == "quantile_mcse"
                    else float("nan")
                )

            bias_values = np.asarray(state["t"][:, k], dtype=float)
            bias_low = _weighted_quantile(bias_values, w, self.q_low)
            bias_high = _weighted_quantile(bias_values, w, self.q_high)
            raw_bias_intervals.append([bias_low, bias_high])
            bias_mean = float((w * bias_values).sum())
            bias_posterior_means.append(bias_mean)
            if self.bias_radius_scale_by_domain is None:
                bias_intervals.append([bias_low, bias_high])
            else:
                bias_radius = max(bias_mean - bias_low, bias_high - bias_mean)
                bias_radius *= float(self.bias_radius_scale_by_domain[k])
                bias_intervals.append(
                    [bias_mean - bias_radius, bias_mean + bias_radius]
                )

        reliability = self._reliability(
            state, telemetry, statistic_values, statistic_mcses
        )
        content_met, content_deficits = self._content_met(telemetry, K)
        statistic_met = [
            bool(value <= tolerance)
            for value, tolerance in zip(statistic_values, self.r_l)
        ]
        precision_now = [
            bool(statistic_met[k] and reliability["guard_pass"][k])
            for k in range(K)
        ]
        evidence_met = [int(n_per_task[k]) >= self.n_min for k in range(K)]

        for k in range(K):
            if self._statuses[k].startswith("UNDETERMINABLE_"):
                continue
            self._streaks[k] = self._streaks[k] + 1 if precision_now[k] else 0
            persistent = self._streaks[k] >= self.persistence
            complete = persistent and evidence_met[k] and content_met[k]
            self._statuses[k] = ESTIMATE_COMPLETE if complete else ACTIVE
            # Evaluate the 60th response before terminalizing an incomplete domain.
            if (
                self._statuses[k] == ACTIVE
                and int(n_per_task[k]) >= self.per_domain_cap
            ):
                self._statuses[k] = UNDETERMINABLE_CAP
                self._terminal_reasons[k] = UNDETERMINABLE_CAP

        # BANK terminalization is deliberately ordered after the post-update cap.
        self._apply_bank_termination(
            telemetry, n_per_task, content_deficits, only_active=True
        )
        precision_persistent = [
            self._streaks[k] >= self.persistence for k in range(K)
        ]
        diagnostics = {
            "confidence": self.confidence,
            "skill_intervals": intervals,
            "bias_intervals": bias_intervals,
            "skill_posterior_mean": [float(x) for x in posterior_means],
            "skill_interval_halfwidth": [float(x) for x in halfwidths],
            "skill_point_centered_radius": [
                float(x) for x in point_centered_radii
            ],
            "skill_point_centered_radius_mcse": [
                float(x) for x in point_centered_radius_mcses
            ],
            "skill_tolerance": self.r_l.tolist(),
            "contraction_fraction": (
                float(self.contraction_fraction_by_domain[0])
                if np.allclose(
                    self.contraction_fraction_by_domain,
                    self.contraction_fraction_by_domain[0],
                    rtol=0.0,
                    atol=0.0,
                )
                else None
            ),
            "contraction_fraction_by_domain": self.contraction_fraction_by_domain.tolist(),
            "prior_halfwidth_multiplier": self.prior_halfwidth_multiplier,
            "band_min": self.band_min,
            "band_edges": (
                None if self.band_edges is None else self.band_edges.tolist()
            ),
            "precision_statistic": self.precision_statistic,
            "precision_statistic_value": [float(x) for x in statistic_values],
            "precision_statistic_mcse": [float(x) for x in statistic_mcses],
            "precision_statistic_met": statistic_met,
            "precision_now": precision_now,
            "precision_persistent": precision_persistent,
            "streak_counts": list(self._streaks),
            "evidence_floor_met": evidence_met,
            "content_floor_met": content_met,
            "content_deficits": content_deficits,
            "statuses": list(self._statuses),
            "terminal_reasons": list(self._terminal_reasons),
            "n_per_task": [int(x) for x in n_per_task],
            "reliability_mode": self.reliability_mode,
            **reliability,
        }
        if (
            self.interval_radius_scale_by_domain is not None
            or self.interval_radius_ramp_max_by_domain is not None
        ):
            diagnostics.update(
                {
                    "raw_skill_intervals": raw_intervals,
                    "raw_skill_point_centered_radius": [
                        float(x) for x in raw_point_centered_radii
                    ],
                    "raw_skill_point_centered_radius_mcse": [
                        float(x) for x in raw_point_centered_radius_mcses
                    ],
                }
            )
            if self.interval_radius_scale_by_domain is not None:
                diagnostics["interval_radius_scale_by_domain"] = (
                    self.interval_radius_scale_by_domain.tolist()
                )
            else:
                diagnostics.update(
                    {
                        "interval_radius_ramp_max_by_domain": (
                            self.interval_radius_ramp_max_by_domain.tolist()
                        ),
                        "interval_radius_ramp_abs_estimate_max": (
                            self.interval_radius_ramp_abs_estimate_max
                        ),
                        "realized_interval_radius_scale_by_domain": [
                            float(value) / float(raw) if float(raw) > 0.0 else 1.0
                            for value, raw in zip(
                                point_centered_radii, raw_point_centered_radii
                            )
                        ],
                    }
                )
        if self.bias_radius_scale_by_domain is not None:
            diagnostics.update(
                {
                    "bias_radius_scale_by_domain": self.bias_radius_scale_by_domain.tolist(),
                    "raw_bias_intervals": raw_bias_intervals,
                    "bias_posterior_mean": bias_posterior_means,
                }
            )
        self._last_diag = diagnostics
        return self.current_decision()

    def current_decision(self) -> StopDecision:
        if self._statuses is None:
            return StopDecision(stop=False, stop_reason="continue")
        stop = all(status != ACTIVE for status in self._statuses)
        return StopDecision(
            stop=stop,
            stop_reason="all_estimated_or_undeterminable" if stop else "continue",
            diagnostics=self._last_diag,
            domain_statuses=list(self._statuses),
            streak_counts=list(self._streaks),
            terminal_reasons=list(self._terminal_reasons),
        )

    def finalize_statuses(self) -> list:
        return list(self._statuses) if self._statuses is not None else []

    def finalize_determinations(self) -> list:
        return [
            DETERMINED if status == ESTIMATE_COMPLETE else status
            for status in self.finalize_statuses()
        ]
