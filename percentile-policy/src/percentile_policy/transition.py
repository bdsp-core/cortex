"""Objective gates for replacing provisional norms with CORTEX-user norms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .model import DOMAINS


@dataclass(frozen=True)
class TransitionPolicy:
    min_effective_n_per_domain: float = 500.0
    target_effective_n_per_domain: float = 1000.0
    max_interval_halfwidth_pp: float = 5.0
    max_anchor_drift_pp: float = 3.0
    min_stable_windows: int = 2
    min_required_strata_coverage: float = 0.80
    max_largest_site_share: float = 0.25


def evaluate_transition(
    metrics: dict[str, Any], policy: TransitionPolicy = TransitionPolicy()
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, observed: Any, required: str) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "required": required,
            }
        )

    domain_metrics = metrics.get("domains") or {}
    for domain in DOMAINS:
        item = domain_metrics.get(domain) or {}
        n_eff = float(item.get("effectiveFirstAttemptN", 0))
        halfwidth = float(item.get("maxCentralIntervalHalfwidthPp", float("inf")))
        drift = float(item.get("maxAnchorDriftPp", float("inf")))
        gate(
            f"{domain}.effective_n",
            n_eff >= policy.min_effective_n_per_domain,
            n_eff,
            f">= {policy.min_effective_n_per_domain:g}",
        )
        gate(
            f"{domain}.precision",
            halfwidth <= policy.max_interval_halfwidth_pp,
            halfwidth,
            f"<= {policy.max_interval_halfwidth_pp:g} percentage points",
        )
        gate(
            f"{domain}.stability",
            drift <= policy.max_anchor_drift_pp,
            drift,
            f"<= {policy.max_anchor_drift_pp:g} percentage points",
        )
    stable_windows = int(metrics.get("stableWindows", 0))
    gate(
        "stable_windows",
        stable_windows >= policy.min_stable_windows,
        stable_windows,
        f">= {policy.min_stable_windows}",
    )
    strata = float(metrics.get("requiredStrataCoverage", 0))
    gate(
        "strata_coverage",
        strata >= policy.min_required_strata_coverage,
        strata,
        f">= {policy.min_required_strata_coverage:.0%}",
    )
    site_share = float(metrics.get("largestSiteShare", 1))
    gate(
        "site_concentration",
        site_share <= policy.max_largest_site_share,
        site_share,
        f"<= {policy.max_largest_site_share:.0%}",
    )
    boolean_gates = {
        "first_attempt_only": "firstAttemptOnly",
        "pre_training_only": "preTrainingOnly",
        "instrument_compatibility": "instrumentCompatibilityValidated",
        "posterior_calibration": "posteriorCalibrationPassed",
        "dif_review": "difReviewPassed",
        "data_quality": "dataQualityPassed",
        "privacy_and_consent": "privacyAndConsentApproved",
        "independent_scientific_review": "independentScientificReviewApproved",
        "governance_cutover_approval": "governanceCutoverApproved",
    }
    for label, key in boolean_gates.items():
        observed = metrics.get(key) is True
        gate(label, observed, metrics.get(key), "true")
    passed = all(item["passed"] for item in gates)
    return {
        "schemaVersion": "cortex-percentile-transition-evaluation/v1",
        "eligibleForGovernedCutover": passed,
        "automaticCutoverAllowed": False,
        "decision": (
            "eligible_for_versioned_shadow_and_governed_cutover"
            if passed
            else "continue_provisional_policy"
        ),
        "policy": asdict(policy),
        "gates": gates,
    }
