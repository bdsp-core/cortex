"""Runtime-independent percentile transforms.

The reference artifact stores an ensemble of bootstrap quantile curves.  A
candidate percentile is obtained by evaluating every candidate posterior draw
against every reference-CDF draw.  This propagates candidate measurement
uncertainty and the provisional reference sampling/fit uncertainty.
"""

from __future__ import annotations

from bisect import bisect_right
from hashlib import sha256
import json
import math
from statistics import NormalDist
from typing import Any

DOMAINS = ("spike", "sz", "lpd", "gpd", "lrda", "grda", "iic")


def _canonical_without_hash(artifact: dict[str, Any]) -> bytes:
    body = dict(artifact)
    body.pop("artifactSha256", None)
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def artifact_content_hash(artifact: dict[str, Any]) -> str:
    """Hash the canonical artifact body, excluding its self-hash field."""
    return sha256(_canonical_without_hash(artifact)).hexdigest()


def verify_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the artifact is usable."""
    errors: list[str] = []
    if artifact.get("schemaVersion") != "cortex-percentile-norm/v1":
        errors.append("unsupported schemaVersion")
    expected = artifact.get("artifactSha256")
    if not isinstance(expected, str) or expected != artifact_content_hash(artifact):
        errors.append("artifactSha256 does not match canonical content")
    if artifact.get("domainOrder") != list(DOMAINS):
        errors.append("domainOrder is not the canonical K=7 order")
    domains = artifact.get("domains")
    if not isinstance(domains, dict):
        errors.append("domains must be an object")
        return errors
    for domain in DOMAINS:
        item = domains.get(domain)
        if not isinstance(item, dict):
            errors.append(f"{domain}: missing domain artifact")
            continue
        curves = item.get("bootstrapQuantiles")
        if not isinstance(curves, list) or not curves:
            errors.append(f"{domain}: bootstrapQuantiles must be non-empty")
            continue
        width = None
        named_curves: list[tuple[str, Any]] = [
            (f"bootstrapQuantiles[{idx}]", curve)
            for idx, curve in enumerate(curves)
        ]
        named_curves.append(("centralQuantiles", item.get("centralQuantiles")))
        named_curves.extend(
            (f"sensitivityQuantiles.{name}", curve)
            for name, curve in (item.get("sensitivityQuantiles") or {}).items()
        )
        for name, curve in named_curves:
            if not isinstance(curve, list) or len(curve) < 3:
                errors.append(f"{domain}: invalid quantile curve {name}")
                break
            if width is None:
                width = len(curve)
            elif len(curve) != width:
                errors.append(f"{domain}: quantile curves have unequal lengths")
                break
            if any(
                not isinstance(x, (int, float)) or not math.isfinite(float(x))
                for x in curve
            ):
                errors.append(f"{domain}: non-finite quantile in {name}")
                break
            if any(curve[i] > curve[i + 1] for i in range(len(curve) - 1)):
                errors.append(f"{domain}: non-monotone quantile in {name}")
                break
    return errors


def cdf_from_quantile_curve(x: float, curve: list[float]) -> float:
    """Invert an equally spaced monotone quantile curve by linear interpolation."""
    if not math.isfinite(x):
        raise ValueError("candidate ell must be finite")
    if x < curve[0]:
        return 0.0
    if x >= curve[-1]:
        return 1.0
    right = bisect_right(curve, x)
    left = right - 1
    p_left = left / (len(curve) - 1)
    p_right = right / (len(curve) - 1)
    x_left, x_right = curve[left], curve[right]
    if x_right <= x_left:
        return (p_left + p_right) / 2.0
    frac = (x - x_left) / (x_right - x_left)
    return p_left + frac * (p_right - p_left)


def _normalize_samples(
    record: dict[str, Any], *, approximation_samples: int = 401
) -> tuple[list[float], list[float], str]:
    samples = record.get("ellSamples")
    if isinstance(samples, list):
        if not samples:
            raise ValueError("ellSamples cannot be empty")
        values = [float(x) for x in samples]
        weights_raw = record.get("weights")
        if weights_raw is None:
            weights = [1.0 / len(values)] * len(values)
        else:
            if not isinstance(weights_raw, list) or len(weights_raw) != len(values):
                raise ValueError("weights must match ellSamples")
            weights = [float(x) for x in weights_raw]
        method = "posterior_particles"
    elif "ellMean" in record and "ellSd" in record:
        mean, sd = float(record["ellMean"]), float(record["ellSd"])
        if not math.isfinite(mean) or not math.isfinite(sd) or sd < 0:
            raise ValueError("ellMean/ellSd must be finite and ellSd non-negative")
        if sd == 0:
            values, weights = [mean], [1.0]
        else:
            normal = NormalDist(mu=mean, sigma=sd)
            values = [
                normal.inv_cdf((i + 0.5) / approximation_samples)
                for i in range(approximation_samples)
            ]
            weights = [1.0 / approximation_samples] * approximation_samples
        method = "normal_approximation_from_mean_sd"
    else:
        raise ValueError("provide ellSamples or ellMean and ellSd")
    if any(not math.isfinite(x) for x in values):
        raise ValueError("candidate ell samples must be finite")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("candidate weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("candidate weights must have positive sum")
    return values, [w / total for w in weights], method


def _mean_rank(curve: list[float], samples: list[float], weights: list[float]) -> float:
    return 100.0 * sum(
        weight * cdf_from_quantile_curve(value, curve)
        for value, weight in zip(samples, weights, strict=True)
    )


def score_domain(
    domain_artifact: dict[str, Any],
    candidate: dict[str, Any],
    *,
    interval_level: float = 0.95,
) -> dict[str, Any]:
    """Score one domain using the full candidate × reference uncertainty grid."""
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must be in (0,1)")
    samples, weights, candidate_method = _normalize_samples(candidate)
    curves: list[list[float]] = domain_artifact["bootstrapQuantiles"]
    reference_weight = 1.0 / len(curves)
    # The rank is bounded in [0,100]. A fixed 0.01-percentage-point histogram
    # gives an interval error far below the artifact's 0.5-point CDF-knot
    # resolution and avoids sorting millions of candidate×reference pairs.
    rank_resolution = 0.01
    rank_bins = int(100.0 / rank_resolution) + 1
    rank_histogram = [0.0] * rank_bins
    point = 0.0
    for curve in curves:
        for value, weight in zip(samples, weights, strict=True):
            rank = 100.0 * cdf_from_quantile_curve(value, curve)
            combined_weight = reference_weight * weight
            point += combined_weight * rank
            bin_index = min(
                rank_bins - 1,
                max(0, int(math.floor(rank / rank_resolution + 0.5))),
            )
            rank_histogram[bin_index] += combined_weight

    def histogram_quantile(probability: float) -> float:
        cumulative = 0.0
        for index, weight in enumerate(rank_histogram):
            cumulative += weight
            if cumulative >= probability:
                return index * rank_resolution
        return 100.0

    alpha = 1.0 - interval_level
    lower = histogram_quantile(alpha / 2.0)
    upper = histogram_quantile(1.0 - alpha / 2.0)

    sensitivity: dict[str, float] = {}
    for name, curve in (domain_artifact.get("sensitivityQuantiles") or {}).items():
        sensitivity[name] = _mean_rank(curve, samples, weights)
    sensitivity_values = list(sensitivity.values())

    return {
        "estimate": point,
        "lower": lower,
        "upper": upper,
        "level": interval_level,
        "candidateMethod": candidate_method,
        "referenceReplicates": len(curves),
        "intervalResolutionPercentagePoints": rank_resolution,
        "sensitivityEstimates": sensitivity,
        "sensitivityRange": (
            [min(sensitivity_values), max(sensitivity_values)]
            if sensitivity_values
            else None
        ),
        "display": {
            "estimate": round(point),
            "lower": round(lower),
            "upper": round(upper),
        },
    }


def score_candidate(
    artifact: dict[str, Any],
    candidate: dict[str, Any],
    *,
    interval_level: float = 0.95,
) -> dict[str, Any]:
    errors = verify_artifact(artifact)
    if errors:
        raise ValueError("invalid norm artifact: " + "; ".join(errors))
    candidate_domains = candidate.get("domains")
    if not isinstance(candidate_domains, dict):
        raise ValueError("candidate must contain a domains object")
    results: dict[str, Any] = {}
    for domain in DOMAINS:
        if domain in candidate_domains:
            results[domain] = score_domain(
                artifact["domains"][domain],
                candidate_domains[domain],
                interval_level=interval_level,
            )
    if not results:
        raise ValueError("candidate has no canonical domains")
    return {
        "scoreSchemaVersion": "cortex-percentile-score/v1",
        "policyStatus": "provisional_historical_calibration_cohort",
        "norm": {
            "id": artifact["normId"],
            "sha256": artifact["artifactSha256"],
            "referenceLabel": artifact["referenceLabel"],
            "fieldwork": artifact.get("fieldwork"),
        },
        "intervalLabel": (
            f"Approximate {interval_level:.0%} uncertainty interval from candidate posterior, "
            "reference-fit uncertainty, and rater bootstrap; excludes unknown "
            "population-representativeness bias."
        ),
        "domains": results,
        "warnings": list(artifact.get("requiredWarnings") or []),
    }
