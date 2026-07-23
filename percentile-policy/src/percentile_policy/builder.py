"""Build a de-identified provisional norm artifact from historical SDT fits."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any

from .model import DOMAINS, artifact_content_hash


@dataclass(frozen=True)
class QualityPolicy:
    min_trials: int = 20
    max_se_ell: float = 1.0
    sigma_lower_bound: float = 0.02
    sigma_upper_bound: float = 5.0
    boundary_tolerance: float = 1e-6


@dataclass(frozen=True)
class FitRecord:
    ell: float
    se_ell: float
    n_trials: int


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _parse_row(
    row: dict[str, str], policy: QualityPolicy
) -> tuple[FitRecord | None, str]:
    if not _truthy(row.get("converged", "")):
        return None, "not_converged"
    try:
        sigma = float(row["sigma"])
        se_sigma = float(row["se_sigma"])
        n_trials = int(float(row["n_trials"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "unparseable"
    if (
        not math.isfinite(sigma)
        or not math.isfinite(se_sigma)
        or sigma <= 0
        or se_sigma < 0
    ):
        return None, "nonfinite_or_invalid"
    if n_trials < policy.min_trials:
        return None, "too_few_trials"
    if (
        sigma <= policy.sigma_lower_bound + policy.boundary_tolerance
        or sigma >= policy.sigma_upper_bound - policy.boundary_tolerance
    ):
        return None, "optimizer_boundary"
    ell = -math.log(sigma)
    se_ell = se_sigma / sigma
    if not math.isfinite(se_ell) or se_ell > policy.max_se_ell:
        return None, "ell_se_above_limit"
    return FitRecord(ell=ell, se_ell=se_ell, n_trials=n_trials), "eligible"


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile from no values")
    position = probability * (len(sorted_values) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return sorted_values[left]
    frac = position - left
    return sorted_values[left] + frac * (
        sorted_values[right] - sorted_values[left]
    )


def _quantile_curve(values: list[float], points: int) -> list[float]:
    ordered = sorted(values)
    return [
        round(_quantile(ordered, i / (points - 1)), 8)
        for i in range(points)
    ]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def _eligible_under(
    rows: list[dict[str, str]], policy: QualityPolicy
) -> list[FitRecord]:
    return [
        record
        for row in rows
        if (record := _parse_row(row, policy)[0]) is not None
    ]


def _load_and_audit(
    source_csv: Path, policy: QualityPolicy
) -> tuple[dict[str, list[FitRecord]], dict[str, Any], dict[str, list[dict[str, str]]]]:
    raw_by_domain: dict[str, list[dict[str, str]]] = {d: [] for d in DOMAINS}
    with source_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            domain = row.get("domain")
            if domain in raw_by_domain:
                raw_by_domain[domain].append(row)
    selected: dict[str, list[FitRecord]] = {}
    audit_domains: dict[str, Any] = {}
    for domain in DOMAINS:
        reasons: Counter[str] = Counter()
        records: list[FitRecord] = []
        for row in raw_by_domain[domain]:
            record, reason = _parse_row(row, policy)
            reasons[reason] += 1
            if record is not None:
                records.append(record)
        if not records:
            raise ValueError(f"{domain}: quality policy selected no records")
        selected[domain] = records
        exclusions = {
            reason: count
            for reason, count in sorted(reasons.items())
            if reason != "eligible"
        }
        audit_domains[domain] = {
            "inputRows": len(raw_by_domain[domain]),
            "eligibleRows": len(records),
            "rowDisposition": dict(sorted(reasons.items())),
            "exclusionReasons": exclusions,
            "ellSummary": _summary([record.ell for record in records]),
            "seEllSummary": _summary([record.se_ell for record in records]),
            "nTrialsSummary": _summary(
                [float(record.n_trials) for record in records]
            ),
        }
    return selected, {"domains": audit_domains}, raw_by_domain


def _assert_manifest_pin(source_csv: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (
        manifest.get("k7_artifacts_sha256", {}).get("sdt_fits_k7.csv")
    )
    observed = _sha256_file(source_csv)
    if expected != observed:
        raise ValueError(
            "historical fit source does not match MANIFEST_k7.json: "
            f"expected {expected}, observed {observed}"
        )
    def portable(path: Path) -> str:
        parts = path.resolve().parts
        if "data" in parts:
            # The workspace itself may live below a `/data` mount.  The
            # repository's own data directory is the last `data` component.
            start = max(i for i, part in enumerate(parts) if part == "data")
            return "/".join(parts[start:])
        return path.name

    return {
        "sourcePath": portable(source_csv),
        "sourceSha256": observed,
        "manifestPath": portable(manifest_path),
        "manifestSha256": _sha256_file(manifest_path),
        "manifestGenerated": manifest.get("generated"),
        "producer": manifest.get("true_producer"),
    }


def build_artifact(
    *,
    source_csv: Path,
    manifest_path: Path,
    bootstrap_replicates: int = 500,
    quantile_points: int = 201,
    seed: int = 20260722,
    quality_policy: QualityPolicy = QualityPolicy(),
    created_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if quantile_points < 101 or quantile_points % 2 == 0:
        raise ValueError("quantile_points must be an odd integer >= 101")
    provenance = _assert_manifest_pin(source_csv, manifest_path)
    selected, audit, raw_by_domain = _load_and_audit(source_csv, quality_policy)
    domain_artifacts: dict[str, Any] = {}
    sensitivity_specs = {
        "stricter_se_ell_0_50": QualityPolicy(
            min_trials=quality_policy.min_trials,
            max_se_ell=0.5,
            sigma_lower_bound=quality_policy.sigma_lower_bound,
            sigma_upper_bound=quality_policy.sigma_upper_bound,
            boundary_tolerance=quality_policy.boundary_tolerance,
        ),
        "minimum_50_trials": QualityPolicy(
            min_trials=50,
            max_se_ell=quality_policy.max_se_ell,
            sigma_lower_bound=quality_policy.sigma_lower_bound,
            sigma_upper_bound=quality_policy.sigma_upper_bound,
            boundary_tolerance=quality_policy.boundary_tolerance,
        ),
    }
    for domain_index, domain in enumerate(DOMAINS):
        records = selected[domain]
        rng = random.Random(seed + 104729 * domain_index)
        curves: list[list[float]] = []
        for _ in range(bootstrap_replicates):
            latent = []
            for _ in range(len(records)):
                record = records[rng.randrange(len(records))]
                latent.append(rng.gauss(record.ell, record.se_ell))
            curves.append(_quantile_curve(latent, quantile_points))
        central = [
            round(
                _quantile(
                    sorted(curve[i] for curve in curves),
                    0.5,
                ),
                8,
            )
            for i in range(quantile_points)
        ]
        sensitivity: dict[str, list[float]] = {}
        sensitivity_n: dict[str, int] = {}
        for name, policy in sensitivity_specs.items():
            records_for_policy = _eligible_under(raw_by_domain[domain], policy)
            sensitivity_n[name] = len(records_for_policy)
            sensitivity[name] = _quantile_curve(
                [record.ell for record in records_for_policy], quantile_points
            )
        domain_artifacts[domain] = {
            "eligibleFitRows": len(records),
            "quantileProbabilities": {
                "start": 0.0,
                "stop": 1.0,
                "count": quantile_points,
                "spacing": "equal",
            },
            "centralQuantiles": central,
            "bootstrapQuantiles": curves,
            "sensitivityQuantiles": sensitivity,
            "sensitivityEligibleRows": sensitivity_n,
        }
    artifact: dict[str, Any] = {
        "schemaVersion": "cortex-percentile-norm/v1",
        "normId": "historical-calibration-k7-provisional-v1",
        "status": "candidate_public_preview",
        "createdUtc": created_utc or datetime.now(timezone.utc).isoformat(),
        "domainOrder": list(DOMAINS),
        "scale": "ell=-log(sigma), frozen K=7 certification instrument scale",
        "referenceLabel": "quality-screened historical mixed-rater calibration fit cohort",
        "fieldwork": "heterogeneous historical source studies; not a single norming window",
        "estimand": (
            "Relative position among eligible historical per-domain calibration "
            "fit records; not a clinician-population percentile."
        ),
        "method": {
            "name": "two_level_rater_and_fit_uncertainty_bootstrap",
            "fitUncertainty": (
                "ell_hat=-log(sigma_hat); SE_ell=SE_sigma/sigma_hat; each "
                "bootstrap fit record receives a Normal(ell_hat, SE_ell) draw"
            ),
            "samplingUncertainty": "equal-weight fit rows resampled with replacement within domain",
            "bootstrapReplicates": bootstrap_replicates,
            "quantilePoints": quantile_points,
            "seed": seed,
        },
        "qualityPolicy": asdict(quality_policy),
        "inputProvenance": provenance,
        "domains": domain_artifacts,
        "requiredWarnings": [
            "Provisional historical calibration-cohort percentile; not representative clinician norms.",
            "The 95% interval excludes unknown population-mismatch and source-composition bias.",
            "Historical fit rows are the sampling units and are not asserted to be unique governed people across sources.",
            "Percentile does not affect or replace the criterion-referenced CORTEX verdict.",
            "Percentile is not percent correct and is not readiness/pass probability.",
        ],
    }
    artifact["artifactSha256"] = artifact_content_hash(artifact)
    audit.update(
        {
            "schemaVersion": "cortex-percentile-audit/v1",
            "normId": artifact["normId"],
            "artifactSha256": artifact["artifactSha256"],
            "qualityPolicy": asdict(quality_policy),
            "inputProvenance": provenance,
            "sensitivityPolicies": {
                name: asdict(policy) for name, policy in sensitivity_specs.items()
            },
        }
    )
    return artifact, audit


def write_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )
    path.write_text(text + "\n", encoding="utf-8")
