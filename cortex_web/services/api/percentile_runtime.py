"""Verified provisional-percentile runtime shared by test and trainer APIs.

The source-of-truth artifact is built by ``percentile-policy``.  Production
ships its compact float32 representation in the SPA's public/norms directory.
This loader verifies both metadata and binary hashes before exposing a scoring
surface; malformed/missing artifacts therefore disable reporting rather than
silently changing a participant's reference distribution.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterable

DOMAINS = ("spike", "sz", "lpd", "gpd", "lrda", "grda", "iic")
RUNTIME_SCHEMA = "cortex-percentile-runtime/v1"
SCORE_SCHEMA = "cortex-percentile-score/v1"


def _canonical_without_hash(value: dict[str, Any]) -> bytes:
    body = dict(value)
    body.pop("metadataSha256", None)
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


class PercentileRuntime:
    """Lazy, thread-safe verifier and scorer for one immutable norm version."""

    def __init__(self, metadata_path: str | Path):
        self.metadata_path = Path(metadata_path)
        self._lock = threading.Lock()
        self._metadata: dict[str, Any] | None = None
        self._values = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        self._load()
        return self._error

    @property
    def ready(self) -> bool:
        self._load()
        return self._metadata is not None and self._values is not None

    @property
    def metadata(self) -> dict[str, Any] | None:
        self._load()
        return self._metadata

    def _load(self) -> None:
        if self._metadata is not None or self._error is not None:
            return
        with self._lock:
            if self._metadata is not None or self._error is not None:
                return
            try:
                import numpy as np

                metadata = json.loads(
                    self.metadata_path.read_text(encoding="utf-8")
                )
                self._validate_metadata(metadata)
                data_path = self.metadata_path.with_name(
                    Path(str(metadata["runtimeDataUrl"])).name
                )
                raw = data_path.read_bytes()
                if sha256(raw).hexdigest() != metadata["runtimeDataSha256"]:
                    raise ValueError("runtime binary SHA-256 mismatch")
                if len(raw) != int(metadata["floatCount"]) * 4:
                    raise ValueError("runtime binary length mismatch")
                values = np.frombuffer(raw, dtype="<f4")
                if not np.isfinite(values).all():
                    raise ValueError("runtime binary contains non-finite values")
                self._validate_curves(metadata, values, np)
                self._metadata, self._values = metadata, values
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _validate_metadata(metadata: dict[str, Any]) -> None:
        if metadata.get("schemaVersion") != RUNTIME_SCHEMA:
            raise ValueError("unsupported percentile runtime schema")
        if metadata.get("scoreSchemaVersion") != SCORE_SCHEMA:
            raise ValueError("unsupported percentile score schema")
        if metadata.get("domainOrder") != list(DOMAINS):
            raise ValueError("percentile domain order mismatch")
        if metadata.get("floatEncoding") != "float32_le":
            raise ValueError("unsupported percentile float encoding")
        actual = sha256(_canonical_without_hash(metadata)).hexdigest()
        if metadata.get("metadataSha256") != actual:
            raise ValueError("runtime metadata SHA-256 mismatch")
        for key in (
            "normId", "normSha256", "runtimeDataSha256",
            "runtimeMetadataUrl", "runtimeDataUrl",
        ):
            if not isinstance(metadata.get(key), str) or not metadata[key]:
                raise ValueError(f"missing runtime metadata field {key}")
        if metadata.get("intervalLevel") != 0.95:
            raise ValueError("public preview requires a 95% interval")

    @staticmethod
    def _validate_curves(metadata, values, np) -> None:
        expected = 0
        for domain in DOMAINS:
            layout = metadata["domains"][domain]
            blocks = [
                ("central", layout["central"]),
                ("bootstrap", layout["bootstrap"]),
                *[
                    (f"sensitivity.{name}", block)
                    for name, block in sorted(layout["sensitivity"].items())
                ],
            ]
            for name, block in blocks:
                offset, count = int(block["offset"]), int(block["count"])
                if offset != expected or count <= 0:
                    raise ValueError(f"{domain} {name} has invalid layout")
                expected += count
                view = values[offset:offset + count]
                if name == "bootstrap":
                    rows, cols = int(block["rows"]), int(block["cols"])
                    if rows * cols != count or rows < 1 or cols < 3:
                        raise ValueError(f"{domain} bootstrap shape mismatch")
                    view = view.reshape(rows, cols)
                    if (np.diff(view, axis=1) < 0).any():
                        raise ValueError(f"{domain} bootstrap curves not monotone")
                elif (np.diff(view) < 0).any():
                    raise ValueError(f"{domain} {name} curve not monotone")
        if expected != int(metadata["floatCount"]) or expected != len(values):
            raise ValueError("percentile float layout is not contiguous")

    def profile(self, *, display: bool) -> dict[str, Any]:
        if not self.ready or self._metadata is None:
            raise RuntimeError(self._error or "percentile runtime unavailable")
        m = self._metadata
        return {
            "scoreSchemaVersion": m["scoreSchemaVersion"],
            "normId": m["normId"],
            "normSha256": m["normSha256"],
            "metadataSha256": m["metadataSha256"],
            "runtimeDataSha256": m["runtimeDataSha256"],
            "runtimeMetadataUrl": m["runtimeMetadataUrl"],
            "runtimeDataUrl": m["runtimeDataUrl"],
            "referenceLabel": m["referenceLabel"],
            "policyStatus": m["policyStatus"],
            "intervalLevel": m["intervalLevel"],
            "display": bool(display),
            "displayCopy": m["display"],
            "requiredWarnings": m["requiredWarnings"],
        }

    def profile_matches(self, profile: dict[str, Any] | None) -> bool:
        if not isinstance(profile, dict) or not self.ready:
            return False
        expected = self.profile(display=bool(profile.get("display")))
        return profile == expected

    def _block(self, domain: str, name: str):
        assert self._metadata is not None and self._values is not None
        block = self._metadata["domains"][domain][name]
        start, count = int(block["offset"]), int(block["count"])
        return self._values[start:start + count], block

    def score_domain(
        self, domain: str, ell_samples: Iterable[float],
        weights: Iterable[float],
    ) -> dict[str, Any]:
        if domain not in DOMAINS or not self.ready:
            raise ValueError(f"percentile runtime unavailable for {domain}")
        import numpy as np

        ell = np.asarray(list(ell_samples), dtype=np.float64)
        w = np.asarray(list(weights), dtype=np.float64)
        if (ell.ndim != 1 or w.ndim != 1 or not len(ell)
                or len(ell) != len(w) or not np.isfinite(ell).all()
                or not np.isfinite(w).all() or (w < 0).any()
                or float(w.sum()) <= 0):
            raise ValueError("candidate particle cloud is invalid")
        w = w / w.sum()
        raw, block = self._block(domain, "bootstrap")
        curves = raw.reshape(int(block["rows"]), int(block["cols"]))
        probabilities = np.linspace(0.0, 1.0, curves.shape[1])
        ranks = np.empty((curves.shape[0], len(ell)), dtype=np.float32)
        for row, curve in enumerate(curves):
            ranks[row] = 100.0 * np.interp(
                ell, curve, probabilities, left=0.0, right=1.0
            )
        point = float((ranks @ w).mean())
        resolution = float(self._metadata[
            "intervalResolutionPercentagePoints"
        ])
        bins = int(round(100.0 / resolution)) + 1
        indices = np.floor(ranks / resolution + 0.5).astype(np.int32).ravel()
        combined = np.tile(w / curves.shape[0], curves.shape[0])
        histogram = np.bincount(indices, weights=combined, minlength=bins)
        cdf = np.cumsum(histogram)

        def quantile(p: float) -> float:
            return min(100.0, int(np.searchsorted(cdf, p, side="left"))
                       * resolution)

        sensitivity: dict[str, float] = {}
        for name, layout in sorted(
            self._metadata["domains"][domain]["sensitivity"].items()
        ):
            curve = self._values[
                int(layout["offset"]):int(layout["offset"]) + int(layout["count"])
            ]
            sensitivity[name] = float(
                np.dot(w, 100.0 * np.interp(
                    ell, curve, np.linspace(0.0, 1.0, len(curve)),
                    left=0.0, right=1.0,
                ))
            )
        alpha = 1.0 - float(self._metadata["intervalLevel"])
        values = list(sensitivity.values())
        return {
            "estimate": point,
            "lower": quantile(alpha / 2.0),
            "upper": quantile(1.0 - alpha / 2.0),
            "level": float(self._metadata["intervalLevel"]),
            "status": self._metadata["policyStatus"],
            "candidateMethod": "posterior_particles",
            "referenceReplicates": curves.shape[0],
            "sensitivityRange": [min(values), max(values)] if values else None,
        }

    def score_domains(
        self, samples_by_domain: dict[str, tuple[Iterable[float], Iterable[float]]]
    ) -> dict[str, Any]:
        if not self.ready or self._metadata is None:
            raise RuntimeError(self._error or "percentile runtime unavailable")
        return {
            "scoreSchemaVersion": self._metadata["scoreSchemaVersion"],
            "policyStatus": self._metadata["policyStatus"],
            "domains": {
                domain: self.score_domain(domain, samples, weights)
                for domain, (samples, weights) in samples_by_domain.items()
                if domain in DOMAINS
            },
        }


def validate_percentile_report(
    expected_profile: dict[str, Any] | None,
    report: Any,
) -> list[str]:
    """Validate the persisted client report against the immutable session stamp."""
    if expected_profile is None:
        return [] if report is None else ["unstamped session reported percentile data"]
    if not isinstance(report, dict):
        return ["stamped session is missing percentile report"]
    if report.get("profile") != expected_profile:
        return ["percentile profile does not match session stamp"]
    status = report.get("status")
    if status in {"unavailable_runtime_error", "unavailable_legacy_client"}:
        if report.get("domains") not in (None, {}):
            return ["unavailable percentile report must not contain domains"]
        return []
    if status != "available":
        return ["invalid percentile report status"]
    domains = report.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
        return ["available percentile report must contain all seven domains"]
    errors: list[str] = []
    for domain in DOMAINS:
        score = domains.get(domain)
        if not isinstance(score, dict):
            errors.append(f"{domain}: invalid percentile score")
            continue
        if score.get("candidateMethod") != "posterior_particles":
            errors.append(f"{domain}: invalid candidate method")
        if score.get("status") != expected_profile.get("policyStatus"):
            errors.append(f"{domain}: invalid percentile policy status")
        if score.get("level") != expected_profile.get("intervalLevel"):
            errors.append(f"{domain}: invalid interval level")
        if not isinstance(score.get("referenceReplicates"), int) \
                or score["referenceReplicates"] < 1:
            errors.append(f"{domain}: invalid reference replicate count")
        for key in ("estimate", "lower", "upper"):
            if not _finite_number(score.get(key)) \
                    or not 0 <= float(score[key]) <= 100:
                errors.append(f"{domain}: invalid {key}")
        if (_finite_number(score.get("lower"))
                and _finite_number(score.get("upper"))
                and float(score["lower"]) > float(score["upper"])):
            errors.append(f"{domain}: inverted percentile interval")
        sensitivity = score.get("sensitivityRange")
        if sensitivity is not None and (
            not isinstance(sensitivity, list) or len(sensitivity) != 2
            or any(not _finite_number(x) or not 0 <= float(x) <= 100
                   for x in sensitivity)
            or float(sensitivity[0]) > float(sensitivity[1])
        ):
            errors.append(f"{domain}: invalid sensitivity range")
    return errors
