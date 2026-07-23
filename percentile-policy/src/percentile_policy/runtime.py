"""Build the compact, immutable runtime representation used by web/API code.

The governed JSON artifact remains the source of truth.  Runtime consumers do
not need its audit prose or repeated JSON number syntax, so the quantile
curves are packed as little-endian float32 values and accompanied by a small,
self-hashed metadata document.  Both runtimes verify the hashes before use.
"""

from __future__ import annotations

from array import array
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

from .model import DOMAINS, verify_artifact

RUNTIME_SCHEMA = "cortex-percentile-runtime/v1"
SCORE_SCHEMA = "cortex-percentile-score/v1"


def _canonical_without_hash(metadata: dict[str, Any]) -> bytes:
    body = dict(metadata)
    body.pop("metadataSha256", None)
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def metadata_content_hash(metadata: dict[str, Any]) -> str:
    return sha256(_canonical_without_hash(metadata)).hexdigest()


def verify_runtime_metadata(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if metadata.get("schemaVersion") != RUNTIME_SCHEMA:
        errors.append("unsupported runtime schemaVersion")
    if metadata.get("scoreSchemaVersion") != SCORE_SCHEMA:
        errors.append("unsupported scoreSchemaVersion")
    if metadata.get("domainOrder") != list(DOMAINS):
        errors.append("domainOrder is not the canonical K=7 order")
    expected = metadata.get("metadataSha256")
    if not isinstance(expected, str) or expected != metadata_content_hash(metadata):
        errors.append("metadataSha256 does not match canonical content")
    if metadata.get("floatEncoding") != "float32_le":
        errors.append("floatEncoding must be float32_le")
    layouts = metadata.get("domains")
    if not isinstance(layouts, dict):
        errors.append("domains must be an object")
        return errors
    expected_offset = 0
    for domain in DOMAINS:
        layout = layouts.get(domain)
        if not isinstance(layout, dict):
            errors.append(f"{domain}: missing runtime layout")
            continue
        for name in ("central", "bootstrap"):
            block = layout.get(name)
            if not isinstance(block, dict):
                errors.append(f"{domain}: missing {name} block")
                continue
            if block.get("offset") != expected_offset:
                errors.append(f"{domain}: {name} offset is not contiguous")
            count = block.get("count")
            if not isinstance(count, int) or count <= 0:
                errors.append(f"{domain}: invalid {name} count")
                continue
            expected_offset += count
        sens = layout.get("sensitivity")
        if not isinstance(sens, dict):
            errors.append(f"{domain}: sensitivity must be an object")
            continue
        for name in sorted(sens):
            block = sens[name]
            if block.get("offset") != expected_offset:
                errors.append(
                    f"{domain}: sensitivity {name} offset is not contiguous"
                )
            count = block.get("count")
            if not isinstance(count, int) or count <= 0:
                errors.append(f"{domain}: invalid sensitivity {name} count")
                continue
            expected_offset += count
    if metadata.get("floatCount") != expected_offset:
        errors.append("floatCount does not match domain layouts")
    return errors


def build_runtime(
    artifact: dict[str, Any],
    *,
    metadata_url: str,
    data_url: str,
) -> tuple[dict[str, Any], bytes]:
    errors = verify_artifact(artifact)
    if errors:
        raise ValueError("invalid norm artifact: " + "; ".join(errors))

    values = array("f")
    layouts: dict[str, Any] = {}

    def add(curve: list[float], **extra: Any) -> dict[str, Any]:
        offset = len(values)
        values.extend(float(x) for x in curve)
        return {"offset": offset, "count": len(curve), **extra}

    for domain in DOMAINS:
        source = artifact["domains"][domain]
        central = add(source["centralQuantiles"])
        bootstrap_rows = source["bootstrapQuantiles"]
        bootstrap = add(
            [x for curve in bootstrap_rows for x in curve],
            rows=len(bootstrap_rows),
            cols=len(bootstrap_rows[0]),
        )
        sensitivity = {
            name: add(curve)
            for name, curve in sorted(
                (source.get("sensitivityQuantiles") or {}).items()
            )
        }
        layouts[domain] = {
            "eligibleFitRows": int(source["eligibleFitRows"]),
            "central": central,
            "bootstrap": bootstrap,
            "sensitivity": sensitivity,
        }

    if sys.byteorder != "little":
        values.byteswap()
    data = values.tobytes()
    metadata: dict[str, Any] = {
        "schemaVersion": RUNTIME_SCHEMA,
        "scoreSchemaVersion": SCORE_SCHEMA,
        "normId": artifact["normId"],
        "normSha256": artifact["artifactSha256"],
        "referenceLabel": artifact["referenceLabel"],
        "policyStatus": "provisional_historical_calibration_cohort",
        "fieldwork": artifact.get("fieldwork"),
        "domainOrder": list(DOMAINS),
        "floatEncoding": "float32_le",
        "floatCount": len(values),
        "runtimeDataSha256": sha256(data).hexdigest(),
        "runtimeMetadataUrl": metadata_url,
        "runtimeDataUrl": data_url,
        "intervalLevel": 0.95,
        "intervalResolutionPercentagePoints": 0.01,
        "tailDisplayMinimum": 5,
        "tailDisplayMaximum": 95,
        "domains": layouts,
        "requiredWarnings": list(artifact.get("requiredWarnings") or []),
        "display": {
            "label": "Preliminary historical calibration-cohort percentile",
            "intervalLabel": "Approximate 95% range",
            "disclosure": (
                "This preview compares you with quality-screened historical "
                "calibration records—not representative clinician norms. It is "
                "not percent correct or readiness and does not affect the "
                "assessment determination."
            ),
        },
    }
    metadata["metadataSha256"] = metadata_content_hash(metadata)
    return metadata, data


def write_runtime(
    artifact: dict[str, Any],
    metadata_path: Path,
    data_path: Path,
    *,
    metadata_url: str,
    data_url: str,
) -> dict[str, Any]:
    metadata, data = build_runtime(
        artifact, metadata_url=metadata_url, data_url=data_url
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    data_path.write_bytes(data)
    return metadata
