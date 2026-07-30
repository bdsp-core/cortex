"""Robustness-floored response-artifact derivation.

The leakage-controlled fit pins the distractor lapse on the zero boundary:
real distractors are fully signal-following, so the fitted likelihood carries
no uniform-noise margin.  The N3b stress result
(research-policy/track-n/n3/N3_REPORT.md) shows the misspecified categorical
engine collapses catastrophically when distractor identity stops following
signal.  This module derives artifacts whose per-draw lapse is floored at a
declared robustness floor, leaving betas, weights, and the recorded fit
untouched, so the insurance is explicit, priced, and reproducible.  Derived
artifacts remain research-only and promotion-forbidden until qualified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def floor_draws(draws: list[dict], floor: float) -> list[dict]:
    if not 0 < floor < 1:
        raise ValueError("robustness floor must be inside (0, 1)")
    floored = []
    for draw in draws:
        row = dict(draw)
        row["distractor_lapse" if "distractor_lapse" in row else "distractorLapse"] = max(
            float(row.get("distractor_lapse", row.get("distractorLapse"))), floor,
        )
        floored.append(row)
    return floored


def _canonical_draws_sha256(draws: list[dict]) -> str:
    canonical = json.dumps(draws, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def derive_floored_crossfit(payload: dict, floor: float, source_name: str) -> dict:
    """Floor the deployable bootstrap draw set; preserve the recorded fit."""
    derived = copy.deepcopy(payload)
    derived["bootstrap"]["draws"] = floor_draws(payload["bootstrap"]["draws"], floor)
    lapses = [draw["distractor_lapse"] for draw in derived["bootstrap"]["draws"]]
    derived["bootstrap"]["distractorLapseInterval95"] = [
        float(np.quantile(lapses, 0.025)), float(np.quantile(lapses, 0.975)),
    ]
    derived["robustnessFloor"] = floor
    derived["flooredFrom"] = source_name
    derived["status"] = "research_only_not_promoted"
    derived["promotionForbidden"] = True
    return derived


def derive_floored_ensemble(artifact: dict, floor: float, source_name: str) -> dict:
    derived = copy.deepcopy(artifact)
    derived["draws"] = floor_draws(artifact["draws"], floor)
    derived["sha256"] = _canonical_draws_sha256(derived["draws"])
    derived["artifactId"] = f"{artifact['artifactId']}-floor{int(round(floor * 100)):03d}"
    derived["robustnessFloor"] = floor
    derived["qualification"] = "exploratory_unqualified"
    derived["promotionForbidden"] = True
    derived["provenance"] = dict(artifact["provenance"])
    derived["provenance"]["flooredFrom"] = (
        f"{source_name} (draws SHA-256 {artifact['sha256']})"
    )
    return derived


def assert_floored(artifact: dict) -> None:
    """Guard: a floored artifact must respect its declared robustness floor."""
    floor = artifact["robustnessFloor"]
    if not 0 < floor < 1:
        raise AssertionError("declared robustness floor must be inside (0, 1)")
    draws = artifact["draws"] if "draws" in artifact else artifact["bootstrap"]["draws"]
    for draw in draws:
        lapse = draw.get("distractorLapse", draw.get("distractor_lapse"))
        if lapse is None or lapse < floor:
            raise AssertionError(
                f"draw lapse {lapse} violates declared robustness floor {floor}"
            )
    if artifact.get("promotionForbidden") is not True:
        raise AssertionError("floored derivations must remain promotion-forbidden")


def _suffix(path: Path, floor: float) -> Path:
    return path.with_name(f"{path.stem}_floor{int(round(floor * 100)):03d}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossfit", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--floors", type=float, nargs="+", default=(0.15, 0.20))
    args = parser.parse_args()
    crossfit = json.loads(args.crossfit.read_text())
    ensemble = json.loads(args.ensemble.read_text())
    for floor in args.floors:
        for base, payload, derive in (
            (args.crossfit, crossfit, derive_floored_crossfit),
            (args.ensemble, ensemble, derive_floored_ensemble),
        ):
            derived = derive(payload, floor, base.name)
            assert_floored(derived)
            output = _suffix(base, floor)
            output.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n")
            print(output)


if __name__ == "__main__":
    main()
