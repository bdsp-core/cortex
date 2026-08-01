"""Tier-spanning floored response ensemble (option-1 experiment).

The deployed ensemble's nine draws express bootstrap uncertainty about the
pooled (novice-dominated) beta and never reach the expert tier, which the
powered campaign showed costs expert-world coverage (0.829). This builder
derives an ensemble whose draws span the tier range instead: quantile-spread
reader-clustered bootstrap draws from the novice and expert fits, floored at
the deployed robustness floor. Same leakage controls, same artifact plumbing;
research-only until qualified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .artifact_floor import floor_draws
from .artifact_rd import _read_csv, bootstrap_artifact
from .artifact_tiers import tier_records


def quantile_spread(draws: list, count: int) -> list:
    """Evenly spaced draws over the beta-sorted bootstrap, endpoints included
    (the same selection rule the qualification CLI applies to an ensemble)."""
    ordered = sorted(draws, key=lambda draw: draw.beta)
    indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int))
    return [ordered[int(index)] for index in indices]


def build(
    records,
    floor: float = 0.15,
    novice_draws: int = 5,
    expert_draws: int = 4,
    bootstrap_draws: int = 200,
    seed: int = 63_300_001,
    workers: int = 1,
) -> dict:
    selected = []
    tiers = {}
    for offset, (tier, count) in enumerate(
        (("expert", expert_draws), ("novice", novice_draws)), start=1,
    ):
        subset = tier_records(records, tier)
        # Seeds match the tier-stratification report (63_300_001 + tier offset).
        draws = bootstrap_artifact(subset, bootstrap_draws, seed + offset, "reader", workers)
        chosen = quantile_spread(draws, count)
        tiers[tier] = {
            "n_wrong_picks": len(subset),
            "bootstrap_draws": bootstrap_draws,
            "seed": seed + offset,
            "selected_betas": [draw.beta for draw in chosen],
        }
        selected.extend(chosen)
    rows = sorted(
        (
            {"beta": draw.beta, "distractor_lapse": draw.distractor_lapse, "weight": 1.0}
            for draw in selected
        ),
        key=lambda row: row["beta"],
    )
    floored = floor_draws(rows, floor)
    weight = 1.0 / len(floored)
    for row in floored:
        row["weight"] = weight
    return {
        "schemaVersion": 1,
        "status": "research_only_not_promoted",
        "model": "iiic_conditional_f1_v1_artifact_ensemble",
        "variant": "tier_spanning_floored",
        "robustnessFloor": floor,
        "nWrongPicks": len(records),
        "tiers": tiers,
        "bootstrap": {
            "clusterBy": "reader",
            "draws": floored,
            "betaInterval95": [floored[0]["beta"], floored[-1]["beta"]],
            "distractorLapseInterval95": [
                min(row["distractor_lapse"] for row in floored),
                max(row["distractor_lapse"] for row in floored),
            ],
        },
        "promotionForbidden": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--floor", type=float, default=0.15)
    parser.add_argument("--novice-draws", type=int, default=5)
    parser.add_argument("--expert-draws", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=63_300_001)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = _read_csv(args.input)
    payload = build(
        records, args.floor, args.novice_draws, args.expert_draws,
        args.bootstrap_draws, args.seed, args.workers,
    )
    payload["provenance"] = {
        "input": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "betas": [round(row["beta"], 4) for row in payload["bootstrap"]["draws"]],
        "lapses": [round(row["distractor_lapse"], 4) for row in payload["bootstrap"]["draws"]],
    }, indent=2))


if __name__ == "__main__":
    main()
