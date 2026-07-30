"""Tier-stratified leakage-controlled distractor fits (expert vs novice).

The pooled crossfit artifact is dominated by novice reads (~96% of wrong
picks), while the DR07 preliminary found experts follow signal ~35% more
sharply.  This tool refits the conditional-distractor layer per reader tier
with the same leakage controls as the pooled artifact and reports how much of
each tier's bootstrap beta mass falls outside the deployed ensemble span.
Audit-only: it never replaces the deployed artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .artifact_rd import (
    DistractorRecord,
    _read_csv,
    bootstrap_artifact,
    crossfit_artifact,
    fit_artifact,
)

TIER_PREFIXES = {"expert": "expert-", "novice": "novice-"}


def tier_records(
    records: list[DistractorRecord], tier: str,
) -> list[DistractorRecord]:
    if tier == "pooled":
        return list(records)
    prefix = TIER_PREFIXES[tier]
    return [record for record in records if record.reader_id.startswith(prefix)]


def stratify(
    records: list[DistractorRecord],
    ensemble_betas: list[float],
    bootstrap_draws: int = 200,
    folds: int = 5,
    seed: int = 63_300_001,
    workers: int = 1,
) -> dict:
    span = (min(ensemble_betas), max(ensemble_betas))
    tiers = {}
    for offset, tier in enumerate(("pooled", "expert", "novice")):
        subset = tier_records(records, tier)
        readers = sorted({record.reader_id for record in subset})
        full = fit_artifact(subset)
        try:
            crossfit = crossfit_artifact(
                subset, folds, salt=f"nway-artifact-v1:{tier}", split_unit="reader",
            )
        except ValueError as error:
            crossfit = {"error": str(error)}
        draws = bootstrap_artifact(
            subset, bootstrap_draws, seed + offset, "reader", workers,
        )
        betas = np.asarray([draw.beta for draw in draws])
        lapses = np.asarray([draw.distractor_lapse for draw in draws])
        tiers[tier] = {
            "n_wrong_picks": len(subset),
            "n_readers": len(readers),
            "full_fit": {"beta": full.beta, "distractor_lapse": full.distractor_lapse},
            "reader_held_out_crossfit": crossfit,
            "bootstrap": {
                "draws": bootstrap_draws,
                "seed": seed + offset,
                "beta_interval95": [
                    float(np.quantile(betas, 0.025)), float(np.quantile(betas, 0.975)),
                ],
                "beta_median": float(np.median(betas)),
                "distractor_lapse_interval95": [
                    float(np.quantile(lapses, 0.025)),
                    float(np.quantile(lapses, 0.975)),
                ],
                "beta_mass_above_ensemble_span": float(np.mean(betas > span[1])),
                "beta_mass_below_ensemble_span": float(np.mean(betas < span[0])),
            },
        }
    return {
        "schema_version": 1,
        "status": "research_only_not_promoted",
        "model": "iiic_conditional_f1_v1_tier_stratified",
        "ensemble_beta_span": list(span),
        "tiers": tiers,
        "promotionForbidden": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=63_300_001)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = _read_csv(args.input)
    ensemble = json.loads(args.ensemble.read_text())
    result = stratify(
        records,
        [draw["beta"] for draw in ensemble["draws"]],
        args.bootstrap_draws,
        args.folds,
        args.seed,
        args.workers,
    )
    result["provenance"] = {
        "input": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "ensemble": args.ensemble.name,
        "ensemble_artifact_id": ensemble["artifactId"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        tier: {
            "n": row["n_wrong_picks"],
            "beta": round(row["full_fit"]["beta"], 4),
            "beta95": [round(v, 4) for v in row["bootstrap"]["beta_interval95"]],
            "mass_above_span": row["bootstrap"]["beta_mass_above_ensemble_span"],
        }
        for tier, row in result["tiers"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
