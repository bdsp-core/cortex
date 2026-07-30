"""Price the λ_d robustness floor on the real bank axes.

For each candidate floor the engine likelihood mixes a uniform share into the
distractor allocation, which costs Fisher information when the world is fully
signal-following (the fitted boundary).  This tool reports the retained
information at each floor relative to the unfloored categorical likelihood and
to the binary reduction, using the same finite-difference machinery as the
bank information audit.  Audit-only: no bank or artifact mutation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .bank_audit import BankSegment, _read_bank, candidate_information
from .qualification import IIIC_GROUP
from .reference import binary_p_yes, signal_z

DEFAULT_LAPSE_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)


def binary_information(
    segment: BankSegment,
    asked_k: int,
    t: np.ndarray,
    l: np.ndarray,
    finite_difference: float = 1e-4,
) -> dict[str, float]:
    """Finite-difference Fisher information of the binary hit/miss channel."""
    t = np.asarray(t, dtype=float).copy()
    l = np.asarray(l, dtype=float).copy()
    s_mean = np.asarray(segment.s_mean, dtype=float)
    s_sd = np.asarray(segment.s_sd, dtype=float)

    def p_yes() -> float:
        z = signal_z(
            l[0, asked_k], t[0, asked_k], s_mean[asked_k], s_sd[asked_k],
        )
        return float(binary_p_yes(np.asarray(z)))

    base = p_yes()
    output = {}
    for parameter, values in (("bias", t), ("skill", l)):
        original = values[0, asked_k]
        values[0, asked_k] = original + finite_difference
        plus = p_yes()
        values[0, asked_k] = original - finite_difference
        minus = p_yes()
        values[0, asked_k] = original
        derivative = (plus - minus) / (2 * finite_difference)
        # Two-outcome channel: (dp)^2/p + (d(1-p))^2/(1-p).
        output[parameter] = float(
            derivative**2 / max(base, 1e-12)
            + derivative**2 / max(1.0 - base, 1e-12)
        )
    return {
        "skill_information": output["skill"],
        "bias_information": output["bias"],
    }


def _price_segment(arguments) -> dict:
    segment, beta, lapse_grid, skill_profiles = arguments
    categorical = {lapse: {"skill": 0.0, "bias": 0.0} for lapse in lapse_grid}
    binary = {"skill": 0.0, "bias": 0.0}
    cells = 0
    for asked_k in IIIC_GROUP:
        for skill in skill_profiles:
            t = np.zeros((1, 7))
            l = np.zeros((1, 7))
            l[:, 1:] = skill
            cells += 1
            row = binary_information(segment, asked_k, t, l)
            binary["skill"] += row["skill_information"]
            binary["bias"] += row["bias_information"]
            for lapse in lapse_grid:
                metrics = candidate_information(
                    segment, asked_k, t, l, beta, lapse,
                )
                categorical[lapse]["skill"] += metrics["skill_information"]
                categorical[lapse]["bias"] += metrics["bias_information"]
    return {
        "cells": cells,
        "binary": binary,
        "categorical": {str(lapse): totals for lapse, totals in categorical.items()},
    }


def price_floors(
    segments: list[BankSegment],
    beta: float = 0.9912,
    lapse_grid: tuple[float, ...] = DEFAULT_LAPSE_GRID,
    skill_profiles: tuple[float, ...] = (-1.0, 0.0, 1.0),
    workers: int = 1,
) -> dict:
    arguments = [
        (segment, beta, lapse_grid, skill_profiles) for segment in segments
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_price_segment, arguments, chunksize=16))
    else:
        rows = [_price_segment(argument) for argument in arguments]
    cells = sum(row["cells"] for row in rows)
    binary = {
        key: sum(row["binary"][key] for row in rows) / cells
        for key in ("skill", "bias")
    }
    table = {}
    for lapse in lapse_grid:
        categorical = {
            key: sum(row["categorical"][str(lapse)][key] for row in rows) / cells
            for key in ("skill", "bias")
        }
        table[str(lapse)] = {
            "mean_skill_information": categorical["skill"],
            "mean_bias_information": categorical["bias"],
            "retention_vs_unfloored_skill": None,
            "retention_vs_unfloored_bias": None,
            "multiple_vs_binary_skill": categorical["skill"] / binary["skill"],
            "multiple_vs_binary_bias": categorical["bias"] / binary["bias"],
        }
    reference_row = table[str(lapse_grid[0])]
    for lapse in lapse_grid:
        row = table[str(lapse)]
        row["retention_vs_unfloored_skill"] = (
            row["mean_skill_information"] / reference_row["mean_skill_information"]
        )
        row["retention_vs_unfloored_bias"] = (
            row["mean_bias_information"] / reference_row["mean_bias_information"]
        )
    return {
        "schema_version": 1,
        "status": "audit_only_no_bank_mutation",
        "config": {
            "beta": beta,
            "lapse_grid": list(lapse_grid),
            "skill_profiles": list(skill_profiles),
            "segments": len(segments),
            "workers": workers,
        },
        "mean_binary_skill_information": binary["skill"],
        "mean_binary_bias_information": binary["bias"],
        "by_lapse": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.9912)
    parser.add_argument(
        "--lapse-grid", type=float, nargs="+", default=DEFAULT_LAPSE_GRID,
    )
    parser.add_argument("--max-segments", type=int, default=4000)
    parser.add_argument("--sample-seed", type=int, default=63_700_001)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.lapse_grid[0] != 0.0:
        raise SystemExit("the first grid value must be 0.0 (the retention reference)")
    segments = _read_bank(args.bank)
    if args.max_segments is not None and len(segments) > args.max_segments:
        rng = np.random.default_rng(args.sample_seed)
        indices = np.sort(rng.choice(
            len(segments), size=args.max_segments, replace=False,
        ))
        segments = [segments[int(index)] for index in indices]
    result = price_floors(
        segments,
        beta=args.beta,
        lapse_grid=tuple(args.lapse_grid),
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "segments": result["config"]["segments"],
        "by_lapse": {
            lapse: {
                "retention_skill": round(row["retention_vs_unfloored_skill"], 4),
                "retention_bias": round(row["retention_vs_unfloored_bias"], 4),
                "multiple_vs_binary_skill": round(row["multiple_vs_binary_skill"], 4),
                "multiple_vs_binary_bias": round(row["multiple_vs_binary_bias"], 4),
            }
            for lapse, row in result["by_lapse"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
