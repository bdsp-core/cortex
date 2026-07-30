"""DR07 identifiability gate for the F1 conditional-distractor fit.

The originating requirement (item 11 of OPTIMAL_DESIGN_RD_RECOMMENDATIONS,
document absent from this repo) demanded a versioned refit with complete
joint-state coverage followed by a governed rank/conditioning check; the
earlier inclusive categorical model died with 20 of 32 joint states
unpopulated. The F1 operationalization pinned in docs/DR07_CRITERION.md
and executed here:

1.  **State coverage** — every one of the 30 off-diagonal (gold class →
    wrong pick) confusion states populated with at least ``--min-state``
    reads in the stable-core fit corpus.
2.  **Identifiability at the optimum** — observed information of
    (beta, lambda_d) at the pooled engine-frame fit. When lambda sits on
    its zero boundary (the persistent finding), identification is
    one-sided: the profile slope in lambda at zero must be positive and
    the beta curvature positive; otherwise the full 2x2 information
    matrix must be positive definite with condition number at most
    ``--condition-max``.
3.  **Split-half stability** — readers split deterministically by id
    hash; the pooled engine-frame beta refit on each half must agree
    within ``--beta-rel-tol`` relative difference.

The gate emits a versioned, hash-chained report. It does not modify any
artifact; stamping ``provenance.dr07`` is a separate deliberate step that
records this report's digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .artifact_engine_frame import (
    engine_frame_records,
    fit_conditional,
    fit_reader_model,
    profile_beta,
    random_effects_log_beta,
)
from .artifact_rd import _design_matrix, _log_probabilities
from .engine_frame_refit import ReadRecord, stage_reads
from .stage_real_artifact import _sha256


def state_coverage(records) -> dict:
    counts = np.zeros((7, 7), dtype=int)
    for record in records:
        counts[record.asked_k, record.raw_pick] += 1
    states = {
        f"{asked}->{pick}": int(counts[asked, pick])
        for asked in range(1, 7) for pick in range(1, 7) if asked != pick
    }
    return {"states": states, "min_count": min(states.values())}


def negative_log_likelihood(records, beta: float, lapse: float) -> float:
    design, positions = _design_matrix(records)
    return -float(_log_probabilities(design, positions, beta, lapse).sum())


def identifiability(records, fitted: dict, condition_max: float) -> dict:
    beta, lapse = fitted["beta"], fitted["distractor_lapse"]
    step_b = max(1e-4, 1e-3 * beta)
    step_l = 1e-4

    def nll(b: float, l: float) -> float:
        return negative_log_likelihood(records, b, max(0.0, l))

    beta_curvature = (
        nll(beta + step_b, lapse) - 2.0 * nll(beta, lapse) + nll(beta - step_b, lapse)
    ) / (step_b * step_b)
    if lapse <= 1e-9:
        lapse_slope = (nll(beta, step_l) - nll(beta, 0.0)) / step_l
        passed = bool(beta_curvature > 0 and lapse_slope > 0)
        return {
            "mode": "boundary_lambda_zero",
            "beta_curvature": float(beta_curvature),
            "lapse_profile_slope_at_zero": float(lapse_slope),
            "pass": passed,
        }
    lapse_curvature = (
        nll(beta, lapse + step_l) - 2.0 * nll(beta, lapse) + nll(beta, lapse - step_l)
    ) / (step_l * step_l)
    cross = (
        nll(beta + step_b, lapse + step_l) - nll(beta + step_b, lapse - step_l)
        - nll(beta - step_b, lapse + step_l) + nll(beta - step_b, lapse - step_l)
    ) / (4.0 * step_b * step_l)
    information = np.asarray([[beta_curvature, cross], [cross, lapse_curvature]])
    eigenvalues = np.linalg.eigvalsh(information)
    condition = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else float("inf")
    return {
        "mode": "interior",
        "information": information.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "condition_number": condition,
        "pass": bool(eigenvalues[0] > 0 and condition <= condition_max),
    }


def split_half(readers: dict, models: dict, mu_z_max: float) -> dict:
    """Population-parameter stability across a deterministic reader split.

    The v1 criterion compared the POOLED beta between halves at a 5%
    tolerance; with the measured between-reader heterogeneity (tau ~0.26)
    the expected half-to-half pooled difference is sqrt(2)*tau/sqrt(n/2)
    ~4.6%, so that check rejects the true model roughly half the time.
    The v2 criterion (docs/DR07_CRITERION.md, amendment A1) gates on the
    population mean mu of log beta agreeing within ``mu_z_max`` standard
    errors; the pooled difference stays reported as a diagnostic.
    """
    half_readers: dict[str, list[str]] = {"a": [], "b": []}
    for reader_id in sorted(readers):
        digest = hashlib.sha256(reader_id.encode()).hexdigest()
        half_readers["a" if int(digest, 16) % 2 == 0 else "b"].append(reader_id)

    halves = {}
    for name, ids in half_readers.items():
        records = {
            reader_id: engine_frame_records(readers[reader_id], models[reader_id])
            for reader_id in ids
        }
        flat = [record for rows in records.values() for record in rows]
        pooled = fit_conditional(flat)
        fits = [
            {
                "reader_id": reader_id,
                **profile_beta(rows, pooled["distractor_lapse"]),
            }
            for reader_id, rows in sorted(records.items())
        ]
        halves[name] = {
            "n_readers": len(ids),
            "n_wrong_picks": len(flat),
            "pooled": pooled,
            "random_effects": random_effects_log_beta(fits),
        }
    effects_a = halves["a"]["random_effects"]
    effects_b = halves["b"]["random_effects"]
    mu_z = abs(effects_a["mu_log_beta"] - effects_b["mu_log_beta"]) / float(np.sqrt(
        effects_a["se_mu_log_beta"] ** 2 + effects_b["se_mu_log_beta"] ** 2
    ))
    pooled_relative = abs(
        halves["a"]["pooled"]["beta"] - halves["b"]["pooled"]["beta"]
    ) / (0.5 * (halves["a"]["pooled"]["beta"] + halves["b"]["pooled"]["beta"]))
    return {
        **{f"half_{name}": half for name, half in halves.items()},
        "mu_z_statistic": float(mu_z),
        "pooled_beta_relative_difference_diagnostic": float(pooled_relative),
        "pass": bool(mu_z <= mu_z_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--novice", type=Path, required=True)
    parser.add_argument("--min-reads", type=int, default=100)
    parser.add_argument("--min-wrong", type=int, default=30)
    parser.add_argument("--interior-l", type=float, default=2.5)
    parser.add_argument("--min-state", type=int, default=5)
    parser.add_argument("--condition-max", type=float, default=1e4)
    parser.add_argument("--mu-z-max", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reads = stage_reads(args.signals, args.case_map, args.expert, args.novice)
    by_reader: dict[str, list[ReadRecord]] = {}
    for read in reads:
        by_reader.setdefault(read.reader_id, []).append(read)

    def wrong_count(rows) -> int:
        return sum(1 for row in rows if row.pick_k != row.asked_k)

    core = {
        reader_id: rows for reader_id, rows in by_reader.items()
        if len(rows) >= args.min_reads and wrong_count(rows) >= args.min_wrong
    }
    models = {reader_id: fit_reader_model(rows) for reader_id, rows in core.items()}
    core = {
        reader_id: rows for reader_id, rows in core.items()
        if abs(models[reader_id]["l"]) <= args.interior_l
    }
    records = [
        record for reader_id, rows in sorted(core.items())
        for record in engine_frame_records(rows, models[reader_id])
    ]

    coverage = state_coverage(records)
    coverage["pass"] = bool(coverage["min_count"] >= args.min_state)
    pooled = fit_conditional(records)
    identified = identifiability(records, pooled, args.condition_max)
    stability = split_half(core, models, args.mu_z_max)

    verdict = bool(coverage["pass"] and identified["pass"] and stability["pass"])
    payload = {
        "schema_version": 1,
        "gate": "dr07_f1_conditional_v1",
        "criterion_doc": "docs/DR07_CRITERION.md",
        "frame": "engine",
        "subset_rule": {
            "min_reads": args.min_reads,
            "min_wrong_picks": args.min_wrong,
            "interior_abs_l_max": args.interior_l,
        },
        "n_readers": len(core),
        "n_wrong_picks": len(records),
        "criterion_version": "v2_amendment_A1_population_mu",
        "thresholds": {
            "min_state_count": args.min_state,
            "condition_max": args.condition_max,
            "mu_z_max": args.mu_z_max,
        },
        "pooled_fit": pooled,
        "checks": {
            "state_coverage": coverage,
            "identifiability": identified,
            "split_half_stability": stability,
        },
        "verdict": "pass" if verdict else "fail",
        "provenance": {
            "source_sha256": {
                str(path): _sha256(path)
                for path in (args.signals, args.case_map, args.expert, args.novice)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": payload["verdict"],
        "state_min_count": coverage["min_count"],
        "identifiability": {k: v for k, v in identified.items() if k != "information"},
        "split_half_mu_z": stability["mu_z_statistic"],
        "split_half_pooled_diagnostic": stability[
            "pooled_beta_relative_difference_diagnostic"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
