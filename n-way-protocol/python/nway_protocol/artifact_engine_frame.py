"""Engine-frame hierarchical conditional-distractor artifact (construction A).

Every prior response artifact was fit in item frame (raw ``s_mean`` axes),
while the engine applies beta to the particle's skill-scaled evidence
``z = e^l (s + t) / sqrt(1 + (e^l s_sd)^2)``; the 2026-07-30 refit
(`reports/engine_frame_refit.json`) showed the frames disagree severalfold
at real readers' inferred sensitivity. This fitter derives the conditional
IN the engine frame, label-free:

1.  Per-reader engine model (per-domain bias t, scalar sensitivity l) by MAP
    on binary asked-class margins under the engine's standard-normal priors —
    the beta-free block, so the conditional never trains on its own output.
2.  A pre-registered stable core: readers with enough reads to identify the
    reader model and an interior sensitivity fit (label-free criteria only).
3.  Per-reader engine-frame beta profile fits (shared lapse), observed-
    information standard errors, and a DerSimonian-Laird random-effects fit
    on log beta giving the reader-population distribution.
4.  Nine ensemble draws at population quantiles (heterogeneity plus mean
    uncertainty), shared fitted lapse, floored at the deployed robustness
    floor. Same payload schema and plumbing as the deployed chain.

A sensitivity panel across nested subsets is emitted alongside the artifact
so the subset rule cannot be shopped after the fact. Research-only until
qualified; promotion remains forbidden by the payload itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtri

from .artifact_floor import floor_draws
from .artifact_rd import DistractorRecord, _design_matrix, _log_probabilities
from .engine_frame_refit import ReadRecord, binary_log_likelihood, stage_reads
from .stage_real_artifact import _sha256

N_DOMAINS = 6
BETA_BOUNDS = (0.01, 100.0)
LAPSE_BOUNDS = (0.0, 0.40)


def reader_arrays(rows: list[ReadRecord]) -> dict:
    return {
        "domain": np.asarray([row.asked_k - 1 for row in rows], dtype=int),
        "s_mean": np.asarray([row.s_mean[row.asked_k] for row in rows]),
        "s_sd": np.asarray([row.s_sd[row.asked_k] for row in rows]),
        "correct": np.asarray([row.pick_k == row.asked_k for row in rows]),
    }


def fit_reader_model(rows: list[ReadRecord]) -> dict:
    """MAP per-domain bias + scalar sensitivity on binary margins."""
    arrays = reader_arrays(rows)

    def objective(params: np.ndarray) -> float:
        l, t = float(params[0]), params[1:]
        sensitivity = np.exp(l)
        z = sensitivity * (arrays["s_mean"] + t[arrays["domain"]])
        z /= np.sqrt(1.0 + np.square(sensitivity * arrays["s_sd"]))
        penalty = 0.5 * (l * l + float(np.dot(t, t)))
        return -float(binary_log_likelihood(z, arrays["correct"]).sum()) + penalty

    fitted = minimize(
        objective,
        x0=np.zeros(1 + N_DOMAINS),
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * (1 + N_DOMAINS),
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"reader model fit failed: {fitted.message}")
    return {"l": float(fitted.x[0]), "t": [float(value) for value in fitted.x[1:]]}


def engine_frame_records(
    rows: list[ReadRecord], model: dict,
) -> list[DistractorRecord]:
    sensitivity = np.exp(model["l"])
    t = np.asarray([0.0] + model["t"])
    records = []
    for row in rows:
        if row.pick_k == row.asked_k:
            continue
        z = sensitivity * (np.asarray(row.s_mean) + t)
        z /= np.sqrt(1.0 + np.square(sensitivity * np.asarray(row.s_sd)))
        z[0] = 0.0
        records.append(DistractorRecord(
            reader_id=row.reader_id,
            source_id=row.source_id,
            asked_k=row.asked_k,
            raw_pick=row.pick_k,
            group=tuple(range(1, 7)),
            z=tuple(z),
        ))
    return records


def fit_conditional(records: list[DistractorRecord]) -> dict:
    design, positions = _design_matrix(records)

    def objective(values: np.ndarray) -> float:
        return -float(_log_probabilities(
            design, positions, float(values[0]), float(values[1]),
        ).sum())

    fitted = minimize(
        objective,
        x0=np.asarray([1.0, 0.02]),
        method="L-BFGS-B",
        bounds=(BETA_BOUNDS, LAPSE_BOUNDS),
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"pooled conditional fit failed: {fitted.message}")
    return {"beta": float(fitted.x[0]), "distractor_lapse": float(fitted.x[1])}


def profile_beta(records: list[DistractorRecord], lapse: float) -> dict:
    """Per-reader beta with the pooled lapse; observed-information SE."""
    design, positions = _design_matrix(records)

    def negative_log_likelihood(beta: float) -> float:
        return -float(_log_probabilities(design, positions, beta, lapse).sum())

    fitted = minimize_scalar(
        negative_log_likelihood, bounds=BETA_BOUNDS, method="bounded",
        options={"xatol": 1e-6},
    )
    beta = float(fitted.x)
    step = max(1e-4, 1e-3 * beta)
    curvature = (
        negative_log_likelihood(beta + step)
        - 2.0 * negative_log_likelihood(beta)
        + negative_log_likelihood(beta - step)
    ) / (step * step)
    standard_error = float(1.0 / np.sqrt(curvature)) if curvature > 0 else float("inf")
    return {"beta": beta, "standard_error": standard_error, "n_wrong_picks": len(records)}


def random_effects_log_beta(fits: list[dict]) -> dict:
    """DerSimonian-Laird random effects on log beta."""
    usable = [
        fit for fit in fits
        if np.isfinite(fit["standard_error"])
        and BETA_BOUNDS[0] * 1.01 < fit["beta"] < BETA_BOUNDS[1] * 0.99
    ]
    y = np.asarray([np.log(fit["beta"]) for fit in usable])
    variance = np.asarray([
        (fit["standard_error"] / fit["beta"]) ** 2 for fit in usable
    ])
    weight = 1.0 / variance
    pooled = float(np.sum(weight * y) / np.sum(weight))
    q = float(np.sum(weight * np.square(y - pooled)))
    k = len(usable)
    denominator = float(np.sum(weight) - np.sum(np.square(weight)) / np.sum(weight))
    tau_squared = max(0.0, (q - (k - 1)) / denominator) if denominator > 0 else 0.0
    total_weight = 1.0 / (variance + tau_squared)
    mu = float(np.sum(total_weight * y) / np.sum(total_weight))
    mu_variance = float(1.0 / np.sum(total_weight))
    return {
        "n_readers_used": k,
        "mu_log_beta": mu,
        "tau_log_beta": float(np.sqrt(tau_squared)),
        "se_mu_log_beta": float(np.sqrt(mu_variance)),
        "cochran_q": q,
    }


def population_draws(effects: dict, lapse: float, count: int) -> list[dict]:
    spread = float(np.sqrt(
        effects["tau_log_beta"] ** 2 + effects["se_mu_log_beta"] ** 2
    ))
    probabilities = (np.arange(count) + 0.5) / count
    return [
        {
            "beta": float(np.exp(effects["mu_log_beta"] + spread * ndtri(p))),
            "distractor_lapse": lapse,
            "weight": 1.0 / count,
        }
        for p in probabilities
    ]


def analyze_subset(
    readers: list[str],
    by_reader: dict[str, list[ReadRecord]],
    models: dict[str, dict],
    min_wrong: int,
) -> dict:
    all_records: list[DistractorRecord] = []
    per_reader_records: dict[str, list[DistractorRecord]] = {}
    for reader_id in readers:
        records = engine_frame_records(by_reader[reader_id], models[reader_id])
        if len(records) >= min_wrong:
            per_reader_records[reader_id] = records
        all_records.extend(records)
    pooled = fit_conditional(all_records)
    fits = [
        {"reader_id": reader_id, **profile_beta(records, pooled["distractor_lapse"])}
        for reader_id, records in sorted(per_reader_records.items())
    ]
    effects = random_effects_log_beta(fits)
    return {
        "n_readers": len(readers),
        "n_readers_with_beta_fit": len(fits),
        "n_wrong_picks": len(all_records),
        "pooled": pooled,
        "random_effects": effects,
        "reader_beta_fits": fits,
        "sensitivity_summary": {
            "mean_l": float(np.mean([models[r]["l"] for r in readers])),
            "min_l": float(min(models[r]["l"] for r in readers)),
            "max_l": float(max(models[r]["l"] for r in readers)),
        },
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
    parser.add_argument("--floor", type=float, default=0.15)
    parser.add_argument("--draws", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-output", type=Path, required=True)
    args = parser.parse_args()

    reads = stage_reads(args.signals, args.case_map, args.expert, args.novice)
    by_reader: dict[str, list[ReadRecord]] = {}
    for read in reads:
        by_reader.setdefault(read.reader_id, []).append(read)

    def wrong_count(rows: list[ReadRecord]) -> int:
        return sum(1 for row in rows if row.pick_k != row.asked_k)

    fit_pool = {
        reader_id: rows for reader_id, rows in by_reader.items()
        if wrong_count(rows) >= args.min_wrong
    }
    models = {reader_id: fit_reader_model(rows) for reader_id, rows in fit_pool.items()}

    subsets = {
        "full": sorted(fit_pool),
        "reads100": sorted(
            reader_id for reader_id, rows in fit_pool.items()
            if len(rows) >= args.min_reads
        ),
        "stable_core": sorted(
            reader_id for reader_id, rows in fit_pool.items()
            if len(rows) >= args.min_reads
            and abs(models[reader_id]["l"]) <= args.interior_l
        ),
    }
    panel = {
        name: analyze_subset(readers, by_reader, models, args.min_wrong)
        for name, readers in subsets.items()
    }

    adopted = panel["stable_core"]
    draws = population_draws(
        adopted["random_effects"], adopted["pooled"]["distractor_lapse"], args.draws,
    )
    floored = floor_draws(sorted(draws, key=lambda row: row["beta"]), args.floor)
    provenance = {
        "method": "engine_frame_hierarchical_v1",
        "adopted_subset": "stable_core",
        "subset_rule": {
            "min_reads": args.min_reads,
            "min_wrong_picks": args.min_wrong,
            "interior_abs_l_max": args.interior_l,
        },
        "reader_model": "per_domain_t_scalar_l_map_standard_normal_prior_binary_margins",
        "population_model": "dersimonian_laird_log_beta;draw_spread=sqrt(tau^2+se_mu^2)",
        "source_sha256": {
            str(path): _sha256(path)
            for path in (args.signals, args.case_map, args.expert, args.novice)
        },
    }
    payload = {
        "schemaVersion": 1,
        "status": "research_only_not_promoted",
        "model": "iiic_conditional_f1_v1_artifact_ensemble",
        "variant": "engine_frame_hierarchical",
        "robustnessFloor": args.floor,
        "nWrongPicks": adopted["n_wrong_picks"],
        "subsetReaders": adopted["n_readers"],
        "randomEffects": adopted["random_effects"],
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
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.panel_output.parent.mkdir(parents=True, exist_ok=True)
    args.panel_output.write_text(json.dumps(
        {"schema_version": 1, "status": "research_only_diagnostic", "panel": panel},
        indent=2, sort_keys=True,
    ) + "\n")
    print(json.dumps({
        "stable_core": {
            key: adopted[key] for key in ("n_readers", "n_wrong_picks", "pooled", "random_effects")
        },
        "floored_betas": [round(row["beta"], 4) for row in floored],
        "floored_lapses": [round(row["distractor_lapse"], 4) for row in floored],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
