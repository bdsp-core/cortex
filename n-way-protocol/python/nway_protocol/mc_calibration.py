"""Categorical-only fixed-history convergence and MCSE calibration study.

This module is deliberately outside the runtime path.  It first creates a
history with the frozen F1 selector/profile, then replays that exact history
with independent particle clouds.  Selection and observed responses therefore
cannot confound numerical comparisons.
"""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np

from .qualification import (
    IIIC_GROUP,
    QualificationConfig,
    _select,
    _truth_and_bank,
    recommended_workers,
)
from .reference import (
    Observation,
    ParticleCloud,
    ess,
    f1_probabilities,
    make_cloud,
    posterior_moments,
    resample_and_rejuvenate,
    update,
    weighted_quantile,
)


@dataclass(frozen=True)
class ReplayProfile:
    name: str
    particles: int
    mh_steps: int
    repeats: int


@dataclass(frozen=True)
class MCCalibrationConfig:
    histories: int = 12
    own_cap: int = 4
    bank_segments: int = 72
    selector_per_domain: int = 5
    history_particles: int = 1200
    history_mh_steps: int = 30
    candidate: ReplayProfile = ReplayProfile("candidate", 1200, 30, 12)
    reference: ReplayProfile = ReplayProfile("reference", 4800, 60, 4)
    ess_fraction: float = 0.5
    beta: float = 0.9912
    distractor_lapse: float = 0.0
    bootstrap_replicates: int = 48
    seed_base: int = 63_100_000


def _qualification_config(config: MCCalibrationConfig) -> QualificationConfig:
    return QualificationConfig(
        replicates=1,
        particles=config.history_particles,
        own_cap=config.own_cap,
        bank_segments=config.bank_segments,
        selector_per_domain=config.selector_per_domain,
        mh_steps=config.history_mh_steps,
        ess_fraction=config.ess_fraction,
        beta=config.beta,
        distractor_lapse=config.distractor_lapse,
        truth_t_mean=0.0,
        truth_l_mean=0.0,
        truth_sd=1.0,
        formal_sbc=True,
    )


def _make_fixed_history(seed: int, config: MCCalibrationConfig):
    """Generate one frozen categorical history with the baseline profile."""
    qualification = _qualification_config(config)
    truth_t, truth_l, s_mean, s_sd, _ = _truth_and_bank(seed, qualification)
    cloud = make_cloud(
        config.history_particles,
        np.eye(7),
        np.eye(7),
        np.random.default_rng(seed + 10_000),
    )
    response_rng = np.random.default_rng(seed + 20_000)
    mh_rng = np.random.default_rng(seed + 30_000)
    remaining = set(range(config.bank_segments))
    counts = np.zeros(7, dtype=int)
    observations: list[Observation] = []
    while np.any(counts[1:] < config.own_cap):
        asked_k, segment_index = _select(
            cloud, "categorical_f1", remaining, counts, s_mean, s_sd,
            qualification,
        )
        probabilities = f1_probabilities(
            truth_t, truth_l, s_mean[segment_index], s_sd[segment_index],
            asked_k, IIIC_GROUP, config.beta, config.distractor_lapse,
        )[0]
        pick = int(response_rng.choice(IIIC_GROUP, p=probabilities))
        observation = Observation(
            kind="categorical_f1",
            asked_k=asked_k,
            segment_index=segment_index,
            raw_pick=pick,
            group=IIIC_GROUP,
        )
        update(
            cloud, observation, s_mean[segment_index], s_sd[segment_index],
            config.beta, config.distractor_lapse,
        )
        if ess(cloud) < config.ess_fraction * cloud.n:
            resample_and_rejuvenate(
                cloud, s_mean, s_sd, config.beta, config.distractor_lapse,
                mh_rng, config.history_mh_steps, 2.38 / math.sqrt(14),
            )
        observations.append(observation)
        remaining.remove(segment_index)
        counts[asked_k] += 1
    return truth_t, truth_l, s_mean, s_sd, observations


def _bootstrap_mcse(
    cloud: ParticleCloud,
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    """Return particle-bootstrap MCSEs for mean and central-95% radius.

    The bootstrap is intentionally a plug-in estimate.  Repeat-cloud errors
    against the high-particle reference determine the required inflation.
    """
    if replicates < 2:
        return math.nan, math.nan
    means = np.empty(replicates)
    radii = np.empty(replicates)
    for index in range(replicates):
        sample = rng.choice(cloud.n, size=cloud.n, replace=True, p=cloud.w)
        drawn = values[sample]
        means[index] = float(drawn.mean())
        radii[index] = float((np.quantile(drawn, 0.975) - np.quantile(drawn, 0.025)) / 2)
    return float(means.std(ddof=1)), float(radii.std(ddof=1))


def _replay(
    seed: int,
    history_index: int,
    profile: ReplayProfile,
    repeat: int,
    config: MCCalibrationConfig,
    fixed_history,
) -> list[dict]:
    truth_t, truth_l, s_mean, s_sd, observations = fixed_history
    cloud_seed = seed + 100_000 + history_index * 1000 + repeat
    cloud = make_cloud(
        profile.particles,
        np.eye(7),
        np.eye(7),
        np.random.default_rng(cloud_seed),
    )
    mh_rng = np.random.default_rng(cloud_seed + 500_000)
    resamples = 0
    acceptances: list[float] = []
    ancestries: list[float] = []
    for observation in observations:
        update(
            cloud, observation, s_mean[observation.segment_index],
            s_sd[observation.segment_index], config.beta,
            config.distractor_lapse,
        )
        if ess(cloud) < config.ess_fraction * cloud.n:
            acceptance, ancestry = resample_and_rejuvenate(
                cloud, s_mean, s_sd, config.beta, config.distractor_lapse,
                mh_rng, profile.mh_steps, 2.38 / math.sqrt(14),
            )
            resamples += 1
            acceptances.append(acceptance)
            ancestries.append(ancestry)

    moments = posterior_moments(cloud)
    bootstrap_rng = np.random.default_rng(cloud_seed + 900_000)
    rows: list[dict] = []
    for parameter, values, means, truth in (
        ("skill", cloud.l, moments["l_mean"], truth_l),
        ("bias", cloud.t, moments["t_mean"], truth_t),
    ):
        for k in IIIC_GROUP:
            low = weighted_quantile(values[:, k], cloud.w, 0.025)
            high = weighted_quantile(values[:, k], cloud.w, 0.975)
            mean_mcse, radius_mcse = _bootstrap_mcse(
                cloud, values[:, k], bootstrap_rng,
                config.bootstrap_replicates,
            )
            rows.append({
                "history": history_index,
                "history_seed": seed,
                "panel": "calibration" if history_index % 2 == 0 else "validation",
                "profile": profile.name,
                "particles": profile.particles,
                "mh_steps": profile.mh_steps,
                "repeat": repeat,
                "parameter": parameter,
                "task_k": k,
                "truth": float(truth[k]),
                "mean": float(means[k]),
                "radius": float((high - low) / 2),
                "covered": bool(low <= truth[k] <= high),
                "mean_plugin_mcse": mean_mcse,
                "radius_plugin_mcse": radius_mcse,
                "resamples": resamples,
                "acceptance": float(np.mean(acceptances)) if acceptances else 0.0,
                "ancestry": float(np.mean(ancestries)) if ancestries else 1.0,
            })
    return rows


def _run_history(history_index: int, config: MCCalibrationConfig) -> list[dict]:
    seed = config.seed_base + history_index
    fixed_history = _make_fixed_history(seed, config)
    rows: list[dict] = []
    for profile in (config.candidate, config.reference):
        for repeat in range(profile.repeats):
            rows.extend(_replay(
                seed, history_index, profile, repeat, config, fixed_history,
            ))
    return rows


def summarize(rows: list[dict], config: MCCalibrationConfig) -> dict:
    reference_rows = [row for row in rows if row["profile"] == config.reference.name]
    reference: dict[tuple[int, str, int], tuple[float, float]] = {}
    for key in {(row["history"], row["parameter"], row["task_k"]) for row in reference_rows}:
        selected = [
            row for row in reference_rows
            if (row["history"], row["parameter"], row["task_k"]) == key
        ]
        reference[key] = (
            float(np.mean([row["mean"] for row in selected])),
            float(np.mean([row["radius"] for row in selected])),
        )

    candidate = [row for row in rows if row["profile"] == config.candidate.name]
    evaluated: list[dict] = []
    for row in candidate:
        reference_mean, reference_radius = reference[
            (row["history"], row["parameter"], row["task_k"])
        ]
        enriched = dict(row)
        enriched["reference_mean"] = reference_mean
        enriched["reference_radius"] = reference_radius
        enriched["mean_error"] = abs(row["mean"] - reference_mean)
        enriched["radius_error"] = abs(row["radius"] - reference_radius)
        enriched["mean_error_ratio"] = enriched["mean_error"] / max(
            row["mean_plugin_mcse"], 1e-12,
        )
        enriched["radius_error_ratio"] = enriched["radius_error"] / max(
            row["radius_plugin_mcse"], 1e-12,
        )
        evaluated.append(enriched)

    result = {
        "schema_version": 1,
        "status": "research_only_not_promoted",
        "config": asdict(config),
        "n_candidate_cells": len(evaluated),
        "parameters": {},
    }
    for parameter in ("skill", "bias"):
        parameter_rows = [row for row in evaluated if row["parameter"] == parameter]
        train = [row for row in parameter_rows if row["panel"] == "calibration"]
        validation = [row for row in parameter_rows if row["panel"] == "validation"]
        mean_inflation = float(np.quantile(
            [row["mean_error_ratio"] for row in train], 0.95,
        ))
        radius_inflation = float(np.quantile(
            [row["radius_error_ratio"] for row in train], 0.95,
        ))
        result["parameters"][parameter] = {
            "selected_mean_inflation_q95": mean_inflation,
            "selected_radius_inflation_q95": radius_inflation,
            "validation_mean_upper_coverage": float(np.mean([
                row["mean_error"] <= 1.645 * mean_inflation * row["mean_plugin_mcse"]
                for row in validation
            ])) if validation else math.nan,
            "validation_radius_upper_coverage": float(np.mean([
                row["radius_error"] <= 1.645 * radius_inflation * row["radius_plugin_mcse"]
                for row in validation
            ])) if validation else math.nan,
            "mean_absolute_radius_error": float(np.mean([
                row["radius_error"] for row in parameter_rows
            ])),
            "mean_candidate_over_reference_radius": float(np.mean([
                row["radius"] / max(row["reference_radius"], 1e-12)
                for row in parameter_rows
            ])),
            "candidate_coverage": float(np.mean([
                row["covered"] for row in parameter_rows
            ])),
            "candidate_mean_resamples": float(np.mean([
                row["resamples"] for row in parameter_rows
            ])),
            "candidate_mean_acceptance": float(np.mean([
                row["acceptance"] for row in parameter_rows
            ])),
            "candidate_mean_ancestry": float(np.mean([
                row["ancestry"] for row in parameter_rows
            ])),
        }
    result["promotion_note"] = (
        "Inflation values are eligible for a locked qualification only after "
        "the calibration and validation panels are enlarged and frozen."
    )
    return result


def run(config: MCCalibrationConfig, workers: int | None = None) -> tuple[list[dict], dict]:
    worker_count = workers or recommended_workers(
        config.histories,
        max(config.history_particles, config.candidate.particles, config.reference.particles),
    )
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        nested = list(executor.map(
            _run_history,
            range(config.histories),
            [config] * config.histories,
        ))
    rows = [row for history_rows in nested for row in history_rows]
    return rows, summarize(rows, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--histories", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = MCCalibrationConfig()
    if args.mode == "smoke":
        config = replace(
            config,
            histories=2,
            own_cap=1,
            bank_segments=18,
            selector_per_domain=2,
            history_particles=96,
            history_mh_steps=2,
            candidate=ReplayProfile("candidate", 96, 2, 2),
            reference=ReplayProfile("reference", 192, 4, 2),
            bootstrap_replicates=8,
        )
    if args.histories is not None:
        config = replace(config, histories=args.histories)
    rows, summary = run(config, args.workers)
    payload = {"summary": summary, "rows": rows}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
