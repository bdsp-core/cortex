"""Fixed-history comparison of research-only SMC variants."""
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
import time
from typing import Literal

import numpy as np

from .mc_calibration import MCCalibrationConfig, _make_fixed_history
from .qualification import IIIC_GROUP, recommended_workers
from .reference import (
    ess,
    make_cloud,
    posterior_moments,
    resample_and_rejuvenate,
    update,
    weighted_quantile,
)
from .smc_rd import make_rqmc_cloud


@dataclass(frozen=True)
class SMCProfile:
    name: str
    resampling: Literal["multinomial", "stratified", "residual", "rqmc_sorted"]
    initialization: Literal["pseudorandom", "rqmc"] = "pseudorandom"


@dataclass(frozen=True)
class SMCExperimentConfig:
    histories: int = 48
    repeats: int = 4
    particles: int = 1200
    mh_steps: int = 30
    own_cap: int = 4
    bank_segments: int = 72
    selector_per_domain: int = 5
    ess_fraction: float = 0.5
    beta: float = 0.9912
    distractor_lapse: float = 0.0
    seed_base: int = 63_400_000
    profiles: tuple[SMCProfile, ...] = (
        SMCProfile("frozen_multinomial", "multinomial"),
        SMCProfile("stratified", "stratified"),
        SMCProfile("residual", "residual"),
        SMCProfile("rqmc_hybrid", "rqmc_sorted", "rqmc"),
    )


def _history_config(config: SMCExperimentConfig) -> MCCalibrationConfig:
    return MCCalibrationConfig(
        histories=1,
        own_cap=config.own_cap,
        bank_segments=config.bank_segments,
        selector_per_domain=config.selector_per_domain,
        history_particles=config.particles,
        history_mh_steps=config.mh_steps,
        ess_fraction=config.ess_fraction,
        beta=config.beta,
        distractor_lapse=config.distractor_lapse,
        seed_base=config.seed_base,
    )


def _run_profile(
    history_index: int,
    profile: SMCProfile,
    repeat: int,
    config: SMCExperimentConfig,
    fixed_history,
) -> dict:
    truth_t, truth_l, s_mean, s_sd, observations = fixed_history
    seed = config.seed_base + 100_000 + history_index * 1000 + repeat
    if profile.initialization == "rqmc":
        cloud = make_rqmc_cloud(config.particles, np.eye(7), np.eye(7), seed)
    else:
        cloud = make_cloud(
            config.particles, np.eye(7), np.eye(7), np.random.default_rng(seed),
        )
    mh_rng = np.random.default_rng(seed + 500_000)
    resamples = 0
    acceptance = []
    ancestry = []
    started = time.perf_counter()
    for observation in observations:
        update(
            cloud, observation, s_mean[observation.segment_index],
            s_sd[observation.segment_index], config.beta,
            config.distractor_lapse,
        )
        if ess(cloud) < config.ess_fraction * cloud.n:
            accepted, ancestors = resample_and_rejuvenate(
                cloud, s_mean, s_sd, config.beta, config.distractor_lapse,
                mh_rng, config.mh_steps, 2.38 / math.sqrt(14),
                resampling_scheme=profile.resampling,
            )
            resamples += 1
            acceptance.append(accepted)
            ancestry.append(ancestors)
    wall_seconds = time.perf_counter() - started
    moments = posterior_moments(cloud)
    result = {
        "history": history_index,
        "profile": profile.name,
        "repeat": repeat,
        "resampling": profile.resampling,
        "initialization": profile.initialization,
        "wall_seconds": wall_seconds,
        "resamples": resamples,
        "acceptance": float(np.mean(acceptance)) if acceptance else 0.0,
        "ancestry": float(np.mean(ancestry)) if ancestry else 1.0,
    }
    for parameter, values, means, truth in (
        ("skill", cloud.l, moments["l_mean"], truth_l),
        ("bias", cloud.t, moments["t_mean"], truth_t),
    ):
        lows = np.asarray([
            weighted_quantile(values[:, k], cloud.w, 0.025) for k in IIIC_GROUP
        ])
        highs = np.asarray([
            weighted_quantile(values[:, k], cloud.w, 0.975) for k in IIIC_GROUP
        ])
        truth_values = truth[list(IIIC_GROUP)]
        result[f"{parameter}_width"] = float(np.mean(highs - lows))
        result[f"{parameter}_coverage"] = float(np.mean(
            (lows <= truth_values) & (truth_values <= highs)
        ))
        result[f"{parameter}_rmse"] = float(np.sqrt(np.mean(np.square(
            means[list(IIIC_GROUP)] - truth_values
        ))))
    return result


def _run_history(history_index: int, config: SMCExperimentConfig) -> list[dict]:
    fixed_history = _make_fixed_history(
        config.seed_base + history_index,
        _history_config(config),
    )
    return [
        _run_profile(history_index, profile, repeat, config, fixed_history)
        for profile in config.profiles
        for repeat in range(config.repeats)
    ]


def summarize(rows: list[dict], config: SMCExperimentConfig) -> dict:
    baseline_name = config.profiles[0].name
    result = {
        "schema_version": 1,
        "status": "research_only_not_promoted",
        "config": asdict(config),
        "profiles": {},
        "interpretation_guard": (
            "Numerical variants receive no credit for narrower intervals unless "
            "coverage and high-fidelity posterior agreement also pass."
        ),
    }
    baseline = [row for row in rows if row["profile"] == baseline_name]
    for profile in config.profiles:
        selected = [row for row in rows if row["profile"] == profile.name]
        metrics = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "skill_width", "bias_width", "skill_coverage", "bias_coverage",
                "skill_rmse", "bias_rmse", "wall_seconds", "resamples",
                "acceptance", "ancestry",
            )
        }
        metrics["skill_width_ratio_vs_frozen"] = metrics["skill_width"] / float(np.mean([
            row["skill_width"] for row in baseline
        ]))
        metrics["bias_width_ratio_vs_frozen"] = metrics["bias_width"] / float(np.mean([
            row["bias_width"] for row in baseline
        ]))
        result["profiles"][profile.name] = metrics
    return result


def run(config: SMCExperimentConfig, workers: int | None = None) -> tuple[list[dict], dict]:
    worker_count = workers or recommended_workers(config.histories, config.particles)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        nested = list(executor.map(
            _run_history,
            range(config.histories),
            [config] * config.histories,
        ))
    rows = [row for history in nested for row in history]
    return rows, summarize(rows, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--histories", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = SMCExperimentConfig()
    if args.mode == "smoke":
        config = replace(
            config,
            histories=4,
            repeats=2,
            particles=96,
            mh_steps=2,
            own_cap=1,
            bank_segments=18,
            selector_per_domain=2,
        )
    if args.histories is not None:
        config = replace(config, histories=args.histories)
    rows, summary = run(config, args.workers)
    payload = {"summary": summary, "rows": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
