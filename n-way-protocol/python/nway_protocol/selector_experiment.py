"""Adaptive categorical selector challengers on matched planted-truth banks."""
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
from typing import Literal

import numpy as np

from .qualification import (
    IIIC_GROUP,
    QualificationConfig,
    _truth_and_bank,
    recommended_workers,
)
from .reference import (
    Observation,
    ess,
    expected_loss,
    f1_probabilities,
    f1_probabilities_artifact,
    make_cloud,
    posterior_moments,
    resample_and_rejuvenate,
    update,
    weighted_quantile,
)

SelectorName = Literal[
    "frozen_total_variance",
    "fisher_augmented_total_variance",
    "bias_weighted_total_variance",
    "mutual_information",
]


@dataclass(frozen=True)
class SelectorExperimentConfig:
    replicates: int = 192
    particles: int = 1200
    mh_steps: int = 30
    own_cap: int = 4
    bank_segments: int = 72
    selector_per_domain: int = 5
    fisher_per_domain: int = 3
    bias_weight: float = 1.5
    ess_fraction: float = 0.5
    beta: float = 0.9912
    distractor_lapse: float = 0.0
    artifact_draws: tuple[tuple[float, float, float], ...] = ()
    truth_artifact_draws: tuple[tuple[float, float, float], ...] = ()
    truth_beta: float | None = None
    truth_distractor_lapse: float | None = None
    seed_base: int = 63_500_000
    selectors: tuple[SelectorName, ...] = (
        "frozen_total_variance",
        "fisher_augmented_total_variance",
        "bias_weighted_total_variance",
        "mutual_information",
    )


def _qualification_config(config: SelectorExperimentConfig) -> QualificationConfig:
    return QualificationConfig(
        replicates=1,
        particles=config.particles,
        own_cap=config.own_cap,
        bank_segments=config.bank_segments,
        selector_per_domain=config.selector_per_domain,
        mh_steps=config.mh_steps,
        ess_fraction=config.ess_fraction,
        beta=config.beta,
        distractor_lapse=config.distractor_lapse,
        truth_t_mean=0.0,
        truth_l_mean=0.0,
        truth_sd=1.0,
        formal_sbc=True,
    )


def _candidate_probabilities(cloud, asked_k, s_mean, s_sd, config):
    return f1_probabilities_artifact(
        cloud.t, cloud.l, s_mean, s_sd, asked_k, IIIC_GROUP,
        config.beta, config.distractor_lapse, config.artifact_draws,
    )


def _weighted_loss(cloud, asked_k, s_mean, s_sd, config) -> float:
    probabilities = _candidate_probabilities(cloud, asked_k, s_mean, s_sd, config)
    moments = posterior_moments(cloud)
    baseline = float(
        config.bias_weight * np.square(moments["t_sd"]).sum()
        + np.square(moments["l_sd"]).sum()
    )
    joint = cloud.w[:, None] * probabilities
    masses = joint.sum(axis=0)
    between = 0.0
    for r, mass in enumerate(masses):
        if mass <= 1e-15:
            continue
        conditional_t = joint[:, r] @ cloud.t / mass
        conditional_l = joint[:, r] @ cloud.l / mass
        between += float(mass * (
            config.bias_weight * np.square(conditional_t - moments["t_mean"]).sum()
            + np.square(conditional_l - moments["l_mean"]).sum()
        ))
    return baseline - between


def _mutual_information(cloud, asked_k, s_mean, s_sd, config) -> float:
    probabilities = _candidate_probabilities(cloud, asked_k, s_mean, s_sd, config)
    marginal = cloud.w @ probabilities
    ratio = np.log(np.maximum(probabilities, 1e-300)) - np.log(np.maximum(marginal, 1e-300))
    return float(np.sum(cloud.w[:, None] * probabilities * ratio))


def _fisher_utility(cloud, asked_k, s_mean, s_sd, config, moments=None) -> float:
    if moments is None:
        moments = posterior_moments(cloud)
    t = moments["t_mean"][None, :].copy()
    l = moments["l_mean"][None, :].copy()
    artifact_draws = getattr(config, "artifact_draws", ())
    if artifact_draws:
        weights = np.asarray([draw[2] for draw in artifact_draws], dtype=float)
        weights /= weights.sum()
        screen_beta = float(sum(
            weight * draw[0] for weight, draw in zip(weights, artifact_draws, strict=True)
        ))
        screen_lapse = float(sum(
            weight * draw[1] for weight, draw in zip(weights, artifact_draws, strict=True)
        ))
    else:
        screen_beta = config.beta
        screen_lapse = config.distractor_lapse
    base = f1_probabilities(
        t, l, s_mean, s_sd, asked_k, IIIC_GROUP,
        screen_beta, screen_lapse,
    )[0]
    utility = 0.0
    step = 1e-3
    for values, sd in ((t, moments["t_sd"]), (l, moments["l_sd"])):
        for k in IIIC_GROUP:
            original = values[0, k]
            values[0, k] = original + step
            plus = f1_probabilities(
                t, l, s_mean, s_sd, asked_k, IIIC_GROUP,
                screen_beta, screen_lapse,
            )[0]
            values[0, k] = original - step
            minus = f1_probabilities(
                t, l, s_mean, s_sd, asked_k, IIIC_GROUP,
                screen_beta, screen_lapse,
            )[0]
            values[0, k] = original
            derivative = (plus - minus) / (2 * step)
            information = float(np.sum(np.square(derivative) / np.maximum(base, 1e-12)))
            variance = float(sd[k] ** 2)
            utility += variance * variance * information / (1 + variance * information)
    return utility


def _shortlist(cloud, remaining, counts, s_mean, s_sd, config, augment):
    ordered = np.asarray(sorted(remaining), dtype=int)
    selected: set[tuple[int, int]] = set()
    moments = posterior_moments(cloud) if augment else None
    for asked_k in IIIC_GROUP:
        if counts[asked_k] >= config.own_cap:
            continue
        n = min(config.selector_per_domain, ordered.size)
        positions = np.unique(np.rint(np.linspace(0, ordered.size - 1, n)).astype(int))
        selected.update((asked_k, int(ordered[position])) for position in positions)
        if augment:
            scored = sorted(
                (
                    _fisher_utility(
                        cloud, asked_k, s_mean[index], s_sd[index], config, moments,
                    ),
                    int(index),
                )
                for index in ordered
            )
            selected.update(
                (asked_k, index)
                for _, index in scored[-config.fisher_per_domain:]
            )
    return sorted(selected)


def _select(cloud, selector, remaining, counts, s_mean, s_sd, config):
    augment = selector != "frozen_total_variance"
    candidates = _shortlist(
        cloud, remaining, counts, s_mean, s_sd, config, augment,
    )
    best = (math.inf, -1, -1)
    for asked_k, segment_index in candidates:
        if selector in ("frozen_total_variance", "fisher_augmented_total_variance"):
            loss = expected_loss(
                cloud, asked_k, s_mean[segment_index], s_sd[segment_index],
                IIIC_GROUP, config.beta, config.distractor_lapse,
                config.artifact_draws,
            )
        elif selector == "bias_weighted_total_variance":
            loss = _weighted_loss(
                cloud, asked_k, s_mean[segment_index], s_sd[segment_index], config,
            )
        else:
            loss = -_mutual_information(
                cloud, asked_k, s_mean[segment_index], s_sd[segment_index], config,
            )
        best = min(best, (loss, asked_k, segment_index))
    if best[1] < 0:
        raise RuntimeError("selector exhausted the adaptive bank")
    return best[1], best[2]


def _run_selector(seed: int, selector: SelectorName, config: SelectorExperimentConfig) -> dict:
    truth_t, truth_l, s_mean, s_sd = _truth_and_bank(seed, _qualification_config(config))
    cloud = make_cloud(
        config.particles, np.eye(7), np.eye(7),
        np.random.default_rng(seed + 10_000),
    )
    response_rng = np.random.default_rng(seed + 20_000)
    mh_rng = np.random.default_rng(seed + 30_000)
    if config.truth_artifact_draws:
        truth_weights = np.asarray(
            [draw[2] for draw in config.truth_artifact_draws], dtype=float,
        )
        if np.any(~np.isfinite(truth_weights)) or np.any(truth_weights <= 0):
            raise ValueError("truth artifact weights must be finite and positive")
        truth_weights /= truth_weights.sum()
        truth_index = int(np.random.default_rng(seed + 40_000).choice(
            len(config.truth_artifact_draws), p=truth_weights,
        ))
        truth_beta, truth_lapse, _ = config.truth_artifact_draws[truth_index]
    else:
        truth_index = -1
        truth_beta = config.beta if config.truth_beta is None else config.truth_beta
        truth_lapse = (
            config.distractor_lapse
            if config.truth_distractor_lapse is None
            else config.truth_distractor_lapse
        )
    remaining = set(range(config.bank_segments))
    counts = np.zeros(7, dtype=int)
    resamples = 0
    acceptance = []
    ancestry = []
    while np.any(counts[1:] < config.own_cap):
        asked_k, segment_index = _select(
            cloud, selector, remaining, counts, s_mean, s_sd, config,
        )
        truth_probabilities = f1_probabilities(
            truth_t, truth_l, s_mean[segment_index], s_sd[segment_index],
            asked_k, IIIC_GROUP,
            truth_beta, truth_lapse,
        )[0]
        pick = int(response_rng.choice(IIIC_GROUP, p=truth_probabilities))
        observation = Observation(
            "categorical_f1", asked_k, segment_index, pick, group=IIIC_GROUP,
        )
        update(
            cloud, observation, s_mean[segment_index], s_sd[segment_index],
            config.beta, config.distractor_lapse, config.artifact_draws,
        )
        if ess(cloud) < config.ess_fraction * cloud.n:
            accepted, ancestors = resample_and_rejuvenate(
                cloud, s_mean, s_sd, config.beta, config.distractor_lapse,
                mh_rng, config.mh_steps, 2.38 / math.sqrt(14),
                artifact_draws=config.artifact_draws,
            )
            resamples += 1
            acceptance.append(accepted)
            ancestry.append(ancestors)
        remaining.remove(segment_index)
        counts[asked_k] += 1
    moments = posterior_moments(cloud)
    result = {
        "seed": seed,
        "selector": selector,
        "truth_artifact_draw_index": truth_index,
        "truth_beta": truth_beta,
        "truth_distractor_lapse": truth_lapse,
        "questions": int(counts.sum()),
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
        truths = truth[list(IIIC_GROUP)]
        result[f"{parameter}_coverage"] = float(np.mean(
            (lows <= truths) & (truths <= highs)
        ))
        result[f"{parameter}_width"] = float(np.mean(highs - lows))
        result[f"{parameter}_rmse"] = float(np.sqrt(np.mean(np.square(
            means[list(IIIC_GROUP)] - truths
        ))))
    return result


def _run_replicate(index: int, config: SelectorExperimentConfig) -> list[dict]:
    seed = config.seed_base + index
    return [_run_selector(seed, selector, config) for selector in config.selectors]


def _mean_ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else math.nan
    return {"mean": mean, "standard_error": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


def summarize(rows: list[dict], config: SelectorExperimentConfig) -> dict:
    frozen_name = "frozen_total_variance"
    by_seed = {
        seed: {row["selector"]: row for row in rows if row["seed"] == seed}
        for seed in sorted({row["seed"] for row in rows})
    }
    result = {
        "schema_version": 1,
        "status": "research_only_not_promoted",
        "config": asdict(config),
        "selectors": {},
        "paired_vs_frozen": {},
    }
    for selector in config.selectors:
        selected = [row for row in rows if row["selector"] == selector]
        result["selectors"][selector] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "skill_coverage", "bias_coverage", "skill_width", "bias_width",
                "skill_rmse", "bias_rmse", "questions", "resamples",
                "acceptance", "ancestry",
            )
        }
        for parameter in ("skill", "bias"):
            result["selectors"][selector][f"{parameter}_coverage_clustered"] = _mean_ci([
                row[f"{parameter}_coverage"] for row in selected
            ])
        if selector == frozen_name:
            continue
        comparison = {}
        for parameter in ("skill", "bias"):
            coverage_differences = [
                arms[selector][f"{parameter}_coverage"]
                - arms[frozen_name][f"{parameter}_coverage"]
                for arms in by_seed.values()
            ]
            coverage = _mean_ci(coverage_differences)
            comparison[f"{parameter}_coverage_difference"] = coverage
            comparison[f"{parameter}_coverage_noninferiority_lcb_pass"] = (
                coverage["ci95"][0] >= -0.03
            )
            width_ratios = [
                arms[selector][f"{parameter}_width"]
                / arms[frozen_name][f"{parameter}_width"]
                for arms in by_seed.values()
            ]
            rmse_ratios = [
                arms[selector][f"{parameter}_rmse"]
                / max(arms[frozen_name][f"{parameter}_rmse"], 1e-12)
                for arms in by_seed.values()
            ]
            comparison[f"{parameter}_width_ratio"] = float(np.mean(width_ratios))
            comparison[f"{parameter}_width_ratio_uncertainty"] = _mean_ci(width_ratios)
            comparison[f"{parameter}_width_ratio_of_means"] = (
                result["selectors"][selector][f"{parameter}_width"]
                / result["selectors"][frozen_name][f"{parameter}_width"]
            )
            comparison[f"{parameter}_rmse_ratio"] = float(np.mean(rmse_ratios))
            comparison[f"{parameter}_rmse_ratio_uncertainty"] = _mean_ci(rmse_ratios)
            comparison[f"{parameter}_rmse_ratio_of_means"] = (
                result["selectors"][selector][f"{parameter}_rmse"]
                / result["selectors"][frozen_name][f"{parameter}_rmse"]
            )
        result["paired_vs_frozen"][selector] = comparison
    result["promotion_note"] = (
        "No selector is promoted by this development run. A locked served-bank "
        "stopping-time qualification remains mandatory."
    )
    return result


def run(config: SelectorExperimentConfig, workers: int | None = None) -> tuple[list[dict], dict]:
    worker_count = workers or recommended_workers(config.replicates, config.particles)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        nested = list(executor.map(
            _run_replicate,
            range(config.replicates),
            [config] * config.replicates,
        ))
    rows = [row for replicate in nested for row in replicate]
    return rows, summarize(rows, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-ensemble", type=Path)
    parser.add_argument("--ensemble-draws", type=int, default=9)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--truth-beta", type=float)
    parser.add_argument("--truth-distractor-lapse", type=float)
    parser.add_argument("--selectors")
    parser.add_argument("--seed-base", type=int)
    args = parser.parse_args()
    config = SelectorExperimentConfig()
    if args.mode == "smoke":
        config = replace(
            config,
            replicates=4,
            particles=96,
            mh_steps=2,
            own_cap=1,
            bank_segments=18,
            selector_per_domain=2,
            fisher_per_domain=1,
        )
    if args.replicates is not None:
        config = replace(config, replicates=args.replicates)
    config = replace(
        config,
        **{
            key: value for key, value in {
                "beta": args.beta,
                "truth_beta": args.truth_beta,
                "truth_distractor_lapse": args.truth_distractor_lapse,
                "seed_base": args.seed_base,
            }.items() if value is not None
        },
    )
    if args.selectors:
        requested = tuple(args.selectors.split(","))
        unknown = set(requested) - set(SelectorExperimentConfig().selectors)
        if unknown or "frozen_total_variance" not in requested:
            raise ValueError("selectors must be known and include frozen_total_variance")
        config = replace(config, selectors=requested)
    if args.artifact_ensemble:
        payload = json.loads(args.artifact_ensemble.read_text())
        draws = sorted(payload["bootstrap"]["draws"], key=lambda draw: draw["beta"])
        n = min(args.ensemble_draws, len(draws))
        indices = np.unique(np.rint(np.linspace(0, len(draws) - 1, n)).astype(int))
        weight = 1 / len(indices)
        config = replace(config, artifact_draws=tuple(
            (
                float(draws[index]["beta"]),
                float(draws[index]["distractor_lapse"]),
                weight,
            )
            for index in indices
        ))
    rows, summary = run(config, args.workers)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"summary": summary, "rows": rows}, indent=2, sort_keys=True,
        ) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
