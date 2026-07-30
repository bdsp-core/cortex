"""Parallel planted-truth qualification harness for binary versus F1 arms."""
from __future__ import annotations

import os

# One BLAS thread per process prevents 40 process workers from each creating a
# second full thread pool.  Process-level parallelism owns the machine budget.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from .reference import (
    Observation,
    ess,
    expected_loss,
    f1_probabilities,
    make_cloud,
    posterior_moments,
    resample_and_rejuvenate,
    update,
    weighted_quantile,
)

IIIC_GROUP = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class QualificationConfig:
    replicates: int = 12
    particles: int = 192
    own_cap: int = 4
    bank_segments: int = 72
    selector_per_domain: int = 5
    mh_steps: int = 2
    ess_fraction: float = 0.5
    beta: float = 0.9912
    distractor_lapse: float = 0.0
    artifact_draws: tuple[tuple[float, float, float], ...] = ()
    truth_beta: float | None = None
    truth_distractor_lapse: float | None = None
    truth_t_mean: float = 0.0
    truth_l_mean: float = 0.2
    truth_sd: float = 0.7
    formal_sbc: bool = False
    signal_sd_scale: float = 1.0


@dataclass
class ReplicateResult:
    seed: int
    arm: Literal["binary", "categorical_f1"]
    skill_rmse: float
    bias_rmse: float
    skill_width: float
    bias_width: float
    skill_coverage: float
    bias_coverage: float
    questions: int
    resamples: int
    mean_acceptance: float
    mean_ancestry: float
    skill_sbc_ranks: list[float]
    bias_sbc_ranks: list[float]


def _truth_and_bank(seed: int, config: QualificationConfig):
    if not np.isfinite(config.signal_sd_scale) or config.signal_sd_scale <= 0:
        raise ValueError("signal_sd_scale must be finite and positive")
    rng = np.random.default_rng(seed)
    truth_t = rng.normal(config.truth_t_mean, config.truth_sd, 7)
    truth_l = rng.normal(config.truth_l_mean, config.truth_sd, 7)
    truth_t[0], truth_l[0] = 0.0, 0.0
    s_mean = rng.normal(0, 1.25, size=(config.bank_segments, 7))
    # Preserve realistic cross-class ambiguity rather than six independent axes.
    common = rng.normal(0, 0.5, size=(config.bank_segments, 1))
    s_mean[:, 1:] += common
    s_sd = rng.uniform(0.02, 0.18, size=(config.bank_segments, 7)) * config.signal_sd_scale
    return truth_t, truth_l, s_mean, s_sd


def _select(
    cloud,
    arm: str,
    remaining: set[int],
    counts: np.ndarray,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    config: QualificationConfig,
) -> tuple[int, int]:
    best = (math.inf, -1, -1)
    ordered = np.array(sorted(remaining), dtype=int)
    for asked_k in IIIC_GROUP:
        if counts[asked_k] >= config.own_cap:
            continue
        # Stable quantile representatives keep smoke and full runs bounded.
        n = min(config.selector_per_domain, ordered.size)
        candidate_positions = np.unique(np.rint(np.linspace(0, ordered.size - 1, n)).astype(int))
        for position in candidate_positions:
            segment_index = int(ordered[position])
            loss = expected_loss(
                cloud, asked_k, s_mean[segment_index], s_sd[segment_index],
                IIIC_GROUP if arm == "categorical_f1" else None,
                config.beta, config.distractor_lapse, config.artifact_draws,
            )
            candidate = (loss, asked_k, segment_index)
            if candidate < best:
                best = candidate
    if best[1] < 0:
        raise RuntimeError("adaptive bank exhausted before direct-domain caps")
    return best[1], best[2]


def _run_arm(seed: int, arm: Literal["binary", "categorical_f1"], config: QualificationConfig):
    truth_t, truth_l, s_mean, s_sd = _truth_and_bank(seed, config)
    prior_corr = np.eye(7)
    cloud = make_cloud(
        config.particles, prior_corr, prior_corr,
        np.random.default_rng(seed + (10_000 if arm == "binary" else 20_000)),
    )
    response_rng = np.random.default_rng(seed + (30_000 if arm == "binary" else 40_000))
    mh_rng = np.random.default_rng(seed + (50_000 if arm == "binary" else 60_000))
    remaining = set(range(config.bank_segments))
    counts = np.zeros(7, dtype=int)
    acceptances, ancestries = [], []
    questions = 0
    while np.any(counts[1:] < config.own_cap):
        asked_k, segment_index = _select(
            cloud, arm, remaining, counts, s_mean, s_sd, config,
        )
        truth_probabilities = f1_probabilities(
            truth_t, truth_l, s_mean[segment_index], s_sd[segment_index],
            asked_k, IIIC_GROUP,
            config.beta if config.truth_beta is None else config.truth_beta,
            (
                config.distractor_lapse
                if config.truth_distractor_lapse is None
                else config.truth_distractor_lapse
            ),
        )[0]
        pick = int(response_rng.choice(IIIC_GROUP, p=truth_probabilities))
        observation = Observation(
            kind="categorical_f1" if arm == "categorical_f1" else "binary",
            asked_k=asked_k,
            segment_index=segment_index,
            raw_pick=pick,
            y=None if arm == "categorical_f1" else int(pick == asked_k),
            group=IIIC_GROUP if arm == "categorical_f1" else (),
        )
        update(
            cloud, observation, s_mean[segment_index], s_sd[segment_index],
            config.beta, config.distractor_lapse, config.artifact_draws,
        )
        if ess(cloud) < config.ess_fraction * config.particles:
            acceptance, ancestry = resample_and_rejuvenate(
                cloud, s_mean, s_sd, config.beta, config.distractor_lapse,
                mh_rng, config.mh_steps, 2.38 / np.sqrt(14),
                artifact_draws=config.artifact_draws,
            )
            acceptances.append(acceptance)
            ancestries.append(ancestry)
        remaining.remove(segment_index)
        counts[asked_k] += 1
        questions += 1

    moments = posterior_moments(cloud)
    skill_covered, bias_covered, skill_widths, bias_widths = [], [], [], []
    for k in IIIC_GROUP:
        l_low = weighted_quantile(cloud.l[:, k], cloud.w, 0.025)
        l_high = weighted_quantile(cloud.l[:, k], cloud.w, 0.975)
        t_low = weighted_quantile(cloud.t[:, k], cloud.w, 0.025)
        t_high = weighted_quantile(cloud.t[:, k], cloud.w, 0.975)
        skill_covered.append(l_low <= truth_l[k] <= l_high)
        bias_covered.append(t_low <= truth_t[k] <= t_high)
        skill_widths.append(l_high - l_low)
        bias_widths.append(t_high - t_low)
    return ReplicateResult(
        seed=seed,
        arm=arm,
        skill_rmse=float(np.sqrt(np.mean(np.square(moments["l_mean"][1:] - truth_l[1:])))),
        bias_rmse=float(np.sqrt(np.mean(np.square(moments["t_mean"][1:] - truth_t[1:])))),
        skill_width=float(np.mean(skill_widths)),
        bias_width=float(np.mean(bias_widths)),
        skill_coverage=float(np.mean(skill_covered)),
        bias_coverage=float(np.mean(bias_covered)),
        questions=questions,
        resamples=len(acceptances),
        mean_acceptance=float(np.mean(acceptances) if acceptances else 0),
        mean_ancestry=float(np.mean(ancestries) if ancestries else 1),
        skill_sbc_ranks=[
            float(cloud.w[cloud.l[:, k] < truth_l[k]].sum()) for k in IIIC_GROUP
        ],
        bias_sbc_ranks=[
            float(cloud.w[cloud.t[:, k] < truth_t[k]].sum()) for k in IIIC_GROUP
        ],
    )


def run_replicate(seed: int, config: QualificationConfig) -> list[ReplicateResult]:
    return [_run_arm(seed, "binary", config), _run_arm(seed, "categorical_f1", config)]


def _mem_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 * 1024**3


def recommended_workers(replicates: int, particles: int) -> int:
    cpu_budget = max(1, (os.cpu_count() or 2) - 2)
    # Conservative allowance for covariance workspaces and Python process
    # overhead.  Keep at least 20% of available RAM unused.
    estimated_per_worker = max(512 * 1024**2, particles * 7 * 8 * 80)
    memory_budget = max(1, int(_mem_available_bytes() * 0.80 // estimated_per_worker))
    return max(1, min(replicates, cpu_budget, memory_budget))


def summarize(results: list[ReplicateResult], config: QualificationConfig) -> dict:
    def mean_ci(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        standard_error = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else math.nan
        return {
            "mean": mean,
            "standard_error": standard_error,
            "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
            "clusters": int(array.size),
        }

    def wilson(successes: int, trials: int) -> list[float]:
        z = 1.96
        p = successes / trials
        denominator = 1 + z * z / trials
        center = (p + z * z / (2 * trials)) / denominator
        radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
        return [center - radius, center + radius]

    def sbc_summary(values: list[float]) -> dict:
        ordered = np.sort(np.asarray(values, dtype=float))
        n = ordered.size
        upper = np.arange(1, n + 1) / n
        lower = np.arange(0, n) / n
        ks = float(max(np.max(upper - ordered), np.max(ordered - lower)))
        histogram, _ = np.histogram(ordered, bins=np.linspace(0, 1, 11))
        return {
            "n": int(n),
            "ks_distance_from_uniform": ks,
            "decile_histogram": histogram.tolist(),
            "formal_sbc": config.formal_sbc,
            "interpretation": (
                "prior-predictive SBC rank diagnostic"
                if config.formal_sbc
                else "population-stress posterior-rank diagnostic; truth prior differs from inference prior"
            ),
        }

    by_arm = {}
    for arm in ("binary", "categorical_f1"):
        rows = [row for row in results if row.arm == arm]
        by_arm[arm] = {
            key: float(np.mean([getattr(row, key) for row in rows]))
            for key in (
                "skill_rmse", "bias_rmse", "skill_width", "bias_width",
                "skill_coverage", "bias_coverage", "questions", "resamples",
                "mean_acceptance", "mean_ancestry",
            )
        }
        by_arm[arm]["skill_sbc"] = sbc_summary([
            rank for row in rows for rank in row.skill_sbc_ranks
        ])
        by_arm[arm]["bias_sbc"] = sbc_summary([
            rank for row in rows for rank in row.bias_sbc_ranks
        ])
        for parameter in ("skill", "bias"):
            values = [getattr(row, f"{parameter}_coverage") for row in rows]
            successes = int(round(sum(values) * 6))
            trials = len(values) * 6
            by_arm[arm][f"{parameter}_coverage_uncertainty"] = {
                "successes": successes,
                "trials": trials,
                "wilson95": wilson(successes, trials),
                "seed_cluster95": mean_ci(values),
            }

    paired = {}
    by_seed: dict[int, dict[str, ReplicateResult]] = {}
    for row in results:
        by_seed.setdefault(row.seed, {})[row.arm] = row
    for parameter in ("skill", "bias"):
        differences = [
            getattr(pair["categorical_f1"], f"{parameter}_coverage")
            - getattr(pair["binary"], f"{parameter}_coverage")
            for pair in by_seed.values()
        ]
        comparison = mean_ci(differences)
        comparison["noninferiority_margin"] = -0.03
        comparison["point_pass"] = comparison["mean"] >= -0.03
        comparison["confidence_pass"] = comparison["ci95"][0] >= -0.03
        paired[f"{parameter}_coverage_nway_minus_binary"] = comparison
    return {
        "schema_version": 1,
        "status": "research_only_unqualified_artifact",
        "config": asdict(config),
        "arms": by_arm,
        "paired_comparisons": paired,
        "ratios": {
            "skill_width_nway_over_binary": (
                by_arm["categorical_f1"]["skill_width"] / by_arm["binary"]["skill_width"]
            ),
            "bias_width_nway_over_binary": (
                by_arm["categorical_f1"]["bias_width"] / by_arm["binary"]["bias_width"]
            ),
            "skill_rmse_nway_over_binary": (
                by_arm["categorical_f1"]["skill_rmse"] / by_arm["binary"]["skill_rmse"]
            ),
            "bias_rmse_nway_over_binary": (
                by_arm["categorical_f1"]["bias_rmse"] / by_arm["binary"]["bias_rmse"]
            ),
        },
    }


def run_parallel(config: QualificationConfig, workers: int | None = None) -> dict:
    workers = workers or recommended_workers(config.replicates, config.particles)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        paired = list(executor.map(
            run_replicate,
            range(62_100_000, 62_100_000 + config.replicates),
            [config] * config.replicates,
        ))
    summary = summarize([row for pair in paired for row in pair], config)
    summary["workers"] = workers
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "qualification"), default="smoke")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--mh-steps", type=int)
    parser.add_argument("--own-cap", type=int)
    parser.add_argument("--bank-segments", type=int)
    parser.add_argument("--selector-per-domain", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--distractor-lapse", type=float)
    parser.add_argument("--truth-beta", type=float)
    parser.add_argument("--truth-distractor-lapse", type=float)
    parser.add_argument("--signal-sd-scale", type=float)
    parser.add_argument("--artifact-ensemble", type=Path)
    parser.add_argument("--ensemble-draws", type=int, default=9)
    parser.add_argument("--formal-sbc", action="store_true")
    args = parser.parse_args()
    if args.mode == "qualification":
        config = QualificationConfig(
            replicates=args.replicates or 1_000,
            particles=1_200,
            own_cap=15,
            bank_segments=420,
            selector_per_domain=32,
            mh_steps=30,
        )
    else:
        config = QualificationConfig(replicates=args.replicates or 12)
    config = replace(
        config,
        **{
            key: value for key, value in {
                "particles": args.particles,
                "mh_steps": args.mh_steps,
                "own_cap": args.own_cap,
                "bank_segments": args.bank_segments,
                "selector_per_domain": args.selector_per_domain,
                "beta": args.beta,
                "distractor_lapse": args.distractor_lapse,
                "truth_beta": args.truth_beta,
                "truth_distractor_lapse": args.truth_distractor_lapse,
                "signal_sd_scale": args.signal_sd_scale,
            }.items() if value is not None
        },
    )
    if args.artifact_ensemble:
        payload = json.loads(args.artifact_ensemble.read_text())
        draws = sorted(
            payload["bootstrap"]["draws"], key=lambda draw: draw["beta"],
        )
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
    if args.formal_sbc:
        config = replace(
            config,
            truth_t_mean=0.0,
            truth_l_mean=0.0,
            truth_sd=1.0,
            formal_sbc=True,
        )
    summary = run_parallel(config, args.workers)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
