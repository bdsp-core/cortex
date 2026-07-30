"""Matched frozen-F1 versus fully integrated n-way R&D experiment."""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np

from .qualification import recommended_workers
from .selector_experiment import SelectorExperimentConfig, _run_selector

FROZEN_ARM = "frozen_f1_v1"
INTEGRATED_ARM = "integrated_ensemble9_fisher_v1"


@dataclass(frozen=True)
class IntegrationExperimentConfig:
    replicates: int = 192
    particles: int = 1200
    mh_steps: int = 30
    own_cap: int = 4
    bank_segments: int = 72
    selector_per_domain: int = 5
    fisher_per_domain: int = 3
    ess_fraction: float = 0.5
    frozen_beta: float = 0.9912
    frozen_distractor_lapse: float = 0.0
    integrated_beta: float = 1.0303243343107302
    integrated_distractor_lapse: float = 0.0
    integrated_draws: tuple[tuple[float, float, float], ...] = ()
    seed_base: int = 63_800_000
    coverage_noninferiority_margin: float = -0.03


def _selector_config(
    config: IntegrationExperimentConfig,
    integrated: bool,
) -> SelectorExperimentConfig:
    return SelectorExperimentConfig(
        replicates=1,
        particles=config.particles,
        mh_steps=config.mh_steps,
        own_cap=config.own_cap,
        bank_segments=config.bank_segments,
        selector_per_domain=config.selector_per_domain,
        fisher_per_domain=config.fisher_per_domain,
        ess_fraction=config.ess_fraction,
        beta=config.integrated_beta if integrated else config.frozen_beta,
        distractor_lapse=(
            config.integrated_distractor_lapse
            if integrated else config.frozen_distractor_lapse
        ),
        artifact_draws=config.integrated_draws if integrated else (),
        truth_artifact_draws=config.integrated_draws,
        seed_base=config.seed_base,
        selectors=(
            ("fisher_augmented_total_variance",)
            if integrated else ("frozen_total_variance",)
        ),
    )


def _run_replicate(index: int, config: IntegrationExperimentConfig) -> list[dict]:
    seed = config.seed_base + index
    frozen = _run_selector(
        seed, "frozen_total_variance", _selector_config(config, False),
    )
    integrated = _run_selector(
        seed, "fisher_augmented_total_variance", _selector_config(config, True),
    )
    if (
        frozen["truth_artifact_draw_index"]
        != integrated["truth_artifact_draw_index"]
    ):
        raise RuntimeError("matched arms received different artifact truths")
    frozen["arm"] = FROZEN_ARM
    integrated["arm"] = INTEGRATED_ARM
    return [frozen, integrated]


def _mean_ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else math.nan
    return {
        "mean": mean,
        "standard_error": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
    }


def summarize(rows: list[dict], config: IntegrationExperimentConfig) -> dict:
    by_seed = {
        seed: {row["arm"]: row for row in rows if row["seed"] == seed}
        for seed in sorted({row["seed"] for row in rows})
    }
    if any(set(arms) != {FROZEN_ARM, INTEGRATED_ARM} for arms in by_seed.values()):
        raise ValueError("every seed must contain one complete matched arm pair")
    result = {
        "schema_version": 1,
        "status": "research_only_not_promoted",
        "comparison": "frozen conditional-F1 versus ensemble9 plus Fisher shortlist",
        "truth_model": (
            "skill/bias prior-predictive SBC under response-artifact heterogeneity "
            "stress; one fixed artifact draw sampled per matched replicate"
        ),
        "config": asdict(config),
        "arms": {},
        "paired_integrated_minus_frozen": {},
        "truth_draw_counts": dict(sorted(Counter(
            arms[FROZEN_ARM]["truth_artifact_draw_index"]
            for arms in by_seed.values()
        ).items())),
    }
    metrics = (
        "skill_coverage", "bias_coverage", "skill_width", "bias_width",
        "skill_rmse", "bias_rmse", "questions", "resamples",
        "acceptance", "ancestry",
    )
    for arm in (FROZEN_ARM, INTEGRATED_ARM):
        selected = [row for row in rows if row["arm"] == arm]
        result["arms"][arm] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in metrics
        }
        for parameter in ("skill", "bias"):
            result["arms"][arm][f"{parameter}_coverage_clustered"] = _mean_ci([
                row[f"{parameter}_coverage"] for row in selected
            ])
    comparison = result["paired_integrated_minus_frozen"]
    for parameter in ("skill", "bias"):
        coverage = _mean_ci([
            arms[INTEGRATED_ARM][f"{parameter}_coverage"]
            - arms[FROZEN_ARM][f"{parameter}_coverage"]
            for arms in by_seed.values()
        ])
        comparison[f"{parameter}_coverage_difference"] = coverage
        comparison[f"{parameter}_coverage_noninferiority_lcb_pass"] = (
            coverage["ci95"][0] >= config.coverage_noninferiority_margin
        )
        for metric in ("width", "rmse"):
            ratios = [
                arms[INTEGRATED_ARM][f"{parameter}_{metric}"]
                / max(arms[FROZEN_ARM][f"{parameter}_{metric}"], 1e-12)
                for arms in by_seed.values()
            ]
            differences = [
                arms[INTEGRATED_ARM][f"{parameter}_{metric}"]
                - arms[FROZEN_ARM][f"{parameter}_{metric}"]
                for arms in by_seed.values()
            ]
            comparison[f"{parameter}_{metric}_ratio"] = _mean_ci(ratios)
            comparison[f"{parameter}_{metric}_difference"] = _mean_ci(differences)
            comparison[f"{parameter}_{metric}_ratio_of_means"] = (
                result["arms"][INTEGRATED_ARM][f"{parameter}_{metric}"]
                / result["arms"][FROZEN_ARM][f"{parameter}_{metric}"]
            )
    result["development_gate"] = {
        "relative_coverage_noninferiority_pass": all(
            comparison[f"{parameter}_coverage_noninferiority_lcb_pass"]
            for parameter in ("skill", "bias")
        ),
        "paired_width_tightening_pass": all(
            comparison[f"{parameter}_width_ratio"]["ci95"][1] < 1
            for parameter in ("skill", "bias")
        ),
        "absolute_coverage_gate": "not_preregistered_not_evaluated",
    }
    strata = {}
    for truth_index in sorted({
        row["truth_artifact_draw_index"] for row in rows
    }):
        selected_pairs = {
            seed: arms for seed, arms in by_seed.items()
            if arms[FROZEN_ARM]["truth_artifact_draw_index"] == truth_index
        }
        example = next(iter(selected_pairs.values()))[FROZEN_ARM]
        stratum = {
            "replicates": len(selected_pairs),
            "truth_beta": example["truth_beta"],
            "truth_distractor_lapse": example["truth_distractor_lapse"],
            "arms": {},
            "paired_integrated_minus_frozen": {},
        }
        for arm in (FROZEN_ARM, INTEGRATED_ARM):
            stratum["arms"][arm] = {
                metric: float(np.mean([
                    arms[arm][metric] for arms in selected_pairs.values()
                ]))
                for metric in (
                    "skill_coverage", "bias_coverage", "skill_width", "bias_width",
                    "skill_rmse", "bias_rmse",
                )
            }
        for parameter in ("skill", "bias"):
            stratum["paired_integrated_minus_frozen"][
                f"{parameter}_coverage_difference"
            ] = float(np.mean([
                arms[INTEGRATED_ARM][f"{parameter}_coverage"]
                - arms[FROZEN_ARM][f"{parameter}_coverage"]
                for arms in selected_pairs.values()
            ]))
            stratum["paired_integrated_minus_frozen"][
                f"{parameter}_width_ratio"
            ] = float(np.mean([
                arms[INTEGRATED_ARM][f"{parameter}_width"]
                / arms[FROZEN_ARM][f"{parameter}_width"]
                for arms in selected_pairs.values()
            ]))
        strata[str(truth_index)] = stratum
    result["artifact_truth_strata"] = strata
    result["promotion_note"] = (
        "This integrates the selected R&D features only inside n-way-protocol. "
        "The artifact remains unqualified and production promotion is forbidden."
    )
    return result


def run(
    config: IntegrationExperimentConfig,
    workers: int | None = None,
) -> tuple[list[dict], dict]:
    if not config.integrated_draws:
        raise ValueError("the integrated comparison requires artifact draws")
    worker_count = workers or recommended_workers(config.replicates, config.particles)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        nested = list(executor.map(
            _run_replicate,
            range(config.replicates),
            [config] * config.replicates,
        ))
    rows = [row for pair in nested for row in pair]
    return rows, summarize(rows, config)


def _load_draws(path: Path) -> tuple[tuple[float, float, float], ...]:
    payload = json.loads(path.read_text())
    return tuple(
        (
            float(draw["beta"]),
            float(draw["distractorLapse"]),
            float(draw["weight"]),
        )
        for draw in payload["draws"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--artifact", type=Path, default=Path(
        "artifacts/iiic_conditional_f1_integrated_ensemble9_rd.json",
    ))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = IntegrationExperimentConfig(integrated_draws=_load_draws(args.artifact))
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
    if args.seed_base is not None:
        config = replace(config, seed_base=args.seed_base)
    rows, summary = run(config, args.workers)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"summary": summary, "rows": rows}, indent=2, sort_keys=True,
        ) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
