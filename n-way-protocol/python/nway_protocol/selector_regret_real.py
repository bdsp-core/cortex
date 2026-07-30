"""Shortlist-regret audit on staged empirical IIIC signal axes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np

from .bank_audit import BankSegment, _read_bank
from .qualification import IIIC_GROUP
from .reference import (
    Observation,
    ess,
    expected_loss,
    f1_probabilities,
    make_cloud,
    posterior_moments,
    resample_and_rejuvenate,
    update,
)
from .selector_experiment import _fisher_utility


@dataclass(frozen=True)
class RealRegretConfig:
    states: int = 24
    particles: int = 1200
    mh_steps: int = 30
    bank_sample: int = 420
    coarse_per_task: int = 32
    entropy_per_task: int = 12
    fisher_per_task: int = 8
    beta: float = 1.0303243343107302
    distractor_lapse: float = 0.0
    ess_fraction: float = 0.5
    seed_base: int = 63_700_000


def _entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def _make_state(index: int, segments: list[BankSegment], config: RealRegretConfig):
    seed = config.seed_base + index
    rng = np.random.default_rng(seed)
    cloud = make_cloud(config.particles, np.eye(7), np.eye(7), rng)
    truth_t = rng.normal(size=(1, 7))
    truth_l = rng.normal(size=(1, 7))
    history_length = (index % 8) * 4
    mh_rng = np.random.default_rng(seed + 100_000)
    for question in range(min(history_length, len(segments))):
        segment = segments[question]
        asked_k = IIIC_GROUP[question % len(IIIC_GROUP)]
        probabilities = f1_probabilities(
            truth_t, truth_l,
            np.asarray(segment.s_mean), np.asarray(segment.s_sd),
            asked_k, IIIC_GROUP, config.beta, config.distractor_lapse,
        )[0]
        pick = int(rng.choice(IIIC_GROUP, p=probabilities))
        observation = Observation(
            "categorical_f1", asked_k, question, pick, group=IIIC_GROUP,
        )
        update(
            cloud, observation, np.asarray(segment.s_mean), np.asarray(segment.s_sd),
            config.beta, config.distractor_lapse,
        )
        if ess(cloud) < config.ess_fraction * cloud.n:
            s_means = np.asarray([segment.s_mean for segment in segments])
            s_sds = np.asarray([segment.s_sd for segment in segments])
            resample_and_rejuvenate(
                cloud, s_means, s_sds, config.beta, config.distractor_lapse,
                mh_rng, config.mh_steps, 2.38 / math.sqrt(14),
            )
    return cloud, history_length


def _baseline_shortlist(cloud, segments: list[BankSegment], config: RealRegretConfig):
    selected: set[tuple[int, int]] = set()
    moments = posterior_moments(cloud)
    for asked_k in IIIC_GROUP:
        ordered = sorted(
            range(len(segments)),
            key=lambda index: (segments[index].s_mean[asked_k], segments[index].seg_id),
        )
        count = min(config.coarse_per_task, len(ordered))
        positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int))
        selected.update((asked_k, ordered[int(position)]) for position in positions)
        width = max(1, math.ceil(len(ordered) / config.coarse_per_task))
        for start in range(0, len(ordered), width):
            group = ordered[start:start + width]
            selected.add((asked_k, min(
                group, key=lambda index: (segments[index].s_sd[asked_k], segments[index].seg_id),
            )))
        entropies = []
        for segment_index in ordered:
            segment = segments[segment_index]
            probabilities = f1_probabilities(
                moments["t_mean"], moments["l_mean"],
                np.asarray(segment.s_mean), np.asarray(segment.s_sd),
                asked_k, IIIC_GROUP, config.beta, config.distractor_lapse,
            )[0]
            entropies.append((_entropy(probabilities), segment.seg_id, segment_index))
        selected.update(
            (asked_k, segment_index)
            for _, _, segment_index in sorted(entropies, key=lambda row: (-row[0], row[1]))[
                :config.entropy_per_task
            ]
        )
    return selected


def _choose_exact(cloud, candidates, segments, config):
    best = (math.inf, -1, -1)
    for asked_k, segment_index in candidates:
        segment = segments[segment_index]
        loss = expected_loss(
            cloud, asked_k, np.asarray(segment.s_mean), np.asarray(segment.s_sd),
            IIIC_GROUP, config.beta, config.distractor_lapse,
        )
        best = min(best, (loss, asked_k, segment_index))
    return best


def _run_state(index: int, full_bank: list[BankSegment], config: RealRegretConfig) -> dict:
    rng = np.random.default_rng(config.seed_base + index + 500_000)
    sample_indices = np.sort(rng.choice(
        len(full_bank), size=min(config.bank_sample, len(full_bank)), replace=False,
    ))
    segments = [full_bank[int(position)] for position in sample_indices]
    cloud, history_length = _make_state(index, segments, config)
    baseline = _baseline_shortlist(cloud, segments, config)
    challenger = set(baseline)
    moments = posterior_moments(cloud)
    for asked_k in IIIC_GROUP:
        scored = sorted(
            (
                _fisher_utility(
                    cloud, asked_k,
                    np.asarray(segment.s_mean), np.asarray(segment.s_sd),
                    config, moments,
                ),
                segment.seg_id,
                segment_index,
            )
            for segment_index, segment in enumerate(segments)
        )
        challenger.update(
            (asked_k, segment_index)
            for _, _, segment_index in scored[-config.fisher_per_task:]
        )
    exhaustive = [
        (asked_k, segment_index)
        for asked_k in IIIC_GROUP
        for segment_index in range(len(segments))
    ]
    baseline_choice = _choose_exact(cloud, baseline, segments, config)
    challenger_choice = _choose_exact(cloud, challenger, segments, config)
    exact_choice = _choose_exact(cloud, exhaustive, segments, config)
    return {
        "state": index,
        "history_length": history_length,
        "candidate_count": len(exhaustive),
        "baseline_shortlist": len(baseline),
        "challenger_shortlist": len(challenger),
        "baseline_loss": baseline_choice[0],
        "challenger_loss": challenger_choice[0],
        "exact_loss": exact_choice[0],
        "baseline_regret": max(0.0, baseline_choice[0] - exact_choice[0]),
        "challenger_regret": max(0.0, challenger_choice[0] - exact_choice[0]),
        "baseline_exact": baseline_choice[1:] == exact_choice[1:],
        "challenger_exact": challenger_choice[1:] == exact_choice[1:],
    }


def summarize(rows: list[dict], config: RealRegretConfig) -> dict:
    return {
        "schema_version": 1,
        "status": "research_only_not_promoted",
        "bank_scope": "deterministic samples from the >=10-vote eligible empirical axis bank",
        "config": asdict(config),
        "states": len(rows),
        "mean_baseline_regret": float(np.mean([row["baseline_regret"] for row in rows])),
        "mean_challenger_regret": float(np.mean([row["challenger_regret"] for row in rows])),
        "p95_baseline_regret": float(np.quantile([row["baseline_regret"] for row in rows], 0.95)),
        "p95_challenger_regret": float(np.quantile([row["challenger_regret"] for row in rows], 0.95)),
        "baseline_exact_recall": float(np.mean([row["baseline_exact"] for row in rows])),
        "challenger_exact_recall": float(np.mean([row["challenger_exact"] for row in rows])),
        "mean_baseline_shortlist": float(np.mean([row["baseline_shortlist"] for row in rows])),
        "mean_challenger_shortlist": float(np.mean([row["challenger_shortlist"] for row in rows])),
        "rows": rows,
    }


def run(bank: list[BankSegment], config: RealRegretConfig, workers: int) -> dict:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(
            _run_state,
            range(config.states),
            [bank] * config.states,
            [config] * config.states,
        ))
    return summarize(rows, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--states", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = RealRegretConfig()
    if args.mode == "smoke":
        config = replace(
            config, states=2, particles=48, mh_steps=1, bank_sample=12,
            coarse_per_task=3, entropy_per_task=1, fisher_per_task=1,
        )
    if args.states is not None:
        config = replace(config, states=args.states)
    result = run(_read_bank(args.bank), config, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
