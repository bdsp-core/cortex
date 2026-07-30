"""Fixed-sequence historical shadow replay on real responses (run family 4).

Replays each real reader's staged response sequence through the reference
engine at the production profile and scores one-step-ahead predictions —
the only real-data model check in the qualification program. Three arms
share the identical asked-class margin model:

- ``binary``: binary-reduced observations (the incumbent).
- ``categorical_deployed``: F1 with the deployed floored ensemble.
- ``categorical_engine_frame``: F1 with the engine-frame hierarchical
  ensemble, rebuilt leave-one-reader-out from the sensitivity panel so the
  replayed reader never informs their own population draws.

The decisive statistic is the conditional-on-wrong log score against the
uniform-distractor baseline ``log(1/5)``: if the fitted conditional does
not beat uniform out-of-reader on sequential prediction, the categorical
channel carries no real information and the program's kill criterion
fires. The asked-class margin score is reported per arm as a guard that
the categorical model does not degrade the incumbent margin.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .artifact_engine_frame import population_draws, random_effects_log_beta
from .artifact_floor import floor_draws
from .engine_frame_refit import ReadRecord, stage_reads
from .qualification import IIIC_GROUP, recommended_workers
from .reference import (
    Observation,
    ess,
    f1_probabilities_artifact,
    binary_p_yes,
    make_cloud,
    resample_and_rejuvenate,
    signal_z,
    update,
)
from .stage_real_artifact import _sha256

PROFILE = {"particles": 1_200, "mh_steps": 30, "ess_fraction": 0.5}
UNIFORM_CONDITIONAL = float(np.log(1.0 / 5.0))


def loo_effects(fits: list[dict], exclude_reader: str) -> dict:
    return random_effects_log_beta([
        fit for fit in fits if fit["reader_id"] != exclude_reader
    ])


def _predictive(cloud, observation, s_mean, s_sd, draws) -> dict:
    """One-step-ahead probabilities before the cloud sees the response."""
    if observation.kind == "binary":
        z = signal_z(
            cloud.l[:, observation.asked_k], cloud.t[:, observation.asked_k],
            np.asarray(s_mean)[observation.asked_k],
            np.asarray(s_sd)[observation.asked_k],
        )
        p_yes = float(np.sum(cloud.w * binary_p_yes(z)))
        p_response = p_yes if observation.y == 1 else 1.0 - p_yes
        return {"margin": float(np.log(max(p_response, 1e-300))), "conditional": None}
    probabilities = f1_probabilities_artifact(
        cloud.t, cloud.l, np.asarray(s_mean), np.asarray(s_sd),
        observation.asked_k, observation.group, 0.0, 0.0, draws,
    )
    marginal = np.sum(cloud.w[:, None] * probabilities, axis=0)
    pick_position = observation.group.index(observation.raw_pick)
    asked_position = observation.group.index(observation.asked_k)
    p_pick = float(marginal[pick_position])
    p_correct = float(marginal[asked_position])
    correct = observation.raw_pick == observation.asked_k
    margin = np.log(max(p_correct if correct else 1.0 - p_correct, 1e-300))
    conditional = None
    if not correct:
        conditional = float(np.log(max(p_pick, 1e-300)) - np.log(max(1.0 - p_correct, 1e-300)))
    return {"margin": float(margin), "conditional": conditional}


def replay_reader(payload: tuple) -> dict:
    reader_id, rows, arm_draws, seed, max_reads = payload
    rows = rows[:max_reads]
    s_means = np.asarray([row.s_mean for row in rows])
    s_sds = np.asarray([row.s_sd for row in rows])
    prior_corr = np.eye(7)
    result = {"reader_id": reader_id, "n_reads": len(rows)}
    for arm_index, (arm, draws) in enumerate(arm_draws.items()):
        cloud = make_cloud(
            PROFILE["particles"], prior_corr, prior_corr,
            np.random.default_rng(seed + 10_000 * (arm_index + 1)),
        )
        mh_rng = np.random.default_rng(seed + 50_000 * (arm_index + 1))
        margin_scores, conditional_scores, resamples = [], [], 0
        for index, row in enumerate(rows):
            observation = Observation(
                kind="binary" if arm == "binary" else "categorical_f1",
                asked_k=row.asked_k,
                segment_index=index,
                raw_pick=row.pick_k,
                y=int(row.pick_k == row.asked_k) if arm == "binary" else None,
                group=IIIC_GROUP if arm != "binary" else (),
            )
            scores = _predictive(cloud, observation, s_means[index], s_sds[index], draws)
            margin_scores.append(scores["margin"])
            if scores["conditional"] is not None:
                conditional_scores.append(scores["conditional"])
            update(cloud, observation, s_means[index], s_sds[index], 0.0, 0.0, draws)
            if ess(cloud) < PROFILE["ess_fraction"] * PROFILE["particles"]:
                resample_and_rejuvenate(
                    cloud, s_means, s_sds, 0.0, 0.0, mh_rng,
                    PROFILE["mh_steps"], 2.38 / np.sqrt(14), artifact_draws=draws,
                )
                resamples += 1
        result[arm] = {
            "mean_margin_log_score": float(np.mean(margin_scores)),
            "n_wrong_picks": len(conditional_scores),
            "mean_conditional_log_score": (
                float(np.mean(conditional_scores)) if conditional_scores else None
            ),
            "resamples": resamples,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--novice", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--deployed-artifact", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.15)
    parser.add_argument("--max-reads", type=int, default=500)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=64_100_001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    panel = json.loads(args.panel.read_text())["panel"]["stable_core"]
    fits = panel["reader_beta_fits"]
    core_readers = {fit["reader_id"] for fit in fits}
    lapse = panel["pooled"]["distractor_lapse"]

    deployed = json.loads(args.deployed_artifact.read_text())
    deployed_rows = (
        deployed["draws"] if "draws" in deployed else deployed["bootstrap"]["draws"]
    )
    deployed_draws = tuple(
        (
            float(draw["beta"]),
            float(draw.get("distractor_lapse", draw.get("distractorLapse"))),
            float(draw["weight"]),
        )
        for draw in deployed_rows
    )

    reads = stage_reads(args.signals, args.case_map, args.expert, args.novice)
    by_reader: dict[str, list[ReadRecord]] = {}
    for read in reads:
        if read.reader_id in core_readers:
            by_reader.setdefault(read.reader_id, []).append(read)

    jobs = []
    for index, (reader_id, rows) in enumerate(sorted(by_reader.items())):
        effects = loo_effects(fits, reader_id)
        loo_draws = floor_draws(
            population_draws(effects, lapse, 9), args.floor,
        )
        arm_draws = {
            "binary": (),
            "categorical_deployed": deployed_draws,
            "categorical_engine_frame": tuple(
                (draw["beta"], draw["distractor_lapse"], draw["weight"])
                for draw in loo_draws
            ),
        }
        jobs.append((reader_id, rows, arm_draws, args.seed + index, args.max_reads))

    workers = args.workers or recommended_workers(len(jobs), PROFILE["particles"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        per_reader = list(executor.map(replay_reader, jobs))

    def pooled(arm: str, key: str) -> dict:
        if key == "mean_conditional_log_score":
            weights = np.asarray(
                [reader[arm]["n_wrong_picks"] for reader in per_reader], dtype=float,
            )
        else:
            weights = np.asarray(
                [reader["n_reads"] for reader in per_reader], dtype=float,
            )
        values = np.asarray([
            np.nan if reader[arm][key] is None else reader[arm][key]
            for reader in per_reader
        ])
        mask = np.isfinite(values) & (weights > 0)
        mean = float(np.sum(weights[mask] * values[mask]) / np.sum(weights[mask]))
        return {"pooled_mean": mean, "readers": int(mask.sum())}

    arms = ("binary", "categorical_deployed", "categorical_engine_frame")
    summary = {
        arm: {
            "margin_log_score": pooled(arm, "mean_margin_log_score"),
            **(
                {"conditional_log_score": pooled(arm, "mean_conditional_log_score")}
                if arm != "binary" else {}
            ),
        }
        for arm in arms
    }
    for arm in ("categorical_deployed", "categorical_engine_frame"):
        gained = summary[arm]["conditional_log_score"]["pooled_mean"] - UNIFORM_CONDITIONAL
        summary[arm]["conditional_bits_over_uniform"] = float(gained / np.log(2.0))
        summary[arm]["beats_uniform_baseline"] = bool(gained > 0)

    payload = {
        "schema_version": 1,
        "status": "research_only_shadow_replay",
        "design": "fixed_staged_order;loo_population_draws;production_profile",
        "profile": PROFILE,
        "max_reads_per_reader": args.max_reads,
        "n_readers": len(per_reader),
        "n_reads_replayed": int(sum(reader["n_reads"] for reader in per_reader)),
        "uniform_conditional_baseline": UNIFORM_CONDITIONAL,
        "summary": summary,
        "per_reader": per_reader,
        "runtime_seconds": round(time.monotonic() - started, 1),
        "provenance": {
            "panel": str(args.panel),
            "deployed_artifact": str(args.deployed_artifact),
            "source_sha256": {
                str(path): _sha256(path)
                for path in (args.signals, args.case_map, args.expert, args.novice)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
