"""Distractor-misspecification monitor: statistic, CUSUM, and OC harness.

Design: docs/MISSPEC_MONITOR_DESIGN.md.  The statistic is fixed-input by
construction — it depends only on (asked class, pick, frozen segment response
axes, frozen artifact draws), never on the session posterior or an RNG.  The
response axes are the raw ``s_mean`` values, matching the frame the artifact
was fitted in (stage_real_artifact.py).  Research-only; the TypeScript engine
port is a separately authorized production change.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

IIIC_GROUP = (1, 2, 3, 4, 5, 6)
UNIFORM_LOG_PROBABILITY = -np.log(len(IIIC_GROUP) - 1)


def mixture_distractor_probabilities(
    asked_k: int,
    z: np.ndarray,
    draws: tuple[tuple[float, float, float], ...],
) -> np.ndarray:
    """Model probability over the five distractors, deployed-mixture predictive."""
    if asked_k not in IIIC_GROUP:
        raise ValueError("asked class must be an IIIC class")
    if not draws:
        raise ValueError("the monitor requires at least one artifact draw")
    distractors = [k for k in IIIC_GROUP if k != asked_k]
    logits = np.asarray([z[k] for k in distractors], dtype=float)
    weights = np.asarray([draw[2] for draw in draws], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("artifact mixture weights are invalid")
    weights = weights / weights.sum()
    output = np.zeros(len(distractors))
    for weight, (beta, lapse, _) in zip(weights, draws, strict=True):
        scaled = beta * logits
        softmax = np.exp(scaled - logsumexp(scaled))
        output += weight * (lapse / len(distractors) + (1 - lapse) * softmax)
    return output


def increment(
    asked_k: int,
    pick_k: int,
    z: np.ndarray,
    draws: tuple[tuple[float, float, float], ...],
) -> float:
    """Per-wrong-pick evidence for the model against the uniform alternative."""
    if pick_k == asked_k or pick_k not in IIIC_GROUP:
        raise ValueError("the monitor consumes wrong picks only")
    distractors = [k for k in IIIC_GROUP if k != asked_k]
    probabilities = mixture_distractor_probabilities(asked_k, z, draws)
    probability = probabilities[distractors.index(pick_k)]
    return float(np.log(max(probability, np.finfo(float).tiny)) - UNIFORM_LOG_PROBABILITY)


@dataclass
class CusumMonitor:
    """Page's CUSUM against the model; trips one-way when evidence for the
    uniform alternative exceeds the threshold."""

    threshold: float
    statistic: float = 0.0
    tripped: bool = False
    wrong_picks: int = 0
    tripped_at: int | None = None

    def observe(self, evidence: float) -> bool:
        if self.tripped:
            return True
        self.wrong_picks += 1
        self.statistic = max(0.0, self.statistic - evidence)
        if self.statistic > self.threshold:
            self.tripped = True
            self.tripped_at = self.wrong_picks
        return self.tripped


def _simulate_max_cusum(
    axes: np.ndarray,
    draws: tuple[tuple[float, float, float], ...],
    truth_beta_scale: float,
    truth_lapse: float,
    sessions: int,
    wrong_picks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-session max-CUSUM paths and the increments that formed them.

    The true world allocates each wrong pick with probability
    ``truth_lapse/5 + (1 - truth_lapse) * softmax(truth_beta_scale * beta_d * z)``
    mixed over the deployed draws, which nests the fitted world
    (scale 1, lapse 0 boundary), partial drift, and the N3b uniform world
    (lapse 1).
    """
    rng = np.random.default_rng(seed)
    segment_indices = rng.integers(0, axes.shape[0], size=(sessions, wrong_picks))
    asked = rng.integers(1, 7, size=(sessions, wrong_picks))
    weights = np.asarray([draw[2] for draw in draws], dtype=float)
    weights = weights / weights.sum()

    increments = np.zeros((sessions, wrong_picks))
    for session in range(sessions):
        for index in range(wrong_picks):
            z = axes[segment_indices[session, index]]
            asked_k = int(asked[session, index])
            distractors = [k for k in IIIC_GROUP if k != asked_k]
            logits = np.asarray([z[k] for k in distractors])
            truth = np.zeros(len(distractors))
            for weight, (beta, _, _) in zip(weights, draws, strict=True):
                scaled = truth_beta_scale * beta * logits
                softmax = np.exp(scaled - logsumexp(scaled))
                truth += weight * (
                    truth_lapse / len(distractors) + (1 - truth_lapse) * softmax
                )
            pick = int(rng.choice(distractors, p=truth / truth.sum()))
            increments[session, index] = increment(asked_k, pick, z, draws)

    paths = np.zeros_like(increments)
    running = np.zeros(sessions)
    for index in range(wrong_picks):
        running = np.maximum(0.0, running - increments[:, index])
        paths[:, index] = running
    return paths, increments


def operating_characteristics(
    axes: np.ndarray,
    draws: tuple[tuple[float, float, float], ...],
    sessions: int = 2000,
    wrong_picks: int = 90,
    false_trip_target: float = 0.01,
    seed: int = 63_800_001,
    alternatives: tuple[tuple[str, float, float], ...] = (
        ("uniform_n3b", 1.0, 1.0),
        ("half_drift_lapse", 1.0, 0.5),
        ("half_drift_beta", 0.5, 0.0),
    ),
) -> dict:
    null_paths, null_increments = _simulate_max_cusum(
        axes, draws, 1.0, 0.0, sessions, wrong_picks, seed,
    )
    max_excursion = null_paths.max(axis=1)
    threshold = float(np.quantile(max_excursion, 1.0 - false_trip_target))
    false_trips = float(np.mean(max_excursion > threshold))

    rows = {}
    for name, beta_scale, lapse in alternatives:
        paths, _ = _simulate_max_cusum(
            axes, draws, beta_scale, lapse, sessions, wrong_picks, seed + 17,
        )
        tripped = paths > threshold
        any_trip = tripped.any(axis=1)
        first = np.where(
            any_trip, tripped.argmax(axis=1) + 1, wrong_picks + 1,
        ).astype(float)
        detected = first[any_trip]
        rows[name] = {
            "truth_beta_scale": beta_scale,
            "truth_lapse": lapse,
            "trip_rate": float(np.mean(any_trip)),
            "median_wrong_picks_to_trip": (
                float(np.median(detected)) if detected.size else None
            ),
            "p90_wrong_picks_to_trip": (
                float(np.quantile(detected, 0.9)) if detected.size else None
            ),
        }
    return {
        "schema_version": 1,
        "status": "research_only_reference_monitor",
        "config": {
            "sessions": sessions,
            "wrong_picks_per_session": wrong_picks,
            "false_trip_target": false_trip_target,
            "seed": seed,
            "n_axes": int(axes.shape[0]),
            "draws": [list(draw) for draw in draws],
        },
        "threshold": threshold,
        "null_false_trip_rate": false_trips,
        "null_mean_increment": float(null_increments.mean()),
        "alternatives": rows,
    }


def load_draws(path: Path) -> tuple[tuple[float, float, float], ...]:
    payload = json.loads(path.read_text())
    draws = payload["draws"] if "draws" in payload else payload["bootstrap"]["draws"]
    return tuple(
        (
            float(draw["beta"]),
            float(draw.get("distractorLapse", draw.get("distractor_lapse"))),
            float(draw["weight"]),
        )
        for draw in draws
    )


def load_axes(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append([float(row[f"s_mean_{k}"]) for k in range(7)])
    return np.asarray(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--axes", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--wrong-picks", type=int, default=90)
    parser.add_argument("--false-trip-target", type=float, default=0.01)
    parser.add_argument("--max-axes", type=int, default=4000)
    parser.add_argument("--sample-seed", type=int, default=63_800_101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draws = load_draws(args.artifact)
    axes = load_axes(args.axes)
    if axes.shape[0] > args.max_axes:
        rng = np.random.default_rng(args.sample_seed)
        axes = axes[np.sort(rng.choice(axes.shape[0], args.max_axes, replace=False))]
    result = operating_characteristics(
        axes,
        draws,
        sessions=args.sessions,
        wrong_picks=args.wrong_picks,
        false_trip_target=args.false_trip_target,
    )
    result["config"]["artifact"] = args.artifact.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "threshold": round(result["threshold"], 3),
        "null_false_trip_rate": result["null_false_trip_rate"],
        "alternatives": {
            name: {
                "trip_rate": row["trip_rate"],
                "median_wrong_picks_to_trip": row["median_wrong_picks_to_trip"],
            }
            for name, row in result["alternatives"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
