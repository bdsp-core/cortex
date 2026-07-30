"""Leakage-controlled conditional-distractor artifact research tooling.

The fitter consumes precomputed response-axis ``z`` values.  Reader/source
connected components are assigned wholly to one fold, preventing either a
reader or a source from leaking across train and validation partitions.
Outputs remain explicitly research-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .reference import f1_probabilities


@dataclass(frozen=True)
class DistractorRecord:
    reader_id: str
    source_id: str
    asked_k: int
    raw_pick: int
    group: tuple[int, ...]
    z: tuple[float, ...]


@dataclass(frozen=True)
class ArtifactDraw:
    beta: float
    distractor_lapse: float
    weight: float = 1.0


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def leakage_components(records: Sequence[DistractorRecord]) -> list[str]:
    union = _UnionFind()
    for record in records:
        union.union(f"reader:{record.reader_id}", f"source:{record.source_id}")
    return [union.find(f"reader:{record.reader_id}") for record in records]


def fold_assignments(
    records: Sequence[DistractorRecord], folds: int, salt: str,
    split_unit: str = "connected",
) -> np.ndarray:
    if folds < 2:
        raise ValueError("at least two folds are required")
    if split_unit == "connected":
        components = leakage_components(records)
    elif split_unit == "reader":
        components = [f"reader:{record.reader_id}" for record in records]
    elif split_unit == "source":
        components = [f"source:{record.source_id}" for record in records]
    else:
        raise ValueError("split_unit must be connected, reader, or source")
    unique = sorted(set(components))
    component_fold = {
        component: int.from_bytes(
            hashlib.sha256(f"{salt}:{component}".encode()).digest()[:8], "big",
        ) % folds
        for component in unique
    }
    return np.asarray([component_fold[component] for component in components], dtype=int)


def _conditional_log_probability(
    record: DistractorRecord, beta: float, lapse: float,
) -> float:
    if record.raw_pick == record.asked_k:
        raise ValueError("artifact fit accepts wrong-pick records only")
    distractors = tuple(k for k in record.group if k != record.asked_k)
    if record.raw_pick not in distractors:
        raise ValueError("raw pick is outside the distractor group")
    logits = beta * np.asarray([record.z[k] for k in distractors], dtype=float)
    log_softmax = logits - logsumexp(logits)
    position = distractors.index(record.raw_pick)
    probability = lapse / len(distractors) + (1 - lapse) * math.exp(log_softmax[position])
    return math.log(max(probability, np.finfo(float).tiny))


def _design_matrix(records: Sequence[DistractorRecord]) -> tuple[np.ndarray, np.ndarray]:
    z_rows = []
    positions = []
    for record in records:
        if record.raw_pick == record.asked_k:
            raise ValueError("artifact fit accepts wrong-pick records only")
        distractors = tuple(k for k in record.group if k != record.asked_k)
        if record.raw_pick not in distractors:
            raise ValueError("raw pick is outside the distractor group")
        z_rows.append([record.z[k] for k in distractors])
        positions.append(distractors.index(record.raw_pick))
    return np.asarray(z_rows, dtype=float), np.asarray(positions, dtype=int)


def _log_probabilities(
    design: np.ndarray, positions: np.ndarray, beta: float, lapse: float,
) -> np.ndarray:
    logits = beta * design
    softmax = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    selected = softmax[np.arange(positions.size), positions]
    probability = lapse / design.shape[1] + (1 - lapse) * selected
    return np.log(np.maximum(probability, np.finfo(float).tiny))


def fit_artifact(records: Sequence[DistractorRecord]) -> ArtifactDraw:
    if not records:
        raise ValueError("artifact fit requires records")

    design, positions = _design_matrix(records)

    def objective(values: np.ndarray) -> float:
        beta, lapse = float(values[0]), float(values[1])
        return -float(_log_probabilities(design, positions, beta, lapse).sum())

    fitted = minimize(
        objective,
        x0=np.asarray([1.0, 0.02]),
        method="L-BFGS-B",
        bounds=((0.05, 5.0), (0.0, 0.40)),
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"distractor fit failed: {fitted.message}")
    return ArtifactDraw(float(fitted.x[0]), float(fitted.x[1]))


def _mean_log_score(records: Sequence[DistractorRecord], draw: ArtifactDraw) -> float:
    design, positions = _design_matrix(records)
    return float(_log_probabilities(
        design, positions, draw.beta, draw.distractor_lapse,
    ).mean())


def crossfit_artifact(
    records: Sequence[DistractorRecord], folds: int = 5,
    salt: str = "nway-artifact-v1", split_unit: str = "connected",
) -> dict:
    assignments = fold_assignments(records, folds, salt, split_unit)
    fold_results = []
    for fold in range(folds):
        train = [record for index, record in enumerate(records) if assignments[index] != fold]
        validation = [record for index, record in enumerate(records) if assignments[index] == fold]
        if not train or not validation:
            continue
        draw = fit_artifact(train)
        fold_results.append({
            "fold": fold,
            "n_train": len(train),
            "n_validation": len(validation),
            "beta": draw.beta,
            "distractor_lapse": draw.distractor_lapse,
            "validation_mean_log_score": _mean_log_score(validation, draw),
        })
    if len(fold_results) < 2:
        raise ValueError("reader/source components do not support at least two populated folds")
    fitted = fit_artifact(records)
    return {
        "full_fit": asdict(fitted),
        "folds": fold_results,
        "crossfit_mean_log_score": float(np.average(
            [row["validation_mean_log_score"] for row in fold_results],
            weights=[row["n_validation"] for row in fold_results],
        )),
        "split": split_unit,
        "salt": salt,
    }


def dual_crossfit_artifact(
    records: Sequence[DistractorRecord], folds: int = 5,
    salt: str = "nway-artifact-v1",
) -> dict:
    """Run independent reader-held-out and source-held-out validation panels."""
    return {
        "reader_held_out": crossfit_artifact(
            records, folds, f"{salt}:reader", "reader",
        ),
        "source_held_out": crossfit_artifact(
            records, folds, f"{salt}:source", "source",
        ),
        "note": (
            "Panels are separate because densely crossed reader/source data can "
            "form one connected component and make simultaneous holdout impossible."
        ),
    }


def bootstrap_artifact(
    records: Sequence[DistractorRecord], draws: int, seed: int,
    cluster_by: str = "reader", workers: int = 1,
) -> list[ArtifactDraw]:
    if draws < 1:
        raise ValueError("at least one bootstrap draw is required")
    if cluster_by == "connected":
        components = leakage_components(records)
    elif cluster_by == "reader":
        components = [record.reader_id for record in records]
    elif cluster_by == "source":
        components = [record.source_id for record in records]
    else:
        raise ValueError("cluster_by must be connected, reader, or source")
    grouped: dict[str, list[DistractorRecord]] = {}
    for component, record in zip(components, records, strict=True):
        grouped.setdefault(component, []).append(record)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    sampled_keys = [rng.choice(keys, size=len(keys), replace=True) for _ in range(draws)]

    def fit_sample(sampled) -> ArtifactDraw:
        sample = [record for key in sampled for record in grouped[str(key)]]
        return fit_artifact(sample)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            output = list(executor.map(fit_sample, sampled_keys))
    else:
        output = [fit_sample(sampled) for sampled in sampled_keys]
    weight = 1 / len(output)
    return [ArtifactDraw(draw.beta, draw.distractor_lapse, weight) for draw in output]


def f1_probabilities_mixture(
    t: np.ndarray,
    l: np.ndarray,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    asked_k: int,
    group: tuple[int, ...],
    draws: Sequence[ArtifactDraw],
) -> np.ndarray:
    if not draws:
        raise ValueError("artifact mixture requires draws")
    weights = np.asarray([draw.weight for draw in draws], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("artifact mixture weights are invalid")
    weights /= weights.sum()
    output = None
    for weight, draw in zip(weights, draws, strict=True):
        probabilities = f1_probabilities(
            t, l, s_mean, s_sd, asked_k, group,
            draw.beta, draw.distractor_lapse,
        )
        if output is None:
            output = np.zeros_like(probabilities)
        output += weight * probabilities
    assert output is not None
    return output


def _read_csv(path: Path) -> list[DistractorRecord]:
    records = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            group = tuple(int(value) for value in row["group"].split("|"))
            z = tuple(float(row[f"z_{k}"]) for k in range(max(group) + 1))
            records.append(DistractorRecord(
                reader_id=row["reader_id"],
                source_id=row["source_id"],
                asked_k=int(row["asked_k"]),
                raw_pick=int(row["raw_pick"]),
                group=group,
                z=z,
            ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument(
        "--split-unit", choices=("dual", "connected", "reader", "source"),
        default="dual",
    )
    parser.add_argument(
        "--bootstrap-cluster", choices=("connected", "reader", "source"),
        default="reader",
    )
    parser.add_argument("--seed", type=int, default=63_300_001)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = _read_csv(args.input)
    crossfit = (
        dual_crossfit_artifact(records, args.folds)
        if args.split_unit == "dual"
        else crossfit_artifact(records, args.folds, split_unit=args.split_unit)
    )
    draws = bootstrap_artifact(
        records, args.bootstrap_draws, args.seed, args.bootstrap_cluster, args.workers,
    )
    payload = {
        "schemaVersion": 1,
        "status": "research_only_not_promoted",
        "model": "iiic_conditional_f1_v1_artifact_ensemble",
        "nWrongPicks": len(records),
        "crossfit": crossfit,
        "bootstrap": {
            "seed": args.seed,
            "workers": args.workers,
            "clusterBy": args.bootstrap_cluster,
            "draws": [asdict(draw) for draw in draws],
            "betaInterval95": list(np.quantile([draw.beta for draw in draws], [0.025, 0.975])),
            "distractorLapseInterval95": list(np.quantile(
                [draw.distractor_lapse for draw in draws], [0.025, 0.975],
            )),
        },
        "promotionForbidden": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
