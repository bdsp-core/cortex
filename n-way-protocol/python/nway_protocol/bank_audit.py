"""Information-oriented audit for candidate n-way banks.

This tool never edits or prunes a bank.  It reports F1 Fisher information and
signal-uncertainty sensitivity so a separately governed curation process can
freeze a candidate bank using training data only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .qualification import IIIC_GROUP
from .reference import f1_probabilities


@dataclass(frozen=True)
class BankSegment:
    segment_index: int
    seg_id: int
    s_mean: tuple[float, ...]
    s_sd: tuple[float, ...]


def candidate_information(
    segment: BankSegment,
    asked_k: int,
    t: np.ndarray,
    l: np.ndarray,
    beta: float,
    distractor_lapse: float,
    finite_difference: float = 1e-4,
) -> dict[str, float]:
    t = np.asarray(t, dtype=float).copy()
    l = np.asarray(l, dtype=float).copy()
    s_mean = np.asarray(segment.s_mean, dtype=float)
    s_sd = np.asarray(segment.s_sd, dtype=float)
    base = f1_probabilities(
        t, l, s_mean, s_sd, asked_k, IIIC_GROUP, beta, distractor_lapse,
    )[0]
    diagonal: dict[str, np.ndarray] = {}
    for parameter, values in (("bias", t), ("skill", l)):
        information = np.zeros(values.shape[1])
        for k in IIIC_GROUP:
            original = values[0, k]
            values[0, k] = original + finite_difference
            plus = f1_probabilities(
                t, l, s_mean, s_sd, asked_k, IIIC_GROUP,
                beta, distractor_lapse,
            )[0]
            values[0, k] = original - finite_difference
            minus = f1_probabilities(
                t, l, s_mean, s_sd, asked_k, IIIC_GROUP,
                beta, distractor_lapse,
            )[0]
            values[0, k] = original
            derivative = (plus - minus) / (2 * finite_difference)
            information[k] = float(np.sum(np.square(derivative) / np.maximum(base, 1e-12)))
        diagonal[parameter] = information
    return {
        "skill_information": float(diagonal["skill"][list(IIIC_GROUP)].sum()),
        "bias_information": float(diagonal["bias"][list(IIIC_GROUP)].sum()),
        "cross_skill_information": float(sum(
            diagonal["skill"][k] for k in IIIC_GROUP if k != asked_k
        )),
        "cross_bias_information": float(sum(
            diagonal["bias"][k] for k in IIIC_GROUP if k != asked_k
        )),
        "focal_skill_information": float(diagonal["skill"][asked_k]),
        "focal_bias_information": float(diagonal["bias"][asked_k]),
    }


def audit_bank(
    segments: Sequence[BankSegment],
    beta: float = 0.9912,
    distractor_lapse: float = 0.0,
    skill_profiles: tuple[float, ...] = (-1.0, 0.0, 1.0),
    uncertainty_scale: float = 0.75,
    top_per_task: int = 20,
    workers: int = 1,
    include_rows: bool = True,
) -> dict:
    arguments = [
        (segment, beta, distractor_lapse, skill_profiles, uncertainty_scale)
        for segment in segments
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested = list(executor.map(_audit_segment, arguments, chunksize=16))
    else:
        nested = [_audit_segment(argument) for argument in arguments]
    rows = [row for segment_rows in nested for row in segment_rows]
    top = {}
    for asked_k in IIIC_GROUP:
        candidates = sorted(
            (row for row in rows if row["asked_k"] == asked_k),
            key=lambda row: (-row["total_information"], row["seg_id"]),
        )
        top[str(asked_k)] = [
            {
                "seg_id": row["seg_id"],
                "total_information": row["total_information"],
                "skill_information": row["skill_information"],
                "bias_information": row["bias_information"],
                "mean_signal_sd": row["mean_signal_sd"],
            }
            for row in candidates[:top_per_task]
        ]
    output = {
        "schema_version": 1,
        "status": "audit_only_no_bank_mutation",
        "config": {
            "beta": beta,
            "distractor_lapse": distractor_lapse,
            "skill_profiles": list(skill_profiles),
            "uncertainty_scale": uncertainty_scale,
            "top_per_task": top_per_task,
            "workers": workers,
        },
        "segments": len(segments),
        "candidate_rows": len(rows),
        "mean_skill_information": float(np.mean([row["skill_information"] for row in rows])),
        "mean_bias_information": float(np.mean([row["bias_information"] for row in rows])),
        "information_quantiles": {
            key: {
                str(q): float(np.quantile([row[key] for row in rows], q))
                for q in (0.1, 0.5, 0.9, 0.99)
            }
            for key in (
                "skill_information", "bias_information", "total_information",
                "reduced_uncertainty_information_ratio",
            )
        },
        "median_reduced_uncertainty_information_ratio": float(np.median([
            row["reduced_uncertainty_information_ratio"] for row in rows
        ])),
        "top_by_asked_task": top,
        "promotion_note": (
            "Information rank is not an authorization to prune. Content, exposure, "
            "source, and held-out coverage gates remain mandatory."
        ),
    }
    if include_rows:
        output["rows"] = rows
    return output


def _audit_segment(arguments) -> list[dict]:
    segment, beta, distractor_lapse, skill_profiles, uncertainty_scale = arguments
    rows = []
    for asked_k in IIIC_GROUP:
        profile_metrics = []
        reduced_metrics = []
        reduced_segment = BankSegment(
            segment.segment_index,
            segment.seg_id,
            segment.s_mean,
            tuple(value * uncertainty_scale for value in segment.s_sd),
        )
        for skill in skill_profiles:
            t = np.zeros((1, 7))
            l = np.zeros((1, 7))
            l[:, 1:] = skill
            profile_metrics.append(candidate_information(
                segment, asked_k, t, l, beta, distractor_lapse,
            ))
            reduced_metrics.append(candidate_information(
                reduced_segment, asked_k, t, l, beta, distractor_lapse,
            ))
        metrics = {
            key: float(np.mean([row[key] for row in profile_metrics]))
            for key in profile_metrics[0]
        }
        reduced_total = float(np.mean([
            row["skill_information"] + row["bias_information"]
            for row in reduced_metrics
        ]))
        total = metrics["skill_information"] + metrics["bias_information"]
        rows.append({
            "segment_index": segment.segment_index,
            "seg_id": segment.seg_id,
            "asked_k": asked_k,
            **metrics,
            "mean_signal_sd": float(np.mean(np.asarray(segment.s_sd)[list(IIIC_GROUP)])),
            "total_information": total,
            "reduced_uncertainty_information_ratio": reduced_total / max(total, 1e-12),
        })
    return rows


def _read_bank(path: Path) -> list[BankSegment]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(BankSegment(
                segment_index=int(row["segment_index"]),
                seg_id=int(row["seg_id"]),
                s_mean=tuple(float(row[f"s_mean_{k}"]) for k in range(7)),
                s_sd=tuple(float(row[f"s_sd_{k}"]) for k in range(7)),
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-per-task", type=int, default=20)
    parser.add_argument("--uncertainty-scale", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--omit-rows", action="store_true")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--sample-seed", type=int, default=63_600_001)
    args = parser.parse_args()
    segments = _read_bank(args.bank)
    if args.max_segments is not None and len(segments) > args.max_segments:
        rng = np.random.default_rng(args.sample_seed)
        indices = np.sort(rng.choice(
            len(segments), size=args.max_segments, replace=False,
        ))
        segments = [segments[int(index)] for index in indices]
    result = audit_bank(
        segments,
        top_per_task=args.top_per_task,
        uncertainty_scale=args.uncertainty_scale,
        workers=args.workers,
        include_rows=not args.omit_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
