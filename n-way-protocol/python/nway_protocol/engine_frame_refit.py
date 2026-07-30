"""Engine-frame per-reader distractor refit (label-free diagnostic).

Every deployed conditional-distractor artifact was fit in ITEM frame: the
staged ``z_k`` columns are raw ``s_mean`` axes with no reader term. The
engine, however, applies beta to the skill-scaled evidence
``z = e^l (s + t) / sqrt(1 + (e^l s_sd)^2)``, which already sharpens picks
for high-sensitivity particles. Item-frame tier fits therefore conflate
reader sensitivity with residual pick temperature.

This diagnostic decomposes the two, using no expertise labels anywhere in
the construction: per-reader sensitivity ``l_r`` is fit on binary
asked-class margins (beta-free), then beta is refit on wrong-picks by
reader-sensitivity decile in both frames. If the item-frame beta gradient
flattens in engine frame, the engine's own scaling explains reader
heterogeneity and no residual-temperature modeling is needed. Tier pools
are refit for reporting comparability with the C1 stratification only.
Research-only diagnostic; no artifact output.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr

from .artifact_rd import (
    DistractorRecord, _design_matrix, _log_probabilities,
)
from .reference import BINARY_LAPSE
from .stage_real_artifact import CLASSES, CLASS_INDEX, _plurality, _sha256, _to_class


@dataclass
class ReadRecord:
    reader_id: str
    source_id: str
    asked_k: int
    pick_k: int
    s_mean: tuple
    s_sd: tuple


def stage_reads(
    signals_path: Path,
    case_map_path: Path,
    expert_path: Path,
    novice_path: Path,
) -> list[ReadRecord]:
    signals = pd.read_csv(signals_path)
    mean_columns = [f"s_mean_{name}" for name in CLASSES]
    sd_columns = [f"s_sd_{name}" for name in CLASSES]
    signals = signals.dropna(subset=mean_columns + sd_columns)
    mean_lookup = {}
    sd_lookup = {}
    for row in signals.itertuples():
        mean_lookup[int(row.seg_id)] = tuple(
            [0.0] + [float(getattr(row, column)) for column in mean_columns]
        )
        sd_lookup[int(row.seg_id)] = tuple(
            [1.0] + [float(getattr(row, column)) for column in sd_columns]
        )
    case_map = pd.read_csv(case_map_path)
    segment_of = dict(zip(case_map["extset_case_id"], case_map["new_seg_id"], strict=True))

    expert = pd.read_excel(expert_path)
    expert_picks: dict[int, list[str | None]] = {}
    for row in expert.itertuples(index=False):
        expert_picks[int(row.case_id)] = [
            _to_class(getattr(row, f"label_expert{index}")) for index in (1, 2, 3, 4)
        ]

    reads: list[ReadRecord] = []

    def append(reader_id: str, case_id: int, gold: str | None, pick: str | None) -> None:
        segment_id = segment_of.get(case_id)
        if gold is None or pick is None or segment_id is None:
            return
        segment_id = int(segment_id)
        if segment_id not in mean_lookup:
            return
        reads.append(ReadRecord(
            reader_id=reader_id,
            source_id=f"case-{case_id}",
            asked_k=CLASS_INDEX[gold],
            pick_k=CLASS_INDEX[pick],
            s_mean=mean_lookup[segment_id],
            s_sd=sd_lookup[segment_id],
        ))

    # Expert reads keep the canonical non-circular rule: each read is judged
    # against the other three experts' plurality only.
    for case_id, picks in expert_picks.items():
        for expert_index, pick in enumerate(picks, start=1):
            gold = _plurality([
                value for index, value in enumerate(picks, start=1)
                if index != expert_index
            ])
            append(f"expert-{expert_index}", case_id, gold, pick)

    gold_of = {case_id: _plurality(picks) for case_id, picks in expert_picks.items()}
    novice = pd.read_csv(novice_path)
    novice = novice[novice["Labeling State"] == "Gold Standard"]
    for row in novice.itertuples(index=False):
        case_id = int(getattr(row, "_0"))
        append(
            f"novice-{getattr(row, '_1')}", case_id,
            gold_of.get(case_id), _to_class(getattr(row, "_5")),
        )
    return reads


def engine_z(l: float, s_mean: np.ndarray, s_sd: np.ndarray) -> np.ndarray:
    """Skill-scaled evidence with the reader's sensitivity and zero bias.

    Bias t is held at zero so the engine frame differs from the item frame by
    the sensitivity scaling alone; the decile contrast is unaffected by a
    shared additive shift.
    """
    sensitivity = np.exp(l)
    return sensitivity * s_mean / np.sqrt(1.0 + np.square(sensitivity * s_sd))


def binary_log_likelihood(z_asked: np.ndarray, correct: np.ndarray) -> np.ndarray:
    signed = np.where(correct, z_asked, -z_asked)
    return np.logaddexp(
        np.log1p(-2.0 * BINARY_LAPSE) + log_ndtr(signed),
        np.log(BINARY_LAPSE),
    )


def fit_reader_sensitivity(
    z_unit_asked: np.ndarray, s_sd_asked: np.ndarray, correct: np.ndarray,
) -> float:
    """MAP scalar l with a standard-normal prior (the engine's session prior)."""

    def objective(l: float) -> float:
        sensitivity = np.exp(l)
        z = sensitivity * z_unit_asked / np.sqrt(1.0 + np.square(sensitivity * s_sd_asked))
        return -float(binary_log_likelihood(z, correct).sum()) + 0.5 * l * l

    fitted = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(fitted.x)


def frame_records(reads: list[ReadRecord], l_of: dict[str, float], frame: str) -> list[DistractorRecord]:
    records = []
    for read in reads:
        if read.pick_k == read.asked_k:
            continue
        s_mean = np.asarray(read.s_mean)
        if frame == "item":
            z = tuple(s_mean)
        else:
            z = tuple(engine_z(l_of[read.reader_id], s_mean, np.asarray(read.s_sd)))
        records.append(DistractorRecord(
            reader_id=read.reader_id,
            source_id=read.source_id,
            asked_k=read.asked_k,
            raw_pick=read.pick_k,
            group=tuple(range(1, 7)),
            z=z,
        ))
    return records


def fit_draw(records: list[DistractorRecord]) -> dict:
    """The canonical conditional fit with diagnostic-wide beta bounds.

    ``fit_artifact`` caps beta at 5.0, which censors engine-frame fits for
    low-sensitivity readers whose shrunken z rescales beta upward.
    """
    design, positions = _design_matrix(records)

    def objective(values: np.ndarray) -> float:
        return -float(_log_probabilities(
            design, positions, float(values[0]), float(values[1]),
        ).sum())

    fitted = minimize(
        objective,
        x0=np.asarray([1.0, 0.02]),
        method="L-BFGS-B",
        bounds=((0.01, 100.0), (0.0, 0.40)),
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"diagnostic distractor fit failed: {fitted.message}")
    return {"beta": float(fitted.x[0]), "distractor_lapse": float(fitted.x[1])}


def fit_both_frames(reads: list[ReadRecord], l_of: dict[str, float]) -> dict:
    return {
        frame: fit_draw(frame_records(reads, l_of, frame))
        for frame in ("item", "engine")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--novice", type=Path, required=True)
    parser.add_argument("--min-reads", type=int, default=30)
    parser.add_argument("--min-wrong", type=int, default=10)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()

    reads = stage_reads(args.signals, args.case_map, args.expert, args.novice)
    n_wrong_total = sum(1 for read in reads if read.pick_k != read.asked_k)

    by_reader: dict[str, list[ReadRecord]] = {}
    for read in reads:
        by_reader.setdefault(read.reader_id, []).append(read)

    l_of: dict[str, float] = {}
    for reader_id, rows in by_reader.items():
        z_unit = np.asarray([row.s_mean[row.asked_k] for row in rows])
        s_sd = np.asarray([row.s_sd[row.asked_k] for row in rows])
        correct = np.asarray([row.pick_k == row.asked_k for row in rows])
        l_of[reader_id] = fit_reader_sensitivity(z_unit, s_sd, correct)

    eligible = [
        reader_id for reader_id, rows in by_reader.items()
        if len(rows) >= args.min_reads
        and sum(1 for row in rows if row.pick_k != row.asked_k) >= args.min_wrong
    ]
    eligible.sort(key=lambda reader_id: l_of[reader_id])
    bins = np.array_split(np.asarray(eligible), args.deciles)

    decile_rows = []
    for index, bin_readers in enumerate(bins):
        bin_reads = [read for reader_id in bin_readers for read in by_reader[reader_id]]
        wrong = sum(1 for read in bin_reads if read.pick_k != read.asked_k)
        correct = len(bin_reads) - wrong
        fits = fit_both_frames(bin_reads, l_of)
        decile_rows.append({
            "decile": index + 1,
            "n_readers": int(len(bin_readers)),
            "n_reads": len(bin_reads),
            "n_wrong_picks": wrong,
            "accuracy": correct / len(bin_reads),
            "mean_l": float(np.mean([l_of[reader_id] for reader_id in bin_readers])),
            **{f"{frame}_frame": fits[frame] for frame in ("item", "engine")},
        })

    # Reporting-only tier pools for comparability with the C1 stratification;
    # labels play no role in the decile construction above.
    pools = {}
    for pool, prefix in (("expert_pool", "expert-"), ("novice_pool", "novice-")):
        pool_reads = [read for read in reads if read.reader_id.startswith(prefix)]
        pools[pool] = {
            "n_reads": len(pool_reads),
            "n_wrong_picks": sum(1 for read in pool_reads if read.pick_k != read.asked_k),
            "mean_l": float(np.mean([
                l_of[reader_id] for reader_id in by_reader if reader_id.startswith(prefix)
            ])),
            **fit_both_frames(pool_reads, l_of),
        }
    pools["all_readers_pooled"] = fit_both_frames(reads, l_of)

    item_betas = [row["item_frame"]["beta"] for row in decile_rows]
    engine_betas = [row["engine_frame"]["beta"] for row in decile_rows]
    summary = {
        "item_frame_beta_range": [min(item_betas), max(item_betas)],
        "engine_frame_beta_range": [min(engine_betas), max(engine_betas)],
        "item_frame_top_over_bottom": item_betas[-1] / item_betas[0],
        "engine_frame_top_over_bottom": engine_betas[-1] / engine_betas[0],
        "gradient_compression": (
            (item_betas[-1] / item_betas[0]) / (engine_betas[-1] / engine_betas[0])
        ),
    }

    payload = {
        "schema_version": 1,
        "status": "research_only_diagnostic",
        "design": "label_free_reader_deciles_by_map_sensitivity",
        "n_reads": len(reads),
        "n_wrong_picks": n_wrong_total,
        "n_readers": len(by_reader),
        "n_eligible_readers": len(eligible),
        "eligibility": {"min_reads": args.min_reads, "min_wrong": args.min_wrong},
        "deciles": decile_rows,
        "pools": pools,
        "summary": summary,
        "runtime_seconds": round(time.monotonic() - started, 1),
        "provenance": {
            "source_sha256": {
                str(path): _sha256(path)
                for path in (args.signals, args.case_map, args.expert, args.novice)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "pools": pools}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
