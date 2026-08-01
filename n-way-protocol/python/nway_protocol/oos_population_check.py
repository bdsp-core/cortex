"""Out-of-sample check of the engine-frame reader-population model.

``artifacts/iiic_conditional_f1_engine_frame_rd.json`` carries a
DerSimonian-Laird population model for engine-frame distractor sharpness
(``mu_log_beta``, ``tau_log_beta``) fit on the CENTAUR gold-panel corpus.
That fit is in-sample by construction: the same 124 readers supplied both
the per-reader engine models and the population summary. This module asks
whether an INDEPENDENT reader population reproduces the same distribution.

The out-of-sample pool is the categorical (``pattern_class``) read stream
from ``data/labels/labels.csv`` restricted to the kong2025 crowdsourcing
and sparcnet50K source datasets — every CENTAUR corpus the artifact was
fit on is excluded, so no reader and no segment read event is shared with
the artifact's fit.

Construction mirrors the artifact exactly and reuses its fitters
(``artifact_engine_frame``); only the staging differs, because this corpus
has no case map and no held-out panel file:

1.  Gold per segment is the plurality of expert-tier raters' votes, one
    vote per rater (their own plurality across repeat presentations), with
    that rater's vote left out when the reader being scored is themselves
    expert-tier. At least three votes and a strict plurality are required.
2.  Readers below ``other``/``unknown`` expertise are dropped so the pool
    is not dominated by near-chance annotators, then the artifact's own
    subset rule is applied: >=100 categorical reads, >=30 wrong picks,
    interior sensitivity ``|l| <= 2.5``.
3.  Pooled conditional fit, per-reader profile betas at the pooled lapse,
    DerSimonian-Laird random effects on log beta -- the artifact's
    ``stable_core`` recipe.

The overall fit is reported alongside a kong-only / sparcnet-only split,
since the two corpora ran different labelling protocols and a disagreement
between them would say the "population" is protocol-specific rather than
universal. Research-only diagnostic; emits no artifact and promotes
nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifact_engine_frame import (
    engine_frame_records,
    fit_conditional,
    fit_reader_model,
    profile_beta,
    random_effects_log_beta,
)
from .engine_frame_refit import ReadRecord
from .stage_real_artifact import CLASSES, CLASS_INDEX, _plurality, _sha256, _to_class

LABEL_TYPE = "pattern_class"
KONG_SOURCES = (
    "iiic_crowdsourcing:kong2025:crowd",
    "iiic_crowdsourcing:kong2025:expert",
)
SPARCNET_SOURCES = ("sparcnet50K",)
OOS_SOURCES = KONG_SOURCES + SPARCNET_SOURCES
EXCLUDED_EXPERTISE = ("other", "unknown")
EXPERT_TIER = "expert"
MIN_EXPERT_VOTES = 3

# reports/../artifacts/iiic_conditional_f1_engine_frame_rd.json -> randomEffects
ARTIFACT = {
    "path": "artifacts/iiic_conditional_f1_engine_frame_rd.json",
    "mu_log_beta": -0.020575228248171405,
    "tau_log_beta": 0.25537783624586124,
    "se_mu_log_beta": 0.02646174635139151,
    "n_readers_used": 124,
    "n_wrong_picks": 58358,
}


def signal_lookups(signals_path: Path) -> tuple[dict, dict]:
    """Segment -> engine-order (s_mean, s_sd) tuples with the index-0 pads."""
    signals = pd.read_csv(signals_path)
    mean_columns = [f"s_mean_{name}" for name in CLASSES]
    sd_columns = [f"s_sd_{name}" for name in CLASSES]
    signals = signals.dropna(subset=mean_columns + sd_columns)
    mean_lookup: dict[int, tuple] = {}
    sd_lookup: dict[int, tuple] = {}
    for row in signals.itertuples():
        segment_id = int(row.seg_id)
        mean_lookup[segment_id] = tuple(
            [0.0] + [float(getattr(row, column)) for column in mean_columns]
        )
        sd_lookup[segment_id] = tuple(
            [1.0] + [float(getattr(row, column)) for column in sd_columns]
        )
    return mean_lookup, sd_lookup


def load_reads(labels_path: Path, raters_path: Path) -> pd.DataFrame:
    """Categorical out-of-sample reads with a mapped class and rater tier."""
    labels = pd.read_csv(labels_path, low_memory=False)
    frame = labels[
        (labels["label_type"] == LABEL_TYPE)
        & (labels["source_dataset"].isin(OOS_SOURCES))
    ].copy()
    frame["pick"] = [_to_class(value) for value in frame["value"]]
    frame = frame.dropna(subset=["pick"])
    raters = pd.read_csv(raters_path)
    tier = dict(zip(raters["rater_id"], raters["expertise_level"], strict=True))
    frame["tier"] = frame["rater_id"].map(tier)
    return frame


def centaur_readers(raters_path: Path) -> set[str]:
    """Reader ids tagged with any CENTAUR roster group.

    The out-of-sample READS are disjoint from the artifact's corpus by
    construction, but a handful of the same people annotated in both
    programmes; this reports how far the reader populations overlap so the
    independence claim can be read with the right caveat.
    """
    raters = pd.read_csv(raters_path)
    flagged = set()
    for rater_id, groups in zip(raters["rater_id"], raters["groups"], strict=True):
        if isinstance(groups, str) and "entaur" in groups:
            flagged.add(f"rater-{rater_id}")
    return flagged


def expert_votes(frame: pd.DataFrame) -> dict[int, dict]:
    """Segment -> {expert rater: that rater's own plurality class}.

    Repeat presentations of a segment to the same rater are collapsed to one
    vote so a single high-volume expert cannot carry a segment's plurality;
    a rater who split their own repeats evenly casts no vote.
    """
    expert = frame[frame["tier"] == EXPERT_TIER]
    votes: dict[int, dict] = {}
    for (segment_id, rater_id), picks in expert.groupby(["seg_id", "rater_id"])["pick"]:
        vote = _plurality(list(picks))
        if vote is not None:
            votes.setdefault(int(segment_id), {})[rater_id] = vote
    return votes


def gold_for(votes: dict[int, dict], segment_id: int, rater_id, is_expert: bool):
    """Leave-own-read-out expert plurality, or None when it is not identified."""
    segment_votes = votes.get(segment_id)
    if segment_votes is None:
        return None
    if is_expert:
        segment_votes = {
            voter: vote for voter, vote in segment_votes.items() if voter != rater_id
        }
    if len(segment_votes) < MIN_EXPERT_VOTES:
        return None
    return _plurality(list(segment_votes.values()))


def stage_reads(
    frame: pd.DataFrame,
    votes: dict[int, dict],
    mean_lookup: dict,
    sd_lookup: dict,
) -> tuple[list[ReadRecord], dict]:
    """Scoreable reads plus the counts at each staging filter."""
    on_signal = frame[frame["seg_id"].isin(mean_lookup.keys())]
    scored = on_signal[~on_signal["tier"].isin(EXCLUDED_EXPERTISE)]
    reads: list[ReadRecord] = []
    for row in scored.itertuples(index=False):
        segment_id = int(row.seg_id)
        gold = gold_for(votes, segment_id, row.rater_id, row.tier == EXPERT_TIER)
        if gold is None:
            continue
        reads.append(ReadRecord(
            reader_id=f"rater-{row.rater_id}",
            source_id=f"seg-{segment_id}",
            asked_k=CLASS_INDEX[gold],
            pick_k=CLASS_INDEX[row.pick],
            s_mean=mean_lookup[segment_id],
            s_sd=sd_lookup[segment_id],
        ))
    stages = {
        "reads_categorical": int(len(frame)),
        "readers_categorical": int(frame["rater_id"].nunique()),
        "reads_on_complete_signal_segments": int(len(on_signal)),
        "reads_after_expertise_filter": int(len(scored)),
        "readers_after_expertise_filter": int(scored["rater_id"].nunique()),
        "reads_with_gold": len(reads),
        "readers_with_gold": len({read.reader_id for read in reads}),
        "segments_with_gold": len({read.source_id for read in reads}),
    }
    return reads, stages


def select_readers(
    reads: list[ReadRecord], min_reads: int, min_wrong: int, interior_l: float,
) -> tuple[dict[str, list[ReadRecord]], dict[str, dict], dict]:
    """Apply the artifact's subset rule and fit each survivor's engine model."""
    by_reader: dict[str, list[ReadRecord]] = {}
    for read in reads:
        by_reader.setdefault(read.reader_id, []).append(read)

    def wrong_count(rows: list[ReadRecord]) -> int:
        return sum(1 for row in rows if row.pick_k != row.asked_k)

    volume_pool = {
        reader_id: rows for reader_id, rows in by_reader.items()
        if len(rows) >= min_reads and wrong_count(rows) >= min_wrong
    }
    models: dict[str, dict] = {}
    failed: list[str] = []
    for reader_id, rows in volume_pool.items():
        try:
            models[reader_id] = fit_reader_model(rows)
        except RuntimeError:
            failed.append(reader_id)
    core = {
        reader_id: volume_pool[reader_id] for reader_id, model in models.items()
        if abs(model["l"]) <= interior_l
    }
    stages = {
        "readers_with_any_gold_read": len(by_reader),
        "readers_min_reads_and_min_wrong": len(volume_pool),
        "readers_reader_model_failed": len(failed),
        "readers_non_interior_sensitivity": len(models) - len(core),
        "readers_stable_core": len(core),
    }
    return core, {reader_id: models[reader_id] for reader_id in core}, stages


def tier_summary(fits: list[dict], tiers: dict[str, str]) -> dict:
    """Per-expertise-tier beta summary, so a split gap can be read against
    reader composition rather than mistaken for a protocol effect."""
    grouped: dict[str, list[float]] = {}
    for fit in fits:
        grouped.setdefault(str(tiers.get(fit["reader_id"])), []).append(
            float(np.log(fit["beta"]))
        )
    return {
        tier: {
            "n_readers": len(values),
            "mean_log_beta": float(np.mean(values)),
            "median_beta": float(np.exp(np.median(values))),
        }
        for tier, values in sorted(grouped.items())
    }


def fit_population(
    core: dict[str, list[ReadRecord]], models: dict[str, dict], tiers: dict[str, str],
) -> dict:
    """Pooled conditional, per-reader profile betas, DL random effects."""
    per_reader = {
        reader_id: engine_frame_records(rows, models[reader_id])
        for reader_id, rows in core.items()
    }
    pooled_records = [record for records in per_reader.values() for record in records]
    pooled = fit_conditional(pooled_records)
    fits = [
        {"reader_id": reader_id, **profile_beta(records, pooled["distractor_lapse"])}
        for reader_id, records in sorted(per_reader.items())
    ]
    effects = random_effects_log_beta(fits)
    sensitivities = [model["l"] for model in models.values()]
    return {
        "n_readers": len(core),
        "n_wrong_picks": len(pooled_records),
        "pooled": pooled,
        "random_effects": effects,
        "sensitivity_summary": {
            "mean_l": float(np.mean(sensitivities)),
            "min_l": float(np.min(sensitivities)),
            "max_l": float(np.max(sensitivities)),
        },
        "beta_quantiles": {
            str(q): float(np.quantile([fit["beta"] for fit in fits], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "tier_summary": tier_summary(fits, tiers),
        "reader_beta_fits": fits,
    }


def compare_to_artifact(effects: dict) -> dict:
    """Two-sample z on mu and the tau ratio against the CENTAUR artifact."""
    difference = effects["mu_log_beta"] - ARTIFACT["mu_log_beta"]
    combined_se = float(np.sqrt(
        effects["se_mu_log_beta"] ** 2 + ARTIFACT["se_mu_log_beta"] ** 2
    ))
    return {
        "mu_difference": float(difference),
        "combined_se_mu": combined_se,
        "z_mu": float(difference / combined_se) if combined_se > 0 else float("inf"),
        "beta_ratio_at_mu": float(np.exp(difference)),
        "tau_ratio": float(effects["tau_log_beta"] / ARTIFACT["tau_log_beta"]),
        "tau_difference": float(effects["tau_log_beta"] - ARTIFACT["tau_log_beta"]),
    }


def analyze(
    frame: pd.DataFrame,
    votes: dict[int, dict],
    mean_lookup: dict,
    sd_lookup: dict,
    min_reads: int,
    min_wrong: int,
    interior_l: float,
    centaur_ids: set[str],
) -> dict:
    tiers = {
        f"rater-{row.rater_id}": row.tier
        for row in frame[["rater_id", "tier"]].drop_duplicates().itertuples(index=False)
    }
    reads, stage_counts = stage_reads(frame, votes, mean_lookup, sd_lookup)
    core, models, reader_stages = select_readers(reads, min_reads, min_wrong, interior_l)
    population = fit_population(core, models, tiers)
    return {
        "sources": sorted(frame["source_dataset"].unique().tolist()),
        "segments_with_expert_gold_pool": len(votes),
        "stages": {
            **stage_counts,
            **reader_stages,
            "readers_stable_core_also_on_centaur_roster": len(
                set(core) & centaur_ids
            ),
        },
        **population,
        "artifact_comparison": compare_to_artifact(population["random_effects"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--raters", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--min-reads", type=int, default=100)
    parser.add_argument("--min-wrong", type=int, default=30)
    parser.add_argument("--interior-l", type=float, default=2.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mean_lookup, sd_lookup = signal_lookups(args.signals)
    frame = load_reads(args.labels, args.raters)
    kong = frame[frame["source_dataset"].isin(KONG_SOURCES)]
    sparcnet = frame[frame["source_dataset"].isin(SPARCNET_SOURCES)]
    shared_votes = expert_votes(frame)
    # Primary split holds gold fixed at the pooled out-of-sample expert panel so a
    # source difference reflects the reading protocol, not gold availability; the
    # isolated variants score each corpus against its own experts only, which for
    # kong2025 identifies gold on a couple of hundred segments and is reported as
    # a supplementary check rather than the headline split.
    splits = {
        "overall": (frame, shared_votes),
        "kong2025": (kong, shared_votes),
        "sparcnet50K": (sparcnet, shared_votes),
        "kong2025_isolated_gold": (kong, expert_votes(kong)),
        "sparcnet50K_isolated_gold": (sparcnet, expert_votes(sparcnet)),
    }
    centaur_ids = centaur_readers(args.raters)
    results = {
        name: analyze(
            subset, votes, mean_lookup, sd_lookup,
            args.min_reads, args.min_wrong, args.interior_l, centaur_ids,
        )
        for name, (subset, votes) in splits.items()
    }
    def cross_source(kong_name: str, sparcnet_name: str) -> dict:
        left = results[kong_name]["random_effects"]
        right = results[sparcnet_name]["random_effects"]
        difference = left["mu_log_beta"] - right["mu_log_beta"]
        combined_se = float(np.sqrt(
            left["se_mu_log_beta"] ** 2 + right["se_mu_log_beta"] ** 2
        ))
        return {
            "mu_difference_kong_minus_sparcnet": float(difference),
            "combined_se_mu": combined_se,
            "z_mu": float(difference / combined_se) if combined_se > 0 else float("inf"),
            "beta_ratio_kong_over_sparcnet": float(np.exp(difference)),
            "tau_ratio_kong_over_sparcnet": float(
                left["tau_log_beta"] / right["tau_log_beta"]
            ) if right["tau_log_beta"] > 0 else float("inf"),
        }

    payload = {
        "schema_version": 1,
        "status": "research_only_diagnostic",
        "analysis": "out_of_sample_reader_population_check",
        "artifact_reference": ARTIFACT,
        "subset_rule": {
            "min_reads": args.min_reads,
            "min_wrong_picks": args.min_wrong,
            "interior_abs_l_max": args.interior_l,
            "excluded_expertise": list(EXCLUDED_EXPERTISE),
            "min_expert_votes": MIN_EXPERT_VOTES,
            "gold_rule": "leave_own_read_out_expert_plurality;one_vote_per_expert_rater",
        },
        "sources": {
            "kong2025": list(KONG_SOURCES),
            "sparcnet50K": list(SPARCNET_SOURCES),
        },
        "source_sha256": {
            str(path): _sha256(path)
            for path in (args.labels, args.raters, args.signals)
        },
        "cross_source_check": {
            "shared_gold": cross_source("kong2025", "sparcnet50K"),
            "isolated_gold": cross_source(
                "kong2025_isolated_gold", "sparcnet50K_isolated_gold",
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            "stages": result["stages"],
            "n_wrong_picks": result["n_wrong_picks"],
            "pooled": result["pooled"],
            "random_effects": result["random_effects"],
            "artifact_comparison": result["artifact_comparison"],
            "tier_summary": result["tier_summary"],
        }
        for name, result in results.items()
    }, indent=2, sort_keys=True))
    print(json.dumps(payload["cross_source_check"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
