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
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from .precision_stop import PRECISION_PER_DOMAIN_CAP, ensure_sidecar_built, get_client
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
    # "own-cap" preserves the historical fixed direct-domain budget exactly.
    # "precision" joins the UNCHANGED production stopping policy through the
    # TS sidecar (QUALIFICATION.md run family 3); own_cap then only remains
    # as a hard per-domain safety cap on top of the policy's own ceiling.
    stopping: Literal["own-cap", "precision"] = "own-cap"
    # When set, the replicate bank is a seeded draw from this served-bank CSV
    # (.artifacts/categorical_bank_axes.csv schema) instead of the synthetic
    # generator.
    real_bank: str | None = None
    # Full-served-bank signal terciles for the precision content floor
    # (production derives these before any per-session draw:
    # cortex_web/services/api/session_bank.py:60-83). Precomputed once in the
    # parent for real banks; per-replicate synthetic banks derive their own.
    precision_band_edges: tuple[tuple[float, float], ...] | None = None


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
    # Final per-domain Precision selection states (precision stopping only;
    # any remaining "ACTIVE" entry means the own-cap safety cap or bank
    # exhaustion ended the session, not the policy).
    end_statuses: list[str] | None = None


@lru_cache(maxsize=2)
def _load_real_bank(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the staged served-bank axes CSV (stage_real_bank.py schema).

    Columns: segment_index, seg_id, s_mean_0..6, s_sd_0..6. The spike axis
    (column 0) is NaN on the IIIC-only bank; the harness never asks domain 0
    and no domain-0 candidate is ever presented to the stopping policy.
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 16:
        raise ValueError(f"unexpected real-bank shape {data.shape} in {path}")
    return data[:, 1].astype(int), data[:, 2:9], data[:, 9:16]


# N(0,1) terciles: structural placeholder edges for a domain with no servable
# signals (the spike axis on the IIIC-only real bank). Such a domain has zero
# candidates, so the policy marks it UNDETERMINABLE_BANK on the first
# evaluation and its band edges are never consulted for a real item.
_PLACEHOLDER_EDGES = (-0.43072729929545744, 0.43072729929545744)


def _tercile_band_edges(s_mean: np.ndarray) -> list[list[float]]:
    """Full-bank signal terciles, mirroring session_bank.py:_linear_quantile
    (NumPy default linear quantile) per domain."""
    edges: list[list[float]] = []
    for k in range(s_mean.shape[1]):
        values = s_mean[:, k]
        values = values[np.isfinite(values)]
        if values.size < 3:
            edges.append(list(_PLACEHOLDER_EDGES))
            continue
        q1 = float(np.quantile(values, 1 / 3))
        q2 = float(np.quantile(values, 2 / 3))
        if not q1 < q2:
            edges.append(list(_PLACEHOLDER_EDGES))
            continue
        edges.append([q1, q2])
    return edges


def _truth_and_bank(seed: int, config: QualificationConfig):
    if not np.isfinite(config.signal_sd_scale) or config.signal_sd_scale <= 0:
        raise ValueError("signal_sd_scale must be finite and positive")
    rng = np.random.default_rng(seed)
    truth_t = rng.normal(config.truth_t_mean, config.truth_sd, 7)
    truth_l = rng.normal(config.truth_l_mean, config.truth_sd, 7)
    truth_t[0], truth_l[0] = 0.0, 0.0
    if config.real_bank is not None:
        bank_seg_ids, bank_mean, bank_sd = _load_real_bank(config.real_bank)
        if config.bank_segments > bank_seg_ids.size:
            raise ValueError(
                f"bank_segments={config.bank_segments} exceeds the "
                f"{bank_seg_ids.size}-segment real bank"
            )
        chosen = np.sort(rng.choice(
            bank_seg_ids.size, size=config.bank_segments, replace=False,
        ))
        return (
            truth_t, truth_l, bank_mean[chosen],
            bank_sd[chosen] * config.signal_sd_scale, bank_seg_ids[chosen],
        )
    s_mean = rng.normal(0, 1.25, size=(config.bank_segments, 7))
    # Preserve realistic cross-class ambiguity rather than six independent axes.
    common = rng.normal(0, 0.5, size=(config.bank_segments, 1))
    s_mean[:, 1:] += common
    s_sd = rng.uniform(0.02, 0.18, size=(config.bank_segments, 7)) * config.signal_sd_scale
    return truth_t, truth_l, s_mean, s_sd, np.arange(config.bank_segments)


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


def _precision_bank_view(
    remaining: set[int],
    seg_ids: np.ndarray,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
) -> dict:
    """Remaining candidate views for the frozen policy's bank telemetry.

    Every remaining segment is one candidate per IIIC domain, carrying only
    the focal (asked-class) signal — the exact view precisionBankArrays
    scatters (src/precision_bridge.ts:12-29). Domain 0 is never served by
    this harness, so it gets no candidates and the policy terminalizes it as
    UNDETERMINABLE_BANK on the first evaluation.
    """
    asked, ids, means, sds = [], [], [], []
    ordered = sorted(remaining)
    for k in IIIC_GROUP:
        for index in ordered:
            asked.append(k)
            ids.append(int(seg_ids[index]))
            means.append(float(s_mean[index, k]))
            sds.append(float(s_sd[index, k]))
    return {"askedK": asked, "segId": ids, "sMean": means, "sSd": sds}


def _run_arm(seed: int, arm: Literal["binary", "categorical_f1"], config: QualificationConfig):
    truth_t, truth_l, s_mean, s_sd, seg_ids = _truth_and_bank(seed, config)
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
    precision_client = None
    session_id = f"{seed}:{arm}"
    last_rejuvenation: dict | None = None
    last_result: dict | None = None
    if config.stopping == "precision":
        precision_client = get_client()
        band_edges = (
            [list(pair) for pair in config.precision_band_edges]
            if config.precision_band_edges is not None
            else _tercile_band_edges(s_mean)
        )
        identity = np.eye(7).tolist()
        precision_client.init_session(session_id, identity, identity, band_edges)
    # own_cap bounds each domain in both modes. Under precision stopping it is
    # only a hard safety cap: the session normally ends when the unchanged
    # policy reports stop (or the bank legitimately empties first).
    while np.any(counts[1:] < config.own_cap):
        if precision_client is not None and not remaining:
            break
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
            last_rejuvenation = {
                "qIndex": questions,
                "acceptanceRate": acceptance,
                "distinctAncestors": int(round(ancestry * config.particles)),
                "distinctAncestorFraction": ancestry,
            }
        remaining.remove(segment_index)
        counts[asked_k] += 1
        questions += 1
        if precision_client is not None:
            # advance.ts:498-514 ordering: the administered item is recorded,
            # then the policy sees the post-update cloud, the incremented
            # per-domain counts, and the post-removal remaining bank.
            last_result = precision_client.evaluate(
                session_id,
                {"k": asked_k, "signal": float(s_mean[segment_index, asked_k])},
                cloud.t, cloud.l, cloud.w, last_rejuvenation,
                counts.tolist(),
                _precision_bank_view(remaining, seg_ids, s_mean, s_sd),
            )
            if last_result["stop"]:
                break

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
        end_statuses=(
            list(last_result["selectionStates"]) if last_result is not None else None
        ),
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


DEFAULT_SEED_BASE = 62_100_000


def run_rows(
    config: QualificationConfig,
    workers: int | None = None,
    seed_base: int = DEFAULT_SEED_BASE,
) -> tuple[list[ReplicateResult], int]:
    workers = workers or recommended_workers(config.replicates, config.particles)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        paired = list(executor.map(
            run_replicate,
            range(seed_base, seed_base + config.replicates),
            [config] * config.replicates,
        ))
    return [row for pair in paired for row in pair], workers


def run_parallel(
    config: QualificationConfig,
    workers: int | None = None,
    seed_base: int = DEFAULT_SEED_BASE,
) -> dict:
    rows, workers = run_rows(config, workers, seed_base)
    summary = summarize(rows, config)
    summary["workers"] = workers
    summary["seed_base"] = seed_base
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
    parser.add_argument("--stopping", choices=("own-cap", "precision"), default="own-cap")
    parser.add_argument("--real-bank", type=Path)
    parser.add_argument("--artifact-ensemble", type=Path)
    parser.add_argument("--ensemble-draws", type=int, default=9)
    parser.add_argument("--formal-sbc", action="store_true")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--rows-output", type=Path)
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
    if args.real_bank:
        config = replace(config, real_bank=str(args.real_bank))
    if args.stopping == "precision":
        # The unchanged production policy owns the stop decision; unless the
        # operator narrows the safety cap explicitly, align it with the frozen
        # per-domain ceiling (precision_policy.ts PRECISION_PER_DOMAIN_CAP) so
        # the policy's own CAP terminalization is what bounds a domain.
        config = replace(
            config,
            stopping="precision",
            own_cap=args.own_cap if args.own_cap is not None else PRECISION_PER_DOMAIN_CAP,
        )
        if config.real_bank is not None:
            _, bank_mean, _ = _load_real_bank(config.real_bank)
            config = replace(config, precision_band_edges=tuple(
                tuple(pair) for pair in _tercile_band_edges(bank_mean)
            ))
        ensure_sidecar_built()
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
    rows, workers = run_rows(config, args.workers, args.seed_base)
    summary = summarize(rows, config)
    summary["workers"] = workers
    summary["seed_base"] = args.seed_base
    if args.rows_output:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_output.open("w") as handle:
            handle.write(json.dumps({
                "config": asdict(config), "seed_base": args.seed_base,
            }, sort_keys=True) + "\n")
            for row in rows:
                # Null optional fields are omitted so pre-precision row files
                # remain byte-identical.
                payload = {
                    key: value for key, value in asdict(row).items()
                    if value is not None
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
