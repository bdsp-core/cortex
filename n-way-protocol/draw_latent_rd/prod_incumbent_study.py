"""Four-arm head-to-head against the PRODUCTION INCUMBENT under Precision stop.

Every prior comparison in this program scored the n-way work against the
BINARY arm. Binary is not the incumbent: since 2026-07-20 production has served
the categorical n-way engine with the UNFLOORED nine-draw static mixture
(``reports/PROD_ENGINE_STATE_FINDINGS.md``). This study therefore runs four
CRN-paired arms against one set of planted truths per seed:

- ``binary``                    the shipping binary arm (reference only).
- ``prod_incumbent_ensemble9``  what production serves today: static
  per-observation mixture over ``iiic_conditional_f1_integrated_ensemble9_rd``
  (unfloored, lambda_d 0.0-0.019).
- ``floored_ensemble9``         the Phase-2 candidate: the same nine draws with
  the owner-approved 0.15 lapse floor.
- ``draw_latent_atoms17``       construction B: a 17-atom draw-latent cloud
  that infers the reader's own (beta, lambda_d) instead of averaging over it.

Stopping is the UNCHANGED production Precision policy on the real served bank,
so ``questions`` is the deployable burden number rather than a fixed cap. Read
burden ONLY next to coverage: the policy stops when the skill posterior radius
contracts inside a fixed tolerance, so an engine whose intervals are too narrow
stops sooner without having earned it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from nway_protocol import qualification
from nway_protocol.qualification import IIIC_GROUP, QualificationConfig, _run_arm

from .harness import _run_categorical_draw_latent, apply_stopping

ROOT = Path(__file__).resolve().parents[1]

BINARY_ARM = "binary"
INCUMBENT_ARM = "prod_incumbent_ensemble9"
FLOORED_ARM = "floored_ensemble9"
DRAW_LATENT_ARM = "draw_latent_atoms17"
ARM_ORDER = (BINARY_ARM, INCUMBENT_ARM, FLOORED_ARM, DRAW_LATENT_ARM)

ARTIFACTS = {
    INCUMBENT_ARM: ROOT / "artifacts/iiic_conditional_f1_integrated_ensemble9_rd.json",
    FLOORED_ARM: ROOT / "artifacts/iiic_conditional_f1_integrated_ensemble9_rd_floor015.json",
    DRAW_LATENT_ARM: ROOT / "artifacts/iiic_conditional_f1_engine_frame_atoms17_rd.json",
}

# Fitted reader population (engine-frame DerSimonian-Laird log-beta random
# effects). The continuum world draws each replicate's own truth beta from it.
TRUTH_LOGNORMAL = (-0.020575228248171405, 0.25537783624586124)
TRUTH_DISTRACTOR_LAPSE = 0.0

NOMINAL = 0.95
ROW_KEYS = (
    "skill_rmse", "bias_rmse", "skill_width", "bias_width",
    "skill_coverage", "bias_coverage", "questions",
)


def load_draws(path: Path) -> tuple[tuple[float, float, float], ...]:
    """Read an artifact's mixture/atom draws in either shipped key style.

    The integrated ensembles publish a top-level camelCase ``draws`` array
    (integration_experiment.py:249-258); the engine-frame artifacts publish
    ``bootstrap.draws`` in snake_case (harness.build_atoms).
    """
    payload = json.loads(path.read_text())
    rows = payload["draws"] if "draws" in payload else payload["bootstrap"]["draws"]
    return tuple(
        (
            float(row["beta"]),
            float(row.get("distractorLapse", row.get("distractor_lapse"))),
            float(row["weight"]),
        )
        for row in rows
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _domain_counts_recorder():
    """Capture the per-domain question counter that ``_run_arm`` mutates.

    ``qualification._run_arm`` never reports per-domain burden, and this study
    may not modify it. ``_select`` is handed the arm's single ``counts`` array
    on every question, and ``_run_arm`` increments that same array in place, so
    holding a reference to it yields the arm's final per-domain counts. The
    wrapper delegates verbatim, and the caller asserts ``sum(counts) ==
    questions`` so a stale or wrong reference cannot pass silently.
    """
    captured: list[np.ndarray] = []
    original = qualification._select

    def recording(cloud, arm, remaining, counts, s_mean, s_sd, config, *args, **kwargs):
        if not captured:
            captured.append(counts)
        return original(cloud, arm, remaining, counts, s_mean, s_sd, config, *args, **kwargs)

    qualification._select = recording
    try:
        yield captured
    finally:
        qualification._select = original


def _row_payload(arm: str, row, domain_counts: list[int]) -> dict:
    payload = {key: value for key, value in asdict(row).items()}
    payload["arm"] = arm
    payload["domain_counts"] = [int(x) for x in domain_counts]
    return payload


def _run_reference_arm(seed: int, kind: str, arm: str, config: QualificationConfig) -> dict:
    with _domain_counts_recorder() as captured:
        row = _run_arm(seed, kind, config)
    counts = captured[0].tolist() if captured else [0] * 7
    if int(sum(counts)) != int(row.questions):
        raise RuntimeError(
            f"per-domain counts {counts} disagree with questions {row.questions} "
            f"for seed {seed} arm {arm}"
        )
    return _row_payload(arm, row, counts)


def run_replicate_four_arm(seed: int, configs: tuple[QualificationConfig, ...]) -> dict:
    binary_config, incumbent_config, floored_config, atoms_config = configs
    out: dict = {"seed": seed, "arms": {}}
    out["arms"][BINARY_ARM] = _run_reference_arm(
        seed, "binary", BINARY_ARM, binary_config,
    )
    out["arms"][INCUMBENT_ARM] = _run_reference_arm(
        seed, "categorical_f1", INCUMBENT_ARM, incumbent_config,
    )
    out["arms"][FLOORED_ARM] = _run_reference_arm(
        seed, "categorical_f1", FLOORED_ARM, floored_config,
    )
    row, diagnostics = _run_categorical_draw_latent(seed, atoms_config)
    payload = _row_payload(DRAW_LATENT_ARM, row, diagnostics["domain_counts"])
    if int(sum(payload["domain_counts"])) != int(row.questions):
        raise RuntimeError(f"draw-latent counts disagree with questions at seed {seed}")
    payload["diagnostics"] = {
        key: value for key, value in diagnostics.items() if key != "domain_counts"
    }
    out["arms"][DRAW_LATENT_ARM] = payload
    return out


# --------------------------------------------------------------------------
# summary statistics
# --------------------------------------------------------------------------

def mean_ci(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else math.nan
    return {
        "mean": mean,
        "standard_error": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "n": int(array.size),
    }


def wilson(successes: int, trials: int) -> list[float]:
    z = 1.96
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return [center - radius, center + radius]


def _coverage_block(values: list[float]) -> dict:
    successes = int(round(sum(values) * len(IIIC_GROUP)))
    trials = len(values) * len(IIIC_GROUP)
    interval = wilson(successes, trials)
    return {
        "mean": float(np.mean(values)),
        "successes": successes,
        "trials": trials,
        "wilson95": interval,
        "seed_cluster95": mean_ci(values),
        "nominal": NOMINAL,
        "verdict": (
            "at_nominal" if interval[0] <= NOMINAL <= interval[1]
            else ("above_nominal" if interval[0] > NOMINAL else "below_nominal")
        ),
    }


def _questions_block(values: list[int]) -> dict:
    array = np.asarray(values, dtype=float)
    block = mean_ci(array)
    block.update({
        "median": float(statistics.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    })
    return block


def _arm_summary(rows: list[dict]) -> dict:
    counts = np.asarray([row["domain_counts"] for row in rows], dtype=float)
    statuses: dict[str, int] = {}
    fully_terminal = 0
    for row in rows:
        end = row.get("end_statuses")
        if not end:
            continue
        for k in IIIC_GROUP:
            statuses[end[k]] = statuses.get(end[k], 0) + 1
        if all(end[k] != "ACTIVE" for k in IIIC_GROUP):
            fully_terminal += 1
    summary = {
        "questions": _questions_block([row["questions"] for row in rows]),
        "questions_per_domain": {
            "mean": [float(x) for x in counts.mean(axis=0)],
            "median": [float(x) for x in np.median(counts, axis=0)],
            "max": [float(x) for x in counts.max(axis=0)],
        },
        "skill_rmse": mean_ci([row["skill_rmse"] for row in rows]),
        "bias_rmse": mean_ci([row["bias_rmse"] for row in rows]),
        "skill_width": mean_ci([row["skill_width"] for row in rows]),
        "bias_width": mean_ci([row["bias_width"] for row in rows]),
        "skill_coverage": _coverage_block([row["skill_coverage"] for row in rows]),
        "bias_coverage": _coverage_block([row["bias_coverage"] for row in rows]),
        "resamples": mean_ci([row["resamples"] for row in rows]),
        "mean_acceptance": mean_ci([row["mean_acceptance"] for row in rows]),
        "mean_ancestry": mean_ci([row["mean_ancestry"] for row in rows]),
        "end_status_counts": dict(sorted(statuses.items())),
        "sessions_all_domains_terminal": fully_terminal,
        "sessions_at_full_cap_budget": sum(
            1 for row in rows
            if all(row["domain_counts"][k] >= 60 for k in IIIC_GROUP)
        ),
        "sessions": len(rows),
    }
    return summary


def _paired(rows_a: dict[int, dict], rows_b: dict[int, dict]) -> dict:
    """Per-seed (CRN) differences A minus B for every reported metric."""
    seeds = sorted(set(rows_a) & set(rows_b))
    out = {}
    for key in ROW_KEYS:
        out[key] = mean_ci([rows_a[s][key] - rows_b[s][key] for s in seeds])
    per_domain = []
    for k in range(7):
        per_domain.append(mean_ci([
            rows_a[s]["domain_counts"][k] - rows_b[s]["domain_counts"][k] for s in seeds
        ]))
    out["questions_per_domain"] = per_domain
    out["paired_seeds"] = len(seeds)
    return out


def _honesty(by_arm: dict[str, list[dict]], summaries: dict[str, dict]) -> dict:
    """Burden read next to coverage, plus the within-arm mechanism check.

    Under a precision-TARGETED rule every arm stops at (approximately) the same
    posterior radius, so the final widths are close to constant by construction
    and the only free quantities are (a) how fast an arm claims that width and
    (b) whether the claim is true. Splitting each arm's own sessions by realized
    coverage exposes the exchange directly: if the sessions that missed stop
    EARLIER than the sessions that covered, the short burden was purchased.
    """
    per_arm = {}
    for arm, rows in by_arm.items():
        covered = [row["questions"] for row in rows if row["skill_coverage"] >= 1.0]
        missed = [row["questions"] for row in rows if row["skill_coverage"] < 1.0]
        domains_ended_early = sum(
            1 for row in rows for k in IIIC_GROUP if row["domain_counts"][k] < 60
        )
        block = {
            "domains_that_stopped_before_the_cap": domains_ended_early,
            "domains_total": len(rows) * len(IIIC_GROUP),
            "sessions_at_full_cap_budget": summaries[arm]["sessions_at_full_cap_budget"],
            "skill_coverage_verdict": summaries[arm]["skill_coverage"]["verdict"],
            "skill_coverage": summaries[arm]["skill_coverage"]["mean"],
            "skill_coverage_wilson95": summaries[arm]["skill_coverage"]["wilson95"],
            "mean_questions": summaries[arm]["questions"]["mean"],
            "mean_skill_width_at_stop": summaries[arm]["skill_width"]["mean"],
            "sessions_full_skill_coverage": len(covered),
            "sessions_with_a_miss": len(missed),
            "questions_when_all_six_covered": mean_ci(covered) if len(covered) > 1 else None,
            "questions_when_a_domain_missed": mean_ci(missed) if len(missed) > 1 else None,
        }
        if len(covered) > 1 and len(missed) > 1:
            gap = float(np.mean(missed) - np.mean(covered))
            se = math.sqrt(
                np.var(missed, ddof=1) / len(missed) + np.var(covered, ddof=1) / len(covered)
            )
            block["miss_minus_covered_questions"] = {
                "mean": gap, "standard_error": se,
                "ci95": [gap - 1.96 * se, gap + 1.96 * se],
            }
        block["burden_reading"] = (
            "honest (coverage interval contains nominal)"
            if block["skill_coverage_verdict"] == "at_nominal"
            else ("purchased (coverage interval entirely below nominal)"
                  if block["skill_coverage_verdict"] == "below_nominal"
                  else "conservative (coverage interval entirely above nominal)")
        )
        per_arm[arm] = block
    honest = [
        arm for arm in ARM_ORDER
        if per_arm[arm]["skill_coverage_verdict"] in ("at_nominal", "above_nominal")
    ]
    matched = {
        "honest_arms": honest,
        "supported": len(honest) >= 2,
    }
    if len(honest) >= 2:
        matched["burden_at_matched_honesty"] = {
            arm: per_arm[arm]["mean_questions"] for arm in honest
        }
        cheapest = min(honest, key=lambda arm: per_arm[arm]["mean_questions"])
        matched["cheapest_honest_arm"] = cheapest
    return {"per_arm": per_arm, "matched_honesty": matched}


def summarize_study(replicates: list[dict], config_by_arm: dict) -> dict:
    by_arm = {arm: [rep["arms"][arm] for rep in replicates] for arm in ARM_ORDER}
    by_arm_seed = {
        arm: {rep["seed"]: rep["arms"][arm] for rep in replicates} for arm in ARM_ORDER
    }
    summaries = {arm: _arm_summary(by_arm[arm]) for arm in ARM_ORDER}
    paired = {
        "draw_latent_atoms17_minus_prod_incumbent": _paired(
            by_arm_seed[DRAW_LATENT_ARM], by_arm_seed[INCUMBENT_ARM],
        ),
        "floored_ensemble9_minus_prod_incumbent": _paired(
            by_arm_seed[FLOORED_ARM], by_arm_seed[INCUMBENT_ARM],
        ),
        "prod_incumbent_minus_binary": _paired(
            by_arm_seed[INCUMBENT_ARM], by_arm_seed[BINARY_ARM],
        ),
        "draw_latent_atoms17_minus_binary": _paired(
            by_arm_seed[DRAW_LATENT_ARM], by_arm_seed[BINARY_ARM],
        ),
        "floored_ensemble9_minus_binary": _paired(
            by_arm_seed[FLOORED_ARM], by_arm_seed[BINARY_ARM],
        ),
    }
    return {
        "schema_version": 1,
        "status": "research_only_unqualified_artifact",
        "study": "prod_incumbent_head_to_head",
        "reference_arm": INCUMBENT_ARM,
        "arms": summaries,
        "paired_comparisons": paired,
        "burden_honesty": _honesty(by_arm, summaries),
        "config": config_by_arm,
    }


def build_configs(
    replicates: int, real_bank: Path, particles: int, mh_steps: int,
    bank_segments: int, selector_per_domain: int, stopping: str,
) -> tuple[QualificationConfig, ...]:
    base = QualificationConfig(
        replicates=replicates,
        particles=particles,
        own_cap=15,
        bank_segments=bank_segments,
        selector_per_domain=selector_per_domain,
        mh_steps=mh_steps,
        ess_fraction=0.5,
        formal_sbc=True,
        truth_t_mean=0.0,
        truth_l_mean=0.0,
        truth_sd=1.0,
        truth_beta_lognormal=TRUTH_LOGNORMAL,
        truth_distractor_lapse=TRUTH_DISTRACTOR_LAPSE,
    )
    base = apply_stopping(base, stopping, real_bank)
    # The binary arm never reads artifact_draws (its likelihood, selector and
    # rejuvenation are all group-free), so it is CRN-identical under any of the
    # three response artifacts; it is run once, on the incumbent's config.
    incumbent = replace(base, artifact_draws=load_draws(ARTIFACTS[INCUMBENT_ARM]))
    floored = replace(base, artifact_draws=load_draws(ARTIFACTS[FLOORED_ARM]))
    atoms = replace(base, artifact_draws=load_draws(ARTIFACTS[DRAW_LATENT_ARM]))
    return incumbent, incumbent, floored, atoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=96)
    parser.add_argument("--seed-base", type=int, default=66_000_000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--particles", type=int, default=1_200)
    parser.add_argument("--mh-steps", type=int, default=30)
    parser.add_argument("--bank-segments", type=int, default=420)
    parser.add_argument("--selector-per-domain", type=int, default=32)
    parser.add_argument(
        "--stopping", choices=("own-cap", "precision"), default="precision",
    )
    parser.add_argument(
        "--real-bank", type=Path,
        default=Path(".artifacts/categorical_bank_axes.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configs = build_configs(
        args.replicates, args.real_bank, args.particles, args.mh_steps,
        args.bank_segments, args.selector_per_domain, args.stopping,
    )
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        replicates = list(executor.map(
            run_replicate_four_arm,
            range(args.seed_base, args.seed_base + args.replicates),
            [configs] * args.replicates,
        ))
    elapsed = time.time() - started

    config_by_arm = {
        "shared": asdict(configs[0]) | {"artifact_draws": None},
        "artifacts": {
            arm: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "n_draws": len(load_draws(path)),
                "min_distractor_lapse": min(d[1] for d in load_draws(path)),
                "max_distractor_lapse": max(d[1] for d in load_draws(path)),
            }
            for arm, path in ARTIFACTS.items()
        },
        "artifact_draws_by_arm": {
            arm: [list(draw) for draw in arm_config.artifact_draws]
            for arm, arm_config in zip(ARM_ORDER, configs, strict=True)
            if arm != BINARY_ARM
        },
        "truth_world": {
            "beta": "lognormal", "mu_log_beta": TRUTH_LOGNORMAL[0],
            "tau_log_beta": TRUTH_LOGNORMAL[1],
            "distractor_lapse": TRUTH_DISTRACTOR_LAPSE,
            "formal_sbc": True,
        },
        "seed_base": args.seed_base,
        "workers": args.workers,
        "wall_clock_seconds": elapsed,
    }
    summary = summarize_study(replicates, config_by_arm)
    summary["rows"] = [
        {"seed": rep["seed"], **{arm: rep["arms"][arm] for arm in ARM_ORDER}}
        for rep in replicates
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "wall_clock_seconds": elapsed,
        "arms": {
            arm: {
                "questions_mean": summary["arms"][arm]["questions"]["mean"],
                "questions_median": summary["arms"][arm]["questions"]["median"],
                "skill_rmse": summary["arms"][arm]["skill_rmse"]["mean"],
                "bias_rmse": summary["arms"][arm]["bias_rmse"]["mean"],
                "skill_coverage": summary["arms"][arm]["skill_coverage"]["mean"],
                "skill_coverage_wilson95": summary["arms"][arm]["skill_coverage"]["wilson95"],
                "bias_coverage": summary["arms"][arm]["bias_coverage"]["mean"],
            }
            for arm in ARM_ORDER
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
