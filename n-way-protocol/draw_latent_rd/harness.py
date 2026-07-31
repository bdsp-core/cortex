"""Viability battery for the draw-latent engine (construction B).

Reuses the governed harness's truth generation, selector, profile, and
summary statistics unchanged; only the categorical arm's inference runs
on the draw-latent cloud. Binary control comes from the shipping
``_run_arm`` verbatim, so every cell is CRN-paired against the incumbent
exactly as in the pre-registered smokes.

Cells (192 seeds, production profile, ~6 min each), in decision order:

- ``ongrid9_exact``   truth on the nine atoms incl. their 0.15 lapse —
  the implementation verdict: joint inference over a prior containing
  the truth must be calibrated (~0.95), or construction B is not viable.
- ``ongrid9_floor``   truth lapse 0 (the deployed floor misspec) —
  prices the floor's own calibration cost under B.
- ``continuum9/17/33`` the pre-registered continuum world against
  atom-count; static averaging scored 0.9227 here.
- ``stress190``       beta = 1.903 (independent expert median) on a
  tau-widened 33-atom grid; static scored 0.8108.
- ``collapse_nest``   uniform-collapse world on the widened grid plus a
  lambda=1 binary-equivalent atom: the graceful-degradation claim.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from nway_protocol.artifact_engine_frame import population_draws
from nway_protocol.artifact_floor import floor_draws
from nway_protocol.qualification import (
    IIIC_GROUP,
    QualificationConfig,
    ReplicateResult,
    _run_arm,
    _select,
    _truth_and_bank,
    recommended_workers,
    summarize,
)
from nway_protocol.reference import (
    Observation,
    ess,
    f1_probabilities,
    posterior_moments,
    weighted_quantile,
)

from .engine import (
    atom_posterior,
    make_draw_cloud,
    resample_and_rejuvenate_draw,
    update_draw_latent,
)

ARTIFACT = Path(__file__).resolve().parents[1] / (
    "artifacts/iiic_conditional_f1_engine_frame_rd.json"
)
OOS_TAU = 0.3581  # independent-population heterogeneity (oos_population_check)


def build_atoms(name: str) -> tuple[tuple[float, float, float], ...]:
    payload = json.loads(ARTIFACT.read_text())
    effects = payload["randomEffects"]
    floor = payload["robustnessFloor"]

    def quantile_grid(count: int, tau: float | None = None) -> list[dict]:
        adjusted = dict(effects)
        if tau is not None:
            adjusted["tau_log_beta"] = tau
        return floor_draws(
            population_draws(adjusted, 0.0, count), floor,
        )

    if name == "atoms9":
        rows = payload["bootstrap"]["draws"]
    elif name == "atoms17":
        rows = quantile_grid(17)
    elif name == "atoms33":
        rows = quantile_grid(33)
    elif name == "widened33":
        rows = quantile_grid(33, tau=OOS_TAU)
    elif name == "nesting34":
        rows = quantile_grid(33, tau=OOS_TAU)
        rows = [dict(row) for row in rows] + [
            {"beta": 0.9796, "distractor_lapse": 1.0, "weight": 1.0}
        ]
        for row in rows:
            row["weight"] = 1.0 / len(rows)
    else:
        raise ValueError(f"unknown atom grid {name}")
    return tuple(
        (float(row["beta"]), float(row["distractor_lapse"]), float(row["weight"]))
        for row in rows
    )


def _run_categorical_draw_latent(
    seed: int, config: QualificationConfig,
) -> tuple[ReplicateResult, dict]:
    truth_t, truth_l, s_mean, s_sd, _seg_ids = _truth_and_bank(seed, config)
    world_beta = config.beta if config.truth_beta is None else config.truth_beta
    if config.truth_beta_lognormal is not None:
        mu, sigma = config.truth_beta_lognormal
        world_beta = float(np.exp(
            mu + sigma * np.random.default_rng(seed + 70_000).standard_normal()
        ))
    if config.truth_beta_choices is not None:
        world_beta = float(np.random.default_rng(seed + 70_000).choice(
            np.asarray(config.truth_beta_choices)
        ))
    world_lapse = (
        config.distractor_lapse
        if config.truth_distractor_lapse is None
        else config.truth_distractor_lapse
    )
    prior_corr = np.eye(7)
    dcloud = make_draw_cloud(
        config.particles, prior_corr, prior_corr, config.artifact_draws,
        np.random.default_rng(seed + 20_000),
    )
    response_rng = np.random.default_rng(seed + 40_000)
    mh_rng = np.random.default_rng(seed + 60_000)
    remaining = set(range(config.bank_segments))
    counts = np.zeros(7, dtype=int)
    acceptances, ancestries = [], []
    questions = 0
    while np.any(counts[1:] < config.own_cap):
        asked_k, segment_index = _select(
            dcloud.cloud, "categorical_f1", remaining, counts, s_mean, s_sd, config,
        )
        truth_probabilities = f1_probabilities(
            truth_t, truth_l, s_mean[segment_index], s_sd[segment_index],
            asked_k, IIIC_GROUP, world_beta, world_lapse,
        )[0]
        pick = int(response_rng.choice(IIIC_GROUP, p=truth_probabilities))
        observation = Observation(
            kind="categorical_f1", asked_k=asked_k, segment_index=segment_index,
            raw_pick=pick, y=None, group=IIIC_GROUP,
        )
        update_draw_latent(dcloud, observation, s_mean[segment_index], s_sd[segment_index])
        if ess(dcloud.cloud) < config.ess_fraction * config.particles:
            acceptance, ancestry = resample_and_rejuvenate_draw(
                dcloud, s_mean, s_sd, mh_rng, config.mh_steps, 2.38 / np.sqrt(14),
            )
            acceptances.append(acceptance)
            ancestries.append(ancestry)
        remaining.remove(segment_index)
        counts[asked_k] += 1
        questions += 1

    cloud = dcloud.cloud
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
    mass = atom_posterior(dcloud)
    diagnostics = {
        "world_beta": world_beta,
        "atoms_surviving": int(np.count_nonzero(np.asarray(mass) > 1e-12)),
        "atom_max_mass": float(max(mass)),
        "map_atom_beta": float(dcloud.atoms[int(np.argmax(mass))][0]),
    }
    row = ReplicateResult(
        seed=seed,
        arm="categorical_f1",
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
    )
    return row, diagnostics


def run_replicate_b(seed: int, config: QualificationConfig):
    binary = _run_arm(seed, "binary", config)
    categorical, diagnostics = _run_categorical_draw_latent(seed, config)
    return binary, categorical, diagnostics


CELLS = {
    "ongrid9_exact": ("atoms9", {"truth_distractor_lapse": 0.15, "grid_truth": True}),
    "ongrid9_floor": ("atoms9", {"grid_truth": True}),
    "continuum9": ("atoms9", {"lognormal": True}),
    "continuum17": ("atoms17", {"lognormal": True}),
    "continuum33": ("atoms33", {"lognormal": True}),
    "stress190": ("widened33", {"truth_beta": 1.903}),
    "collapse_nest": ("nesting34", {"truth_distractor_lapse": 1.0}),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--replicates", type=int, default=192)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed-base", type=int, default=62_100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    atom_name, overrides = CELLS[args.cell]
    atoms = build_atoms(atom_name)
    payload = json.loads(ARTIFACT.read_text())
    effects = payload["randomEffects"]
    config = QualificationConfig(
        replicates=args.replicates, particles=1_200, own_cap=15,
        bank_segments=420, selector_per_domain=32, mh_steps=30,
        artifact_draws=atoms, formal_sbc=True,
        truth_t_mean=0.0, truth_l_mean=0.0, truth_sd=1.0,
    )
    if overrides.get("grid_truth"):
        config = replace(config, truth_beta_choices=tuple(a[0] for a in atoms))
    if overrides.get("lognormal"):
        config = replace(config, truth_beta_lognormal=(
            effects["mu_log_beta"], effects["tau_log_beta"],
        ))
    if "truth_beta" in overrides:
        config = replace(config, truth_beta=overrides["truth_beta"])
    if "truth_distractor_lapse" in overrides:
        config = replace(
            config, truth_distractor_lapse=overrides["truth_distractor_lapse"],
        )

    workers = args.workers or recommended_workers(config.replicates, config.particles)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            run_replicate_b,
            range(args.seed_base, args.seed_base + config.replicates),
            [config] * config.replicates,
        ))
    rows = [row for binary, categorical, _ in results for row in (binary, categorical)]
    diagnostics = [diag for _, _, diag in results]
    summary = summarize(rows, config)
    summary["workers"] = workers
    summary["cell"] = args.cell
    summary["atom_grid"] = {
        "name": atom_name,
        "n_atoms": len(atoms),
        "betas": [round(atom[0], 4) for atom in atoms],
        "lapses": [round(atom[1], 4) for atom in atoms],
    }
    summary["draw_latent_diagnostics"] = {
        "mean_atoms_surviving": float(np.mean([d["atoms_surviving"] for d in diagnostics])),
        "mean_atom_max_mass": float(np.mean([d["atom_max_mass"] for d in diagnostics])),
        "map_atom_tracks_world": float(np.mean([
            abs(np.log(d["map_atom_beta"] / d["world_beta"])) < 0.2
            for d in diagnostics if d["world_beta"] > 0
        ])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    nw = summary["arms"]["categorical_f1"]
    print(json.dumps({
        "cell": args.cell,
        "nway_skill_coverage": nw["skill_coverage"],
        "wilson95": nw["skill_coverage_uncertainty"]["wilson95"],
        "binary_skill_coverage": summary["arms"]["binary"]["skill_coverage"],
        "diagnostics": summary["draw_latent_diagnostics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
