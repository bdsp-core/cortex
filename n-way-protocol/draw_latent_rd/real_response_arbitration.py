"""Phase-1c real-response arbitration: which engine describes REAL picks?

Construction B fixes ensemble aggregation; it does not by itself fix the
asked!=gold fit-population mismatch found in production
(reports/PROD_ENGINE_STATE_FINDINGS.md). Before replacing the incumbent we
must know which engine better describes the responses production actually
received. This replays the real prod n-way sittings (extracted with only
session_id/trial_index/seg_id/task_k/pick/is_correct) through each candidate
engine and scores the ONE-STEP-AHEAD predictive log-likelihood of the
observed picks — the prequential log score, a proper scoring rule: before
each trial the engine's filtered posterior assigns a probability to the pick
that then arrives; the engine is scored on that probability and only then
told the answer.

Arms (identical prior, identical trials; they differ ONLY in the response
model, so the contrast isolates the object under arbitration):

- ``binary``            the shipping binary reduction y = (pick == asked),
  extended to the categorical channel by uniform allocation of its wrong-pick
  mass over the five distractors (the monitor's uniform alternative).
- ``prod_incumbent``    unfloored nine-draw static mixture — deployed today.
- ``floored_ensemble9`` the same nine draws under the 0.15 lapse floor.
- ``draw_latent_atoms17``  construction B on the 17-atom floored grid.
- ``draw_latent_nesting34`` construction B on the promotion-candidate grid:
  33 quantile atoms at the out-of-sample heterogeneity (tau = 0.3581) plus
  the lambda_d = 1 binary-nesting atom.

Configuration mirrors production: 1,200 particles, ESS 0.5, 30 MH steps,
proposal 2.38/sqrt(14), and the SERVED manifest corrL/corrT prior blocks
(.artifacts/prod_manifest_engine_config.json), not the harness identity
prior. Response axes are the RAW ``s_mean``/``s_sd`` staged bank axes — the
frame the artifact was fitted in (stage_real_artifact.py); no skill scaling
is applied anywhere (the misspec-monitor evidence-frame rule).

Spike trials (task_k = 0) are excluded from scoring and from updating: they
are binary in every arm, their axes are absent from the IIIC bank, and under
any of these engines they carry no information that differs across arms.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nway_protocol.qualification import IIIC_GROUP
from nway_protocol.reference import (
    Observation,
    binary_p_yes,
    ess,
    f1_probabilities_artifact,
    make_cloud,
    resample_and_rejuvenate,
    signal_z,
    update,
)

from .engine import (
    atom_posterior,
    make_draw_cloud,
    _observation_log_probability,
    resample_and_rejuvenate_draw,
    update_draw_latent,
)
from .harness import build_atoms
from .prod_incumbent_study import ARTIFACTS, FLOORED_ARM, INCUMBENT_ARM, load_draws

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = 68_000_000  # disjoint from 62.1M dev / 63.8M monitor / 65M locked / 66M head-to-head
N_PARTICLES = 1_200
ESS_FRACTION = 0.5
MH_STEPS = 30
PROPOSAL_SCALE = 2.38 / np.sqrt(14)

BINARY = "binary"
FLOORED = "floored_ensemble9"
ATOMS17 = "draw_latent_atoms17"
NESTING34 = "draw_latent_nesting34"
ARM_ORDER = (BINARY, "prod_incumbent", FLOORED, ATOMS17, NESTING34)


def load_trials(path: Path) -> dict[str, list[tuple[int, int, int]]]:
    """session_id -> ordered [(seg_id, asked_k, pick)] for IIIC trials only."""
    sessions: dict[str, list[tuple[int, int, int, int]]] = {}
    skipped = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            asked = int(row["task_k"])
            if asked not in IIIC_GROUP:
                continue
            pick = int(row["pick"])
            if pick not in IIIC_GROUP:
                skipped += 1
                continue
            sessions.setdefault(row["session_id"], []).append(
                (int(row["trial_index"]), int(row["seg_id"]), asked, pick),
            )
    if skipped:
        raise ValueError(f"{skipped} IIIC trials carry picks outside the group")
    return {
        session: [(seg, asked, pick) for _, seg, asked, pick in sorted(rows)]
        for session, rows in sessions.items()
    }


def load_bank(path: Path) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    seg_row: dict[int, int] = {}
    means, sds = [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            seg_row[int(float(row["seg_id"]))] = len(means)
            means.append([float(row[f"s_mean_{k}"]) for k in range(7)])
            sds.append([float(row[f"s_sd_{k}"]) for k in range(7)])
    return seg_row, np.asarray(means), np.asarray(sds)


def load_prior(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    return np.asarray(payload["corrT"]), np.asarray(payload["corrL"])


def _self_check(corr_t: np.ndarray, corr_l: np.ndarray) -> None:
    """The three predictive constructions must agree where they must nest."""
    rng = np.random.default_rng(7)
    cloud = make_cloud(64, corr_t, corr_l, rng)
    s_mean = rng.normal(size=7)
    s_sd = np.full(7, 0.1)
    draws = ((0.97, 0.15, 1.0),)
    static = f1_probabilities_artifact(
        cloud.t, cloud.l, s_mean, s_sd, 2, IIIC_GROUP, 1.0, 0.0, draws,
    )
    assert np.allclose(static.sum(axis=1), 1.0)
    dcloud = make_draw_cloud(64, corr_t, corr_l, draws, np.random.default_rng(7))
    lp = _observation_log_probability(
        dcloud,
        Observation(kind="categorical_f1", asked_k=2, segment_index=0,
                    raw_pick=3, group=IIIC_GROUP),
        s_mean, s_sd,
    )
    position = IIIC_GROUP.index(3)
    assert np.allclose(np.exp(lp), static[:, position], atol=1e-12)
    z = signal_z(cloud.l[:, 2], cloud.t[:, 2], s_mean[2], s_sd[2])
    own = binary_p_yes(z)
    binary_pick = float(cloud.w @ ((1.0 - own) / 5.0))
    total = binary_pick * 5 + float(cloud.w @ own)
    assert abs(total - 1.0) < 1e-12


def replay_session(job: tuple) -> dict:
    (session, trials, arm, draws, corr_t, corr_l, seed,
     s_means, s_sds, seg_row) = job
    cloud_rng = np.random.default_rng(seed)
    mh_rng = np.random.default_rng(seed + 50)
    is_draw_latent = arm in (ATOMS17, NESTING34)
    if is_draw_latent:
        dcloud = make_draw_cloud(N_PARTICLES, corr_t, corr_l, draws, cloud_rng)
        cloud = dcloud.cloud
    else:
        cloud = make_cloud(N_PARTICLES, corr_t, corr_l, cloud_rng)
    log_predictions: list[float] = []
    matched_asked: list[bool] = []
    rejuvenations = 0
    acceptances: list[float] = []
    for seg_id, asked_k, pick in trials:
        row = seg_row[seg_id]
        s_mean, s_sd = s_means[row], s_sds[row]
        position = IIIC_GROUP.index(pick)
        observation = Observation(
            kind="categorical_f1", asked_k=asked_k, segment_index=row,
            raw_pick=pick, group=IIIC_GROUP,
        )
        if arm == BINARY:
            z = signal_z(cloud.l[:, asked_k], cloud.t[:, asked_k],
                         s_mean[asked_k], s_sd[asked_k])
            own = binary_p_yes(z)
            predicted = (
                float(cloud.w @ own) if pick == asked_k
                else float(cloud.w @ (1.0 - own)) / 5.0
            )
            binary_observation = Observation(
                kind="binary", asked_k=asked_k, segment_index=row,
                raw_pick=pick, y=int(pick == asked_k),
            )
            update(cloud, binary_observation, s_mean, s_sd, 1.0, 0.0)
        elif is_draw_latent:
            lp = _observation_log_probability(dcloud, observation, s_mean, s_sd)
            predicted = float(cloud.w @ np.exp(lp))
            update_draw_latent(dcloud, observation, s_mean, s_sd)
        else:
            probabilities = f1_probabilities_artifact(
                cloud.t, cloud.l, s_mean, s_sd, asked_k, IIIC_GROUP,
                1.0, 0.0, draws,
            )
            predicted = float(cloud.w @ probabilities[:, position])
            update(cloud, observation, s_mean, s_sd, 1.0, 0.0, draws)
        log_predictions.append(float(np.log(max(predicted, np.finfo(float).tiny))))
        matched_asked.append(pick == asked_k)
        if ess(cloud) < ESS_FRACTION * N_PARTICLES:
            rejuvenations += 1
            if is_draw_latent:
                acceptance, _ = resample_and_rejuvenate_draw(
                    dcloud, s_means, s_sds, mh_rng, MH_STEPS, PROPOSAL_SCALE,
                )
            else:
                acceptance, _ = resample_and_rejuvenate(
                    cloud, s_means, s_sds, 1.0, 0.0, mh_rng, MH_STEPS,
                    PROPOSAL_SCALE, artifact_draws=draws if arm != BINARY else (),
                )
            acceptances.append(acceptance)
    log_array = np.asarray(log_predictions)
    matched = np.asarray(matched_asked)
    out = {
        "session": session,
        "arm": arm,
        "n_trials": len(trials),
        "sum_log_predictive": float(log_array.sum()),
        "mean_log_predictive": float(log_array.mean()),
        "sum_log_predictive_matched_asked": float(log_array[matched].sum()),
        "sum_log_predictive_other_pick": float(log_array[~matched].sum()),
        "n_matched_asked": int(matched.sum()),
        "rejuvenations": rejuvenations,
        "mean_acceptance": float(np.mean(acceptances)) if acceptances else None,
        "final_ess": float(ess(cloud)),
    }
    if is_draw_latent:
        mass = atom_posterior(dcloud)
        top = int(np.argmax(mass))
        out["atom_posterior"] = {
            "atoms_alive": int(np.count_nonzero(np.asarray(mass) > 1e-12)),
            "max_mass": float(max(mass)),
            "map_atom_beta": float(dcloud.atoms[top][0]),
            "map_atom_lapse": float(dcloud.atoms[top][1]),
        }
        if arm == NESTING34:
            out["atom_posterior"]["nesting_atom_mass"] = float(mass[-1])
    return out


def paired_block(rows_a: dict[str, float], rows_b: dict[str, float], rng) -> dict:
    sessions = sorted(set(rows_a) & set(rows_b))
    differences = np.asarray([rows_a[s] - rows_b[s] for s in sessions])
    resamples = rng.integers(0, len(sessions), size=(10_000, len(sessions)))
    bootstrap = differences[resamples].sum(axis=1)
    return {
        "sessions": len(sessions),
        "total_difference": float(differences.sum()),
        "mean_per_session": float(differences.mean()),
        "bootstrap95_total": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "sessions_won": int((differences > 0).sum()),
        "sessions_lost": int((differences < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path,
                        default=ROOT / ".artifacts/prod_nway_trials_20260801.csv")
    parser.add_argument("--bank", type=Path,
                        default=ROOT / ".artifacts/categorical_bank_axes.csv")
    parser.add_argument("--prior", type=Path,
                        default=ROOT / ".artifacts/prod_manifest_engine_config.json")
    parser.add_argument("--workers", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sessions = load_trials(args.trials)
    seg_row, s_means, s_sds = load_bank(args.bank)
    corr_t, corr_l = load_prior(args.prior)
    _self_check(corr_t, corr_l)

    arm_draws = {
        BINARY: (),
        "prod_incumbent": load_draws(ARTIFACTS[INCUMBENT_ARM]),
        FLOORED: load_draws(ARTIFACTS[FLOORED_ARM]),
        ATOMS17: build_atoms("atoms17"),
        NESTING34: build_atoms("nesting34"),
    }
    ordered_sessions = sorted(sessions)
    jobs = []
    for arm_index, arm in enumerate(ARM_ORDER):
        for session_index, session in enumerate(ordered_sessions):
            jobs.append((
                session, sessions[session], arm, arm_draws[arm], corr_t, corr_l,
                SEED_BASE + 100 * session_index + arm_index,
                s_means, s_sds, seg_row,
            ))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(replay_session, jobs))

    by_arm: dict[str, dict[str, float]] = {arm: {} for arm in ARM_ORDER}
    for row in rows:
        by_arm[row["arm"]][row["session"]] = row["sum_log_predictive"]
    totals = {
        arm: {
            "total_log_predictive": float(sum(by_arm[arm].values())),
            "mean_log_predictive_per_trial": float(
                sum(r["sum_log_predictive"] for r in rows if r["arm"] == arm)
                / sum(r["n_trials"] for r in rows if r["arm"] == arm)
            ),
        }
        for arm in ARM_ORDER
    }
    rng = np.random.default_rng(SEED_BASE + 999)
    contrasts = {
        "nesting34_minus_incumbent": paired_block(
            by_arm[NESTING34], by_arm["prod_incumbent"], rng),
        "atoms17_minus_incumbent": paired_block(
            by_arm[ATOMS17], by_arm["prod_incumbent"], rng),
        "floored_minus_incumbent": paired_block(
            by_arm[FLOORED], by_arm["prod_incumbent"], rng),
        "binary_minus_incumbent": paired_block(
            by_arm[BINARY], by_arm["prod_incumbent"], rng),
        "nesting34_minus_binary": paired_block(
            by_arm[NESTING34], by_arm[BINARY], rng),
        "atoms17_minus_binary": paired_block(
            by_arm[ATOMS17], by_arm[BINARY], rng),
    }
    gate = contrasts["nesting34_minus_incumbent"]
    verdict = {
        "gate_arm": NESTING34,
        "beats_incumbent_total": gate["total_difference"] > 0,
        "beats_incumbent_ci_excludes_zero": gate["bootstrap95_total"][0] > 0,
        "both_below_binary": (
            contrasts["nesting34_minus_binary"]["total_difference"] < 0
            and contrasts["binary_minus_incumbent"]["total_difference"] > 0
        ),
    }
    payload = {
        "schema_version": 1,
        "study": "real_response_arbitration_phase1c",
        "status": "real_prod_responses_no_pii",
        "config": {
            "n_particles": N_PARTICLES, "ess_fraction": ESS_FRACTION,
            "mh_steps": MH_STEPS, "seed_base": SEED_BASE,
            "prior": "served manifest corrL/corrT (v1.6-k7-35k)",
            "axes_frame": "raw s_mean/s_sd staged bank axes",
            "sessions": len(ordered_sessions),
            "trials_scored": sum(len(t) for t in sessions.values()),
        },
        "arm_totals": totals,
        "paired_contrasts": contrasts,
        "verdict": verdict,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm_totals": totals, "verdict": verdict,
                      "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
