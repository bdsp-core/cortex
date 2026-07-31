"""Draw-latent particle engine (construction B, oracle variant).

Each particle carries an artifact-atom index in addition to (t, l): the
ensemble becomes a proper discrete prior over (beta, lambda_d) that the
session's own responses resolve, instead of a fixed-weight blur applied
to every observation. Atom assignments ride resampling as lineage;
Metropolis-Hastings rejuvenation moves (t, l) conditional on each
particle's atom, which leaves the joint target invariant. With a single
atom the engine reduces exactly to the shipping mixture path (tested).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nway_protocol.reference import (
    Observation,
    ParticleCloud,
    _log_prior,
    f1_probabilities,
    log_probability,
    make_cloud,
    resample_indices,
)


@dataclass
class DrawCloud:
    cloud: ParticleCloud
    atoms: tuple[tuple[float, float, float], ...]
    draw: np.ndarray
    history: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.cloud.t.shape[0])


def make_draw_cloud(
    n_particles: int,
    corr_t: np.ndarray,
    corr_l: np.ndarray,
    atoms: tuple[tuple[float, float, float], ...],
    rng: np.random.Generator,
) -> DrawCloud:
    cloud = make_cloud(n_particles, corr_t, corr_l, rng)
    weights = np.asarray([atom[2] for atom in atoms], dtype=float)
    weights = weights / weights.sum()
    draw = rng.choice(len(atoms), size=n_particles, p=weights)
    return DrawCloud(cloud=cloud, atoms=atoms, draw=draw)


def _categorical_log_probability(
    dcloud: DrawCloud,
    observation: Observation,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
) -> np.ndarray:
    cloud = dcloud.cloud
    lp = np.empty(dcloud.n)
    pick_position = observation.group.index(observation.raw_pick)
    for atom_index in np.unique(dcloud.draw):
        subset = np.where(dcloud.draw == atom_index)[0]
        beta, lapse, _ = dcloud.atoms[int(atom_index)]
        probabilities = f1_probabilities(
            cloud.t[subset], cloud.l[subset], s_mean, s_sd,
            observation.asked_k, observation.group, beta, lapse,
        )
        lp[subset] = np.log(probabilities[:, pick_position])
    return lp


def _observation_log_probability(
    dcloud: DrawCloud,
    observation: Observation,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    t: np.ndarray | None = None,
    l: np.ndarray | None = None,
) -> np.ndarray:
    """Per-particle log probability under each particle's own atom.

    ``t``/``l`` override the cloud state for rejuvenation proposals; the
    atom assignment is always the cloud's current lineage.
    """
    cloud = dcloud.cloud
    t = cloud.t if t is None else t
    l = cloud.l if l is None else l
    if observation.kind == "binary":
        return log_probability(t, l, observation, s_mean, s_sd, 1.0, 0.0)
    lp = np.empty(t.shape[0])
    pick_position = observation.group.index(observation.raw_pick)
    for atom_index in np.unique(dcloud.draw):
        subset = np.where(dcloud.draw == atom_index)[0]
        beta, lapse, _ = dcloud.atoms[int(atom_index)]
        probabilities = f1_probabilities(
            t[subset], l[subset], s_mean, s_sd,
            observation.asked_k, observation.group, beta, lapse,
        )
        lp[subset] = np.log(probabilities[:, pick_position])
    return lp


def update_draw_latent(
    dcloud: DrawCloud,
    observation: Observation,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
) -> None:
    cloud = dcloud.cloud
    if np.any(~np.isfinite(cloud.w)) or np.any(cloud.w < 0) or cloud.w.sum() <= 0:
        raise ValueError("invalid pre-update weights")
    lp = _observation_log_probability(dcloud, observation, s_mean, s_sd)
    from scipy.special import logsumexp

    log_weights = np.where(cloud.w > 0, np.log(cloud.w), -np.inf) + lp
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise ValueError("posterior update has zero mass")
    cloud.w = np.exp(log_weights - normalizer)
    cloud.log_lik = cloud.log_lik + lp
    dcloud.history.append(observation)


def resample_and_rejuvenate_draw(
    dcloud: DrawCloud,
    s_means: np.ndarray,
    s_sds: np.ndarray,
    rng: np.random.Generator,
    n_steps: int,
    proposal_scale: float,
) -> tuple[float, float]:
    cloud = dcloud.cloud
    ancestors = resample_indices(
        cloud.w, rng, "multinomial",
        np.concatenate([cloud.t, cloud.l], axis=1),
    )
    distinct_fraction = float(np.unique(ancestors).size / dcloud.n)
    cloud.t = cloud.t[ancestors].copy()
    cloud.l = cloud.l[ancestors].copy()
    cloud.log_prior = cloud.log_prior[ancestors].copy()
    cloud.log_lik = cloud.log_lik[ancestors].copy()
    dcloud.draw = dcloud.draw[ancestors].copy()
    cloud.w.fill(1.0 / dcloud.n)
    acceptance = []
    for _ in range(n_steps):
        theta = np.concatenate([cloud.t, cloud.l], axis=1)
        covariance = np.cov(theta, rowvar=False) + np.eye(2 * cloud.k) * 1e-6
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(covariance)
            factor = vectors @ np.diag(np.sqrt(np.maximum(values, 1e-6)))
        proposal = theta + proposal_scale * (rng.normal(size=theta.shape) @ factor.T)
        proposed_t, proposed_l = proposal[:, :cloud.k], proposal[:, cloud.k:]
        proposed_prior = _log_prior(proposed_t, proposed_l, cloud.corr_t, cloud.corr_l)
        proposed_lik = np.zeros(dcloud.n)
        for observation in dcloud.history:
            proposed_lik += _observation_log_probability(
                dcloud, observation,
                s_means[observation.segment_index], s_sds[observation.segment_index],
                proposed_t, proposed_l,
            )
        log_alpha = proposed_prior + proposed_lik - cloud.log_prior - cloud.log_lik
        accepted = np.log(rng.random(dcloud.n)) < log_alpha
        cloud.t[accepted] = proposed_t[accepted]
        cloud.l[accepted] = proposed_l[accepted]
        cloud.log_prior[accepted] = proposed_prior[accepted]
        cloud.log_lik[accepted] = proposed_lik[accepted]
        acceptance.append(float(accepted.mean()))
    return float(np.mean(acceptance) if acceptance else 0.0), distinct_fraction


def atom_posterior(dcloud: DrawCloud) -> list[float]:
    """Posterior mass per atom — the session's inferred draw distribution."""
    mass = np.zeros(len(dcloud.atoms))
    np.add.at(mass, dcloud.draw, dcloud.cloud.w)
    return [float(value) for value in mass]
