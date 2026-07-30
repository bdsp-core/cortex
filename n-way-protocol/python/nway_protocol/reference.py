"""Numerically stable reference implementation of the conditional F1 link.

This module intentionally does not import the browser engine.  It is the
independent oracle used for TypeScript fixtures and statistical qualification.
The spike family remains binary; the IIIC family consumes the complete pick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr

BINARY_LAPSE = 0.025


@dataclass(frozen=True)
class Observation:
    kind: Literal["binary", "categorical_f1"]
    asked_k: int
    segment_index: int
    raw_pick: int
    y: int | None = None
    group: tuple[int, ...] = ()


@dataclass
class ParticleCloud:
    t: np.ndarray
    l: np.ndarray
    w: np.ndarray
    log_prior: np.ndarray
    log_lik: np.ndarray
    corr_t: np.ndarray
    corr_l: np.ndarray
    history: list[Observation] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.w.size)

    @property
    def k(self) -> int:
        return int(self.t.shape[1])


def signal_z(l: np.ndarray, t: np.ndarray, s: np.ndarray, s_sd: np.ndarray) -> np.ndarray:
    sensitivity = np.exp(l)
    z = sensitivity * (s + t)
    return z / np.sqrt(1.0 + np.square(sensitivity * s_sd))


def binary_p_yes(z: np.ndarray) -> np.ndarray:
    return BINARY_LAPSE + (1.0 - 2.0 * BINARY_LAPSE) * ndtr(z)


def log_p_binary(z: np.ndarray, y: int) -> np.ndarray:
    if y not in (0, 1):
        raise ValueError("binary y must be zero or one")
    cdf = np.log1p(-2.0 * BINARY_LAPSE) + log_ndtr(z if y else -z)
    return np.logaddexp(cdf, np.log(BINARY_LAPSE))


def f1_probabilities(
    t: np.ndarray,
    l: np.ndarray,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    asked_k: int,
    group: tuple[int, ...],
    beta: float,
    distractor_lapse: float,
) -> np.ndarray:
    """Return an (N, len(group)) response-probability matrix."""
    if asked_k not in group or len(group) < 3:
        raise ValueError("asked class must belong to a categorical group")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not 0 <= distractor_lapse <= 1:
        raise ValueError("distractor_lapse must be in [0,1]")
    t = np.atleast_2d(np.asarray(t, dtype=np.float64))
    l = np.atleast_2d(np.asarray(l, dtype=np.float64))
    s_mean = np.asarray(s_mean, dtype=np.float64)
    s_sd = np.asarray(s_sd, dtype=np.float64)
    z = signal_z(l, t, s_mean[None, :], s_sd[None, :])
    own = binary_p_yes(z[:, asked_k])
    distractors = tuple(k for k in group if k != asked_k)
    logits = beta * z[:, distractors]
    softmax = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    q = distractor_lapse / len(distractors) + (1.0 - distractor_lapse) * softmax
    output = np.zeros((t.shape[0], len(group)), dtype=np.float64)
    for position, outcome in enumerate(group):
        if outcome == asked_k:
            output[:, position] = own
        else:
            output[:, position] = (1.0 - own) * q[:, distractors.index(outcome)]
    return output


def f1_probabilities_artifact(
    t: np.ndarray,
    l: np.ndarray,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    asked_k: int,
    group: tuple[int, ...],
    beta: float,
    distractor_lapse: float,
    artifact_draws: tuple[tuple[float, float, float], ...] = (),
) -> np.ndarray:
    """Evaluate either the frozen plug-in artifact or an opt-in mixture."""
    if not artifact_draws:
        return f1_probabilities(
            t, l, s_mean, s_sd, asked_k, group, beta, distractor_lapse,
        )
    weights = np.asarray([draw[2] for draw in artifact_draws], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("artifact mixture weights are invalid")
    weights /= weights.sum()
    t = np.atleast_2d(np.asarray(t, dtype=np.float64))
    l = np.atleast_2d(np.asarray(l, dtype=np.float64))
    s_mean = np.asarray(s_mean, dtype=np.float64)
    s_sd = np.asarray(s_sd, dtype=np.float64)
    z = signal_z(l, t, s_mean[None, :], s_sd[None, :])
    own = binary_p_yes(z[:, asked_k])
    distractors = tuple(k for k in group if k != asked_k)
    q = np.zeros((t.shape[0], len(distractors)), dtype=np.float64)
    for weight, (draw_beta, draw_lapse, _) in zip(weights, artifact_draws, strict=True):
        if not np.isfinite(draw_beta) or draw_beta <= 0:
            raise ValueError("artifact mixture beta must be finite and positive")
        if not 0 <= draw_lapse <= 1:
            raise ValueError("artifact mixture lapse must be in [0,1]")
        logits = draw_beta * z[:, distractors]
        softmax = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        q += weight * (
            draw_lapse / len(distractors) + (1 - draw_lapse) * softmax
        )
    output = np.zeros((t.shape[0], len(group)), dtype=np.float64)
    for position, outcome in enumerate(group):
        if outcome == asked_k:
            output[:, position] = own
        else:
            output[:, position] = (1 - own) * q[:, distractors.index(outcome)]
    return output


def log_probability(
    cloud_t: np.ndarray,
    cloud_l: np.ndarray,
    observation: Observation,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    beta: float,
    distractor_lapse: float,
    artifact_draws: tuple[tuple[float, float, float], ...] = (),
) -> np.ndarray:
    z = signal_z(
        cloud_l[:, observation.asked_k], cloud_t[:, observation.asked_k],
        np.asarray(s_mean)[observation.asked_k],
        np.asarray(s_sd)[observation.asked_k],
    )
    if observation.kind == "binary":
        if observation.y not in (0, 1):
            raise ValueError("binary observation is missing y")
        return log_p_binary(z, int(observation.y))
    probabilities = f1_probabilities_artifact(
        cloud_t, cloud_l, s_mean, s_sd, observation.asked_k,
        observation.group, beta, distractor_lapse, artifact_draws,
    )
    try:
        pick_position = observation.group.index(observation.raw_pick)
    except ValueError as exc:
        raise ValueError("pick is outside the response group") from exc
    return np.log(probabilities[:, pick_position])


def _log_prior(t: np.ndarray, l: np.ndarray, corr_t: np.ndarray, corr_l: np.ndarray) -> np.ndarray:
    inv_t = np.linalg.inv(corr_t)
    inv_l = np.linalg.inv(corr_l)
    det_t = np.linalg.slogdet(corr_t)[1]
    det_l = np.linalg.slogdet(corr_l)[1]
    return (
        -0.5 * np.einsum("ni,ij,nj->n", t, inv_t, t) - 0.5 * det_t
        -0.5 * np.einsum("ni,ij,nj->n", l, inv_l, l) - 0.5 * det_l
    )


def make_cloud(
    n_particles: int,
    corr_t: np.ndarray,
    corr_l: np.ndarray,
    rng: np.random.Generator,
) -> ParticleCloud:
    k = int(corr_l.shape[0])
    t = rng.multivariate_normal(np.zeros(k), corr_t, size=n_particles)
    l = rng.multivariate_normal(np.zeros(k), corr_l, size=n_particles)
    return ParticleCloud(
        t=t,
        l=l,
        w=np.full(n_particles, 1.0 / n_particles),
        log_prior=_log_prior(t, l, corr_t, corr_l),
        log_lik=np.zeros(n_particles),
        corr_t=np.asarray(corr_t),
        corr_l=np.asarray(corr_l),
    )


def clone_cloud(cloud: ParticleCloud) -> ParticleCloud:
    return ParticleCloud(
        t=cloud.t.copy(), l=cloud.l.copy(), w=cloud.w.copy(),
        log_prior=cloud.log_prior.copy(), log_lik=cloud.log_lik.copy(),
        corr_t=cloud.corr_t, corr_l=cloud.corr_l,
        history=list(cloud.history),
    )


def update(
    cloud: ParticleCloud,
    observation: Observation,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    beta: float,
    distractor_lapse: float,
    artifact_draws: tuple[tuple[float, float, float], ...] = (),
) -> None:
    if np.any(~np.isfinite(cloud.w)) or np.any(cloud.w < 0) or cloud.w.sum() <= 0:
        raise ValueError("invalid pre-update weights")
    lp = log_probability(
        cloud.t, cloud.l, observation, s_mean, s_sd, beta, distractor_lapse,
        artifact_draws,
    )
    log_weights = np.where(cloud.w > 0, np.log(cloud.w), -np.inf) + lp
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise ValueError("posterior update has zero mass")
    next_weights = np.exp(log_weights - normalizer)
    next_log_lik = cloud.log_lik + lp
    if np.any(~np.isfinite(next_log_lik)):
        raise ValueError("posterior update has non-finite likelihood")
    cloud.w = next_weights
    cloud.log_lik = next_log_lik
    cloud.history.append(observation)


def ess(cloud: ParticleCloud) -> float:
    return float(1.0 / np.square(cloud.w).sum())


def history_log_likelihood(
    cloud: ParticleCloud,
    proposed_t: np.ndarray,
    proposed_l: np.ndarray,
    s_means: np.ndarray,
    s_sds: np.ndarray,
    beta: float,
    distractor_lapse: float,
    artifact_draws: tuple[tuple[float, float, float], ...] = (),
) -> np.ndarray:
    result = np.zeros(proposed_t.shape[0])
    for observation in cloud.history:
        result += log_probability(
            proposed_t, proposed_l, observation,
            s_means[observation.segment_index], s_sds[observation.segment_index],
            beta, distractor_lapse, artifact_draws,
        )
    return result


def resample_and_rejuvenate(
    cloud: ParticleCloud,
    s_means: np.ndarray,
    s_sds: np.ndarray,
    beta: float,
    distractor_lapse: float,
    rng: np.random.Generator,
    n_steps: int,
    proposal_scale: float,
    resampling_scheme: Literal[
        "multinomial", "stratified", "residual", "rqmc_sorted"
    ] = "multinomial",
    artifact_draws: tuple[tuple[float, float, float], ...] = (),
) -> tuple[float, float]:
    ancestors = resample_indices(
        cloud.w,
        rng,
        resampling_scheme,
        np.concatenate([cloud.t, cloud.l], axis=1),
    )
    distinct_fraction = float(np.unique(ancestors).size / cloud.n)
    cloud.t = cloud.t[ancestors].copy()
    cloud.l = cloud.l[ancestors].copy()
    cloud.log_prior = cloud.log_prior[ancestors].copy()
    cloud.log_lik = cloud.log_lik[ancestors].copy()
    cloud.w.fill(1.0 / cloud.n)
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
        proposed_lik = history_log_likelihood(
            cloud, proposed_t, proposed_l, s_means, s_sds,
            beta, distractor_lapse, artifact_draws,
        )
        log_alpha = proposed_prior + proposed_lik - cloud.log_prior - cloud.log_lik
        accepted = np.log(rng.random(cloud.n)) < log_alpha
        cloud.t[accepted] = proposed_t[accepted]
        cloud.l[accepted] = proposed_l[accepted]
        cloud.log_prior[accepted] = proposed_prior[accepted]
        cloud.log_lik[accepted] = proposed_lik[accepted]
        acceptance.append(float(accepted.mean()))
    return float(np.mean(acceptance) if acceptance else 0.0), distinct_fraction


def resample_indices(
    weights: np.ndarray,
    rng: np.random.Generator,
    scheme: Literal["multinomial", "stratified", "residual", "rqmc_sorted"],
    particles: np.ndarray | None = None,
) -> np.ndarray:
    """Research-capable resampling with multinomial as the frozen default."""
    weights = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("resampling weights are invalid")
    weights = weights / weights.sum()
    n = weights.size
    if scheme == "multinomial":
        # Preserve the baseline RNG operation exactly.
        return rng.choice(n, size=n, replace=True, p=weights)
    if scheme == "stratified":
        points = (np.arange(n) + rng.random(n)) / n
        return np.searchsorted(np.cumsum(weights), points, side="right")
    if scheme == "residual":
        deterministic = np.floor(n * weights).astype(int)
        fixed = np.repeat(np.arange(n), deterministic)
        remainder_n = n - fixed.size
        if remainder_n == 0:
            return fixed
        remainder = n * weights - deterministic
        remainder /= remainder.sum()
        random_part = rng.choice(n, size=remainder_n, replace=True, p=remainder)
        output = np.concatenate([fixed, random_part])
        rng.shuffle(output)
        return output
    if scheme == "rqmc_sorted":
        if particles is None or particles.shape[0] != n:
            raise ValueError("rqmc_sorted resampling requires one state row per weight")
        # A one-dimensional principal-component ordering is a deterministic
        # space-filling surrogate. Scrambled Sobol uniforms then replace IID
        # multinomial uniforms. This is an SQMC precursor, not a claim of a
        # complete Gerber-Chopin Hilbert-ordered SQMC implementation.
        centered = particles - particles.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(n - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0:
            direction = -direction
        order = np.argsort(centered @ direction, kind="stable")
        from scipy.stats import qmc
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        power = int(np.ceil(np.log2(max(n, 1))))
        points = np.sort(
            qmc.Sobol(d=1, scramble=True, seed=seed).random_base2(power)[:n, 0]
        )
        selected = np.searchsorted(np.cumsum(weights[order]), points, side="right")
        return order[np.minimum(selected, n - 1)]
    raise ValueError(f"unknown resampling scheme: {scheme}")


def posterior_moments(cloud: ParticleCloud) -> dict[str, np.ndarray]:
    t_mean = cloud.w @ cloud.t
    l_mean = cloud.w @ cloud.l
    t_sd = np.sqrt(np.maximum(cloud.w @ np.square(cloud.t) - np.square(t_mean), 0))
    l_sd = np.sqrt(np.maximum(cloud.w @ np.square(cloud.l) - np.square(l_mean), 0))
    return {"t_mean": t_mean, "l_mean": l_mean, "t_sd": t_sd, "l_sd": l_sd}


def expected_loss(
    cloud: ParticleCloud,
    asked_k: int,
    s_mean: np.ndarray,
    s_sd: np.ndarray,
    group: tuple[int, ...] | None,
    beta: float,
    distractor_lapse: float,
    artifact_draws: tuple[tuple[float, float, float], ...] = (),
) -> float:
    moments = posterior_moments(cloud)
    baseline = float(np.square(moments["t_sd"]).sum() + np.square(moments["l_sd"]).sum())
    if group is None:
        z = signal_z(
            cloud.l[:, asked_k], cloud.t[:, asked_k],
            s_mean[asked_k], s_sd[asked_k],
        )
        yes = binary_p_yes(z)
        probabilities = np.column_stack([yes, 1.0 - yes])
    else:
        probabilities = f1_probabilities_artifact(
            cloud.t, cloud.l, s_mean, s_sd, asked_k, group,
            beta, distractor_lapse, artifact_draws,
        )
    joint = cloud.w[:, None] * probabilities
    response_mass = joint.sum(axis=0)
    between = 0.0
    theta = np.concatenate([cloud.t, cloud.l], axis=1)
    overall = np.concatenate([moments["t_mean"], moments["l_mean"]])
    for r, mass in enumerate(response_mass):
        if mass <= 1e-15:
            continue
        conditional = joint[:, r] @ theta / mass
        between += float(mass * np.square(conditional - overall).sum())
    return baseline - between


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(ordered_values[min(np.searchsorted(cumulative, q), len(values) - 1)])
