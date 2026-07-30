"""Opt-in RQMC initialization and resampling helpers for isolated R&D."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .reference import ParticleCloud, _log_prior


def make_rqmc_cloud(
    n_particles: int,
    corr_t: np.ndarray,
    corr_l: np.ndarray,
    seed: int,
) -> ParticleCloud:
    """Create a scrambled-Sobol prior cloud.

    Together with ``rqmc_sorted`` resampling this is a bounded hybrid
    experiment. MH mutation remains randomized, so it is not labeled a full
    SQMC implementation or eligible for production.
    """
    k = int(corr_l.shape[0])
    power = int(np.ceil(np.log2(max(n_particles, 1))))
    uniforms = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random_base2(power)[:n_particles]
    normals = ndtri(np.clip(uniforms, 1e-12, 1 - 1e-12))
    t = normals[:, :k] @ np.linalg.cholesky(corr_t).T
    l = normals[:, k:] @ np.linalg.cholesky(corr_l).T
    return ParticleCloud(
        t=t,
        l=l,
        w=np.full(n_particles, 1 / n_particles),
        log_prior=_log_prior(t, l, corr_t, corr_l),
        log_lik=np.zeros(n_particles),
        corr_t=np.asarray(corr_t),
        corr_l=np.asarray(corr_l),
    )
