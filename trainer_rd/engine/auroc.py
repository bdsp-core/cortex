"""Minimal vendored stand-in for the main repo's `auroc` module (Step 0).

Provides exactly the three names `core_mcmc_general.py` imports:
    auroc_from_l, auroc_quantiles_from_particles_hier,
    auroc_quantiles_from_particles_brute

AUROC(ℓ) = Φ( √2 / √(exp(−2ℓ) + 1) )   (HOW_THE_TEST_WORKS.md §1.1)

Monotone in ℓ only — bias does not affect discrimination. ℓ = 0 ⇒ AUROC ≈ 0.8413.
When porting back to the main repo, the repo's own module wins; this file exists
so the scratch prototypes can import the engine unchanged.
"""
import numpy as np
from scipy.stats import norm


def auroc_from_l(l):
    """AUROC as a function of log-skill ℓ. Vectorized; returns shape of l."""
    l = np.asarray(l, dtype=np.float64)
    return norm.cdf(np.sqrt(2.0) / np.sqrt(np.exp(-2.0 * l) + 1.0))


def _wquantile(values, weights, q):
    """Weighted quantile via linear interpolation on the weighted CDF.

    Same construction as core_mcmc_general._wquantile_col (kept consistent so
    AUROC CIs and posterior-summary CIs are computed identically).
    """
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    cw = cw / cw[-1]
    return float(np.interp(q, cw, values[order]))


def auroc_quantiles_from_particles_hier(state, alphas=(0.025, 0.975)):
    """Per-task weighted quantiles of AUROC(ℓ_k) from the joint cloud.

    state: dict with "l" (N, K) and "w" (N,). Returns array (K, len(alphas)),
    so callers can do `q[:, 0], q[:, 1]` (see core_mcmc_general._auroc_ci).
    """
    w = np.asarray(state["w"], dtype=np.float64)
    w = w / w.sum()
    A = auroc_from_l(state["l"])                     # (N, K)
    K = A.shape[1]
    out = np.empty((K, len(alphas)))
    for k in range(K):
        for j, a in enumerate(alphas):
            out[k, j] = _wquantile(A[:, k], w, float(a))
    return out


def auroc_quantiles_from_particles_brute(state, alphas=(0.025, 0.975)):
    """Brute-state variant; same (N, K) cloud layout, same computation."""
    return auroc_quantiles_from_particles_hier(state, alphas=alphas)
