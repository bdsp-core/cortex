"""AUROC machinery.

The single-task AUROC formula (per /Users/mwestover/GithubRepos/ideal-test
src/viz_skill_threshold.py:39-42 and Agent report):

    AUROC = Phi( sqrt(2) * mu / sqrt(sigma^2 + sigma_s^2) )

where sigma is the internal-noise parameter, mu is the reference signal mean (=1),
and sigma_s is the reference signal spread (=1, probit scale).

In our parameterization Y ~ Bernoulli(Phi(exp(l)(s + t))) we have sigma = exp(-l),
so:

    AUROC(l) = Phi( sqrt(2) / sqrt(exp(-2 l) + 1) )

Bias t does NOT enter the formula — AUROC is a discrimination measure independent
of the operating-point bias.

Convention throughout this module:
- "halfwidth" = (q97.5 - q2.5) / 2 of the weighted posterior on AUROC for one task.
- "stopping criterion" = max_k halfwidth_k < delta_target.  No relative scaling —
  with AUROC bounded on [0.5, 1] for a skilled rater (l > 0), a fixed absolute
  threshold like 0.05 corresponds to "5% absolute AUROC error".
"""
import numpy as np
from scipy.stats import norm

MU_S = 1.0          # reference signal mean (single-task convention)
SIGMA_S = 1.0       # reference signal spread


def auroc_from_l(l):
    """AUROC as a function of log-skill l. Vectorized."""
    sigma = np.exp(-l)
    u = np.sqrt(2.0) * MU_S / np.sqrt(sigma ** 2 + SIGMA_S ** 2)
    return norm.cdf(u)


def auroc_quantiles_from_particles_hier(particles, alphas=(0.025, 0.5, 0.975)):
    """Weighted quantiles of AUROC posterior per task, for the hier 4D-cloud state.

    Returns (K, len(alphas)) array.
    """
    K = particles["l"].shape[1]
    w = particles["w"] / particles["w"].sum()
    out = np.zeros((K, len(alphas)))
    for k in range(K):
        auc = auroc_from_l(particles["l"][:, k])
        idx = np.argsort(auc)
        cdf = np.cumsum(w[idx])
        cdf /= cdf[-1]
        for j, a in enumerate(alphas):
            out[k, j] = float(np.interp(a, cdf, auc[idx]))
    return out


def auroc_quantiles_from_particles_brute(brute, alphas=(0.025, 0.5, 0.975)):
    """Same, for the brute K-clouds-list state."""
    K = len(brute)
    out = np.zeros((K, len(alphas)))
    for k in range(K):
        w = brute[k]["w"] / brute[k]["w"].sum()
        auc = auroc_from_l(brute[k]["l"])
        idx = np.argsort(auc)
        cdf = np.cumsum(w[idx])
        cdf /= cdf[-1]
        for j, a in enumerate(alphas):
            out[k, j] = float(np.interp(a, cdf, auc[idx]))
    return out


def auroc_mean_from_particles_hier(particles):
    K = particles["l"].shape[1]
    w = particles["w"] / particles["w"].sum()
    return np.array([float((w * auroc_from_l(particles["l"][:, k])).sum())
                     for k in range(K)])


def auroc_mean_from_particles_brute(brute):
    means = []
    for pset in brute:
        w = pset["w"] / pset["w"].sum()
        means.append(float((w * auroc_from_l(pset["l"])).sum()))
    return np.array(means)


def auroc_halfwidths_hier(particles, alpha=0.05):
    q = auroc_quantiles_from_particles_hier(particles,
                                             alphas=(alpha / 2, 1 - alpha / 2))
    return (q[:, 1] - q[:, 0]) / 2.0


def auroc_halfwidths_brute(brute, alpha=0.05):
    q = auroc_quantiles_from_particles_brute(brute,
                                              alphas=(alpha / 2, 1 - alpha / 2))
    return (q[:, 1] - q[:, 0]) / 2.0


def inflate_ci(lo, hi, c):
    """Inflate a CI symmetrically around its center by factor c >= 1.

    Used to apply a post-hoc inflation factor on a saved CI trajectory so we can
    determine when an inflated CI would stop.
    """
    mid = (lo + hi) / 2.0
    hw = (hi - lo) / 2.0
    return mid - c * hw, mid + c * hw
