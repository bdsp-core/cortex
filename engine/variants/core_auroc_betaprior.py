"""AUROC-aware Beta(κ)-prior-on-r SMC for K=2.

Mirrors core_learnr.run_session_learnr but:
1. r-grid weights stay equal to the supplied prior (no marginal-lik updates).
2. Stopping is on AUROC posterior halfwidth.
3. Trajectory of mixture AUROC CI is logged.
"""
import numpy as np
from scipy.stats import beta as beta_dist

from core import (
    sample_prior_hier_rho, response_prob, simulate_response,
    weighted_quantile, ess, resample_jitter_hier,
    expected_loss_hier_vec, SIGNAL_GRID,
)
from auroc import auroc_from_l

DEFAULT_R_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


def beta_prior_weights(r_grid, mean, kappa):
    """Discretized Beta(αβ) prior weights over r_grid with α+β = κ."""
    a = mean * kappa
    b = (1 - mean) * kappa
    eps = 1e-3
    rg = np.clip(r_grid, eps, 1 - eps)
    dens = beta_dist.pdf(rg, a, b)
    return dens / dens.sum()


def init_states(N_per, r_grid, rho_assumed, rng):
    return [sample_prior_hier_rho(N_per, float(r), rho_assumed,
             np.random.default_rng(rng.integers(0, 2**31-1)))
            for r in r_grid]


def reweight_one(particles, k, s, y):
    p = response_prob(s, particles["t"][:, k], particles["l"][:, k])
    lik = p if y == 1 else (1.0 - p)
    w = particles["w"] * lik
    particles["w"] = w / w.sum()


def mixture_auroc_ci(states, prior_w, alpha=0.05):
    """Returns lo, hi arrays of length K (=2). Mixture across r-grid."""
    K = states[0]["t"].shape[1]
    los = np.zeros(K); his = np.zeros(K)
    for k in range(K):
        # collect (auc, weight) pairs across grid
        auc_all = []
        w_all = []
        for sg, gw in zip(states, prior_w):
            auc = auroc_from_l(sg["l"][:, k])
            auc_all.append(auc)
            w_all.append(gw * (sg["w"] / sg["w"].sum()))
        auc_all = np.concatenate(auc_all)
        w_all = np.concatenate(w_all)
        w_all /= w_all.sum()
        los[k] = weighted_quantile(auc_all, w_all, alpha / 2)
        his[k] = weighted_quantile(auc_all, w_all, 1 - alpha / 2)
    return los, his


def expected_loss_mixture(states, prior_w, k, signals):
    losses = np.zeros_like(signals, dtype=float)
    for sg, gw in zip(states, prior_w):
        if gw < 1e-6:
            continue
        losses += gw * expected_loss_hier_vec(sg, k, signals)
    return losses


def choose_item_mixture(states, prior_w):
    losses = np.stack([expected_loss_mixture(states, prior_w, k, SIGNAL_GRID)
                       for k in range(2)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


def run_session_auroc_betaprior(true_params, prior_w, r_grid=DEFAULT_R_GRID,
                                 N_per=400, max_q=400, delta_auroc=0.05,
                                 seed=0, run_until_max=False, log_trajectory=False,
                                 alpha=0.05):
    rng = np.random.default_rng(seed)
    K = len(true_params) // 2
    states = init_states(N_per, r_grid, 0.0, rng)

    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    lo, hi = mixture_auroc_ci(states, prior_w, alpha)
    if log_trajectory:
        los_t, his_t = [lo.copy()], [hi.copy()]
    n_q = 0
    stop_step = None
    for q in range(max_q):
        k, s = choose_item_mixture(states, prior_w)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        for sg in states:
            reweight_one(sg, k, s, y)
            if ess(sg["w"]) < N_per / 4:
                # find this entry in states and reassign
                pass
        # actually do resample step properly:
        for gi, sg in enumerate(states):
            if ess(sg["w"]) < N_per / 4:
                states[gi] = resample_jitter_hier(sg, rng)
        n_q += 1
        lo, hi = mixture_auroc_ci(states, prior_w, alpha)
        if log_trajectory:
            los_t.append(lo.copy())
            his_t.append(hi.copy())
        if stop_step is None:
            hw = (hi - lo) / 2.0
            if hw.max() < delta_auroc:
                stop_step = n_q
                if not run_until_max:
                    break

    out = {
        "n_questions": stop_step if stop_step is not None else n_q,
        "stopped_early": stop_step is not None,
        "true_auroc": true_auroc,
        "final_lo": lo,
        "final_hi": hi,
    }
    if log_trajectory:
        out["lo_traj"] = np.array(los_t)
        out["hi_traj"] = np.array(his_t)
    return out
