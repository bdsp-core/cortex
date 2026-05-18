"""K-task version of the learn-r SMC (used by exp4_K48)."""
import numpy as np

from core import (
    response_prob, simulate_response, weighted_quantile, ess, _kernel_smooth,
    SIGNAL_GRID,
)
from core_K import (
    sample_prior_hier_K, posterior_ci_hier_K,
    expected_loss_hier_vec_K, resample_jitter_hier_K,
)

DEFAULT_R_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


def init_learnr_K(N_per, K, r_grid, rho_assumed, rng):
    states = []
    for r in r_grid:
        sub_rng = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
        states.append(sample_prior_hier_K(N_per, K, float(r), rho_assumed, sub_rng))
    log_marg = np.zeros(len(r_grid))
    return states, log_marg


def _likelihood_and_update_one_K(particles, k, s, y):
    p = response_prob(s, particles["t"][:, k], particles["l"][:, k])
    lik = p if y == 1 else (1.0 - p)
    p_y = float((particles["w"] * lik).sum())
    p_y = max(p_y, 1e-12)
    w = particles["w"] * lik
    particles["w"] = w / w.sum()
    return p_y


def mixture_weights(log_marg, prior_weights=None):
    """Posterior weights over the r_grid.

    prior_weights: optional non-uniform prior over r_grid (shape == log_marg).
                   If None, uniform.
    """
    m = log_marg - log_marg.max()
    w = np.exp(m)
    if prior_weights is not None:
        w = w * prior_weights
    return w / w.sum()


def mixture_posterior_ci_K(states, log_marg, alpha=0.05, prior_weights=None):
    grid_w = mixture_weights(log_marg, prior_weights)
    K = states[0]["t"].shape[1]
    arrs = []
    weights = []
    for sg, gw in zip(states, grid_w):
        cols = []
        for k in range(K):
            cols.append(sg["t"][:, k])
            cols.append(sg["l"][:, k])
        arrs.append(np.column_stack(cols))
        weights.append(gw * sg["w"])
    big = np.concatenate(arrs, axis=0)
    big_w = np.concatenate(weights)
    big_w /= big_w.sum()
    hw = []
    los = []
    his = []
    for j in range(2 * K):
        lo = weighted_quantile(big[:, j], big_w, alpha / 2)
        hi = weighted_quantile(big[:, j], big_w, 1 - alpha / 2)
        hw.append((hi - lo) / 2)
        los.append(lo)
        his.append(hi)
    return np.array(hw), np.array(los), np.array(his)


def mixture_posterior_means_K(states, log_marg, prior_weights=None):
    grid_w = mixture_weights(log_marg, prior_weights)
    K = states[0]["t"].shape[1]
    arrs = []
    weights = []
    for sg, gw in zip(states, grid_w):
        cols = []
        for k in range(K):
            cols.append(sg["t"][:, k])
            cols.append(sg["l"][:, k])
        arrs.append(np.column_stack(cols))
        weights.append(gw * sg["w"])
    big = np.concatenate(arrs, axis=0)
    big_w = np.concatenate(weights)
    big_w /= big_w.sum()
    return (big * big_w[:, None]).sum(axis=0)


def expected_loss_mixture_K(states, log_marg, k, signals, prior_weights=None):
    grid_w = mixture_weights(log_marg, prior_weights)
    losses = np.zeros_like(signals, dtype=float)
    for sg, gw in zip(states, grid_w):
        if gw < 1e-6:
            continue
        losses += gw * expected_loss_hier_vec_K(sg, k, signals)
    return losses


def choose_item_mixture_K(states, log_marg, prior_weights=None):
    K = states[0]["t"].shape[1]
    losses = np.stack([expected_loss_mixture_K(states, log_marg, k, SIGNAL_GRID,
                                                prior_weights)
                       for k in range(K)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


def run_session_learnr_K(true_params, K, r_grid=DEFAULT_R_GRID, N_per=400,
                         rho_assumed=0.0, max_q=400, delta=0.4, seed=0,
                         prior_weights=None, update_r=True):
    """K-task model-averaging hierarchical SMC.

    update_r: if True, marginal-likelihood updates the posterior over r (this
              is the original "learn r" behaviour). If False, r-weights stay
              fixed at `prior_weights` (useful for exp6's robust-prior design).
    """
    rng = np.random.default_rng(seed)
    states, log_marg = init_learnr_K(N_per, K, r_grid, rho_assumed, rng)

    n_q = 0
    for q in range(max_q):
        k, s = choose_item_mixture_K(states, log_marg, prior_weights)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        for g, sg in enumerate(states):
            p_y = _likelihood_and_update_one_K(sg, k, s, y)
            if update_r:
                log_marg[g] += np.log(p_y)
            if ess(sg["w"]) < N_per / 4:
                states[g] = resample_jitter_hier_K(sg, rng)
        n_q += 1
        cis, _, _ = mixture_posterior_ci_K(states, log_marg, prior_weights=prior_weights)
        if np.nanmax(cis) < delta:
            break

    grid_w = mixture_weights(log_marg, prior_weights)
    r_post_mean = float((grid_w * r_grid).sum())
    r_post_mode = float(r_grid[int(np.argmax(grid_w))])
    _, los, his = mixture_posterior_ci_K(states, log_marg, prior_weights=prior_weights)
    return {
        "n_questions": n_q,
        "stopped_early": n_q < max_q,
        "r_posterior": grid_w,
        "r_posterior_mean": r_post_mean,
        "r_posterior_mode": r_post_mode,
        "final_means": mixture_posterior_means_K(states, log_marg, prior_weights),
        "final_lo": los,
        "final_hi": his,
    }
