"""Hierarchical SMC that *learns* r from the data via discrete grid model averaging.

Approach:
  - r ~ uniform over a fixed discrete grid r_grid (e.g. {0, 0.1, ..., 1.0}).
  - For each r in the grid, run a parallel SMC with that assumed r.
  - Track log marginal likelihood log p(y_{1:t} | r_g) per grid point by
    accumulating log p(y_t | r_g, history) at each step, where
        p(y_t | r_g, history) = sum_i w_i^{t-1} * lik_i.
  - Posterior over r: softmax(log_marg_lik).
  - Mixture posterior on (t_k, l_k) := concat all clouds with weight
        w_i (mixture) = (posterior_r_g) * (w_i within-grid).
  - Item selection: weighted sum of per-grid expected-loss vectors.
  - Stopping: max CI half-width of the mixture posterior across (t1,l1,t2,l2) < delta.

This is a clean, low-risk way to add r as an unknown.
"""
import numpy as np
from scipy.stats import norm

from core import (
    sample_prior_hier_rho, response_prob, simulate_response,
    weighted_mean_var, ess, weighted_quantile, _kernel_smooth,
    SIGNAL_GRID, expected_loss_hier_vec, posterior_ci_hier_4d,
    update_hier, resample_jitter_hier,
)

DEFAULT_R_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


def init_learnr(N_per, r_grid, rho_assumed, rng):
    """Initialise one SMC per r value. N_per particles each."""
    states = []
    for r in r_grid:
        sub_rng = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
        states.append(sample_prior_hier_rho(N_per, float(r), rho_assumed, sub_rng))
    log_marg = np.zeros(len(r_grid))  # log p(y_{1:t} | r_g), unnormalized in r
    return states, log_marg


def _likelihood_and_update_one(particles, k, s, y):
    """Reweight one SMC; return the marginal predictive p(y | history) used for log_marg."""
    p = response_prob(s, particles["t"][:, k], particles["l"][:, k])
    lik = p if y == 1 else (1.0 - p)
    p_y = float((particles["w"] * lik).sum())
    p_y = max(p_y, 1e-12)
    w = particles["w"] * lik
    particles["w"] = w / w.sum()
    return p_y


def mixture_weights(log_marg):
    """Posterior weights over the r_grid (uniform prior assumed)."""
    m = log_marg - log_marg.max()
    w = np.exp(m)
    return w / w.sum()


def mixture_posterior_ci(states, log_marg, alpha=0.05):
    """CI half-widths of the mixture posterior on (t1, l1, t2, l2)."""
    grid_w = mixture_weights(log_marg)
    arrs = []
    weights = []
    for sg, gw in zip(states, grid_w):
        arrs.append(np.column_stack([sg["t"][:, 0], sg["l"][:, 0],
                                     sg["t"][:, 1], sg["l"][:, 1]]))
        weights.append(gw * sg["w"])
    big = np.concatenate(arrs, axis=0)
    big_w = np.concatenate(weights)
    big_w /= big_w.sum()
    hw = []
    for j in range(4):
        lo = weighted_quantile(big[:, j], big_w, alpha / 2)
        hi = weighted_quantile(big[:, j], big_w, 1 - alpha / 2)
        hw.append((hi - lo) / 2)
    return np.array(hw)


def mixture_posterior_means(states, log_marg):
    grid_w = mixture_weights(log_marg)
    arrs = []
    weights = []
    for sg, gw in zip(states, grid_w):
        arrs.append(np.column_stack([sg["t"][:, 0], sg["l"][:, 0],
                                     sg["t"][:, 1], sg["l"][:, 1]]))
        weights.append(gw * sg["w"])
    big = np.concatenate(arrs, axis=0)
    big_w = np.concatenate(weights)
    big_w /= big_w.sum()
    return (big * big_w[:, None]).sum(axis=0)


def expected_loss_mixture(states, log_marg, k, signals):
    """Weighted sum of per-grid expected losses."""
    grid_w = mixture_weights(log_marg)
    losses = np.zeros_like(signals, dtype=float)
    for sg, gw in zip(states, grid_w):
        if gw < 1e-6:
            continue
        losses += gw * expected_loss_hier_vec(sg, k, signals)
    return losses


def choose_item_mixture(states, log_marg):
    losses = np.stack([expected_loss_mixture(states, log_marg, k, SIGNAL_GRID)
                       for k in range(2)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


def run_session_learnr(true_params, r_grid=DEFAULT_R_GRID, N_per=500, rho_assumed=0.0,
                       max_q=400, delta=0.4, seed=0):
    """One session of the learn-r hierarchical SMC. Brute force not handled here."""
    rng = np.random.default_rng(seed)
    states, log_marg = init_learnr(N_per, r_grid, rho_assumed, rng)

    n_q = 0
    for q in range(max_q):
        k, s = choose_item_mixture(states, log_marg)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        for g, sg in enumerate(states):
            p_y = _likelihood_and_update_one(sg, k, s, y)
            log_marg[g] += np.log(p_y)
            if ess(sg["w"]) < N_per / 4:
                states[g] = resample_jitter_hier(sg, rng)
        n_q += 1
        cis = mixture_posterior_ci(states, log_marg)
        if np.nanmax(cis) < delta:
            break

    grid_w = mixture_weights(log_marg)
    r_post_mean = float((grid_w * r_grid).sum())
    r_post_mode = float(r_grid[int(np.argmax(grid_w))])
    return {
        "n_questions": n_q,
        "stopped_early": n_q < max_q,
        "r_posterior": grid_w,
        "r_posterior_mean": r_post_mean,
        "r_posterior_mode": r_post_mode,
        "final_means": mixture_posterior_means(states, log_marg),
    }
