"""SMC sampler with MCMC rejuvenation, Beta-prior over r, AUROC-based stopping.

K=2-task. Maintains an SMC per r-grid value, with prior weights fixed (no
marginal-likelihood update — at K=2 the data doesn't identify r anyway).
"""
import numpy as np
from scipy.stats import beta as beta_dist

from core_mcmc import (
    make_state_hier, update, ess,
    resample_and_rejuvenate, _expected_loss_vec,
    SIGNAL_GRID, simulate_response,
)
from auroc import auroc_from_l
from core import weighted_quantile


DEFAULT_R_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


def beta_prior_weights(r_grid, mean, kappa):
    a = mean * kappa
    b = (1 - mean) * kappa
    eps = 1e-3
    rg = np.clip(r_grid, eps, 1 - eps)
    dens = beta_dist.pdf(rg, a, b)
    return dens / dens.sum()


def init_states_mcmc_grid(N_per, K, r_grid, rng):
    return [make_state_hier(N_per, K, float(r),
                             np.random.default_rng(rng.integers(0, 2**31 - 1)))
            for r in r_grid]


def mixture_auroc_ci(states, prior_w, alpha=0.05):
    """Mixture posterior on AUROC across r-grid SMCs. Returns lo, hi shape (K,)."""
    K = states[0]["t"].shape[1]
    los = np.zeros(K)
    his = np.zeros(K)
    for k in range(K):
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
        losses += gw * _expected_loss_vec(sg, k, signals)
    return losses


def choose_item_mixture(states, prior_w):
    K = states[0]["t"].shape[1]
    losses = np.stack([expected_loss_mixture(states, prior_w, k, SIGNAL_GRID)
                       for k in range(K)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


def run_session_mcmc_betaprior(true_params, prior_w, r_grid=DEFAULT_R_GRID,
                                N_per=400, K=2, max_q=400, delta_auroc=0.05,
                                seed=0, run_until_max=False, log_trajectory=False,
                                alpha=0.05, n_mh_steps=15, proposal_scale=0.5,
                                ess_threshold_frac=0.5):
    rng = np.random.default_rng(seed)
    states = init_states_mcmc_grid(N_per, K, r_grid, rng)
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
        # update every grid SMC
        for state in states:
            update(state, k, s, y)
        # rejuvenate any grid whose ESS dropped
        for state in states:
            if ess(state["w"]) < ess_threshold_frac * len(state["w"]):
                resample_and_rejuvenate(state, rng, n_mh_steps, proposal_scale)
        n_q += 1
        lo, hi = mixture_auroc_ci(states, prior_w, alpha)
        if log_trajectory:
            los_t.append(lo.copy())
            his_t.append(hi.copy())
        if stop_step is None:
            if np.max((hi - lo) / 2.0) < delta_auroc:
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
