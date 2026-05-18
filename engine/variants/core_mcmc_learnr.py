"""SMC sampler with MCMC rejuvenation, learn-r via marginal likelihood, AUROC stopping.

K-task generalization. Maintains an SMC per r-grid value. After each observation:
  - update grid g's particles via reweighting + log_lik increment
  - accumulate log marginal likelihood for grid g
  - posterior on r is softmax of log marginal likelihoods
  - mixture posterior on (t, l) uses posterior r-weights
  - resample+MCMC each grid independently when its ESS drops

Different from core_mcmc_betaprior.py: the r-grid weights are updated by data,
not held fixed.
"""
import numpy as np
from scipy.stats import norm

from core_mcmc import (
    make_state_hier, ess, resample_and_rejuvenate, _expected_loss_vec,
    SIGNAL_GRID, simulate_response,
)
from core import response_prob, weighted_quantile
from auroc import auroc_from_l

DEFAULT_R_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


def init_states_learnr_mcmc(N_per, K, r_grid, rng):
    states = [make_state_hier(N_per, K, float(r),
                                np.random.default_rng(rng.integers(0, 2**31 - 1)))
              for r in r_grid]
    log_marg = np.zeros(len(r_grid))
    return states, log_marg


def _reweight_one(state, k, s, y):
    """Update state's particles for observation (k, s, y). Returns p(y|history)."""
    p = response_prob(s, state["t"][:, k], state["l"][:, k])
    lik = p if y == 1 else (1.0 - p)
    p_y_history = float((state["w"] * lik).sum())
    p_y_history = max(p_y_history, 1e-12)
    log_p_obs = np.log(p) if y == 1 else np.log(1.0 - p)
    state["log_lik"] += log_p_obs
    w = state["w"] * lik
    state["w"] = w / w.sum()
    state["history"].append((int(k), float(s), int(y)))
    return p_y_history


def posterior_r_weights(log_marg, prior_log_w=None):
    """Posterior weights over r-grid. prior_log_w defaults to uniform."""
    if prior_log_w is None:
        prior_log_w = np.zeros_like(log_marg)
    m = log_marg + prior_log_w
    m = m - m.max()
    w = np.exp(m)
    return w / w.sum()


def mixture_auroc_ci(states, post_r_w, alpha=0.05):
    K = states[0]["t"].shape[1]
    los = np.zeros(K)
    his = np.zeros(K)
    for k in range(K):
        auc_all = []
        w_all = []
        for sg, gw in zip(states, post_r_w):
            auc = auroc_from_l(sg["l"][:, k])
            auc_all.append(auc)
            w_all.append(gw * (sg["w"] / sg["w"].sum()))
        auc_all = np.concatenate(auc_all)
        w_all = np.concatenate(w_all)
        w_all /= w_all.sum()
        los[k] = weighted_quantile(auc_all, w_all, alpha / 2)
        his[k] = weighted_quantile(auc_all, w_all, 1 - alpha / 2)
    return los, his


def expected_loss_mixture(states, post_r_w, k, signals):
    losses = np.zeros_like(signals, dtype=float)
    for sg, gw in zip(states, post_r_w):
        if gw < 1e-6:
            continue
        losses += gw * _expected_loss_vec(sg, k, signals)
    return losses


def choose_item_mixture(states, post_r_w):
    K = states[0]["t"].shape[1]
    losses = np.stack([expected_loss_mixture(states, post_r_w, k, SIGNAL_GRID)
                       for k in range(K)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


def run_session_mcmc_learnr(true_params, K, r_grid=DEFAULT_R_GRID,
                             N_per=400, max_q=400, delta_auroc=0.05, seed=0,
                             run_until_max=False, log_trajectory=False, alpha=0.05,
                             n_mh_steps=15, proposal_scale=0.5,
                             ess_threshold_frac=0.5):
    rng = np.random.default_rng(seed)
    states, log_marg = init_states_learnr_mcmc(N_per, K, r_grid, rng)
    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    post_r_w = posterior_r_weights(log_marg)
    lo, hi = mixture_auroc_ci(states, post_r_w, alpha)
    if log_trajectory:
        los_t, his_t = [lo.copy()], [hi.copy()]
    n_q = 0
    stop_step = None
    for q in range(max_q):
        k, s = choose_item_mixture(states, post_r_w)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        for g, state in enumerate(states):
            p_y = _reweight_one(state, k, s, y)
            log_marg[g] += np.log(p_y)
            if ess(state["w"]) < ess_threshold_frac * len(state["w"]):
                resample_and_rejuvenate(state, rng, n_mh_steps, proposal_scale)
        n_q += 1
        post_r_w = posterior_r_weights(log_marg)
        lo, hi = mixture_auroc_ci(states, post_r_w, alpha)
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
        "r_posterior": post_r_w.tolist(),
        "r_posterior_mean": float((post_r_w * r_grid).sum()),
    }
    if log_trajectory:
        out["lo_traj"] = np.array(los_t)
        out["hi_traj"] = np.array(his_t)
    return out
