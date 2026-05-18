"""Session drivers with AUROC-based stopping and AUROC-trajectory logging.

Works for any K (uses core_K functions throughout; K=2 is just a special case).
"""
import numpy as np

from core import simulate_response, ess
from core_K import (
    sample_prior_hier_K, sample_prior_brute_K,
    update_hier_K, update_brute_K,
    choose_item_hier_K, choose_item_brute_K,
    resample_jitter_hier_K, resample_jitter_brute_K,
)
from auroc import (
    auroc_from_l,
    auroc_quantiles_from_particles_hier,
    auroc_quantiles_from_particles_brute,
)


def _auroc_ci_hier(particles, alpha=0.05):
    q = auroc_quantiles_from_particles_hier(particles,
                                             alphas=(alpha / 2, 1 - alpha / 2))
    return q[:, 0], q[:, 1]


def _auroc_ci_brute(brute, alpha=0.05):
    q = auroc_quantiles_from_particles_brute(brute,
                                              alphas=(alpha / 2, 1 - alpha / 2))
    return q[:, 0], q[:, 1]


def run_session_auroc(method, true_params, K, r_assumed, rho_assumed=0.0,
                     max_q=500, delta_auroc=0.05, N=2500, seed=0,
                     run_until_max=False, log_trajectory=False, alpha=0.05):
    """One session with AUROC-based stopping.

    Args:
      method: "hier" or "brute"
      true_params: (2K,) array of true (t_1, l_1, ..., t_K, l_K)
      K: number of tasks
      r_assumed, rho_assumed: estimator's prior parameters (hier only)
      delta_auroc: stopping threshold on per-task AUROC posterior halfwidth
      run_until_max: if True, ignore stopping criterion and run to max_q
                     (useful for trajectory logging + post-hoc inflation analysis)
      log_trajectory: if True, return (max_q+1, K) arrays of lo/hi AUROC bounds
                      at every step (step 0 = prior).

    Returns dict with:
      n_questions       (int) - when AUROC criterion fired (or max_q)
      stopped_early     (bool)
      true_auroc        (K,) ndarray
      final_lo, final_hi  (K,) ndarrays
      [if log_trajectory] lo_traj, hi_traj  (n_steps+1, K) ndarrays
    """
    rng = np.random.default_rng(seed)
    assert len(true_params) == 2 * K

    if method == "hier":
        state = sample_prior_hier_K(N, K, r_assumed, rho_assumed, rng)
        update_fn = update_hier_K
        choose_fn = choose_item_hier_K
        resample_fn = resample_jitter_hier_K
        ess_fn = lambda c: ess(c["w"])
        ci_fn = _auroc_ci_hier
    elif method == "brute":
        state = sample_prior_brute_K(N, K, rng)
        update_fn = update_brute_K
        choose_fn = choose_item_brute_K
        resample_fn = resample_jitter_brute_K
        ess_fn = lambda c: min(ess(p["w"]) for p in c)
        ci_fn = _auroc_ci_brute
    else:
        raise ValueError(method)

    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    lo, hi = ci_fn(state, alpha)
    if log_trajectory:
        los, his = [lo.copy()], [hi.copy()]

    n_q = 0
    stop_step = None
    for q in range(max_q):
        k, s = choose_fn(state)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update_fn(state, k, s, y)
        if ess_fn(state) < N / 4:
            state = resample_fn(state, rng)
        n_q += 1
        lo, hi = ci_fn(state, alpha)
        if log_trajectory:
            los.append(lo.copy())
            his.append(hi.copy())
        if stop_step is None:
            hw = (hi - lo) / 2.0
            if np.max(hw) < delta_auroc:
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
        out["lo_traj"] = np.array(los)
        out["hi_traj"] = np.array(his)
    return out


def first_crossing(lo_traj, hi_traj, delta, inflation_c=1.0):
    """Find the first question index where inflated AUROC halfwidth < delta for all tasks.

    lo_traj, hi_traj: shape (n_steps+1, K) — bounds at step 0..n_steps.
    Returns int (question count), or None if never crossed.
    """
    mid = (lo_traj + hi_traj) / 2.0
    hw = (hi_traj - lo_traj) / 2.0 * inflation_c
    all_below = (hw.max(axis=1) < delta)
    idx = np.where(all_below)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def coverage_at_step(lo_traj, hi_traj, true_auroc, inflation_c=1.0):
    """For each step in the trajectory, is the true AUROC inside the inflated CI?

    Returns shape (n_steps+1, K) bool array.
    """
    mid = (lo_traj + hi_traj) / 2.0
    hw = (hi_traj - lo_traj) / 2.0 * inflation_c
    lo_i = mid - hw
    hi_i = mid + hw
    return (true_auroc[None, :] >= lo_i) & (true_auroc[None, :] <= hi_i)
