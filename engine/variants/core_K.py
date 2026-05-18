"""Generalization to K>=2 tasks.

Same symmetric diagonal hierarchical model as core.py:
    G_bias  ~ N(0, 1)
    G_skill = rho * G_bias + sqrt(1 - rho^2) * N(0, 1)
    t_k = sqrt(r) * G_bias  + sqrt(1 - r) * eps_t_k     for k = 1..K
    l_k = sqrt(r) * G_skill + sqrt(1 - r) * eps_l_k     for k = 1..K

State: {"t": (N, K), "l": (N, K), "w": (N,)}.
Brute-force: list of K independent particle sets {"t": (N,), "l": (N,), "w": (N,)}.
"""
import numpy as np
from scipy.stats import norm

from core import (
    response_prob, simulate_response, weighted_mean_var, ess,
    weighted_quantile, _kernel_smooth, SIGNAL_GRID,
    LAPSE_RATE,   # Phase 2: single-source the spike-paper lapse constant
)


# ───────────────────────────── Generative model (K tasks) ─────────────────────────────

def sample_true_expert_K(K, r, rho, rng):
    """Returns (t1, l1, t2, l2, ..., tK, lK)."""
    lam = np.sqrt(r)
    psi = np.sqrt(1.0 - r)
    G_bias = rng.standard_normal()
    G_skill = rho * G_bias + np.sqrt(1.0 - rho ** 2) * rng.standard_normal()
    eps = rng.standard_normal(2 * K)
    out = np.empty(2 * K)
    for k in range(K):
        out[2 * k] = lam * G_bias + psi * eps[2 * k]
        out[2 * k + 1] = lam * G_skill + psi * eps[2 * k + 1]
    return out


def sample_prior_hier_K(N, K, r, rho, rng):
    lam = np.sqrt(r)
    psi = np.sqrt(1.0 - r)
    G_bias = rng.standard_normal(N)
    G_skill = rho * G_bias + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(N)
    eps_t = rng.standard_normal((N, K))
    eps_l = rng.standard_normal((N, K))
    t = lam * G_bias[:, None] + psi * eps_t
    l = lam * G_skill[:, None] + psi * eps_l
    return {"t": t, "l": l, "w": np.full(N, 1.0 / N)}


def sample_prior_brute_K(N, K, rng):
    return [{"t": rng.standard_normal(N),
             "l": rng.standard_normal(N),
             "w": np.full(N, 1.0 / N)} for _ in range(K)]


# ───────────────────────────── Posterior summaries (K tasks) ─────────────────────────────

def posterior_sds_hier_K(particles):
    K = particles["t"].shape[1]
    sds = []
    for k in range(K):
        for arr in (particles["t"][:, k], particles["l"][:, k]):
            _, v = weighted_mean_var(arr, particles["w"])
            sds.append(np.sqrt(max(v, 0.0)))
    return np.array(sds)


def posterior_sds_brute_K(brute):
    sds = []
    for k in range(len(brute)):
        for arr in (brute[k]["t"], brute[k]["l"]):
            _, v = weighted_mean_var(arr, brute[k]["w"])
            sds.append(np.sqrt(max(v, 0.0)))
    return np.array(sds)


def posterior_ci_hier_K(particles, alpha=0.05):
    w = particles["w"] / particles["w"].sum()
    K = particles["t"].shape[1]
    hw = []
    for k in range(K):
        for arr in (particles["t"][:, k], particles["l"][:, k]):
            lo = weighted_quantile(arr, w, alpha / 2)
            hi = weighted_quantile(arr, w, 1 - alpha / 2)
            hw.append((hi - lo) / 2)
    return np.array(hw)


def posterior_ci_brute_K(brute, alpha=0.05):
    hw = []
    for k in range(len(brute)):
        w = brute[k]["w"] / brute[k]["w"].sum()
        for key in ("t", "l"):
            arr = brute[k][key]
            lo = weighted_quantile(arr, w, alpha / 2)
            hi = weighted_quantile(arr, w, 1 - alpha / 2)
            hw.append((hi - lo) / 2)
    return np.array(hw)


# ───────────────────────────── SMC update (K tasks) ─────────────────────────────

def update_hier_K(particles, k, s, y):
    p = response_prob(s, particles["t"][:, k], particles["l"][:, k])
    lik = p if y == 1 else (1.0 - p)
    w = particles["w"] * lik
    particles["w"] = w / w.sum()
    return particles


def update_brute_K(brute, k, s, y):
    pset = brute[k]
    p = response_prob(s, pset["t"], pset["l"])
    lik = p if y == 1 else (1.0 - p)
    w = pset["w"] * lik
    pset["w"] = w / w.sum()
    return brute


def resample_jitter_hier_K(particles, rng, h=0.1):
    K = particles["t"].shape[1]
    N = len(particles["w"])
    idx = rng.choice(N, size=N, p=particles["w"])
    state = np.column_stack([particles["t"][idx], particles["l"][idx]])  # (N, 2K)
    state = _kernel_smooth(state, rng, h)
    return {"t": state[:, :K], "l": state[:, K:],
            "w": np.full(N, 1.0 / N)}


def resample_jitter_brute_K(brute, rng, h=0.1):
    out = []
    for pset in brute:
        N = len(pset["w"])
        idx = rng.choice(N, size=N, p=pset["w"])
        state = np.column_stack([pset["t"][idx], pset["l"][idx]])
        state = _kernel_smooth(state, rng, h)
        out.append({"t": state[:, 0], "l": state[:, 1],
                    "w": np.full(N, 1.0 / N)})
    return out


# ───────────────────────────── Item selection (K tasks) ─────────────────────────────

def expected_loss_hier_vec_K(particles, k, signals):
    K = particles["t"].shape[1]
    t_k = particles["t"][:, k]
    l_k = particles["l"][:, k]
    z = np.exp(l_k)[None, :] * (signals[:, None] + t_k[None, :])
    # Phase 2 hardening: replace np.clip(norm.cdf(z),1e-9,1-1e-9) (collapses
    # to exactly 1.0 at |z|>~6, biasing confident-rater item selection) with
    # the spike-paper Eq. 2 lapse mixture — identical to core_mcmc._p_response_yes
    # and core.response_prob (F0.1/FIX-T1.4). Bounded in [λ,1−λ]; no clip needed.
    p = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z)
    w = particles["w"]
    p_yes = (p * w).sum(axis=1)
    w_y1 = p * w
    w_y1 = w_y1 / w_y1.sum(axis=1, keepdims=True)
    w_y0 = (1.0 - p) * w
    w_y0 = w_y0 / w_y0.sum(axis=1, keepdims=True)

    def var_vec(arr, weights):
        mu = (weights * arr).sum(axis=1)
        return (weights * (arr - mu[:, None]) ** 2).sum(axis=1)

    total_y1 = sum(var_vec(particles["t"][:, kk], w_y1) for kk in range(K)) + \
               sum(var_vec(particles["l"][:, kk], w_y1) for kk in range(K))
    total_y0 = sum(var_vec(particles["t"][:, kk], w_y0) for kk in range(K)) + \
               sum(var_vec(particles["l"][:, kk], w_y0) for kk in range(K))
    return p_yes * total_y1 + (1.0 - p_yes) * total_y0


def expected_loss_brute_vec_K(brute, k, signals):
    pset = brute[k]
    t = pset["t"]
    l = pset["l"]
    z = np.exp(l)[None, :] * (signals[:, None] + t[None, :])
    # Phase 2 hardening: replace np.clip(norm.cdf(z),1e-9,1-1e-9) (collapses
    # to exactly 1.0 at |z|>~6, biasing confident-rater item selection) with
    # the spike-paper Eq. 2 lapse mixture — identical to core_mcmc._p_response_yes
    # and core.response_prob (F0.1/FIX-T1.4). Bounded in [λ,1−λ]; no clip needed.
    p = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z)
    w = pset["w"]
    p_yes = (p * w).sum(axis=1)
    w_y1 = p * w
    w_y1 = w_y1 / w_y1.sum(axis=1, keepdims=True)
    w_y0 = (1.0 - p) * w
    w_y0 = w_y0 / w_y0.sum(axis=1, keepdims=True)

    def var_vec(arr, weights):
        mu = (weights * arr).sum(axis=1)
        return (weights * (arr - mu[:, None]) ** 2).sum(axis=1)

    var_y1 = var_vec(t, w_y1) + var_vec(l, w_y1)
    var_y0 = var_vec(t, w_y0) + var_vec(l, w_y0)
    expected_var_k = p_yes * var_y1 + (1.0 - p_yes) * var_y0
    other_var_sum = 0.0
    for kk, other in enumerate(brute):
        if kk == k:
            continue
        _, vt_o = weighted_mean_var(other["t"], other["w"])
        _, vl_o = weighted_mean_var(other["l"], other["w"])
        other_var_sum += vt_o + vl_o
    return expected_var_k + other_var_sum


def choose_item_hier_K(particles):
    K = particles["t"].shape[1]
    losses = np.stack([expected_loss_hier_vec_K(particles, k, SIGNAL_GRID)
                       for k in range(K)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


def choose_item_brute_K(brute):
    K = len(brute)
    losses = np.stack([expected_loss_brute_vec_K(brute, k, SIGNAL_GRID)
                       for k in range(K)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


# ───────────────────────────── Session driver (K tasks) ─────────────────────────────

def run_session_K(method, true_params, K, r_assumed, rho_assumed=0.0,
                  max_q=400, delta=0.4, N=2500, seed=0):
    rng = np.random.default_rng(seed)
    if method == "hier":
        state = sample_prior_hier_K(N, K, r_assumed, rho_assumed, rng)
        update_fn = update_hier_K
        choose_fn = choose_item_hier_K
        resample_fn = resample_jitter_hier_K
        ess_fn = lambda c: ess(c["w"])
        ci_fn = posterior_ci_hier_K
    elif method == "brute":
        state = sample_prior_brute_K(N, K, rng)
        update_fn = update_brute_K
        choose_fn = choose_item_brute_K
        resample_fn = resample_jitter_brute_K
        ess_fn = lambda c: min(ess(p["w"]) for p in c)
        ci_fn = posterior_ci_brute_K
    else:
        raise ValueError(method)

    n_q = 0
    for q in range(max_q):
        k, s = choose_fn(state)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update_fn(state, k, s, y)
        if ess_fn(state) < N / 4:
            state = resample_fn(state, rng)
        n_q += 1
        cis = ci_fn(state)
        if np.nanmax(cis) < delta:
            break

    return {"n_questions": n_q, "stopped_early": n_q < max_q}
