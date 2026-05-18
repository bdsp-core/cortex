"""Shared library extracted from explore_two_tasks.ipynb.

Canonical functions for the symmetric diagonal hierarchical 2-task model and the
brute-force baseline. Matches the notebook semantics exactly.

Notation:
  r   = cross-task correlation (Corr(t1, t2) = Corr(l1, l2))   = lambda^2
  rho = bias-skill correlation (Corr(G_bias, G_skill))
"""
import numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr, logsumexp  # FIX-T0.7: stable log Φ; FIX-T1.4: lapse mixture

SIGNAL_GRID = np.linspace(-3.0, 3.0, 11)

# FIX-T1.4 + F0.1: Lapse rate (matches spike paper CLAUDE.md §2.1 Eq. 2).
# Response model: P(y=1 | c, ℓ, θ; λ) = λ + (1 − 2λ)·Φ(exp(ℓ)·(c+θ)).
# Symmetric lapse: floor λ, ceiling 1 − λ. The previous parametrization
# (1−λ)·Φ + 0.5λ corresponded to a 4AFC guess rate and was algebraically
# inconsistent with train_val_split_and_fit.py:probit_lapse_nll.
LAPSE_RATE = 0.025
_LOG_LAPSE = np.log(LAPSE_RATE)
_LOG_ONE_MINUS_TWO_LAPSE = np.log1p(-2.0 * LAPSE_RATE)


# ───────────────────────────── Generative model ─────────────────────────────

def sample_true_expert_rho(r, rho, rng):
    lam = np.sqrt(r)
    psi = np.sqrt(1.0 - r)
    G_bias = rng.standard_normal()
    G_skill = rho * G_bias + np.sqrt(1.0 - rho ** 2) * rng.standard_normal()
    eps = rng.standard_normal(4)
    return np.array([lam * G_bias + psi * eps[0],
                     lam * G_skill + psi * eps[1],
                     lam * G_bias + psi * eps[2],
                     lam * G_skill + psi * eps[3]])


def sample_prior_hier_rho(N, r, rho, rng):
    lam = np.sqrt(r)
    psi = np.sqrt(1.0 - r)
    G_bias = rng.standard_normal(N)
    G_skill = rho * G_bias + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(N)
    eps = rng.standard_normal((N, 4))
    t = np.column_stack([lam * G_bias + psi * eps[:, 0],
                         lam * G_bias + psi * eps[:, 2]])
    l = np.column_stack([lam * G_skill + psi * eps[:, 1],
                         lam * G_skill + psi * eps[:, 3]])
    return {"t": t, "l": l, "w": np.full(N, 1.0 / N)}


def sample_prior_brute(N, rng):
    return [{"t": rng.standard_normal(N),
             "l": rng.standard_normal(N),
             "w": np.full(N, 1.0 / N)} for _ in range(2)]


def _marg_z(l, s, t, s_sd=0.0):
    """Item-signal latent z, EXACTLY marginalized over s ~ N(s, s_sd²).

    Phase 3.5: the segment signal is now a posterior (s_mean, s_sd) rather
    than an error-free scalar.  Marginalizing the probit-lapse likelihood
    over a Gaussian s is closed-form (Gaussian–probit convolution):

        E_s[ Φ(e^ℓ (s + t)) ] = Φ(  e^ℓ (s_mean + t)
                                   / √(1 + e^{2ℓ}·s_sd²) )

    so we attenuate z by 1/√(1 + (e^ℓ·s_sd)²).  At s_sd == 0.0 the factor
    is exactly 1.0 and z is BIT-IDENTICAL to the pre-Phase-3.5 engine
    (IEEE-754: 1.0+0.0=1.0, √1.0=1.0, z/1.0=z) — the parity@s_sd→0 gate
    holds by construction, so the whole validated suite is unchanged.
    """
    el = np.exp(l)
    z = el * (s + t)
    if np.all(s_sd == 0.0):              # bit-exact fast path / default
        return z
    return z / np.sqrt(1.0 + (el * s_sd) ** 2)


def response_prob(s, t, l, s_sd=0.0):
    """P(y=1 | s, t, ℓ) — spike-paper Eq. 2 lapse mixture, marginalized
    over the segment-signal posterior s ~ N(s, s_sd²).

    FIX-T0.7: Computed via log_ndtr for numerical stability in the tails
    (replaces np.clip(norm.cdf(z), 1e-9, 1-1e-9), which collapses to 1.0 at
    |z| ≳ 6 in float64 due to catastrophic cancellation in 1−Φ(z)).

    F0.1: spike-paper Eq. 2 lapse: p ∈ [λ, 1 − λ]: p = λ + (1 − 2λ)·Φ(z).
    Phase 3.5: s_sd=0.0 (default) ⇒ bit-identical to the prior engine.
    """
    z = _marg_z(l, s, t, s_sd)
    log_phi = log_ndtr(z)
    p = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * np.exp(log_phi)
    return p


def log_response_prob(s, t, l, y, s_sd=0.0):
    """log P(y | s, t, ℓ), lapse mixture, tail-stable, marginalized over
    s ~ N(s, s_sd²) (Phase 3.5; s_sd=0 ⇒ bit-identical to prior engine).

    FIX-T0.7 + F0.1: uses log_ndtr + logsumexp on the spike-paper lapse mixture,
    avoiding the underflow that np.log(1 − norm.cdf(z)) suffers for z ≫ 0.

        log p(y=1) = logsumexp(log(1−2λ) + log_ndtr( z), log(λ))
        log p(y=0) = logsumexp(log(1−2λ) + log_ndtr(−z), log(λ))
    """
    z = _marg_z(l, s, t, s_sd)
    log_phi = log_ndtr(z) if y == 1 else log_ndtr(-z)
    a = _LOG_ONE_MINUS_TWO_LAPSE + log_phi
    b = np.broadcast_to(_LOG_LAPSE, np.shape(a))
    return logsumexp(np.stack([a, b], axis=0), axis=0)


def simulate_response(s, t_true, l_true, rng):
    return int(rng.random() < response_prob(s, t_true, l_true))


# ───────────────────────────── Posterior summaries ─────────────────────────────

def weighted_mean_var(values, weights):
    mu = (weights * values).sum()
    var = (weights * (values - mu) ** 2).sum()
    return mu, var


def posterior_means_hier(particles):
    means = []
    for k in range(2):
        for arr in (particles["t"][:, k], particles["l"][:, k]):
            mu, _ = weighted_mean_var(arr, particles["w"])
            means.append(mu)
    return np.array(means)


def posterior_means_brute(brute):
    means = []
    for k in range(2):
        for arr in (brute[k]["t"], brute[k]["l"]):
            mu, _ = weighted_mean_var(arr, brute[k]["w"])
            means.append(mu)
    return np.array(means)


def posterior_sds_hier(particles):
    sds = []
    for k in range(2):
        for arr in (particles["t"][:, k], particles["l"][:, k]):
            _, v = weighted_mean_var(arr, particles["w"])
            sds.append(np.sqrt(max(v, 0.0)))
    return np.array(sds)


def posterior_sds_brute(brute):
    sds = []
    for k in range(2):
        for arr in (brute[k]["t"], brute[k]["l"]):
            _, v = weighted_mean_var(arr, brute[k]["w"])
            sds.append(np.sqrt(max(v, 0.0)))
    return np.array(sds)


def ess(weights):
    return 1.0 / (weights ** 2).sum()


def weighted_quantile(values, weights, q):
    sorted_idx = np.argsort(values)
    cumulative_weights = np.cumsum(weights[sorted_idx])
    cumulative_weights /= cumulative_weights[-1]
    return float(np.interp(q, cumulative_weights, values[sorted_idx]))


def posterior_ci_hier_4d(particles, alpha=0.05):
    w = particles["w"] / particles["w"].sum()
    hw = []
    for k in range(2):
        for arr in (particles["t"][:, k], particles["l"][:, k]):
            lo = weighted_quantile(arr, w, alpha / 2)
            hi = weighted_quantile(arr, w, 1 - alpha / 2)
            hw.append((hi - lo) / 2)
    return np.array(hw)


def posterior_ci_brute(brute, alpha=0.05):
    hw = []
    for k in range(2):
        w = brute[k]["w"] / brute[k]["w"].sum()
        for key in ("t", "l"):
            arr = brute[k][key]
            lo = weighted_quantile(arr, w, alpha / 2)
            hi = weighted_quantile(arr, w, 1 - alpha / 2)
            hw.append((hi - lo) / 2)
    return np.array(hw)


def posterior_quantiles_hier(particles, alpha=0.05):
    """Returns (lo, hi) arrays of length 4 — used for coverage checks."""
    w = particles["w"] / particles["w"].sum()
    los, his = [], []
    for k in range(2):
        for arr in (particles["t"][:, k], particles["l"][:, k]):
            los.append(weighted_quantile(arr, w, alpha / 2))
            his.append(weighted_quantile(arr, w, 1 - alpha / 2))
    return np.array(los), np.array(his)


def posterior_quantiles_brute(brute, alpha=0.05):
    los, his = [], []
    for k in range(2):
        w = brute[k]["w"] / brute[k]["w"].sum()
        for key in ("t", "l"):
            arr = brute[k][key]
            los.append(weighted_quantile(arr, w, alpha / 2))
            his.append(weighted_quantile(arr, w, 1 - alpha / 2))
    return np.array(los), np.array(his)


# ───────────────────────────── SMC update ─────────────────────────────

def update_hier(particles, k, s, y):
    # FIX-T0.7 + FIX-T1.4: work in log-space using log_response_prob (lapse-mixture probit)
    # instead of `p if y==1 else (1-p)`. Avoids underflow when |z| is large and per-particle
    # likelihoods span many orders of magnitude.
    log_lik = log_response_prob(s, particles["t"][:, k], particles["l"][:, k], y)
    log_w = np.log(particles["w"]) + log_lik
    log_w -= log_w.max()
    w = np.exp(log_w)
    particles["w"] = w / w.sum()
    return particles


def update_brute(brute, k, s, y):
    # FIX-T0.7 + FIX-T1.4: log-space SMC update with lapse-mixture probit likelihood.
    pset = brute[k]
    log_lik = log_response_prob(s, pset["t"], pset["l"], y)
    log_w = np.log(pset["w"]) + log_lik
    log_w -= log_w.max()
    w = np.exp(log_w)
    pset["w"] = w / w.sum()
    return brute


def _kernel_smooth(state, rng, h=0.1):
    a = np.sqrt(1.0 - h ** 2)
    mean = state.mean(axis=0)
    centered = state - mean
    cov = centered.T @ centered / state.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_cov = eigvecs * np.sqrt(eigvals)
    noise = rng.standard_normal(state.shape) @ (h * sqrt_cov).T
    return a * state + (1.0 - a) * mean + noise


def resample_jitter_hier(particles, rng, h=0.1):
    N = len(particles["w"])
    idx = rng.choice(N, size=N, p=particles["w"])
    state = np.column_stack([particles["t"][idx], particles["l"][idx]])
    state = _kernel_smooth(state, rng, h)
    return {"t": state[:, :2], "l": state[:, 2:],
            "w": np.full(N, 1.0 / N)}


def resample_jitter_brute(brute, rng, h=0.1):
    out = []
    for pset in brute:
        N = len(pset["w"])
        idx = rng.choice(N, size=N, p=pset["w"])
        state = np.column_stack([pset["t"][idx], pset["l"][idx]])
        state = _kernel_smooth(state, rng, h)
        out.append({"t": state[:, 0], "l": state[:, 1],
                    "w": np.full(N, 1.0 / N)})
    return out


# ───────────────────────────── Item selection ─────────────────────────────

def expected_loss_hier_vec(particles, k, signals, signal_sds=None):
    t_k = particles["t"][:, k]
    l_k = particles["l"][:, k]
    el = np.exp(l_k)[None, :]
    z = el * (signals[:, None] + t_k[None, :])
    if signal_sds is not None:
        # Phase 3.5: marginalize each candidate item over its s posterior
        # N(signal, signal_sd²); signal_sds=None ⇒ bit-identical (default).
        z = z / np.sqrt(1.0 + (el * np.asarray(signal_sds)[:, None]) ** 2)
    # FIX-T0.7: log_ndtr replaces np.clip(norm.cdf(z), 1e-9, 1-1e-9) for stability.
    # F0.1: spike-paper Eq. 2 lapse mixture; p ∈ [λ, 1 − λ] without ad-hoc clipping.
    p = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * np.exp(log_ndtr(z))
    w = particles["w"]
    p_yes = (p * w).sum(axis=1)
    w_y1 = p * w
    w_y1 = w_y1 / w_y1.sum(axis=1, keepdims=True)
    w_y0 = (1.0 - p) * w
    w_y0 = w_y0 / w_y0.sum(axis=1, keepdims=True)

    def var_vec(arr, weights):
        mu = (weights * arr).sum(axis=1)
        return (weights * (arr - mu[:, None]) ** 2).sum(axis=1)

    total_y1 = sum(var_vec(particles["t"][:, kk], w_y1) for kk in range(2)) + \
               sum(var_vec(particles["l"][:, kk], w_y1) for kk in range(2))
    total_y0 = sum(var_vec(particles["t"][:, kk], w_y0) for kk in range(2)) + \
               sum(var_vec(particles["l"][:, kk], w_y0) for kk in range(2))
    return p_yes * total_y1 + (1.0 - p_yes) * total_y0


def expected_loss_brute_vec(brute, k, signals):
    pset = brute[k]
    t = pset["t"]
    l = pset["l"]
    z = np.exp(l)[None, :] * (signals[:, None] + t[None, :])
    # FIX-T0.7: log_ndtr replaces np.clip(norm.cdf(z), 1e-9, 1-1e-9) for stability.
    # F0.1: spike-paper Eq. 2 lapse mixture; p ∈ [λ, 1 − λ] without ad-hoc clipping.
    p = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * np.exp(log_ndtr(z))
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
    other = brute[1 - k]
    _, vt_o = weighted_mean_var(other["t"], other["w"])
    _, vl_o = weighted_mean_var(other["l"], other["w"])
    return expected_var_k + vt_o + vl_o


def choose_item_hier(particles):
    losses = np.stack([expected_loss_hier_vec(particles, k, SIGNAL_GRID)
                       for k in range(2)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


def choose_item_brute(brute):
    losses = np.stack([expected_loss_brute_vec(brute, k, SIGNAL_GRID)
                       for k in range(2)], axis=0)
    k_best, s_idx = np.unravel_index(np.argmin(losses), losses.shape)
    return int(k_best), float(SIGNAL_GRID[s_idx])


# ───────────────────────────── Session driver ─────────────────────────────

def run_session_rho(method, true_params, r_assumed, rho_assumed=0.0,
                    max_q=300, delta=0.4, N=2500, seed=0, jitter_h=0.1,
                    return_final_state=False):
    """Run one session.

    `r_assumed` and `rho_assumed` control the estimator's prior.
    Data generation happens BEFORE this call (true_params is precomputed using
    the *true* r and rho, which may differ from the assumed values).
    """
    rng = np.random.default_rng(seed)
    if method == "hier":
        state = sample_prior_hier_rho(N, r_assumed, rho_assumed, rng)
        sd_fn = posterior_sds_hier
        update_fn = update_hier
        choose_fn = choose_item_hier
        resample_fn = lambda c, r: resample_jitter_hier(c, r, h=jitter_h)
        ess_fn = lambda c: ess(c["w"])
        ci_fn = posterior_ci_hier_4d
        means_fn = posterior_means_hier
        quantile_fn = posterior_quantiles_hier
    elif method == "brute":
        state = sample_prior_brute(N, rng)
        sd_fn = posterior_sds_brute
        update_fn = update_brute
        choose_fn = choose_item_brute
        resample_fn = lambda c, r: resample_jitter_brute(c, r, h=jitter_h)
        ess_fn = lambda c: min(ess(c[0]["w"]), ess(c[1]["w"]))
        ci_fn = posterior_ci_brute
        means_fn = posterior_means_brute
        quantile_fn = posterior_quantiles_brute
    else:
        raise ValueError(method)

    questions = []
    for q in range(max_q):
        k, s = choose_fn(state)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update_fn(state, k, s, y)
        if ess_fn(state) < N / 4:
            state = resample_fn(state, rng)
        questions.append((k, s, y))
        cis = ci_fn(state)
        if np.nanmax(cis) < delta:
            break

    final_means = means_fn(state)
    final_los, final_his = quantile_fn(state)
    out = {
        "n_questions": len(questions),
        "stopped_early": len(questions) < max_q,
        "final_means": final_means,
        "final_lo": final_los,
        "final_hi": final_his,
    }
    if return_final_state:
        out["final_state"] = state
    return out
