"""Methodology-compliant brute-force baseline with MCMC-rejuvenation SMC.

This module implements the brute-force tester from `slides_mcmc.md`:
  - K independent 2-D MCMCs (one per domain / task) — i.e., K separate 2-D
    particle clouds with N(0,1) × N(0,1) priors and no information sharing
    across tasks. This is the "methodology-compliant" brute baseline used as
    the comparator against the hierarchical (hier) sampler in `core_mcmc.py`.
  - Item selection: signal optimization within a task; task selection greedy
    by per-task expected variance reduction (the operational choice in
    `core_K.choose_item_brute_K`).

The earlier `core_mcmc.make_state_brute` implements brute as a single 2K-D
particle cloud with global resampling — that is NOT the methodology brute and
gives the joint sampler a per-task effective N bounded by N, vs the K·N
particle-effort budget the methodology specifies. See STATE_LOG_2026_05_11.md.

Particle budget convention: N particles *per task* (so total particles K·N
for brute, vs hier's N total in a 2K-D cloud). Matches `core_K.py`.

Per-task state dict:
    {"t": (N,), "l": (N,), "w": (N,),
     "log_prior": (N,), "log_lik": (N,),
     "history": [(s, y), ...]}    # history is per-task, no k field

MCMC proposal scale: default 1.5 in 2-D (Roberts-Gelman-Gilks optimum is
2.38/sqrt(d) ≈ 1.68 for d=2). The 0.5 default in `core_mcmc.py` was for 2K-D.

Last revised 2026-05-14: applied Tier-0/Tier-1 corrections from reviewer panel.
  - FIX-T0.7: replaced log(clip(norm.cdf(z))) with scipy.special.log_ndtr(z)
    for numerical stability in the tails.
  - FIX-T1.4: introduced lapse rate λ = LAPSE_RATE = 0.025 in every likelihood
    evaluation (matching spike paper CLAUDE.md §2.1):
        P(y=1 | c, ℓ, θ; λ) = (1 − λ)·Φ(exp(ℓ)·(c+θ)) + 0.5·λ
  - FIX-T1.3: certification stopping in `run_session_mcmc_brute_k_cert` is now
    MCSE-buffered with Z_BUFFER = 2.0 to control Monte-Carlo error in the tail
    pass-probability estimates.
  - FIX-T0.5: removed deprecated KL-based item selection (`_kl_vec_brute`,
    `choose_item_kl_brute`) — STATE_LOG §3.1 documents the fatal flaw.
"""
import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm

from auroc import auroc_from_l, auroc_quantiles_from_particles_brute

# Same signal grid as core_mcmc.py and core_K.py — keep consistent.
SIGNAL_GRID = np.linspace(-3.0, 3.0, 11)

# FIX-T1.4 + F0.1: lapse rate (spike paper CLAUDE.md §2.1 Eq. 2).
# Response model: P(y=1 | c, ℓ, θ; λ) = λ + (1 − 2λ)·Φ(exp(ℓ)·(c+θ)).
# Floor λ, ceiling 1 − λ (symmetric). The previous parametrization
# (1 − λ)·Φ + 0.5λ (floor λ/2, ceiling 1 − λ/2) corresponded to a 4AFC
# guess rate, not the binary task — it was algebraically inconsistent with
# train_val_split_and_fit.py:probit_lapse_nll.
LAPSE_RATE = 0.025

# FIX-T1.3: MCSE buffer multiplier for certification stopping.
# Stop PASS only when pass_prob − Z_BUFFER · MCSE ≥ stop_thresh.
Z_BUFFER = 2.0

# Precomputed log-mixture constants for the lapse model (F0.1).
_LOG_LAPSE = float(np.log(LAPSE_RATE))
_LOG_ONE_MINUS_TWO_LAPSE = float(np.log1p(-2.0 * LAPSE_RATE))


# ───────────── log-likelihood helpers (F0.1 lapse mixture, FIX-T0.7 log_ndtr) ─────────────

def _log_p_y(z, y):
    """Log P(y | z; LAPSE_RATE) under spike-paper symmetric lapse mixture.

    P(y=1) = λ + (1 − 2λ)·Φ(z)
    P(y=0) = λ + (1 − 2λ)·Φ(−z)

    Computed via logsumexp on log_ndtr for numerical stability in the tails.
    `z` may be any shape; `y` is 0 or 1 (scalar). Returns array with shape of z.
    """
    z = np.asarray(z)
    if y == 1:
        a = _LOG_ONE_MINUS_TWO_LAPSE + log_ndtr(z)
    else:
        a = _LOG_ONE_MINUS_TWO_LAPSE + log_ndtr(-z)
    b = np.full_like(a, _LOG_LAPSE)
    return logsumexp(np.stack([a, b], axis=0), axis=0)


def _p_y1(z):
    """P(y=1 | z; LAPSE_RATE) under spike-paper symmetric lapse mixture.

    Used wherever a probability (not its log) is needed — e.g. the predictive
    weights inside the expected-variance item selection. norm.cdf is fine here
    because we are not taking its log.
    """
    return LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z)


# ───────────── log densities for a single task (vectorized over N particles) ─────────────

def _log_prior_2d(t, l):
    """Independent N(0,1) prior per parameter for a single task. t, l shape (N,)."""
    return -0.5 * (t * t + l * l)


def _log_lik_history_2d(t, l, history):
    """Cumulative log-likelihood of one task's history given particles.

    history: list of (s, y) tuples (no k field — these are task-local queries).
    Returns shape (N,). FIX-T0.7 + FIX-T1.4: uses log_ndtr through the lapse
    mixture log-likelihood `_log_p_y`.
    """
    if not history:
        return np.zeros(t.shape[0])
    ll = np.zeros(t.shape[0])
    for (s, y) in history:
        z = np.exp(l) * (s + t)
        ll += _log_p_y(z, y)
    return ll


# ───────────── state construction ─────────────

def make_state_brute_k(N, K, rng):
    """K independent particle clouds, each with N(0,1) × N(0,1) prior.

    Returns a list of K state dicts. Each dict is independently resampleable
    and rejuvenable. RNG consumption: 2*K*N samples (N for each of t_k, l_k).
    Matches `core_K.sample_prior_brute_K` ordering.
    """
    states = []
    for _ in range(K):
        t = rng.standard_normal(N)
        l = rng.standard_normal(N)
        states.append({
            "t": t, "l": l,
            "w": np.full(N, 1.0 / N),
            "log_prior": _log_prior_2d(t, l),
            "log_lik": np.zeros(N),
            "history": [],
        })
    return states


# ───────────── reweight on observation ─────────────

def update_brute_k(states, k, s, y):
    """Reweight task k's particles by likelihood of (s, y). Other tasks untouched.

    Updates `log_lik` in place; appends (s, y) to task k's history.
    FIX-T0.7 + FIX-T1.4: lapse-mixture likelihood via `_log_p_y` / log_ndtr.
    """
    state = states[k]
    z = np.exp(state["l"]) * (s + state["t"])
    log_p_obs = _log_p_y(z, y)
    state["log_lik"] += log_p_obs
    w = state["w"] * np.exp(log_p_obs)
    s_w = w.sum()
    if s_w <= 0:
        state["w"] = np.full_like(state["w"], 1.0 / len(state["w"]))
    else:
        state["w"] = w / s_w
    state["history"].append((float(s), int(y)))


def ess(w):
    return 1.0 / (w * w).sum()


# ───────────── MCMC rejuvenation for a single task ─────────────

def mh_rejuvenate_2d(state, n_steps, proposal_scale, rng):
    """Run n_steps of Metropolis-Hastings on one task's 2-D particle cloud.

    Proposal: random-walk Gaussian with covariance proposal_scale^2 * cov_cloud
    where cov_cloud is the empirical 2x2 covariance of (t, l). Acceptance ratio
    uses cached log_prior + log_lik; recomputes both at the proposal point.

    Returns the mean acceptance rate across the n_steps.
    """
    N = state["t"].shape[0]
    accepts = []
    for _ in range(n_steps):
        theta = np.column_stack([state["t"], state["l"]])  # (N, 2)
        cov = np.cov(theta, rowvar=False)
        if cov.ndim == 0:
            cov = np.atleast_2d(cov)
        cov = cov + 1e-6 * np.eye(2)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            L = eigvecs * np.sqrt(eigvals)
        eps = rng.standard_normal((N, 2))
        theta_new = theta + proposal_scale * (eps @ L.T)
        t_new = theta_new[:, 0]
        l_new = theta_new[:, 1]

        lp_new = _log_prior_2d(t_new, l_new)
        ll_new = _log_lik_history_2d(t_new, l_new, state["history"])
        log_alpha = (lp_new + ll_new) - (state["log_prior"] + state["log_lik"])
        u = rng.random(N)
        accept = np.log(u) < log_alpha
        state["t"] = np.where(accept, t_new, state["t"])
        state["l"] = np.where(accept, l_new, state["l"])
        state["log_prior"] = np.where(accept, lp_new, state["log_prior"])
        state["log_lik"] = np.where(accept, ll_new, state["log_lik"])
        accepts.append(float(accept.mean()))
    return float(np.mean(accepts))


def resample_and_rejuvenate_2d(state, rng, n_mh_steps=15, proposal_scale=1.5):
    """Multinomial resample then MCMC rejuvenate, in place, for a single task."""
    N = state["t"].shape[0]
    idx = rng.choice(N, size=N, p=state["w"])
    state["t"] = state["t"][idx]
    state["l"] = state["l"][idx]
    state["log_prior"] = state["log_prior"][idx]
    state["log_lik"] = state["log_lik"][idx]
    state["w"] = np.full(N, 1.0 / N)
    return mh_rejuvenate_2d(state, n_mh_steps, proposal_scale, rng)


# ───────────── item selection ─────────────

def _expected_loss_brute_vec(states, k, signals):
    """Expected total variance over all (t_kk, l_kk) after a question on task k.

    Within-task signal optimization (vectorized over signals); cross-task task
    selection emerges from picking argmin over (k, s).

    Note: variance of tasks kk != k is unchanged by a hypothetical query on
    task k (independent priors + task-local likelihoods → exact factorization).
    So adding them as a constant doesn't affect within-task signal optimization,
    but it DOES matter for the cross-task argmin (loss_k = ΔVar_k + sum_{kk!=k} Var_kk;
    argmin_k loss_k = argmax_k ΔVar_k). Mirrors `core_K.expected_loss_brute_vec_K`.

    FIX-T1.4: predictive probability uses the lapse-mixture P(y=1 | z).
    """
    pset = states[k]
    t = pset["t"]
    l = pset["l"]
    z = np.exp(l)[None, :] * (signals[:, None] + t[None, :])
    p = _p_y1(z)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
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
    for kk, other in enumerate(states):
        if kk == k:
            continue
        mu_t = (other["w"] * other["t"]).sum()
        mu_l = (other["w"] * other["l"]).sum()
        vt = (other["w"] * (other["t"] - mu_t) ** 2).sum()
        vl = (other["w"] * (other["l"] - mu_l) ** 2).sum()
        other_var_sum += vt + vl
    return expected_var_k + other_var_sum


def _coarse_to_fine_argmin(loss_of, n, n_coarse, m_bracket=3):
    """Deterministic top-M coarse-to-fine argmin over ordered candidates.

    Mirrors core_mcmc._coarse_to_fine_argmin (kept local to avoid an
    import cycle: core_mcmc late-imports this module).  See that
    docstring for the algorithm and rationale (single-bracket
    coarse-to-fine misses ~0.19% of steps on locally-bimodal EV
    surfaces and the miss cascades chaotically; refining the M lowest
    coarse cells gives 0.00% disagreement, M=3).  `loss_of(idx_array)`
    returns the loss at those integer indices.  Deterministic (no RNG).
    """
    full = np.arange(n)
    if n_coarse is None or n <= int(n_coarse):
        L = loss_of(full)
        return int(full[int(np.argmin(L))])
    nc = int(n_coarse)
    coarse = np.unique(np.round(np.linspace(0, n - 1, nc)).astype(int))
    Lc = loss_of(coarse)
    h = int(np.ceil(n / nc))
    order = np.argsort(Lc, kind="stable")[:max(1, int(m_bracket))]
    windows = [coarse]
    for j in order:
        c = int(coarse[int(j)])
        windows.append(np.arange(max(0, c - h), min(n - 1, c + h) + 1))
    cand = np.unique(np.concatenate(windows))
    Lcand = loss_of(cand)
    return int(cand[int(np.argmin(Lcand))])


def choose_item_brute_k(states, bank_signals=None, active_domains=None,
                        n_subsample=None, bank_segids=None):
    """Pick (k, s) globally minimizing expected total posterior variance.

    bank_signals: optional list of K arrays (one per domain). If None, uses
    SIGNAL_GRID for all domains. Per-domain arrays may differ in length.
    active_domains: optional list of domain indices to consider (for cert sessions).
    n_subsample: F4.1.  Coarse-to-fine argmin with ~n_subsample coarse
                 points + a local full-resolution refine per domain
                 (~5–6× faster; recovers the exact argmin, no n_q bias).
                 None = full grid.
    bank_segids: optional list of K arrays parallel to bank_signals giving each
                 candidate's seg_id. When provided, the chosen item's seg_id is
                 APPENDED to the return → (k, s, seg_id) — parity with
                 `core_mcmc.choose_item`'s Phase-7 sub-3-C real-rater-lookup
                 extension. None (default) ⇒ (k, s), BYTE-IDENTICAL to pre-edit.
    """
    K = len(states)
    candidates = bank_signals if bank_signals is not None else [SIGNAL_GRID] * K
    domains = active_domains if active_domains is not None else list(range(K))
    best_loss = np.inf
    best_k, best_s = domains[0], float(np.asarray(candidates[domains[0]])[0])
    best_segid = -1
    for k in domains:
        sigs = np.asarray(candidates[k])
        idx = _coarse_to_fine_argmin(
            lambda ii: _expected_loss_brute_vec(states, k, sigs[ii]),
            len(sigs), n_subsample)
        loss = float(_expected_loss_brute_vec(states, k, sigs[idx:idx + 1])[0])
        if loss < best_loss:
            best_loss = loss
            best_k = k
            best_s = float(sigs[idx])
            if bank_segids is not None:
                best_segid = int(np.asarray(bank_segids[k])[idx])
    if bank_segids is not None:
        return best_k, best_s, best_segid
    return best_k, best_s


def random_item_brute_k(states, rng, bank_signals=None, active_domains=None,
                        bank_segids=None):
    """Null baseline: pick (k, s) uniformly at random.

    The TRUE base comparison for the Phase-1 ablation — no adaptive item
    selection at all.  Domain k ~ Uniform(active domains); signal s ~
    Uniform(that domain's bank).  Uses the session `rng` so the sequence
    is deterministic and consistent with the parallel-determinism
    guarantee (a worker seeded with `seed` reproduces it bitwise).

    bank_segids: optional list of K arrays parallel to bank_signals; when
                 given, the chosen seg_id is APPENDED to the return →
                 (k, s, seg_id). None (default) ⇒ (k, s).  RNG consumption is
                 IDENTICAL either way (one integers() per draw), so seeding /
                 bit-reproducibility is unchanged.
    """
    K = len(states)
    candidates = bank_signals if bank_signals is not None else [SIGNAL_GRID] * K
    domains = active_domains if active_domains is not None else list(range(K))
    k = int(domains[rng.integers(0, len(domains))])
    sigs = np.asarray(candidates[k], dtype=float)
    si = int(rng.integers(0, len(sigs)))
    s = float(sigs[si])
    if bank_segids is not None:
        return k, s, int(np.asarray(bank_segids[k])[si])
    return k, s


# ───────────── AUROC CI from K-list state ─────────────

def auroc_ci_brute_k(states, alpha=0.05):
    """Returns (lo, hi) arrays of shape (K,) — per-task AUROC CIs."""
    q = auroc_quantiles_from_particles_brute(states, alphas=(alpha / 2, 1 - alpha / 2))
    return q[:, 0], q[:, 1]


# ───────────── response simulation ─────────────

def simulate_response(s, t_true, l_true, rng):
    """Simulate a response under the lapse-mixture probit model (FIX-T1.4)."""
    z = np.exp(l_true) * (s + t_true)
    p = float(np.clip(_p_y1(z), 0.0, 1.0))
    return int(rng.random() < p)


# ── PUB-CLEANUP[estimation-ladder]: posterior read-out (opt-in; paper-sim only) ──
# These two helpers + the `capture_posterior`/`y_source`/`bank_segids` kwargs on
# run_session_mcmc_brute_k (and the seg_id threading in choose_item_brute_k /
# random_item_brute_k) exist ONLY to support the random→brute→hier estimation
# ablation (NEJM-AI Paper-2). The shipped CORTEX app never calls this driver.
# Strip or guard before the public paper-code release — see
# docs/PUBLICATION_CODE_CLEANUP.md (grep "PUB-CLEANUP[estimation-ladder]").

def _wquantile(values, weights, q):
    """Weighted quantile via linear interpolation on the weighted CDF.

    Mirrors `core.weighted_quantile` (kept local to avoid an import).
    """
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return float(np.interp(q, cw, values[order]))


def _posterior_summary_brute_k(states):
    """Per-task weighted posterior mean/SD/central-95% interval of (ℓ, θ).

    Returns 8 arrays of shape (K,): mean_l, sd_l, q025_l, q975_l, then the
    θ analogues. Used only when `capture_posterior=True`; the K independent
    clouds each contribute their own marginal (no pooling — by construction).
    """
    K = len(states)
    ml = np.empty(K); sl = np.empty(K); q025l = np.empty(K); q975l = np.empty(K)
    mt = np.empty(K); st = np.empty(K); q025t = np.empty(K); q975t = np.empty(K)
    for k in range(K):
        w = states[k]["w"]
        w = w / w.sum()
        l = states[k]["l"]; t = states[k]["t"]
        ml[k] = float((w * l).sum())
        sl[k] = float(np.sqrt(max((w * (l - ml[k]) ** 2).sum(), 0.0)))
        mt[k] = float((w * t).sum())
        st[k] = float(np.sqrt(max((w * (t - mt[k]) ** 2).sum(), 0.0)))
        q025l[k] = _wquantile(l, w, 0.025); q975l[k] = _wquantile(l, w, 0.975)
        q025t[k] = _wquantile(t, w, 0.025); q975t[k] = _wquantile(t, w, 0.975)
    return ml, sl, q025l, q975l, mt, st, q025t, q975t


# ───────────── session driver ─────────────

def run_session_mcmc_brute_k(true_params, K, max_q=400, delta_auroc=0.05,
                              N=2500, seed=0, run_until_max=False,
                              log_trajectory=False, alpha=0.05,
                              n_mh_steps=15, proposal_scale=1.5,
                              ess_threshold_frac=0.5,
                              bank_signals=None, select="ev",
                              n_subsample=None,
                              bank_segids=None, y_source=None,
                              capture_posterior=False):
    """Methodology-compliant brute force session with MCMC-rejuvenation SMC.

    Args:
      true_params: length 2K array (t_0, l_0, t_1, l_1, ..., t_{K-1}, l_{K-1})
      K: number of tasks
      N: particles per task (total particle-effort = K * N)
      proposal_scale: MH step size; 1.5 is near the 2-D optimal (Roberts-Gelman-Gilks)
      ess_threshold_frac: trigger per-task resample+MCMC when that task's ESS < this * N.
                          Only the queried task's ESS is checked each step.
      select: "ev" (default, Global-EV adaptive `choose_item_brute_k`) or
              "random" (uniform domain+signal `random_item_brute_k` — the
              true null baseline; same independent posterior, no adaptive
              selection).  `method="random"` in `run_session_mcmc_auroc`
              forwards here with select="random".
      bank_segids: optional list of K arrays parallel to bank_signals giving
              each candidate's seg_id.  Brings this path to parity with
              `core_mcmc.run_session_mcmc_auroc`'s hier real-rater-lookup
              interface — when given, the chosen seg_id is passed to `y_source`.
              None (default) ⇒ unchanged.
      y_source: optional callable `y_source(k, seg_id, s) → int Y ∈ {0,1}` used
              to obtain the response instead of the default `simulate_response`
              draw at true_params.  Mirrors `core_mcmc.run_session_mcmc_auroc`.
              None (default) ⇒ BYTE-IDENTICAL Bernoulli draw at true_params.
      capture_posterior: if True, record per-question weighted posterior
              mean/sd/central-95% interval of (ℓ, θ) per task into the output
              (`mean_l_traj`/`sd_l_traj`/`q025_l_traj`/`q975_l_traj` + `_t_`
              analogues, shape (n_q+1, K)).  Off (default) ⇒ output unchanged.

    Returns dict matching `core_mcmc.run_session_mcmc_auroc`'s output (same keys).
    The `method` key in the returned dict reflects `select`.
    """
    rng = np.random.default_rng(seed)
    assert len(true_params) == 2 * K
    if select not in ("ev", "random"):
        raise ValueError(f"select must be 'ev' or 'random'; got {select!r}")

    states = make_state_brute_k(N, K, rng)
    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    lo, hi = auroc_ci_brute_k(states, alpha)
    if log_trajectory:
        los, his = [lo.copy()], [hi.copy()]
    if capture_posterior:
        ml, sl, q025l, q975l, mt, st, q025t, q975t = _posterior_summary_brute_k(states)
        pml, psl, pq025l, pq975l = [ml], [sl], [q025l], [q975l]
        pmt, pst, pq025t, pq975t = [mt], [st], [q025t], [q975t]
    n_q = 0
    stop_step = None
    accept_rates = []

    for q in range(max_q):
        if select == "random":
            sel = random_item_brute_k(states, rng, bank_signals,
                                      bank_segids=bank_segids)
        else:
            sel = choose_item_brute_k(states, bank_signals,
                                      n_subsample=n_subsample,
                                      bank_segids=bank_segids)
        if bank_segids is not None:
            k, s, seg_id = sel
        else:
            k, s = sel
            seg_id = None
        if y_source is None:
            t_true = true_params[k * 2]
            l_true = true_params[k * 2 + 1]
            y = simulate_response(s, t_true, l_true, rng)
        else:
            y = int(y_source(k, seg_id, s))
        update_brute_k(states, k, s, y)
        if ess(states[k]["w"]) < ess_threshold_frac * N:
            ar = resample_and_rejuvenate_2d(states[k], rng, n_mh_steps, proposal_scale)
            accept_rates.append(ar)
        n_q += 1
        lo, hi = auroc_ci_brute_k(states, alpha)
        if log_trajectory:
            los.append(lo.copy())
            his.append(hi.copy())
        if capture_posterior:
            ml, sl, q025l, q975l, mt, st, q025t, q975t = _posterior_summary_brute_k(states)
            pml.append(ml); psl.append(sl); pq025l.append(q025l); pq975l.append(q975l)
            pmt.append(mt); pst.append(st); pq025t.append(q025t); pq975t.append(q975t)
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
        "mean_acceptance_rate": float(np.mean(accept_rates)) if accept_rates else float("nan"),
        "n_rejuvenations": len(accept_rates),
        "delta_auroc": float(delta_auroc),
        "method": "random" if select == "random" else "brute",
    }
    if log_trajectory:
        out["lo_traj"] = np.array(los)
        out["hi_traj"] = np.array(his)
    if capture_posterior:
        out["mean_l_traj"] = np.array(pml); out["sd_l_traj"] = np.array(psl)
        out["q025_l_traj"] = np.array(pq025l); out["q975_l_traj"] = np.array(pq975l)
        out["mean_t_traj"] = np.array(pmt); out["sd_t_traj"] = np.array(pst)
        out["q025_t_traj"] = np.array(pq025t); out["q975_t_traj"] = np.array(pq975t)
    return out


# ───────────── certification support for brute-K state ─────────────

def _pass_probs_brute(states, l_star):
    """P(l_k > l*_k) for each domain from K independent particle clouds."""
    return np.array([(states[k]["w"] * (states[k]["l"] > l_star[k])).sum()
                     for k in range(len(states))])


def _pass_prob_joint_brute(states, l_star, active_domains):
    """P(all active l_k > l*_k) for the brute model. The K clouds are
    INDEPENDENT, so the joint conjunction probability is exactly the
    product of the active per-domain marginals (unified-merge IUT
    'joint' rule, brute parity with engine_mode_b._pass_prob_joint)."""
    pp = _pass_probs_brute(states, l_star)
    out = 1.0
    for k in active_domains:
        out *= float(pp[k])
    return out


def _n_eff_per_domain(states):
    """Per-domain effective sample size, N_eff_k = (sum w_k)^2 / sum(w_k^2).

    Used by FIX-T1.3 MCSE-buffered stopping. Weights are normalized in
    `update_brute_k`, so (sum w)^2 ≈ 1 and N_eff = 1 / sum(w^2). We compute
    the ratio explicitly for safety (avoids reliance on perfect normalization).
    """
    out = np.empty(len(states))
    for k, st in enumerate(states):
        w = st["w"]
        s1 = float(w.sum())
        s2 = float((w * w).sum())
        out[k] = (s1 * s1) / s2 if s2 > 0 else 0.0
    return out


# Note: KL-based item selection (`_kl_vec_brute`, `choose_item_kl_brute`) was
# removed on 2026-05-14 (FIX-T0.5) — see STATE_LOG_2026_05_13.md §3.1 for the
# fatal flaw (uses θ̂ as a fixed point estimate, picks zero-Fisher items
# whenever θ̂ ≠ θ_true). Use `choose_item_brute_k` instead.


def run_session_mcmc_brute_k_cert(
    true_params, K, l_star, max_q=400,
    stop_thresh=None, alpha=0.05, delta_indiff=0.15,
    N=500, seed=0, n_mh_steps=15, proposal_scale=1.5,
    ess_threshold_frac=0.5, bank_signals=None, virtual_pairs=None,
    n_min_guard=20,
    use_iut_stopping=True,
    iut_rule="joint",          # unified-merge: parity with engine_mode_b
    fisher_imin=0.0,
):
    """Brute-force (K independent clouds) certification session.

    Brute uses K independent 2-D N(0,1) priors per (t_k, l_k) — no cross-domain
    information sharing.  This is the methodology-baseline comparator for hier.

    Stopping (parity with core_mcmc.run_session_mcmc_certification):
        PASS (per-domain): p_k - z_buffer * mcse_k >= stop_thresh
        FAIL (per-domain): p_k + z_buffer * mcse_k <= 1 - stop_thresh,
                           AND per_domain_n[k] >= n_min_guard,
                           AND domain_fi[k] >= fisher_imin
        IUT PASS (Berger 1982): if use_iut_stopping=True, the test PASSes
                                jointly once ALL active per-domain marginals
                                clear stop_thresh simultaneously.
    """
    rng = np.random.default_rng(seed)
    assert len(true_params) == 2 * K
    l_star = np.asarray(l_star, dtype=float)
    if iut_rule not in ("joint", "berger_marginals"):
        raise ValueError(
            f"iut_rule must be 'joint' (default) or 'berger_marginals'; "
            f"got {iut_rule!r}.")

    if stop_thresh is None:
        stop_thresh = (1.0 - alpha) ** (1.0 / K)  # Sidak correction

    states = make_state_brute_k(N, K, rng)
    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    active_domains = list(range(K))
    decisions = np.zeros(K, dtype=int)
    per_domain_n = np.zeros(K, dtype=int)
    domain_fi = np.zeros(K, dtype=float)        # FIX-T1.11 parity
    virtual_granted = np.zeros(K, dtype=bool)
    n_q = 0
    accept_rates = []

    # F0.1: Fisher info for ℓ under spike-paper Eq. 2 mixture P = λ + (1−2λ)Φ(z).
    _ONE_MINUS_TWO_LAPSE_SQ = (1.0 - 2.0 * LAPSE_RATE) ** 2

    for _ in range(max_q):
        if not active_domains:
            break

        k, s = choose_item_brute_k(states, bank_signals, active_domains)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update_brute_k(states, k, s, y)
        per_domain_n[k] += 1
        n_q += 1

        # FIX-T1.11 + F0.1 parity: accumulate Fisher info for l at l*_k.
        w_k = states[k]["w"]
        t_hat_k = float(w_k @ states[k]["t"])
        z_fi = float(np.exp(l_star[k]) * (s + t_hat_k))
        phi_fi = float(norm.pdf(z_fi))
        p_fi = float(LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z_fi))
        domain_fi[k] += (_ONE_MINUS_TWO_LAPSE_SQ * phi_fi ** 2 * z_fi ** 2
                         / max(p_fi * (1.0 - p_fi), 1e-9))

        if ess(states[k]["w"]) < ess_threshold_frac * N:
            ar = resample_and_rejuvenate_2d(states[k], rng, n_mh_steps, proposal_scale)
            accept_rates.append(ar)

        pass_probs = _pass_probs_brute(states, l_star)
        # FIX-T1.3: per-domain MCSE for buffered stopping.
        n_eff = _n_eff_per_domain(states)
        mcse = np.sqrt(np.maximum(pass_probs * (1.0 - pass_probs), 0.0)
                       / np.maximum(n_eff, 1.0))

        if virtual_pairs:
            for (src_k, tgt_k, src_thresh) in virtual_pairs:
                if (tgt_k in active_domains
                        and pass_probs[src_k] - Z_BUFFER * mcse[src_k] >= src_thresh):
                    decisions[tgt_k] = 1
                    virtual_granted[tgt_k] = True
                    active_domains.remove(tgt_k)

        newly_decided = []

        # Unified-merge IUT toggle (parity with engine_mode_b): default
        # "joint" (audit-endorsed) = product of active marginals (brute
        # clouds are independent), MCSE-buffered with the conservative
        # min active per-domain N_eff; "berger_marginals" = FIX-T1.9
        # min-of-marginals, preserved verbatim.
        if use_iut_stopping and active_domains:
            if iut_rule == "joint":
                p_joint = _pass_prob_joint_brute(states, l_star,
                                                 active_domains)
                n_eff_j = min(max(n_eff[k], 1.0) for k in active_domains)
                mcse_j = np.sqrt(
                    max(p_joint * (1.0 - p_joint), 0.0) / n_eff_j)
                conjunction_pass = (
                    p_joint - Z_BUFFER * mcse_j >= stop_thresh)
            else:  # "berger_marginals" — FIX-T1.9 verbatim
                conjunction_pass = all(
                    pass_probs[k_act] - Z_BUFFER * mcse[k_act] >= stop_thresh
                    for k_act in active_domains
                )
            if conjunction_pass:
                for k_act in list(active_domains):
                    decisions[k_act] = 1
                    newly_decided.append(k_act)
        else:
            for k_act in list(active_domains):
                if pass_probs[k_act] - Z_BUFFER * mcse[k_act] >= stop_thresh:
                    decisions[k_act] = 1
                    newly_decided.append(k_act)

        # Per-domain FAIL with FIX-T1.7 + T1.11 guards.
        for k_act in list(active_domains):
            if k_act in newly_decided:
                continue
            if pass_probs[k_act] + Z_BUFFER * mcse[k_act] <= 1.0 - stop_thresh:
                fi_ok = (domain_fi[k_act] >= fisher_imin) if fisher_imin > 0 else True
                if fi_ok and per_domain_n[k_act] >= n_min_guard:
                    decisions[k_act] = -1
                    newly_decided.append(k_act)

        for k_dec in newly_decided:
            active_domains.remove(k_dec)

    pass_probs_final = _pass_probs_brute(states, l_star)
    n_eff_final = _n_eff_per_domain(states)
    mcse_final = np.sqrt(
        np.maximum(pass_probs_final * (1.0 - pass_probs_final), 0.0)
        / np.maximum(n_eff_final, 1.0)
    )

    return {
        "n_questions": int(n_q),
        "per_domain_n": per_domain_n,
        "decisions": decisions,
        "stopped_early": len(active_domains) == 0,
        "active_at_end": list(active_domains),
        "true_auroc": true_auroc,
        "pass_probs_final": pass_probs_final,
        "virtual_granted": virtual_granted,
        "stop_thresh_used": float(stop_thresh),
        "mean_acceptance_rate": float(np.mean(accept_rates)) if accept_rates else float("nan"),
        "n_rejuvenations": len(accept_rates),
        "mcse_final": mcse_final,
        "z_buffer": float(Z_BUFFER),
        "n_min_guard": int(n_min_guard),
        "domain_fi": domain_fi,
        "use_iut_stopping": bool(use_iut_stopping),
    }
