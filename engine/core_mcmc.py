"""SMC sampler with Metropolis-Hastings rejuvenation, AUROC-based stopping.

Certification extension (run_session_mcmc_certification):
  - Binary pass/fail stopping: P(l_k > l*_k | data) >= stop_thresh (Sidak-corrected),
    with Monte-Carlo standard-error buffer (Z_BUFFER * mcse) to guard against
    finite-N particle noise.
  - Item selection: expected posterior-variance reduction (EV); marginalises over
    the FULL posterior in (t, l) for ALL domains.  KL-based selection has been
    REMOVED (it conditioned on a point estimate of theta_hat_k and selected
    near-zero-Fisher-information items whenever theta_hat differed from theta_true;
    see STATE_LOG_2026_05_13 §3.1).
  - Active-domain tracking: stops asking questions in decided domains.
  - The virtual sz->grda certification mechanism has been REMOVED.  The relevant
    quantity for virtual-cert is the per-examinee POSTERIOR correlation
    Corr(l_sz, l_grda | data), bounded above by the prior r (~0.378), NOT the
    cross-rater population correlation.  Under the prior the correctly derived
    bound P(l_grda > l*) >= Phi(0.38 * 1.645) ~= 0.73 is BELOW threshold.  The
    joint particle cloud's pass_probs[grda] subsumes any virtual-cert update
    automatically through the proper Bayesian update via the prior covariance.

Replaces the West kernel-jitter resampling (core.py / core_K.py) with proper
MCMC moves targeting the current posterior.  Preserves the posterior support
exactly — no Gaussian-blob drift over many resampling steps.

Per-particle state additions vs the kernel-jitter SMC:
  log_prior   - cached log prior density at the particle's current position
  log_lik     - cumulative log likelihood up to the current observation count

The particle's log posterior is log_prior + log_lik.  On reweighting after an
observation, log_lik is incremented in place.  On MCMC moves, the acceptance
ratio uses the cached values, with the likelihood at the proposed position
computed by sweeping the question history (vectorized over particles).

Hierarchical prior on (t_1,...,t_K, l_1,...,l_K):
  Default: unstructured 6x6 covariance Sigma_l (and Sigma_t) fit from the
  empirical SPARCNET cross-domain rater matrix
  (Sigma_l_fitted.npy).  Backward-compat: a compound-symmetry fallback
  Sigma = (1-r) I + r J is used when r_assumed is supplied without an explicit
  Sigma matrix.  Note: the binary stopping rule is a Bayesian
  posterior-threshold test under adaptive sampling; it is NOT a Wald SPRT
  (the boundary log-odds = log(stop_thresh / (1-stop_thresh)) coincide
  numerically, but the operating characteristics differ under adaptive
  question selection).

Response model includes a uniform lapse rate (Spike paper CLAUDE.md §2.1):
  P(y=1 | c, l, theta; lambda) = (1 - lambda) Phi(exp(l) (c + theta)) + 0.5 lambda

Last revised 2026-05-14: applied Tier-0/Tier-1 corrections from reviewer panel
  (T0.5 dead code, T0.7 log_ndtr, T1.1 virtual-cert removal, T1.2 docstring,
  T1.3 MCSE buffer, T1.4 lapse rate, T1.6 unstructured Sigma, T1.7 Corr_l
  prior-scale fix + N_min guard, T1.8 boundary prior, T1.9 IUT stopping,
  T1.10 boundary item selection, T1.11 Fisher info guard).

FIX-T1.7 — prior scale misspecification (added 2026-05-14):
  The fitted Sigma_l has marginal std ~ 0.10-0.24 (inter-rater spread), NOT
  the scale of plausible examinee log-discrimination skills.  Using it as a
  zero-mean prior places l*_k at 2-4 prior SDs above zero, causing FAIL to
  fire after q=1 for skilled examinees.  The bridge now uses Corr_l (unit
  diagonal) from Sigma_l_fitted.npy.  Additionally, run_session_mcmc_cert-
  ification accepts n_min_guard (default 20): FAIL is blocked until that many
  questions have been asked in a domain, preventing prior-induced premature
  decisions even if use_corr_l is inadvertently disabled.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr, logsumexp  # FIX-T0.7: numerically stable log Phi

from auroc import (
    auroc_from_l,
    auroc_quantiles_from_particles_hier,
    auroc_quantiles_from_particles_brute,
)


# ───────────── module constants ─────────────
LAPSE_RATE = 0.025          # FIX-T1.4 + F0.1: matches spike paper CLAUDE.md §2.1 Eq. 2
Z_BUFFER = 2.0              # FIX-T1.3: ~95% MC envelope around pass_probs[k]
# F0.1 (2026-05-15): spike-paper-canonical symmetric lapse mixture.
# P(y=1|z) = λ + (1 − 2λ)·Φ(z);  floor = λ, ceiling = 1 − λ.
# Previous parametrization `(1−λ)·Φ + 0.5λ` (floor = λ/2, ceiling = 1 − λ/2)
# corresponded to a 4AFC-style guess rate and is algebraically inconsistent
# with `train_val_split_and_fit.py:probit_lapse_nll`.
_LOG_LAPSE = float(np.log(LAPSE_RATE))
_LOG_ONE_MINUS_TWO_LAPSE = float(np.log1p(-2.0 * LAPSE_RATE))


def _log_p_response(z, y):
    """log P(y | z; LAPSE_RATE) with z = exp(l) (c + theta).

    FIX-T0.7 + FIX-T1.4 + F0.1: uses scipy.special.log_ndtr for log Phi and
    logsumexp to mix in the lambda lapse floor (spike paper Eq. 2):
        P(y=1|z) = λ + (1 − 2λ)·Φ(z)
        P(y=0|z) = λ + (1 − 2λ)·Φ(−z)

    Accepts z as scalar or array; y in {0, 1}.  Returns shape of z.
    """
    z = np.asarray(z, dtype=np.float64)
    if y == 1:
        a = _LOG_ONE_MINUS_TWO_LAPSE + log_ndtr(z)
    else:
        a = _LOG_ONE_MINUS_TWO_LAPSE + log_ndtr(-z)
    # logsumexp(a, log(lambda)) elementwise
    b = np.full_like(z, _LOG_LAPSE)
    return logsumexp(np.stack([a, b], axis=0), axis=0)


def _p_response_yes(z):
    """P(y=1 | z; LAPSE_RATE), vectorized.  Used for item-selection moments.

    F0.1: spike-paper Eq. 2 form: P(y=1|z) = λ + (1 − 2λ)·Φ(z).
    """
    z = np.asarray(z, dtype=np.float64)
    return LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z)


# ───────────── log prior densities (vectorized over N particles) ─────────────

# FIX-T1.6: regression checkpoint — empirical correlations from
# cross_domain_rater_matrix.csv (15 SPARCNET raters, columns l_{sz,lpd,gpd,lrda,grda,iic}):
#   rho(l_sz, l_grda) = 0.6481
#   rho(l_sz, l_lpd ) = 0.5379
#   rho(l_sz, l_gpd ) = 0.4120
#   rho(l_sz, l_lrda) = 0.1668
#   rho(l_sz, l_iic ) = 0.4302
# Mean off-diagonal corr  ≈ 0.218 (compound-symmetry r ≈ 0.378 used previously
# was an over-statement of inter-task pooling for some pairs; sz<->grda is the
# strongest pair but still well below the per-rater value of 0.974 quoted in
# STATE_LOG §4.1, which conflated cross-rater and within-rater statistics).
# The current default prior uses the unstructured Sigma loaded from
# Sigma_l_fitted.npy when available.

def _compound_symmetry_cov(r, K):
    """Sigma = (1-r) I + r J, the prior used pre-T1.6.  r in [0, 1)."""
    if r >= 1.0 - 1e-9:
        r = 1.0 - 1e-9
    return (1.0 - r) * np.eye(K) + r * np.ones((K, K))


def _log_mvn_zero_mean(x, Sigma_inv, log_det):
    """log N(x | 0, Sigma) up to constant.  x: (N, K).  Returns (N,)."""
    # quadratic form rowwise: x Sigma^{-1} x^T diagonal
    qf = np.einsum("ni,ij,nj->n", x, Sigma_inv, x)
    return -0.5 * qf - 0.5 * log_det  # the (K/2) log(2 pi) term cancels in MH ratio


def _precompute_prior_pieces(Sigma_l, Sigma_t, K):
    """Cache (Sigma_inv, log_det) for both blocks, with jitter for stability."""
    jitter = 1e-9 * np.eye(K)
    Sl = np.asarray(Sigma_l, dtype=np.float64) + jitter
    St = np.asarray(Sigma_t, dtype=np.float64) + jitter
    Sl_inv = np.linalg.inv(Sl)
    St_inv = np.linalg.inv(St)
    sign_l, ldet_l = np.linalg.slogdet(Sl)
    sign_t, ldet_t = np.linalg.slogdet(St)
    if sign_l <= 0 or sign_t <= 0:
        raise ValueError("Sigma matrices must be positive definite.")
    L_l = np.linalg.cholesky(Sl)
    L_t = np.linalg.cholesky(St)
    return {
        "Sigma_l": Sl, "Sigma_t": St,
        "Sigma_l_inv": Sl_inv, "Sigma_t_inv": St_inv,
        "log_det_l": ldet_l, "log_det_t": ldet_t,
        "L_l": L_l, "L_t": L_t,
    }


def log_prior_hier(t, l, r, K, Sigma_l=None, Sigma_t=None, _pieces=None, l_prior_mean=None):
    """log p(t, l) under hierarchical prior.

    FIX-T1.6: now supports an unstructured Sigma_l (and Sigma_t).  Backward-
    compat: if Sigma_l is None, falls back to compound-symmetry Sigma = (1-r) I + r J.
    FIX-T1.8: l_prior_mean shifts the l block; evaluates log N(l | l_prior_mean, Sigma_l).

    t, l shape (N, K).  Returns shape (N,) (up to a constant that cancels in the
    Metropolis ratio).
    """
    l_c = l if l_prior_mean is None else (l - np.asarray(l_prior_mean, dtype=float))
    if _pieces is not None:
        return _log_mvn_zero_mean(t, _pieces["Sigma_t_inv"], _pieces["log_det_t"]) + \
               _log_mvn_zero_mean(l_c, _pieces["Sigma_l_inv"], _pieces["log_det_l"])
    if Sigma_l is None:
        Sigma_l = _compound_symmetry_cov(r, K)
    if Sigma_t is None:
        Sigma_t = Sigma_l  # historically Sigma_t = Sigma_l under symmetry
    pieces = _precompute_prior_pieces(Sigma_l, Sigma_t, K)
    return _log_mvn_zero_mean(t, pieces["Sigma_t_inv"], pieces["log_det_t"]) + \
           _log_mvn_zero_mean(l_c, pieces["Sigma_l_inv"], pieces["log_det_l"])


def log_prior_brute_K(t, l, K):
    """Brute force: independent N(0,1) per parameter. Shape (N, K) t, l."""
    return -0.5 * ((t * t).sum(axis=1) + (l * l).sum(axis=1))


# ───────────── Sigma loader (FIX-T1.6) ─────────────

import os as _os

# Verbatim from the methodology repo. engine/Sigma_l_fitted.npy is a
# symlink to the single frozen repo-root artifact (Phase 2; zero drift —
# the file is immutable). Phase 3/5 migrate the engine to PI's fitted Sigma.
_SIGMA_FITTED_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "Sigma_l_fitted.npy"
)


def load_fitted_Sigma(path=None):
    """Load fitted Sigma_l, Sigma_t from Sigma_l_fitted.npy (FIX-T1.6).

    Returns dict with keys 'Sigma_l', 'Sigma_t' (and metadata).  Raises
    FileNotFoundError if absent.
    """
    p = path if path is not None else _SIGMA_FITTED_PATH
    obj = np.load(p, allow_pickle=True).item()
    return obj


# ───────────── log likelihood given history (vectorized) ─────────────

def _log_lik_history(t, l, history, history_s_sd=None):
    """Given particles (N, K), compute cumulative log P(history | particle).

    history: list of (k, s, y) tuples.  Returns shape (N,).
    history_s_sd: optional parallel list of per-question s posterior SDs
        (Phase 3.5).  None ⇒ all 0.0 ⇒ z BIT-IDENTICAL to the prior engine
        (so rejuvenation-replay stays consistent with `update`, and the
        whole validated suite is unchanged when s_sd is absent).

    FIX-T0.7 + FIX-T1.4: uses log_ndtr and the lapse-rate response model.
    Phase 3.5: closed-form s-marginalization, z attenuated by
    1/√(1+(e^ℓ·s_sd)²) — same formula as engine.core._marg_z.
    """
    N = t.shape[0]
    ll = np.zeros(N)
    sds = history_s_sd if history_s_sd is not None else [0.0] * len(history)
    for (k, s, y), s_sd in zip(history, sds):
        el = np.exp(l[:, k])
        z = el * (s + t[:, k])
        if s_sd != 0.0:
            z = z / np.sqrt(1.0 + (el * s_sd) ** 2)
        ll += _log_p_response(z, y)
    return ll


# ───────────── prior sampling ─────────────

def sample_prior_hier_K(N, K, r, rng, Sigma_l=None, Sigma_t=None, _pieces=None, l_prior_mean=None):
    """Sample from the hierarchical prior. Returns t, l of shape (N, K).

    FIX-T1.6: when Sigma_l (and/or Sigma_t) is provided, samples from
    N(0, Sigma) for each block via Cholesky.  Falls back to compound-symmetry
    Sigma = (1-r) I + r J when Sigma_l is None.
    FIX-T1.8: l_prior_mean shifts the l block to N(l_prior_mean, Sigma_l).
    """
    if _pieces is not None:
        eps_t = rng.standard_normal((N, K))
        eps_l = rng.standard_normal((N, K))
        t = eps_t @ _pieces["L_t"].T
        l = eps_l @ _pieces["L_l"].T
    elif Sigma_l is None:
        # legacy compound-symmetry path (faithful to pre-T1.6 behavior)
        lam = np.sqrt(r)
        psi = np.sqrt(1.0 - r)
        G_bias = rng.standard_normal(N)
        G_skill = rng.standard_normal(N)
        eps_t = rng.standard_normal((N, K))
        eps_l = rng.standard_normal((N, K))
        t = lam * G_bias[:, None] + psi * eps_t
        l = lam * G_skill[:, None] + psi * eps_l
    else:
        if Sigma_t is None:
            Sigma_t = Sigma_l
        pieces = _precompute_prior_pieces(Sigma_l, Sigma_t, K)
        eps_t = rng.standard_normal((N, K))
        eps_l = rng.standard_normal((N, K))
        t = eps_t @ pieces["L_t"].T
        l = eps_l @ pieces["L_l"].T
    if l_prior_mean is not None:
        l = l + np.asarray(l_prior_mean, dtype=float)
    return t, l


def sample_prior_brute_K(N, K, rng):
    """Independent N(0,1) per task per parameter."""
    return rng.standard_normal((N, K)), rng.standard_normal((N, K))


# ───────────── state object ─────────────

def make_state_hier(N, K, r_assumed, rng, Sigma_l=None, Sigma_t=None, l_prior_mean=None):
    """Initialise particles from hierarchical prior.

    FIX-T1.6: Sigma_l (and Sigma_t) override r_assumed.  When neither is given,
    falls back to compound-symmetry Sigma = (1-r) I + r J.
    FIX-T1.8: l_prior_mean shifts the l prior mean; pass l_star for boundary prior.
    """
    pieces = None
    if Sigma_l is not None:
        if Sigma_t is None:
            Sigma_t = Sigma_l
        pieces = _precompute_prior_pieces(Sigma_l, Sigma_t, K)
    t, l = sample_prior_hier_K(N, K, r_assumed, rng,
                                Sigma_l=Sigma_l, Sigma_t=Sigma_t, _pieces=pieces,
                                l_prior_mean=l_prior_mean)
    lp = log_prior_hier(t, l, r_assumed, K,
                         Sigma_l=Sigma_l, Sigma_t=Sigma_t, _pieces=pieces,
                         l_prior_mean=l_prior_mean)
    return {
        "t": t, "l": l,
        "w": np.full(N, 1.0 / N),
        "log_prior": lp,
        "log_lik": np.zeros(N),
        "history": [],
        "r_assumed": r_assumed,
        "K": K,
        "is_hier": True,
        "_prior_pieces": pieces,      # may be None (legacy CS path)
        "_l_prior_mean": l_prior_mean,  # FIX-T1.8: stored for MH rejuvenation
    }


def make_state_brute(N, K, rng):
    t, l = sample_prior_brute_K(N, K, rng)
    return {
        "t": t, "l": l,
        "w": np.full(N, 1.0 / N),
        "log_prior": log_prior_brute_K(t, l, K),
        "log_lik": np.zeros(N),
        "history": [],
        "K": K,
        "is_hier": False,
        "_prior_pieces": None,
    }


def _log_prior_of(state, t, l):
    if state["is_hier"]:
        return log_prior_hier(t, l, state["r_assumed"], state["K"],
                               _pieces=state.get("_prior_pieces"),
                               l_prior_mean=state.get("_l_prior_mean"))
    return log_prior_brute_K(t, l, state["K"])


# ───────────── reweight on new observation ─────────────

def update(state, k, s, y, s_sd=0.0):
    """Reweight particles by likelihood of (k, s, y).  Update history & log_lik.

    FIX-T0.7 + FIX-T1.4: log_ndtr-based likelihood with lapse-rate mixture.
    Phase 3.5: s_sd = posterior SD of the segment signal; the likelihood is
    marginalized over s ~ N(s, s_sd²) via z /= √(1+(e^ℓ·s_sd)²) (same
    formula as engine.core._marg_z). s_sd=0.0 (default) ⇒ BIT-IDENTICAL to
    the prior engine; s_sd is also recorded in a parallel history list so
    rejuvenation-replay stays consistent.
    """
    el = np.exp(state["l"][:, k])
    z = el * (s + state["t"][:, k])
    if s_sd != 0.0:
        z = z / np.sqrt(1.0 + (el * s_sd) ** 2)
    log_p_obs = _log_p_response(z, y)
    state["log_lik"] += log_p_obs
    w = state["w"] * np.exp(log_p_obs)
    s_w = w.sum()
    if s_w <= 0:
        # numerical fallback — reset to uniform
        state["w"] = np.full_like(state["w"], 1.0 / len(state["w"]))
    else:
        state["w"] = w / s_w
    state["history"].append((int(k), float(s), int(y)))
    # Phase 3.5: parallel additive list (NOT a 4-tuple — keeps bridge/
    # audit/tests that unpack (k,s,y) ripple-free). All-zeros ⇒ bit-exact.
    state.setdefault("history_s_sd", []).append(float(s_sd))


def ess(w):
    return 1.0 / (w * w).sum()


# ───────────── MCMC rejuvenation ─────────────

def mh_rejuvenate(state, n_steps, proposal_scale, rng):
    """Run n_steps of Metropolis-Hastings on each particle in parallel.

    Proposal: random-walk Gaussian with covariance proposal_scale^2 * cov_cloud.
    Acceptance: min(1, exp(log_post_new - log_post_old)) using cached log_prior
                and log_lik (recomputed at proposal from full history).

    Returns average acceptance rate over the n_steps for diagnostics.
    """
    N, K = state["t"].shape
    accepts = []
    for _ in range(n_steps):
        theta = np.column_stack([state["t"], state["l"]])  # (N, 2K)
        cov = np.cov(theta, rowvar=False)
        if cov.ndim == 0:
            cov = np.atleast_2d(cov)
        cov = cov + 1e-6 * np.eye(2 * K)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            L = eigvecs * np.sqrt(eigvals)
        eps = rng.standard_normal((N, 2 * K))
        theta_new = theta + proposal_scale * (eps @ L.T)
        t_new = theta_new[:, :K]
        l_new = theta_new[:, K:]

        lp_new = _log_prior_of(state, t_new, l_new)
        ll_new = _log_lik_history(t_new, l_new, state["history"],
                                  state.get("history_s_sd"))
        log_alpha = (lp_new + ll_new) - (state["log_prior"] + state["log_lik"])
        u = rng.random(N)
        accept = np.log(u) < log_alpha
        state["t"] = np.where(accept[:, None], t_new, state["t"])
        state["l"] = np.where(accept[:, None], l_new, state["l"])
        state["log_prior"] = np.where(accept, lp_new, state["log_prior"])
        state["log_lik"] = np.where(accept, ll_new, state["log_lik"])
        accepts.append(float(accept.mean()))
    return float(np.mean(accepts))


def resample_and_rejuvenate(state, rng, n_mh_steps=15, proposal_scale=0.5,
                              diag_callback=None, q_index=None):
    """Multinomial resample then run MCMC rejuvenation.

    F2.5: Optional `diag_callback(record)` receives a dict summarising the
    rejuvenation event (accept_rate, lag-1 autocorrelation per coord, ESS).
    The callback runs only if non-None; otherwise the function is unchanged.
    """
    N = state["t"].shape[0]
    idx = rng.choice(N, size=N, p=state["w"])
    state["t"] = state["t"][idx]
    state["l"] = state["l"][idx]
    state["log_prior"] = state["log_prior"][idx]
    state["log_lik"] = state["log_lik"][idx]
    state["w"] = np.full(N, 1.0 / N)
    # F2.5: capture pre-MCMC clouds for lag-1 autocorr diagnostic
    if diag_callback is not None:
        t_before = state["t"].copy()
        l_before = state["l"].copy()
    # MCMC rejuvenation
    accept_rate = mh_rejuvenate(state, n_mh_steps, proposal_scale, rng)
    if diag_callback is not None:
        from diagnostics import rejuvenation_event_diag
        try:
            event_id = state.setdefault("_diag_event_counter", 0)
            state["_diag_event_counter"] += 1
            record = rejuvenation_event_diag(
                event_id=event_id,
                q_index=int(q_index) if q_index is not None else -1,
                n_mh_steps=int(n_mh_steps),
                accept_rate=float(accept_rate),
                t_before=t_before, t_after=state["t"],
                l_before=l_before, l_after=state["l"],
                weights_after=state["w"],
            )
            diag_callback(record)
        except Exception as e:
            # Diagnostics failures must not break the engine.
            import warnings
            warnings.warn(f"resample_and_rejuvenate diag_callback failed: {e}",
                          RuntimeWarning)
    return accept_rate


# ───────────── adaptive item selection (re-uses AUROC + posterior var) ─────────────

SIGNAL_GRID = np.linspace(-3.0, 3.0, 11)


def _expected_loss_vec(state, k, signals, signal_sds=None):
    """Expected total posterior variance over (t_1..t_K, l_1..l_K) after one
    question on task k at each signal level. Vectorized across signals.

    FIX-T1.4: response probability incorporates the lapse-rate mixture.
    Phase 3.5: signal_sds (per-candidate s posterior SD) marginalizes each
    candidate item over its s posterior so item selection prefers
    high-precision items; signal_sds=None (default) ⇒ BIT-IDENTICAL.
    """
    K = state["t"].shape[1]
    t_k = state["t"][:, k]
    l_k = state["l"][:, k]
    el = np.exp(l_k)[None, :]
    z = el * (signals[:, None] + t_k[None, :])
    if signal_sds is not None:
        z = z / np.sqrt(1.0 + (el * np.asarray(signal_sds)[:, None]) ** 2)
    p = _p_response_yes(z)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    w = state["w"]
    p_yes = (p * w).sum(axis=1)
    w_y1 = p * w
    w_y1 = w_y1 / w_y1.sum(axis=1, keepdims=True)
    w_y0 = (1.0 - p) * w
    w_y0 = w_y0 / w_y0.sum(axis=1, keepdims=True)

    def var_vec(arr, weights):
        mu = (weights * arr).sum(axis=1)
        return (weights * (arr - mu[:, None]) ** 2).sum(axis=1)

    total_y1 = sum(var_vec(state["t"][:, kk], w_y1) for kk in range(K)) + \
               sum(var_vec(state["l"][:, kk], w_y1) for kk in range(K))
    total_y0 = sum(var_vec(state["t"][:, kk], w_y0) for kk in range(K)) + \
               sum(var_vec(state["l"][:, kk], w_y0) for kk in range(K))
    return p_yes * total_y1 + (1.0 - p_yes) * total_y0


def _coarse_to_fine_argmin(loss_of, n, n_coarse, m_bracket=3):
    """Deterministic top-M coarse-to-fine argmin over ordered candidates.

    F4.1.  Pure coarse subsampling is near-optimal *per question* but its
    small per-step errors cascade chaotically over a session (a single
    different pick reshapes the posterior and decorrelates every
    downstream choice → empirically +43% n_q at n_coarse=40 on a 250-item
    bank, biasing the OC/simulation estimates).  Single-bracket
    coarse-to-fine still misses ~0.19% of steps when the
    expected-loss-vs-difficulty surface is locally bimodal (the coarse
    grid can land its minimum in the wrong basin).  Instead:

      Stage 1 — evaluate the loss on a coarse evenly-spaced grid of
                ~n_coarse indices spanning [0, n-1].
      Stage 2 — re-evaluate at full resolution on the local windows
                [c-h, c+h] (h = ceil(n / n_coarse)) around EACH of the
                m_bracket lowest coarse points, and return the global
                argmin over stage1 ∪ all refine windows.

    Refining the top-M (not just the single best) coarse cells brackets
    every competitive basin of the at-most-weakly-multimodal EV surface.
    Empirically (scripts/diag_coarse_to_fine.py) M=3 yields **0.00%**
    disagreement vs the exhaustive full-grid argmin over 6 seeds × 350
    steps × 6 domains — i.e. the session trajectory is bitwise identical
    to full-grid (no cascade), so n_q (and the AUROC estimates) are
    exactly unbiased — while only ~n_coarse + M·(2h+1) losses are
    evaluated.  Measured isolated `choose_item` wall-clock speedup
    (scripts/diag_choose_item_speed.py): ≈1.9× on a 250-item bank,
    ≈4.2× at 600, ≈7.5× on the ~1705-item production bank (the ratio
    grows with bank size; the per-question end-to-end gain is
    Amdahl-diluted by the rest of the SMC pipeline).  In the
    measure-zero event of a deeper-than-M miss, the single-bracket data
    bounds the EV cost at ~1e-5 (the stopping rule is insensitive at
    that scale), so the optimisation degrades gracefully rather than
    silently biasing results.

    `loss_of(idx_array)` must return the loss at those integer indices.
    Deterministic (no RNG) → preserves bitwise serial==parallel.
    Returns the integer argmin index in [0, n).
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


def choose_item(state, bank_signals=None, active_domains=None,
                n_subsample=None, bank_sds=None, return_sd=False):
    """Pick (k, s) globally minimising expected total posterior variance.

    bank_signals: optional list of K arrays (one per domain) of candidate
    signal levels. If None, falls back to SIGNAL_GRID for all domains.
    Per-domain arrays may differ in length (e.g. lrda bank has 230 items).
    bank_sds:  Phase 3.5 — optional list of K arrays parallel to
               bank_signals giving each candidate's s posterior SD.  When
               provided, item selection marginalises each candidate over
               its s posterior (via _expected_loss_vec(signal_sds=...)),
               so the selector prefers high-precision items.  None
               (default) ⇒ BIT-IDENTICAL to the prior engine.
    return_sd: if True, return (k, s, s_sd) — the chosen item's s posterior
               SD (0.0 when bank_sds is None) — so the session can
               marginalise the observation likelihood too.  Default False
               ⇒ returns (k, s), so every existing caller/test is unchanged.
    active_domains: optional list of domain indices to consider (for cert sessions).
    n_subsample: F4.1 OC-campaign enabler.  If set, use the deterministic
                 coarse-to-fine argmin (`_coarse_to_fine_argmin`) with
                 ~n_subsample coarse points + a local full-resolution
                 refine per domain instead of scanning the full bank.
                 Recovers the exact argmin on the smooth EV surface (no
                 n_q bias) at ~5–6× lower cost.  None (default) = full
                 grid (backward-compatible, byte-identical to pre-F4.1).
    """
    K = state["t"].shape[1]
    candidates = bank_signals if bank_signals is not None else [SIGNAL_GRID] * K
    domains = active_domains if active_domains is not None else list(range(K))
    best_loss = np.inf
    best_k, best_s = domains[0], float(np.asarray(candidates[domains[0]])[0])
    best_sd = 0.0
    for k in domains:
        sigs = np.asarray(candidates[k])
        sds = (np.asarray(bank_sds[k]) if bank_sds is not None else None)
        idx = _coarse_to_fine_argmin(
            lambda ii: _expected_loss_vec(
                state, k, sigs[ii],
                signal_sds=(sds[ii] if sds is not None else None)),
            len(sigs), n_subsample)
        loss = float(_expected_loss_vec(
            state, k, sigs[idx:idx + 1],
            signal_sds=(sds[idx:idx + 1] if sds is not None else None))[0])
        if loss < best_loss:
            best_loss = loss
            best_k = k
            best_s = float(sigs[idx])
            best_sd = float(sds[idx]) if sds is not None else 0.0
    return (best_k, best_s, best_sd) if return_sd else (best_k, best_s)


# ───────────── AUROC CI (same as before) ─────────────

def _auroc_ci(state, alpha=0.05):
    # use the hier-style quantile helper (treats state["t"]/["l"] as (N,K) cloud)
    q = auroc_quantiles_from_particles_hier(state, alphas=(alpha / 2, 1 - alpha / 2))
    return q[:, 0], q[:, 1]


# ───────────── session driver ─────────────

def simulate_response(s, t_true, l_true, rng):
    """Sample y ~ Bernoulli(p) where p = λ + (1 − 2λ)·Φ(exp(l)·(s + t)).

    F0.1: spike-paper Eq. 2 symmetric lapse mixture (floor λ, ceiling 1 − λ).
    """
    z = np.exp(l_true) * (s + t_true)
    p = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * float(norm.cdf(z))
    return int(rng.random() < p)


def run_session_mcmc_auroc(method, true_params, K, r_assumed,
                            max_q=400, delta_auroc=0.05, N=2500, seed=0,
                            run_until_max=False, log_trajectory=False, alpha=0.05,
                            n_mh_steps=15, proposal_scale=0.5,
                            ess_threshold_frac=0.5,
                            brute_proposal_scale=None,
                            bank_signals=None,
                            Sigma_l=None, Sigma_t=None,
                            n_subsample=None, bank_sds=None):
    """Session with MCMC-rejuvenation SMC + AUROC-based stopping (Mode-A).

    Multi-AUROC Precision Protocol (Paper 1, 2026-05-15):
      Stop when max_k halfwidth_{0.95}(AUROC_k) < delta_auroc.
      Item selection via Global EV (`choose_item`) — minimises total posterior
      variance over (t_1..t_K, l_1..l_K).  This is the Bayesian-optimal A-design.

    Args:
      method:
        - "hier"  — joint 2K-D cloud with hierarchical prior.  When `Sigma_l`
                    (and optionally `Sigma_t`) are provided, uses the fitted
                    unstructured covariance.  Otherwise falls back to
                    compound-symmetry `Sigma = (1−r)I + rJ` with `r_assumed`.
        - "brute" — methodology-compliant: K independent 2-D SMCs, one per
                    task, each with N particles. Total particle-effort K·N.
                    Forwards to `core_mcmc_brute_k.run_session_mcmc_brute_k`.
                    `Sigma_l`/`r_assumed` are IGNORED for brute (each task gets
                    an independent N(0,1) prior by construction — this is the
                    "no information sharing" comparator).

      FIX-T0.5: the previously available "brute_joint" branch (single 2K-D
      cloud with independent N(0,1) prior) has been removed.  It produced
      biased comparisons when paired with c* calibration.

      true_params, K, r_assumed: as before.
      Sigma_l, Sigma_t: F1.1 (2026-05-15). Unstructured prior covariance
                        matrices for the l-block and t-block respectively.
                        Pass to enable cross-domain "borrowing of strength".
                        Forwarded to `make_state_hier`. None → CS fallback.
      n_mh_steps: number of MH steps per rejuvenation event.
      proposal_scale: multiplier on cloud covariance for MH in 2K-D.
                      RGG-optimal ~ 2.38 / sqrt(2K).
      brute_proposal_scale: separate scale for the per-task 2-D MCMC under
                            method="brute".  Defaults to 1.5 (~ 2.38/sqrt(2)).
      ess_threshold_frac: trigger resample+MCMC when ESS < this * N.
      log_trajectory: if True, returns `lo_traj`, `hi_traj` (per-question lo/hi
                      AUROC CIs, shape (n_q+1, K)).  Used for post-hoc δ-sweep
                      reporting (compute stop points at multiple δ from one run).
    """
    # Dispatch "brute" / "random" to the methodology-compliant independent-
    # posterior implementation in core_mcmc_brute_k.py.  Both use K
    # independent 2-D SMCs (N(0,1) prior); they differ ONLY in item
    # selection: "brute" = Global-EV adaptive, "random" = uniform null
    # baseline.  Sigma_l/r_assumed are unused for both.  The clean Phase-1
    # ablation ladder is random → brute → hier (each adds one capability:
    # adaptive selection, then hierarchical pooling).
    if method in ("brute", "random"):
        # Late import to avoid circular dependency at module load.
        from core_mcmc_brute_k import run_session_mcmc_brute_k
        ps = 1.5 if brute_proposal_scale is None else float(brute_proposal_scale)
        return run_session_mcmc_brute_k(
            true_params, K=K, max_q=max_q, delta_auroc=delta_auroc, N=N,
            seed=seed, run_until_max=run_until_max,
            log_trajectory=log_trajectory, alpha=alpha,
            n_mh_steps=n_mh_steps, proposal_scale=ps,
            ess_threshold_frac=ess_threshold_frac,
            bank_signals=bank_signals,
            select=("random" if method == "random" else "ev"),
            n_subsample=n_subsample,
        )

    rng = np.random.default_rng(seed)
    assert len(true_params) == 2 * K
    if method == "hier":
        # F1.1: pass Sigma_l/Sigma_t through so unstructured fitted prior is used.
        state = make_state_hier(N, K, r_assumed, rng,
                                 Sigma_l=Sigma_l, Sigma_t=Sigma_t)
    else:
        raise ValueError(
            f"Unknown method {method!r}; valid: 'hier', 'brute', 'random'")

    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    lo, hi = _auroc_ci(state, alpha)
    if log_trajectory:
        los, his = [lo.copy()], [hi.copy()]
    n_q = 0
    stop_step = None
    accept_rates = []

    for q in range(max_q):
        if bank_sds is not None:
            # Phase 3.5: item selection AND the observation likelihood
            # marginalise over the segment-signal posterior s ~ N(s,s_sd²).
            k, s, s_sd = choose_item(state, bank_signals,
                                     n_subsample=n_subsample,
                                     bank_sds=bank_sds, return_sd=True)
        else:                       # default path — BIT-IDENTICAL to prior
            k, s = choose_item(state, bank_signals, n_subsample=n_subsample)
            s_sd = 0.0
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update(state, k, s, y, s_sd=s_sd)
        if ess(state["w"]) < ess_threshold_frac * N:
            ar = resample_and_rejuvenate(state, rng, n_mh_steps, proposal_scale)
            accept_rates.append(ar)
        n_q += 1
        lo, hi = _auroc_ci(state, alpha)
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
        "mean_acceptance_rate": float(np.mean(accept_rates)) if accept_rates else float("nan"),
        "n_rejuvenations": len(accept_rates),
        "delta_auroc": float(delta_auroc),
        "method": method,
    }
    if log_trajectory:
        out["lo_traj"] = np.array(los)
        out["hi_traj"] = np.array(his)
    return out


# ───────────── F1.4: post-hoc delta-sweep helper ─────────────

def post_hoc_delta_sweep(lo_traj, hi_traj, deltas):
    """Given a per-question (lo, hi) AUROC CI trajectory from a Mode-A run with
    `log_trajectory=True`, report the stop-step under each candidate δ in
    `deltas`.

    This avoids re-running the same session at every δ.  Run once at the
    tightest δ (e.g. 0.025) with `run_until_max=True`, then call this helper
    to report what would have happened at δ ∈ {0.025, 0.05, 0.10} from the
    SAME trajectory.

    Args:
      lo_traj, hi_traj: arrays of shape (T, K) where T is the number of
                        post-update timepoints (T = max_q + 1, indexed by q).
                        `lo_traj[0]`, `hi_traj[0]` are the prior CIs.
      deltas:           iterable of candidate AUROC HW thresholds.

    Returns:
      dict mapping each delta to its stop-step (1-indexed n_questions, or
      None if the session never reached HW < δ within max_q).

    The stop step is the FIRST q ≥ 1 such that max_k HW_q,k < δ, matching
    the production loop in `run_session_mcmc_auroc`.
    """
    lo_traj = np.asarray(lo_traj)
    hi_traj = np.asarray(hi_traj)
    hw_traj = (hi_traj - lo_traj) / 2.0          # shape (T, K)
    hw_max = hw_traj.max(axis=1)                 # shape (T,)
    out = {}
    # Index 0 is the prior; first post-update step is index 1.
    for d in deltas:
        d_f = float(d)
        idx = np.where(hw_max[1:] < d_f)[0]
        out[d_f] = (int(idx[0]) + 1) if idx.size > 0 else None
    return out



# ───────────── Mode-B binary certification: RELOCATED ─────────────
# F3.1 (2026-05-15): the Mode-B engine (run_session_mcmc_certification,
# choose_item_cert, _pass_probs_hier, _pass_prob_joint, the boundary-prior
# / Berger-IUT / Fisher-guard / n_min-guard logic) has been MOVED to
# `engine_mode_b.py`.  Under the Multi-AUROC reframe Mode-B is deprecated
# for Paper 1 (Mode-A = run_session_mcmc_auroc above).  Mode-B is preserved
# verbatim for Paper 2.  Import it explicitly:
#     from engine_mode_b import run_session_mcmc_certification
# The shared SMC/MCMC primitives it needs (make_state_hier with the
# Mode-A-inert l_prior_mean kwarg, update, ess, resample_and_rejuvenate,
# simulate_response, LAPSE_RATE, Z_BUFFER, SIGNAL_GRID) remain in this
# module.  See CHANGELOG.md.
