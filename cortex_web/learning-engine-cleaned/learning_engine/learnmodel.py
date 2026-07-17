"""Learning-with-feedback SDT model: simulation, likelihood, and estimation.

Implements the state-space model of docs/learning_with_feedback_design_note.tex:

    observation   p_k = lam + (1-2*lam) * Phi((s_k - t_k)/sigma_k)
    criterion     t_{k+1}       = t_k + alpha_t * delta_k + xi_t
    skill         log s_{k+1}   = log s_k - alpha_s * w_k * (log s_k - log s_inf) + xi_s

with the soft prediction error delta_k = p_k - ystar_k (default), the hard
form delta_k = y_k - ystar_k, or the lapse-free internal form
delta_k = Phi(z_k) - ystar_k, and the 85%-rule weight (w_form)

    "wilson" (default)  w_k = |z_k| * exp((1 - z_k^2)/2)      [derived, no rho]
    "gauss"  (ablation) w_k = exp(-(|z_k| - ZSTAR)^2 / (2 rho^2)).

Stimuli live on a signed difficulty axis; the ground-truth label is
ystar = 1[s > 0], so |s| is difficulty and sign(s) is the answer.

Key structural fact exploited throughout: when the process noises q_t, q_s are
zero the state trajectory is a *deterministic* function of the parameters and
the observed history, so the likelihood is an exact product of Bernoullis. Only
q > 0 forces a particle filter.
"""
# Copied from ideal-test-learning/sim/learnmodel.py on 2026-07-16 (D45 close-out state).
# Only import paths were rewritten for packaging; the code is otherwise verbatim.

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri, expit, logit
from scipy.optimize import minimize

# Legacy Gaussian-weight centre. Phi^{-1}(0.85) = 1.0364; this is the lapse-free
# value. See zstar_exact() for the lapse-corrected version. Kept only for the
# "gauss" ablation form; the default "wilson" form has no free constant (D9).
ZSTAR = 1.0364

# The one blessed lapse value: the *design* constant used for placement and for
# derived quantities. The observation-model lapse is still estimated per learner.
LAPSE_DESIGN = 0.025

PARAM_NAMES = ["t0", "sigma0", "sigma_inf", "alpha_t", "alpha_s", "rho", "lam"]
LAM_MAX = 0.25
EPS = 1e-9


def zstar_exact(lam: float) -> float:
    """Signal strength (in units of sigma) giving exactly 85% correct at lapse lam."""
    return float(ndtri((0.85 - lam) / (1.0 - 2.0 * lam)))


def learn_weight(z_abs, rho, w_form: str = "wilson", zstar: float = ZSTAR):
    """Difficulty-appropriateness weight w(|z|) gating the skill update.

    "wilson"  w(z) = z * exp((1 - z^2)/2)  =  z*phi(z)/phi(1).
              The exact shape from the Wilson et al. (2019) derivation of the
              85% rule (per-trial learning ∝ Δ·phi(βΔ)), normalised to peak
              value 1 at its argmax z = 1. Parameter-free: rho is ignored.
    "gauss"   the memo's original soft Gaussian bump around zstar with width
              rho. Kept as the ablation baseline; rho is unidentifiable in
              adaptive designs (the alpha_sigma–rho ridge, D3).
    """
    z_abs = np.asarray(z_abs, dtype=float)
    if w_form == "wilson":
        return z_abs * np.exp(0.5 * (1.0 - z_abs ** 2))
    if w_form == "gauss":
        return np.exp(-((z_abs - zstar) ** 2) / (2.0 * np.asarray(rho) ** 2))
    if w_form == "flat":                 # G2 truth generator: difficulty-blind
        return np.ones_like(z_abs)
    raise ValueError(w_form)


def zstar_place(w_form: str = "wilson", zstar: float = ZSTAR) -> float:
    """Difficulty (in sigma units) where the learning weight peaks — the
    skill-mode placement target. Computed, never hard-coded: 1.0 for the
    Wilson form (its analytic argmax), zstar for the legacy Gaussian form."""
    return 1.0 if w_form == "wilson" else zstar


# --- population-learned gate basis (G2/G4): shared by inference (jaxmodel)
#     and deployment (policy). Simplex weights c fix the shape's scale.
#     The basis is CONFIGURABLE (D45 item 4): every helper takes optional
#     `centers`; the module defaults are the historical basis. Widen when
#     a fit rails (gate_railed) — a peak at the support edge means
#     "outside the representable range", not a measured optimum (D42).
def gate_centers(z_max=3.2, J=8):
    """Bump centers + width for a gate basis reaching z_max."""
    c = np.linspace(0.0, z_max, J)
    return c, float(c[1] - c[0])


W_CENTERS, W_H = gate_centers()
G_POWERS = np.array([0.5, 1.0, 2.0, 4.0])


def _basis(centers):
    if centers is None:
        return W_CENTERS, W_H
    c = np.asarray(centers, float)
    return c, float(c[1] - c[0])


def wilson_simplex(centers=None):
    """Projection of the Wilson shape onto the bump weights (the prior mean)."""
    zc, _ = _basis(centers)
    w = zc * np.exp(0.5 * (1.0 - zc ** 2))
    w = np.maximum(w, 1e-4)
    return w / w.sum()


def learn_weight_coef(z_abs, c, centers=None):
    """The LEARNED gate W(|z|) = sum_j c_j B_j(|z|) (numpy; policy side)."""
    zc, h = _basis(centers)
    B = np.exp(-0.5 * ((np.asarray(z_abs, float)[..., None] - zc) / h) ** 2)
    return B @ np.asarray(c, float)


def zstar_from_coef(c, zgrid=None, centers=None):
    """Placement target under the learned gate: argmax of W-hat."""
    zc, _ = _basis(centers)
    zg = np.linspace(0.05, zc[-1], 160) if zgrid is None else zgrid
    return float(zg[np.argmax(learn_weight_coef(zg, c, centers=centers))])


def gate_railed(c, centers=None):
    """True if the fitted gate's argmax falls beyond the second-to-last
    bump center — the outermost bump's territory, where the basis cannot
    resolve a peak from a censored shoulder (D42: argmax 3.15 with the
    basis ending at 3.2 was 'easier than representable', not an optimum).
    A railed fit means: widen the basis and refit."""
    zc, _ = _basis(centers)
    return bool(zstar_from_coef(c, centers=centers) >= zc[-2])


# ----------------------------------------------------------------------------
# parameter transforms: unconstrained psi <-> natural params
# ----------------------------------------------------------------------------

def to_natural(psi: np.ndarray) -> np.ndarray:
    """psi (..., 7) unconstrained -> natural (..., 7)."""
    psi = np.asarray(psi, dtype=float)
    out = np.empty_like(psi)
    out[..., 0] = psi[..., 0]                     # t0
    out[..., 1] = np.exp(psi[..., 1])             # sigma0
    out[..., 2] = np.exp(psi[..., 2])             # sigma_inf
    out[..., 3] = np.exp(psi[..., 3])             # alpha_t
    out[..., 4] = np.exp(psi[..., 4])             # alpha_s
    out[..., 5] = np.exp(psi[..., 5])             # rho
    out[..., 6] = LAM_MAX * expit(psi[..., 6])    # lam
    return out


def to_unconstrained(nat: np.ndarray) -> np.ndarray:
    nat = np.asarray(nat, dtype=float)
    out = np.empty_like(nat)
    out[..., 0] = nat[..., 0]
    out[..., 1] = np.log(nat[..., 1])
    out[..., 2] = np.log(nat[..., 2])
    out[..., 3] = np.log(nat[..., 3])
    out[..., 4] = np.log(nat[..., 4])
    out[..., 5] = np.log(nat[..., 5])
    out[..., 6] = logit(np.clip(nat[..., 6] / LAM_MAX, 1e-6, 1 - 1e-6))
    return out


# ----------------------------------------------------------------------------
# forward simulation
# ----------------------------------------------------------------------------

def _delta(form, p, z, y_k, ystar_k):
    """Prediction error: soft (model probability incl. lapse), hard (own binary
    response), or internal (lapse-free perceptual probability Phi(z))."""
    if form == "soft":
        return p - ystar_k
    if form == "hard":
        return y_k - ystar_k
    if form == "internal":
        return ndtr(z) - ystar_k
    raise ValueError(form)


def simulate(
    theta: dict,
    s: np.ndarray,
    rng: np.random.Generator,
    form: str = "soft",
    q_t: float = 0.0,
    q_s: float = 0.0,
    feedback: np.ndarray | None = None,
    zstar: float = ZSTAR,
    w_form: str = "gauss",
    g_pow: float = 1.0,
):
    """Simulate one learner over the stimulus sequence `s`.

    Returns dict with y, ystar, t_traj, sigma_traj (length K states, pre-trial).
    """
    K = len(s)
    ystar = (s > 0).astype(float)
    fb = np.ones(K) if feedback is None else np.asarray(feedback, dtype=float)

    t = float(theta["t0"])
    logsig = np.log(float(theta["sigma0"]))
    log_inf = np.log(float(theta["sigma_inf"]))
    a_t, a_s, rho, lam = (float(theta[k]) for k in ("alpha_t", "alpha_s", "rho", "lam"))
    psi0 = float(theta.get("psi0", 1.0))

    y = np.empty(K)
    rt = np.empty(K)
    t_traj = np.empty(K)
    sig_traj = np.empty(K)

    for k in range(K):
        t_traj[k] = t
        sig = np.exp(logsig)
        sig_traj[k] = sig

        z = (s[k] - t) / sig
        p = lam + (1.0 - 2.0 * lam) * ndtr(z)
        y[k] = rng.random() < p
        rt[k] = sample_rt(psi0, abs(z), 0.0, rng)   # F=0: no fatigue kind here

        delta = _delta(form, p, z, y[k], ystar[k])
        if g_pow != 1.0:                 # G2 truth generator: nonlinear link
            delta = np.sign(delta) * np.abs(delta) ** g_pow
        w = learn_weight(abs(z), rho, w_form, zstar)

        t = t + a_t * delta + (rng.normal(0.0, q_t) if q_t > 0 else 0.0)
        logsig = (
            logsig
            - a_s * fb[k] * w * (logsig - log_inf)
            + (rng.normal(0.0, q_s) if q_s > 0 else 0.0)
        )

    return dict(y=y, ystar=ystar, s=s, rt=rt, t_traj=t_traj, sigma_traj=sig_traj)


def state_trajectory(nat: np.ndarray, s: np.ndarray, ystar: np.ndarray,
                     y: np.ndarray | None = None, form: str = "soft",
                     fb: np.ndarray | None = None, zstar: float = ZSTAR,
                     w_form: str = "gauss"):
    """Deterministic state trajectory implied by natural params `nat` (7,)."""
    t0, sig0, sig_inf, a_t, a_s, rho, lam = nat
    K = len(s)
    fb = np.ones(K) if fb is None else fb
    t, logsig, log_inf = t0, np.log(sig0), np.log(sig_inf)
    t_traj, sig_traj = np.empty(K), np.empty(K)
    for k in range(K):
        t_traj[k] = t
        sig = np.exp(logsig)
        sig_traj[k] = sig
        z = (s[k] - t) / sig
        p = lam + (1 - 2 * lam) * ndtr(z)
        delta = _delta(form, p, z, None if y is None else y[k], ystar[k])
        w = learn_weight(abs(z), rho, w_form, zstar)
        t = t + a_t * delta
        logsig = logsig - a_s * fb[k] * w * (logsig - log_inf)
    return t_traj, sig_traj


# ----------------------------------------------------------------------------
# exact (deterministic-dynamics) likelihood, batched over parameter vectors
# ----------------------------------------------------------------------------

def nll_batch(psi: np.ndarray, s: np.ndarray, ystar: np.ndarray, y: np.ndarray,
              form: str = "soft", fb: np.ndarray | None = None,
              zstar: float = ZSTAR, w_form: str = "gauss") -> np.ndarray:
    """Negative log-likelihood for a batch of unconstrained params psi (B, 7).

    Vectorised over the batch dimension; loops over trials. This is what makes
    finite-difference gradients cheap.
    """
    psi = np.atleast_2d(psi)
    nat = to_natural(psi)
    t = nat[:, 0].copy()
    logsig = np.log(nat[:, 1])
    log_inf = np.log(nat[:, 2])
    a_t, a_s, rho, lam = nat[:, 3], nat[:, 4], nat[:, 5], nat[:, 6]

    K = len(s)
    fb = np.ones(K) if fb is None else fb
    nll = np.zeros(psi.shape[0])

    for k in range(K):
        sig = np.exp(logsig)
        z = (s[k] - t) / sig
        p = lam + (1.0 - 2.0 * lam) * ndtr(z)
        p = np.clip(p, EPS, 1.0 - EPS)
        nll -= y[k] * np.log(p) + (1.0 - y[k]) * np.log1p(-p)

        delta = _delta(form, p, z, y[k], ystar[k])
        w = learn_weight(np.abs(z), rho, w_form, zstar)
        t = t + a_t * delta
        logsig = logsig - a_s * fb[k] * w * (logsig - log_inf)

    bad = ~np.isfinite(nll)
    nll[bad] = 1e12
    return nll


def _obj_and_grad(psi, s, ystar, y, form, fb, zstar, w_form, prior_mean, prior_prec, h=1e-5):
    """Objective (nll + Gaussian prior on psi) and central-difference gradient."""
    d = len(psi)
    batch = np.repeat(psi[None, :], 2 * d + 1, axis=0)
    for j in range(d):
        batch[1 + 2 * j, j] += h
        batch[2 + 2 * j, j] -= h
    vals = nll_batch(batch, s, ystar, y, form=form, fb=fb, zstar=zstar, w_form=w_form)

    f = vals[0]
    g = np.empty(d)
    for j in range(d):
        g[j] = (vals[1 + 2 * j] - vals[2 + 2 * j]) / (2 * h)

    if prior_prec is not None:
        r = psi - prior_mean
        f = f + 0.5 * np.sum(prior_prec * r * r)
        g = g + prior_prec * r
    return f, g


DEFAULT_INIT = np.array([0.0, 1.0, 0.7, 0.10, 0.006, 0.8, LAPSE_DESIGN])


def fit_learner(s, ystar, y, form="soft", fb=None, zstar=ZSTAR,
                init=None, prior_mean=None, prior_prec=None,
                fixed: dict | None = None, n_restarts: int = 2, seed: int = 0,
                w_form: str = "gauss"):
    """MLE / MAP for one learner under deterministic dynamics.

    `fixed` maps param name -> value, held constant (implemented by projecting
    the gradient; used for the "lapse fixed at wrong value" experiment).
    Under w_form="wilson" the weight has no width parameter, so rho is inert
    and automatically pinned to keep the Hessian non-singular.
    Returns dict with nat, psi, nll, cov (unconstrained), success.
    """
    rng = np.random.default_rng(seed)
    if init is None:
        init = DEFAULT_INIT
    psi0 = to_unconstrained(np.asarray(init, float))

    if w_form == "wilson":
        fixed = dict(fixed or {})
        fixed.setdefault("rho", 1.0)

    fixed_idx = []
    if fixed:
        nat_fix = np.asarray(DEFAULT_INIT, float).copy()
        for name, v in fixed.items():
            j = PARAM_NAMES.index(name)
            nat_fix[j] = v
            fixed_idx.append(j)
        psi_fix = to_unconstrained(nat_fix)
        psi0 = psi0.copy()
        for j in fixed_idx:
            psi0[j] = psi_fix[j]

    bounds = [(-3, 3), (np.log(0.05), np.log(20)), (np.log(0.02), np.log(20)),
              (np.log(1e-4), np.log(3)), (np.log(1e-6), np.log(0.5)),
              (np.log(0.05), np.log(10)), (-8, 6)]
    for j in fixed_idx:
        bounds[j] = (psi0[j], psi0[j])

    best = None
    for r in range(n_restarts):
        start = psi0.copy() if r == 0 else psi0 + rng.normal(0, 0.35, size=7)
        for j in fixed_idx:
            start[j] = psi0[j]
        start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(
                _obj_and_grad, start, jac=True, method="L-BFGS-B", bounds=bounds,
                args=(s, ystar, y, form, fb, zstar, w_form, prior_mean, prior_prec),
                options=dict(maxiter=400, ftol=1e-10, gtol=1e-7),
            )
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        return dict(success=False)

    psi_hat = best.x
    cov = _hessian_cov(psi_hat, s, ystar, y, form, fb, zstar, w_form, fixed_idx)
    return dict(nat=to_natural(psi_hat), psi=psi_hat, nll=float(best.fun),
                cov=cov, success=True)


def _hessian_cov(psi, s, ystar, y, form, fb, zstar, w_form, fixed_idx, h=1e-4):
    """Finite-difference Hessian of the nll -> asymptotic covariance."""
    d = len(psi)
    free = [j for j in range(d) if j not in fixed_idx]
    H = np.zeros((len(free), len(free)))
    pts, idx = [], {}

    def key(a, b, sa, sb):
        return (a, b, sa, sb)

    for ii, i in enumerate(free):
        for jj, j in enumerate(free):
            if jj < ii:
                continue
            for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = psi.copy()
                p[i] += sa * h
                p[j] += sb * h
                idx[key(i, j, sa, sb)] = len(pts)
                pts.append(p)
    vals = nll_batch(np.array(pts), s, ystar, y, form=form, fb=fb, zstar=zstar,
                     w_form=w_form)
    for ii, i in enumerate(free):
        for jj, j in enumerate(free):
            if jj < ii:
                continue
            v = (vals[idx[key(i, j, 1, 1)]] - vals[idx[key(i, j, 1, -1)]]
                 - vals[idx[key(i, j, -1, 1)]] + vals[idx[key(i, j, -1, -1)]]) / (4 * h * h)
            H[ii, jj] = H[jj, ii] = v
    try:
        cov_free = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_free = np.full((len(free), len(free)), np.nan)
    cov = np.full((d, d), np.nan)
    for ii, i in enumerate(free):
        for jj, j in enumerate(free):
            cov[i, j] = cov_free[ii, jj]
    return cov


# ----------------------------------------------------------------------------
# stimulus designs
# ----------------------------------------------------------------------------

STATIC_DESIGNS = ("wide", "narrow", "easy", "twopoint", "medium")
ONLINE_DESIGNS = ("oracle85", "threshold")


def design_stimuli(kind: str, K: int, rng: np.random.Generator,
                   sigma_hint: float = 1.0, t_hint: float = 0.0) -> np.ndarray:
    """Signed stimulus sequences under *static* (pre-randomised) designs."""
    if kind == "wide":                       # the real study: random difficulty
        levels = np.linspace(0.15, 3.0, 10)
        mag = rng.choice(levels, size=K)
    elif kind == "medium":                   # narrower random band
        mag = rng.uniform(0.6, 1.6, size=K)
    elif kind == "narrow":                   # near threshold, but non-adaptive
        mag = np.abs(rng.normal(0.0, 0.25 * sigma_hint, size=K)) + 0.05
    elif kind == "easy":                     # all easy items
        mag = rng.uniform(2.0, 3.0, size=K)
    elif kind == "twopoint":                 # only two difficulty levels
        mag = rng.choice([0.5, 2.5], size=K)
    else:
        raise ValueError(kind)
    sign = rng.choice([-1.0, 1.0], size=K)
    return sign * mag


def simulate_online(theta: dict, K: int, rng: np.random.Generator, kind: str,
                    form: str = "soft", q_t: float = 0.0, q_s: float = 0.0,
                    zstar: float = ZSTAR, jitter: float = 0.02,
                    w_form: str = "gauss"):
    """Simulate under a *state-dependent* design that sees the true (t_k, sigma_k).

    `oracle85`   places every trial at the learning weight's peak, i.e.
                 |s_k - t_k| / sigma_k = zstar_place(w_form), so w_k == 1 on
                 every trial and (under "gauss") rho drops out of the likelihood.
    `threshold`  places every trial at the learner's current criterion
                 (measurement-optimal, ~50% correct).
    """
    t = float(theta["t0"])
    logsig = np.log(float(theta["sigma0"]))
    log_inf = np.log(float(theta["sigma_inf"]))
    a_t, a_s, rho, lam = (float(theta[k]) for k in ("alpha_t", "alpha_s", "rho", "lam"))

    s = np.empty(K); y = np.empty(K); ystar = np.empty(K)
    t_traj = np.empty(K); sig_traj = np.empty(K)

    for k in range(K):
        sig = np.exp(logsig)
        t_traj[k] = t; sig_traj[k] = sig
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if kind == "oracle85":
            offset = zstar_place(w_form, zstar) * sig
        elif kind == "threshold":
            offset = 0.05 * sig
        else:
            raise ValueError(kind)
        sk = t + sign * offset + rng.normal(0, jitter * sig)
        s[k] = sk
        ystar[k] = float(sk > 0)

        z = (sk - t) / sig
        p = lam + (1 - 2 * lam) * ndtr(z)
        y[k] = rng.random() < p
        delta = _delta(form, p, z, y[k], ystar[k])
        w = learn_weight(abs(z), rho, w_form, zstar)
        t = t + a_t * delta + (rng.normal(0, q_t) if q_t > 0 else 0.0)
        logsig = logsig - a_s * w * (logsig - log_inf) + (rng.normal(0, q_s) if q_s > 0 else 0.0)

    return dict(y=y, ystar=ystar, s=s, t_traj=t_traj, sigma_traj=sig_traj)


# ----------------------------------------------------------------------------
# population sampling
# ----------------------------------------------------------------------------

POP = dict(
    t0=(0.30, 0.35),          # mean, sd  (normal)
    sigma0=(np.log(1.20), 0.22),   # lognormal
    sigma_inf=(np.log(0.55), 0.20),
    alpha_t=(np.log(0.12), 0.35),
    alpha_s=(np.log(0.008), 0.35),  # tau_sigma ~ 125 trials
    rho=(np.log(0.70), 0.20),
    lam=(np.log(LAPSE_DESIGN), 0.30),   # design constant 0.025 (D8); still estimated
    psi0=(1.0, 0.15),          # per-learner base log-RT (G1); normal
)

# Chronometric truth for simulation (G1/D21): log r = psi0_i - psi1*|z| +
# psi2*F_k + noise, with F_k the learner's ACCUMULATED fatigue rise in
# log-sigma units. Easier decisions are faster; fatigue slows responding.
# These are simulation ground truths (like POP), estimated in every fit.
CHRONO = dict(psi1=0.15, psi2=0.5, tau_r=0.35)


def sample_rt(psi0, z_abs, F, rng, chrono=CHRONO):
    """One response time from the chronometric law."""
    m = psi0 - chrono["psi1"] * z_abs + chrono["psi2"] * F
    return float(np.exp(m + rng.normal(0.0, chrono["tau_r"])))


def sample_population(n: int, rng: np.random.Generator, pop: dict | None = None):
    pop = POP if pop is None else pop
    out = []
    for _ in range(n):
        th = dict(
            t0=rng.normal(*pop["t0"]),
            sigma0=np.exp(rng.normal(*pop["sigma0"])),
            sigma_inf=np.exp(rng.normal(*pop["sigma_inf"])),
            alpha_t=np.exp(rng.normal(*pop["alpha_t"])),
            alpha_s=np.exp(rng.normal(*pop["alpha_s"])),
            rho=np.exp(rng.normal(*pop["rho"])),
            lam=np.exp(rng.normal(*pop["lam"])),
            psi0=rng.normal(*pop.get("psi0", (1.0, 0.15))),
        )
        # keep the floor genuinely below the start so there is a curve to learn
        th["sigma_inf"] = min(th["sigma_inf"], 0.92 * th["sigma0"])
        out.append(th)
    return out


def nat_to_dict(nat) -> dict:
    return {k: float(v) for k, v in zip(PARAM_NAMES, nat)}


def dict_to_nat(d) -> np.ndarray:
    return np.array([d[k] for k in PARAM_NAMES], float)
