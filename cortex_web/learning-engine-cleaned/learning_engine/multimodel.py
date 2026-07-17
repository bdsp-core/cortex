"""Multi-signal (per-signal skill/bias) learning model — the real-data
generalisation of the binary SDT trainer (decision log D30).

Per participant i and signal channel m in {1..M}:

    state        t_im (criterion), u_im = log sigma_im (skill)
    evidence     z_m  = (s_km - t_m) / exp(u_m + r_f * fpos_k)
    response     P(resp = m) = lam/M + (1 - lam) * softmax(z)_m
    criterion    t_m    += alpha_t[m]  * delta_m
    skill        u_m    -= alpha_s[m] * w(|z_m|) * (u_m - u_inf[m])
    chronometric log r_k ~ N(psi0_i - psi1 * margin_k + psi2 * F_k, tau_r^2)

with delta_m the per-channel prediction error (soft p_m - ystar_m by
default; every channel updates every trial — the soft deltas sum to zero
across channels), w the Wilson gate (which naturally allocates skill
learning to the channels for which the trial is appropriately difficult),
margin_k = z_(1) - z_(2) the decision margin (the multiclass |z|), and
F_k = r_f * fpos_k the transient fatigue rise (fpos resets each session).

Hierarchical transfer structure (user decision, D28):
  * per-signal population rates alpha_t[m], alpha_s[m], floors u_inf[m]
    (cohort-level, per the settled D6 stance), partially pooled across
    signals through their own hyperprior;
  * per-participant initial states with a ONE-FACTOR skill structure
        u0_im  = mu_u0[m] + gamma[m] * eta_i + tau_u[m] * eps_im
    (eta_i ~ N(0,1) the general-ability factor, gamma[m] >= 0 loadings —
    the static snapshot's 64% first-eigenvalue share, D29), criteria
    independent per signal;
  * per-participant lapse lam_i and base speed psi0_i, hierarchical.

Deterministic dynamics (q = 0): the state trajectory is a deterministic
function of parameters and stimuli, so the likelihood is an exact product
of categoricals — the D2 trick carries over unchanged.
"""
# Copied from ideal-test-learning/sim/multimodel.py on 2026-07-16 (D45 close-out state).
# Only import paths were rewritten for packaging; the code is otherwise verbatim.

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import ndtr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

from .learnmodel import W_CENTERS, W_H, wilson_simplex

EPS = 1e-9
FORMS = ("soft", "hard", "internal", "none")


# ---------------------------------------------------------------------------
# numpy forward simulation (synthetic truth for the recovery test)
# ---------------------------------------------------------------------------

def _softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def _wilson(z_abs, z0=1.0):
    za = z_abs / z0
    return za * np.exp(0.5 * (1.0 - za ** 2))


def simulate_multi(theta, s, ystar, fpos, rng, form="soft", chrono=None,
                   gate_z0=1.0):
    """One participant. theta: dict with t0 (M,), u0 (M,), u_inf (M,),
    alpha_t (M,), alpha_s (M,), lam, r_f, psi0. s: (K, M) evidence,
    ystar: (K, M) one-hot gold, fpos: (K,) transient ramp position.
    Returns y (K, M) one-hot, rt (K,), t_traj/sig_traj (K, M)."""
    K, M = s.shape
    t = theta["t0"].astype(float).copy()
    u = theta["u0"].astype(float).copy()
    u_inf = theta["u_inf"]
    a_t, a_s = theta["alpha_t"], theta["alpha_s"]
    lam, r_f, psi0 = theta["lam"], theta["r_f"], theta["psi0"]
    ch = chrono or dict(psi1=0.15, psi2=0.5, tau_r=0.35)

    y = np.zeros((K, M)); rt = np.empty(K)
    t_traj = np.empty((K, M)); sig_traj = np.empty((K, M))
    for k in range(K):
        t_traj[k] = t; sig_traj[k] = np.exp(u)
        z = (s[k] - t) / np.exp(u + r_f * fpos[k])
        p = lam / M + (1 - lam) * _softmax(z)
        c = rng.choice(M, p=p)
        y[k, c] = 1.0
        zs = np.sort(z)[::-1]
        margin = zs[0] - zs[1]
        rt[k] = np.exp(psi0 - ch["psi1"] * margin + ch["psi2"] * r_f * fpos[k]
                       + rng.normal(0, ch["tau_r"]))
        if form == "soft":
            delta = p - ystar[k]
        elif form == "hard":
            delta = y[k] - ystar[k]
        elif form == "internal":
            delta = _softmax(z) - ystar[k]
        elif form == "none":
            delta = np.zeros(M)
        else:
            raise ValueError(form)
        w = _wilson(np.abs(z), gate_z0)
        t = t + a_t * delta
        u = u - a_s * w * (u - u_inf)
    return dict(y=y, rt=rt, t_traj=t_traj, sig_traj=sig_traj)


def make_multi_cohort(L=40, K=300, M=4, rng=None, form="soft",
                      pop=None, chrono=None, gate_z0=1.0):
    """Synthetic cohort with the real data's geometry: gold-channel evidence
    ambiguous (mean ~ -0.2), non-gold clearly negative (~ -1.3); sessions of
    ~25-150 trials with a saturating transient fatigue ramp; a one-factor
    skill structure across signals."""
    rng = rng or np.random.default_rng(0)
    pop = pop or {}
    mu_u0 = pop.get("mu_u0", np.linspace(0.15, 0.45, M))
    gamma = pop.get("gamma", np.linspace(0.35, 0.15, M))
    tau_u = pop.get("tau_u", 0.18)
    mu_t0 = pop.get("mu_t0", np.linspace(-0.15, 0.25, M))
    tau_t = pop.get("tau_t", 0.25)
    floor_drop = pop.get("floor_drop", np.linspace(0.45, 0.25, M))
    alpha_t = pop.get("alpha_t", np.linspace(0.10, 0.05, M))
    alpha_s = pop.get("alpha_s", np.linspace(0.012, 0.005, M))
    lam0 = pop.get("lam", 0.03)
    rf_scale = pop.get("rf_scale", 0.15)
    chrono = chrono or dict(psi1=0.15, psi2=0.5, tau_r=0.35)

    S = np.empty((L, K, M)); YS = np.zeros((L, K, M)); Y = np.zeros((L, K, M))
    RT = np.empty((L, K)); FP = np.empty((L, K))
    TT = np.empty((L, K, M)); SS = np.empty((L, K, M))
    thetas = []
    for i in range(L):
        gold = rng.integers(0, M, size=K)
        s = rng.normal(-1.3, 0.5, size=(K, M))
        s[np.arange(K), gold] = rng.normal(-0.1, 0.6, size=K)
        s = np.clip(s, -2.33, 2.33)
        ys = np.zeros((K, M)); ys[np.arange(K), gold] = 1.0
        # sessions: lengths lognormal around ~60 trials
        fpos = np.empty(K); k = 0
        while k < K:
            ln = int(np.clip(rng.lognormal(np.log(60), 0.6), 15, 200))
            idx = np.arange(min(ln, K - k))
            fpos[k:k + len(idx)] = 1.0 - np.exp(-idx / 60.0)
            k += len(idx)
        eta = rng.normal()
        u0 = mu_u0 + gamma * eta + tau_u * rng.normal(size=M)
        th = dict(
            t0=mu_t0 + tau_t * rng.normal(size=M),
            u0=u0,
            u_inf=u0 - floor_drop * np.exp(0.3 * rng.normal(size=M)),
            alpha_t=alpha_t, alpha_s=alpha_s,
            lam=float(np.clip(lam0 * np.exp(0.3 * rng.normal()), 0.005, 0.2)),
            r_f=float(np.abs(rng.normal(0, rf_scale))),
            psi0=float(rng.normal(1.0, 0.15)), eta=eta,
        )
        r = simulate_multi(th, s, ys, fpos, rng, form=form, chrono=chrono,
                           gate_z0=gate_z0)
        S[i], YS[i], Y[i] = s, ys, r["y"]
        RT[i], FP[i] = r["rt"], fpos
        TT[i], SS[i] = r["t_traj"], r["sig_traj"]
        thetas.append(th)
    mask = np.ones((L, K))
    return dict(s=S, ystar=YS, y=Y, rt=RT, fpos=FP, mask=mask,
                t_traj=TT, sig_traj=SS, thetas=thetas, L=L, K=K, M=M,
                pop=dict(mu_u0=mu_u0, gamma=gamma, tau_u=tau_u, mu_t0=mu_t0,
                         tau_t=tau_t, floor_drop=floor_drop, alpha_t=alpha_t,
                         alpha_s=alpha_s, lam=lam0, rf_scale=rf_scale),
                chrono=chrono)


# ---------------------------------------------------------------------------
# JAX exact likelihood (deterministic dynamics)
# ---------------------------------------------------------------------------

def _gate(z, gate_form="wilson", gate_z0=1.0, w_coef=None,
          w_centers=None):
    """Skill-learning gate w(|z|): the Wilson shape (peak at gate_z0), a
    flat/difficulty-blind ablation, or the LEARNED simplex-basis shape
    (D23 convention: the simplex fixes scale, the rate carries magnitude).
    `w_centers` overrides the default basis (D45 item 4: widen when a
    fit rails at the support edge)."""
    if w_coef is not None:
        zc = jnp.asarray(W_CENTERS if w_centers is None else w_centers)
        h = zc[1] - zc[0]
        B = jnp.exp(-0.5 * ((jnp.abs(z)[..., None] - zc) / h) ** 2)
        return B @ w_coef
    if gate_form == "wilson":
        za = jnp.abs(z) / gate_z0
        return za * jnp.exp(0.5 * (1.0 - za ** 2))
    if gate_form == "flat":
        return jnp.ones_like(z)
    raise ValueError(gate_form)


def multi_lls_one(t0, u0, u_inf, alpha_t, alpha_s, lam, r_f,
                  s, ystar, y, mask, fpos, form="soft",
                  rt=None, rt_ok=None, psi0=0.0, psi1=0.0, psi2=0.0,
                  tau_r=1.0, gate_form="wilson", gate_z0=1.0,
                  upd_mask=None, qs=None, qs_ok=None, w_coef=None,
                  kap=None, dk=None, w_centers=None,
                  dmask=None, is_bin=None):
    """Per-trial log-likelihood vector (K,) for one participant.

    `mask` gates which trials CONTRIBUTE likelihood; `upd_mask` (default =
    mask) gates which trials UPDATE the state — a temporal train/eval split
    passes mask=eval, upd_mask=all so the dynamics keep evolving through
    the evaluation segment while only eval trials are scored.
    `kap` (M,) + `dk` (K,): difficulty-indexed skill (D45 item 3) — the
    effective log-noise on trial k is u + kap * dk_k, with dk the item's
    standardized difficulty; kap = 0 recovers scalar skill.
    MIXED LINKS (integration plan P1.3): `dmask` (K, M) marks the state
    dims trial k addresses (one-hot for a binary detection trial; the
    group's dims for an n-way pick; default all-ones = the historical
    single-group behavior). `is_bin` (K,) selects the binary probit
    link, whose response/gold are carried in y/ystar on the trial's
    domain column. Updates are confined to the trial's dims."""
    M = t0.shape[0]
    has_rt = rt is not None
    logr = jnp.log(rt) if has_rt else jnp.zeros(s.shape[0])
    ok = rt_ok if has_rt else jnp.zeros(s.shape[0])
    um = mask if upd_mask is None else upd_mask
    qs_ = jnp.zeros(s.shape[0]) if qs is None else qs
    qok_ = jnp.zeros(s.shape[0]) if qs_ok is None else qs_ok
    dk_ = jnp.zeros(s.shape[0]) if dk is None else dk
    kap_ = jnp.zeros(M) if kap is None else kap
    dm_ = jnp.ones(s.shape) if dmask is None else dmask
    ib_ = jnp.zeros(s.shape[0]) if is_bin is None else is_bin

    def step(carry, x):
        t, u = carry
        sk, ysk, yk, mk, umk, fk, lrk, okk, qk, qokk, dkk, dmk, ibk = x
        z = (sk - t) / jnp.exp(u + r_f * fk + kap_ * dkk)
        # n-way link: softmax restricted to the trial's dims
        zm = jnp.where(dmk > 0, z, -1e30)
        sm = jax.nn.softmax(zm)
        G = jnp.maximum(dmk.sum(), 1.0)
        p = (lam / G) * dmk + (1.0 - lam) * sm
        p = jnp.clip(p, EPS, 1.0)
        ll_nway = jnp.sum(yk * jnp.log(p))
        # binary link: probit on the trial's single domain, the shared
        # (1 - 2 lam) design convention
        z_dom = jnp.sum(z * dmk)
        p1 = jnp.clip(lam + (1.0 - 2.0 * lam) * ndtr(z_dom),
                      EPS, 1.0 - EPS)
        y_dom = jnp.sum(yk * dmk)
        ys_dom = jnp.sum(ysk * dmk)
        ll = jnp.where(ibk > 0,
                       y_dom * jnp.log(p1)
                       + (1.0 - y_dom) * jnp.log(1.0 - p1),
                       ll_nway)
        if has_rt:
            # top-2 margin via max ops: 2x cheaper than sort inside the
            # scan gradient (ties have measure zero; the isfinite guard
            # covers M=1)
            m1 = jnp.max(z)
            m2 = jnp.max(jnp.where(z >= m1, -jnp.inf, z))
            margin = m1 - jnp.where(jnp.isfinite(m2), m2, m1)
            mr = psi0 - psi1 * margin + psi2 * r_f * fk
            ll = ll + okk * (-0.5 * ((lrk - mr) / tau_r) ** 2
                             - jnp.log(tau_r))
        if form == "soft":
            delta = jnp.where(ibk > 0, dmk * (p1 - ys_dom), p - ysk)
        elif form == "hard":
            delta = yk - ysk
        elif form == "internal":
            delta = sm - ysk
        elif form == "none":
            delta = jnp.zeros(M)
        elif form == "qsoft":
            # exploratory (D28: qscore never ground truth): the learner
            # reinforces the CHOSEN channel toward the seen graded score
            delta = qokk * yk * (p - qk)
        else:
            raise ValueError(form)
        delta = delta * dmk                 # updates confined to dims
        w = _gate(z, gate_form, gate_z0, w_coef, w_centers)
        t_n = t + alpha_t * delta
        u_n = u - alpha_s * w * (u - u_inf) * dmk
        return (t + umk * (t_n - t), u + umk * (u_n - u)), mk * ll

    (_, _), lls = lax.scan(step, (t0, u0),
                           (s, ystar, y, mask, um, fpos, logr, ok,
                            qs_, qok_, dk_, dm_, ib_))
    return lls


def multi_loglik_one(t0, u0, u_inf, alpha_t, alpha_s, lam, r_f,
                     s, ystar, y, mask, fpos, form="soft",
                     rt=None, rt_ok=None, psi0=0.0, psi1=0.0, psi2=0.0,
                     tau_r=1.0, gate_form="wilson", gate_z0=1.0,
                     upd_mask=None, qs=None, qs_ok=None, w_coef=None,
                     kap=None, dk=None, w_centers=None,
                     dmask=None, is_bin=None):
    """Exact log-likelihood for one participant. Shapes: t0/u0/u_inf/
    alpha_t/alpha_s (M,); s/ystar/y (K, M); mask/fpos (K,); rt (K,)."""
    return multi_lls_one(t0, u0, u_inf, alpha_t, alpha_s, lam, r_f,
                         s, ystar, y, mask, fpos, form=form, rt=rt,
                         rt_ok=rt_ok, psi0=psi0, psi1=psi1, psi2=psi2,
                         tau_r=tau_r, gate_form=gate_form,
                         gate_z0=gate_z0, upd_mask=upd_mask, qs=qs,
                         qs_ok=qs_ok, w_coef=w_coef, kap=kap,
                         dk=dk, w_centers=w_centers, dmask=dmask,
                         is_bin=is_bin).sum()


def multi_traj_one(t0, u0, u_inf, alpha_t, alpha_s, lam, r_f,
                   s, ystar, y, mask, fpos, form="soft"):
    """Deterministic (t_k, sigma_k) per-channel trajectory."""
    M = t0.shape[0]

    def step(carry, x):
        t, u = carry
        sk, ysk, yk, mk, fk = x
        z = (sk - t) / jnp.exp(u + r_f * fk)
        sm = jax.nn.softmax(z)
        p = jnp.clip(lam / M + (1.0 - lam) * sm, EPS, 1.0)
        if form == "soft":
            delta = p - ysk
        elif form == "hard":
            delta = yk - ysk
        elif form == "internal":
            delta = sm - ysk
        else:
            delta = jnp.zeros(M)
        w = jnp.abs(z) * jnp.exp(0.5 * (1.0 - z ** 2))
        t_n = t + alpha_t * delta
        u_n = u - alpha_s * w * (u - u_inf)
        return (t + mk * (t_n - t), u + mk * (u_n - u)), (t, jnp.exp(u))

    _, out = lax.scan(step, (t0, u0), (s, ystar, y, mask, fpos))
    return out


# ---------------------------------------------------------------------------
# hierarchical NumPyro model with the one-factor transfer structure
# ---------------------------------------------------------------------------

def hier_multi_model(s, ystar, y, mask, fpos, rt=None, rt_ok=None,
                     form="soft", fatigue=True, static=False,
                     qs=None, qs_ok=None, shapes=False, upd_mask=None,
                     dk=None, kappa=False, w_centers=None,
                     dmask=None, is_bin=None):
    L, K, M = s.shape
    dm = jnp.ones_like(s) if dmask is None else dmask       # (L, K, M)
    ib = jnp.zeros(s.shape[:2]) if is_bin is None else is_bin
    um = mask if upd_mask is None else upd_mask   # dynamics gate (D45:
    # fit with state updates on the trials where feedback was SHOWN)

    # --- per-signal population, partially pooled through hyperpriors ---
    if not static:
        mu_at = numpyro.sample("hyper_mu_at", dist.Normal(-2.2, 1.0))
        sd_at = numpyro.sample("hyper_sd_at", dist.HalfNormal(0.7))
        mu_as = numpyro.sample("hyper_mu_as", dist.Normal(-5.0, 1.2))
        sd_as = numpyro.sample("hyper_sd_as", dist.HalfNormal(0.7))
    with numpyro.plate("signal", M):
        if not static:
            la_t = numpyro.sample("la_t", dist.Normal(mu_at, sd_at))
            la_s = numpyro.sample("la_s", dist.Normal(mu_as, sd_as))
            # the floor is identified only through the dynamics; in static
            # mode these latents are inert and their hierarchical funnel
            # only degrades sampling — skip them entirely
            mu_dinf = numpyro.sample("mu_dinf", dist.Normal(-1.2, 0.8))
            tau_dinf = numpyro.sample("tau_dinf", dist.HalfNormal(0.5))
        mu_t0 = numpyro.sample("mu_t0", dist.Normal(0.0, 0.6))
        mu_u0 = numpyro.sample("mu_u0", dist.Normal(0.0, 0.6))
        tau_t0 = numpyro.sample("tau_t0", dist.HalfNormal(0.5))
        tau_u0 = numpyro.sample("tau_u0", dist.HalfNormal(0.5))
        gamma = numpyro.sample("gamma", dist.HalfNormal(0.4))
    if static:      # frozen states: the no-learning baseline (R4 predictive test)
        alpha_t = numpyro.deterministic("alpha_t", jnp.zeros(M))
        alpha_s = numpyro.deterministic("alpha_s", jnp.zeros(M))
    else:
        alpha_t = numpyro.deterministic("alpha_t", jnp.exp(la_t))
        alpha_s = numpyro.deterministic("alpha_s", jnp.exp(la_s))

    # --- learned gate (G2/D23): simplex weights, Dirichlet centred on the
    #     Wilson projection (prior fallback is safe; movement = data) ---
    if shapes:
        zc = W_CENTERS if w_centers is None else np.asarray(w_centers)
        J = len(zc)
        w_coef = numpyro.sample(
            "c_w", dist.Dirichlet(J * jnp.array(wilson_simplex(zc)) + 0.5))
    else:
        w_coef = None

    # --- difficulty-indexed skill (D45 item 3): effective log-noise on
    #     trial k is u + kappa_m * d_k; kappa = 0 recovers scalar skill ---
    if kappa:
        mu_kap = numpyro.sample("mu_kap", dist.Normal(0.0, 0.3))
        tau_kap = numpyro.sample("tau_kap", dist.HalfNormal(0.3))
        with numpyro.plate("signal_kap", M):
            z_kap = numpyro.sample("z_kap", dist.Normal(0, 1))
        kap = numpyro.deterministic("kappa", mu_kap + tau_kap * z_kap)
    else:
        kap = None
    dk_ = jnp.zeros(s.shape[:2]) if dk is None else jnp.asarray(dk)

    # --- lapse (participant-level) ---
    mu_lam = numpyro.sample("mu_lam", dist.Normal(-3.5, 0.8))
    tau_lam = numpyro.sample("tau_lam", dist.HalfNormal(0.5))

    # --- participants: general-ability factor + per-signal deviations ---
    with numpyro.plate("learner", L):
        eta = numpyro.sample("eta", dist.Normal(0, 1))
        z_lam = numpyro.sample("z_lam", dist.Normal(0, 1))
    with numpyro.plate("ls_m", M), numpyro.plate("ls_l", L):
        z_t0 = numpyro.sample("z_t0", dist.Normal(0, 1))
        z_u0 = numpyro.sample("z_u0", dist.Normal(0, 1))
        if not static:
            z_dinf = numpyro.sample("z_dinf", dist.Normal(0, 1))

    t0 = numpyro.deterministic("t0", mu_t0 + tau_t0 * z_t0)          # (L, M)
    u0 = numpyro.deterministic(
        "u0", mu_u0 + gamma * eta[:, None] + tau_u0 * z_u0)          # (L, M)
    if static:
        u_inf = numpyro.deterministic("u_inf", u0)   # inert when alpha = 0
    else:
        # floor = u0 - softplus(drop): always strictly below the start
        dinf = mu_dinf + tau_dinf * z_dinf
        u_inf = numpyro.deterministic("u_inf", u0 - jax.nn.softplus(dinf))
    lam = numpyro.deterministic(
        "lam", jnp.clip(jnp.exp(mu_lam + tau_lam * z_lam), 1e-4, 0.4))

    # --- fatigue (transient; per-participant, non-negative) ---
    if fatigue:
        tau_f = numpyro.sample("tau_f", dist.HalfNormal(0.5))
        with numpyro.plate("learner_f", L):
            z_f = numpyro.sample("z_f", dist.HalfNormal(1.0))
        r_f = numpyro.deterministic("r_f", tau_f * z_f)
    else:
        r_f = jnp.zeros(L)

    # --- chronometric channel ---
    has_rt = rt is not None
    if has_rt:
        mu_psi0 = numpyro.sample("mu_psi0", dist.Normal(1.0, 1.0))
        tau_psi0 = numpyro.sample("tau_psi0", dist.HalfNormal(0.5))
        with numpyro.plate("learner_r", L):
            z_psi0 = numpyro.sample("z_psi0", dist.Normal(0, 1))
        psi0 = numpyro.deterministic("psi0", mu_psi0 + tau_psi0 * z_psi0)
        psi1 = numpyro.sample("psi1", dist.HalfNormal(0.5))
        psi2 = numpyro.sample("psi2", dist.HalfNormal(2.0))
        tau_r = numpyro.sample("tau_r", dist.HalfNormal(0.5))
        rt_ = jnp.asarray(rt)
        ok_ = jnp.asarray(rt_ok)

        def one(t0_, u0_, ui_, lam_, rf_, p0_, s_k, ys_k, y_k, m_k, um_k,
                f_k, r_k, ok_k, q_k, qok_k, d_k, dm_k, ib_k):
            return multi_loglik_one(t0_, u0_, ui_, alpha_t, alpha_s, lam_,
                                    rf_, s_k, ys_k, y_k, m_k, f_k, form=form,
                                    rt=r_k, rt_ok=ok_k, psi0=p0_, psi1=psi1,
                                    psi2=psi2, tau_r=tau_r, upd_mask=um_k,
                                    qs=q_k, qs_ok=qok_k, w_coef=w_coef,
                                    kap=kap, dk=d_k, w_centers=w_centers,
                                    dmask=dm_k, is_bin=ib_k)
        ll = jax.vmap(one)(t0, u0, u_inf, lam, r_f, psi0,
                           s, ystar, y, mask, um, fpos, rt_, ok_,
                           _zeros_like(qs, s), _zeros_like(qs_ok, s), dk_,
                           dm, ib)
    else:
        def one(t0_, u0_, ui_, lam_, rf_, s_k, ys_k, y_k, m_k, um_k, f_k,
                q_k, qok_k, d_k, dm_k, ib_k):
            return multi_loglik_one(t0_, u0_, ui_, alpha_t, alpha_s, lam_,
                                    rf_, s_k, ys_k, y_k, m_k, f_k, form=form,
                                    upd_mask=um_k, qs=q_k, qs_ok=qok_k,
                                    w_coef=w_coef, kap=kap, dk=d_k,
                                    w_centers=w_centers, dmask=dm_k,
                                    is_bin=ib_k)
        ll = jax.vmap(one)(t0, u0, u_inf, lam, r_f, s, ystar, y, mask, um,
                           fpos, _zeros_like(qs, s), _zeros_like(qs_ok, s),
                           dk_, dm, ib)
    numpyro.factor("obs", ll.sum())


def _zeros_like(x, s):
    return jnp.zeros(s.shape[:2]) if x is None else jnp.asarray(x)


POP_SITES = ("hyper_mu_at", "hyper_sd_at", "hyper_mu_as", "hyper_sd_as",
             "la_t", "la_s", "mu_t0", "mu_u0", "mu_dinf", "tau_t0",
             "tau_u0", "tau_dinf", "gamma", "mu_lam", "tau_lam", "tau_f",
             "mu_psi0", "tau_psi0", "psi1", "psi2", "tau_r",
             "mu_kap", "tau_kap")


def run_hier_multi(coh, warmup=400, samples=400, chains=2, seed=0,
                   form="soft", use_rt=True, fatigue=True, progress=True,
                   max_tree_depth=9, target_accept=0.85, static=False,
                   mask_key="mask", chain_method=None, dense_pop=False,
                   shapes=False, upd_key=None, kappa=False, dk_key=None,
                   w_centers=None, dmask_key=None, bin_key=None):
    s = jnp.array(coh["s"], dtype=jnp.float64)
    ystar = jnp.array(coh["ystar"], dtype=jnp.float64)
    y = jnp.array(coh["y"], dtype=jnp.float64)
    mask = jnp.array(coh[mask_key], dtype=jnp.float64)
    upd = (None if upd_key is None
           else jnp.array(coh[upd_key], dtype=jnp.float64))
    dk = (None if dk_key is None
          else jnp.array(coh[dk_key], dtype=jnp.float64))
    dm = (None if dmask_key is None
          else jnp.array(coh[dmask_key], dtype=jnp.float64))
    ib = (None if bin_key is None
          else jnp.array(coh[bin_key], dtype=jnp.float64))
    fpos = jnp.array(coh["fpos"], dtype=jnp.float64)
    if use_rt:
        rt_np = np.asarray(coh["rt"], dtype=float)
        ok = (np.isfinite(rt_np) & (rt_np > 0)
              & (np.asarray(coh[mask_key]) > 0))
        rt = jnp.array(np.where(ok, rt_np, 1.0), dtype=jnp.float64)
        rt_ok = jnp.array(ok.astype(float), dtype=jnp.float64)
    else:
        rt = rt_ok = None
    if form == "qsoft":
        q_np = np.asarray(coh["qscore"], dtype=float)
        qok = np.isfinite(q_np) & (np.asarray(coh[mask_key]) > 0)
        qs = jnp.array(np.where(qok, q_np, 0.0), dtype=jnp.float64)
        qs_ok = jnp.array(qok.astype(float), dtype=jnp.float64)
    else:
        qs = qs_ok = None

    if chain_method is None:
        # independent chains beat the vectorized batch when tree depths
        # differ across chains; needs >= `chains` XLA host devices
        # (XLA_FLAGS=--xla_force_host_platform_device_count=N)
        chain_method = ("parallel" if jax.device_count() >= chains
                        else "vectorized")
    if dense_pop:
        # one dense mass block over the population-level sites: the
        # correlated (rates x floors x state-prior) directions saturate
        # tree depth under a diagonal metric at real-data scale
        drop = set()
        if static:
            drop |= {"hyper_mu_at", "hyper_sd_at", "hyper_mu_as",
                     "hyper_sd_as", "la_t", "la_s", "mu_dinf", "tau_dinf"}
        if not fatigue:
            drop.add("tau_f")
        if rt is None:
            drop |= {"mu_psi0", "tau_psi0", "psi1", "psi2", "tau_r"}
        if not kappa:
            drop |= {"mu_kap", "tau_kap"}
        dense_mass = [tuple(s for s in POP_SITES if s not in drop)]
    else:
        dense_mass = False
    kernel = NUTS(hier_multi_model, init_strategy=init_to_median(num_samples=20),
                  max_tree_depth=max_tree_depth, target_accept_prob=target_accept,
                  dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method=chain_method,
                progress_bar=progress)
    wc = None if w_centers is None else jnp.asarray(w_centers,
                                                    dtype=jnp.float64)
    mcmc.run(jax.random.PRNGKey(seed), s, ystar, y, mask, fpos, rt, rt_ok,
             form, fatigue, static, qs, qs_ok, shapes, upd, dk, kappa, wc,
             dm, ib, extra_fields=("diverging", "num_steps"))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return mcmc, post


def pooled_loglik(coh, post, form="soft"):
    """Per-participant pooled log-likelihood (categorical channel only) at
    the fitted model's posterior-MEAN parameters — the plug-in profile
    statistic of the rule adjudication (D1: equal parameter counts across
    forms, so margins compare directly). Returns (total, per-participant)."""
    s = jnp.array(coh["s"], dtype=jnp.float64)
    ystar = jnp.array(coh["ystar"], dtype=jnp.float64)
    y = jnp.array(coh["y"], dtype=jnp.float64)
    mask = jnp.array(coh["mask"], dtype=jnp.float64)
    fpos = jnp.array(coh["fpos"], dtype=jnp.float64)
    pm = {k: jnp.array(post[k].mean(0)) for k in
          ("t0", "u0", "u_inf", "lam", "alpha_t", "alpha_s")}
    rf = (jnp.array(post["r_f"].mean(0)) if "r_f" in post
          else jnp.zeros(coh["L"]))

    def one(t0_, u0_, ui_, lam_, rf_, s_k, ys_k, y_k, m_k, f_k):
        return multi_loglik_one(t0_, u0_, ui_, pm["alpha_t"], pm["alpha_s"],
                                lam_, rf_, s_k, ys_k, y_k, m_k, f_k,
                                form=form)
    lls = np.asarray(jax.vmap(one)(pm["t0"], pm["u0"], pm["u_inf"], pm["lam"],
                                   rf, s, ystar, y, mask, fpos))
    return float(lls.sum()), lls
