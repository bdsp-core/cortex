"""Training filter — bootstrap SMC with propagation (G1 port of trainer_rd
training_filter.TaskFilter).

The belief-update engine for training. Per-task 2-D cloud over (θ_k, ℓ_k) in
ENGINE coords. Unlike the eval engine, the trainer filter does NOT use MH
rejuvenation: once the learner's state evolves, the history-replay likelihood
targets the wrong (trajectory) posterior (F1). Instead each trial is:

  reweight  (Bayes on the lapse-mixture likelihood, with s_sd attenuation)
  propagate (sample each particle forward through the transition kernel T —
             the SAME dynamics learner_sim assumes, so the filter is correctly
             specified against the learner)
  resample  (systematic when ESS < frac·N; process noise supplies diversity)

`propagate` (no MH) is the only genuinely new numeric vs the eval engine. This is
a VERBATIM port of the scratch `TaskFilter`; bit-parity is gated by
`tests/test_trainer_g1_filter.py` against golden vectors from the scratch module.

Deferred to the next increment (they need the `Learner` truth-kernel, ported to
trainer/sim/): `steady_state_sd`, `recommended_sd_floor`, `filter_from_seed_task`.
The mastery-gate `sd_floor` is re-pinned against production numerics then.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from trainer.conventions import (LAPSE_RATE, SKILL_MODE_MULTIPLIER,
                                 engine_to_plan, auroc_from_ell)
from trainer.dynamics import LearnerParams

_DEFAULT_Z = 2.0
_DEFAULT_ALPHA = 0.05

# Gauss–Hermite nodes for E[w(s_real)] under s_real ~ N(s, s_sd²)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(11)
_GH_WN = _GH_W / np.sqrt(np.pi)
# 21-node rule for the exact conditional kernel.
_GHE_X, _GHE_W = np.polynomial.hermite.hermgauss(21)
_GHE_WN = _GHE_W / np.sqrt(np.pi)


class TaskFilter:
    """Bootstrap particle filter for ONE task's (θ, ℓ) state under training.

    particles: theta (N,), ell (N,), w (N,). Dynamics params come from a
    LearnerParams (the filter's assumed model).
    """

    def __init__(self, theta, ell, params: LearnerParams, *,
                 ess_frac=0.5, seed=0, w=None, p_static=0.0, smear_w=False,
                 exact_kernel=True, jump_eps=0.0, jump_kappa=8.0):
        self.theta = np.asarray(theta, dtype=np.float64).copy()
        self.ell = np.asarray(ell, dtype=np.float64).copy()
        self.N = self.theta.shape[0]
        self.w = (np.full(self.N, 1.0 / self.N) if w is None
                  else np.asarray(w, dtype=np.float64).copy())
        self.w /= self.w.sum()
        self.p = params
        self.ess_frac = float(ess_frac)
        self.rng = np.random.default_rng(seed)
        # static-hypothesis mixture (opt-in). p_static=0 (default) consumes no
        # RNG and is bit-identical to shipped.
        self.p_static = float(p_static)
        self.learn = (np.ones(self.N)
                      if self.p_static <= 0.0
                      else (self.rng.random(self.N)
                            >= self.p_static).astype(np.float64))
        # smear_w=True propagates skill with E[w(s_real)] (11-node Gauss–Hermite).
        self.smear_w = bool(smear_w)
        # exact conditional transition kernel (default True). Bit-identical to
        # the simple branch at s_sd=0 / rule="static" / y=None.
        self.exact_kernel = bool(exact_kernel)
        # contaminated-innovation ("jump") transition noise (opt-in). jump_eps=0
        # (default) consumes no RNG and is bit-identical.
        self.jump_eps = float(jump_eps)
        self.jump_kappa = float(jump_kappa)

    def _jump_scale(self):
        """Per-particle innovation-SD multiplier. 1.0 when off."""
        if self.jump_eps <= 0.0:
            return 1.0
        ind = self.rng.random(self.N) < self.jump_eps
        return np.where(ind, self.jump_kappa, 1.0)

    # ── observation likelihood (engine coords, s_sd attenuation) ──
    def _p_yes(self, s, s_sd=0.0):
        el = np.exp(self.ell)
        z = el * (float(s) + self.theta)
        if s_sd:
            z = z / np.sqrt(1.0 + (el * s_sd) ** 2)
        return LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z)

    def reweight(self, s, y, s_sd=0.0):
        """Bayes step. Returns the one-step predictive likelihood
        p(y | history) = Σ_i w_i·like_i."""
        p = self._p_yes(s, s_sd)
        like = p if y == 1 else (1.0 - p)
        w = self.w * like
        tot = w.sum()
        self.w = (w / tot) if tot > 0 else np.full(self.N, 1.0 / self.N)
        return float(tot)

    # ── transition kernel T per particle (mirrors learner_sim.step) ──
    def propagate(self, s, y_star, feedback=True, *, y=None, s_sd=0.0):
        """Sample the belief forward through T.

        y: the OBSERVED response. Under the hard rule the learner's criterion
        update uses its own realized y — which the filter observed — so the
        exact conditional kernel is δ = y − y* for every particle. Falls back to
        per-particle Bernoulli sampling only when y is unknown.

        Both channels are f-gated: no feedback ⇒ no y* ⇒ no update."""
        if self.p.rule == "static":
            return
        s = float(s)
        jscale = self._jump_scale()          # scalar 1.0 when off
        sigma = np.exp(-np.clip(self.ell, -40.0, 40.0))  # plan coord σ
        t = -self.theta                      # plan coord t
        f = 1.0 if feedback else 0.0
        if self.exact_kernel and s_sd and y is not None:
            # ── exact conditional kernel ──
            nodes = s + np.sqrt(2.0) * float(s_sd) * _GHE_X         # (21,)
            zz = (nodes[:, None] - t[None, :]) / sigma[None, :]     # (21,N)
            p_node = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(zz)
            like = p_node if y == 1 else (1.0 - p_node)
            qn = _GHE_WN[:, None] * like
            Z = qn.sum(axis=0)
            qn = qn / np.where(Z > 0, Z, 1.0)
            d_n = np.abs(nodes[:, None] - t[None, :]) / sigma[None, :]
            w_node = np.exp(-((d_n - SKILL_MODE_MULTIPLIER) ** 2)
                            / (2.0 * self.p.rho ** 2))
            w_weight = (qn * w_node).sum(axis=0)                    # E[w|y]
            Vw = np.maximum((qn * w_node ** 2).sum(axis=0)
                            - w_weight ** 2, 0.0)
            if self.p.rule == "soft":
                Ep = (qn * p_node).sum(axis=0)                      # E[p|y]
                Vp = np.maximum((qn * p_node ** 2).sum(axis=0)
                                - Ep ** 2, 0.0)
                delta = Ep - y_star
            else:                            # hard: δ = y − y* exact
                delta = float(y) - y_star
                Vp = 0.0
            sd_t = np.sqrt((self.p.q_t * jscale) ** 2
                           + f * self.learn * (self.p.alpha_t ** 2) * Vp)
            t_new = (t + f * self.learn * self.p.alpha_t * delta
                     + sd_t * self.rng.standard_normal(self.N))
            log_sig = np.log(sigma)
            gap = log_sig - np.log(self.p.sigma_inf)
            g_sigma = -f * self.learn * w_weight * gap
            sd_s = np.sqrt((self.p.q_sigma * jscale) ** 2 + f * self.learn
                           * (self.p.alpha_sigma * gap) ** 2 * Vw)
            log_sig_new = (log_sig + self.p.alpha_sigma * g_sigma
                           + sd_s * self.rng.standard_normal(self.N))
            self.theta = -t_new
            self.ell = -log_sig_new
            return
        if self.p.rule == "soft":
            delta = self._p_yes(s, s_sd) - y_star
        elif y is not None:                  # hard, conditioned on observed y
            delta = float(y) - y_star
        else:                                # hard, y unknown: marginal sample
            p = self._p_yes(s, s_sd)
            yhat = (self.rng.random(self.N) < p).astype(np.float64)
            delta = yhat - y_star
        if self.jump_eps > 0.0:                       # jump branch
            noise_t = self.p.q_t * jscale * self.rng.standard_normal(self.N)
        else:                                         # bit-identical path
            noise_t = self.rng.normal(0, self.p.q_t, self.N)
        t_new = (t + f * self.learn * self.p.alpha_t * delta + noise_t)
        if self.smear_w and s_sd:
            nodes = s + np.sqrt(2.0) * float(s_sd) * _GH_X      # (11,)
            d = np.abs(nodes[:, None] - t[None, :]) / sigma[None, :]
            w_weight = _GH_WN @ np.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2)
                                       / (2.0 * self.p.rho ** 2))
        else:
            d = np.abs(s - t) / sigma
            w_weight = np.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2)
                              / (2.0 * self.p.rho ** 2))
        log_sig = np.log(sigma)
        log_inf = np.log(self.p.sigma_inf)
        g_sigma = -f * self.learn * w_weight * (log_sig - log_inf)
        if self.jump_eps > 0.0:                       # jump branch
            noise_s = (self.p.q_sigma * jscale
                       * self.rng.standard_normal(self.N))
        else:                                         # bit-identical path
            noise_s = self.rng.normal(0, self.p.q_sigma, self.N)
        log_sig_new = log_sig + self.p.alpha_sigma * g_sigma + noise_s
        # back to engine coords
        self.theta = -t_new
        self.ell = -log_sig_new

    def maybe_resample(self):
        if self.ess() < self.ess_frac * self.N:
            # systematic resampling: one uniform offset, stratified positions
            u = (self.rng.random() + np.arange(self.N)) / self.N
            idx = np.searchsorted(np.cumsum(self.w), u)
            idx = np.minimum(idx, self.N - 1)        # guard fp roundoff at cw≈1
            self.theta = self.theta[idx]
            self.ell = self.ell[idx]
            if self.p_static > 0.0:
                self.learn = self.learn[idx]
            self.w = np.full(self.N, 1.0 / self.N)
            # distinct-lineage count of THIS resample (MCSE optimism term)
            self._n_anc = int(np.unique(idx).size)
            return True
        return False

    # ── one training trial ──
    def step(self, s, y, y_star, *, s_sd=0.0, feedback=True):
        """reweight on the observed y → resample if degenerate → propagate the
        belief forward through T. Order matches the eval engine (reweight then
        move), with propagation replacing MH (F1)."""
        self.reweight(s, y, s_sd)
        self.maybe_resample()
        self.propagate(s, y_star, feedback, y=y, s_sd=s_sd)

    def propagate_gap(self, dt_seconds, *, widen=0.0):
        """LT2 hook: between-session decay. Phase-2 default = identity (+ optional
        isotropic variance widening). The fitted Ebbinghaus kernel replaces this
        body in Phase 3 — call site stays put."""
        if widen > 0.0:
            self.ell = self.ell + self.rng.normal(0, widen, self.N)
            self.theta = self.theta + self.rng.normal(0, widen, self.N)

    def boundary_jump(self, eps, sd, theta_scale=1.0):
        """Session-boundary regime-shift hazard (opt-in). At each boundary every
        particle independently receives, with hazard eps, a N(0, sd²) shock on ℓ
        and a N(0, (theta_scale·sd)²) shock on θ (one shared indicator).
        eps<=0 (default) is bit-identical (never called)."""
        if eps <= 0.0 or sd <= 0.0 or self.p.rule == "static":
            return
        ind = self.rng.random(self.N) < float(eps)
        n = int(ind.sum())
        if n == 0:
            return
        self.ell[ind] = self.ell[ind] + float(sd) * self.rng.standard_normal(n)
        if theta_scale > 0.0:
            self.theta[ind] = (self.theta[ind] + float(theta_scale)
                               * float(sd) * self.rng.standard_normal(n))

    # ── summaries ──
    def ess(self):
        return 1.0 / float((self.w * self.w).sum())

    def mean(self):
        """Weighted posterior mean (theta, ell)."""
        return (float((self.w * self.theta).sum()),
                float((self.w * self.ell).sum()))

    def sd(self):
        mt, ml = self.mean()
        return (float(np.sqrt((self.w * (self.theta - mt) ** 2).sum())),
                float(np.sqrt((self.w * (self.ell - ml) ** 2).sum())))

    def plan_mean(self):
        """Posterior mean in plan coords (σ via exp(−ℓ) at the mean ℓ, t=−θ)."""
        mt, ml = self.mean()
        sigma, t = engine_to_plan(mt, ml)
        return float(sigma), float(t)

    def pass_mass(self, ell_star):
        """π = P(ℓ > ℓ*) and its MC standard error (AD6, F14). MCSE denominator
        = min(ESS, distinct lineages at the last resample) — the honest
        leading-order bound; conservative (the gate fires no earlier)."""
        pi = float((self.w * (self.ell > ell_star)).sum())
        n_eff = min(self.ess(), float(getattr(self, "_n_anc", self.N)))
        mcse = float(np.sqrt(max(pi * (1 - pi), 0.0) / max(n_eff, 1.0)))
        return pi, mcse

    def is_mastered(self, ell_star, *, sd_floor, alpha=_DEFAULT_ALPHA,
                    Z=_DEFAULT_Z):
        """AD6-style graduation on the filtered posterior (F14), floor-aware (F5):
        require the skill SD to have contracted to/below `sd_floor` AND the lower
        confidence bound on the pass-mass to clear 1−α."""
        pi, mcse = self.pass_mass(ell_star)
        _, sd_ell = self.sd()
        return bool(sd_ell <= sd_floor and (pi - Z * mcse) >= 1.0 - alpha)


# ── gate-calibration helpers (need the research Learner truth-kernel; imported
#    lazily so the runtime filter never pulls trainer.sim) ────────────────────
def recommended_sd_floor(params: LearnerParams, *, s_sd=0.0, mult=1.5,
                         floor_min=0.15, **kw):
    """Mastery-gate sd_floor from the MEASURED information floor (F5/MA-6):
    mult × the steady-state filtered ℓ-SD under the deployment physics. Recompute
    whenever the physics or the kernel changes instead of trusting a constant."""
    _, sd_l = steady_state_sd(params, s_sd=s_sd, **kw)
    return float(max(mult * sd_l, floor_min))


def steady_state_sd(params: LearnerParams, *, n_particles=2000, n_trials=400,
                    seed=0, s_sd=0.0, filter_kwargs=None):
    """Information-balanced filtered-posterior SD floor (F5).

    The floor is the SD the filter SETTLES to under the actual training regime,
    where reweight-driven contraction balances propagate-driven inflation. We
    measure it by tracking a learner already AT the skill floor (σ = σ_∞,
    criterion unbiased) under representative informative trials (skill-mode
    placement, veridical feedback, simulated responses), averaging the posterior
    SD over the back half. The mastery gate's `sd_floor` must sit at or above this
    value, else no learner can ever satisfy it. `s_sd`: stimulus uncertainty of
    the training items; s_sd=0 reproduces the synthetic regime."""
    from trainer.sim import Learner  # lazy: research kernel, not runtime
    rng = np.random.default_rng(seed)
    sig_inf = float(np.broadcast_to(params.sigma_inf, (1,))[0])
    lnr = Learner([sig_inf], [0.0],
                  LearnerParams(**{**params.__dict__}), seed=seed)
    th0 = 0.0 + 0.25 * rng.standard_normal(n_particles)
    el0 = -np.log(sig_inf) + 0.25 * rng.standard_normal(n_particles)
    f = TaskFilter(th0, el0, params, ess_frac=0.5, seed=seed + 1,
                   **(filter_kwargs or {}))
    sds, sign = [], 1
    for k in range(n_trials):
        mt, ml = f.mean()
        sig_hat, t_hat = engine_to_plan(mt, ml)
        s = t_hat + sign * SKILL_MODE_MULTIPLIER * sig_hat
        sign *= -1
        y_star = int(s > 0.0)
        s_real = s + (s_sd * rng.standard_normal() if s_sd else 0.0)
        y = lnr.step(s_real, 0, y_star=y_star, feedback=True)  # learner ≈ floor
        f.step(s, y, y_star, s_sd=s_sd, feedback=True)
        if k > n_trials // 2:
            sds.append(f.sd())
    sds = np.array(sds)
    return float(sds[:, 0].mean()), float(sds[:, 1].mean())
