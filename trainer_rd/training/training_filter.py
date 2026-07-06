"""Step 3 — training filter: bootstrap SMC with propagation.

The belief-update engine for training. Per-task 2-D cloud over (θ_k, ℓ_k) in
ENGINE coords (PROJECT_MEMORY.md D2). Unlike the eval engine, the trainer
filter does NOT use MH rejuvenation: once the learner's state evolves the
history-replay likelihood targets the wrong (trajectory) posterior (F1/D5).
Instead each trial is:

  reweight  (Bayes on the lapse-mixture likelihood, with s_sd attenuation)
  propagate (sample each particle forward through the transition kernel T —
             the SAME dynamics as learner_sim, so the filter is correctly
             specified against the simulator)
  resample  (multinomial when ESS < frac·N; process noise supplies diversity)

The propagate step mirrors learner_sim.Learner.step EXACTLY (soft/hard rule,
σ relaxation toward σ_∞, 85%-weight, feedback gate), evaluated per particle.

Mastery (F14): reuse AD6 semantics on the filtered posterior — graduate task k
when P(ℓ_k > ℓ*_k) − Z·mcse ≥ 1−α AND the posterior SD has contracted past a
floor-aware gate (F5).

LT2 hook (D9): `propagate_gap(dt_seconds)` — between-session decay kernel,
identity (+ optional variance widening) in Phase 2; the fitted Ebbinghaus
kernel (memory §3A) drops in here without touching the per-trial loop.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from training.bridge_conventions import (LAPSE_RATE, SKILL_MODE_MULTIPLIER,
                                engine_to_plan, auroc_from_ell)
from training.learner_sim import Learner, LearnerParams

_DEFAULT_Z = 2.0
_DEFAULT_ALPHA = 0.05

# Gauss–Hermite nodes for E[w(s_real)] under s_real ~ N(s, s_sd²) (M12 F30)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(11)
_GH_WN = _GH_W / np.sqrt(np.pi)
# 21-node rule for the exact conditional kernel (M15 MA-1). The conditional
# moments E[p^m | y] have a likelihood factor that concentrates node mass in
# one tail, a harder integrand than F30's smooth unconditional bump: measured
# worst-case error vs an 81-node reference over the realistic regime
# (s_sd/σ ≤ 2) is 2.2e-2 at 11 nodes but ≤ 2.0e-3 at 21 — i.e. per-trial
# kernel-mean error ≤ α_t·2e-3 ≈ 4e-4, two orders below q_t. smear_w keeps
# the 11-node rule (its F30 behavior is bit-pinned by the M12 SBC).
_GHE_X, _GHE_W = np.polynomial.hermite.hermgauss(21)
_GHE_WN = _GHE_W / np.sqrt(np.pi)


class TaskFilter:
    """Bootstrap particle filter for ONE task's (θ, ℓ) state under training.

    particles: theta (N,), ell (N,), w (N,). Dynamics params come from a
    LearnerParams (the filter's assumed model — fixed in Phase 2, D7/D10).
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
        # M12 robustification (opt-in): static-hypothesis mixture. With prior
        # prob p_static a particle carries "learning inactive" — the
        # DETERMINISTIC transition terms (criterion delta-update, σ relaxation)
        # are gated off for it; process noise still applies. Keeps posterior
        # support on non-improving trajectories so the belief cannot converge
        # to the model's σ_∞ attractor on prior dynamics alone (finding F28).
        # p_static=0 (default) consumes no RNG and is bit-identical to shipped.
        self.p_static = float(p_static)
        self.learn = (np.ones(self.N)
                      if self.p_static <= 0.0
                      else (self.rng.random(self.N)
                            >= self.p_static).astype(np.float64))
        # M12 finding F30 (opt-in): the learner experiences w(s_real), not
        # w(s_mean); without smearing the filter assumes faster skill progress
        # than the learner actually makes under bank-level s_sd, over-
        # estimating ℓ exactly in the graduation window (SBC n≈50 dip).
        # smear_w=True propagates with E[w(s_real)] (11-node Gauss–Hermite).
        self.smear_w = bool(smear_w)
        # M15 MA-1 (opt-in): EXACT conditional transition kernel. The learner's
        # response y and its state update share the SAME s_real draw, so the
        # correct kernel conditions on the observed y: moments are taken under
        # the node posterior p(s_real|θ,y) ∝ N(s;s_mean,s_sd²)·P(y|s_real,θ) —
        #   mean  E[p|y] (soft δ), E[w|y] (skill channel, both rules),
        #   var   q² + (gain)²·Var[·|y] injected per particle.
        # Audit measurement (studies/study_exact_kernel_audit.py): cov_t@90
        # 0.631→0.908, cov_ℓ 0.771→0.909, RMSE also improves — the shipped
        # kernel is overconfident (t-intervals ~40% too narrow) under bank
        # s_sd. Supersedes smear_w (which is its unconditional-mean ℓ-half).
        # Bit-identical to shipped at s_sd=0 / rule="static" / y=None.
        # M17 (D26, user-ratified): DEFAULT True in scratch — the closed-loop
        # SBC (F57) completed the evidence; exact_kernel=False remains for
        # ablations/historical comparisons. Floor context at s_sd=0.85:
        # 1.5×ℓ-floor = 0.246 (default params) / 0.131 (benchmark params),
        # so ModeThresholds.sd_floor=0.23 stays in-band, erring toward FG
        # safety (M17 priority: FG ≥ lateness > speed > retention).
        self.exact_kernel = bool(exact_kernel)
        # M23 (F80, opt-in): contaminated-innovation ("jump") transition
        # noise. The 2026-07-03 tester sessions show discrete regime shifts
        # (USER-E d′ 0.8→3.0 across a 64 s sitting break; USER-C −0.3→2.0
        # across one day) that a Gaussian random walk at q_σ=0.02 makes
        # astronomically improbable, so the filter grinds a full 40-trial
        # session of likelihood to traverse what the learner did instantly.
        # Per-trial innovations become the scale mixture
        #     η ~ (1−ε)·N(0, q²) + ε·N(0, (κq)²)
        # with ONE Bernoulli(ε) regime indicator per particle per trial
        # shared across both channels (a strategy shift moves the whole
        # observer). SMC marginalizes the indicator exactly by sampling it —
        # no static parameter lives on a particle (F4-safe). Stationary-floor
        # price is E[η²]/q² = 1 + ε(κ²−1); keep ε(κ²−1) ≲ 0.6 and re-measure
        # `steady_state_sd` when enabling. jump_eps=0 (default) consumes no
        # RNG and is bit-identical to M22. Constants for the sandbox are
        # pinned by studies/study_m23_regime.py.
        self.jump_eps = float(jump_eps)
        self.jump_kappa = float(jump_kappa)

    def _jump_scale(self):
        """Per-particle innovation-SD multiplier (M23/F80). 1.0 when off."""
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
        p(y | history) = Σ_i w_i·like_i — the prequential evidence increment
        the M15 σ_∞-mixture (mixture_filter.py) accumulates per stratum."""
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
        exact conditional kernel is δ = y − y* for every particle (F21).
        Falls back to per-particle Bernoulli sampling only when y is unknown.

        s_sd: when the presented stimulus is uncertain (s_real ~ N(s, s_sd²)),
        the learner's soft-rule δ uses p(s_real); its conditional mean given
        the particle is the attenuated p (F20). With exact_kernel=False the
        extra Var[p(s_real)] and the s_real-dependence of the skill weight
        are dropped ("folded into the process noise") — the M15 audit (MA-1)
        measured that omission as first-order for t-calibration; the
        exact_kernel=True path conditions every moment on the observed y.
        Both channels are f-gated (MA-3): no feedback ⇒ no y* ⇒ no update."""
        if self.p.rule == "static":
            return
        s = float(s)
        jscale = self._jump_scale()          # M23/F80; scalar 1.0 when off
        # M18 hygiene: clip ℓ before exponentiating — Nelder–Mead excursions
        # during dynamics fitting (F62) can push particles to |ℓ| where
        # σ = exp(−ℓ) under/overflows; the clip only touches states whose
        # response probabilities are saturated at the lapse floor anyway.
        sigma = np.exp(-np.clip(self.ell, -40.0, 40.0))  # plan coord σ
        t = -self.theta                      # plan coord t
        # MA-3 (F22 fix): the criterion δ needs y*, which only feedback
        # reveals — both channels are f-gated (f=1.0 ⇒ bit-identical).
        f = 1.0 if feedback else 0.0
        if self.exact_kernel and s_sd and y is not None:
            # ── exact conditional kernel (MA-1) ──
            # node posterior p(s_real | θ, y) on the 21-point GH grid
            nodes = s + np.sqrt(2.0) * float(s_sd) * _GHE_X         # (21,)
            zz = (nodes[:, None] - t[None, :]) / sigma[None, :]     # (11,N)
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
            else:                            # hard: δ = y − y* exact (F21)
                delta = float(y) - y_star
                Vp = 0.0
            # conditional variance rides the same gates as the mean: a
            # non-learning / no-feedback particle has no α·δ term, hence
            # no Var[δ|y] inflation either
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
        if self.jump_eps > 0.0:                       # M23/F80 jump branch
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
        if self.jump_eps > 0.0:                       # M23/F80 jump branch
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
            # systematic resampling: one uniform offset, stratified positions —
            # strictly lower resampling variance than multinomial at equal cost
            u = (self.rng.random() + np.arange(self.N)) / self.N
            idx = np.searchsorted(np.cumsum(self.w), u)
            idx = np.minimum(idx, self.N - 1)        # guard fp roundoff at cw≈1
            self.theta = self.theta[idx]
            self.ell = self.ell[idx]
            if self.p_static > 0.0:
                self.learn = self.learn[idx]
            self.w = np.full(self.N, 1.0 / self.N)
            # MA-6: distinct-lineage count of THIS resample — the leading
            # term of the post-resample MCSE optimism (weights reset to
            # uniform while the support collapsed to n_anc atoms). Not
            # compounded across resamples: the process noise re-mixes the
            # cloud over ~1/α trials, so older ancestry is forgotten.
            self._n_anc = int(np.unique(idx).size)
            return True
        return False

    # ── one training trial ──
    def step(self, s, y, y_star, *, s_sd=0.0, feedback=True):
        """reweight on the observed y → resample if degenerate → propagate
        the belief forward through T. Order matches the eval engine (reweight
        then move), with propagation replacing MH (F1)."""
        self.reweight(s, y, s_sd)
        self.maybe_resample()
        self.propagate(s, y_star, feedback, y=y, s_sd=s_sd)

    def propagate_gap(self, dt_seconds, *, widen=0.0):
        """LT2 hook (D9): between-session decay. Phase-2 default = identity
        (+ optional isotropic variance widening to model forgetting drift).
        The fitted Ebbinghaus retrievability/stability kernel (memory §3A)
        replaces this body in Phase 3 — call site stays put."""
        if widen > 0.0:
            self.ell = self.ell + self.rng.normal(0, widen, self.N)
            self.theta = self.theta + self.rng.normal(0, widen, self.N)

    def boundary_jump(self, eps, sd, theta_scale=1.0):
        """M23 (F80): session-boundary regime-shift hazard.

        The 2026-07-03 tester data localize the discrete performance jumps
        at SITTING BOUNDARIES (USER-E's d′ 0.8→3.0 across a 64 s restart;
        USER-C's −0.3→2.0 overnight), where the per-trial kernel never
        moves and the GapAnchor only acts on gaps ≥ 4 h. Model: at each
        boundary every particle independently receives, with hazard eps, a
        N(0, sd²) shock on ℓ and a N(0, (theta_scale·sd)²) shock on θ (one
        shared indicator — a strategy shift moves the whole observer; the
        observed jumps are σ-dominant, E: Δlnσ ≈ 1.3 vs Δt ≈ 0.1, so the
        criterion channel takes a scaled shock; direction is the
        likelihood's job). Unlike `propagate_gap(widen=)` this is a
        MIXTURE: the (1−eps) bulk of the belief is untouched, so a stable
        learner's posterior barely inflates (added variance eps·sd² on ℓ)
        while the tail mass lets one session of likelihood traverse a
        regime shift instead of three. eps<=0 (default: never called) is
        bit-identical to M22. Constants pinned by
        studies/study_m23_regime.py."""
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
        """π = P(ℓ > ℓ*) and its MC standard error (AD6, F14).

        MCSE denominator (M15 MA-6): min(ESS, distinct lineages at the last
        resample). ESS alone reads N right after every resample although the
        support holds only n_anc atoms; the min is the honest leading-order
        bound. Conservative direction — the gate fires no earlier."""
        pi = float((self.w * (self.ell > ell_star)).sum())
        n_eff = min(self.ess(), float(getattr(self, "_n_anc", self.N)))
        mcse = float(np.sqrt(max(pi * (1 - pi), 0.0) / max(n_eff, 1.0)))
        return pi, mcse

    def is_mastered(self, ell_star, *, sd_floor, alpha=_DEFAULT_ALPHA,
                    Z=_DEFAULT_Z):
        """AD6-style graduation on the filtered posterior (F14), floor-aware
        (F5): require the skill SD to have contracted to/below `sd_floor` AND
        the lower confidence bound on the pass-mass to clear 1−α."""
        pi, mcse = self.pass_mass(ell_star)
        _, sd_ell = self.sd()
        return bool(sd_ell <= sd_floor and (pi - Z * mcse) >= 1.0 - alpha)


def filter_from_seed_task(seed, task, params: LearnerParams, *,
                          inflated=True, ess_frac=0.5, rng_seed=0) -> TaskFilter:
    """Build a per-task TaskFilter from a TrainingSeed (Step 1) marginal (D2)."""
    theta, ell, w = seed.task_cloud(task, inflated=inflated)
    return TaskFilter(theta, ell, params, ess_frac=ess_frac, seed=rng_seed, w=w)


def recommended_sd_floor(params: LearnerParams, *, s_sd=0.0, mult=1.5,
                         floor_min=0.15, **kw):
    """Mastery-gate sd_floor from the MEASURED information floor (F5/MA-6):
    mult × the steady-state filtered ℓ-SD under the deployment physics.
    ModeThresholds.sd_floor = 0.23 hard-codes 1.5× the synthetic (s_sd=0)
    floor 0.154; the real-bank floor is ≈0.179 (⇒ ≈0.27 here) and the exact
    kernel (MA-1) raises it further — recompute whenever the physics or the
    kernel changes instead of trusting the constant."""
    _, sd_l = steady_state_sd(params, s_sd=s_sd, **kw)
    return float(max(mult * sd_l, floor_min))


def steady_state_sd(params: LearnerParams, *, n_particles=2000, n_trials=400,
                    seed=0, s_sd=0.0, filter_kwargs=None):
    """Information-balanced filtered-posterior SD floor (F5).

    The floor is NOT pure diffusion (process noise alone random-walks the
    variance up without bound — there is no steady state without information).
    It is the SD the filter SETTLES to under the actual training regime, where
    reweight-driven contraction balances propagate-driven inflation. We measure
    it by tracking a learner already AT the skill floor (σ = σ_∞, criterion
    unbiased) under representative informative trials (skill-mode placement,
    veridical feedback, simulated responses), and averaging the posterior SD
    over the back half of the run. The mastery gate's `sd_floor` must sit at or
    above this value, else no learner can ever satisfy it.

    s_sd: stimulus uncertainty of the training items (F20). The real bank has
    median s_sd ≈ 0.6–1.0, which raises the floor (per-trial information is
    attenuated); s_sd=0 reproduces the synthetic M3 regime."""
    rng = np.random.default_rng(seed)
    sig_inf = float(np.broadcast_to(params.sigma_inf, (1,))[0])
    lnr = Learner([sig_inf], [0.0],
                  LearnerParams(**{**params.__dict__}), seed=seed)
    # filter starts at a realistic post-eval spread around the true floor state
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
