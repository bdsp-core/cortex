"""M15 MA-2 — per-learner σ_∞ mixture filter (the principled F28/F29 fix).

WHY. The shipped TaskFilter conditions on the dynamics parameters — in
particular the skill ceiling σ_∞ — at their population point values (D7/D10).
Graduatability is decided by sign(ℓ_∞ − ℓ*), so a filter that KNOWS ℓ_∞ > ℓ*
a priori pulls every particle toward an above-cut attractor and eventually
certifies the prior's prophecy regardless of data (F28: static sub-cut
learner declared 30/30; F29: P(declare)=1.00 at every ceiling-to-cut margin).
The F31 `p_static` mixture is the 2-point special case of the fix below.

WHAT. Bayesian model averaging over a small discrete grid of ceiling
hypotheses: J parallel TaskFilter strata, stratum j conditioned on
ℓ_∞ = grid_j, with prior weights from a truncated Gaussian and posterior
weights updated by each stratum's PREQUENTIAL evidence — the one-step
predictive likelihood p_j(y_k | history) that `TaskFilter.reweight` already
computes as its normalizer:

    log ω_j ← log ω_j + log p_j(y_k | H_k),   ω ∝ exp(log ω)

This is exact Bayes over the grid (adaptive item selection cancels — the
policy is a function of the observed history only, the same ignorability
argument that licenses the filter itself). A DISCRETE mixture avoids the
static-parameter path degeneracy that rightly killed per-particle σ_∞
estimation in F4: stratum weights live outside the particle system, so no
resampling step can collapse the ceiling hypothesis space.

PARAMETER JUSTIFICATION (each estimation choice, per the M15 audit):
  grid center  ell_inf_mean — the D10 empirical-Bayes ceiling
      exp(−expert_ℓ_mean_k) (§2B): the only data-grounded ceiling estimate.
  grid spread  tau = 0.30 (default) — bracketed from two sides:
      UPPER: the ceiling spread cannot exceed the cross-rater CURRENT-skill
      spread (skill = ceiling + transient deficit ⇒ Var(skill) ≥
      Var(ceiling)); EXTSET fit gives population σ_ℓ = 0.434 [0.414, 0.456]
      (F43/F51). LOWER: the grid must resolve the F29 stress envelope
      Δ = ℓ_∞ − ℓ* ∈ [−0.2, +0.3] with usable prior mass on both signs;
      τ = 0.3 puts P(ceiling < cut) ≈ Φ((0.534−0.765)/0.3) = 0.22 for
      domain3 — non-negligible "not trainable to the bar" mass, consistent
      with the F28 heterogeneity draws. Report τ ∈ {0.2, 0.45} sensitivity.
  grid size    J = 7 over ±2.2τ — covers 97% of the prior; spacing
      0.73τ ≈ 0.22 in ℓ is below the filter's own resolution (the F5
      information floor 0.15–0.19 ℓ-SD), so finer grids buy nothing.
      The grid is extended downward if needed so that at least one stratum
      sits ≥ 0.15 BELOW ℓ* — resolving the decision sign is the point.
  static stratum p_static_stratum (default 0, study 0.15) — the "does not
      learn at all" hypothesis (rule="static"), subsuming F31(a); prior mass
      is second-order because posterior odds move exponentially in the
      prequential log-evidence.

OUTPUT. Same API as TaskFilter (mean/sd/pass_mass/is_mastered/ess/
plan_mean/step/propagate_gap + pooled .theta/.ell/.w/.learn/.p for the
policy scorers), plus `trainability(ell_star)` = Σ ω_j·1[grid_j > ℓ*] —
P(this learner's ceiling clears the cut | data), the honest recommendation
statistic D18 asks for. Note for the pooled scorers: `.p` carries the BASE
σ_∞; the σ-reward's (log σ − log σ_∞) factor is per-particle common, so
candidate RANKING within the skill term is unaffected by which σ_∞ is used.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm as _norm

from training.learner_sim import LearnerParams
from training.training_filter import TaskFilter, _DEFAULT_ALPHA, _DEFAULT_Z


def declared_hazard_scale(n_confirms, *, strength=5.0, floor=0.0,
                          geometric=None):
    """M26 (F89): evidence-adaptive Gate-4 hazard multiplier for a task
    that has DECLARED and then re-confirmed across n_confirms boundaries.

    Model. The F81 fixed-share update is exact Bayes for a hidden-Markov
    ceiling that re-draws from the prior with a KNOWN per-boundary hazard
    ρ (pinned 0.10 by study_m23_regime arm G). For a task that has
    demonstrated the declaration gate, each subsequent boundary followed
    by a fresh gate demonstration is an (approximate) observation that no
    downward re-draw happened at that boundary. With ρ ~ Beta(a0, b0)
    (mean ρ0 = a0/c0, prior strength c0 = a0 + b0) and n such
    observations, the posterior-mean hazard is

        ρ_n = ρ0 · c0 / (c0 + n),

    and because BOTH Gate-4 components are linear in their hazard — the
    fixed-share weight re-mix (1−ρ)ω + ρ·prior in ρ, and the two-component
    state-jump mixture in ε_b — plugging the posterior mean in is EXACT
    for the marginal one-boundary update. So the adaptive hazard is the
    pinned (ε_b, w_share) scaled by c0/(c0+n). Λ (the shock SIZE) and the
    θ-scale are not hazard probabilities and do not decay.

    Approximation recorded honestly: a re-draw that lands ABOVE the cut
    also re-confirms, so counting every re-confirmation as "no re-draw"
    over-credits the no-shift hypothesis by P(re-draw clears cut) ≈ the
    prior above-cut mass; the study prices the consequence (regression-
    detection lag, arm G) instead of correcting the count.

    `floor` keeps a permanent fraction of the pinned hazard (the redraw
    process is not claimed stationary-forever; a floor bounds how much
    safety machinery evidence can ever remove). `geometric`, if given,
    replaces the Beta form with λ^n — the aggressive ablation the study
    compares against. n_confirms=0 returns 1.0 (bit-identical to M23:
    a task that has never re-confirmed pays the full pinned hazard).
    Constants pinned by studies/study_m26_hazard.py."""
    n = max(int(n_confirms), 0)
    if geometric is not None:
        s = float(geometric) ** n
    else:
        s = float(strength) / (float(strength) + n)
    return max(float(floor), s)


class AdaptiveBoundaryHazard:
    """M26 (F89): per-task declaration-lifecycle state for the evidence-
    adaptive boundary hazard. Tracks, per task,
      ever     — the task has EVER satisfied the declaration gate
                 (the F86 `ever_declared` key; revocation does not clear it),
      n        — accumulated post-boundary re-confirmations (each Gate-4
                 boundary survived-and-re-demonstrated counts ONCE),
      pending  — a boundary has been applied since the gate was last
                 demonstrated (the "debt" a re-confirmation pays).
    Increment rule: gate demonstrated (D33 confirm, provisional watch, or
    in-session re-mastery) while pending ⇒ n += 1, pending cleared. A D33
    REVOCATION resets n to 0 (direct evidence a re-draw happened; the full
    pinned hazard returns because scale(0) = 1). Tasks that never declared
    always get scale 1.0 — the F80 breakout machinery is untouched.
    The caller applies `scale(k)` to (ε_b, w_share) at boundary time and
    then calls `on_boundary(k)`."""

    def __init__(self, n_tasks, *, strength=5.0, floor=0.0, geometric=None):
        self.strength, self.floor, self.geometric = strength, floor, geometric
        self.ever = [False] * int(n_tasks)
        self.n = [0] * int(n_tasks)
        self.pending = [False] * int(n_tasks)

    def scale(self, k):
        if not self.ever[k]:
            return 1.0
        return declared_hazard_scale(self.n[k], strength=self.strength,
                                     floor=self.floor,
                                     geometric=self.geometric)

    def on_boundary(self, k):
        if self.ever[k]:
            self.pending[k] = True

    def on_gate_pass(self, k):
        """The declaration gate was observed satisfied for task k."""
        if self.ever[k] and self.pending[k]:
            self.n[k] += 1
        self.ever[k] = True
        self.pending[k] = False

    def on_revoked(self, k):
        self.n[k] = 0
        self.pending[k] = False

    def credit_from_pi(self, k, pi, *, alpha=0.05, collapse=0.5):
        """Session-open credit rule. The full declaration gate CANNOT be
        the per-boundary evidence unit: its sd-floor conjunct is exactly
        what the hazard injects, so under the full pinned hazard a
        declared task re-fires the gate ~never (measured: zero
        re-demonstrations in 5 post-declaration sessions at the real
        2-task service rate) and the decay would be unreachable. The
        decision-relevant \"no downward re-draw\" observation is the
        MASS condition alone at the session open — π ≥ 1−α over the
        evidence the previous session accumulated against that session's
        re-widened prior (the gate minus its hazard-injected conjunct;
        a satisfied gate implies it). π < collapse (indifference) is
        positive evidence FOR a re-draw and resets the count, the
        evidence-driven analog of a D33 revocation."""
        if not self.ever[k]:
            return
        if pi >= 1.0 - alpha:
            self.on_gate_pass(k)
        elif pi < collapse:
            self.on_revoked(k)

    # sandbox persistence (plain-json meta, state_io rule: no pickle)
    def to_json(self):
        return {"ever": list(self.ever), "n": list(self.n),
                "pending": list(self.pending)}

    @classmethod
    def from_json(cls, d, n_tasks, **kw):
        h = cls(n_tasks, **kw)
        if d:
            h.ever = [bool(x) for x in d.get("ever", h.ever)]
            h.n = [int(x) for x in d.get("n", h.n)]
            h.pending = [bool(x) for x in d.get("pending", h.pending)]
        return h


class SigmaInfMixtureFilter:
    """Bayesian-model-averaged bank of TaskFilters over a ceiling grid."""

    def __init__(self, theta, ell, params: LearnerParams, *,
                 ell_inf_mean, tau=0.30, J=7, span=2.2,
                 p_static_stratum=0.0, ell_star=None, min_below_cut=0.15,
                 seed=0, w=None, forget=1.0, **filter_kwargs):
        theta = np.asarray(theta, dtype=np.float64)
        ell = np.asarray(ell, dtype=np.float64)
        grid = ell_inf_mean + np.linspace(-span, span, int(J)) * float(tau)
        # guarantee the decision boundary is straddled (see docstring)
        if ell_star is not None and grid.min() > ell_star - min_below_cut:
            grid = np.append(grid, ell_star - min_below_cut)
            grid.sort()
        self.grid = grid
        # truncated-Gaussian prior on the grid (+ optional static stratum)
        lp = -0.5 * ((grid - ell_inf_mean) / float(tau)) ** 2
        pw = np.exp(lp - lp.max())
        pw = pw / pw.sum()
        self.static_idx = None
        self.strata = []
        params_j = [replace(params, sigma_inf=float(np.exp(-g)))
                    for g in grid]
        if p_static_stratum > 0.0:
            self.static_idx = len(params_j)
            params_j.append(replace(params, rule="static"))
            pw = np.append(pw * (1.0 - p_static_stratum), p_static_stratum)
        self.log_w = np.log(pw)
        self._log_prior_w = np.log(pw).copy()   # M23 F81: fixed-share target
        for j, pj in enumerate(params_j):
            self.strata.append(TaskFilter(theta, ell, pj, seed=seed + 101 * j,
                                          w=w, **filter_kwargs))
        self.p = params                      # base params for pooled scorers
        # M23 (F81, opt-in): dynamic-model-averaging forgetting (Raftery,
        # Kárný & Ettler 2010). Exact BMA integrates the WHOLE history, so a
        # regime change leaves the ceiling weights hostage to stale evidence:
        # USER-E's guessing onboarding drove trainability to 0.34 right
        # before a d′ 3.0 breakout session; recovery took 11 trials (USER-C:
        # 0.36, 23 trials) while Gate-1 allocation discounts the task. With
        # forgetting the prior weights are flattened each trial,
        #     ω_j ← ω_j^γ / Σ ω^γ   (then × the prequential evidence),
        # i.e. centered log-weights shrink by γ toward uniform: evidence
        # from k trials back is discounted γ^k, effective window
        # N_eff = 1/(1−γ). Steady-state log-odds cap at (mean per-trial
        # LLR)·N_eff, so the careless/static protection (F54) must be
        # re-verified when enabling — studies/study_m23_regime.py arm C
        # pins γ with false-graduation guardrails. forget=1.0 (default) is
        # bit-identical exact BMA (M22 behavior).
        self.forget = float(forget)

    # ── weights ──
    def weights(self):
        lw = self.log_w - logsumexp(self.log_w)
        return np.exp(lw)

    def trainability(self, ell_star):
        """P(this learner's ceiling clears the cut | data). The static
        stratum counts as not-trainable."""
        om = self.weights()
        ok = np.zeros_like(om)
        ok[: len(self.grid)] = (self.grid > ell_star)
        return float((om * ok).sum())

    # ── one training trial (mirrors TaskFilter.step order) ──
    def step(self, s, y, y_star, *, s_sd=0.0, feedback=True):
        if self.forget < 1.0:
            # DMA forgetting BEFORE this trial's evidence (M23/F81). log_w
            # is a representative vector (normalization is implicit), so
            # scaling it by γ implements ω^γ regardless of centering:
            # exp(γ(log ω + c)) ∝ ω^γ.
            self.log_w = self.forget * self.log_w
        for j, f in enumerate(self.strata):
            pred = f.reweight(s, y, s_sd)
            f.maybe_resample()
            f.propagate(s, y_star, feedback, y=y, s_sd=s_sd)
            self.log_w[j] += np.log(max(pred, 1e-300))
        self.log_w -= self.log_w.max()       # keep numerics bounded

    def propagate_gap(self, dt_seconds, *, widen=0.0):
        for f in self.strata:
            f.propagate_gap(dt_seconds, widen=widen)

    def boundary_jump(self, eps, sd, theta_scale=1.0):
        """M23 (F80): session-boundary regime-shift hazard, applied to every
        non-static stratum (see TaskFilter.boundary_jump). Weights are not
        touched — the shock is state uncertainty, not ceiling evidence."""
        for f in self.strata:
            f.boundary_jump(eps, sd, theta_scale)

    def boundary_shift(self, eps, sd, theta_scale=1.0, w_share=0.0):
        """M23 (F80/F81): the full session-boundary regime-shift update —
        state shock (boundary_jump) plus FIXED-SHARE ceiling re-mixing
        (Herbster & Warmuth 1998):

            ω ← (1 − w_share)·ω + w_share·ω_prior

        This is exact Bayes for a hidden-Markov ceiling that, with
        probability w_share per sitting boundary, re-draws from the prior —
        the model the 2026-07-03 data demand: a strategy epiphany changes
        the REACHABLE ceiling, so ceiling evidence older than the boundary
        is partially stale (USER-E's guessing onboarding held trainability
        at 0.34 into a d′ 3.0 session). Unlike per-trial DMA forgetting
        (`forget=`) — which flattens on EVERY trial and permanently holds
        the cross-strata variance above the declaration floor
        (study_m23_regime arm B: γ=0.98 drops the well-specified
        declaration rate 100% → 30% at ~6× lateness; ablations only) —
        fixed-share re-injects weight variance once per sitting. That
        still raises the ACHIEVABLE end-of-session sd_ℓ cycle floor
        (0.226–0.263 at the pinned hazard vs the M22 gate 0.23), so
        enabling it requires re-measuring the declaration sd_floor per F5
        (arm G: sandbox floor 0.23 → 0.33; declaration then 100% @ median
        94 vs M22's 126). w_share=0 leaves weights untouched; never
        calling this is bit-identical to M22."""
        self.boundary_jump(eps, sd, theta_scale)
        if w_share > 0.0:
            w = self.weights()
            prior = np.exp(self._log_prior_w
                           - logsumexp(self._log_prior_w))
            self.log_w = np.log((1.0 - float(w_share)) * w
                                + float(w_share) * prior + 1e-300)

    # ── pooled cloud (policy scorers: trainer_greedy reads these) ──
    @property
    def theta(self):
        return np.concatenate([f.theta for f in self.strata])

    @property
    def ell(self):
        return np.concatenate([f.ell for f in self.strata])

    @property
    def w(self):
        om = self.weights()
        return np.concatenate([o * f.w for o, f in zip(om, self.strata)])

    @property
    def learn(self):
        return np.concatenate([f.learn for f in self.strata])

    # ── summaries (law of total expectation / variance over strata) ──
    def ess(self):
        w = self.w
        return 1.0 / float((w * w).sum())

    def mean(self):
        om = self.weights()
        ms = np.array([f.mean() for f in self.strata])       # (J, 2)
        mt, ml = (om[:, None] * ms).sum(axis=0)
        return float(mt), float(ml)

    def sd(self):
        om = self.weights()
        ms = np.array([f.mean() for f in self.strata])
        ss = np.array([f.sd() for f in self.strata])
        mt, ml = (om[:, None] * ms).sum(axis=0)
        vt = (om * (ss[:, 0] ** 2 + ms[:, 0] ** 2)).sum() - mt ** 2
        vl = (om * (ss[:, 1] ** 2 + ms[:, 1] ** 2)).sum() - ml ** 2
        return float(np.sqrt(max(vt, 0.0))), float(np.sqrt(max(vl, 0.0)))

    def plan_mean(self):
        mt, ml = self.mean()
        return float(np.exp(-ml)), float(-mt)

    def predictive_p(self, s, s_sd=0.0, lapse=0.025):
        """Posterior-predictive P(y=1 | item) over the pooled cloud,
        s_sd-attenuated (M21: feeds the ConsistencyMonitor's null)."""
        att = np.sqrt(np.exp(-2.0 * self.ell) + float(s_sd) ** 2)
        z = (float(s) + self.theta) / att
        p = lapse + (1.0 - 2.0 * lapse) * _norm.cdf(z)
        return float(np.sum(self.w * p))

    def pass_mass(self, ell_star):
        """Mixture π = Σ ω_j π_j; MCSE combines stratum MCSEs as independent
        (ω treated as known — their own MC noise is second-order because the
        prequential evidence is a sum over all N particles per stratum)."""
        om = self.weights()
        parts = [f.pass_mass(ell_star) for f in self.strata]
        pi = float(sum(o * p for o, (p, _) in zip(om, parts)))
        mcse = float(np.sqrt(sum((o * m) ** 2 for o, (_, m) in zip(om, parts))))
        return pi, mcse

    def is_mastered(self, ell_star, *, sd_floor, alpha=_DEFAULT_ALPHA,
                    Z=_DEFAULT_Z):
        """AD6 gate on the MIXTURE posterior (same semantics as TaskFilter:
        the mixture makes π honest, the gate formula is unchanged)."""
        pi, mcse = self.pass_mass(ell_star)
        _, sd_l = self.sd()
        return bool(sd_l <= sd_floor and (pi - Z * mcse) >= 1.0 - alpha)
