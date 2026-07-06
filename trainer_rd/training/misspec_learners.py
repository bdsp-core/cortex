"""M12/V1 — mis-specified learner zoo (rung 3 of the verification ladder).

Simulated learners that deliberately DIFFER from the model the training filter
assumes (learner_sim soft R–W: probit-lapse response, R–W criterion on the
soft prediction error, exponential log-σ relaxation toward a fixed floor,
λ = 0.025). Each class deviates on EXACTLY ONE named axis so robustness can be
attributed; everything else is inherited from learner_sim.Learner.

Axes (PROJECT_MEMORY §3D rung 3):
  dynamics family   PowerLawLearner, PlateauLearner, DriftingCeilingLearner,
                    MomentumCriterionLearner
  observation model HeavyTailLinkLearner, AsymmetricLapseLearner, FatigueLearner
  adversarial       AntiLearner, careless_learner(), static_below_cut()

Ground truth for graduation honesty (link misspec makes latent σ alone
insufficient): pool accuracy A = E_pool[p_correct] under the learner's TRUE
response function, with the bank's stimulus noise smeared in
(s_real ~ N(s_mean, s_sd²), Gauss–Hermite). The pass bar A_bar is the same
functional at the borderline assumed learner (probit, σ = σ*, t = 0,
λ = LAPSE_RATE). Declared-mastered with A < A_bar ⇒ FALSE GRADUATION.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t

from training.bridge_conventions import LAPSE_RATE
from training.learner_sim import Learner, LearnerParams

# t3 link scaled to match the probit slope at z=0 (so the deviation is purely
# tail weight, not overall discriminability)
_T3_SCALE = float(norm.pdf(0.0) / student_t(df=3).pdf(0.0))

# ── pool-accuracy ground truth ─────────────────────────────────────────────

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(21)


def smeared_p_yes(learner, s_mean, s_sd, task=0):
    """E[p_yes(s_real)] for s_real ~ N(s_mean, s_sd²), via 21-node
    Gauss–Hermite. Works for ANY learner exposing p_yes(s, task); uses the
    learner's CURRENT state (incl. time-varying lapse)."""
    s_mean = np.atleast_1d(np.asarray(s_mean, dtype=np.float64))
    s_sd = np.atleast_1d(np.asarray(s_sd, dtype=np.float64))
    out = np.zeros_like(s_mean)
    for x, w in zip(_GH_X, _GH_W):
        nodes = s_mean + np.sqrt(2.0) * s_sd * x
        p = np.array([learner.p_yes(si, task) for si in nodes])
        out += (w / np.sqrt(np.pi)) * p
    return out


def pool_accuracy(learner, pool, task=0):
    """A = mean over pool items of P(correct) under the learner's true
    response function, stimulus noise included."""
    p1 = smeared_p_yes(learner, pool.s_mean, pool.s_sd, task)
    p_correct = np.where(pool.y_star == 1, p1, 1.0 - p1)
    return float(p_correct.mean())


def accuracy_bar(pool, sigma_star, t_star=0.0):
    """A_bar: pool accuracy of the WORST learner the gate is designed to
    accept (probit, σ = σ*, |t| = t*, λ = LAPSE_RATE); min over the two bias
    signs. Closed form: smearing a probit gives Φ((s_mean−t)/sqrt(σ²+s_sd²)).
    Declared-mastered with A < A_bar ⇒ unambiguously outside the designed
    acceptance envelope ⇒ false graduation."""
    denom = np.sqrt(sigma_star ** 2 + pool.s_sd ** 2)
    out = []
    for t in {+abs(t_star), -abs(t_star)}:
        p1 = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(
            (pool.s_mean - t) / denom)
        out.append(float(np.where(pool.y_star == 1, p1, 1.0 - p1).mean()))
    return min(out)


# ── dynamics-family deviations (response model inherited) ─────────────────

class PowerLawLearner(Learner):
    """Law-of-practice skill curve: σ(u) = σ_∞ + (σ0−σ_∞)(1+u/u0)^(−β), u =
    accumulated difficulty-weighted feedback practice (same w·f gating as the
    assumed model, so the deviation is purely the CURVE SHAPE: faster early,
    much slower late than exponential). Criterion dynamics inherited (R–W).
    Implemented in increment form: logσ steps by the curve increment + the
    usual q_σ random walk."""

    def __init__(self, sigma0, t0, params, *, beta=0.7, u0=10.0, seed=0):
        super().__init__(sigma0, t0, params, seed=seed)
        self.beta, self.u0 = float(beta), float(u0)
        self.sigma00 = self.sigma.copy()        # curve anchor
        self.u = np.zeros(self.K)

    def _curve(self, u, task):
        return (self.sigma_inf[task] + (self.sigma00[task] - self.sigma_inf[task])
                * (1.0 + u / self.u0) ** (-self.beta))

    def step(self, s, task, y_star, feedback=True):
        s, task = float(s), int(task)
        y = self.respond(s, task)
        p = self.p_yes(s, task)
        delta = (p - y_star) if self.p.rule == "soft" else (y - y_star)
        self.t[task] = (self.t[task] + self.p.alpha_t * delta
                        + self.rng.normal(0.0, self.p.q_t))
        du = (1.0 if feedback else 0.0) * self.skill_weight(s, task)
        inc = np.log(self._curve(self.u[task] + du, task)) - \
            np.log(self._curve(self.u[task], task))
        self.u[task] += du
        self.sigma[task] = float(np.exp(np.log(self.sigma[task]) + inc
                                        + self.rng.normal(0.0, self.p.q_sigma)))
        return y


class PlateauLearner(PowerLawLearner):
    """Stagewise/insight learning: σ sits on discrete plateaus and drops
    abruptly when accumulated practice u crosses stage thresholds. Long flat
    stretches + sudden jumps — the anti-case for smooth-relaxation tracking
    and trailing-mean gates."""

    def __init__(self, sigma0, t0, params, *, u_steps=(10.0, 25.0), seed=0):
        super().__init__(sigma0, t0, params, seed=seed)
        self.u_steps = tuple(float(x) for x in u_steps)

    def _curve(self, u, task):
        lo, hi = np.log(self.sigma_inf[task]), np.log(self.sigma00[task])
        stage = sum(u >= th for th in self.u_steps)        # 0..len(u_steps)
        frac = stage / len(self.u_steps)
        return float(np.exp(hi + frac * (lo - hi)))


class DriftingCeilingLearner(Learner):
    """Non-stationary skill floor: σ_∞ drifts upward over the run (attention/
    strategy degradation), here 0.82σ*→~0.95σ* by trial 400. The assumed model
    holds at every instant EXCEPT that its fixed-σ_∞ premise is false."""

    def __init__(self, sigma0, t0, params, *, drift_per_trial=3.7e-4,
                 ceil_cap=None, seed=0):
        super().__init__(sigma0, t0, params, seed=seed)
        self.drift = float(drift_per_trial)
        self.cap = ceil_cap                      # cap on σ_∞ (e.g. 0.97σ*)

    def step(self, s, task, y_star, feedback=True):
        y = super().step(s, task, y_star, feedback)
        new = self.sigma_inf[int(task)] * np.exp(self.drift)
        if self.cap is not None:
            new = min(new, self.cap)
        self.sigma_inf[int(task)] = new
        return y


class MomentumCriterionLearner(Learner):
    """Criterion driven by an EMA-integrated prediction error (sluggish,
    temporally correlated updates) instead of the trial-local R–W delta.
    Same steady-state gain; different transient + autocorrelation."""

    def __init__(self, sigma0, t0, params, *, kappa=0.15, seed=0):
        super().__init__(sigma0, t0, params, seed=seed)
        self.kappa = float(kappa)
        self.e = np.zeros(self.K)

    def step(self, s, task, y_star, feedback=True):
        s, task = float(s), int(task)
        y = self.respond(s, task)
        p = self.p_yes(s, task)
        delta = (p - y_star) if self.p.rule == "soft" else (y - y_star)
        self.e[task] = (1.0 - self.kappa) * self.e[task] + self.kappa * delta
        self.t[task] = (self.t[task] + self.p.alpha_t * self.e[task]
                        + self.rng.normal(0.0, self.p.q_t))
        f = 1.0 if feedback else 0.0
        w = self.skill_weight(s, task)
        log_sig = np.log(self.sigma[task])
        g = -f * w * (log_sig - np.log(self.sigma_inf[task]))
        self.sigma[task] = float(np.exp(log_sig + self.p.alpha_sigma * g
                                        + self.rng.normal(0.0, self.p.q_sigma)))
        return y


# ── observation-model deviations (dynamics inherited) ─────────────────────

class HeavyTailLinkLearner(Learner):
    """t₃ response link, slope-matched to the probit at z=0: identical local
    discriminability, far heavier error tails on easy items (the classic
    psychometric-function mismatch; mimics extra lapse that scales with |z|)."""

    def p_yes(self, s, task):
        z = (float(s) - self.t[task]) / self.sigma[task]
        return self.p.lapse + (1.0 - 2.0 * self.p.lapse) * \
            float(student_t(df=3).cdf(_T3_SCALE * z))


class RealLinkLearner(Learner):
    """EXTSET-fitted observation model (M13.1/W1): asymmetric lapse +
    optional slope-matched t_ν tail, parameters from the task1 binary-frame
    fit (`figures/data_extset_link.npz`). Deviates on EXACTLY the observation
    axis — R–W dynamics inherited unchanged, so the rung-3 stress isolates
    "the real link" against the filter's assumed probit-λ0.025."""

    def __init__(self, sigma0, t0, params, *, lam_fa, lam_miss, nu=None,
                 seed=0):
        super().__init__(sigma0, t0, params, seed=seed)
        self.lam_fa, self.lam_miss = float(lam_fa), float(lam_miss)
        self.nu = None if nu is None else float(nu)
        if self.nu is not None:
            self._scale = float(norm.pdf(0) / student_t(df=self.nu).pdf(0))

    def p_yes(self, s, task):
        z = (float(s) - self.t[task]) / self.sigma[task]
        base = (float(student_t(df=self.nu).cdf(self._scale * z))
                if self.nu is not None else float(norm.cdf(z)))
        return self.lam_fa + (1.0 - self.lam_fa - self.lam_miss) * base


class AsymmetricLapseLearner(Learner):
    """Asymmetric lapse: p = λ_fa + (1−λ_fa−λ_miss)Φ(z). A response bias that
    does NOT live in the criterion — it cannot be trained away by feedback,
    and to a symmetric-lapse filter it masquerades as t ≠ 0. The nastiest
    bias-mode confound."""

    def __init__(self, sigma0, t0, params, *, lam_fa=0.08, lam_miss=0.01,
                 seed=0):
        super().__init__(sigma0, t0, params, seed=seed)
        self.lam_fa, self.lam_miss = float(lam_fa), float(lam_miss)

    def p_yes(self, s, task):
        z = (float(s) - self.t[task]) / self.sigma[task]
        return self.lam_fa + (1.0 - self.lam_fa - self.lam_miss) * norm.cdf(z)


class FatigueLearner(Learner):
    """State-dependent lapse: λ(n) ramps λ0→λ_max over n_ramp trials of time-
    on-task (filter assumes constant λ=0.025). Dynamics inherited."""

    def __init__(self, sigma0, t0, params, *, lam_max=0.12, n_ramp=300,
                 seed=0):
        super().__init__(sigma0, t0, params, seed=seed)
        self.lam0 = params.lapse
        self.lam_max, self.n_ramp = float(lam_max), int(n_ramp)
        self.n = 0

    def p_yes(self, s, task):
        lam = self.lam0 + (self.lam_max - self.lam0) * min(self.n / self.n_ramp, 1.0)
        z = (float(s) - self.t[task]) / self.sigma[task]
        return lam + (1.0 - 2.0 * lam) * norm.cdf(z)

    def step(self, s, task, y_star, feedback=True):
        y = super().step(s, task, y_star, feedback)
        self.n += 1
        return y


# ── adversarial learners ───────────────────────────────────────────────────

class AntiLearner(Learner):
    """Feedback moves the criterion the WRONG way (sign-flipped R–W); skill
    static above the mastery bar. Must never graduate; |t| clipped at ±4 to
    keep the (diverging) simulation finite."""

    def step(self, s, task, y_star, feedback=True):
        s, task = float(s), int(task)
        y = self.respond(s, task)
        p = self.p_yes(s, task)
        delta = (p - y_star) if self.p.rule == "soft" else (y - y_star)
        self.t[task] = float(np.clip(
            self.t[task] - self.p.alpha_t * delta
            + self.rng.normal(0.0, self.p.q_t), -4.0, 4.0))
        return y                                  # σ untouched


def careless_learner(sigma0, t0, params: LearnerParams, seed=0):
    """λ=0.35, learning rates ×0.25: max attainable accuracy 0.65 < any bar,
    so ANY graduation is false. Tests whether massive unmodeled lapse fools
    the filter (it should read as low skill instead)."""
    q = LearnerParams(alpha_t=0.25 * params.alpha_t,
                      alpha_sigma=0.25 * params.alpha_sigma,
                      sigma_inf=params.sigma_inf, q_t=params.q_t,
                      q_sigma=params.q_sigma, rho=params.rho, lapse=0.35,
                      rule=params.rule)
    return Learner(sigma0, t0, q, seed=seed)


def static_below_cut(sigma_star, seed=0, *, frac=1.25):
    """Non-learner parked just below the bar (σ = frac·σ*, t=0): the pure
    false-graduation null."""
    q = LearnerParams(rule="static")
    return Learner([frac * sigma_star], [0.0], q, seed=seed)
