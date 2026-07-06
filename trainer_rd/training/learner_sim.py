"""Step 2 — learner simulator: the transition kernel T (the learning dynamics).

Executable ground truth for the trainer. Implements the plan's postulated
dynamics (learning_algorithm_plan.md §4.4–4.5) as a direct transcription, so
the simulator IS the model the Step-3 filter assumes.

State is stored in PLAN coordinates (σ, t) — the coords the dynamics equations
are literally written in (σ = perceptual noise, t = criterion). Convert to
ENGINE coords (θ = −t, ℓ = −log σ) with `engine_state()` for the filter / eval
engine. See PROJECT_MEMORY.md §2 and bridge_conventions.py.

Dynamics (per task, independent in Phase 2 — no cross-task transfer, D2/F8):

  response   y ~ Bernoulli( p ),  p = λ + (1−2λ)·Φ((s − t)/σ)
  criterion  t' = t + f·α_t·δ + ξ_t,         ξ_t ~ N(0, q_t²)
                 δ_soft = p − y*   (default)   |  δ_hard = y − y*  (ablation)
                 (f-gated since M15/MA-3: δ needs y*, which only feedback
                  reveals — the ungated form was F22, informationally
                  impossible and the cause of the F34-i replay inflation)
  skill      logσ' = logσ + α_σ·gσ + ξ_σ,     ξ_σ ~ N(0, q_σ²)
                 gσ = −f · w(s) · (logσ − logσ_∞)
                 w(s) = exp( −(|s−t|/σ − m)² / (2ρ²) ),  m = SKILL_MODE_MULTIPLIER

α_σ = 1/τ_σ when w=f=1. f = 1 iff veridical feedback delivered this trial.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from training.bridge_conventions import (LAPSE_RATE, SKILL_MODE_MULTIPLIER,
                                plan_to_engine)

RULES = ("soft", "hard", "static")


@dataclass
class LearnerParams:
    """Per-learner dynamics parameters (one set, broadcast across tasks unless
    given as arrays). σ_∞ defaults are the data-grounded expert ceilings
    (PROJECT_MEMORY.md §2B / D10) — caller passes the per-task vector.
    """
    alpha_t: float = 0.15          # criterion learning rate (D7 placeholder)
    alpha_sigma: float = 0.02      # skill rate = 1/τ_σ (τ≈50 trials)
    sigma_inf: float = 0.5         # skill floor (override per task from §2B)
    q_t: float = 0.05              # criterion process-noise SD
    q_sigma: float = 0.02          # log-skill process-noise SD
    rho: float = 0.5               # 85%-weight width
    lapse: float = LAPSE_RATE
    rule: str = "soft"

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")


class Learner:
    """A simulated trainee with per-task true state (σ_k, t_k) evolving under T."""

    def __init__(self, sigma0, t0, params: LearnerParams, seed=0):
        self.sigma = np.atleast_1d(np.asarray(sigma0, dtype=np.float64)).copy()
        self.t = np.atleast_1d(np.asarray(t0, dtype=np.float64)).copy()
        if self.sigma.shape != self.t.shape:
            raise ValueError("sigma0 and t0 must have the same shape")
        self.K = self.sigma.shape[0]
        self.p = params
        self.sigma_inf = np.broadcast_to(
            np.asarray(params.sigma_inf, dtype=np.float64), (self.K,)).copy()
        self.rng = np.random.default_rng(seed)

    # ── observation ──
    def p_yes(self, s, task):
        z = (float(s) - self.t[task]) / self.sigma[task]
        return self.p.lapse + (1.0 - 2.0 * self.p.lapse) * norm.cdf(z)

    def respond(self, s, task):
        return int(self.rng.random() < self.p_yes(s, task))

    # ── skill-weight (85% rule, Eq. weight) ──
    def skill_weight(self, s, task):
        d = abs(float(s) - self.t[task]) / self.sigma[task]
        return float(np.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2)
                            / (2.0 * self.p.rho ** 2)))

    # ── one trial: respond, then update state from feedback ──
    def step(self, s, task, y_star, feedback=True):
        """Present signal s on `task`, return the learner's response y, and
        advance (σ_task, t_task) by one transition. y_star ∈ {0,1} is the
        ground-truth label; feedback gates skill learning (f)."""
        s = float(s)
        task = int(task)
        y = self.respond(s, task)
        if self.p.rule == "static":
            return y
        p = self.p_yes(s, task)
        # criterion prediction error — f-gated (MA-3 fix of F22): δ requires
        # the true label y*, which the learner only has when feedback fires
        f = 1.0 if feedback else 0.0
        if self.p.rule == "soft":
            delta = p - y_star
        else:                                  # hard
            delta = y - y_star
        t_new = (self.t[task] + f * self.p.alpha_t * delta
                 + self.rng.normal(0.0, self.p.q_t))
        # skill relaxation toward floor, gated by feedback + difficulty weight
        w = self.skill_weight(s, task)
        log_sig = np.log(self.sigma[task])
        g_sigma = -f * w * (log_sig - np.log(self.sigma_inf[task]))
        log_sig_new = (log_sig + self.p.alpha_sigma * g_sigma
                       + self.rng.normal(0.0, self.p.q_sigma))
        self.t[task] = t_new
        self.sigma[task] = float(np.exp(log_sig_new))
        return y

    # ── coordinate views ──
    def engine_state(self):
        """Return (theta, ell) engine-coord vectors for the filter/eval engine."""
        theta, ell = plan_to_engine(self.sigma, self.t)
        return theta, ell

    def state(self):
        """Return a copy of (sigma, t) plan-coord vectors."""
        return self.sigma.copy(), self.t.copy()
