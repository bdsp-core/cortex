"""Learner simulator — the transition kernel T (G1 port of trainer_rd
learner_sim.Learner). RESEARCH-ONLY: the executable ground-truth learner used to
calibrate the gate `sd_floor` (`trainer.filter.steady_state_sd`) and to drive
closed-loop SBC / OC studies. Not part of the deployment runtime.

State is stored in PLAN coordinates (σ, t) — the coords the dynamics equations are
written in. `engine_state()` converts to engine coords (θ = −t, ℓ = −log σ). The
simulator IS the model the filter (`trainer/filter.py`) assumes, so the filter is
correctly specified against it. Verbatim numerics port; `LearnerParams` is shared
from `trainer/dynamics.py`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from trainer.conventions import SKILL_MODE_MULTIPLIER, plan_to_engine
from trainer.dynamics import LearnerParams  # noqa: F401  (re-exported for callers)


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

    # ── skill-weight (85% rule) ──
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
        # criterion prediction error — f-gated (δ requires the true label y*,
        # which the learner only has when feedback fires)
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
