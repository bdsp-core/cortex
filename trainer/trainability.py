"""Per-learner σ_∞ mixture filter — the trainability BMA (G1 port of trainer_rd
mixture_filter). Verbatim numerics; bit-parity gated by
`tests/test_trainer_g1_trainability.py` against scratch golden vectors.

WHY. A TaskFilter that conditions on the skill ceiling σ_∞ at its population point
value can certify its own prior (a static sub-cut learner gets declared because the
filter "knows" ℓ_∞ > ℓ*). WHAT. Bayesian model averaging over a discrete grid of
ceiling hypotheses: J parallel TaskFilter strata, stratum j conditioned on
ℓ_∞ = grid_j, with prior weights from a truncated Gaussian and posterior weights
updated by each stratum's PREQUENTIAL evidence (the one-step predictive likelihood
`TaskFilter.reweight` returns). A DISCRETE mixture avoids the static-parameter path
degeneracy that kills per-particle σ_∞ estimation.

OUTPUT. Same API as TaskFilter, plus `trainability(ell_star)` = Σ ω_j·1[grid_j > ℓ*]
— P(this learner's ceiling clears the cut | data), the honest recommendation
statistic (D18): the trainer recommends, the exam re-certifies.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm as _norm

from trainer.dynamics import LearnerParams
from trainer.filter import TaskFilter, _DEFAULT_ALPHA, _DEFAULT_Z


def declared_hazard_scale(n_confirms, *, strength=5.0, floor=0.0,
                          geometric=None):
    """Evidence-adaptive Gate-4 hazard multiplier for a task that has DECLARED and
    re-confirmed across n_confirms boundaries (opt-in, F89). With ρ ~ Beta(a0, b0)
    (mean ρ0, strength c0) and n "no re-draw" observations the posterior-mean
    hazard is ρ_n = ρ0·c0/(c0+n); both Gate-4 components are linear in their
    hazard, so scaling the pinned constants by c0/(c0+n) is exact for the marginal
    one-boundary update. `floor` keeps a permanent fraction; `geometric` replaces
    the Beta form with λ^n. n_confirms=0 returns 1.0 (full pinned hazard)."""
    n = max(int(n_confirms), 0)
    if geometric is not None:
        s = float(geometric) ** n
    else:
        s = float(strength) / (float(strength) + n)
    return max(float(floor), s)


class AdaptiveBoundaryHazard:
    """Per-task declaration-lifecycle state for the evidence-adaptive boundary
    hazard (opt-in, F89). Tracks `ever` (task ever satisfied the declaration
    gate), `n` (post-boundary re-confirmations), `pending` (a boundary applied
    since the gate was last demonstrated). The caller applies `scale(k)` to
    (ε_b, w_share) at boundary time, then calls `on_boundary(k)`."""

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
        """Session-open credit rule: the decision-relevant "no downward re-draw"
        observation is the MASS condition alone at session open — π ≥ 1−α credits
        a re-confirmation; π < collapse (indifference) is positive evidence FOR a
        re-draw and resets the count (the evidence-driven analog of a revocation)."""
        if not self.ever[k]:
            return
        if pi >= 1.0 - alpha:
            self.on_gate_pass(k)
        elif pi < collapse:
            self.on_revoked(k)

    # sandbox persistence (plain-json meta, no pickle)
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
        self._log_prior_w = np.log(pw).copy()   # F81: fixed-share target
        for j, pj in enumerate(params_j):
            self.strata.append(TaskFilter(theta, ell, pj, seed=seed + 101 * j,
                                          w=w, **filter_kwargs))
        self.p = params                      # base params for pooled scorers
        # dynamic-model-averaging forgetting (opt-in, F81): ω_j ← ω_j^γ / Σ ω^γ,
        # effective window N_eff = 1/(1−γ). forget=1.0 (default) is bit-identical
        # exact BMA.
        self.forget = float(forget)

    # ── weights ──
    def weights(self):
        lw = self.log_w - logsumexp(self.log_w)
        return np.exp(lw)

    def trainability(self, ell_star):
        """P(this learner's ceiling clears the cut | data). The static stratum
        counts as not-trainable."""
        om = self.weights()
        ok = np.zeros_like(om)
        ok[: len(self.grid)] = (self.grid > ell_star)
        return float((om * ok).sum())

    # ── one training trial (mirrors TaskFilter.step order) ──
    def step(self, s, y, y_star, *, s_sd=0.0, feedback=True):
        if self.forget < 1.0:
            # DMA forgetting BEFORE this trial's evidence (F81)
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
        """Session-boundary regime-shift hazard, applied to every non-static
        stratum. Weights are not touched — the shock is state uncertainty, not
        ceiling evidence."""
        for f in self.strata:
            f.boundary_jump(eps, sd, theta_scale)

    def boundary_shift(self, eps, sd, theta_scale=1.0, w_share=0.0):
        """The full session-boundary regime-shift update — state shock plus
        FIXED-SHARE ceiling re-mixing (Herbster & Warmuth):
            ω ← (1 − w_share)·ω + w_share·ω_prior
        exact Bayes for a hidden-Markov ceiling that re-draws from the prior with
        probability w_share per boundary. w_share=0 leaves weights untouched;
        never calling this is bit-identical."""
        self.boundary_jump(eps, sd, theta_scale)
        if w_share > 0.0:
            w = self.weights()
            prior = np.exp(self._log_prior_w
                           - logsumexp(self._log_prior_w))
            self.log_w = np.log((1.0 - float(w_share)) * w
                                + float(w_share) * prior + 1e-300)

    # ── pooled cloud (policy scorers read these) ──
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
        s_sd-attenuated."""
        att = np.sqrt(np.exp(-2.0 * self.ell) + float(s_sd) ** 2)
        z = (float(s) + self.theta) / att
        p = lapse + (1.0 - 2.0 * lapse) * _norm.cdf(z)
        return float(np.sum(self.w * p))

    def pass_mass(self, ell_star):
        """Mixture π = Σ ω_j π_j; MCSE combines stratum MCSEs as independent."""
        om = self.weights()
        parts = [f.pass_mass(ell_star) for f in self.strata]
        pi = float(sum(o * p for o, (p, _) in zip(om, parts)))
        mcse = float(np.sqrt(sum((o * m) ** 2 for o, (_, m) in zip(om, parts))))
        return pi, mcse

    def is_mastered(self, ell_star, *, sd_floor, alpha=_DEFAULT_ALPHA,
                    Z=_DEFAULT_Z):
        """AD6 gate on the MIXTURE posterior (same semantics as TaskFilter: the
        mixture makes π honest, the gate formula is unchanged)."""
        pi, mcse = self.pass_mass(ell_star)
        _, sd_l = self.sd()
        return bool(sd_l <= sd_floor and (pi - Z * mcse) >= 1.0 - alpha)
