"""M16 — Phase-3 offline dynamics fitting (the F4/D7 deferred estimator).

WHAT. Per-learner maximum-likelihood (weakly-penalized) estimation of the
learning-dynamics parameters from logged training trials, via the particle
filter's PREQUENTIAL EVIDENCE: the marginal likelihood of a state-space model
factorizes as  log p(y_1:n | φ) = Σ_k log p(y_k | y_1:k−1, φ),  and each term
is exactly the normalizer `TaskFilter.reweight` already returns. No new
inference machinery — the filter IS the likelihood evaluator.

ESTIMATION CHOICES (each justified):
  parameters fitted  φ = (log α_t, log α_σ, ℓ_∞) — the three that decide
      trainer behavior (rates → timing, ceiling → graduatability, F54).
      q_t, q_σ, ρ are HELD at base values: they are weakly identified from
      single sessions (noise variances confound with rate variability at
      n ≪ τ; the F4/D7 argument) and mis-setting them is the characterized-
      safe direction (F17/D14). Sensitivity is a prespecified follow-up.
  log/−log transforms  keep positivity without constrained optimization.
  CRN evidence  the filter seed is FIXED across objective evaluations, making
      the evidence deterministic in φ (piecewise-smooth: resample branch
      points move discretely — standard simulated-ML practice; Nelder–Mead
      is used precisely because the objective is not differentiable).
  weak penalty  −log N(φ; φ_0, 1.5²·I) in transformed space. SD 1.5 in log
      space = a ×4.5 one-sigma band — wider than the largest heterogeneity
      stressed in F28 (±4×), so genuine signal is essentially unbiased while
      runaway excursions on evidence plateaus are clipped. This makes the
      estimator an EB-flavored MAP, honestly labeled.
  n_particles = 300  evidence MC error is O(1/√N) per trial and CRN cancels
      most of it across φ; 300 keeps a 150–300-trial fit at ~5–20 s. The
      fitting filter uses the EXACT kernel (M15 F53) so fitted dynamics do
      not absorb kernel bias — the audit's Phase-3 prescription.

USE. `fit_learner(trials, base_params)` → FitResult; `filter_evidence`
doubles as the held-out scorer (score_from= for temporal train/test splits)
and the rule-form comparator (F12: fit under rule="soft" and rule="hard",
compare held-out evidence).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from training.learner_sim import LearnerParams
from training.training_filter import TaskFilter


@dataclass
class FitResult:
    params: LearnerParams
    x: np.ndarray                 # (log α_t, log α_σ, ℓ_∞) at the optimum
    logZ: float                   # penalized objective's evidence part
    n_trials: int
    converged: bool


def filter_evidence(trials, params: LearnerParams, *, n_particles=300,
                    seed=0, prior_sd=0.30, prior_mean=(0.0, 0.0),
                    exact_kernel=True, score_from=0):
    """Prequential log-evidence Σ_{k≥score_from} log p(y_k | H_k, params).

    trials: iterable of (s, y, y_star, s_sd) or (s, y, y_star, s_sd, fb) —
    the optional 5th element marks whether feedback was shown on that trial
    (M17/EXTSET: reads without consensus gold enter with fb=0 — the response
    still informs the STATE via reweight, but no dynamics update fires,
    which is exactly the MA-3 semantics). prior_*: the initial-belief spec
    the LOGGING trainer used (the pilot analog is the eval-seed cloud).
    score_from: index of the first SCORED trial — the belief still filters
    through the earlier trials (that is the point of a temporal split:
    predict the second half given the first under the candidate dynamics)."""
    rng = np.random.default_rng(seed)
    th0 = prior_mean[0] + prior_sd * rng.standard_normal(n_particles)
    el0 = prior_mean[1] + prior_sd * rng.standard_normal(n_particles)
    f = TaskFilter(th0, el0, params, seed=seed + 1,
                   exact_kernel=exact_kernel)
    logZ = 0.0
    for k, row in enumerate(trials):
        s, y, y_star, s_sd = row[0], row[1], row[2], row[3]
        fb = bool(row[4]) if len(row) > 4 else True
        pred = f.reweight(float(s), int(y), float(s_sd))
        if k >= score_from:
            logZ += float(np.log(max(pred, 1e-300)))
        f.maybe_resample()
        f.propagate(float(s), int(y_star), fb, y=int(y), s_sd=float(s_sd))
    return logZ


def fit_learner(trials, base_params: LearnerParams, *, rule=None,
                n_particles=300, seed=0, maxiter=150, penalty_sd=1.5,
                prior_sd=0.30, prior_mean=(0.0, 0.0)):
    """Penalized-ML fit of (α_t, α_σ, σ_∞) on one learner's logged trials.

    rule: dynamics rule to fit under (default: base_params.rule) — pass
    "soft"/"hard" explicitly for the F12 rule-form comparison.
    prior_sd/prior_mean: the initial-belief spec of the evidence filter —
    must match how the logging trainer was seeded."""
    rule = rule or base_params.rule
    x0 = np.array([np.log(base_params.alpha_t),
                   np.log(base_params.alpha_sigma),
                   -np.log(base_params.sigma_inf)])

    def unpack(x):
        return replace(base_params, alpha_t=float(np.exp(x[0])),
                       alpha_sigma=float(np.exp(x[1])),
                       sigma_inf=float(np.exp(-x[2])), rule=rule)

    def nobj(x):
        lz = filter_evidence(trials, unpack(x), n_particles=n_particles,
                             seed=seed, prior_sd=prior_sd,
                             prior_mean=prior_mean)
        pen = 0.5 * float(((x - x0) / penalty_sd) ** 2 @ np.ones(3))
        return -(lz - pen)

    res = minimize(nobj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3,
                            "fatol": 1e-3, "adaptive": True})
    xh = np.asarray(res.x)
    pen = 0.5 * float(((xh - x0) / penalty_sd) ** 2 @ np.ones(3))
    return FitResult(params=unpack(xh), x=xh, logZ=float(-res.fun - pen),
                     n_trials=len(trials), converged=bool(res.success))
