"""Step 6 — Tier-1 one-step greedy policy (plan §6.1, Eq. Qgreedy) + ablations.

The myopic comparator to the Tier-2 mode policy: at each trial pick the bank
item maximizing expected immediate reward under the current belief,

    Q(s) = Σ_i w_i · R(θ^(i), θ̄'^(i))                              (Eq. Qgreedy)

with R = β_t(|t| − |t'|) + β_σ(σ − σ') + β_r·ret  (Eq. reward), and the
deterministic next state θ̄' from the transition means (drop process noise).
Under the soft rule θ̄' is independent of the response y, so the per-particle
reward collapses to one term (plan §6.1 remark).

This module also exposes the F6 degeneracies the review flagged, as REGRESSION
witnesses (PROJECT_MEMORY.md F6):
  * F6a retention spam — an unconstrained retention bonus is maximized by
    re-serving an already-easy "mastered" item every trial;
  * F6b base-rate exploit — the |t|-reduction reward is maximized by one-sided
    label selection (an implicit, unintended base-rate manipulation).
The CONSTRAINED greedy (label-balanced, retention eligibility-gated) does not
exhibit them; the UNCONSTRAINED variant does — that contrast is the test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from training.bridge_conventions import SKILL_MODE_MULTIPLIER, LAPSE_RATE
from training.training_filter import TaskFilter


@dataclass
class RewardWeights:
    beta_t: float = 1.0
    beta_sigma: float = 1.0
    beta_r: float = 0.5


def _expected_reward(filt: TaskFilter, s, y_star, weights: RewardWeights,
                     ret_bonus=0.0, s_sd=0.0, bar_ell=None):
    """Q(s): weighted expected one-step reward over the particle cloud.

    Uses the filter's OWN assumed dynamics (the transition means) to form θ̄',
    exactly as Tier-1 prescribes. Soft rule ⇒ y-independent (single term).
    Accepts scalar s (returns float) or an (n,) array of candidate signals
    paired with (n,) y_star (returns (n,) Q) — the array form broadcasts the
    whole bank against the cloud in one shot.

    s_sd (F20): stimulus uncertainty. The prediction-error term uses the
    attenuated P(yes); the skill-weight term uses the noise-smeared expected
    weight ρ/√(ρ²+τ²)·exp(−(|d|−m)²/(2(ρ²+τ²))), τ = s_sd/σ — exact at
    s_sd=0, fold-then-smear approximation otherwise (fine for ranking).

    bar_ell (M24/F85, opt-in): when set, the σ-progress term is credited
    ONLY for particles still BELOW the bar (ℓ_i < bar_ell) — progress that
    reduces the distance-to-cut, not progress past it. None (default) is
    bit-identical to the plan's Eq. (reward)."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))[:, None]      # (n,1)
    y_star = np.atleast_1d(np.asarray(y_star, dtype=np.float64))[:, None]
    ret_bonus = np.atleast_1d(np.asarray(ret_bonus, dtype=np.float64))[:, None]
    s_sd = np.atleast_1d(np.asarray(s_sd, dtype=np.float64))
    s_sd = (np.zeros_like(s) + s_sd[:, None]) if s_sd.size > 1 else \
        np.full_like(s, float(s_sd[0]))
    p = filt.p
    sigma = np.exp(-filt.ell)[None, :]                               # (1,N)
    t = -filt.theta[None, :]
    el = np.exp(filt.ell)[None, :]
    z = el * (s + filt.theta[None, :])
    z = z / np.sqrt(1.0 + (el * s_sd) ** 2)            # attenuation (F20)
    p_yes = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * ndtr(z)
    delta = p_yes - y_star                              # soft prediction error
    # MA-9: a p_static filter asserts `1−learn` of the mass is non-learning;
    # its expected reward must be zero for that mass (learn=1 ⇒ bit-identical)
    learn = filt.learn[None, :]
    t_new = t + learn * p.alpha_t * delta
    d = np.abs(s - t) / sigma
    v = p.rho ** 2 + (s_sd / sigma) ** 2               # smeared weight (F20)
    w_weight = (p.rho / np.sqrt(v)
                * np.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2) / (2.0 * v)))
    log_sig_new = (np.log(sigma)
                   - learn * p.alpha_sigma * w_weight
                   * (np.log(sigma) - np.log(p.sigma_inf)))
    sigma_new = np.exp(log_sig_new)
    sig_gain = sigma - sigma_new
    if bar_ell is not None:                 # M24/F85 bar-referenced credit
        sig_gain = sig_gain * (filt.ell[None, :] < float(bar_ell))
    R = (weights.beta_t * (np.abs(t) - np.abs(t_new))
         + weights.beta_sigma * sig_gain
         + weights.beta_r * ret_bonus)
    Q = (R * filt.w[None, :]).sum(axis=1)
    return float(Q[0]) if Q.shape[0] == 1 else Q


def greedy_select(filt: TaskFilter, cands, weights: RewardWeights, *,
                  label_balance=0, constrain_balance=True,
                  retention_eligible=None):
    """Pick the candidate maximizing Q(s).

    constrain_balance : if True, the |t|-reward cannot be farmed by one-sided
        labels — candidates whose label worsens the running balance beyond a
        tolerance are dropped (F6b). If False, no constraint (the exploit shows).
    retention_eligible : optional bool array over cands; when provided, the
        retention bonus is granted ONLY to eligible (due) items (F6a). When None,
        no retention bonus is modeled.
    """
    n = len(cands)
    s = cands.s_mean
    y = cands.y_star
    allowed = np.ones(n, dtype=bool)
    if constrain_balance:
        # if we already lean +, disallow further +-label items (and vice versa)
        if label_balance >= 1:
            allowed &= (y == 0)
        elif label_balance <= -1:
            allowed &= (y == 1)
        if not allowed.any():
            allowed = np.ones(n, dtype=bool)
    ret = (np.asarray(retention_eligible, dtype=np.float64)
           if retention_eligible is not None else np.zeros(n))
    Q = _expected_reward(filt, s, y.astype(np.float64), weights, ret_bonus=ret,
                         s_sd=cands.s_sd)
    Q = np.where(allowed, Q, -np.inf)
    return int(np.argmax(Q)), Q
