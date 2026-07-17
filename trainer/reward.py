"""One-step expected-reward scorer (G1 port of trainer_rd trainer_greedy's shared
`RewardWeights` + `_expected_reward`). The tier-2 policy's bias-correction and
expected-progress scorers are thin wrappers over this. Verbatim numerics port;
bit-parity gated by `tests/test_trainer_g1_policy.py`. (The full tier-1
`greedy_select` ablation is not shipped.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from trainer.conventions import SKILL_MODE_MULTIPLIER, LAPSE_RATE


@dataclass
class RewardWeights:
    beta_t: float = 1.0
    beta_sigma: float = 1.0
    beta_r: float = 0.5


def _expected_reward(filt, s, y_star, weights: RewardWeights,
                     ret_bonus=0.0, s_sd=0.0, bar_ell=None):
    """Q(s): weighted expected one-step reward over the particle cloud, using the
    filter's OWN assumed dynamics (the transition means) to form θ̄'. Soft rule ⇒
    y-independent. Accepts scalar s (returns float) or an (n,) array of candidate
    signals paired with (n,) y_star (returns (n,) Q) — the array form broadcasts
    the whole bank against the cloud in one shot.

    s_sd (F20): stimulus uncertainty; the prediction-error term uses the
    attenuated P(yes), the skill-weight term the noise-smeared expected weight.
    bar_ell (opt-in): credit σ-progress only for particles still below the bar."""
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
    learn = filt.learn[None, :]                        # p_static mass ⇒ no reward
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
    if bar_ell is not None:                 # bar-referenced credit (F85)
        sig_gain = sig_gain * (filt.ell[None, :] < float(bar_ell))
    R = (weights.beta_t * (np.abs(t) - np.abs(t_new))
         + weights.beta_sigma * sig_gain
         + weights.beta_r * ret_bonus)
    Q = (R * filt.w[None, :]).sum(axis=1)
    return float(Q[0]) if Q.shape[0] == 1 else Q
