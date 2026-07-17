"""Canonical parameter-convention bridge (G1 port of trainer_rd bridge_conventions).

Verbatim numerics port of `trainer_rd/training/bridge_conventions.py` — bit-parity
gated by `tests/test_trainer_g1_conventions.py` against golden vectors generated
from the scratch module. Only the module docstring/home changed; every float
operation is identical so the port reproduces the scratch bitwise.

The trainer plan and the eval engine (engine/core.py) parameterize the SAME
probit-lapse observation model two ways:

    plan   : z = (s − t)/σ,        state (σ, t),  skill = 1/σ
    engine : z = exp(ℓ)·(s + θ),   state (θ, ℓ),  skill = exp(ℓ)

so the exact bridge is  σ = exp(−ℓ),  t = −θ  (⚠ SIGN FLIP on the criterion),
and both sides share  P(y=1 | z) = λ + (1 − 2λ)·Φ(z)  with λ = 0.025 locked.

D6: skill-building mode targets the Wilson et al. (2019) optimum (training error
Φ(−1) ≈ 0.1587), giving SKILL_MODE_MULTIPLIER = m(Φ(1)) ≈ 1.0770 under λ = 0.025.
"""
import numpy as np
from scipy.stats import norm

# Must equal engine/core.py LAPSE_RATE (🔒 locked invariant; asserted in
# tests/test_trainer_g1_conventions.py rather than imported, so this module
# stays dependency-light — same contract as the scratch bridge).
LAPSE_RATE = 0.025

# Wilson et al. (2019) optimal training accuracy: error rate Φ(−1) ≈ 0.1587.
WILSON_OPT_ACC = float(norm.cdf(1.0))            # ≈ 0.841345


# ───────────── parameter conversions ─────────────

def engine_to_plan(theta, ell):
    """(θ, ℓ) engine coords → (σ, t) plan coords. Vectorized."""
    theta = np.asarray(theta, dtype=np.float64)
    ell = np.asarray(ell, dtype=np.float64)
    return np.exp(-ell), -theta


def plan_to_engine(sigma, t):
    """(σ, t) plan coords → (θ, ℓ) engine coords. Vectorized; σ > 0."""
    sigma = np.asarray(sigma, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return -t, -np.log(sigma)


# ───────────── observation model (both conventions) ─────────────

def p_yes_plan(s, sigma, t, lapse=LAPSE_RATE):
    """P(y=1) in plan coords: λ + (1−2λ)·Φ((s − t)/σ)."""
    z = (np.asarray(s, dtype=np.float64) - t) / sigma
    return lapse + (1.0 - 2.0 * lapse) * norm.cdf(z)


def p_yes_engine(s, theta, ell, lapse=LAPSE_RATE):
    """P(y=1) in engine coords: λ + (1−2λ)·Φ(exp(ℓ)·(s + θ))."""
    z = np.exp(np.asarray(ell, dtype=np.float64)) * (np.asarray(s, dtype=np.float64) + theta)
    return lapse + (1.0 - 2.0 * lapse) * norm.cdf(z)


# ───────────── difficulty placement (lapse-corrected) ─────────────

def difficulty_multiplier(target_acc, lapse=LAPSE_RATE):
    """The |s − t|/σ giving expected accuracy `target_acc` under the lapse model.

    m = Φ⁻¹((a − λ)/(1 − 2λ)). Requires λ < a < 1 − λ (the achievable band).
    """
    a = float(target_acc)
    if not (lapse < a < 1.0 - lapse):
        raise ValueError(
            f"target accuracy {a} outside achievable band ({lapse}, {1.0 - lapse})")
    return float(norm.ppf((a - lapse) / (1.0 - 2.0 * lapse)))


def accuracy_at_multiplier(m, lapse=LAPSE_RATE):
    """Inverse of difficulty_multiplier: expected accuracy at |s − t|/σ = m."""
    return float(lapse + (1.0 - 2.0 * lapse) * norm.cdf(float(m)))


# Decision D6: the skill-building mode placement.
SKILL_MODE_MULTIPLIER = difficulty_multiplier(WILSON_OPT_ACC)      # ≈ 1.0770


def skill_mode_multiplier(lapse=LAPSE_RATE, target_acc=WILSON_OPT_ACC):
    """λ-parameterized D6 placement constant (port scaffolding). At port, derive
    placement constants from the instrument's λ through this function rather than
    importing the frozen SKILL_MODE_MULTIPLIER; asymmetric lapse generalizes via
    the same formula with the label-conditional λ."""
    return difficulty_multiplier(target_acc, lapse=lapse)


# ───────────── skill ↔ AUROC ─────────────

def auroc_from_ell(ell):
    """AUROC(ℓ) = Φ(√2/√(exp(−2ℓ)+1)); identical to auroc.auroc_from_l."""
    ell = np.asarray(ell, dtype=np.float64)
    return norm.cdf(np.sqrt(2.0) / np.sqrt(np.exp(-2.0 * ell) + 1.0))


def auroc_from_sigma(sigma):
    """AUROC in plan coords: σ = exp(−ℓ) ⇒ AUROC = Φ(√2/√(σ² + 1))."""
    sigma = np.asarray(sigma, dtype=np.float64)
    return norm.cdf(np.sqrt(2.0) / np.sqrt(sigma * sigma + 1.0))


def sigma_star_from_ell_star(ell_star):
    """Trainer mastery target σ*_k = exp(−ℓ*_k) from a cert cut-score (F14)."""
    return np.exp(-np.asarray(ell_star, dtype=np.float64))
