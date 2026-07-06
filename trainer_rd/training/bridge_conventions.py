"""Step 0 — canonical parameter-convention bridge (PROJECT_MEMORY.md §2).

The trainer plan (learning_algorithm_plan.md) and the eval engine
(core_mcmc_general.py) parameterize the SAME probit-lapse observation model
two different ways:

    plan   : z = (s − t)/σ,        state (σ, t),  skill = 1/σ
    engine : z = exp(ℓ)·(s + θ),   state (θ, ℓ),  skill = exp(ℓ)

so the exact bridge is

    σ = exp(−ℓ),   t = −θ          (⚠ SIGN FLIP on the bias/criterion)

and both sides share  P(y=1 | z) = λ + (1 − 2λ)·Φ(z)  with λ = 0.025 locked.

This module is the ONLY place conversions and difficulty-placement constants
may be defined. Finding F2 (PROJECT_MEMORY.md §3): the plan's Eq.
(eq:eightyfive) constant 1.04 is the *no-lapse* 85% point and yields 83.3%
accuracy under λ = 0.025. The lapse-corrected placement multiplier is

    m(a) = Φ⁻¹( (a − λ) / (1 − 2λ) )        # |s − t|/σ for target accuracy a

Decision D6: skill-building mode targets the Wilson et al. (2019) optimum
(training ERROR rate Φ(−1) ≈ 0.1587, i.e. accuracy Φ(1) ≈ 0.8413), giving
SKILL_MODE_MULTIPLIER = m(Φ(1)) ≈ 1.0770 under λ = 0.025.
"""
import numpy as np
from scipy.stats import norm

# Must equal core_mcmc_general.LAPSE_RATE (🔒 locked invariant; asserted in
# test_step0.py rather than imported, so this module stays dependency-light).
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


# Decision D6 (PROJECT_MEMORY.md §4): the skill-building mode placement.
SKILL_MODE_MULTIPLIER = difficulty_multiplier(WILSON_OPT_ACC)      # ≈ 1.0770


def skill_mode_multiplier(lapse=LAPSE_RATE, target_acc=WILSON_OPT_ACC):
    """λ-parameterized D6 placement constant (M15 MA-4, port scaffolding).

    The frozen SKILL_MODE_MULTIPLIER is m(Φ(1); λ=0.025), but the EXTSET
    Bayesian refit puts the real lapses at λ_fa≈0.0024 / λ_miss≈0.0079–0.0095
    (F43/F51, P(both < 0.025 | data)=0.9999). At those values the correct
    multiplier is 1.007–1.028 and the frozen 1.0772 trains at ~85.2–85.8%
    accuracy instead of the Wilson-optimal 84.1%. At port, derive the
    constant from the instrument's λ through this function (and every other
    placement constant through difficulty_multiplier) rather than importing
    the frozen value; asymmetric lapse generalizes via the same formula with
    the label-conditional λ."""
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
