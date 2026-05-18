"""F0.1 + T1.4 regression — lapse rate λ = 0.025 under spike-paper Eq. 2.

Pins the spike-paper-canonical symmetric lapse mixture:
    P(y=1 | z) = λ + (1 − 2λ)·Φ(z)
with floor λ and ceiling 1 − λ.

The previous parametrization `(1 − λ)·Φ(z) + 0.5·λ` (floor λ/2, ceiling 1 − λ/2)
corresponded to a 4AFC guess rate and was algebraically inconsistent with
`train_val_split_and_fit.py:probit_lapse_nll` in the spike repo. See spike paper
CLAUDE.md §2.1 Eq. 2 for the canonical form.

Pins:
  - LAPSE_RATE = 0.025 in core_mcmc.py, core_mcmc_brute_k.py, core.py.
  - P(y=1 | z=10)  = λ + (1−2λ)·Φ(10)  ≈ 0.975  (NOT 0.9875).
  - P(y=1 | z=-10) = λ + (1−2λ)·Φ(-10) ≈ 0.025  (NOT 0.0125).
  - log P(y=1 | z=10)  ≈ log(0.975) ≈ -0.02532.
  - log P(y=0 | z=10)  ≈ log(0.025) ≈ -3.6889.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

import core
import core_mcmc
import core_mcmc_brute_k


# ---------------------------------------------------------------------------
# constant consistency across modules
# ---------------------------------------------------------------------------

def test_lapse_rate_constant_consistent():
    assert core_mcmc.LAPSE_RATE == 0.025
    assert core_mcmc_brute_k.LAPSE_RATE == 0.025
    assert core.LAPSE_RATE == 0.025


# ---------------------------------------------------------------------------
# numerical correctness of the spike-paper symmetric lapse mixture (F0.1)
# ---------------------------------------------------------------------------

def _expected_p_y1(z, lam=0.025):
    """Spike-paper Eq. 2: P(y=1|z) = λ + (1 − 2λ)·Φ(z)."""
    return lam + (1.0 - 2.0 * lam) * float(norm.cdf(z))


def test_p_response_yes_upper_tail():
    """At z = +10 the symmetric lapse caps at 1 − λ = 0.975, NOT 0.9875."""
    p = float(core_mcmc._p_response_yes(10.0))
    np.testing.assert_allclose(p, _expected_p_y1(10.0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(p, 1.0 - 0.025, atol=1e-9)
    assert p < 1.0 - 1e-3, "lapse-mixture P(y=1) at z=10 must be strictly below 1"


def test_p_response_yes_lower_tail():
    """At z = -10 the symmetric lapse floors at λ = 0.025, NOT 0.0125."""
    p = float(core_mcmc._p_response_yes(-10.0))
    np.testing.assert_allclose(p, _expected_p_y1(-10.0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(p, 0.025, atol=1e-9)
    assert p > 1e-3, "lapse-mixture P(y=1) at z=-10 must be strictly above 0"


def test_p_response_yes_at_zero():
    """At z = 0 the symmetric lapse passes through 0.5 (Φ(0) = 0.5)."""
    p = float(core_mcmc._p_response_yes(0.0))
    np.testing.assert_allclose(p, 0.5, atol=1e-12)


def test_brute_p_y1_matches_hier():
    """Brute and hier helpers must produce the same response probabilities."""
    z = np.linspace(-12, 12, 25)
    a = core_mcmc._p_response_yes(z)
    b = core_mcmc_brute_k._p_y1(z)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_core_response_prob_matches_hier():
    """core.response_prob must agree with core_mcmc._p_response_yes."""
    z = np.linspace(-12, 12, 25)
    # core.response_prob takes (s, t, l) but for a direct comparison we use l=0, t=0, s=z
    a = core_mcmc._p_response_yes(z)
    b = np.array([core.response_prob(zi, 0.0, 0.0) for zi in z])
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# log-likelihood numerics at the saturating ends (F0.1)
# ---------------------------------------------------------------------------

def test_log_p_response_y1_at_z10():
    """log P(y=1 | z=10) ≈ log(0.975) ≈ -0.02532."""
    val = float(core_mcmc._log_p_response(np.array(10.0), 1))
    np.testing.assert_allclose(val, np.log(0.975), rtol=1e-6, atol=1e-9)


def test_log_p_response_y0_at_z10():
    """log P(y=0 | z=10) ≈ log(0.025) ≈ -3.6889 (the spike-paper lapse floor)."""
    val = float(core_mcmc._log_p_response(np.array(10.0), 0))
    np.testing.assert_allclose(val, np.log(0.025), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(val, -3.6888794541139363, atol=1e-6)


def test_log_p_response_y1_at_z_minus10():
    """log P(y=1 | z=-10) ≈ log(0.025) (symmetric to y=0 at z=+10)."""
    val = float(core_mcmc._log_p_response(np.array(-10.0), 1))
    np.testing.assert_allclose(val, np.log(0.025), rtol=1e-6, atol=1e-9)


def test_brute_log_p_y_matches_hier_log_p_response():
    """`_log_p_y` (brute) and `_log_p_response` (hier) must agree pointwise."""
    z_grid = np.linspace(-12, 12, 25)
    for y in (0, 1):
        a = core_mcmc._log_p_response(z_grid, y)
        b = core_mcmc_brute_k._log_p_y(z_grid, y)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_core_log_response_prob_matches_hier():
    """core.log_response_prob must agree with core_mcmc._log_p_response."""
    z_grid = np.linspace(-12, 12, 25)
    for y in (0, 1):
        a = core_mcmc._log_p_response(z_grid, y)
        b = np.array([core.log_response_prob(zi, 0.0, 0.0, y) for zi in z_grid])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# F0.1: numerical equivalence with spike-paper probit_lapse_nll
# ---------------------------------------------------------------------------

def _spike_paper_p_y1(c, t, sigma, lam=0.025):
    """Replicates train_val_split_and_fit.probit_lapse_nll's response model:
        z = (c - t) / sigma
        p = λ + (1 − 2λ) · Φ(z)
    The multi-main engine uses ℓ = log(1/σ) so exp(ℓ) = 1/σ and z = exp(ℓ)(c + θ)
    where θ corresponds to -t in the spike paper notation (sign of bias).
    """
    z = (c - t) / sigma
    return lam + (1.0 - 2.0 * lam) * float(norm.cdf(z))


def test_equivalence_with_spike_paper_response_model():
    """F0.1 regression: multi-main P(y=1) matches spike-paper Eq. 2 to 1e-12.

    Battery: a 5×5×3 grid of (sigma, theta_spike, c) crossed with both y values.
    The multi-main engine parametrizes l = log(1/sigma) and t = -theta_spike
    (different bias sign convention). With z_multi = exp(l)·(c + t_multi) and
    z_spike = (c - theta_spike)/sigma = exp(l)·(c - theta_spike), the equivalence
    is: t_multi = -theta_spike. Test by setting t_multi = -theta_spike.
    """
    sigmas = np.array([0.3, 0.5, 0.8, 1.2, 2.0])
    thetas_spike = np.array([-1.0, -0.3, 0.0, 0.4, 1.0])
    cs = np.array([-2.0, -0.5, 0.5])

    for sigma in sigmas:
        l_multi = np.log(1.0 / sigma)
        for theta_s in thetas_spike:
            t_multi = -theta_s   # bias sign convention difference
            for c in cs:
                # spike paper form
                p_spike = _spike_paper_p_y1(c, theta_s, sigma)
                # multi-main form via _p_response_yes
                z_multi = np.exp(l_multi) * (c + t_multi)
                p_multi = float(core_mcmc._p_response_yes(z_multi))
                np.testing.assert_allclose(
                    p_multi, p_spike, rtol=1e-12, atol=1e-13,
                    err_msg=f"sigma={sigma} theta_s={theta_s} c={c}: "
                            f"p_multi={p_multi} vs p_spike={p_spike}",
                )


def test_equivalence_with_spike_paper_log_likelihood():
    """F0.1 regression: multi-main log P(y) matches spike-paper log to 1e-12.

    Spike paper NLL contribution per trial: y·log(p) + (1−y)·log(1−p) with
        p = λ + (1 − 2λ)·Φ((c − θ)/σ).
    Multi-main: log P(y|z) via logsumexp on the lapse mixture.
    """
    sigmas = np.array([0.3, 0.5, 0.8, 1.2])
    thetas_spike = np.array([-0.5, 0.0, 0.5])
    cs = np.array([-1.5, 0.0, 1.5])

    for sigma in sigmas:
        l_multi = np.log(1.0 / sigma)
        for theta_s in thetas_spike:
            t_multi = -theta_s
            for c in cs:
                z_multi = np.exp(l_multi) * (c + t_multi)
                for y in (0, 1):
                    p_spike = _spike_paper_p_y1(c, theta_s, sigma)
                    expected = np.log(p_spike if y == 1 else 1.0 - p_spike)
                    got = float(core_mcmc._log_p_response(np.array(z_multi), y))
                    np.testing.assert_allclose(
                        got, expected, rtol=1e-10, atol=1e-12,
                        err_msg=f"sigma={sigma} theta_s={theta_s} c={c} y={y}: "
                                f"got={got} expected={expected}",
                    )
