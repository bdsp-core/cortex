"""T0.7 regression — numerical correctness of `scipy.special.log_ndtr`.

The historical bug pattern was `np.log(np.clip(norm.cdf(z), 1e-9, 1-1e-9))`
which (a) underflows to log(1) = 0 in float64 for z >> 6 because
1 - Phi(z) cancels into round-off, and (b) saturates at log(1e-9) ~= -20.7
for z << -6 instead of the correct ~-z**2/2 asymptote.

This test pins both behaviors so a future "let's go back to norm.cdf" cleanup
will fail loudly.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm


def test_log_ndtr_matches_log_cdf_in_safe_range():
    """For |z| <= 5, log_ndtr(z) and log(norm.cdf(z)) must agree to ~1e-12."""
    z = np.linspace(-5.0, 5.0, 501)
    expected = np.log(norm.cdf(z))
    actual = log_ndtr(z)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_log_ndtr_finite_in_extreme_tails():
    """For z in [-10, 10], log_ndtr stays finite and monotone increasing."""
    z = np.linspace(-10.0, 10.0, 401)
    a = log_ndtr(z)
    assert np.all(np.isfinite(a)), "log_ndtr produced non-finite values in [-10, 10]"
    # monotone non-decreasing in z
    assert np.all(np.diff(a) >= -1e-12), "log_ndtr should be monotone non-decreasing"


def test_log_ndtr_upper_tail_pins_value():
    """log_ndtr(10) ≈ -7.62e-24 (correct).  np.log(norm.cdf(10)) saturates at 0."""
    actual = float(log_ndtr(10.0))
    # Reference: log(1 - 7.620e-24) ≈ -7.620e-24
    np.testing.assert_allclose(actual, -7.62e-24, rtol=5e-2, atol=1e-26)
    # Document the BAD behavior of the legacy pattern so it's explicit:
    legacy = float(np.log(norm.cdf(10.0)))
    assert legacy == 0.0, (
        f"log(norm.cdf(10)) used to underflow to 0.0; got {legacy}.  "
        "If this changes (e.g. scipy improves norm.cdf precision) the regression "
        "rationale changes — review FIX-T0.7 and update this test."
    )


def test_log_ndtr_lower_tail_pins_value():
    """log_ndtr(-10) ≈ -52.65 (correct ~ -z**2/2).  Old clip would saturate at log(1e-9) = -20.72."""
    actual = float(log_ndtr(-10.0))
    # Asymptote: log Phi(-z) ≈ -z**2/2 - log(z * sqrt(2*pi))
    expected = -10.0 ** 2 / 2.0 - np.log(10.0 * np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(actual, expected, rtol=5e-2)
    # The legacy pattern would have floored this at log(1e-9) = -20.72 → not -52.65.
    legacy_floor = float(np.log(1e-9))
    assert actual < legacy_floor - 5.0, (
        f"log_ndtr(-10)={actual} should be far below the legacy clip floor {legacy_floor}; "
        "if not, the asymptotic improvement T0.7 buys is gone."
    )
