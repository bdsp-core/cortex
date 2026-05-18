"""T3.6 — numerical bound that documents WHY virtual cert was removed (FIX-T1.1).

Under joint normality with posterior correlation ρ_post on (ℓ_sz, ℓ_grda) and
P(ℓ_sz > ℓ*) = 0.95 (i.e. ℓ_sz threshold = 1.645 standard deviations above
the marginal mean), the conditional bound is

    P(ℓ_grda > ℓ* | ℓ_sz > ℓ*)  ≥  Φ(ρ_post · 1.645).

This test pins:
  - For ρ_post = 0.378 (compound-symmetry r used previously): bound ≈ 0.733.
  - For ρ_post = 0.6481 (the empirical sz–grda correlation found in W1-A):
    bound ≈ 0.857.
  - Both values are STRICTLY BELOW the original 0.95 virtual-cert threshold,
    so the virtual mechanism was unjustified at that level.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


_Z_95 = float(norm.ppf(0.95))  # ≈ 1.6448536


def _bound(rho):
    return float(norm.cdf(rho * _Z_95))


def test_compound_symmetry_bound_is_073():
    rho = 0.378
    b = _bound(rho)
    np.testing.assert_allclose(b, 0.733, atol=1e-2)
    assert b < 0.95, (
        f"under compound-symmetry r=0.378 the virtual-cert bound is {b:.3f}, "
        "BELOW 0.95 — virtual cert at the 0.95 threshold is unjustified"
    )


def test_empirical_sz_grda_bound_is_086():
    rho = 0.6481  # empirical from cross_domain_rater_matrix.csv (W1-A)
    b = _bound(rho)
    np.testing.assert_allclose(b, 0.857, atol=1e-2)
    assert b < 0.95, (
        f"under empirical ρ_sz_grda=0.6481 the bound is {b:.3f}, still below 0.95; "
        "virtual cert at 0.95 threshold remains unjustified by joint normality"
    )


def test_bound_monotone_in_rho():
    """Sanity: bound must be monotone increasing in ρ."""
    rhos = np.linspace(0.0, 0.99, 20)
    bs = np.array([_bound(r) for r in rhos])
    diffs = np.diff(bs)
    assert np.all(diffs > 0), "bound must be strictly increasing in ρ"


def test_rho_required_for_095_bound():
    """What ρ would actually justify a 0.95 virtual cert?  Φ(ρ·1.645) = 0.95
    ⇒ ρ·1.645 = 1.645 ⇒ ρ = 1.0.  Document this explicitly."""
    # Find the ρ such that bound(ρ) ≥ 0.95
    # Solving: norm.cdf(ρ · z95) = 0.95 ⇒ ρ · z95 = z95 ⇒ ρ = 1.
    rho_required = 1.0
    b = _bound(rho_required)
    np.testing.assert_allclose(b, 0.95, atol=1e-9)
    # i.e., perfect posterior correlation is the ONLY way the simple bound
    # hits 0.95 — confirming virtual cert at that threshold cannot be backed
    # by any finite ρ_post.
