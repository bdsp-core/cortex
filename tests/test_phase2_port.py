"""Phase 2: PI-variant likelihood hardening — numeric-equivalence tests.

Verifies the core_K hardening (the only genuinely unhardened variant code)
matches the canonical hardened likelihood and no longer collapses in the
tails, and that the ported variants import against the hardened core.
"""
import numpy as np
import pytest
from scipy.stats import norm

import core  # hardened canonical core (engine/)
import core_mcmc
import core_K  # ported variant (engine/variants/)


def test_core_k_uses_single_sourced_lapse_constant():
    assert core_K.LAPSE_RATE is core.LAPSE_RATE == 0.025


def test_core_k_source_has_no_unhardened_cdf_clip():
    import inspect
    src = inspect.getsource(core_K)
    assert "np.clip(norm.cdf(z), 1e-9, 1.0 - 1e-9)" not in src, (
        "residual unhardened tail-collapsing likelihood in core_K")
    assert "LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z)" in src


@pytest.mark.parametrize("z", [-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 50.0])
def test_hardened_p_matches_canonical_and_is_bounded(z):
    zc = float(z)
    p_hardened = core.LAPSE_RATE + (1.0 - 2.0 * core.LAPSE_RATE) * norm.cdf(zc)
    # exactly the canonical item-selection probability
    assert p_hardened == pytest.approx(
        float(core_mcmc._p_response_yes(np.array([zc]))[0]), abs=0.0)
    # bounded strictly inside (0,1) by construction — no clip needed
    assert core.LAPSE_RATE <= p_hardened <= 1.0 - core.LAPSE_RATE


def test_old_clip_form_collapses_in_tail_new_does_not():
    """Demonstrates the bug the hardening fixes: np.clip(norm.cdf)..."""
    z = 8.0
    old = float(np.clip(norm.cdf(z), 1e-9, 1.0 - 1e-9))
    new = core.LAPSE_RATE + (1.0 - 2.0 * core.LAPSE_RATE) * norm.cdf(z)
    assert old == 1.0 - 1e-9            # collapses to the clip ceiling
    assert new < 1.0 - core.LAPSE_RATE + 1e-12
    assert abs(new - old) > 1e-3        # materially different in the tail


def test_core_k_expected_loss_runs_and_is_finite():
    """Functional smoke: the hardened expected-loss objective executes."""
    rng = np.random.default_rng(0)
    N, K = 64, 2
    particles = {
        "t": rng.normal(0, 0.5, size=(N, K)),
        "l": rng.normal(0.3, 0.3, size=(N, K)),
        "w": np.full(N, 1.0 / N),
    }
    signals = np.linspace(-3, 3, 11)
    out = core_K.expected_loss_hier_vec_K(particles, 0, signals)
    assert out.shape == signals.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("mod", [
    "core_learnr", "core_learnr_K", "core_mcmc_learnr",
    "core_mcmc_betaprior", "core_auroc_betaprior", "core_auroc",
])
def test_ported_variants_import_against_hardened_core(mod):
    import importlib
    m = importlib.import_module(mod)
    assert m is not None
