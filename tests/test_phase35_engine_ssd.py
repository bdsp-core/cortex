"""Phase 3.5 — engine s_sd propagation: parity gate + closed-form correctness.

The engine now marginalizes the probit-lapse likelihood over the segment
signal posterior s ~ N(s, s_sd²) in closed form
(z attenuated by 1/√(1+(e^ℓ·s_sd)²)).  These tests assert:

  (1) PARITY@s_sd→0 is BIT-EXACT — s_sd=0.0 (the default) reproduces the
      pre-Phase-3.5 engine to the last bit, so the entire validated suite
      is unchanged by construction;
  (2) the closed-form marginalization equals a Monte-Carlo marginal
      (validates the Gaussian–probit convolution identity is correct);
  (3) attenuation is monotone — more s_sd ⇒ p shrinks toward the lapse
      midpoint and per-item Fisher information decreases (the honest
      power-deflation that is the whole point of Phase 3.5).
"""
import numpy as np
import pytest
from scipy.special import log_ndtr

import core
import core_mcmc

LAM = core.LAPSE_RATE


def _baseline_p(s, t, l):
    """Pre-Phase-3.5 formula, written out independently."""
    z = np.exp(l) * (s + t)
    return LAM + (1.0 - 2.0 * LAM) * np.exp(log_ndtr(z))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parity_response_prob_bit_exact_at_ssd0(seed):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1.5, 500)
    t = rng.normal(0, 0.6, 500)
    l = rng.normal(0.3, 0.4, 500)
    base = _baseline_p(s, t, l)
    # default arg AND explicit 0.0 must both be bit-identical to baseline
    np.testing.assert_array_equal(core.response_prob(s, t, l), base)
    np.testing.assert_array_equal(
        core.response_prob(s, t, l, s_sd=0.0), base)
    # _marg_z at s_sd=0 is exactly e^l (s+t)
    np.testing.assert_array_equal(
        core._marg_z(l, s, t, 0.0), np.exp(l) * (s + t))


@pytest.mark.parametrize("y", [0, 1])
def test_parity_log_response_prob_bit_exact(y):
    rng = np.random.default_rng(7)
    s, t, l = (rng.normal(0, 1.2, 300), rng.normal(0, .5, 300),
               rng.normal(.2, .3, 300))
    a = core.log_response_prob(s, t, l, y)
    b = core.log_response_prob(s, t, l, y, s_sd=0.0)
    np.testing.assert_array_equal(a, b)


def test_parity_expected_loss_hier_vec_default():
    rng = np.random.default_rng(3)
    N, K = 256, 3
    part = {"t": rng.normal(0, .5, (N, K)), "l": rng.normal(.3, .4, (N, K)),
            "w": np.full(N, 1.0 / N)}
    sig = np.linspace(-3, 3, 11)
    np.testing.assert_array_equal(
        core.expected_loss_hier_vec(part, 1, sig),
        core.expected_loss_hier_vec(part, 1, sig, signal_sds=None))


def test_parity_update_bit_exact_and_history_zeros():
    """update() with default s_sd reproduces the prior reweight exactly
    and records an all-zero parallel history_s_sd."""
    rng = np.random.default_rng(0)
    st = core_mcmc.make_state_hier(400, 2, 0.3, rng)
    st2 = {k: (v.copy() if isinstance(v, np.ndarray) else
               (list(v) if isinstance(v, list) else v))
           for k, v in st.items()}
    core_mcmc.update(st, 0, 0.7, 1)              # default s_sd
    core_mcmc.update(st2, 0, 0.7, 1, s_sd=0.0)   # explicit 0
    np.testing.assert_array_equal(st["w"], st2["w"])
    np.testing.assert_array_equal(st["log_lik"], st2["log_lik"])
    assert st["history_s_sd"] == [0.0]


def test_closed_form_marginal_matches_monte_carlo():
    """E_{s~N(s0,sd²)}[Φ-lapse] ≈ closed-form attenuated p (validates the
    Gaussian–probit convolution identity)."""
    rng = np.random.default_rng(42)
    s0, t, l, sd = 0.4, 0.1, np.float64(0.25), 0.8
    closed = float(core.response_prob(s0, t, l, s_sd=sd))
    draws = rng.normal(s0, sd, 400_000)
    mc = float(np.mean(core.response_prob(draws, t, l)))   # s_sd=0 per draw
    assert abs(closed - mc) < 3e-3, (closed, mc)


def test_attenuation_monotone_and_information_deflates():
    """Larger s_sd pulls p toward the lapse midpoint and lowers the
    item's discriminating slope — the intended honest power deflation."""
    l, t = 0.5, 0.0
    s_grid = np.linspace(-2, 2, 9)
    p0 = core.response_prob(s_grid, t, l, s_sd=0.0)
    p1 = core.response_prob(s_grid, t, l, s_sd=0.5)
    p2 = core.response_prob(s_grid, t, l, s_sd=1.5)
    mid = 0.5  # lapse-symmetric midpoint
    # every point: more s_sd ⇒ closer to 0.5 (never further)
    assert np.all(np.abs(p2 - mid) <= np.abs(p1 - mid) + 1e-12)
    assert np.all(np.abs(p1 - mid) <= np.abs(p0 - mid) + 1e-12)
    # slope (discrimination) strictly deflates
    assert (np.ptp(p2) < np.ptp(p1) < np.ptp(p0))
