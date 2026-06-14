"""Drift-guard for the corr_t wiring of _sbc_engine_k7.py (2026-06-11).

The K=7 SBC engine gained an env-gated prior mode (SBC_PRIOR_MODE) so the SBC
rank-uniformity check can run on the live fitted prior (corr_l shipped / corr_t
proposed) instead of only the scalar compound-symmetry default. These gates pin:

  * default (unset) mode is "cs" → _prior_sigmas() == (None, None) so
    make_state_hier / sample_prior_hier_K take the BYTE-IDENTICAL legacy
    compound-symmetry path;
  * corr_l → (Corr_l, Corr_l); corr_t → (Corr_l, Corr_t) from the same
    Sigma_l_fitted_k7.npy the live engine consumes;
  * Corr_t is a real lever (≠ Corr_l), symmetric and positive-definite.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DRIVERS = os.path.join(_REPO, "paper_sims", "drivers")
if _DRIVERS not in sys.path:
    sys.path.insert(0, _DRIVERS)

import _sbc_engine_k7 as sbc  # noqa: E402


def test_default_mode_is_compound_symmetry():
    """Unset SBC_PRIOR_MODE ⇒ cs ⇒ no fitted matrices (legacy byte-identical)."""
    assert sbc.PRIOR_MODE == "cs"
    saved = sbc.PRIOR_MODE
    try:
        sbc.PRIOR_MODE = "cs"
        assert sbc._prior_sigmas() == (None, None)
    finally:
        sbc.PRIOR_MODE = saved


def test_corr_l_and_corr_t_modes_select_fitted_matrices():
    saved = sbc.PRIOR_MODE
    try:
        sbc.PRIOR_MODE = "corr_l"
        Sl1, St1 = sbc._prior_sigmas()
        sbc.PRIOR_MODE = "corr_t"
        Sl2, St2 = sbc._prior_sigmas()
    finally:
        sbc.PRIOR_MODE = saved
    # shapes
    for M in (Sl1, St1, Sl2, St2):
        assert np.asarray(M).shape == (7, 7)
    # corr_l: Sigma_t == Sigma_l == Corr_l ; same l-block across both modes
    assert np.array_equal(Sl1, St1)
    assert np.array_equal(Sl1, Sl2)
    # corr_t: Sigma_t is the distinct fitted Corr_t
    assert not np.allclose(St2, St1)
    assert np.allclose(St2, St2.T)
    assert np.linalg.eigvalsh(np.asarray(St2, float)).min() > 0


def test_bad_mode_raises():
    saved = sbc.PRIOR_MODE
    try:
        sbc.PRIOR_MODE = "bogus"
        with pytest.raises(ValueError):
            sbc._prior_sigmas()
    finally:
        sbc.PRIOR_MODE = saved


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
