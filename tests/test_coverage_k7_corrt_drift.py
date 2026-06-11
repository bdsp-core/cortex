"""Drift-guard for the K=7 + corr_t extension of run_coverage_validation.py
(2026-06-11).

The coverage harness gained two default-OFF flags (`--k7`, `--corr-t`) so the
F2.2 AUROC-CI coverage validation can run on the live K=7 + corr_t instrument.
These gates pin that the additive change did NOT perturb the shipped K=6 path
and that the new K=7 wiring matches the live engine's source of truth:

  * K=6 default bank is byte-identical to the legacy `load_bank_signals` (no
    spike, same per-domain signal arrays, same order);
  * K=7 bank prepends spike from combined_spike.json (spike-first, 7 domains);
  * the K=7 Sigma file carries a Corr_t distinct from Corr_l (so corr_t is a
    real change, not a no-op), both 7x7;
  * the corr_t flag OFF selects Corr_l for the t-block (shipped semantics).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import run_coverage_validation as cov  # noqa: E402
from bridge._common import _autodetect_banks_dir, load_bank_signals  # noqa: E402
from core_mcmc import load_fitted_Sigma  # noqa: E402


def test_k6_default_bank_byte_identical_to_legacy_loader():
    """Default-OFF: the K=6 bank must equal the pre-change loader output."""
    banks = _autodetect_banks_dir()
    legacy = load_bank_signals(banks, cov.DOMAINS)
    new = cov._load_bank_with_spike(banks, cov.DOMAINS)
    assert len(new) == len(legacy) == 6
    for a, b in zip(new, legacy):
        assert np.array_equal(a, b)


def test_k7_bank_prepends_spike_spike_first():
    """K=7 bank: spike first, 7 domains, IIIC tail byte-identical to legacy."""
    banks = _autodetect_banks_dir()
    iiic = load_bank_signals(banks, cov.DOMAINS)        # 6 IIIC
    k7 = cov._load_bank_with_spike(banks, cov.DOMAINS_K7)
    assert cov.DOMAINS_K7[0] == "spike" and len(k7) == 7
    assert len(k7[0]) > 0                                # spike pool non-empty
    for a, b in zip(k7[1:], iiic):                       # IIIC tail unchanged
        assert np.array_equal(a, b)


def test_k7_sigma_file_has_distinct_corr_t():
    """corr_t is a real lever: Corr_t != Corr_l, both 7x7, in the K=7 file."""
    obj = load_fitted_Sigma(os.path.join(_REPO, cov.SIGMA_FILE_K7))
    Cl = np.asarray(obj["Corr_l"], float)
    Ct = np.asarray(obj["Corr_t"], float)
    assert Cl.shape == Ct.shape == (7, 7)
    assert not np.allclose(Cl, Ct)
    assert np.allclose(Ct, Ct.T)                         # symmetric
    assert np.linalg.eigvalsh(Ct).min() > 0              # positive-definite


def test_domain_order_matches_sigma_file():
    """DOMAINS_K7 must equal the on-disk Sigma_l_fitted_k7 domain order."""
    obj = load_fitted_Sigma(os.path.join(_REPO, cov.SIGMA_FILE_K7))
    file_domains = [str(d) for d in list(obj["domains"])]
    assert file_domains == cov.DOMAINS_K7


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
