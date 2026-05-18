"""T0.6 regression — Šidák correction formula.

Historical bug: `1 - alpha**(1/K)` (≈ 0.381 at α=0.05, K=6).
Correct form:    `(1 - alpha)**(1/K)` (≈ 0.9915 at α=0.05, K=6).

Pins:
  - cert_config.yaml stop_thresh_sidak = 0.9915 (within 0.001).
  - The session driver's default stop_thresh (when None and alpha=0.05, K=6)
    matches the Šidák formula (and NOT the buggy form).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import core_mcmc
import engine_mode_b  # F3.1: Mode-B relocated

# F3.3: this file now lives at tests/mode_b/, so the repo root is two
# levels up (was one level when it lived at tests/).
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def test_sidak_formula_arithmetic():
    """Pin both the correct value and the "wrong-formula bug" sentinel."""
    K = 6
    alpha = 0.05
    correct = (1.0 - alpha) ** (1.0 / K)
    buggy = 1.0 - alpha ** (1.0 / K)
    np.testing.assert_allclose(correct, 0.9915, atol=1e-3)
    # The "wrong-formula bug" gives 1 - 0.05**(1/6) = 0.393 (NOT 0.95).
    np.testing.assert_allclose(buggy, 0.393, atol=2e-3)
    assert correct > 0.95 > buggy, (
        "Šidák-correct stop_thresh must be above 0.95 and the buggy form below"
    )


def test_cert_config_has_correct_sidak_value():
    """F1.2 (2026-05-15): Šidák threshold moved to `mode_b_legacy` block when
    cert_config.yaml is in Mode-A.  Test still verifies the value is correct
    in the legacy block (Mode-B is the deprecated binary-cert path)."""
    yaml = pytest.importorskip("yaml")
    cfg_path = os.path.join(_PROJECT_ROOT, "cert_config.yaml")
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    # Mode-A configs nest Mode-B keys under `mode_b_legacy`; Mode-B configs
    # keep them at the top level.  Accept either location.
    legacy = cfg.get("mode_b_legacy", cfg)
    assert "stop_thresh_sidak" in legacy, (
        "cert_config.yaml missing stop_thresh_sidak "
        "(checked both top level and mode_b_legacy block)"
    )
    np.testing.assert_allclose(legacy["stop_thresh_sidak"], 0.9915, atol=1e-3)


def test_session_default_stop_thresh_uses_correct_sidak():
    """Run a tiny session with stop_thresh=None and inspect stop_thresh_used."""
    K = 6
    alpha = 0.05
    rng = np.random.default_rng(0)
    l_true = rng.normal(0.4, 0.3, K)
    t_true = rng.normal(0.0, 0.5, K)
    true_params = []
    for k in range(K):
        true_params.append(float(t_true[k]))
        true_params.append(float(l_true[k]))
    out = engine_mode_b.run_session_mcmc_certification(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        l_star=np.zeros(K), max_q=5, N=100, seed=0, ess_threshold_frac=0.9,
        alpha=alpha, stop_thresh=None,
    )
    expected = (1.0 - alpha) ** (1.0 / K)
    np.testing.assert_allclose(out["stop_thresh_used"], expected, rtol=1e-9)
    # And explicitly NOT the buggy value.
    buggy = 1.0 - alpha ** (1.0 / K)
    assert abs(out["stop_thresh_used"] - buggy) > 0.5
