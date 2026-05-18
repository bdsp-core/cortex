"""Shared pytest fixtures for the certification engine test suite (W2-A).

Adds the project root to sys.path so tests can `import core_mcmc`,
`import core_mcmc_brute_k`, `import core` directly without an installed
package.  Provides synthetic-rater fixtures and a loaded `cert_config.yaml`
fixture so individual tests don't have to duplicate that boilerplate.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Engine reproducibility contract (methodology repo requirements.txt): the
# SMC/MCMC + linear-algebra paths are bit-exact ONLY with single-threaded
# BLAS. `tests/test_parallel_determinism.py` asserts serial==parallel
# BITWISE; eigvalsh/solve reduction order depends on BLAS thread count.
# These caps MUST be set BEFORE numpy is imported, so they live here at the
# top of conftest (pytest imports conftest before any test module). Runtime
# entrypoints document the same requirement in requirements.txt.
# ---------------------------------------------------------------------------
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
           "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402  (must follow the BLAS caps above)
import pytest  # noqa: E402

# ---------------------------------------------------------------------------
# Unified-repo layout (Phase 2): the validated engine keeps its flat
# bare-name imports (`import core_mcmc`, `from core import ...`,
# `from core_K import ...`) so MINE's 59-test suite runs VERBATIM — no
# import rewrites to the validated numerics. The flat modules now live in
# engine/ (core engine) and engine/variants/ (ported PI variants); both
# dirs go on sys.path. _PROJECT_ROOT stays the repo root so cert_config.yaml
# and Phase-0/1 path-based tests resolve unchanged.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _p in (_PROJECT_ROOT,
           os.path.join(_PROJECT_ROOT, "engine"),
           os.path.join(_PROJECT_ROOT, "engine", "variants"),
           os.path.join(_PROJECT_ROOT, "bridge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# F0.1 (2026-05-15): register custom pytest markers used in this suite.
# Run fast tests only:    pytest -m "not slow and not nightly"
# Run paper-grade tests:  pytest -m "slow"
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that take >1 min (SBC, coverage, OC pilots); excluded by default"
    )
    config.addinivalue_line(
        "markers",
        "nightly: long-running tests (FWER 1000+ sessions, gold-chain references)"
    )


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Per-test scratch directory (alias of pytest's tmp_path)."""
    return tmp_path


@pytest.fixture
def synthetic_true_params():
    """Factory: build a clearly-pass synthetic rater.

    Usage:
        def test_x(synthetic_true_params):
            tp = synthetic_true_params(K=3, seed=0)
            l_true = tp["l_true"]; t_true = tp["t_true"]

    `l_true` is drawn N(0.4, 0.3**2) so the rater clearly passes l* = 0;
    `t_true` is drawn N(0, 0.5**2).
    """
    def _make(K: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        return {
            "l_true": rng.normal(0.4, 0.3, size=K),
            "t_true": rng.normal(0.0, 0.5, size=K),
        }
    return _make


@pytest.fixture
def cert_config():
    """Load cert_config.yaml as a plain dict.  Skips if PyYAML missing."""
    yaml = pytest.importorskip("yaml")
    cfg_path = os.path.join(_PROJECT_ROOT, "cert_config.yaml")
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)


def true_params_array(t_true: np.ndarray, l_true: np.ndarray) -> list:
    """Pack (t_0, l_0, t_1, l_1, ...) order required by the session drivers."""
    K = len(t_true)
    out = []
    for k in range(K):
        out.append(float(t_true[k]))
        out.append(float(l_true[k]))
    return out
