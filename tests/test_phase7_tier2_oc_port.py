"""Phase 7 sub-step 2 — Tier-2 OC simstudy port-fidelity tests.

The `scripts/run_tier2_oc_simstudy.py` port is faithful to the
methodology reference (`run_phase4_simstudy.py` in the sibling
methodology repo) with exactly two documented path-only edits + one
K-grid extension (K=7 added). This file gates those invariants:

  * Reference's design constants are preserved (L_GRID, SIGMA_L_CONDS,
    METHODS, DELTAS, N_PARTICLES, TIGHTEST_DELTA, MAX_Q caps for the
    methodology-reference K ∈ {2,4,6,8}).
  * K_GRID extends the reference's {2,4,6,8} with K=7 (= the
    production deployment dim).
  * Each method has a MAX_Q cap for every K in K_GRID (no missing
    cells).
  * `_sigma_l` returns well-shaped K×K matrices at K=7 for all 3
    Σ_l conditions (no K=6-specific branch leaks).
  * The task graph builds at K=7 (--n-reps 1 produces 30 tasks: 6 ℓ
    × 1 K × 5 methods × seeds — random+brute uncrossed with Σ_l,
    hier crossed with 3 Σ_l conds).
  * Output directory is `results/phase2_validation/` (the unified
    plan's Phase-7 sub-2 location).

These are fast unit tests (no engine session runs); the full --n-reps
1 K=7 pilot is a separate execution deliverable (~minutes).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_tier2_oc_simstudy.py"


@pytest.fixture(scope="module")
def runtier2():
    """Load the port as a module (avoids side-effects on package
    imports; the script auto-configures BLAS at import time)."""
    # Ensure unified engine/ + scripts/ on path before loading (the
    # script itself does this, but `spec_from_file_location` evaluates
    # the script's preamble — keep the parent's env consistent).
    for p in (str(REPO / "scripts"), str(REPO / "engine"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("tier2_oc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_design_constants_preserved_from_methodology_reference(runtier2):
    """The reference design surface (L, Σ_l conds, methods, deltas,
    N_PARTICLES, tightest δ) is preserved byte-for-byte."""
    assert runtier2.L_GRID == [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    assert runtier2.SIGMA_L_CONDS == ["empirical", "independent", "cs0.7"]
    assert runtier2.METHODS == ["random", "brute", "hier"]
    assert runtier2.DELTAS == [0.025, 0.05, 0.10]
    assert runtier2.N_PARTICLES == 1000
    assert runtier2.TIGHTEST_DELTA == 0.025


def test_methodology_reference_max_q_caps_preserved(runtier2):
    """The reference's K ∈ {2,4,6,8} cap entries are preserved (no
    accidental drift during the port)."""
    ref = {
        "hier":   {2: 1500, 4: 2500, 6: 4000, 8: 6000},
        "brute":  {2: 1500, 4: 2500, 6: 4000, 8: 6000},
        "random": {2: 8000, 4: 14000, 6: 20000, 8: 28000},
    }
    for m, caps in ref.items():
        for K, v in caps.items():
            assert runtier2.MAX_Q_BY_METHOD_K[m][K] == v, (
                f"port drift: MAX_Q_BY_METHOD_K[{m!r}][{K}] = "
                f"{runtier2.MAX_Q_BY_METHOD_K[m][K]} (methodology "
                f"reference: {v})")


def test_k7_added_to_k_grid(runtier2):
    """K=7 added as the production deployment dim (Phase 7 sub-2)."""
    assert 7 in runtier2.K_GRID
    # K_GRID is the reference's plus K=7
    assert set(runtier2.K_GRID) == {2, 4, 6, 7, 8}


def test_every_k_has_a_max_q_cap_for_every_method(runtier2):
    """No missing cells — every (method, K) pair has a cap."""
    for m in runtier2.METHODS:
        caps = runtier2.MAX_Q_BY_METHOD_K[m]
        for K in runtier2.K_GRID:
            assert K in caps, (
                f"missing cell: MAX_Q_BY_METHOD_K[{m!r}][{K}]")
    # K=7 caps interpolated between K=6 and K=8 (sanity: 6 < 7 < 8 cap)
    for m in ("hier", "brute", "random"):
        c6 = runtier2.MAX_Q_BY_METHOD_K[m][6]
        c7 = runtier2.MAX_Q_BY_METHOD_K[m][7]
        c8 = runtier2.MAX_Q_BY_METHOD_K[m][8]
        assert c6 < c7 < c8, (
            f"{m} caps non-monotone: 6={c6}, 7={c7}, 8={c8}")


def test_sigma_l_k7_well_shaped(runtier2):
    """_sigma_l produces well-shaped, finite, symmetric K=7 matrices
    for all 3 Σ_l conditions (no K=6-specific branch leaks)."""
    for cond in ("empirical", "independent", "cs0.7"):
        S = runtier2._sigma_l(cond, 7)
        assert S.shape == (7, 7), f"{cond}: shape {S.shape}"
        assert np.isfinite(S).all(), f"{cond}: non-finite entries"
        assert np.allclose(S, S.T), f"{cond}: not symmetric"
        # All cells PSD (no eigenvalue ≤ 0)
        evals = np.linalg.eigvalsh(S)
        assert (evals > -1e-12).all(), (
            f"{cond}: not PSD (min eigenvalue {evals.min():.2e})")
    # independent ≡ identity at K=7
    np.testing.assert_array_equal(
        runtier2._sigma_l("independent", 7), np.eye(7))
    # cs0.7: off-diag exactly 0.7
    cs07 = runtier2._sigma_l("cs0.7", 7)
    assert cs07[0, 1] == 0.7
    # empirical at K≠6: matched-mean compound-symmetry (mean off-diag
    # of the K=6 reference Corr_l)
    emp7 = runtier2._sigma_l("empirical", 7)
    mean_off = float((emp7.sum() - 7) / (7 * 7 - 7))
    # mean_off ≈ 0.2504 from the fitted Corr_l (K=6 era).
    assert 0.10 < mean_off < 0.45, (
        f"empirical-at-K=7 matched-mean off-diag {mean_off:.3f} "
        f"unexpectedly far from the methodology reference ~0.25")


def test_k7_pilot_task_graph_builds(runtier2):
    """`build_tasks(n_reps=1, k_grid=[7])` produces the expected 30
    sessions: 6 ℓ × 1 K × 5 methods (random + brute + 3 hier Σ_l-conds)."""
    tasks = runtier2.build_tasks(n_reps=1, seed_base=1000, k_grid=[7])
    assert len(tasks) == 30, (
        f"K=7 --n-reps 1 produced {len(tasks)} tasks (expected 30)")
    # exactly 6 random, 6 brute, 18 hier (6 ℓ × 3 conds)
    by_method = {}
    for t in tasks:
        by_method.setdefault(t["method"], []).append(t)
    assert len(by_method["random"]) == 6
    assert len(by_method["brute"]) == 6
    assert len(by_method["hier"]) == 18
    # hier crossed across all 3 conds
    hier_conds = sorted({t["sigma_l_cond"] for t in by_method["hier"]})
    assert hier_conds == ["cs0.7", "empirical", "independent"]
    # random/brute uncrossed
    assert all(t["sigma_l_cond"] == "na"
               for t in by_method["random"] + by_method["brute"])


def test_output_dir_is_phase2_validation(runtier2):
    """Plan §"Phase 7" sub-2 places OC tables under
    results/phase2_validation/ (the methodology reference wrote
    `results/phase4_simstudy/`)."""
    assert runtier2.OUT_DIR.endswith(os.path.join(
        "results", "phase2_validation"))
