"""Phase 7 sub-step 4 drift-guard tests.

Pins the small surface that sub-7.4 edits — does NOT re-run the
expensive figure regeneration. Two halves:

  sub-7.4-A (Paper-1 main figure)
    * data/curated_banks/ vendored in-repo (D9 self-contained), 7
      JSONs, md5 matches docs/_manifests/reference_consulted.md5.
    * bridge._common._autodetect_banks_dir prefers the in-repo path
      over the sibling-repo fallback.
    * scripts/run_phase1_experiments_v2._KS_B includes K=7 (production
      deployment dim added to the synthetic K-scaling experiment).
    * _MAX_Q_BY_METHOD_K has a K=7 cap for every method, monotone
      cap(6) < cap(7) < cap(8) per method (matches the precedent set
      in scripts/run_tier2_oc_simstudy.py).

  sub-7.4-B (Tier-2 OC figures)
    * scripts/run_phase4_figures.ROWS_PATH defaults to the Tier-2
      output (results/phase2_validation/tier2_oc_simstudy_rows.json),
      matching scripts/run_tier2_oc_simstudy.py:33.
    * _subplot_grid handles 1..7 K-cells (1×1 .. 3×3) — the K=7-only
      pilot writes a single panel, the full grid stays 2×2/2×3.
"""
from __future__ import annotations

import hashlib
import os

import pytest


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_BANK_DIR = os.path.join(_REPO, "data", "curated_banks")
_BANK_FILES = (
    "combined_spike.json",
    "sparcnet_gpd.json",
    "sparcnet_grda.json",
    "sparcnet_iic.json",
    "sparcnet_lpd.json",
    "sparcnet_lrda.json",
    "sparcnet_sz.json",
)


# ── sub-7.4-A: bank vendor + bridge precedence + ExpB K=7 ─────────────
def test_curated_banks_vendored_in_repo():
    """All 7 SPARCNET bank JSONs must live in data/curated_banks/."""
    for fn in _BANK_FILES:
        path = os.path.join(_BANK_DIR, fn)
        assert os.path.isfile(path), f"missing in-repo bank: {path}"


def test_curated_banks_md5_matches_manifest():
    """In-repo banks must byte-match docs/_manifests/reference_consulted.md5."""
    manifest = os.path.join(_REPO, "docs", "_manifests",
                            "reference_consulted.md5")
    expected = {}
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line or "curated_banks" not in line:
                continue
            md5, path = line.split(None, 1)
            expected[os.path.basename(path)] = md5
    assert set(expected.keys()) == set(_BANK_FILES), (
        f"manifest curated_banks set mismatches expected: "
        f"manifest={sorted(expected.keys())} expected={sorted(_BANK_FILES)}"
    )
    for fn, want in expected.items():
        with open(os.path.join(_BANK_DIR, fn), "rb") as f:
            got = hashlib.md5(f.read()).hexdigest()
        assert got == want, f"{fn} md5 drift: got={got} want={want}"


def test_autodetect_banks_dir_prefers_in_repo():
    """bridge._common._autodetect_banks_dir() must return the in-repo path
    when both in-repo and sibling-repo banks exist."""
    import sys
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from bridge._common import _autodetect_banks_dir
    got = _autodetect_banks_dir()
    assert os.path.abspath(got) == _BANK_DIR, (
        f"_autodetect_banks_dir resolved to {got}, expected {_BANK_DIR}"
    )


def test_run_phase1_v2_ks_b_includes_k7():
    """The synthetic K-scaling experiment must cover the production K=7."""
    import sys
    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_phase1_experiments_v2 as m
    assert 7 in m._KS_B, f"_KS_B={m._KS_B} missing K=7"


def test_run_phase1_v2_max_q_caps_monotone():
    """Every method must have a K=7 cap with cap(6) < cap(7) < cap(8)."""
    import sys
    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_phase1_experiments_v2 as m
    for method, caps in m._MAX_Q_BY_METHOD_K.items():
        assert 7 in caps, f"{method!r} missing K=7 cap"
        assert caps[6] < caps[7] < caps[8], (
            f"{method!r} caps non-monotone: "
            f"K=6={caps[6]} K=7={caps[7]} K=8={caps[8]}"
        )


# ── sub-7.4-B: run_phase4_figures path + subplot grid ─────────────────
def test_run_phase4_figures_rows_path_points_at_tier2_output():
    """ROWS_PATH must default to the Tier-2 OC simstudy output."""
    import sys
    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_phase4_figures as m
    expected = os.path.join(_REPO, "results", "phase2_validation",
                            "tier2_oc_simstudy_rows.json")
    assert os.path.abspath(m.ROWS_PATH) == expected, (
        f"ROWS_PATH={m.ROWS_PATH} expected={expected}"
    )


def test_run_phase4_figures_subplot_grid_handles_1_to_7():
    """_subplot_grid must produce a valid (nrow, ncol, figsize) for n=1..7."""
    import sys
    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_phase4_figures import _subplot_grid
    for n in range(1, 8):
        nrow, ncol, figsize = _subplot_grid(n)
        assert nrow >= 1 and ncol >= 1
        assert nrow * ncol >= n, (
            f"_subplot_grid({n}) = ({nrow},{ncol}) cannot hold {n} panels"
        )
        assert isinstance(figsize, tuple) and len(figsize) == 2
