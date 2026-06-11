"""Guard the v15 credentialed cut-scores + the staged (non-live) wiring (2026-06-11).

  * cert_config carries an `ell_star_unified_v15` block with the credentialed
    panel cut-scores (spike=Super8; IIIC=Super8∪Bonobo; lpd robust-trimmed);
  * the LIVE default block is still `ell_star_unified_v14` (frozen instrument —
    v15 is staged, opt-in only);
  * the codified lpd robust-trim regenerates ell*=0.3059 (drop credentialed
    experts with lpd ell<=0.19), proving the once-hand-assembled value reproduces.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _p in (os.path.join(_REPO, "scripts"),
           os.path.join(_REPO, "pipeline", "reference_calibration")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cortex_policy_k7 import load_ell_star_k7  # noqa: E402

CODES = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]

# Pinned credentialed v15 cut-scores (build_ell_star_v15 / panel_comparison).
V15 = {"spike": 0.2383, "sz": 0.1548, "lpd": 0.3059, "gpd": 0.2570,
       "lrda": 0.3214, "grda": 0.3540, "iic": 0.3042}
# Pinned shipped v14 cut-scores (the live frozen instrument).
V14 = {"spike": 0.3251, "sz": 0.2560, "lpd": 0.5337, "gpd": 0.3297,
       "lrda": 0.4793, "grda": 0.4865, "iic": 0.4418}


def test_v15_block_values():
    got = load_ell_star_k7(CODES, block_name="ell_star_unified_v15")
    for c, v in zip(CODES, got):
        assert abs(v - V15[c]) < 1e-3, f"{c}: {v} != {V15[c]}"


def test_live_default_is_still_v14_frozen():
    """The default (no block_name) must remain v14 — the live instrument is frozen."""
    got = load_ell_star_k7(CODES)
    for c, v in zip(CODES, got):
        assert abs(v - V14[c]) < 1e-3, f"default drifted off v14 for {c}: {v}"
    # and v15 is genuinely different (lower cuts)
    v15 = load_ell_star_k7(CODES, block_name="ell_star_unified_v15")
    assert all(b < a for a, b in zip(got, v15)), "v15 cuts should all be < v14"


def test_lpd_robust_trim_reproduces_0_3059():
    from build_ell_star_v15 import _panel_calibration, LPD_TRIM_ELL, _credentialed
    from run_youden_calibration_k7 import _load_sdt_fits
    s8b = _credentialed("Super8") | _credentialed("Bonobo")
    fits = _load_sdt_fits("lpd")
    untrimmed = _panel_calibration("lpd", s8b, fits, lpd_trim=False)["l_star"]
    trimmed = _panel_calibration("lpd", s8b, fits, lpd_trim=True)["l_star"]
    assert abs(untrimmed - 0.186067) < 1e-4          # the unstable raw value
    assert abs(trimmed - 0.3059) < 1e-3              # PI-confirmed robust value
    assert LPD_TRIM_ELL == 0.19


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
