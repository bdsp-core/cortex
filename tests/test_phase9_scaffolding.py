"""Phase 9 Layer 2/3/5/6a scaffolding — schema + import + helper-function tests.

These tests verify the K=7 scaffolding scripts at the structure level WITHOUT
requiring the heavy NUTS compute to complete (which can take hours). They run
in seconds and gate-test:

  * Module imports and basic constants
  * K=7 task ordering invariants (spike first, then 6 IIIC)
  * Mapping consistency between cortex_policy_k7 _KEY_FOR_CODE and v13 cert_config keys
  * Helper functions (Sigma_l pairwise covariance, Youden, Cohen-d) on synthetic data
  * cert_config v13 has all 7 ell_star entries
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# ── Module imports ────────────────────────────────────────────────────────

def test_layer2_assemble_outputs_k7_imports():
    from pipeline.joint_calibration import assemble_outputs_k7 as a
    assert a.TASKS == ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    assert len(a.TASK_LABEL_FILTER) == 7
    assert a.SPIKE_SOURCES == {"sn1_combined_v2"}
    # Disjoint
    assert a.IIIC_SOURCES.isdisjoint(a.SPIKE_SOURCES)


def test_layer3_build_engine_inputs_k7_imports():
    import build_engine_inputs_k7 as b
    assert b.K7_DOMAINS == ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]


def test_layer5_build_sigma_l_k7_imports():
    import build_sigma_l_k7 as b
    assert b.K7_DOMAINS == ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]


def test_layer6a_cortex_engine_inputs_k7_imports():
    import cortex_engine_inputs_k7 as c
    assert c.K == 7
    assert c.TASK_CODES == ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    assert c.SPIKE_TASK_INDEX == 0
    assert c.IIIC_TASK_INDICES == [1, 2, 3, 4, 5, 6]
    # spike is the first task family; rest are IIIC
    assert c.TASK_FAMILIES == ["spike"] + ["iiic"] * 6


def test_layer6a_cortex_policy_k7_imports():
    import cortex_policy_k7 as p
    assert set(p._KEY_FOR_CODE_K7) == {
        "spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"}
    assert p._KEY_FOR_CODE_K7["spike"] == "combined_spike"
    for c in ("sz", "lpd", "gpd", "lrda", "grda", "iic"):
        assert p._KEY_FOR_CODE_K7[c] == f"sparcnet_{c}"


# ── cert_config v13 has all 7 ell_star entries ────────────────────────────

def test_v13_cert_config_has_all_7_ell_star():
    import yaml
    cfg_path = REPO / "calibration" / "cert_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    v13 = cfg["ell_star_unified_v13"]["tasks"]
    expected_keys = {"combined_spike", "sparcnet_sz", "sparcnet_lpd",
                     "sparcnet_gpd", "sparcnet_lrda", "sparcnet_grda",
                     "sparcnet_iic"}
    assert set(v13.keys()) >= expected_keys
    # All ell_star values are finite
    for k in expected_keys:
        ell = v13[k]["ell_star"]
        assert np.isfinite(ell), f"v13 {k} ell_star is not finite: {ell}"


def test_cortex_policy_k7_load_ell_star_v14_is_default():
    """K=7 ell_star load defaults to v14 (Phase-9 ship state)."""
    import cortex_policy_k7 as p
    codes = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    ell_v14 = p.load_ell_star_k7(codes)
    assert len(ell_v14) == 7
    # v14: 5 of 6 IIIC tasks reproduce v13 ell* exactly (lpd, gpd, lrda,
    # grda, iic — panel byte-stable under cross-task expansion); sz and
    # spike differ due to spike's inclusion in cross-task expert score +
    # spike methodology change from 70/30 TRAIN to uniform CV-top-14.
    assert abs(ell_v14[2] - 0.5337430687087749) < 1e-9, "lpd v14 != v13"
    assert abs(ell_v14[3] - 0.3297002787536504) < 1e-9, "gpd v14 != v13"
    assert abs(ell_v14[4] - 0.4792595265871899) < 1e-9, "lrda v14 != v13"
    assert abs(ell_v14[5] - 0.48645101728803) < 1e-9, "grda v14 != v13"
    assert abs(ell_v14[6] - 0.4418131962513707) < 1e-9, "iic v14 != v13"


def test_cortex_policy_k7_load_ell_star_v13_explicit():
    """K=7 loader can still read v13 (legacy K=6+spike-special)."""
    import cortex_policy_k7 as p
    codes = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    ell_v13 = p.load_ell_star_k7(codes, block_name="ell_star_unified_v13")
    assert len(ell_v13) == 7
    # v13 spike (combined_spike) is 0.2543
    assert abs(ell_v13[0] - 0.2542662861051924) < 1e-9
    # v13 sz is 0.4550
    assert abs(ell_v13[1] - 0.45504292981835925) < 1e-9


# ── Helper functions (synthetic data) ─────────────────────────────────────

def test_build_sigma_l_k7_pairwise_cov_known():
    """Synthetic 2-task pairwise covariance: known answer."""
    import build_sigma_l_k7 as b
    values = {
        "spike": {1: 0.0, 2: 1.0, 3: 2.0},      # mean 1, var 1
        "sz": {2: 1.0, 3: 2.0, 4: 3.0},         # mean 2, var 1
    }
    Sigma, n_overlap, meta = b._pairwise_cov(values, ["spike", "sz"])
    # n_overlap[spike, spike] = 3; [sz, sz] = 3; [spike, sz] = 2 (overlap: rids 2, 3)
    assert n_overlap[0, 0] == 3
    assert n_overlap[1, 1] == 3
    assert n_overlap[0, 1] == 2 and n_overlap[1, 0] == 2
    # diagonal variance ≈ 1.0 (sample variance ddof=1)
    assert abs(Sigma[0, 0] - 1.0) < 1e-9
    assert abs(Sigma[1, 1] - 1.0) < 1e-9
    # off-diagonal: cov([1, 2], [1, 2]) = 0.5 (ddof=1)
    assert abs(Sigma[0, 1] - 0.5) < 1e-9
    assert Sigma[0, 1] == Sigma[1, 0]


def test_build_sigma_l_k7_pairwise_cov_insufficient_overlap():
    """If <2 raters in overlap, returns NaN for that cell."""
    import build_sigma_l_k7 as b
    values = {
        "spike": {1: 0.0, 2: 1.0},
        "sz": {3: 1.0, 4: 2.0},          # zero overlap with spike
    }
    Sigma, n_overlap, _meta = b._pairwise_cov(values, ["spike", "sz"])
    assert n_overlap[0, 1] == 0
    assert np.isnan(Sigma[0, 1])


def test_build_sigma_l_k7_to_corr():
    """Σ → correlation; diagonal becomes 1.0."""
    import build_sigma_l_k7 as b
    Sigma = np.array([[2.0, 0.6], [0.6, 0.5]])
    Corr = b._to_corr(Sigma)
    assert abs(Corr[0, 0] - 1.0) < 1e-9
    assert abs(Corr[1, 1] - 1.0) < 1e-9
    # Corr(0, 1) = 0.6 / sqrt(2.0 * 0.5) = 0.6 / 1.0 = 0.6
    assert abs(Corr[0, 1] - 0.6) < 1e-9


# ── TASKS table consistency across all K=7 modules ───────────────────────

def test_k7_task_ordering_consistent_across_modules():
    """All K=7 modules must agree on task order: spike, sz, lpd, gpd, lrda, grda, iic."""
    expected = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    from pipeline.joint_calibration import (
        prep_k7, assemble_outputs_k7, baseline_k6, variant_selection_k7)
    import build_engine_inputs_k7, build_sigma_l_k7
    import cortex_engine_inputs_k7, cortex_policy_k7
    assert prep_k7.TASKS == expected
    assert assemble_outputs_k7.TASKS == expected
    assert build_engine_inputs_k7.K7_DOMAINS == expected
    assert build_sigma_l_k7.K7_DOMAINS == expected
    assert cortex_engine_inputs_k7.TASK_CODES == expected
    assert list(cortex_policy_k7._KEY_FOR_CODE_K7) == expected
    # baseline_k6 + variant_selection use IIIC-only and ALL_TASKS respectively
    assert baseline_k6.K6_IIIC_TASKS == expected[1:]
    assert variant_selection_k7.ALL_TASKS == expected
