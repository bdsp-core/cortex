"""Phase 9 Layer 1 — variant_selection_k7 gate-logic unit tests.

Tests the gate helpers (Youden J, Cohen-d, s_j drift) without requiring
actual model posteriors. Drift-guard discipline.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.joint_calibration import variant_selection_k7, baseline_k6


# ── _youden_j ─────────────────────────────────────────────────────────────

def test_youden_j_perfect_separation():
    """Perfectly separable values → J = 1.0."""
    values = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    is_pos = np.array([0, 0, 0, 1, 1, 1], bool)
    j = variant_selection_k7._youden_j(values, is_pos)
    assert abs(j - 1.0) < 1e-9, f"perfect separation should give J=1; got {j}"


def test_youden_j_no_separation():
    """No separation (all overlap) → J near 0."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 200)
    is_pos = rng.integers(0, 2, 200).astype(bool)
    j = variant_selection_k7._youden_j(values, is_pos)
    assert -0.2 < j < 0.3, f"random separation should give J ≈ 0; got {j}"


def test_youden_j_partial_overlap():
    """Partial overlap → J between 0 and 1."""
    rng = np.random.default_rng(42)
    pos = rng.normal(1.0, 1.0, 100)
    neg = rng.normal(-1.0, 1.0, 100)
    values = np.concatenate([pos, neg])
    is_pos = np.array([True] * 100 + [False] * 100)
    j = variant_selection_k7._youden_j(values, is_pos)
    # Expected J ~ 0.5-0.7 for this configuration
    assert 0.4 < j < 0.9, f"partial overlap should give moderate J; got {j}"


def test_youden_j_degenerate():
    """All positives or all negatives → NaN."""
    values = np.array([0.1, 0.2, 0.3])
    is_pos_all = np.array([True, True, True])
    j = variant_selection_k7._youden_j(values, is_pos_all)
    assert np.isnan(j), "all-positive should give NaN"


def test_youden_j_phase35_szj_threshold_calibration():
    """Sanity-check that the G3 threshold (sz J ≥ 0.40) corresponds to a
    reasonable separation. With Cohen's d ≈ 0.7 between expert (μ=0.6)
    and crowd (μ=-0.6) on N=200 raters, we should get J > 0.40."""
    rng = np.random.default_rng(42)
    expert_ell = rng.normal(0.6, 1.0, 100)
    crowd_ell = rng.normal(-0.6, 1.0, 100)
    values = np.concatenate([expert_ell, crowd_ell])
    is_expert = np.array([True] * 100 + [False] * 100)
    j = variant_selection_k7._youden_j(values, is_expert)
    assert j >= 0.40, (
        f"d=1.2 separation should comfortably clear J=0.40; got {j}")


# ── _cohen_d ─────────────────────────────────────────────────────────────

def test_cohen_d_known_separation():
    """Known +1 SD separation → d = 1.0 (within sample noise)."""
    rng = np.random.default_rng(42)
    a = rng.normal(1.0, 1.0, 1000)
    b = rng.normal(0.0, 1.0, 1000)
    d = variant_selection_k7._cohen_d(a, b)
    assert abs(d - 1.0) < 0.1, f"d=1 separation expected; got {d}"


def test_cohen_d_zero():
    """Same-mean distributions → d ≈ 0."""
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.0, 1000)
    b = rng.normal(0.0, 1.0, 1000)
    d = variant_selection_k7._cohen_d(a, b)
    assert abs(d) < 0.1, f"zero separation expected; got {d}"


def test_cohen_d_degenerate():
    """Single observation → NaN."""
    d = variant_selection_k7._cohen_d(np.array([1.0]), np.array([0.0, 1.0]))
    assert np.isnan(d)


# ── s_j_drift_against_k6 ─────────────────────────────────────────────────

def test_s_j_drift_zero():
    """K=7 == K=6 → max drift = 0."""
    seg_ids = np.array([1, 2, 3, 4, 5])
    s_mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    s_sd = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    baseline = baseline_k6.K6Baseline(
        task="test", seg_ids=seg_ids, s_mean=s_mean, s_sd=s_sd,
        s_sd_calibrated=s_sd, nuts_rhat_s_j=None, nuts_rhat_ell=None,
        nuts_rhat_t=None, n_seg=5,
        rater_ids=np.array([]), rater_sigma=np.array([]),
        rater_theta=np.array([]), rater_converged=np.array([], bool))
    drift = baseline_k6.s_j_drift_against_k6(seg_ids, s_mean, baseline)
    assert drift["n_overlap"] == 5
    assert drift["max_abs_drift"] == 0.0


def test_s_j_drift_known():
    """Known per-seg drift."""
    seg_ids = np.array([1, 2, 3])
    k6_s = np.array([0.1, 0.2, 0.3])
    k7_s = np.array([0.15, 0.25, 0.35])  # uniform +0.05 drift
    baseline = baseline_k6.K6Baseline(
        task="test", seg_ids=seg_ids, s_mean=k6_s, s_sd=k6_s * 0.1,
        s_sd_calibrated=k6_s * 0.1, nuts_rhat_s_j=None, nuts_rhat_ell=None,
        nuts_rhat_t=None, n_seg=3,
        rater_ids=np.array([]), rater_sigma=np.array([]),
        rater_theta=np.array([]), rater_converged=np.array([], bool))
    drift = baseline_k6.s_j_drift_against_k6(seg_ids, k7_s, baseline)
    assert abs(drift["max_abs_drift"] - 0.05) < 1e-9


def test_s_j_drift_partial_overlap():
    """K=7 and K=6 with partial seg_id overlap."""
    k6_seg = np.array([1, 2, 3, 4])
    k6_s = np.array([0.1, 0.2, 0.3, 0.4])
    k7_seg = np.array([2, 3, 4, 5])
    k7_s = np.array([0.21, 0.32, 0.40, 0.5])  # drifts on overlap: 0.01, 0.02, 0
    baseline = baseline_k6.K6Baseline(
        task="test", seg_ids=k6_seg, s_mean=k6_s, s_sd=k6_s * 0.1,
        s_sd_calibrated=k6_s * 0.1, nuts_rhat_s_j=None, nuts_rhat_ell=None,
        nuts_rhat_t=None, n_seg=4,
        rater_ids=np.array([]), rater_sigma=np.array([]),
        rater_theta=np.array([]), rater_converged=np.array([], bool))
    drift = baseline_k6.s_j_drift_against_k6(k7_seg, k7_s, baseline)
    assert drift["n_overlap"] == 3  # segs 2, 3, 4
    assert abs(drift["max_abs_drift"] - 0.02) < 1e-9
