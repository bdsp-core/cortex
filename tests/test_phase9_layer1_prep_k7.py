"""Phase 9 Layer 1 — prep_k7 + model_k7 smoke tests.

No compute; verifies the module-level structure, schemas, and a small
subsample-based prep load for both spike and IIIC families. Real NUTS/SVI
runs happen via the variant_comparison orchestrator (not in pytest).
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.joint_calibration import prep_k7, model_k7


# ── prep_k7 module-level constants ────────────────────────────────────────

def test_prep_k7_tasks_are_7():
    """K=7: spike + 6 IIIC."""
    assert prep_k7.TASKS == ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    assert len(prep_k7.TASK_TO_CLASS) == 7


def test_prep_k7_iiic_collapse():
    """{bipd, birds} → other applies to IIIC tasks only."""
    assert prep_k7._canon_iiic_value("bipd") == "other"
    assert prep_k7._canon_iiic_value("birds") == "other"
    assert prep_k7._canon_iiic_value("seizure") == "seizure"


def test_prep_k7_label_type_split():
    """spike uses label_type='spike' (NOT pattern_class)."""
    assert prep_k7.SPIKE_LABEL_TYPE == "spike"
    assert prep_k7.IIIC_LABEL_TYPE == "pattern_class"


def test_prep_k7_source_filters_disjoint():
    """Spike vs IIIC source-dataset sets are disjoint in labels.csv."""
    assert prep_k7.SPIKE_SOURCES_LABELS.isdisjoint(prep_k7.IIIC_SOURCES_LABELS)
    # spike labels.csv source is collapsed to sn1_combined_v2 (single value)
    assert prep_k7.SPIKE_SOURCES_LABELS == {"sn1_combined_v2"}


# ── prep_k7.load_task end-to-end (small subsample) ────────────────────────

@pytest.mark.parametrize("task", ["spike", "sz", "iic"])
def test_prep_k7_load_task_small(task):
    """Verifies load_task returns a coherent TaskData for spike + IIIC.

    Uses subsample=5000 to keep test fast (~10 s); skips compute-heavy parts.
    """
    td = prep_k7.load_task(task, subsample=5000, seed=42)
    assert td.task == task
    assert td.n_obs > 0
    assert td.n_seg > 0
    assert td.n_rater > 0
    assert td.n_src > 0
    # Index arrays match n_obs
    assert td.seg_idx.shape[0] == td.n_obs
    assert td.rater_idx.shape[0] == td.n_obs
    assert td.src_idx.shape[0] == td.n_obs
    assert td.y.shape[0] == td.n_obs
    # Anchor masks match n_seg
    assert td.scale_anchor_seg.shape[0] == td.n_seg
    assert td.loc_anchor_seg.shape[0] == td.n_seg
    # y is binary
    assert set(np.unique(td.y).tolist()).issubset({0, 1})
    # pos_rate in [0, 1]
    assert 0.0 <= td.pos_rate <= 1.0
    # tier vocabulary plausibly includes expected tiers
    tiers = set(td.rater_expertise)
    assert tiers, f"task {task} has no expertise tiers"
    # Spike-specific: sub-source granularity from segments.csv
    if task == "spike":
        assert td.n_src <= 3, "spike has at most 3 sub-sources"
        for s in td.src_names:
            assert s.startswith("sn1_combined_v2:"), (
                f"spike sub-source {s!r} must start with sn1_combined_v2:")
    else:
        # IIIC: 6 source datasets in labels.csv
        for s in td.src_names:
            assert s in prep_k7.IIIC_SOURCES_LABELS, (
                f"IIIC source {s!r} not in expected IIIC source set")


def test_prep_k7_spike_anchor_masks():
    """Spike scale anchor: sn1:sn1 sub-source segs only.
    Location anchor: non-empty mid-difficulty band.
    """
    td = prep_k7.load_task("spike", subsample=5000, seed=42)
    assert td.scale_anchor_seg.any(), "spike scale anchor empty"
    assert td.loc_anchor_seg.any(), "spike location anchor empty"


def test_prep_k7_iiic_anchor_masks():
    """IIIC scale anchor: sparcnet50K ∩ kong:crowd. Location: centaur gold."""
    td = prep_k7.load_task("iic", subsample=5000, seed=42)
    # With only 5k subsample, the anchor sets may be empty by chance;
    # check that the masks have the right shape but allow zeros.
    assert td.scale_anchor_seg.shape[0] == td.n_seg
    assert td.loc_anchor_seg.shape[0] == td.n_seg


# ── model_k7 dispatcher + variants ────────────────────────────────────────

def test_model_k7_variants_registered():
    """All three variants A/B/C are registered in the dispatcher."""
    assert set(model_k7.VARIANTS.keys()) == {"A", "B", "C"}
    for v in ("A", "B", "C"):
        fn = model_k7.get_model(v)
        assert callable(fn), f"variant {v} model not callable"


def test_model_k7_lapse_invariant():
    """LAMBDA = 0.025 is preserved across the K=7 transition (D2 invariant)."""
    assert model_k7.LAMBDA == 0.025


def test_model_k7_compute_source_weights():
    """V_C helper: weights sum to ~n_obs (mean = 1) per the normalization."""
    src_idx = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    w = model_k7.compute_source_weights(src_idx, n_src=3)
    assert w.shape == src_idx.shape
    # Mean is normalized to 1.0
    assert abs(float(w.mean()) - 1.0) < 1e-9
    # Smaller sources get higher per-obs weight
    assert w[src_idx == 2].mean() > w[src_idx == 0].mean()
