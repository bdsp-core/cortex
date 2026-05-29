"""Phase 9 Layer 6b — build + fetch infrastructure smoke tests.

Verifies the production-bank build script's helpers + the per-session
fetcher are correct + deterministic without requiring the full ~37 GB
production bank to exist. The offline fallback bundle
(cortex_app/cortex_offline_fallback.h5 + .manifest.json) is the test
fixture — it has the same schema as production, just smaller (~1000 segs).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

FALLBACK_H5 = REPO / "cortex_app" / "cortex_offline_fallback.h5"
FALLBACK_MANIFEST = REPO / "cortex_app" / "cortex_offline_fallback.manifest.json"


# ── module imports ────────────────────────────────────────────────────────

def test_build_cortex_bank_k7_production_imports():
    import build_cortex_bank_k7_production as b
    assert b.MIN_N_RATERS == 5
    assert b.FALLBACK_PER_TASK == 150
    assert set(b.IIIC_CLASSES) == {"seizure", "lpd", "gpd", "lrda",
                                    "grda", "other"}
    # SN1 sub-sources used for spike payloads
    assert "sn1_combined_v2:sn1" in b.SPIKE_SUBSOURCES


def test_cortex_session_bank_fetch_imports():
    import cortex_session_bank_fetch as f
    assert f.DEFAULT_PER_TASK == 60  # v1.2.0: lowered from 200
    assert f.TASK_CODES == ("spike", "sz", "lpd", "gpd", "lrda", "grda", "iic")
    # Pattern-class → engine task code mapping
    assert f._CLS_TO_CODE["seizure"] == "sz"
    assert f._CLS_TO_CODE["other"] == "iic"
    assert f._CLS_TO_CODE["spike"] == "spike"


# ── offline fallback existence + schema ───────────────────────────────────

@pytest.mark.skipif(not FALLBACK_H5.exists(),
                    reason="offline fallback not built; run "
                           "build_cortex_bank_k7_production.py --mode fallback")
def test_offline_fallback_schema():
    """Fallback bundle has expected /spike + /iiic groups with correct schema."""
    with h5py.File(FALLBACK_H5, "r") as f:
        assert set(f.keys()) >= {"spike", "iiic"}
        # Spike: eeg10s only, attrs include s_mean/s_sd/n_raters
        first_spike = list(f["spike"].keys())[0]
        sg = f[f"spike/{first_spike}"]
        assert "eeg10s" in sg
        assert sg["eeg10s"].shape == (20, 1281)
        for attr in ("seg_id", "n_raters", "s_mean", "family"):
            assert attr in sg.attrs
        assert sg.attrs["family"] == "spike"
        # IIIC: eeg30s + sdata + sfreqs + stimes, attrs include pattern_class
        first_iiic = list(f["iiic"].keys())[0]
        ig = f[f"iiic/{first_iiic}"]
        assert "eeg30s" in ig
        assert ig["eeg30s"].shape == (20, 6000)
        for ds in ("sdata", "sfreqs", "stimes"):
            assert ds in ig
        assert ig.attrs["family"] == "iiic"
        assert ig.attrs["pattern_class"] in (
            "seizure", "lpd", "gpd", "lrda", "grda", "other")


@pytest.mark.skipif(not FALLBACK_MANIFEST.exists(),
                    reason="offline fallback manifest not built")
def test_offline_fallback_manifest_schema():
    """Manifest has required top-level + per-seg fields."""
    manifest = json.loads(FALLBACK_MANIFEST.read_text())
    assert manifest["schema_version"] == "production_v1"
    assert manifest["phase"] == 9
    assert manifest["K"] == 7
    assert "filter_criteria" in manifest
    assert "segments" in manifest
    # Per-seg required fields
    seg = manifest["segments"][0]
    for field in ("seg_id", "family", "pattern_class",
                  "source_dataset", "n_raters", "group_path",
                  "payload_sha256"):
        assert field in seg, f"manifest seg missing {field}"
    # Family check
    families = {s["family"] for s in manifest["segments"]}
    assert families == {"spike", "iiic"}
    # IIIC pattern_class coverage
    iiic_classes = {s["pattern_class"] for s in manifest["segments"]
                    if s["family"] == "iiic"}
    assert iiic_classes >= {"seizure", "lpd", "gpd", "lrda", "grda", "other"}


# ── fetcher: stratified sampling on the fallback bundle ───────────────────

@pytest.mark.skipif(not FALLBACK_MANIFEST.exists(),
                    reason="fallback manifest not built")
def test_session_fetch_deterministic_with_session_id():
    """Same session_id → same sampled segments. Pure-Python; no file I/O."""
    import cortex_session_bank_fetch as f
    manifest = f._load_manifest(str(FALLBACK_MANIFEST))
    a = f.sample_session_segments(manifest, "ses_unit_test_001", per_task=30)
    b = f.sample_session_segments(manifest, "ses_unit_test_001", per_task=30)
    c = f.sample_session_segments(manifest, "ses_unit_test_002", per_task=30)
    ids_a = sorted([s["seg_id"] for s in a])
    ids_b = sorted([s["seg_id"] for s in b])
    ids_c = sorted([s["seg_id"] for s in c])
    assert ids_a == ids_b, "fetcher not deterministic across runs"
    assert ids_a != ids_c, "different session_ids gave identical samples"


@pytest.mark.skipif(not FALLBACK_MANIFEST.exists(),
                    reason="fallback manifest not built")
def test_session_fetch_per_task_balance():
    """Per-task allocation balances across spike + 6 IIIC pattern_classes."""
    import cortex_session_bank_fetch as f
    manifest = f._load_manifest(str(FALLBACK_MANIFEST))
    sampled = f.sample_session_segments(
        manifest, "ses_balance_test", per_task=30)
    spike_n = sum(1 for s in sampled if s["family"] == "spike")
    iiic_by_cls = {}
    for s in sampled:
        if s["family"] == "iiic":
            iiic_by_cls[s["pattern_class"]] = (
                iiic_by_cls.get(s["pattern_class"], 0) + 1)
    # Spike: target per_task (may be less if pool < per_task)
    assert spike_n > 0
    # All 6 IIIC classes represented (or pool too small)
    assert len(iiic_by_cls) == 6
    # Balance within ~1 stratum (per_stratum = per_task//N_STRATA = 3)
    for cls, n in iiic_by_cls.items():
        assert n >= 1, f"class {cls} under-sampled: {n}"


@pytest.mark.skipif(not FALLBACK_H5.exists() or not FALLBACK_MANIFEST.exists(),
                    reason="fallback bundle not built")
def test_session_materialise_end_to_end(tmp_path):
    """Full pipeline: manifest → sample → fetch into a session h5."""
    import cortex_session_bank_fetch as f
    out = tmp_path / "session_test.h5"
    result = f.fetch_session_bank(
        str(FALLBACK_MANIFEST),
        session_id="ses_e2e_test",
        out_path=out,
        per_task=20,
        offline_fallback_path=FALLBACK_H5)
    assert result == out
    assert out.exists()
    with h5py.File(out, "r") as h:
        assert h.attrs["session_id"] == "ses_e2e_test"
        assert h.attrs["K"] == 7
        n_spike = len(h.get("spike", {}))
        n_iiic = len(h.get("iiic", {}))
        # At target = 20/task × 7 = 140; spike may be < 20 if pool is tiny
        assert 0 < n_spike <= 20
        assert n_iiic >= 6  # at least 1 per IIIC class
        # Round-tripped attrs survive the copy
        if n_spike > 0:
            first_spike = list(h["spike"].keys())[0]
            sg = h[f"spike/{first_spike}"]
            assert "s_mean" in sg.attrs
            assert "n_raters" in sg.attrs


@pytest.mark.skipif(not FALLBACK_H5.exists(),
                    reason="fallback bundle not built (needed for fall-back)")
def test_session_fetch_falls_back_on_missing_manifest(tmp_path):
    """If the manifest path is bad, fall back to copying the offline bundle."""
    import cortex_session_bank_fetch as f
    out = tmp_path / "session_offline.h5"
    bogus_manifest = tmp_path / "does_not_exist.json"
    result = f.fetch_session_bank(
        str(bogus_manifest),
        session_id="ses_offline_test",
        out_path=out,
        per_task=20,
        offline_fallback_path=FALLBACK_H5)
    assert result == out
    with h5py.File(out, "r") as h:
        assert h.attrs.get("offline_fallback") is True or \
               h.attrs.get("offline_fallback") == np.True_
        assert h.attrs["session_id"] == "ses_offline_test"


# ── pattern_class → engine task code mapping consistency ─────────────────

def test_cls_to_code_covers_all_k7_tasks():
    """All K=7 engine task codes are reachable via _CLS_TO_CODE values."""
    import cortex_session_bank_fetch as f
    target_codes = set(f.TASK_CODES)
    mapped_codes = set(f._CLS_TO_CODE.values())
    assert target_codes == mapped_codes


# ── manifest filter_criteria documents drops (Nature defensibility) ──────

@pytest.mark.skipif(not FALLBACK_MANIFEST.exists(),
                    reason="fallback manifest not built")
def test_manifest_filter_criteria_documents_drops():
    """Per Layer-6b investigation: manifest records the seg drops so Nature
    reviewers can see WHY the bank is the size it is."""
    manifest = json.loads(FALLBACK_MANIFEST.read_text())
    iiic = manifest["filter_criteria"]["iiic"]
    assert iiic["min_n_raters"] >= 5
    assert iiic["spec_filter"] == "morgoth1_recompute only"
    assert iiic["shape_filter"] == "(20, 6000) only"
    # Drop counters present (even if zero in fallback)
    for k in ("n_skipped_missing_from_spec", "n_skipped_kong_precomputed",
              "n_skipped_other_spec_source", "n_skipped_bad_shape"):
        assert k in iiic
