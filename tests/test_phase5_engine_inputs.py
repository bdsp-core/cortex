"""Phase 5 — engine_inputs provenance/layout reconciliation (D3).

Audit found the original plan stale: `scripts/build_engine_inputs.py`
read the *sibling* repo and produced the 219-row legacy artifact,
while the live `sdt_fits.csv` is the Phase-3 recompute on the unified
`data/labels` (true producer = `pipeline/run_unified_calibration.py`
STEP d). Phase 5 = reconcile provenance (no content regen): the
builder is now a no-sibling MANIFEST verifier/regenerator; the stray
`sdt_fits.phase3.csv` was removed; D3 shared-lineage with
`deployment_prior` is recorded + cross-checked.

These tests assert the reconciliation invariants. They do NOT invoke
`build_engine_inputs.main()` (it rewrites MANIFEST.json, incl. the
`generated` date — calling it in the suite would churn a tracked
file); instead they exercise the pure `_verify()` and assert on the
committed MANIFEST.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EI = REPO / "data" / "engine_inputs"
MANIFEST = EI / "MANIFEST.json"
BUILDER = REPO / "scripts" / "build_engine_inputs.py"
DEPLOY_SUMMARY = REPO / "data" / "deployment_prior" / "summary.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "_p5_build_engine_inputs", BUILDER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_builder_is_sibling_free_d9():
    """D9: the reconciled builder must NOT read the sibling repo."""
    src = BUILDER.read_text()
    assert "ilae-skill-certification-test-main" not in src
    assert "data/prepared" not in src
    assert "SIBLING" not in src


def test_builder_self_check_passes():
    """The pure _verify() (no MANIFEST write) reports zero failures —
    the same contract tests/test_phase3_calibration asserts."""
    m = _load_builder()
    failures = m._verify()
    assert failures == [], f"engine_inputs self-check failures: {failures}"
    # the D3 anchor helper returns the live data/labels sha256
    a = m._d3_anchor()
    assert len(a["labels_csv_sha256"]) == 64
    assert len(a["raters_csv_sha256"]) == 64


def test_manifest_is_truthful_and_not_stale():
    """The committed MANIFEST describes the REAL (Phase-3/data-labels)
    artifact: true producer recorded, output_sha256 == live files,
    self-check passed, 14,214 rows, NO sibling source_dir."""
    import hashlib

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    mf = json.loads(MANIFEST.read_text())
    assert "run_unified_calibration.py STEP d" in mf["true_producer"]
    assert mf["sdt_fits_rows"] == 14214
    assert mf["self_check"]["passed"] is True
    assert mf["self_check"]["failures"] == []
    # output_sha256 matches the LIVE files (the prior MANIFEST didn't)
    for fname, sha in mf["output_sha256"].items():
        if "Sigma_l_fitted.npy" in fname:
            continue                       # repo-root frozen, separate
        p = EI / fname
        assert p.exists(), f"MANIFEST lists missing {fname}"
        assert _sha(p) == sha, f"MANIFEST output_sha256 stale for {fname}"
    # no sibling lineage as the *source* (recorded only historically)
    assert "source_dir" not in mf
    assert "superseded_legacy" in mf


def test_d3_shared_lineage_engine_inputs_eq_deployment_prior():
    """Plan D3: engine_inputs and deployment_prior are provably
    derived from the SAME unified data/labels corpus — identical
    sha256 anchors, and the live corpus matches both."""
    import hashlib

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    ei = json.loads(MANIFEST.read_text())["data_labels_provenance"]
    dp = json.loads(DEPLOY_SUMMARY.read_text())["data_labels_provenance"]
    assert ei["labels_csv_sha256"] == dp["labels_csv_sha256"]
    assert ei["raters_csv_sha256"] == dp["raters_csv_sha256"]
    # and both equal the LIVE corpus
    assert ei["labels_csv_sha256"] == _sha(
        REPO / "data" / "labels" / "labels.csv")
    assert ei["raters_csv_sha256"] == _sha(
        REPO / "data" / "labels" / "raters.csv")
    d3 = json.loads(MANIFEST.read_text())[
        "d3_shared_lineage_with_deployment_prior"]
    assert d3["checked"] is True
    assert d3["shares_data_labels_lineage"] is True


def test_stray_phase3_snapshot_removed_and_authoritative_set():
    """The footgun stray is gone; the authoritative/historical pair
    stays; README points at the real producer (not the stale claim)."""
    assert not (EI / "sdt_fits.phase3.csv").exists()
    assert (EI / "sdt_fits.csv").exists()                 # authoritative
    assert (EI / "sdt_fits.legacy_oldcorpus.csv").exists()  # historical
    readme = (EI / "README.md").read_text()
    assert "run_unified_calibration.py` STEP d" in readme
    assert "D3" in readme
    # the stale pre-Phase-5 claims must be gone
    assert "Built + verified by `scripts/build_engine_inputs.py`" \
        not in readme
    assert "Reads the sibling `data/prepared/`" not in readme
