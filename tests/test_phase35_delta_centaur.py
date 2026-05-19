"""Phase 3.5 gate: δ_Centaur sensitivity regression.

Asserts on the produced artifact (skip if the cheap probe has not been
run — same pattern as the other Phase-3.5 gate tests). Checks the
robustness invariant + schema, NOT brittle smoke-scale numbers:

  * δ_Centaur is partially non-identified against the deterministic
    Centaur-gold location anchor. The honest claim is that zeroing it
    perturbs the unified s_j NO MORE than a same-model/different-seed
    SVI re-run (ratio <= ~1.5) while preserving signal shape
    (Pearson r >= 0.95). That is a robust directional claim safe to
    gate; the raw shift magnitude (smoke-scale SVI noise) is NOT.

Run: pytest tests/test_phase35_delta_centaur.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RPT = REPO / "calibration" / "joint" / "delta_centaur_sensitivity.json"


def _load():
    if not RPT.exists():
        pytest.skip(
            "delta_centaur_sensitivity.json not produced — run "
            "`python -m pipeline.joint_calibration."
            "delta_centaur_sensitivity` first")
    return json.loads(RPT.read_text())


def test_report_schema():
    d = _load()
    for k in ("scope", "probe_tasks", "min_s_j_pearson_r_delta_swap",
              "max_delta_shift_over_noise_floor_ratio", "robust",
              "interpretation", "per_task"):
        assert k in d, f"missing key {k!r}"
    assert "CHEAP" in d["scope"] and "NOT a full refit" in d["scope"]
    assert len(d["per_task"]) >= 1
    for task, v in d["per_task"].items():
        for k in ("delta_centaur_baseline_estimate",
                  "s_j_pearson_r_base_vs_fixed",
                  "s_j_max_abs_shift_delta_swap",
                  "s_j_max_abs_shift_svi_noise_floor",
                  "delta_shift_over_noise_floor_ratio"):
            assert k in v, f"{task} missing {k!r}"


def test_delta_centaur_robust_vs_noise_floor():
    """Zeroing δ_Centaur must perturb s_j no more than re-running SVI
    (ratio <= 1.5) and preserve shape (r >= 0.95)."""
    d = _load()
    assert d["min_s_j_pearson_r_delta_swap"] >= 0.95, (
        "s_j shape not preserved under δ_Centaur=0 — genuine "
        "dependence, gauge NOT robust")
    assert d["max_delta_shift_over_noise_floor_ratio"] <= 1.5, (
        "δ_Centaur swap perturbs s_j MORE than the SVI noise floor — "
        "δ_Centaur is identified and materially affects the unified "
        "signal (re-scope to a production refit)")
    assert d["robust"] is True
