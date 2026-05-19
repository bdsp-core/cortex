"""Phase 3.5 gate: plug-in vs uncertainty-propagated AUROC regression.

Asserts on the produced artifact (skip if the multi-minute harness has
not been run — same pattern as test_phase3_calibration.py). Checks ONLY
theoretically-guaranteed invariants, NOT a fragile effect size:

  * s_sd marginalisation can only WIDEN the per-item likelihood and
    REDUCE per-item Fisher information (closed-form, proven exact by
    test_phase35_engine_ssd). Therefore at a fixed budget the
    uncertainty arm's AUROC CI halfwidth must be >= plug-in's and its
    coverage must be >= plug-in's. This is a hard directional law, safe
    to regression-gate.
  * The realised magnitude is honestly SMALL (~2%); we assert a loose
    sanity band (1.0 <= ratio < 1.5) to catch a blow-up/sign bug
    WITHOUT pinning a number (pinning would be p-hacky and brittle).

Run: pytest tests/test_phase35_plugin_vs_uncertainty.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RPT = REPO / "calibration" / "joint" / "plugin_vs_uncertainty_auroc.json"


def _load():
    if not RPT.exists():
        pytest.skip(
            "plugin_vs_uncertainty_auroc.json not produced yet — run "
            "`python -m pipeline.joint_calibration."
            "plugin_vs_uncertainty_auroc` first")
    return json.loads(RPT.read_text())


def test_report_schema():
    d = _load()
    for k in ("metric", "fixed_q_per_session", "n_sessions_per_arm",
              "mean_ci_halfwidth_plugin", "mean_ci_halfwidth_uncert",
              "ci_halfwidth_ratio_uncert_over_plugin",
              "coverage_plugin", "coverage_uncert",
              "directional_invariant_holds", "headline_finding", "rows"):
        assert k in d, f"missing key {k!r}"
    # fixed-budget metric (NOT the rejected stop-at-δ confounded one)
    assert "FIXED budget" in d["metric"] or "fixed-budget" in d["metric"]
    assert d["fixed_q_per_session"] >= 50
    assert d["n_sessions_per_arm"] == len(d["rows"]) >= 16


def test_directional_invariant_holds():
    """Hard law: marginalising item error cannot NARROW the AUROC CI
    nor REDUCE coverage at a fixed budget."""
    d = _load()
    assert d["mean_ci_halfwidth_uncert"] >= d["mean_ci_halfwidth_plugin"], (
        "uncertainty CI is narrower than plug-in — violates the "
        "Fisher-deflation law (engine s_sd wiring regression?)")
    assert d["coverage_uncert"] >= d["coverage_plugin"], (
        "uncertainty coverage below plug-in — directional violation")
    assert d["directional_invariant_holds"] is True


def test_effect_magnitude_sane_band():
    """Honest small effect: loose band catches a blow-up / sign bug
    without pinning a brittle number."""
    d = _load()
    ratio = d["ci_halfwidth_ratio_uncert_over_plugin"]
    assert 1.0 <= ratio < 1.5, f"CI-halfwidth ratio {ratio} out of band"
    # both arms conservative for the synthetic regime (honest finding:
    # the plug-in engine was NOT badly overstating power here)
    assert 0.90 <= d["coverage_plugin"] <= 1.0
    assert 0.90 <= d["coverage_uncert"] <= 1.0
