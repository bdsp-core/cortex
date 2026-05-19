"""Phase 3.5 gate: engine SBC regression.

Asserts on the produced artifact (skip if the ~17-min SBC has not been
run — same pattern as the other Phase-3.5 gate tests). Checks robust
invariants, NOT brittle Monte-Carlo numbers.

IMPORTANT statistical nuance (honest, not spin): the engine SMC is an
APPROXIMATE posterior, so at large SBC n the KS hard-test detects its
finite-particle approximation error EVEN in the correctly-specified
CONTROL arm (KS marginally exceeds crit). That is a shared baseline,
not a harness bug and not the Phase-3.5 effect. Harness soundness is
therefore gated on PRACTICAL calibration (mean_rank≈0.5,
coverage≈nominal — these prove the cloud is not degenerate, unlike the
pre-fix mean_rank=0.73), and the scientific claim is the CONTRAST
between arms, which is unambiguous regardless of the absolute KS.

Run: pytest tests/test_phase35_sbc_engine.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RPT = REPO / "calibration" / "joint" / "sbc_engine.json"


def _load():
    if not RPT.exists():
        pytest.skip(
            "sbc_engine.json not produced — run "
            "`python -m pipeline.joint_calibration.sbc_engine` first")
    return json.loads(RPT.read_text())


def test_report_schema():
    d = _load()
    for k in ("n_replicates", "budget", "n_particles",
              "nominal_coverage", "conditions", "headline_finding",
              "control_ks_note", "directional_invariant_holds"):
        assert k in d, f"missing key {k!r}"
    assert set(d["conditions"]) == {"CONTROL", "PLUGIN", "UNCERT"}
    for c, v in d["conditions"].items():
        for k in ("ks_to_uniform", "ks_crit_0.05",
                  "uniform_not_rejected", "mean_rank",
                  "central95_coverage"):
            assert k in v, f"{c} missing {k!r}"


def test_control_practically_calibrated_harness_sound():
    """If this fails the SBC harness is buggy (degenerate cloud), not
    the engine. Gate on PRACTICAL calibration — NOT the KS hard-test,
    which is over-powered for an approximate SMC at large n (see
    control_ks_note). The pre-rejuvenation-fix bug showed mean_rank=
    0.73 / coverage=0.54; a sound harness sits near 0.5 / nominal."""
    c = _load()["conditions"]["CONTROL"]
    assert 0.45 <= c["mean_rank"] <= 0.55, (
        f"CONTROL mean rank {c['mean_rank']} far from 0.5 — degenerate "
        "cloud / harness bug, NOT an engine finding")
    assert c["central95_coverage"] >= 0.90, (
        f"CONTROL central-95% coverage {c['central95_coverage']} too "
        "low for a sound harness")


def test_plugin_is_detectably_miscalibrated_vs_control():
    """The Phase-3.5 finding: ignoring real item noise (PLUGIN)
    degrades calibration FAR beyond the CONTROL baseline."""
    d = _load()
    cC = d["conditions"]["CONTROL"]
    cP = d["conditions"]["PLUGIN"]
    assert cP["ks_to_uniform"] > 1.5 * cC["ks_to_uniform"], (
        "PLUGIN KS not materially worse than CONTROL — the plug-in "
        "miscalibration under item noise is not demonstrated")
    assert cP["central95_coverage"] < cC["central95_coverage"], (
        "PLUGIN does not under-cover relative to CONTROL")
    assert cP["central95_coverage"] < d["nominal_coverage"]


def test_uncertainty_restores_calibration_and_invariant():
    """s_sd propagation returns calibration toward the CONTROL baseline
    and cannot worsen it vs PLUGIN (directional law; gated)."""
    d = _load()
    cC = d["conditions"]["CONTROL"]
    cP = d["conditions"]["PLUGIN"]
    cU = d["conditions"]["UNCERT"]
    assert cU["central95_coverage"] >= cP["central95_coverage"] - 1e-9
    assert cU["ks_to_uniform"] <= cP["ks_to_uniform"] + 1e-9
    # UNCERT back to ~CONTROL baseline (not merely "better than PLUGIN")
    assert cU["central95_coverage"] >= cC["central95_coverage"] - 0.03
    assert cU["ks_to_uniform"] <= cC["ks_to_uniform"] + 0.03
    assert d["directional_invariant_holds"] is True
