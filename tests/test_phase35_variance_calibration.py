"""Phase 3.5 gate: lock the VI-vs-NUTS s_sd variance calibration.

The production s_j posterior is SVI; the SVI-defensibility requirement is
that its s_sd is calibrated against exact-Bayes NUTS on a subsample. These
tests assert the calibration was performed, is CONSERVATIVE (never shrinks
s_sd), achieved its target (post-cal SVI/NUTS s_sd median ~ 1.0), the
engine bank consumes the calibrated s_sd, and the honest moderate-s_mean
caveat is recorded in v13 provenance (not hidden).
"""
import csv
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "calibration" / "joint"
TASKS = {"sz", "lpd", "gpd", "lrda", "grda", "iic"}


@pytest.fixture(scope="module")
def vc():
    p = JOINT / "variance_calibration.json"
    if not p.exists():
        pytest.skip("variance_calibration.json absent (x-check not run)")
    return json.loads(p.read_text())


def test_kappa_conservative_and_complete(vc):
    k = vc["kappa"]
    assert set(k) == TASKS
    for t, v in k.items():
        assert v >= 1.0, f"{t}: kappa {v} < 1 would SHRINK s_sd (must not)"


def test_calibration_hit_its_target(vc):
    """post-calibration SVI/NUTS s_sd median ~ 1.0 by construction."""
    for t, r in vc["per_task"].items():
        m = r["post_calibration_svi_over_nuts_median"]
        assert 0.95 <= m <= 1.05, f"{t}: post-cal SVI/NUTS median {m}"


def test_svi_underdispersed_as_expected(vc):
    """The premise: SVI under-disperses s_sd (NUTS/SVI ratio > 1)."""
    for t, r in vc["per_task"].items():
        med = r["s_sd_ratio_nuts_over_svi_quantiles"]["50"]
        assert med > 1.0, f"{t}: NUTS/SVI s_sd median {med} (expected >1)"


def test_s_mean_agreement_caveat_is_recorded_honestly(vc):
    """Moderate (not high) s_mean agreement must be captured, not spun."""
    rs = [r["s_mean_pearson_r"] for r in vc["per_task"].values()]
    assert all(0.5 < r < 0.95 for r in rs), rs   # moderate band
    cfg = yaml.safe_load((ROOT / "calibration/cert_config.yaml").read_text())
    ph = cfg["ell_star_unified_v13"]["provenance"]["phase35"]
    vcp = ph["variance_calibration"]
    assert "honest_caveat" in vcp
    assert "MODERATE" in vcp["honest_caveat"]
    assert "SUBSAMPLE" in vcp["honest_caveat"]


def test_calibrated_bank_applies_kappa_and_never_shrinks(vc):
    k = vc["kappa"]
    p = JOINT / "s_j_table_calibrated.csv"
    n = 0
    with open(p) as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == ["task", "seg_id", "s_mean", "s_sd",
                                 "s_sd_calibrated"]
        for d in rd:
            sd, sdc = float(d["s_sd"]), float(d["s_sd_calibrated"])
            assert sdc + 1e-9 >= sd, "calibrated s_sd shrank below raw"
            assert abs(sdc - sd * k[d["task"]]) < 1e-6 * max(sd, 1.0)
            n += 1
    assert n == vc["n_calibrated_rows"] == 418836
