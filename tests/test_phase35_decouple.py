"""Phase 3.5 part-2 — lock the converged decouple as regression-tested.

Guards the three decisions that the decouple converged on, so they
cannot silently regress:
  (1) DECOUPLE: canonical engine_inputs/sdt_fits.csv is the erratum-fixed
      VERBATIM two-stage output — NOT the joint per-rater params (which
      gave the documented sz J=0.089 degeneracy).
  (2) ERRATUM FIX: build_calibration_inputs applies {bipd,birds}->other
      uniformly, so the `iic` one-vs-rest positive set includes the
      Centaur-novice bipd/birds reads.
  (3) cert_config v13: erratum-fixed two-stage ell* + joint s_j/s_sd
      artifacts + phase35 provenance; mode_b_legacy preserved.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EI = ROOT / "data" / "engine_inputs"
JOINT = ROOT / "calibration" / "joint"


def _cfg():
    return yaml.safe_load((ROOT / "calibration" / "cert_config.yaml").read_text())


# ---- (1) DECOUPLE: sdt_fits.csv is two-stage, not the joint side-car ----

def test_canonical_sdt_fits_is_two_stage_not_joint():
    """DECOUPLE invariant, tested from MEASURED separation (no magic
    constants): the canonical file is the erratum-fixed two-stage output;
    the joint per-rater params live ONLY in the side-car and are on a very
    different gauge-fixed scale. Discriminate canonical vs the side-car
    directly (the side-car is the ground truth for "what the joint scale
    looks like")."""
    side = JOINT / "joint_per_rater_params.csv"
    assert side.exists(), "joint side-car missing (assemble not run)"

    def med_by_dom(path, dom):
        v = [float(r["sigma"]) for r in csv.DictReader(open(path))
             if r["domain"] == dom and r["sigma"] not in ("", "nan")]
        return (float(np.median(v)), len(v)) if v else (float("nan"), 0)

    # Structural invariant (measured): the verbatim two-stage pipeline
    # emits non-converged / empty-σ rows for low-trial raters (canonical
    # has ~2520); the SVI joint fit converges every rater (side-car has
    # exactly 0). Identical-shape files with the side-car's 0-empty
    # signature ⇒ sdt_fits.csv was overwritten by the joint params.
    def empties(path):
        r = list(csv.DictReader(open(path)))
        return sum(1 for x in r if x["sigma"] in ("", "nan")
                   or str(x["converged"]).lower() != "true")
    e_canon, e_side = empties(EI / "sdt_fits.csv"), empties(side)
    assert e_canon > 0 and e_side == 0, (
        f"canonical non-converged/empty σ rows={e_canon} (two-stage "
        f"expects >0), side-car={e_side} (joint expects 0) — decouple "
        "violated: sdt_fits.csv looks like the joint params")
    for dom in ("sz", "iic", "lpd"):
        cm, cn = med_by_dom(EI / "sdt_fits.csv", dom)
        sm, sn = med_by_dom(side, dom)
        # canonical (two-stage) σ median must be materially BELOW the
        # joint side-car's for the same domain (measured ratios ≈
        # 0.34–0.62; gate at 0.75 with margin). Equality ⇒ overwrite.
        assert cm < 0.75 * sm, (
            f"{dom}: canonical σ median {cm:.3f} not materially below "
            f"joint side-car {sm:.3f} — sdt_fits.csv may be the joint "
            "params (decouple violated)")


def test_decouple_documented_in_v13_provenance():
    ph = _cfg()["ell_star_unified_v13"]["provenance"]["phase35"]
    assert "DECOUPLE" in ph["decision"]
    assert "0.089" in ph["decision"]  # the sz failure that forced it


# ---- (2) ERRATUM FIX: uniform {bipd,birds}->other ----

@pytest.mark.slow
def test_erratum_iic_positive_includes_birds_bipd():
    """The corrected build_calibration_inputs must count Centaur-novice
    bipd/birds as iic-POSITIVE (Phase-3 mislabelled them negative)."""
    from pipeline.build_calibration_inputs import build_task
    import pandas as pd
    labels = pd.read_csv(ROOT / "data/labels/labels.csv", low_memory=False)
    raters = pd.read_csv(ROOT / "data/labels/raters.csv", low_memory=False)
    pc = labels[labels.label_type == "pattern_class"]
    v = pc.value.astype(str).str.strip()
    expect_pos = int(v.isin(["other", "bipd", "birds"]).sum())
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s = build_task("sparcnet_iic", labels, raters, Path(d))
        got = pd.read_csv(Path(d) / "sparcnet_iic.csv")["Y"].sum()
    assert int(got) == expect_pos, (
        f"iic positives {got} != {expect_pos} (other+bipd+birds) — "
        "erratum reintroduced")


def test_erratum_recorded_and_iic_J_improved():
    ph = _cfg()["ell_star_unified_v13"]["provenance"]["phase35"]
    assert "{bipd,birds}->other" in ph["phase3_erratum_fixed"]
    iicJ = _cfg()["ell_star_unified_v13"]["tasks"]["sparcnet_iic"]["youden_j"]
    assert iicJ > 0.75, f"iic J={iicJ:.3f} (erratum fix gave 0.814)"
    assert ph["iiic_min_J"] >= 0.372  # the Phase-3.5 gate floor


# ---- (3) v13 structure + joint artifacts ----

def test_cert_config_v13_shape():
    c = _cfg()
    assert c["config_version"] == 13
    assert "ell_star_unified_v13" in c and "ell_star_unified_v12" not in c
    assert "mode_b_legacy" in c
    tasks = c["ell_star_unified_v13"]["tasks"]
    for t in ("sparcnet_sz", "sparcnet_lpd", "sparcnet_gpd",
              "sparcnet_lrda", "sparcnet_grda", "sparcnet_iic",
              "combined_spike"):
        assert t in tasks


def test_joint_sj_artifacts_present_and_schemas():
    sj = JOINT / "s_j_table.csv"
    den = ROOT / "data" / "labels" / "iiic_segment_signals.csv"
    assert sj.is_file() and den.is_file()
    h = next(csv.reader(open(sj)))
    assert h == ["task", "seg_id", "s_mean", "s_sd"]
    dh = next(csv.reader(open(den)))
    assert dh[0] == "seg_id" and "s_mean_iic" in dh and "s_sd_iic" in dh
    # s_sd must be present and non-degenerate (the whole point)
    import pandas as pd
    t = pd.read_csv(sj)
    assert t["s_sd"].median() > 0.05, "s_sd collapsed — power fix void"
    assert set(t["task"].unique()) == {
        "sz", "lpd", "gpd", "lrda", "grda", "iic"}
