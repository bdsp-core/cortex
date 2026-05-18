"""Phase-3 reference-faithful calibration tests.

Fast tests run by default; the heavy end-to-end orchestrator is NOT
re-run here (it is run once by the orchestrator itself). Tests assert on
its produced artifacts plus the byte-identity / schema invariants.

Run all:   pytest -m "" tests/test_phase3_calibration.py -q
Skip slow: pytest tests/test_phase3_calibration.py -q
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
REF_SRC = (
    REPO.parent
    / "ilae-skill-certification-test-main"
    / "src"
)
REF_MULTI = (
    REPO.parent / "ilae-skill-certification-test-multi-main"
)
REF_CALIB = REPO / "pipeline" / "reference_calibration"
WORK_PREPARED = REPO / "pipeline" / "_calib_work" / "data" / "prepared"
ENGINE_INPUTS = REPO / "data" / "engine_inputs"
CALIB_DIR = REPO / "calibration"

ALL_TASKS = [
    "combined_spike",
    "sparcnet_sz",
    "sparcnet_lpd",
    "sparcnet_gpd",
    "sparcnet_lrda",
    "sparcnet_grda",
    "sparcnet_iic",
]


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ── (a) carried scripts byte-identical to reference src/ ────────────────────
def test_fit_main_effects_byte_identical():
    a = REF_SRC / "fit_main_effects.py"
    b = REF_CALIB / "fit_main_effects.py"
    assert _md5(a) == _md5(b), (
        f"fit_main_effects md5 mismatch: {_md5(a)} vs {_md5(b)}"
    )
    assert _md5(b) == "fcf149b4f9c173ab14e9980ee3b82a13"


def test_fit_sdt_per_domain_byte_identical():
    a = REF_SRC / "fit_sdt_per_domain.py"
    b = REF_CALIB / "fit_sdt_per_domain.py"
    assert _md5(a) == _md5(b)
    assert _md5(b) == "6b90d59dcd0aaa878f9a52802254044b"


def test_run_youden_byte_identical():
    a = REF_MULTI / "run_youden_calibration.py"
    b = REF_CALIB / "run_youden_calibration.py"
    assert _md5(a) == _md5(b)
    assert _md5(b) == "f8bcfbd61b7a89c70e4dfec3cf8227db"


def test_youden_sigma_star_extract_byte_identical():
    """The extracted function body must equal reference lines 211-225."""
    ref = (
        REF_SRC / "train_val_split_and_fit.py"
    ).read_text().splitlines()
    ref_fn = "\n".join(ref[210:225])  # lines 211..225 (1-based)
    ext = (
        REF_CALIB / "youden_sigma_star_ref.py"
    ).read_text().splitlines()
    start = next(
        i
        for i, l in enumerate(ext)
        if l.startswith("def youden_sigma_star(")
    )
    ext_fn = "\n".join(ext[start : start + 15])
    assert ext_fn == ref_fn, "youden_sigma_star extract not byte-identical"


# ── (e) reference invariants present literally in carried fit_sdt ───────────
def test_reference_invariants_in_fit_sdt():
    txt = (REF_CALIB / "fit_sdt_per_domain.py").read_text()
    assert "LAMBDA = 0.025" in txt, "LAMBDA = 0.025 missing"
    assert "1.0 / 1.7" in txt, "1.0 / 1.7 (LOGIT_TO_PROBIT) missing"
    assert "MIN_FIT_TRIALS = 20" in txt


def test_reference_constants_in_extract():
    txt = (REF_CALIB / "youden_sigma_star_ref.py").read_text()
    assert "LAMBDA = 0.025" in txt
    assert "SEED = 42" in txt
    assert "EXPERT_TRAIN_FRAC = 0.70" in txt


# ── (b) build_calibration_inputs schema correctness ─────────────────────────
@pytest.mark.parametrize("task", ALL_TASKS)
def test_prepared_dense_ids_and_meta(task):
    csv = WORK_PREPARED / f"{task}.csv"
    meta_p = WORK_PREPARED / f"{task}_meta.json"
    if not csv.exists():
        pytest.skip(f"{csv} not produced yet (run orchestrator first)")
    df = pd.read_csv(csv)
    assert list(df.columns) == ["case_id", "rater_id", "Y"]
    # dense 0-based contiguous
    for col in ("case_id", "rater_id"):
        u = df[col].unique()
        assert u.min() == 0
        assert u.max() == len(u) - 1, f"{task}.{col} not contiguous 0-based"
    assert set(df["Y"].unique()) <= {0, 1}
    meta = json.load(open(meta_p))
    n_rater = df["rater_id"].max() + 1
    assert len(meta["rater_index_to_name"]) == n_rater
    assert meta["provenance"]["n_obs"] == len(df)


def test_y_binarization_counts_vs_labels():
    """Spot-check Y positive counts against labels.csv for each task."""
    labels = pd.read_csv(
        REPO / "data" / "labels" / "labels.csv", low_memory=False
    )
    expect = {}
    # combined_spike (UN-FOLD 2026-05-18): CLEAN sn1 binary ONLY —
    # value in {'0','1'}; Y=1 iff value=='1'. Centaur-IED spike-SUBTYPE
    # rows are EXCLUDED from the spike cert task (retained in labels.csv).
    sp = labels[labels.label_type == "spike"]
    spv = sp.value.astype(str).str.strip()
    expect["combined_spike"] = int((spv == "1").sum())
    pc = labels[labels.label_type == "pattern_class"]
    pcv = pc.value.astype(str).str.strip()
    for cls, task in [
        ("seizure", "sparcnet_sz"),
        ("lpd", "sparcnet_lpd"),
        ("gpd", "sparcnet_gpd"),
        ("lrda", "sparcnet_lrda"),
        ("grda", "sparcnet_grda"),
    ]:
        expect[task] = int((pcv == cls).sum())
    # sparcnet_iic: ERRATUM-CORRECTED — uniform {bipd,birds}->other
    # collapse across all sources, so the positive set is
    # {other, bipd, birds} (NOT 'other' only; the Phase-3 bug).
    expect["sparcnet_iic"] = int(pcv.isin(["other", "bipd", "birds"]).sum())
    for task, n_pos in expect.items():
        csv = WORK_PREPARED / f"{task}.csv"
        if not csv.exists():
            pytest.skip(f"{csv} not produced yet")
        df = pd.read_csv(csv)
        assert int(df["Y"].sum()) == n_pos, (
            f"{task}: Y positives {int(df['Y'].sum())} != {n_pos}"
        )


# ── (c) cert_config — CURRENT version (single source so a future bump is
#       a one-line change; Phase-3.5 moved v12 -> v13) ──────────────────────
_VER = 13
_VKEY = "ell_star_unified_v13"


def _load_cfg():
    p = CALIB_DIR / "cert_config.yaml"
    if not p.exists():
        pytest.skip("cert_config not emitted yet (run orchestrator)")
    return yaml.safe_load(p.read_text())


def test_cert_config_parses_and_version():
    assert _load_cfg()["config_version"] == _VER


def test_cert_config_has_7_ell_star_and_provenance():
    blk = _load_cfg()[_VKEY]
    assert set(blk["tasks"].keys()) == set(ALL_TASKS)
    assert len(blk["tasks"]) == 7
    prov = blk["provenance"]
    assert len(prov["labels_csv_sha256"]) == 64
    assert "carried_script_md5" in prov
    assert prov["seed"] == 42
    assert "d7_independent_panel" in prov


def test_cert_config_mode_b_legacy_preserved():
    cfg = _load_cfg()
    assert "mode_b_legacy" in cfg
    assert "l_star_per_domain" in cfg["mode_b_legacy"]
    # repo-root cert_config left untouched at v11
    root = yaml.safe_load((REPO / "cert_config.yaml").read_text())
    assert root["config_version"] == 11


# ── (d) every Q2 candidate resolved ─────────────────────────────────────────
def test_all_q2_candidates_resolved():
    """Stable correctness invariant (run-order-independent): every one of
    the 29 Q2-locked candidates must join the regenerated sdt_fits.csv by
    the Youden join key (rater_name == unified canonical_name), including
    the 4 known name-variant raters. NOTE: applied_mappings is a non-
    idempotent delta (empty on a re-run where the on-disk matrix is already
    reconciled) — it is NOT a correctness property, so it is not asserted
    here; resolution/joinability is."""
    cfg = _load_cfg()
    pool = cfg[_VKEY]["provenance"]["candidate_pool"]
    assert pool["n_candidates"] == 29
    assert pool["n_resolved"] == 29, (
        f"only {pool['n_resolved']}/29 resolved"
    )
    assert pool["unresolved"] == [], f"unresolved: {pool['unresolved']}"

    # Independent check against the Q2-LOCKED original (not the in-place
    # reconciled matrix): each candidate's resolved name is present in
    # sdt_fits.csv rater_name, so the verbatim Youden driver joins all 29.
    import pandas as pd
    q2 = pd.read_csv(
        ENGINE_INPUTS / "cross_domain_rater_matrix.q2locked.csv"
    )
    assert len(q2) == 29
    sdt = pd.read_csv(ENGINE_INPUTS / "sdt_fits.csv")
    sdt_names = set(sdt["rater_name"].astype(str).str.strip())
    recon = pd.read_csv(ENGINE_INPUTS / "cross_domain_rater_matrix.csv")
    recon_names = set(
        recon["confirmed_sparcnet_name"].astype(str).str.strip()
    )
    # the reconciled matrix the driver consumes must be 29 names that all
    # exist in sdt_fits (the actual join the verbatim Youden performs)
    assert len(recon) == 29
    missing = sorted(recon_names - sdt_names)
    assert not missing, f"candidates not joinable to sdt_fits: {missing}"
    # the 4 known name-variant raters must be present under their unified
    # canonical spelling (proves the R5 reconciliation actually happened)
    for canon in ("Aaron F. Struck", "Hiba A. Haider",
                  "Jonathan J. Halford", "Olga Taraschenko"):
        assert canon in sdt_names, (
            f"R5 name-variant rater {canon!r} absent from sdt_fits "
            "(candidate silently dropped from the CV panel)"
        )


@pytest.mark.slow
def test_unified_sdt_fits_has_6_iiic_domains():
    p = ENGINE_INPUTS / "sdt_fits.csv"
    if not p.exists():
        pytest.skip("sdt_fits.csv not assembled yet")
    df = pd.read_csv(p)
    assert set(df["domain"].unique()) == {
        "sz",
        "lpd",
        "gpd",
        "lrda",
        "grda",
        "iic",
    }
    assert (ENGINE_INPUTS / "sdt_fits.legacy_oldcorpus.csv").exists()
