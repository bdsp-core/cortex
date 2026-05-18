"""Phase 1 data-unification verification.

Asserts the "nothing lost" guarantee and every R1-R6 / gold-ingest
invariant against the actual on-disk files. Heavy whole-corpus checks are
marked `slow` (run them in the Phase-1 gate via `pytest -m ""`); structural
checks run by default.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "labels"
MINE_SRC = ROOT.parent / "ilae-skill-certification-test-multi-main"
PI_SRC = ROOT.parent / "ilae-skill-certification-test-multi-main-PI-code"

GOLD_RIDS = {"97", "99000001", "99000002", "99000003"}
CANON_6 = {"seizure", "lpd", "gpd", "lrda", "grda", "other"}


def _rows(p, cols=None):
    with open(p) as f:
        r = csv.DictReader(f)
        for d in r:
            yield d if cols is None else tuple(d[c] for c in cols)


def _data_rows(p):
    with open(p) as f:
        return sum(1 for _ in f) - 1


# ---- canonical table sizes -------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("labels.csv", 2_115_793), ("segments.csv", 95_327),
    ("raters.csv", 5_306), ("datasets.csv", 5),
    ("segment_labels.csv", 95_327),
])
def test_canonical_table_sizes(name, expected):
    assert _data_rows(LBL / name) == expected


# ---- "NOTHING LOST": MINE labels.csv subset-of unified ---------------------

@pytest.mark.slow
def test_nothing_lost_mine_subset_of_unified():
    if not MINE_SRC.exists():
        pytest.skip("merge-time check; MINE source sibling not present "
                    "(permanent record in DATA_PROVENANCE.md / manifest)")
    cols = ["seg_id", "rater_id", "label_type", "value", "source_dataset"]
    mine = Counter(_rows(MINE_SRC / "data/labels/labels.csv", cols))
    uni = Counter(_rows(LBL / "labels.csv", cols))
    missing = sum(max(0, c - uni.get(k, 0)) for k, c in mine.items())
    assert missing == 0, f"{missing} MINE observations lost"
    assert sum(uni.values()) - sum(mine.values()) == 812_592  # +PI +gold


@pytest.mark.slow
def test_labels_prefix_unchanged_vs_PI():
    """The first 2,095,793 rows must be PI's labels.csv byte-for-byte
    (gold ingest was strictly append-only)."""
    if not PI_SRC.exists():
        pytest.skip("PI source sibling not present")
    pi = (PI_SRC / "data/labels/labels.csv").read_bytes()
    uni = (LBL / "labels.csv").read_bytes()
    assert uni.startswith(pi), "pre-existing labels rows were modified"
    assert len(uni) > len(pi), "no rows appended"


# ---- gold-panel ingest correctness ----------------------------------------

@pytest.mark.slow
def test_gold_ingest_rows():
    segids = {d["seg_id"] for d in _rows(LBL / "segments.csv")}
    aligned, gold, gseg, rids, vals = set(), 0, set(), set(), set()
    for d in _rows(LBL / "labels.csv"):
        sd = d["source_dataset"]
        if sd == "centaur_2025_iiic":
            aligned.add(d["seg_id"])
        elif sd == "centaur_iiic_expert":
            gold += 1
            gseg.add(d["seg_id"]); rids.add(d["rater_id"])
            vals.add(d["value"])
            assert d["label_type"] == "pattern_class"
    assert gold == 20_000
    assert len(gseg) == 5_000
    assert gseg <= segids, "gold seg_ids not all in segments.csv"
    assert gseg <= aligned, "gold seg_ids not aligned w/ centaur_2025_iiic"
    assert rids == GOLD_RIDS, f"unexpected gold rater_ids {rids}"
    assert vals <= CANON_6, f"gold values not 6-class: {vals - CANON_6}"


def test_gold_raters_added_and_mbw_unchanged():
    by_id = {}
    for d in _rows(LBL / "raters.csv"):
        by_id.setdefault(d["rater_id"], []).append(d)
    for rid in ("99000001", "99000002", "99000003"):
        assert len(by_id[rid]) == 1
        assert by_id[rid][0]["expertise_level"] == "expert"
    assert len(by_id["97"]) == 1
    assert by_id["97"][0]["canonical_name"] == "M. Brandon Westover"


def test_datasets_has_gold_entry():
    ids = {d["dataset_id"] for d in _rows(LBL / "datasets.csv")}
    assert "centaur_iiic_expert" in ids


def test_ingest_provenance_sidecar():
    p = (LBL / "external/centaur_iiic_goldpanel_raw"
         / "INGEST_PROVENANCE.json")
    j = json.loads(p.read_text())
    assert j["n_observations_added"] == 20_000
    assert j["taxonomy"]["collapse"] == {"bipd": "other", "birds": "other"}
    assert j["append_only"] is True


# ---- R1 / R2 / R4 / R5 / provenance ---------------------------------------

def test_R1_segment_labels_has_k7_column():
    with open(LBL / "segment_labels.csv") as f:
        hdr = next(csv.reader(f))
    assert "iiic_vote_other" in hdr      # the K=7 'other' task
    assert "iiic_vote_seizure" in hdr


def test_R2_consolidated_not_shipped():
    assert not (LBL / "consolidated").exists()


def test_R4_legacy_sigma_archived_not_in_data():
    assert (ROOT / "archive" / "Sigma_l_fitted.npy").is_file()
    assert not list((ROOT / "data").rglob("Sigma_l_fitted.npy"))


def test_R5_gold_ingest_collision_free():
    ids = Counter(d["rater_id"] for d in _rows(LBL / "raters.csv"))
    for rid in ("97", "99000001", "99000002", "99000003"):
        assert ids[rid] == 1, f"rater_id {rid} not unique ({ids[rid]})"


def test_R5_numeric_id_join_is_unsafe_known_hazard():
    """LOCKED KNOWN HAZARD (Phase 5 must resolve): engine_inputs/sdt_fits
    `rater_id` is a domain-local index whose values numerically COINCIDE
    with PI raters.csv rater_ids but denote different entities. Proven by:
    looking up each sdt_fits rater_id in PI raters.csv by numeric id yields
    a canonical_name that disagrees with sdt_fits.rater_name for the large
    majority of rows. => engine_inputs must be joined by NAME, never id.
    If this ever fails, identities were reconciled (likely Phase 5) —
    update this test and DATA_PROVENANCE R5."""
    pi_name = {d["rater_id"]: d["canonical_name"]
               for d in _rows(LBL / "raters.csv")}
    rows = list(_rows(ROOT / "data/engine_inputs/sdt_fits.csv"))
    checked = mismatch = 0
    for d in rows:
        rid, nm = d["rater_id"], d["rater_name"]
        if rid in pi_name:
            checked += 1
            if pi_name[rid] != nm:
                mismatch += 1
    assert checked > 0
    frac = mismatch / checked
    assert frac > 0.5, (
        f"numeric-id join now mostly agrees ({1-frac:.2%}) — engine_inputs "
        "identities may be reconciled; update R5 docs/tests")


def test_carry_forwards_present():
    assert (ROOT / "data/SENSITIVE.md").is_file()
    if MINE_SRC.exists():
        a = hashlib.md5((ROOT / "data/SENSITIVE.md").read_bytes()).hexdigest()
        b = hashlib.md5(
            (MINE_SRC / "data/SENSITIVE.md").read_bytes()).hexdigest()
        assert a == b, "SENSITIVE.md is not MINE's authoritative copy"
    ei = ROOT / "data/engine_inputs"
    for f in ("sdt_fits.csv", "cross_domain_rater_matrix.csv",
              "MANIFEST.json", "README.md"):
        assert (ei / f).is_file()
    prov = LBL / "external/centaur_iiic_goldpanel_raw"
    for f in ("centaur_iiic_expert_labels.xlsx",
              "centaur_iiic_novice_labels.csv",
              "centaur_iiic_novice_survey_results.csv",
              "AUDIT_centaur_iiic_novice_expert.md"):
        assert (prov / f).is_file()


def test_native7_recoverable_from_raw_xlsx():
    """Native 7-class must remain recoverable (birds/bipd present in the
    untouched raw provenance xlsx)."""
    import openpyxl
    p = (LBL / "external/centaur_iiic_goldpanel_raw"
         / "centaur_iiic_expert_labels.xlsx")
    ws = openpyxl.load_workbook(p, read_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    next(it)
    vals = set()
    for r in it:
        vals.update(str(x).strip().strip("'").lower() for x in r[2:6])
    assert {"birds", "bipd"} <= vals, "native-7 classes lost from raw xlsx"
