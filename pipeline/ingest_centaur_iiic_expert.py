"""Ingest the Centaur-IIIC 4-expert gold panel into the canonical lake.

Phase 1 / Centaur-gold decision (2026-05-18): the 4-expert gold panel
(`centaur_iiic_expert_labels.xlsx`, 5000 cases x {mbw,cal,matt,tianyu})
exists ONLY in the methodology repo's raw files and is absent from PI's
canonical labels.csv. It is the gold reference for Paper-1 Centaur-IIIC
validation, so it is ingested as a new source_dataset='centaur_iiic_expert'
(modeled on PI's ingest_* convention), with the raw xlsx kept verbatim as
provenance under external/centaur_iiic_goldpanel_raw/.

Design guarantees:
  * Repo-relative paths only (no hardcoded /Users/... — pre-satisfies the
    Phase-4 path-parameterization requirement for this new script).
  * APPEND-ONLY to labels.csv/raters.csv/datasets.csv: every pre-existing
    row is preserved byte-for-byte (nothing lost). Timestamped *.bak.*.csv
    written first (gitignored) for reversibility.
  * IDEMPOTENT: aborts if source_dataset 'centaur_iiic_expert' already
    present.
  * Joins are asserted, not assumed: every gold case_id must crosswalk to
    a seg_id that (a) exists in segments.csv and (b) already carries the
    aligned centaur_2025_iiic novice reads.
  * Taxonomy: gold is 7-class {seizure,lpd,gpd,lrda,grda,bipd,birds};
    canonical pattern_class is 6-class {...,other}. Per AUDIT section 6
    ("the mapping is clean: collapse {bipd,birds}->other -> exact 6-class
    compatibility"), bipd/birds -> other. The native 7-class labels remain
    losslessly recoverable from the carried raw xlsx (Paper-2 / native-7).
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[1]
LBL = REPO / "data" / "labels"
RAW = LBL / "external" / "centaur_iiic_goldpanel_raw" / "centaur_iiic_expert_labels.xlsx"
CROSSWALK = LBL / "external" / "centaur_2025" / "case_id_to_seg_id.csv"

F_LABELS = LBL / "labels.csv"
F_RATERS = LBL / "raters.csv"
F_DATASETS = LBL / "datasets.csv"

SOURCE_DATASET = "centaur_iiic_expert"
GOLD_7 = {"seizure", "lpd", "gpd", "lrda", "grda", "bipd", "birds"}
CANON_6 = {"seizure", "lpd", "gpd", "lrda", "grda", "other"}
COLLAPSE = {"bipd": "other", "birds": "other"}  # AUDIT section 6

EXPERTS = ["mbw", "cal", "matt", "tianyu"]
MBW_RID = "97"  # existing canonical rater (M. Brandon Westover)
# Collision-free block (verified Phase 1: max existing rater_id 10,620,815;
# PI Centaur block is 10_000_000-10_999_999; nothing >= 99_000_000):
GOLD_RID = {"cal": "99000001", "matt": "99000002", "tianyu": "99000003"}

RATERS_COLS = ["rater_id", "canonical_name", "aliases", "groups",
               "expertise_level", "affiliation", "years_eeg", "neurologist",
               "epileptologist", "board_certified", "read_eeg_flag",
               "rct_arm", "trial_pool", "n_segments_scored_roster",
               "is_person"]


def _die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _read_csv_rows(path: Path):
    with open(path) as f:
        r = csv.reader(f)
        return next(r), [row for row in r]


def main() -> None:
    print("=== ingest Centaur-IIIC 4-expert gold panel ===", flush=True)
    for p in (RAW, CROSSWALK, F_LABELS, F_RATERS, F_DATASETS):
        if not p.exists():
            _die(f"missing required input: {p}")

    # ---- idempotency guard -------------------------------------------------
    with open(F_LABELS) as f:
        rd = csv.DictReader(f)
        for row in rd:
            if row["source_dataset"] == SOURCE_DATASET:
                _die(f"'{SOURCE_DATASET}' already present in labels.csv "
                     "(already ingested) - nothing to do.")

    # ---- crosswalk centaur_case_id -> seg_id ------------------------------
    cw = {}
    with open(CROSSWALK) as f:
        for d in csv.DictReader(f):
            cw[str(d["centaur_case_id"])] = int(d["new_seg_id"])
    print(f"  crosswalk: {len(cw):,} centaur_case_id -> seg_id")

    # ---- gold xlsx --------------------------------------------------------
    wb = openpyxl.load_workbook(RAW, read_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    expected_hdr = ["case_id", "origin", "label_mbw", "label_cal",
                    "label_matt", "label_tianyu"]
    if hdr != expected_hdr:
        _die(f"unexpected gold xlsx header {hdr} != {expected_hdr}")
    gold = [r for r in it]
    print(f"  gold xlsx: {len(gold):,} cases x 4 experts")

    # ---- segments.csv membership + centaur_2025_iiic alignment ------------
    seg_ids = set()
    with open(LBL / "segments.csv") as f:
        rd = csv.DictReader(f)
        idc = rd.fieldnames[0]
        for d in rd:
            seg_ids.add(int(d[idc]))
    aligned = set()
    with open(F_LABELS) as f:
        for d in csv.DictReader(f):
            if d["source_dataset"] == "centaur_2025_iiic":
                aligned.add(int(d["seg_id"]))

    # ---- build the 20,000 expert observations -----------------------------
    new_rows = []
    val_dist = {}
    collapsed = 0
    missing_cw = missing_seg = unaligned = 0
    for row in gold:
        case_id = str(row[0])
        if case_id not in cw:
            missing_cw += 1
            continue
        seg = cw[case_id]
        if seg not in seg_ids:
            missing_seg += 1
            continue
        if seg not in aligned:
            unaligned += 1
            continue
        labels = dict(zip(["mbw", "cal", "matt", "tianyu"], row[2:6]))
        for ex in EXPERTS:
            raw_v = str(labels[ex]).strip().strip("'").strip('"').lower()
            if raw_v not in GOLD_7:
                _die(f"case {case_id} expert {ex}: value {raw_v!r} "
                     f"not in 7-class set {sorted(GOLD_7)}")
            v6 = COLLAPSE.get(raw_v, raw_v)
            if raw_v in COLLAPSE:
                collapsed += 1
            assert v6 in CANON_6, v6
            rid = MBW_RID if ex == "mbw" else GOLD_RID[ex]
            new_rows.append([str(seg), rid, "pattern_class", v6,
                             SOURCE_DATASET])
            val_dist[v6] = val_dist.get(v6, 0) + 1

    for nm, cnt in (("crosswalk-missing", missing_cw),
                    ("segment-missing", missing_seg),
                    ("unaligned (no centaur_2025_iiic reads)", unaligned)):
        if cnt:
            _die(f"{cnt} gold cases {nm} - join not clean, refusing to "
                 "ingest a partial/unaligned panel.")
    expected = len(gold) * len(EXPERTS)
    if len(new_rows) != expected:
        _die(f"row count {len(new_rows)} != expected {expected}")
    print(f"  built {len(new_rows):,} expert observations "
          f"({len(gold)} cases x 4); {collapsed:,} bipd/birds -> other")
    print(f"  6-class value distribution: {val_dist}")

    # ---- backups (reversibility) ------------------------------------------
    ts = time.strftime("%Y%m%dT%H%M%S")
    for p in (F_LABELS, F_RATERS, F_DATASETS):
        shutil.copy2(p, p.with_suffix(f".bak.{ts}.csv"))
    print(f"  backups written (*.bak.{ts}.csv)")

    # ---- append to labels.csv (preserve every existing row) ---------------
    with open(F_LABELS, "rb") as f:
        f.seek(-1, 2)
        needs_nl = f.read(1) != b"\n"
    with open(F_LABELS, "a", newline="") as f:
        if needs_nl:
            f.write("\n")
        w = csv.writer(f)
        w.writerows(new_rows)

    # ---- append 3 new rater rows (cal/matt/tianyu); mbw=97 reused ---------
    new_raters = []
    for ex in ("cal", "matt", "tianyu"):
        rec = {c: "" for c in RATERS_COLS}
        rec.update({
            "rater_id": GOLD_RID[ex],
            "canonical_name": f"Centaur-IIIC gold expert: {ex}",
            "aliases": json.dumps([f"centaur_expert:{ex}"]),
            "groups": json.dumps([SOURCE_DATASET]),
            "expertise_level": "expert",
            "neurologist": "False", "epileptologist": "False",
            "board_certified": "False", "read_eeg_flag": "False",
            "is_person": "True",
        })
        new_raters.append([rec[c] for c in RATERS_COLS])
    with open(F_RATERS, "rb") as f:
        f.seek(-1, 2)
        needs_nl = f.read(1) != b"\n"
    with open(F_RATERS, "a", newline="") as f:
        if needs_nl:
            f.write("\n")
        csv.writer(f).writerows(new_raters)

    # ---- append datasets.csv row ------------------------------------------
    ds_row = [
        SOURCE_DATASET,
        "Centaur-IIIC 4-expert gold panel (mbw/cal/matt/tianyu), 5000 IIIC "
        "cases, complete (no missingness). 7-class native; ingested with "
        "{bipd,birds}->other per AUDIT section 6. Native-7 recoverable from "
        "external/centaur_iiic_goldpanel_raw/centaur_iiic_expert_labels.xlsx.",
        "data/labels/external/centaur_iiic_goldpanel_raw/"
        "centaur_iiic_expert_labels.xlsx",
        str(len(gold)), str(len(EXPERTS)), "pattern_class",
        "Centaur-IIIC gold panel (Paper-1 validation cohort); see "
        "AUDIT_centaur_iiic_novice_expert.md",
    ]
    with open(F_DATASETS, "rb") as f:
        f.seek(-1, 2)
        needs_nl = f.read(1) != b"\n"
    with open(F_DATASETS, "a", newline="") as f:
        if needs_nl:
            f.write("\n")
        csv.writer(f).writerow(ds_row)

    # ---- provenance sidecar ----------------------------------------------
    prov = {
        "ingested": ts,
        "source_dataset": SOURCE_DATASET,
        "n_cases": len(gold),
        "n_experts": len(EXPERTS),
        "n_observations_added": len(new_rows),
        "rater_id_assignment": {"mbw": MBW_RID, **GOLD_RID},
        "taxonomy": {
            "gold_native": "7-class {seizure,lpd,gpd,lrda,grda,bipd,birds}",
            "canonical": "6-class pattern_class",
            "collapse": COLLAPSE,
            "n_collapsed_to_other": collapsed,
            "native_7_recoverable_from":
                "external/centaur_iiic_goldpanel_raw/"
                "centaur_iiic_expert_labels.xlsx",
            "rationale": "AUDIT_centaur_iiic_novice_expert.md section 6",
        },
        "join": "case_id -> external/centaur_2025/case_id_to_seg_id.csv "
                "-> seg_id; asserted in segments.csv AND aligned with "
                "centaur_2025_iiic novice reads",
        "value_distribution_6class": val_dist,
        "append_only": True,
        "backups": f"*.bak.{ts}.csv",
    }
    prov_path = (LBL / "external" / "centaur_iiic_goldpanel_raw" /
                 "INGEST_PROVENANCE.json")
    prov_path.write_text(json.dumps(prov, indent=2))
    print(f"  provenance -> {prov_path.relative_to(REPO)}")
    print("  DONE (append-only; originals preserved; reversible via "
          f"*.bak.{ts}.csv)")


if __name__ == "__main__":
    main()
