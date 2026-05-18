"""Build the unified label tables in data/labels/.

Inputs:
  - ideal-test/data/h5_v2_public/SN1_combined_v2.h5     (canonical spike corpus)
  - ideal-test/data/sparcnet50K_expert_labels_deIDed.csv (IIIC option-B raw)
  - pd-rda-profiler/data/labels/{segments,labels,segment_labels,annotations}.csv
  - data/labels/raters_aliases_draft.yaml               (alias resolution)

Outputs (overwrites):
  - data/labels/raters.csv          (one row per canonical rater)
  - data/labels/segments.csv        (one row per unique EEG segment, with S3 URI)
  - data/labels/labels.csv          (long-form: seg_id, rater_id, label_type, value, source_dataset)
  - data/labels/segment_labels.csv  (wide-form aggregates per segment)
  - data/labels/datasets.csv        (registry of source datasets)

EEG-file pointers go in segments.s3_uri (best guess; provenance documented).
"""
from __future__ import annotations
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


# Phase 1 (unified merge): ROOT made repo-relative and the module-level
# mkdir guarded so `build_segment_labels` imports side-effect-free for R1
# regeneration. IDEALTEST/PROFILER are only touched by main()/raw readers
# (not run here); kept env-overridable for any full rebuild. (Also
# pre-satisfies Phase-4 path-parameterization for this carried script.)
import os

ROOT = Path(os.environ.get(
    "ILAE_REPO_ROOT", Path(__file__).resolve().parents[1]))
IDEALTEST = Path(os.environ.get("ILAE_IDEALTEST",
                                "/Users/mwestover/GithubRepos/ideal-test"))
PROFILER = Path(os.environ.get("ILAE_PROFILER",
                               "/Users/mwestover/GithubRepos/pd-rda-profiler"))
OUT = ROOT / "data/labels"
try:  # only meaningful for a full rebuild; never on import for R1
    OUT.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# S3 paths verified from upstream documentation + GROND pipeline scripts
# (pd-rda-profiler/code/data_management/*.py):
#
# 1) Spike segments (sn1_combined_v2): one h5 file holding ALL segments.
#    s3://bdsp-opendata-restricted/spike-test/v2/SN1_combined_v2.h5
#    Segments are at /eeg/signals[s3_h5_index].
#
# 2) IIIC raw monopolar segments (the BIDS-named files):
#    s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/{SUBTYPE}/
#         segments_raw/{BIDS_filename}.mat
#    where SUBTYPE is one of {LPD, GPD, LRDA, GRDA, SEIZURE, IIIC}
#    (IIIC subfolder = "iiic_other" catch-all).
#    Plus the BIPD overflow:
#    s3://bdsp-opendata-credentialed/morgoth2/data/internal_dataset/BIPD/{BIDS_filename}.mat
#
# 3) IIIC pre-processed bipolar segments (the curated GROND subset, 9,857 files):
#    s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/{profiler_mat_file}
#    where profiler_mat_file is the renamed "{mrn}_seg{idx}.mat" form.
#    Only available for pd-rda-profiler segments (subset of segments where
#    GROND processed the monopolar source into a clean bipolar version).
SPIKE_H5_S3 = "s3://bdsp-opendata-restricted/spike-test/v2/SN1_combined_v2.h5"
S3_MORGOTH1 = "s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset"
S3_MORGOTH2 = "s3://bdsp-opendata-credentialed/morgoth2/data/internal_dataset"
IIIC_FREQ3_BIPOLAR_PREFIX = "s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/"

# Map our internal subtype label → morgoth subfolder
SUBTYPE_TO_S3_FOLDER = {
    "lpd":     (S3_MORGOTH1, "LPD",     "segments_raw"),
    "gpd":     (S3_MORGOTH1, "GPD",     "segments_raw"),
    "lrda":    (S3_MORGOTH1, "LRDA",    "segments_raw"),
    "grda":    (S3_MORGOTH1, "GRDA",    "segments_raw"),
    "seizure": (S3_MORGOTH1, "SEIZURE", "segments_raw"),
    "other":   (S3_MORGOTH1, "IIIC",    "segments_raw"),  # iiic_other catch-all
    "bipd":    (S3_MORGOTH2, "BIPD",    ""),               # morgoth2 has no /segments_raw/
}


def build_morgoth_s3_uri(bids_filename, subtype):
    """Construct s3:// URI for an IIIC raw monopolar segment."""
    if not bids_filename or pd.isna(bids_filename):
        return None
    subtype = (subtype or "").lower()
    if subtype not in SUBTYPE_TO_S3_FOLDER:
        return None
    base, folder, subdir = SUBTYPE_TO_S3_FOLDER[subtype]
    fname = bids_filename if bids_filename.endswith(".mat") else bids_filename + ".mat"
    if subdir:
        return f"{base}/{folder}/{subdir}/{fname}"
    return f"{base}/{folder}/{fname}"


def sparcnet_subtype_from_votes(row):
    """Determine IIIC class from sparcnet50K vote counts (argmax)."""
    votes = {
        "seizure": row.get("total_sz", 0) or 0,
        "lpd":     row.get("total_lpd", 0) or 0,
        "gpd":     row.get("total_gpd", 0) or 0,
        "lrda":    row.get("total_lrda", 0) or 0,
        "grda":    row.get("total_grda", 0) or 0,
        "other":   row.get("Total-other", 0) or 0,
    }
    if sum(votes.values()) == 0:
        return None
    return max(votes, key=votes.get)

NONPERSON_NAMES = {"IIIC_crowd", "corrected", "pending"}


# ───────────── helpers ─────────────

def strip_quotes(s):
    s = str(s).strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def norm_name(name):
    """Same canonicalization used to build raters_aliases_draft.yaml."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKC", str(name).strip())
    s = strip_quotes(s)
    s = re.sub(r"[.,]", "", s)
    s = s.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def name_key(name):
    return norm_name(name).lower()


def load_alias_map():
    """Read raters_aliases_draft.yaml and return alias_key -> canonical_name."""
    yaml_path = OUT / "raters_aliases_draft.yaml"
    with yaml_path.open() as f:
        entries = yaml.safe_load(f)
    alias_to_canon = {}
    for entry in entries:
        canon = entry["canonical"]
        alias_to_canon[name_key(canon)] = canon
        for v in entry.get("variants", []):
            alias_to_canon[name_key(v)] = canon
    return alias_to_canon


def canon_name(name, alias_map):
    """Return canonical name for `name`, falling back to a tidied version."""
    k = name_key(name)
    if k in alias_map:
        return alias_map[k]
    return strip_quotes(name).strip()


# ───────────── readers ─────────────

def read_spike_h5(path):
    """Open the v2 h5; return segment df, rater df, and long-form (seg_idx, rater_idx, value)."""
    with h5py.File(path, "r") as f:
        n_seg, n_rat = f["/scores/matrix"].shape
        seg = pd.DataFrame({
            "h5_idx": np.arange(n_seg),
            "file_key":    [s.decode() if isinstance(s, bytes) else s for s in f["/segments/file_key"][:]],
            "seg_key":     [s.decode() if isinstance(s, bytes) else s for s in f["/segments/seg_key"][:]],
            "pat_key":     [s.decode() if isinstance(s, bytes) else s for s in f["/segments/pat_key"][:]],
            "source":      [s.decode() if isinstance(s, bytes) else s for s in f["/segments/source"][:]],
            "pat_age":     f["/segments/pat_age"][:],
            "sex":         [s.decode() if isinstance(s, bytes) else s for s in f["/segments/sex"][:]],
        })
        rat = pd.DataFrame({
            "h5_idx":         np.arange(n_rat),
            "name":           [s.decode() if isinstance(s, bytes) else s for s in f["/experts/name"][:]],
            "group":          [s.decode() if isinstance(s, bytes) else s for s in f["/experts/group"][:]],
            "affiliation":    [s.decode() if isinstance(s, bytes) else s for s in f["/experts/affiliation"][:]],
            "years_eeg":      f["/experts/years_eeg"][:],
            "neurologist":    f["/experts/neurologist"][:].astype(bool),
            "epileptologist": f["/experts/epileptologist"][:].astype(bool),
            "board_certified":f["/experts/board_certified"][:].astype(bool),
            "read_eeg_flag":  f["/experts/read_eeg"][:].astype(bool),
            "rct_arm":        [s.decode() if isinstance(s, bytes) else s for s in f["/experts/rct_arm"][:]],
            "trial_pool":     [s.decode() if isinstance(s, bytes) else s for s in f["/experts/trial_pool"][:]],
        })
        scores = f["/scores/matrix"][:]  # (n_seg, n_rat) int8
    # Long form: only entries that are 0 or 1 (skip -1 unscored)
    seg_idx, rat_idx = np.where(scores >= 0)
    long = pd.DataFrame({
        "h5_seg_idx": seg_idx,
        "h5_rat_idx": rat_idx,
        "value": scores[seg_idx, rat_idx].astype(np.int8),
    })
    print(f"  spike h5: {n_seg} segments × {n_rat} raters; {len(long):,} non-missing labels")
    return seg, rat, long


def read_sparcnet50k(path):
    """Read sparcnet50K wide CSV; return segments df + long-form (seg_idx, rater_name, vote, label_type)."""
    df = pd.read_csv(path)
    non_rater = ("file", "mrn", "old_token", "old_file", "cpd_center", "cpd_start", "cpd_end",
                 "Total-other", "total_sz", "total_lpd", "total_gpd", "total_lrda", "total_grda",
                 "sparcnet_split", "sn2_split")
    rater_cols = [c for c in df.columns if c not in non_rater]
    # Segments table
    seg = df[["file", "mrn", "old_token", "old_file", "cpd_center", "cpd_start", "cpd_end",
              "total_sz", "total_lpd", "total_gpd", "total_lrda", "total_grda", "Total-other",
              "sparcnet_split", "sn2_split"]].copy()
    seg.rename(columns={"file": "file_key", "mrn": "mrn",
                        "old_token": "old_token", "old_file": "old_file_recording",
                        "cpd_center": "window_center_s", "cpd_start": "window_start_s",
                        "cpd_end": "window_end_s",
                        "Total-other": "iiic_vote_other"}, inplace=True)
    seg["sparcnet_idx"] = np.arange(len(seg))
    # Long-form via stack (drops NaN cells automatically).
    # Encoding (verified): 0=other, 1=seizure, 2=lpd, 3=gpd, 4=lrda, 5=grda.
    votes = df[rater_cols].copy()
    votes.index = np.arange(len(votes))   # sparcnet_idx
    long = votes.stack().rename("vote_code").reset_index()
    long.columns = ["sparcnet_idx", "rater_col", "vote_code"]
    print(f"  sparcnet50K: {len(seg)} segments × {len(rater_cols)} raters; {len(long):,} votes")
    return seg, rater_cols, long


def read_profiler_labels():
    profiler_dir = PROFILER / "data/labels"
    segments  = pd.read_csv(profiler_dir / "segments.csv")
    labels    = pd.read_csv(profiler_dir / "labels.csv")
    seg_labels = pd.read_csv(profiler_dir / "segment_labels.csv")
    annotations = pd.read_csv(profiler_dir / "annotations.csv")
    print(f"  profiler segments: {len(segments)}; labels: {len(labels)}; "
          f"seg_labels: {len(seg_labels)}; annotations: {len(annotations)}")
    return segments, labels, seg_labels, annotations


def read_idealtest_expertise():
    """Roster of expert raters with metadata (idealtest)."""
    df = pd.read_csv(IDEALTEST / "data/raters_expertise_for_elijah.csv")
    return df


def read_idealtest_crowd_metadata():
    df = pd.read_csv(IDEALTEST / "data/combined_spike/crowd_users_metadata.csv")
    return df


# ───────────── build raters.csv ─────────────

def build_raters(spike_rat, sparcnet_rater_cols, profiler_labels, profiler_annot,
                 idealtest_expertise, crowd_meta, alias_map):
    """Collect every rater observed and produce canonical raters table."""
    # Aggregate per canonical name
    raters = defaultdict(lambda: {
        "aliases": set(),
        "group_obs": set(),
        "affiliation": None,
        "years_eeg": None,
        "neurologist": None,
        "epileptologist": None,
        "board_certified": None,
        "read_eeg_flag": None,
        "rct_arm": None,
        "trial_pool": None,
        "expertise_level": None,
        "n_segments_scored_roster": None,
        "is_person": True,
    })

    def update(canon, alias, **fields):
        r = raters[canon]
        if alias is not None:
            r["aliases"].add(alias)
        for k, v in fields.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, bytes):
                v = v.decode()
            v = strip_quotes(str(v)).strip() if isinstance(v, str) else v
            # Don't overwrite a non-empty value with empty
            if isinstance(r.get(k), str) and r[k] and not v:
                continue
            if k == "group_obs":
                r["group_obs"].add(v)
            else:
                r[k] = v

    # 1) Spike h5
    for _, row in spike_rat.iterrows():
        nm = row["name"]
        canon = canon_name(nm, alias_map)
        update(canon, alias=nm,
               group_obs=row["group"],
               affiliation=row["affiliation"],
               years_eeg=row["years_eeg"],
               neurologist=row["neurologist"],
               epileptologist=row["epileptologist"],
               board_certified=row["board_certified"],
               read_eeg_flag=row["read_eeg_flag"],
               rct_arm=row["rct_arm"],
               trial_pool=row["trial_pool"])

    # 2) Sparcnet50K rater columns
    for col in sparcnet_rater_cols:
        canon = canon_name(col, alias_map)
        update(canon, alias=col, group_obs="sparcnet50K")

    # 3) pd-rda-profiler labels.csv + annotations.csv (column 'rater')
    for r in profiler_labels["rater"].dropna().unique():
        canon = canon_name(r, alias_map)
        update(canon, alias=r, group_obs="profiler_iiic")
        if canon in NONPERSON_NAMES:
            raters[canon]["is_person"] = False
    for r in profiler_annot["rater"].dropna().unique():
        canon = canon_name(r, alias_map)
        update(canon, alias=r, group_obs="profiler_iiic")
        if canon in NONPERSON_NAMES:
            raters[canon]["is_person"] = False

    # 4) idealtest expertise roster
    for _, row in idealtest_expertise.iterrows():
        nm = row["rater_name"]
        canon = canon_name(nm, alias_map)
        update(canon, alias=nm,
               group_obs=row["group"],
               affiliation=row["affiliation"],
               years_eeg=row["years_eeg"],
               neurologist=row["neurologist"],
               epileptologist=row["epileptologist"],
               board_certified=row["board_certified"],
               read_eeg_flag=row["read_eeg_flag"],
               trial_pool=row["trial_pool"],
               rct_arm=row["rct_arm"],
               expertise_level=row["expertise_level"],
               n_segments_scored_roster=row.get("n_segments_scored"))

    # 5) crowd users metadata (Centaur) — these are user IDs (numeric handles)
    for _, row in crowd_meta.iterrows():
        uid = f"crowd_{row['user_ids']}"
        update(uid, alias=uid, group_obs="Crowd",
               neurologist=bool(row.get("neurologist")),
               epileptologist=bool(row.get("epileptologist")),
               read_eeg_flag=bool(row.get("read_eeg")))

    # Build final DataFrame
    out_rows = []
    for canon in sorted(raters):
        r = raters[canon]
        out_rows.append({
            "canonical_name": canon,
            "aliases": json.dumps(sorted(r["aliases"])),
            "groups":  json.dumps(sorted(r["group_obs"])),
            "expertise_level": r["expertise_level"],
            "affiliation": r["affiliation"],
            "years_eeg": r["years_eeg"],
            "neurologist": r["neurologist"],
            "epileptologist": r["epileptologist"],
            "board_certified": r["board_certified"],
            "read_eeg_flag": r["read_eeg_flag"],
            "rct_arm": r["rct_arm"],
            "trial_pool": r["trial_pool"],
            "n_segments_scored_roster": r["n_segments_scored_roster"],
            "is_person": r["is_person"],
        })
    df = pd.DataFrame(out_rows)
    df.insert(0, "rater_id", np.arange(len(df)))
    return df


# ───────────── build segments.csv ─────────────

def build_segments(spike_seg, sparcnet_seg, profiler_seg):
    """Combine three sources into one segments table with universal seg_id and S3 URI."""
    rows = []
    # 1) spike segments: all live inside SN1_combined_v2.h5 on S3.
    for _, r in spike_seg.iterrows():
        source = r["source"]  # 'sn1' / 'bonobo_only' / 'fabio_spikeed'
        rows.append({
            "source_dataset": f"sn1_combined_v2:{source}",
            "file_key": r["file_key"],
            "seg_key": r["seg_key"],
            "pat_key": r["pat_key"],
            "mrn": None,
            "old_token": None,
            "old_file_recording": None,
            "subtype": None,    # spike segments use binary spike/non-spike, no IIIC subtype
            "window_center_s": None,
            "window_start_s": None,
            "window_end_s": None,
            "pat_age": r["pat_age"] if not np.isnan(r["pat_age"]) else None,
            "sex": r["sex"] if r["sex"] else None,
            "h5_local_path": str(IDEALTEST / "data/h5_v2_public/SN1_combined_v2.h5"),
            "spike_h5_idx": int(r["h5_idx"]),
            "sparcnet_idx": None,
            "profiler_mat_file": None,
            "s3_uri": SPIKE_H5_S3,
            "s3_uri_bipolar": None,
            "s3_h5_index": int(r["h5_idx"]),  # slice /eeg/signals[h5_idx] in the h5
            "s3_uri_note": "spike segments are inside the h5; load + slice by s3_h5_index",
        })

    # 2) sparcnet50K IIIC segments: raw monopolar at morgoth1 (subfolder = vote argmax).
    for _, r in sparcnet_seg.iterrows():
        subtype = sparcnet_subtype_from_votes(r)
        s3_uri_raw = build_morgoth_s3_uri(r["file_key"], subtype) if subtype else None
        rows.append({
            "source_dataset": "sparcnet50K",
            "file_key": r["file_key"],
            "seg_key": None,
            "pat_key": None,
            "mrn": r["mrn"],
            "old_token": r["old_token"],
            "old_file_recording": r["old_file_recording"],
            "subtype": subtype,
            "window_center_s": r["window_center_s"],
            "window_start_s": r["window_start_s"],
            "window_end_s": r["window_end_s"],
            "pat_age": None,
            "sex": None,
            "h5_local_path": None,
            "spike_h5_idx": None,
            "sparcnet_idx": int(r["sparcnet_idx"]),
            "profiler_mat_file": None,
            "s3_uri": s3_uri_raw,
            "s3_uri_bipolar": None,        # no bipolar version for sparcnet-only segments
            "s3_h5_index": None,
            "s3_uri_note": None if s3_uri_raw else
                ("could not determine S3 subfolder: sparcnet50K segment has zero "
                 "votes across all IIIC classes (n_votes=0)"),
        })

    # 3) pd-rda-profiler segments: TWO S3 locations available:
    #    - raw monopolar: morgoth1/{SUBTYPE}/segments_raw/{BIDS_filename}
    #    - pre-processed bipolar: iiic-freq3/data/eeg/{mat_file}
    # The BIDS filename comes from `eeg_file` when present, else from `mat_file`
    # (for some rows the profiler stored the BIDS form directly in mat_file).
    for _, r in profiler_seg.iterrows():
        bids = r.get("eeg_file")
        if pd.isna(bids) or not bids:
            mf = r.get("mat_file", "")
            if isinstance(mf, str) and mf.startswith("sub-"):
                bids = mf
        s3_uri_raw = build_morgoth_s3_uri(bids, r.get("subtype"))
        s3_uri_bipolar = IIIC_FREQ3_BIPOLAR_PREFIX + r["mat_file"]
        rows.append({
            "source_dataset": "pd_rda_profiler",
            "file_key": r["mat_file"],
            "seg_key": None,
            "pat_key": r["patient_id"],
            "mrn": None,
            "old_token": None,
            "old_file_recording": r.get("eeg_file"),
            "subtype": r.get("subtype"),
            "window_center_s": None,
            "window_start_s": None,
            "window_end_s": r.get("duration_s"),
            "pat_age": None,
            "sex": None,
            "h5_local_path": None,
            "spike_h5_idx": None,
            "sparcnet_idx": None,
            "profiler_mat_file": r["mat_file"],
            "s3_uri": s3_uri_raw,
            "s3_uri_bipolar": s3_uri_bipolar,
            "s3_h5_index": None,
            "s3_uri_note": None if s3_uri_raw else
                f"unmapped subtype '{r.get('subtype')}' — no morgoth subfolder",
        })

    df = pd.DataFrame(rows)
    df.insert(0, "seg_id", np.arange(len(df)))
    return df


# ───────────── build labels.csv ─────────────

def build_labels(spike_long, spike_seg, spike_rat,
                 sparcnet_long, sparcnet_seg, sparcnet_rater_cols,
                 profiler_long_df, profiler_seg,
                 segments_df, raters_df, alias_map):
    """Long-form labels table with universal seg_id + rater_id."""
    canon_to_rater_id = dict(zip(raters_df["canonical_name"], raters_df["rater_id"]))

    rows = []

    # ── 1) Spike labels (h5 v2) ──
    # Map h5_idx → seg_id and h5_idx → rater_id
    spike_seg_to_segid = {h5_idx: segid for h5_idx, segid in
                          zip(segments_df.loc[segments_df["spike_h5_idx"].notna(), "spike_h5_idx"].astype(int),
                              segments_df.loc[segments_df["spike_h5_idx"].notna(), "seg_id"])}
    spike_rat_to_raterid = {}
    for _, rr in spike_rat.iterrows():
        canon = canon_name(rr["name"], alias_map)
        spike_rat_to_raterid[int(rr["h5_idx"])] = canon_to_rater_id[canon]

    n = len(spike_long)
    print(f"  building {n:,} spike label rows...")
    seg_ids = np.array([spike_seg_to_segid[i] for i in spike_long["h5_seg_idx"].values])
    rater_ids = np.array([spike_rat_to_raterid[i] for i in spike_long["h5_rat_idx"].values])
    rows.append(pd.DataFrame({
        "seg_id": seg_ids,
        "rater_id": rater_ids,
        "label_type": "spike",
        "value": spike_long["value"].values.astype(int).astype(str),
        "source_dataset": "sn1_combined_v2",
    }))

    # ── 2) Sparcnet50K IIIC labels ──
    # Map sparcnet_idx → seg_id
    sn_idx_to_segid = {int(idx): segid for idx, segid in
                       zip(segments_df.loc[segments_df["sparcnet_idx"].notna(), "sparcnet_idx"].astype(int),
                           segments_df.loc[segments_df["sparcnet_idx"].notna(), "seg_id"])}
    # Map rater_col → rater_id (via alias map)
    sn_col_to_raterid = {}
    for col in sparcnet_rater_cols:
        canon = canon_name(col, alias_map)
        sn_col_to_raterid[col] = canon_to_rater_id[canon]

    # Translate vote codes to label_type + value.
    # Verified by cross-checking against 'total_*' columns:
    #   0=other, 1=seizure, 2=lpd, 3=gpd, 4=lrda, 5=grda.
    print(f"  building {len(sparcnet_long):,} sparcnet IIIC label rows...")
    sn_label_rows = []
    code_to_class = {0: "other", 1: "seizure", 2: "lpd", 3: "gpd", 4: "lrda", 5: "grda"}
    skipped = 0
    for _, row in sparcnet_long.iterrows():
        sid = sn_idx_to_segid.get(int(row["sparcnet_idx"]))
        rid = sn_col_to_raterid.get(row["rater_col"])
        if sid is None or rid is None:
            skipped += 1
            continue
        code = int(row["vote_code"]) if not pd.isna(row["vote_code"]) else None
        if code is None:
            continue
        cls = code_to_class.get(code)
        sn_label_rows.append({"seg_id": sid, "rater_id": rid,
                              "label_type": "pattern_class",
                              "value": cls if cls else str(code),
                              "source_dataset": "sparcnet50K"})
    if skipped:
        print(f"    sparcnet: skipped {skipped} rows w/ missing seg or rater mapping")
    rows.append(pd.DataFrame(sn_label_rows))

    # ── 3) pd-rda-profiler labels (long-form already) ──
    print(f"  building {len(profiler_long_df):,} profiler IIIC label rows...")
    profiler_mat_to_segid = {mf: sid for mf, sid in
                              zip(segments_df.loc[segments_df["profiler_mat_file"].notna(), "profiler_mat_file"],
                                  segments_df.loc[segments_df["profiler_mat_file"].notna(), "seg_id"])}
    prof_rows = []
    skipped = 0
    for _, row in profiler_long_df.iterrows():
        # join profiler_labels.csv (segment_id, mat_file, rater, label_type, value, ...) to segments.csv
        sid = profiler_mat_to_segid.get(row["mat_file"])
        rid = canon_to_rater_id.get(canon_name(row["rater"], alias_map))
        if sid is None or rid is None:
            skipped += 1
            continue
        prof_rows.append({"seg_id": sid, "rater_id": rid,
                          "label_type": row["label_type"],
                          "value": str(row["value"]),
                          "source_dataset": "pd_rda_profiler"})
    if skipped:
        print(f"    profiler: skipped {skipped} rows w/ missing seg or rater mapping")
    rows.append(pd.DataFrame(prof_rows))

    out = pd.concat(rows, ignore_index=True)
    return out


def build_segment_labels(labels_df, segments_df):
    """Wide-form per-segment aggregates."""
    rows = []
    by_seg = labels_df.groupby("seg_id")
    for seg_id, grp in by_seg:
        agg = {"seg_id": seg_id}
        # spike votes
        spk = grp[grp["label_type"] == "spike"]
        if len(spk):
            v = pd.to_numeric(spk["value"], errors="coerce").dropna()
            agg["spike_n_votes"] = len(v)
            agg["spike_pos_votes"] = int((v == 1).sum())
            agg["spike_pos_frac"] = float((v == 1).mean()) if len(v) else None
        # IIIC pattern_class votes
        pc = grp[grp["label_type"] == "pattern_class"]
        if len(pc):
            agg["iiic_n_votes"] = len(pc)
            for cls in ["seizure", "lpd", "gpd", "lrda", "grda", "other"]:
                agg[f"iiic_vote_{cls}"] = int((pc["value"] == cls).sum())
            counts = pc["value"].value_counts()
            agg["iiic_plurality"] = counts.idxmax() if len(counts) else None
            agg["iiic_plurality_frac"] = float(counts.iloc[0] / counts.sum()) if len(counts) else None
        rows.append(agg)
    df = pd.DataFrame(rows).set_index("seg_id")
    # left-join onto segments
    merged = segments_df.set_index("seg_id").join(df).reset_index()
    return merged


def build_datasets():
    return pd.DataFrame([
        {"dataset_id": "sn1_combined_v2",
         "description": "SN1 + bonobo13K + Centaur + SpikeEd combined spike corpus (v2).",
         "source_path": str(IDEALTEST / "data/h5_v2_public/SN1_combined_v2.h5"),
         "n_segments_in_source": 20521,
         "n_raters_in_source": 2574,
         "label_types": "spike",
         "paper": "Westover, Nascimento, Jing — spike skill paper; SpikeEd RCT (Nascimento 2024 Epileptic Disord)"},
        {"dataset_id": "sparcnet50K",
         "description": "Raw sparcnet IIIC expert annotations (50,478 cases × 124 raters).",
         "source_path": str(IDEALTEST / "data/sparcnet50K_expert_labels_deIDed.csv"),
         "n_segments_in_source": 50478,
         "n_raters_in_source": 124,
         "label_types": "pattern_class",
         "paper": "SPaRCNet IIIC annotations"},
        {"dataset_id": "pd_rda_profiler",
         "description": "Curated pd-rda-profiler IIIC labels with frequency / laterality / spatial labels.",
         "source_path": str(PROFILER / "data/labels/labels.csv"),
         "n_segments_in_source": 13557,
         "n_raters_in_source": 9,
         "label_types": "pattern_class,frequency_hz,laterality,spatial_channels,spatial_extent,discharge_times,wave_times",
         "paper": "pd-rda-profiler internal annotations"},
    ])


# ───────────── main ─────────────

def main():
    print("Loading alias map...")
    alias_map = load_alias_map()
    print(f"  {len(alias_map):,} alias entries → canonical")

    print("Reading spike h5 v2...")
    spike_seg, spike_rat, spike_long = read_spike_h5(IDEALTEST / "data/h5_v2_public/SN1_combined_v2.h5")

    print("Reading sparcnet50K IIIC...")
    sparcnet_seg, sparcnet_rater_cols, sparcnet_long = read_sparcnet50k(IDEALTEST / "data/sparcnet50K_expert_labels_deIDed.csv")

    print("Reading pd-rda-profiler...")
    profiler_seg, profiler_labels, profiler_seglabels, profiler_annot = read_profiler_labels()

    print("Reading idealtest rater expertise roster + crowd metadata...")
    idealtest_expertise = read_idealtest_expertise()
    crowd_meta = read_idealtest_crowd_metadata()

    print()
    print("Building raters.csv...")
    raters_df = build_raters(spike_rat, sparcnet_rater_cols, profiler_labels, profiler_annot,
                              idealtest_expertise, crowd_meta, alias_map)
    raters_df.to_csv(OUT / "raters.csv", index=False)
    print(f"  → {OUT/'raters.csv'}: {len(raters_df):,} canonical raters")

    print("Building segments.csv...")
    segments_df = build_segments(spike_seg, sparcnet_seg, profiler_seg)
    segments_df.to_csv(OUT / "segments.csv", index=False)
    print(f"  → {OUT/'segments.csv'}: {len(segments_df):,} unique segments")

    print("Building labels.csv (long-form; this is the big one)...")
    labels_df = build_labels(spike_long, spike_seg, spike_rat,
                              sparcnet_long, sparcnet_seg, sparcnet_rater_cols,
                              profiler_labels, profiler_seg,
                              segments_df, raters_df, alias_map)
    labels_df.to_csv(OUT / "labels.csv", index=False)
    print(f"  → {OUT/'labels.csv'}: {len(labels_df):,} long-form labels")

    print("Building segment_labels.csv (wide aggregates)...")
    seg_labels_df = build_segment_labels(labels_df, segments_df)
    seg_labels_df.to_csv(OUT / "segment_labels.csv", index=False)
    print(f"  → {OUT/'segment_labels.csv'}: {len(seg_labels_df):,} rows")

    print("Building datasets.csv...")
    datasets_df = build_datasets()
    datasets_df.to_csv(OUT / "datasets.csv", index=False)
    print(f"  → {OUT/'datasets.csv'}: {len(datasets_df)} datasets")

    print("\nDone.")


if __name__ == "__main__":
    main()
