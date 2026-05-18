"""Phase-3 data adapter: unified labels.csv -> reference-schema prepared CSVs.

This is the ONLY new glue in the reference-faithful calibration pipeline.
It does NOT touch any numerical/statistical logic. Its sole job is to
translate the unified corpus (`data/labels/labels.csv` + `raters.csv`) into
the exact long-format schema that the *verbatim-carried* reference scripts
(`pipeline/reference_calibration/fit_main_effects.py` and
`fit_sdt_per_domain.py`) consume:

    data/prepared/{name}.csv         cols: case_id,rater_id,Y
                                     case_id, rater_id are DENSE 0-based
                                     contiguous ints; Y in {0,1}
    data/prepared/{name}_meta.json   {"rater_index_to_name": [...canonical
                                     names by dense rater idx...], plus
                                     provenance: n_obs, source label filter}
    data/prepared/{name}_rater_ids.csv  traceability:
                                     dense_rater_id, unified_rater_id,
                                     canonical_name, expertise_level

Task definitions (one-vs-rest binary, EXACT per Phase-3 spec):

  combined_spike : ALL rows where label_type=='spike' (nothing dropped).
                   value in {'0','1'} -> Y=int(value) (sn1 binary
                   spike-detection cohort). The unified spike corpus also
                   carries Centaur-2025-IED spike-SUBTYPE labels
                   ('ied','vertex wave','posts','wickets','bets','other');
                   for those, Y = 1 iff value=='ied' else 0 — the
                   clinically meaningful IED-vs-benign-variant binary spike
                   contrast (decision 2026-05-18; folds the 167,503-row
                   Centaur-IED cohort into the spike task). DATA-ENTRY-POINT
                   definition only — recorded in meta provenance; no
                   methodology altered.
  sparcnet_sz    : label_type=='pattern_class'; Y = 1 if value=='seizure' else 0.
  sparcnet_lpd   : Y = 1 if value=='lpd'  else 0.
  sparcnet_gpd   : Y = 1 if value=='gpd'  else 0.
  sparcnet_lrda  : Y = 1 if value=='lrda' else 0.
  sparcnet_grda  : Y = 1 if value=='grda' else 0.
  sparcnet_iic   : Y = 1 if value=='other' else 0.

Dense id construction: within each task subset, seg_id and rater_id are
remapped to 0..N-1 in STABLE SORTED order of the original unified id, so
the mapping is deterministic and reproducible. rater_index_to_name is the
unified `canonical_name` (NOT a numeric id) — the R5-safe key required by
the carried run_youden_calibration.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Task -> (label_type, positive-class spec)
#   spec == "__spike_ied__" -> combined_spike: value in {0,1} → Y=int(value)
#                              (sn1 binary spike-detection); else (Centaur
#                              spike-subtype rows) → Y = 1 if value=='ied'
#                              else 0 (IED vs benign variant: vertex wave,
#                              POSTS, wickets, BETS, other). Decision
#                              2026-05-18: fold the centaur_2025_ied cohort
#                              in as a clinically meaningful binary spike
#                              contrast (nothing lost).
#   spec == <class>         -> Y = 1 if value == <class> else 0
TASKS: dict[str, tuple[str, str]] = {
    "combined_spike": ("spike", "__spike_ied__"),
    "sparcnet_sz": ("pattern_class", "seizure"),
    "sparcnet_lpd": ("pattern_class", "lpd"),
    "sparcnet_gpd": ("pattern_class", "gpd"),
    "sparcnet_lrda": ("pattern_class", "lrda"),
    "sparcnet_grda": ("pattern_class", "grda"),
    "sparcnet_iic": ("pattern_class", "other"),
}

ALL_TASKS = list(TASKS.keys())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_task(
    name: str,
    labels: pd.DataFrame,
    raters: pd.DataFrame,
    prepared_dir: Path,
    labels_sha256: str | None = None,
) -> dict:
    """Build {name}.csv, {name}_meta.json, {name}_rater_ids.csv.

    Returns a per-task summary dict.
    """
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; valid: {ALL_TASKS}")
    label_type, spec = TASKS[name]

    sub = labels[labels["label_type"] == label_type].copy()
    n_raw = len(sub)

    if spec == "__spike_ied__":
        # combined_spike (decision 2026-05-18): keep ALL spike rows.
        #  - value in {'0','1'}  -> Y = int(value)  (sn1 binary
        #    spike-detection cohort, ~964,206 rows)
        #  - else (Centaur-2025-IED spike-SUBTYPE rows: ied / vertex wave /
        #    posts / wickets / bets / other, ~167,503) -> Y = 1 iff the
        #    label is an IED, else 0 (IED vs benign variant — a clinically
        #    meaningful binary spike contrast). Nothing dropped.
        val = sub["value"].astype(str).str.strip()
        is_bin = val.isin(["0", "1"])
        Y = np.where(
            is_bin,
            np.where(val == "1", 1, 0),
            (val.str.lower() == "ied").astype(int),
        ).astype(np.int64)
        sub = sub.copy()
        sub["Y"] = Y
        n_bin = int(is_bin.sum())
        source_filter = (
            "label_type=='spike'; value in {'0','1'} -> Y=int(value) "
            f"({n_bin} sn1 binary rows); else Y = 1 if value=='ied' else 0 "
            f"({len(sub) - n_bin} Centaur-IED IED-vs-benign rows). "
            "All spike rows retained."
        )
    else:
        val = sub["value"].astype(str).str.strip()
        # ERRATUM FIX (decouple, 2026-05-18): apply the canonical 6-class
        # collapse {bipd,birds}->other UNIFORMLY across ALL sources before
        # one-vs-rest (AUDIT s6 / Phase-1 decision). Phase-3 omitted this,
        # so ~30,511 Centaur-novice bipd/birds reads were mislabelled
        # NEGATIVE for the `iic` task. Collapsing here fixes `iic`
        # (Y=1 iff collapsed value=='other') and leaves sz/lpd/gpd/lrda/
        # grda UNCHANGED (collapsed 'other' != those classes).
        val = val.replace({"bipd": "other", "birds": "other"})
        sub = sub.copy()
        sub["Y"] = (val == spec).astype(np.int64)
        source_filter = (
            f"label_type=='{label_type}'; {{bipd,birds}}->other (uniform "
            f"6-class collapse, erratum fix); Y = 1 if value=='{spec}' else 0"
        )

    n_obs = len(sub)
    if n_obs == 0:
        raise RuntimeError(f"task {name}: zero observations after filter")

    # Dense 0-based contiguous ids in STABLE SORTED order of unified id.
    seg_uniq = np.sort(sub["seg_id"].unique())
    rater_uniq = np.sort(sub["rater_id"].unique())
    seg_map = {int(s): i for i, s in enumerate(seg_uniq)}
    rater_map = {int(r): i for i, r in enumerate(rater_uniq)}

    case_id = sub["seg_id"].map(seg_map).to_numpy(dtype=np.int64)
    rater_id = sub["rater_id"].map(rater_map).to_numpy(dtype=np.int64)
    Y = sub["Y"].to_numpy(dtype=np.int64)

    long = pd.DataFrame({"case_id": case_id, "rater_id": rater_id, "Y": Y})

    prepared_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(prepared_dir / f"{name}.csv", index=False)

    # canonical_name by dense rater idx (R5-safe key, NOT numeric id)
    rmap = dict(zip(raters["rater_id"].astype(int), raters["canonical_name"]))
    emap = dict(
        zip(raters["rater_id"].astype(int), raters["expertise_level"])
    )
    rater_index_to_name: list[str] = []
    rid_rows = []
    for dense_idx, uni_rid in enumerate(rater_uniq):
        uni_rid = int(uni_rid)
        cname = rmap.get(uni_rid, f"rater_{uni_rid}")
        if pd.isna(cname):
            cname = f"rater_{uni_rid}"
        cname = str(cname)
        rater_index_to_name.append(cname)
        elvl = emap.get(uni_rid, "")
        if pd.isna(elvl):
            elvl = ""
        rid_rows.append(
            {
                "dense_rater_id": dense_idx,
                "unified_rater_id": uni_rid,
                "canonical_name": cname,
                "expertise_level": str(elvl),
            }
        )

    pd.DataFrame(rid_rows).to_csv(
        prepared_dir / f"{name}_rater_ids.csv", index=False
    )

    pos = int(Y.sum())
    meta = {
        "rater_index_to_name": rater_index_to_name,
        "provenance": {
            "task": name,
            "label_type": label_type,
            "source_label_filter": source_filter,
            "n_obs": int(n_obs),
            "n_obs_pre_filter": int(n_raw),
            "n_case": int(len(seg_uniq)),
            "n_rater": int(len(rater_uniq)),
            "n_pos": pos,
            "pos_rate": float(pos / n_obs),
            "labels_csv_sha256": labels_sha256,
            "id_mapping": (
                "seg_id/rater_id remapped to dense 0-based contiguous ints "
                "in stable sorted order of the unified id"
            ),
        },
    }
    with open(prepared_dir / f"{name}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return {
        "task": name,
        "n_obs": int(n_obs),
        "n_case": int(len(seg_uniq)),
        "n_rater": int(len(rater_uniq)),
        "n_pos": pos,
        "pos_rate": round(float(pos / n_obs), 4),
    }


def build_all(
    labels_csv: Path,
    raters_csv: Path,
    prepared_dir: Path,
    tasks: list[str] | None = None,
) -> dict:
    """Build all (or the given subset of) tasks. Returns {task: summary}."""
    tasks = tasks or ALL_TASKS
    labels_sha = _sha256(labels_csv)
    labels = pd.read_csv(labels_csv, low_memory=False)
    raters = pd.read_csv(raters_csv, low_memory=False)
    out = {}
    for name in tasks:
        out[name] = build_task(
            name, labels, raters, prepared_dir, labels_sha256=labels_sha
        )
    out["_labels_csv_sha256"] = labels_sha
    return out


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parent.parent
    res = build_all(
        repo / "data" / "labels" / "labels.csv",
        repo / "data" / "labels" / "raters.csv",
        repo / "data" / "prepared",
        tasks=sys.argv[1:] or None,
    )
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(
            f"  {v['task']:16s} n_obs={v['n_obs']:>8d} "
            f"n_case={v['n_case']:>6d} n_rater={v['n_rater']:>5d} "
            f"pos_rate={v['pos_rate']:.4f}"
        )
