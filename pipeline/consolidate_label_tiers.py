"""Tier the CANONICAL lake (labels.csv) into expert/experienced/novice
files — LOSSLESS, self-verifying. R3-CORRECTED for the unified repo.

WHY THIS DIFFERS FROM THE ORIGINAL (R3, decided 2026-05-18):
  The original script (methodology repo) unioned labels.csv AND the raw
  Centaur files (centaur_iiic_novice_labels.csv / _expert_labels.xlsx /
  _survey) as *disjoint* sources, with row-conservation hardwired to the
  pre-ingest corpus (1,303,201 + 125,865 + 20,000 + 119). PI's canonical
  labels.csv ALREADY ingested Centaur 2025, and Phase 1 additionally
  ingested the 4-expert gold panel as source_dataset='centaur_iiic_expert'.
  Re-unioning the raw Centaur files here would DOUBLE-COUNT Centaur. So
  this version tiers ONLY the canonical labels.csv; the raw Centaur files
  remain provenance-only under external/centaur_iiic_goldpanel_raw/ and are
  intentionally NOT re-unioned.

SHIPPING NOTE: consolidated/ is deliberately NOT part of the shipped repo
  (zero engine/deployment consumers; it is a losslessly reconstructible
  convenience view). The canonical tier truth is raters.csv.expertise_level.
  This script is retained for ad-hoc reconstruction only; running it writes
  to a caller-chosen out dir and never to the shipped tree by default.

Tiering rule (unchanged from the original lake path):
  raters.csv.expertise_level ∈ {expert, experienced, novice} → that tier
  (tier_source='raters_csv_expertise'); anything else/missing → novice
  (tier_source='untiered_default'). Originals are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import List

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS)
LBL = os.path.join(REPO, "data", "labels")
F_LABELS = os.path.join(LBL, "labels.csv")
F_RATERS = os.path.join(LBL, "raters.csv")

SCHEMA = ["obs_id", "tier", "tier_source", "source_file", "src_row_idx",
          "source_dataset", "seg_id", "rater_id", "label_type", "value"]
TIERS = ("expert", "experienced", "novice")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    lab = pd.read_csv(F_LABELS, dtype=str, keep_default_na=False,
                      low_memory=False)
    raters = pd.read_csv(F_RATERS, dtype=str, keep_default_na=False)
    n = len(lab)

    exp_lvl = raters.set_index("rater_id")["expertise_level"].to_dict()
    lvl = lab["rater_id"].map(exp_lvl).fillna("")
    is_tiered = lvl.isin(list(TIERS))
    tier = lvl.where(is_tiered, "novice")

    out = pd.DataFrame({
        "obs_id": np.arange(n),
        "tier": tier.values,
        "tier_source": np.where(is_tiered, "raters_csv_expertise",
                                "untiered_default"),
        "source_file": "labels.csv",
        "src_row_idx": np.arange(n),
        "source_dataset": lab["source_dataset"].values,
        "seg_id": lab["seg_id"].values,
        "rater_id": lab["rater_id"].values,
        "label_type": lab["label_type"].values,
        "value": lab["value"].values,
    })[SCHEMA].astype(str)

    paths = {}
    for t in TIERS:
        p = os.path.join(out_dir, {"expert": "experts",
                                   "experienced": "experienced",
                                   "novice": "novice"}[t] + ".csv")
        out[out["tier"] == t].reset_index(drop=True).to_csv(p, index=False)
        paths[t] = p
    return lab, out, paths


def verify(lab: pd.DataFrame, paths) -> str:
    """Reconstruct labels.csv EXACTLY from the on-disk tier files."""
    disk = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False)
                      for p in paths.values()], ignore_index=True)
    if len(disk) != len(lab):
        raise SystemExit(f"ABORT: disk rows {len(disk)} != labels "
                         f"{len(lab)} (row conservation FAILED)")
    disk["src_row_idx"] = disk["src_row_idx"].astype(int)
    g = disk.sort_values("src_row_idx")
    rec = pd.DataFrame({
        "seg_id": g.seg_id.values, "rater_id": g.rater_id.values,
        "label_type": g.label_type.values, "value": g.value.values,
        "source_dataset": g.source_dataset.values,
    })
    if not rec.equals(lab.reset_index(drop=True)):
        raise SystemExit("ABORT: labels.csv round-trip MISMATCH "
                         "(no tier file trusted)")
    return f"labels.csv round-trip OK ({len(rec):,} rows exact)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(LBL, "consolidated"),
                    help="ad-hoc output dir (NOT shipped by default)")
    args = ap.parse_args()
    print("=== consolidate (R3-corrected: canonical labels.csv only) ===",
          flush=True)
    lab, out, paths = build(args.out_dir)
    chk = verify(lab, paths)
    print("  OK " + chk, flush=True)
    tier_counts = (out.groupby(["tier", "tier_source"]).size()
                   .reset_index(name="n").to_dict("records"))
    manifest = {
        "generated_for": "unified repo (R3-corrected)",
        "source": {"labels.csv": {"sha256": sha256(F_LABELS),
                                  "rows": len(lab)},
                   "raters.csv": {"sha256": sha256(F_RATERS)}},
        "tiering_rule": "raters.csv.expertise_level; else novice "
                        "(untiered_default)",
        "row_conservation": {"labels_csv": len(lab), "total": len(out)},
        "tier_breakdown": tier_counts,
        "round_trip": "PASS - labels.csv reconstructed exactly",
        "r3_note": "raw Centaur files NOT re-unioned (provenance only); "
                   "avoids the Centaur double-count of the original "
                   "disjoint-union design on PI's ingested labels.csv",
        "schema": SCHEMA,
    }
    with open(os.path.join(args.out_dir,
                           "consolidation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for r in tier_counts:
        print(f"  {r['tier']:<11} {r['tier_source']:<22} {r['n']:>9,}")
    print(f"  -> {os.path.relpath(args.out_dir, REPO)}/ "
          "(ad-hoc; not shipped)")


if __name__ == "__main__":
    main()
