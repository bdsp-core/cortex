"""Post-hoc patch for v2 ExpA results: replace R0..R26 with canonical names.

The v2 experiment script had a bug — it used `row.get('canonical_name', ...)`
but the rater matrix column is `confirmed_canonical_name`, so the rater_id
fell through to the default `R{ri}` placeholder.  This script maps those
placeholders back to real names using the deterministic row order in
`cross_domain_rater_matrix.csv` (the same order the experiment iterates).
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
RESULTS_DIR = os.path.join(ENGINE_REPO, "results", "phase1_figures")
MATRIX_PATH = ("/Users/elikeldsen/Documents/Research/spike-test-project/"
               "ilae-skill-certification-test-main/data/prepared/"
               "cross_domain_rater_matrix.csv")

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]


def load_canonical_names():
    df = pd.read_csv(MATRIX_PATH)
    needed = [f"sigma_{d}" for d in DOMAINS] + [f"theta_{d}" for d in DOMAINS]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df["confirmed_canonical_name"].tolist()


def patch_expA():
    names = load_canonical_names()
    in_path = os.path.join(RESULTS_DIR, "expA_real_k6_v2.json")
    out_path = in_path  # in-place
    with open(in_path) as f:
        obj = json.load(f)

    mapping = {f"R{i}": names[i] for i in range(len(names))}
    n_patched = 0
    for row in obj["rows"]:
        rid = row.get("rater_id", "")
        if rid in mapping:
            row["rater_id"] = mapping[rid]
            n_patched += 1
    obj["config"]["rater_id_patched_v2"] = True
    obj["config"]["n_rows_patched"] = n_patched

    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Patched {n_patched} rows in {out_path}")
    print(f"  rater 0 → {names[0]!r}")
    print(f"  rater 6 → {names[6]!r}")  # should be M. Brandon Westover
    return names


if __name__ == "__main__":
    patch_expA()
