"""Verify + regenerate the data/engine_inputs/ MANIFEST (Phase 5).

HISTORY: this script ORIGINALLY built data/engine_inputs/ by vendoring
the legacy out-of-repo prepared sparcnet per-domain SDT fits (219-row,
15/29-rater era — recorded historically in the MANIFEST). Phase 3
SUPERSEDED that: the engine's
sdt_fits.csv is now produced by `pipeline/run_unified_calibration.py`
STEP d — the reference-faithful CV-Youden recompute on the UNIFIED
`data/labels/` corpus (14,214 rows, 6 IIIC domains), and is already
exercised + gated green by tests/test_phase3_calibration.py +
Mode-A/Mode-B.

So this script no longer BUILDS engine_inputs (re-running the old
sibling path would REGRESS to the 219-row legacy + re-introduce the
sibling dependency — a D9/D3 violation). Phase 5 repurposes it as a
self-verifying MANIFEST regenerator for the CURRENT in-repo artifact:

  • zero sibling-repo reads (D9);
  • record the TRUE producer (run_unified_calibration STEP d) + the
    D3 `data/labels` sha256 lineage anchor, and CROSS-CHECK it equals
    the anchor in data/deployment_prior/summary.json (proving
    engine_inputs and deployment_prior derive from the SAME unified
    corpus — plan D3);
  • record live output_sha256 (the prior MANIFEST was stale — its
    output_sha256 did not match the files on disk);
  • SELF-CHECK the gated correctness contract (sdt_fits.csv 6 IIIC
    domains; the 29 Q2-locked candidates + the 4 R5 name-variants
    join sdt_fits by canonical name) — the same invariants
    tests/test_phase3_calibration.py asserts.

`main()` returns 0 iff every verification passes (so the gate/tests
can invoke it); it regenerates MANIFEST.json from the live artifact.
The old sibling-prepared build is preserved historically as
`sdt_fits.legacy_oldcorpus.csv` (recorded, never rebuilt). The frozen
repo-root `Sigma_l_fitted.npy` (older 15-rater era) is an immutable
vendored input — recorded, never regenerated.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from typing import Dict, List

import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS)
ENGINE_INPUTS = os.path.join(REPO, "data", "engine_inputs")
LABELS_DIR = os.path.join(REPO, "data", "labels")
DEPLOY_SUMMARY = os.path.join(
    REPO, "data", "deployment_prior", "summary.json")
MANIFEST = os.path.join(ENGINE_INPUTS, "MANIFEST.json")

# Live data artifacts whose sha256 the MANIFEST records (NOT MANIFEST/
# README themselves). Sigma_l_fitted.npy is repo-root (frozen vendored).
_OUTPUT_FILES = [
    "sdt_fits.csv",
    "cross_domain_rater_matrix.csv",
    "cross_domain_rater_matrix.q2locked.csv",
    "sdt_fits.legacy_oldcorpus.csv",
    "sdt_fits.spike.csv",
]
_R5_CANON = ("Aaron F. Struck", "Hiba A. Haider",
             "Jonathan J. Halford", "Olga Taraschenko")
_IIIC_DOMAINS = {"sz", "lpd", "gpd", "lrda", "grda", "iic"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _verify() -> List[str]:
    """Return a list of failure strings (empty ⇒ all invariants hold).
    Mirrors the gated tests/test_phase3_calibration contract."""
    fail: List[str] = []
    sdt_p = os.path.join(ENGINE_INPUTS, "sdt_fits.csv")
    q2_p = os.path.join(ENGINE_INPUTS,
                        "cross_domain_rater_matrix.q2locked.csv")
    recon_p = os.path.join(ENGINE_INPUTS,
                           "cross_domain_rater_matrix.csv")
    legacy_p = os.path.join(ENGINE_INPUTS,
                            "sdt_fits.legacy_oldcorpus.csv")
    stray_p = os.path.join(ENGINE_INPUTS, "sdt_fits.phase3.csv")

    if not os.path.exists(sdt_p):
        return ["sdt_fits.csv MISSING (run pipeline/"
                "run_unified_calibration.py STEP d)"]
    sdt = pd.read_csv(sdt_p)
    doms = set(sdt["domain"].unique())
    if doms != _IIIC_DOMAINS:
        fail.append(f"sdt_fits domains {sorted(doms)} != "
                    f"{sorted(_IIIC_DOMAINS)}")
    if not os.path.exists(legacy_p):
        fail.append("sdt_fits.legacy_oldcorpus.csv missing (the "
                    "historical sibling-prepared 219-row build)")
    if os.path.exists(stray_p):
        fail.append("sdt_fits.phase3.csv present — Phase 5 removed this "
                    "stray, NOT-engine-consumed snapshot (it differs "
                    "materially from the authoritative sdt_fits.csv)")
    sdt_names = set(sdt["rater_name"].astype(str).str.strip())
    for p, col, lbl in ((q2_p, None, "q2locked"),
                        (recon_p, "confirmed_sparcnet_name", "recon")):
        if not os.path.exists(p):
            fail.append(f"{os.path.basename(p)} MISSING")
            continue
        df = pd.read_csv(p)
        if len(df) != 29:
            fail.append(f"{lbl} has {len(df)} rows, expected 29 "
                        "(Q2-locked candidate pool)")
        if col:
            miss = sorted(set(df[col].astype(str).str.strip())
                          - sdt_names)
            if miss:
                fail.append(f"candidates not joinable to sdt_fits: "
                            f"{miss}")
    for canon in _R5_CANON:
        if canon not in sdt_names:
            fail.append(f"R5 name-variant {canon!r} absent from "
                        "sdt_fits (candidate silently dropped)")
    return fail


def _d3_anchor() -> Dict[str, str]:
    """sha256 of the UNIFIED source corpus the engine_inputs derive
    from (via run_unified_calibration STEP d). The SAME corpus
    deployment_prior derives from — plan D3."""
    return {
        "labels_csv_sha256": sha256(
            os.path.join(LABELS_DIR, "labels.csv")),
        "raters_csv_sha256": sha256(
            os.path.join(LABELS_DIR, "raters.csv")),
        "labels_csv_path": "data/labels/labels.csv",
        "raters_csv_path": "data/labels/raters.csv",
    }


def main() -> int:
    print("=== verify + regenerate data/engine_inputs/MANIFEST.json "
          "(Phase 5; no sibling reads) ===", flush=True)
    fail = _verify()

    anchor = _d3_anchor()
    # D3 cross-check: engine_inputs and deployment_prior MUST share the
    # same data/labels sha256 (both derived from the one unified corpus).
    d3 = {"checked": False}
    if os.path.exists(DEPLOY_SUMMARY):
        dp = json.load(open(DEPLOY_SUMMARY)).get(
            "data_labels_provenance", {})
        match = (dp.get("labels_csv_sha256") == anchor["labels_csv_sha256"]
                 and dp.get("raters_csv_sha256")
                 == anchor["raters_csv_sha256"])
        d3 = {"checked": True,
              "deployment_prior_summary": "data/deployment_prior/"
              "summary.json",
              "shares_data_labels_lineage": bool(match)}
        if not match:
            fail.append("D3 VIOLATION: engine_inputs data/labels "
                        "sha256 != deployment_prior/summary.json "
                        "data_labels_provenance (NOT the same corpus)")
    else:
        fail.append("deployment_prior/summary.json absent — cannot "
                    "verify D3 shared lineage")

    out_sha = {f: sha256(os.path.join(ENGINE_INPUTS, f))
               for f in _OUTPUT_FILES
               if os.path.exists(os.path.join(ENGINE_INPUTS, f))}
    sigma_l = os.path.join(REPO, "Sigma_l_fitted.npy")
    if os.path.exists(sigma_l):
        out_sha["Sigma_l_fitted.npy(repo-root,frozen)"] = sha256(sigma_l)
    n_rows = (len(pd.read_csv(os.path.join(ENGINE_INPUTS,
              "sdt_fits.csv")))
              if os.path.exists(os.path.join(ENGINE_INPUTS,
                                             "sdt_fits.csv")) else None)

    manifest = {
        "generated": datetime.date.today().isoformat(),
        "regenerated_by": "scripts/build_engine_inputs.py (Phase 5: "
        "verify + regenerate; NO sibling reads)",
        "true_producer": "pipeline/run_unified_calibration.py STEP d "
        "— the reference-faithful CV-Youden recompute on the UNIFIED "
        "data/labels corpus (Phase 3). This script does NOT build "
        "engine_inputs; it verifies + records provenance for it.",
        "authoritative_artifact": "sdt_fits.csv (engine_paths.SDT_FITS; "
        "engine-consumed; gated by tests/test_phase3_calibration.py + "
        "Mode-A/Mode-B). sdt_fits.legacy_oldcorpus.csv = the historical "
        "sibling-prepared 219-row build (recorded, never rebuilt). "
        "Phase 5 REMOVED the stray sdt_fits.phase3.csv (a superseded, "
        "not-engine-consumed snapshot that differed materially).",
        "sdt_fits_rows": n_rows,
        "data_labels_provenance": anchor,
        "d3_shared_lineage_with_deployment_prior": d3,
        "output_sha256": out_sha,
        "frozen_vendored_input": "Sigma_l_fitted.npy (repo root) — "
        "older 15-rater-era Corr_l; immutable, recorded, never rebuilt.",
        "self_check": {
            "contract": "mirrors tests/test_phase3_calibration "
            "(6 IIIC domains; 29 Q2-locked + 4 R5 name-variants join "
            "sdt_fits by canonical rater_name) + D3 cross-check",
            "passed": not fail,
            "failures": fail,
        },
        "superseded_legacy": {
            "old_builder": "this file PRE-Phase-5 vendored the legacy "
            "out-of-repo prepared per-domain SDT fits (219-row, "
            "preserved as sdt_fits.legacy_oldcorpus.csv). RETIRED — "
            "re-running that path would regress engine_inputs + violate "
            "D9 (no out-of-repo reads) / D3.",
            "consolidated_dir": "the original plan's Phase-5 gate "
            "referenced data/consolidated/ — it does NOT exist "
            "(retired in Phase 1, zero consumers); dropped from the "
            "Phase-5 gate.",
        },
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {MANIFEST}")
    print(f"  D3 shared-lineage with deployment_prior: "
          f"{d3.get('shares_data_labels_lineage')}")
    if fail:
        print("  SELF-CHECK FAILED:", flush=True)
        for x in fail:
            print(f"    - {x}")
        return 1
    print("  self-check PASSED (engine_inputs provenance reconciled).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
