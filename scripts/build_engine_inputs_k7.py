"""Phase 9 Layer 3 — build K=7 engine inputs.

K=7 sibling of `scripts/build_engine_inputs.py` (the K=6 IIIC predecessor).
Produces the K=7 engine input artifacts the SMC engine reads at runtime:

  1. `data/engine_inputs/sdt_fits_k7.csv` — concatenation of the v13 K=6
     `sdt_fits.csv` (6 IIIC; 14,214 rows) + the v13 spike sdt_fits
     `sdt_fits.spike.csv` (2,551 rows) with domain rebadged to 'spike'.
     Total ~16,765 rows × 9 cols. **V_B preserves D2:** these per-rater
     fits come directly from the byte-verbatim two-stage path
     (`pipeline/reference_calibration/fit_sdt_per_domain.py`; md5
     `6b90d59…`); the K=7 V_B joint fit is consumed only for s_j/s_sd
     (Layer 2).

  2. `data/engine_inputs/cross_domain_rater_matrix_k7.csv` — 29 Q2-locked
     candidates × 7 task triples (σ, θ, ℓ) per task. Extended from the
     K=6 cross_domain_rater_matrix.csv with 3 new columns:
     sigma_spike, theta_spike, l_spike (joined on canonical name from
     sdt_fits.spike.csv).

  3. `data/engine_inputs/MANIFEST_k7.json` — provenance + D3 sha256 chain
     (extends K=6 D3 anchor with the V_B joint posterior shas + assemble
     summary path; closes audit G3 by chaining intermediate fits).

NUTS R̂ is recorded in `calibration/joint/{task}_k7_B_summary.json` per task;
this script consumes only the SVI posteriors for s_j extraction (V_B
methodology). NUTS validation is a downstream consumer that reads the same
NPZ files.

Usage:
    .venv/bin/python scripts/build_engine_inputs_k7.py --variant B
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = Path(os.path.dirname(_THIS))
ENGINE_INPUTS = REPO / "data" / "engine_inputs"
LABELS_DIR = REPO / "data" / "labels"
JOINT = REPO / "calibration" / "joint"

# K=7 task table
K7_DOMAINS = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
# Q2-locked 29 candidates (from K=6 cross_domain_rater_matrix); same panel
# is used at K=7 with spike fits joined on canonical_name.
# R5 name variants (joined on canonical_name in sdt_fits)
_R5_CANON = ("Aaron F. Struck", "Hiba A. Haider",
             "Jonathan J. Halford", "Olga Taraschenko")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_sdt_fits_k7() -> tuple[Path, int]:
    """Concatenate v13 IIIC sdt_fits + spike sdt_fits → sdt_fits_k7.csv.

    Both are byte-verbatim two-stage fitter outputs (D2 invariant). V_B's
    joint posterior is NOT consumed here per Phase-3.5 decoupling
    (cert_config.yaml:278-283).
    """
    iiic_path = ENGINE_INPUTS / "sdt_fits.csv"
    spike_path = ENGINE_INPUTS / "sdt_fits.spike.csv"
    if not iiic_path.exists():
        raise FileNotFoundError(
            f"K=6 IIIC sdt_fits missing: {iiic_path} — Phase-3 calibration "
            "must run first")
    if not spike_path.exists():
        raise FileNotFoundError(
            f"K=6 spike sdt_fits missing: {spike_path} — Phase-3 calibration "
            "must run first")

    iiic = pd.read_csv(iiic_path)
    spike = pd.read_csv(spike_path)

    # Sanity: domains
    iiic_doms = set(iiic["domain"].astype(str).unique())
    expected_iiic = {"sz", "lpd", "gpd", "lrda", "grda", "iic"}
    if iiic_doms != expected_iiic:
        raise RuntimeError(f"IIIC sdt_fits domains {iiic_doms} != {expected_iiic}")
    spike_doms = set(spike["domain"].astype(str).unique())
    if spike_doms != {"spike"}:
        raise RuntimeError(f"spike sdt_fits domains {spike_doms} != {{'spike'}}")

    k7 = pd.concat([spike, iiic], ignore_index=True)
    out = ENGINE_INPUTS / "sdt_fits_k7.csv"
    k7.to_csv(out, index=False)
    return out, len(k7)


def build_cross_domain_matrix_k7(variant: str) -> tuple[Path, int]:
    """Extend the 29-rater K=6 cross_domain_rater_matrix to K=7 by adding
    spike columns. Spike (σ, θ, ℓ) are joined on canonical_name from
    sdt_fits.spike.csv. ℓ is computed from σ via ell = -log(sigma) (the
    standard mapping; matches engine_paths convention)."""
    import numpy as np

    src = ENGINE_INPUTS / "cross_domain_rater_matrix.csv"
    if not src.exists():
        raise FileNotFoundError(f"K=6 matrix missing: {src}")
    mat = pd.read_csv(src)
    if len(mat) != 29:
        raise RuntimeError(f"K=6 matrix has {len(mat)} rows; expected 29")

    spike = pd.read_csv(ENGINE_INPUTS / "sdt_fits.spike.csv")
    # Name lookup with R5 variants — try `confirmed_sparcnet_name` first
    # then `confirmed_canonical_name`
    name_col_primary = ("confirmed_sparcnet_name"
                        if "confirmed_sparcnet_name" in mat.columns
                        else "confirmed_canonical_name")
    name_to_spike: dict[str, tuple[float, float, float]] = {}
    for _, r in spike.iterrows():
        n = str(r["rater_name"]).strip()
        sigma = float(r["sigma"])
        theta = float(r["theta"])
        ell = -float(np.log(max(sigma, 1e-12)))
        name_to_spike[n] = (sigma, theta, ell)

    new_rows = []
    for _, row in mat.iterrows():
        n = str(row.get(name_col_primary, "")).strip()
        # Try canonical_name fallback
        if n not in name_to_spike and "confirmed_canonical_name" in mat.columns:
            n = str(row.get("confirmed_canonical_name", "")).strip()
        if n in name_to_spike:
            sigma_spike, theta_spike, l_spike = name_to_spike[n]
        else:
            # Defensive: if no spike fit for this candidate, mark NaN
            sigma_spike = theta_spike = l_spike = float("nan")
        new_row = row.to_dict()
        new_row["sigma_spike"] = sigma_spike
        new_row["theta_spike"] = theta_spike
        new_row["l_spike"] = l_spike
        new_rows.append(new_row)

    out_df = pd.DataFrame(new_rows)
    out = ENGINE_INPUTS / "cross_domain_rater_matrix_k7.csv"
    out_df.to_csv(out, index=False)
    return out, len(out_df)


def build_manifest_k7(variant: str,
                      sdt_fits_k7: Path, n_sdt_rows: int,
                      cdm_k7: Path, n_cdm_rows: int) -> Path:
    """Emit MANIFEST_k7.json with D3 sha256 chain extended through
    intermediate K=7 fits (closes audit G3 — the K=6 D3 anchored only
    the input labels.csv; K=7 chains include the joint posteriors)."""
    labels_p = LABELS_DIR / "labels.csv"
    raters_p = LABELS_DIR / "raters.csv"

    # K=7 V_B joint posterior shas (per-task)
    joint_post_shas = {}
    for t in K7_DOMAINS:
        p = JOINT / f"{t}_k7_{variant}_posterior.npz"
        if p.exists():
            joint_post_shas[t] = sha256_path(p)
        else:
            joint_post_shas[t] = None  # may be missing if NUTS not yet run

    # Layer-2 assemble outputs shas
    assemble_paths = {
        "segment_signals": LABELS_DIR / "segment_signals.csv",
        "s_j_table_k7": JOINT / f"s_j_table_k7_{variant}.csv",
        "joint_per_rater_sidecar": (
            JOINT / f"joint_per_rater_params_k7_{variant}.csv"),
        "assemble_summary_k7": (
            JOINT / f"assemble_summary_k7_{variant}.json"),
    }
    assemble_shas = {
        k: sha256_path(p) if p.exists() else None
        for k, p in assemble_paths.items()
    }

    manifest = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": 9,
        "K": 7,
        "variant": variant,
        "true_producer": (
            "scripts/build_engine_inputs_k7.py — concatenates v13 byte-"
            "verbatim two-stage IIIC sdt_fits + spike sdt_fits (D2 invariant). "
            "V_B joint posterior NPZs are consumed by Layer 2 "
            "(assemble_outputs_k7.py) for s_j; per-rater (σ̂, θ̂) come "
            "from the byte-verbatim two-stage path per Phase-3.5 decoupling."),
        "sdt_fits_k7_rows": n_sdt_rows,
        "cross_domain_rater_matrix_k7_rows": n_cdm_rows,
        "k7_domains": K7_DOMAINS,
        "data_labels_provenance": {
            "labels_csv_sha256": sha256_path(labels_p),
            "raters_csv_sha256": sha256_path(raters_p),
            "labels_csv_path": str(labels_p.relative_to(REPO)),
            "raters_csv_path": str(raters_p.relative_to(REPO)),
            "v13_iiic_sdt_fits_sha256": sha256_path(
                ENGINE_INPUTS / "sdt_fits.csv"),
            "v13_spike_sdt_fits_sha256": sha256_path(
                ENGINE_INPUTS / "sdt_fits.spike.csv"),
        },
        "k7_joint_posterior_sha256": joint_post_shas,
        "assemble_outputs_sha256": assemble_shas,
        "k7_artifacts_sha256": {
            "sdt_fits_k7.csv": sha256_path(sdt_fits_k7),
            "cross_domain_rater_matrix_k7.csv": sha256_path(cdm_k7),
        },
        "phase35_decoupling_preserved": (
            "V_B follows Phase-3.5 cert_config.yaml:278-283 — joint posterior "
            "used for s_j ONLY; per-rater fits remain byte-verbatim two-stage. "
            "D2 invariant (md5 pin on fit_sdt_per_domain.py) preserved."),
    }
    manifest_p = ENGINE_INPUTS / "MANIFEST_k7.json"
    manifest_p.write_text(json.dumps(manifest, indent=2))
    return manifest_p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="B",
                    help="winning variant (default B)")
    args = ap.parse_args(argv)
    variant = args.variant

    print(f"=== Phase 9 Layer 3: build K=7 engine_inputs (variant V_{variant}) ===",
          flush=True)

    print("  [1] building sdt_fits_k7.csv (concat IIIC + spike v13 outputs)",
          flush=True)
    sdt_path, n_sdt = build_sdt_fits_k7()
    print(f"      → {sdt_path.name}: {n_sdt:,} rows (6 IIIC + 1 spike)",
          flush=True)

    print("  [2] building cross_domain_rater_matrix_k7.csv (29 × 7 tasks)",
          flush=True)
    cdm_path, n_cdm = build_cross_domain_matrix_k7(variant)
    print(f"      → {cdm_path.name}: {n_cdm} rows × {n_cdm and len(pd.read_csv(cdm_path).columns)} cols",
          flush=True)

    print("  [3] building MANIFEST_k7.json (D3 chain extended)", flush=True)
    manifest_p = build_manifest_k7(variant, sdt_path, n_sdt, cdm_path, n_cdm)
    print(f"      → {manifest_p.name}", flush=True)
    print("=== Layer 3 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
