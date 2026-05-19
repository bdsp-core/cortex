"""Single source of truth for the engine's data-input paths.

Before: every consumer reached into the *sibling* repo via a hardcoded
`../ilae-skill-certification-test-main/data/prepared/...` relative path
(≈10 call sites) — fragile and non-portable.

Now: all engine data inputs live in this repo at `data/engine_inputs/`,
produced + verified by `scripts/build_engine_inputs.py`. Provenance,
sha256, and the lossless/reproduction proofs are in
`data/engine_inputs/MANIFEST.json`.

Two inputs are intentionally NOT moved into data/engine_inputs/ —
they were never the fragility problem (already in-repo, not cross-repo)
and are referenced by their repo-root location in many places; moving
them would add blast radius with zero fragility reduction and risk
duplicate→drift:
  • `Sigma_l_fitted.npy`  — FROZEN 15-rater-era artifact, repo root
  • `cert_config.yaml`    — hand-maintained config, repo root
This module still exposes them so every consumer can resolve all
engine inputs through one place.
"""
import os

# Phase 2 (unified merge): engine_paths.py now lives in engine/, so the
# repo root is one level up. Path-only adjustment — no value/logic change.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENGINE_INPUTS_DIR = os.path.join(_REPO, "data", "engine_inputs")

#: 29-rater cross-domain σ/θ/ℓ panel (engine reads this verbatim).
#: Byte-identical vendored copy of the former sibling-repo file.
RATER_MATRIX = os.path.join(ENGINE_INPUTS_DIR,
                            "cross_domain_rater_matrix.csv")

#: Tidy long per-rater SDT fits (219 rows; the documented source of
#: truth from which RATER_MATRIX is reproducible — see MANIFEST).
SDT_FITS = os.path.join(ENGINE_INPUTS_DIR, "sdt_fits.csv")

#: Frozen fitted hierarchical-prior covariance bundle (LEGACY 15-rater-era;
#: engine migrates to PI's fitted 12x12->14x14 Sigma in Phase 3/5). Single
#: real file at repo root; archive/ and engine/ are symlinks to it (Phase-2;
#: zero drift — immutable). Original methodology-repo path text preserved.
SIGMA_L = os.path.join(_REPO, "Sigma_l_fitted.npy")

#: Hand-maintained config. Canonical location is the repo root.
CERT_CONFIG = os.path.join(_REPO, "cert_config.yaml")

#: Phase-4 deployment integration. The frozen clinical-deployment
#: artifact (Σ prior, case bank, ℓ* thresholds, reference sim). PI
#: scripts hardcoded `/Users/mwestover/.../data/deployment_prior`;
#: resolved here so the ported deployment runtime carries zero
#: absolute paths (UNIFIED_REPO_MERGE_PLAN.md Phase 4 step 2).
DEPLOYMENT_PRIOR = os.path.join(_REPO, "data", "deployment_prior")

#: Phase-3.5 unified reference-faithful calibration (config_version 13,
#: 7 per-task ℓ*). The repo-root `cert_config.yaml` (CERT_CONFIG) is the
#: untouched legacy v11; deployment consumes THIS v13 single lineage
#: (decision 2026-05-18; plan D2 "resolve three-lineage conflict").
CALIB_CERT_CONFIG = os.path.join(_REPO, "calibration", "cert_config.yaml")


def sdt_fits_domain(domain):
    """Per-domain SDT-fits DataFrame, a drop-in for the legacy
    `pd.read_csv(sparcnet_{domain}_sdt_fits.csv)` (identical columns,
    dtypes and values — verified in build_engine_inputs.py)."""
    import pandas as pd
    df = pd.read_csv(SDT_FITS)
    return (df[df["domain"] == domain]
            .drop(columns=["domain"]).reset_index(drop=True))
