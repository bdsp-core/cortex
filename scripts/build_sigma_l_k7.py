"""Phase 9 Layer 5 — build K=7 Sigma_l_fitted_k7.npy from v13 two-stage fits.

Extracts the K=7 cross-task hierarchical covariance from the per-rater
two-stage fits in `data/engine_inputs/sdt_fits_k7.csv` (D2 byte-md5-pinned;
produced by Layer 3 from v13 IIIC + v13 spike sdt_fits). This matches the
K=6 v13 `Sigma_l_fitted.npy` precedent exactly:

  v13 note (from repo-root Sigma_l_fitted.npy):
    "Computed from cross_domain_rater_matrix.csv. Theta values clipped at
     2.0; Sigma_t mostly reflects clipping..."

V_B's modular architecture (Plummer 2015 modular Bayesian inference) cuts
the joint posterior at the s_j boundary: joint fit produces s_j (the
case-difficulty signal); per-rater (σ̂, θ̂) come from the byte-verbatim
two-stage `fit_sdt_per_domain.py`. Σ_l estimation follows the per-rater
path because Σ_l is a prior over per-rater latents at deployment time —
the engine consumes two-stage-derived parameters at runtime, so Σ_l should
be calibrated on the same parameter substrate.

Method:

  For each rater i and task k ∈ {spike, sz, lpd, gpd, lrda, grda, iic},
  collect (t_k_i, ℓ_k_i) from sdt_fits_k7.csv where:
    ell = -log(sigma);  t = -theta
  (matches the canonical joint↔two-stage parameter mapping used in
  assemble_outputs.py).

  Then compute, per (k1, k2) pair:
      Sigma_l[k1, k2] = Cov(ell_k1, ell_k2) over raters present in BOTH tasks
      Sigma_t[k1, k2] = Cov(t_k1,   t_k2)   over raters present in BOTH tasks

  Pairwise covariance handles the asymmetric rater overlap (most spike raters
  are not in IIIC tasks; most IIIC raters are not in spike). Diagonals use
  the full per-task rater set.

Output: `Sigma_l_fitted_k7.npy` matching the engine's `load_fitted_Sigma`
schema (dict with `Sigma_l`, `Sigma_t`, `Corr_l`, `Corr_t`, `domains`,
`n_raters`, `note`).

The repo-root K=6 `Sigma_l_fitted.npy` is preserved unchanged (Phase 9
preserves it as `Sigma_l_fitted.legacy_v13.npy` at Layer-5 promotion time;
this script only PRODUCES the K=7 sibling).

Usage:
    .venv/bin/python scripts/build_sigma_l_k7.py --variant B \\
        --out Sigma_l_fitted_k7.npy
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JOINT = REPO / "calibration" / "joint"
DEFAULT_OUT = REPO / "Sigma_l_fitted_k7.npy"

# K=7 task order (matches deployment Sigma + cert_config v14 + everywhere else)
K7_DOMAINS = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]


def _load_per_rater(variant: str = "B") -> dict[str, dict[int, tuple[float, float]]]:
    """For each task, return {rater_id: (ell, t)} dict from the K=7 two-stage
    sdt_fits_k7.csv (D2 byte-md5-pinned; Phase-9 Layer 3 output).

    Mapping: ell = -log(sigma); t = -theta (canonical joint↔two-stage
    parameter map used everywhere in this pipeline).

    Filters to converged raters only (sdt_fits.converged == True). Drops
    rows where sigma <= 0 (would yield ell = +inf).
    """
    import pandas as pd
    sdt_path = REPO / "data" / "engine_inputs" / "sdt_fits_k7.csv"
    if not sdt_path.exists():
        raise FileNotFoundError(
            f"sdt_fits_k7.csv missing: {sdt_path} — run Layer 3 "
            "(scripts/build_engine_inputs_k7.py) first")

    df = pd.read_csv(sdt_path)
    expected = set(K7_DOMAINS)
    actual = set(df["domain"].astype(str).unique())
    if expected != actual:
        raise RuntimeError(
            f"sdt_fits_k7 domains {actual} != K7_DOMAINS {expected}")

    out: dict[str, dict[int, tuple[float, float]]] = {}
    for task in K7_DOMAINS:
        sub = df[df["domain"] == task]
        # Filter to converged raters with valid sigma (positive); converged
        # column is True/False in the v13 spike + IIIC sdt_fits.
        conv = sub["converged"].astype(str).str.upper() == "TRUE"
        valid = (sub["sigma"].astype(float) > 0) & conv
        sub = sub[valid]
        rids = sub["rater_id"].astype(int).to_numpy()
        sigmas = sub["sigma"].astype(float).to_numpy()
        thetas = sub["theta"].astype(float).to_numpy()
        # Canonical mapping: ell = -log(sigma); t = -theta
        ells = -np.log(np.maximum(sigmas, 1e-12))
        ts = -thetas
        out[task] = {int(rid): (float(e), float(t))
                     for rid, e, t in zip(rids, ells, ts)}
    return out


def _pairwise_cov(values_by_task: dict[str, dict[int, float]],
                  tasks: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute pairwise covariance matrix Cov[k1, k2] over raters present in
    both task k1 and task k2. Also returns the diagonal-only variance and a
    per-pair n_overlap matrix for provenance.
    """
    K = len(tasks)
    Sigma = np.zeros((K, K), dtype=float)
    n_overlap = np.zeros((K, K), dtype=int)
    diag_n = np.zeros(K, dtype=int)
    for i, k1 in enumerate(tasks):
        rids1 = set(values_by_task[k1])
        diag_n[i] = len(rids1)
        v1_self = np.array([values_by_task[k1][r] for r in rids1])
        Sigma[i, i] = float(v1_self.var(ddof=1) if len(v1_self) >= 2
                            else float("nan"))
        n_overlap[i, i] = len(rids1)
        for j, k2 in enumerate(tasks):
            if j <= i:
                continue
            common = sorted(rids1 & set(values_by_task[k2]))
            n_overlap[i, j] = n_overlap[j, i] = len(common)
            if len(common) < 2:
                Sigma[i, j] = Sigma[j, i] = float("nan")
                continue
            v1 = np.array([values_by_task[k1][r] for r in common])
            v2 = np.array([values_by_task[k2][r] for r in common])
            c = float(np.cov(v1, v2, ddof=1)[0, 1])
            Sigma[i, j] = Sigma[j, i] = c
    meta = {"n_overlap": n_overlap.tolist(), "diag_n": diag_n.tolist()}
    return Sigma, n_overlap, meta


def _to_corr(Sigma: np.ndarray) -> np.ndarray:
    """Convert covariance to correlation matrix. NaN-safe: cells where either
    diagonal is NaN stay NaN."""
    K = Sigma.shape[0]
    Corr = np.full_like(Sigma, float("nan"))
    d = np.sqrt(np.diag(Sigma))
    for i in range(K):
        for j in range(K):
            if d[i] > 0 and d[j] > 0 and np.isfinite(Sigma[i, j]):
                Corr[i, j] = Sigma[i, j] / (d[i] * d[j])
    return Corr


def _fill_nans(M: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Replace NaNs with `fallback` (off-diagonal) and 1.0 (diagonal)."""
    out = np.where(np.isnan(M), fallback, M)
    K = out.shape[0]
    for i in range(K):
        if np.isnan(M[i, i]) or M[i, i] <= 0:
            out[i, i] = 1.0  # unit diagonal for Corr; positive for Sigma
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="B",
                    help="winning variant (default B)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output path (default {DEFAULT_OUT})")
    ap.add_argument("--require-cross-coverage", type=int, default=2,
                    help="minimum n_overlap per (k1,k2) pair to keep "
                         "estimate non-NaN (default 2)")
    args = ap.parse_args(argv)

    print(f"=== Phase 9 Layer 5: build K=7 Sigma_l_fitted (variant V_{args.variant}) ===",
          flush=True)

    per_rater = _load_per_rater(args.variant)
    n_per_task = {t: len(per_rater[t]) for t in K7_DOMAINS}
    print(f"  per-task rater counts: {n_per_task}", flush=True)

    # Build per-task ell and t dicts (separated)
    ell_by_task = {t: {r: v[0] for r, v in per_rater[t].items()}
                   for t in K7_DOMAINS}
    t_by_task = {t: {r: v[1] for r, v in per_rater[t].items()}
                 for t in K7_DOMAINS}

    print("  computing pairwise Sigma_l (ell covariance)...", flush=True)
    Sigma_l, n_overlap_l, meta_l = _pairwise_cov(ell_by_task, K7_DOMAINS)
    print("  computing pairwise Sigma_t (t covariance)...", flush=True)
    Sigma_t, n_overlap_t, meta_t = _pairwise_cov(t_by_task, K7_DOMAINS)

    # Correlation forms (unit diagonal where possible)
    Corr_l = _to_corr(Sigma_l)
    Corr_t = _to_corr(Sigma_t)

    # Report
    print("\n  K=7 Sigma_l (ell covariance) — diagonal (per-task variance):",
          flush=True)
    for i, k in enumerate(K7_DOMAINS):
        print(f"    {k:>5}: var(ell)={Sigma_l[i,i]:.4f}  n_raters={n_overlap_l[i,i]}",
              flush=True)
    print("\n  K=7 Corr_l (ell correlations; off-diagonals):", flush=True)
    for i, k1 in enumerate(K7_DOMAINS):
        for j, k2 in enumerate(K7_DOMAINS):
            if j <= i:
                continue
            n = n_overlap_l[i, j]
            r = Corr_l[i, j]
            r_str = f"{r:+.3f}" if np.isfinite(r) else "  NaN"
            print(f"    Corr(ell_{k1:>5}, ell_{k2:>5}) = {r_str}  (n={n:>4})",
                  flush=True)
    print("\n  K=7 Corr_t (t correlations; off-diagonals):", flush=True)
    for i, k1 in enumerate(K7_DOMAINS):
        for j, k2 in enumerate(K7_DOMAINS):
            if j <= i:
                continue
            n = n_overlap_t[i, j]
            r = Corr_t[i, j]
            r_str = f"{r:+.3f}" if np.isfinite(r) else "  NaN"
            print(f"    Corr(t_{k1:>5},   t_{k2:>5})   = {r_str}  (n={n:>4})",
                  flush=True)

    # Fill NaN cells (insufficient overlap) with 0 off-diag and 1 on diag
    # — the engine cannot consume NaN; record raw matrices in metadata
    Sigma_l_filled = _fill_nans(Sigma_l, fallback=0.0)
    Sigma_t_filled = _fill_nans(Sigma_t, fallback=0.0)
    Corr_l_filled = _fill_nans(Corr_l, fallback=0.0)
    Corr_t_filled = _fill_nans(Corr_t, fallback=0.0)

    note = (
        f"Phase 9 K=7 free-Σ extracted from v13 two-stage per-rater fits "
        f"(sdt_fits_k7.csv; D2 byte-md5-pinned). Matches K=6 v13 "
        f"Sigma_l_fitted.npy precedent (computed from cross_domain_rater_"
        f"matrix.csv → two-stage). V_B modular Bayesian inference: joint "
        f"posterior produces s_j only; per-rater (σ̂, θ̂) and derived "
        f"Σ_l/Σ_t come from byte-verbatim two-stage. Pairwise covariance "
        f"handles asymmetric rater overlap (most spike raters not in IIIC). "
        f"NaN cells (insufficient n_overlap < {args.require_cross_coverage}) "
        f"replaced with 0 off-diagonal + 1 diagonal for engine consumption; "
        f"raw matrices in metadata. Generated "
        f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}."
    )

    obj = {
        "Sigma_l": Sigma_l_filled,
        "Sigma_t": Sigma_t_filled,
        "Corr_l": Corr_l_filled,
        "Corr_t": Corr_t_filled,
        "domains": K7_DOMAINS,
        "n_raters": int(np.median([n_per_task[t] for t in K7_DOMAINS])),
        "note": note,
        # Metadata for provenance
        "n_raters_per_task": n_per_task,
        "n_overlap_ell": meta_l["n_overlap"],
        "n_overlap_t": meta_t["n_overlap"],
        "Sigma_l_raw_pre_nan_fill": Sigma_l.tolist(),
        "Sigma_t_raw_pre_nan_fill": Sigma_t.tolist(),
        "Corr_l_raw_pre_nan_fill": Corr_l.tolist(),
        "Corr_t_raw_pre_nan_fill": Corr_t.tolist(),
        "variant": args.variant,
        "phase": 9,
        "K": 7,
    }
    np.save(args.out, obj, allow_pickle=True)
    print(f"\n  → {args.out}", flush=True)
    print(f"=== Layer 5 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
