"""Per-domain Youden ell*_k calibration with cross-validated expert selection.

Framework (CV top-N):
  For each held-out domain d:
    1. Compute each candidate rater's mean ell across the OTHER 5 domains.
    2. Take the top-N candidates by this cross-domain score as the expert panel
       for domain d.
    3. Compute Youden index J on domain d's ell values, separating these
       experts from all remaining SPARCNET raters not in the candidate pool
       (plus any candidates outside the top-N).

Why CV: defining experts by ell on the held-out domain (and then deriving
the threshold on the same data) is circular and inflates J by construction.
Selecting experts on the OTHER 5 domains is non-circular: the held-out
domain's ell distribution is not used to construct its own expert panel.

Candidate pool: 29 raters in cross_domain_rater_matrix.csv (15 ALL_7
cross-domain raters + 14 additional Bonobo SPARCNET annotators added 2026-05-14).

Comparison vs prior calibration schemes (BOOTSTRAP_N=2000):

  Scheme               | min J | mean J |
  ---------------------|-------|--------|
  ALL_7 (legacy N=15)  | 0.078 | 0.211  | coverage-based; lpd near-zero
  Naive N=29 expansion | 0.000 | 0.128  | non-expert pool collapses
  CV top-14 (this fn)  | 0.372 | 0.521  | 5x stronger min J vs legacy
  CV top-15            | 0.286 | 0.475  | slight degradation

Output: youden_ell_star.json (replaces deprecated interim_ell_star.json)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _REPO_ROOT.parent

# In-repo engine inputs (byte-identical vendored copies of the former
# sibling-repo files — see data/engine_inputs/MANIFEST.json).
_ENGINE_INPUTS = _REPO_ROOT / "data" / "engine_inputs"
MATRIX_CSV = _ENGINE_INPUTS / "cross_domain_rater_matrix.csv"
SDT_FITS_CSV = _ENGINE_INPUTS / "sdt_fits.csv"   # tidy long (219 rows)
OUT_JSON = _REPO_ROOT / "youden_ell_star.json"

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
PANEL_SIZE = 14   # CV top-N expert panel; calibrated as Youden-optimal trade-off
BOOTSTRAP_N = 2000
SEED = 42

# Raters with sigma <= SIGMA_FLOOR hit the optimizer's lower bound and produce
# unreliable ell estimates.  The floor equals the lapse rate (lambda=0.025):
# below it the response model is degenerate.  These are excluded from calibration.
SIGMA_FLOOR = 0.025


# ── data loaders ─────────────────────────────────────────────────────────────
def _load_sdt_fits(domain: str) -> dict[str, float]:
    """Return {sparcnet_rater_name: ell} for converged raters with sigma > floor."""
    rater_ell: dict[str, float] = {}
    n_clipped = 0
    with open(SDT_FITS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["domain"] != domain:
                continue
            if row["converged"].strip().lower() != "true":
                continue
            sigma = float(row["sigma"])
            if sigma <= SIGMA_FLOOR:
                n_clipped += 1
                continue
            rater_ell[row["rater_name"].strip()] = -math.log(sigma)
    if n_clipped:
        print(f"  [{domain}] excluded {n_clipped} rater(s) with sigma <= {SIGMA_FLOOR} (optimizer boundary)")
    return rater_ell


def _load_candidates() -> list[dict[str, str]]:
    """Return [{canonical, sparcnet}] from cross_domain_rater_matrix.csv."""
    out = []
    with open(MATRIX_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append({
                "canonical": row["confirmed_canonical_name"].strip(),
                "sparcnet":  row["confirmed_sparcnet_name"].strip(),
            })
    return out


# ── Youden index on empirical CDFs ───────────────────────────────────────────
def _youden_one(expert_vals: np.ndarray, non_expert_vals: np.ndarray) -> tuple[float, float]:
    """Return (ell*, J) where J = max_ell [F_nonexpert(ell) - F_expert(ell)]."""
    e = np.sort(expert_vals)
    ne = np.sort(non_expert_vals)
    if e.size == 0 or ne.size == 0:
        return float("nan"), float("nan")
    grid = np.sort(np.unique(np.concatenate([e, ne])))
    F_e  = np.searchsorted(e,  grid, side="right") / e.size
    F_ne = np.searchsorted(ne, grid, side="right") / ne.size
    diff = F_ne - F_e
    idx = int(np.argmax(diff))
    return float(grid[idx]), float(diff[idx])


def _run_calibration(expert_vals: np.ndarray, non_expert_vals: np.ndarray,
                     bootstrap_n: int = BOOTSTRAP_N, seed: int = SEED) -> dict:
    l_star, J = _youden_one(expert_vals, non_expert_vals)
    if not np.isfinite(J):
        return {"l_star": float("nan"), "sigma_star": float("nan"), "youden_J": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_expert": int(expert_vals.size), "n_non_expert": int(non_expert_vals.size)}
    rng = np.random.default_rng(seed)
    boots = np.empty(bootstrap_n, dtype=np.float64)
    for b in range(bootstrap_n):
        e_b  = rng.choice(expert_vals,     size=expert_vals.size,     replace=True)
        ne_b = rng.choice(non_expert_vals, size=non_expert_vals.size, replace=True)
        boots[b], _ = _youden_one(e_b, ne_b)
    return {
        "l_star":            float(l_star),
        "sigma_star":        float(math.exp(-l_star)),
        "youden_J":          float(J),
        "ci_low":            float(np.percentile(boots, 2.5)),
        "ci_high":           float(np.percentile(boots, 97.5)),
        "n_expert":          int(expert_vals.size),
        "n_non_expert":      int(non_expert_vals.size),
        "expert_ell_mean":   float(np.mean(expert_vals)),
        "expert_ell_median": float(np.median(expert_vals)),
        "non_expert_ell_mean":   float(np.mean(non_expert_vals)),
        "non_expert_ell_median": float(np.median(non_expert_vals)),
    }


# ── CV expert panel selection ────────────────────────────────────────────────
def _cv_expert_panel(domain: str, fits_by_domain: dict[str, dict[str, float]],
                     candidates: list[dict[str, str]], N: int = PANEL_SIZE) -> set[str]:
    """Select top-N candidates by mean ell across OTHER 5 domains (CV-stable)."""
    other_doms = [d for d in DOMAINS if d != domain]
    scored = []
    for c in candidates:
        sn = c["sparcnet"]
        other_ells = [fits_by_domain[d][sn] for d in other_doms if sn in fits_by_domain[d]]
        if other_ells:
            scored.append((sn, c["canonical"], float(np.mean(other_ells))))
    scored.sort(key=lambda x: -x[2])
    return {s[0] for s in scored[:N]}, [(s[1], s[2]) for s in scored[:N]]


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    candidates = _load_candidates()
    fits_by_domain = {d: _load_sdt_fits(d) for d in DOMAINS}

    print(f"\nCandidate pool: {len(candidates)} raters from cross_domain_rater_matrix.csv")
    print(f"Panel size (CV top-N): {PANEL_SIZE}\n")

    results: dict[str, dict] = {}
    panels: dict[str, list[tuple[str, float]]] = {}

    for d in DOMAINS:
        panel_set, panel_with_scores = _cv_expert_panel(d, fits_by_domain, candidates, PANEL_SIZE)
        panels[d] = panel_with_scores

        rater_ell = fits_by_domain[d]
        expert_vals = np.array(
            [ell for name, ell in rater_ell.items() if name in panel_set],
            dtype=np.float64,
        )
        non_expert_vals = np.array(
            [ell for name, ell in rater_ell.items() if name not in panel_set],
            dtype=np.float64,
        )

        if expert_vals.size == 0 or non_expert_vals.size == 0:
            print(f"  [{d}] SKIP: insufficient data (n_exp={expert_vals.size}, n_non={non_expert_vals.size})")
            continue

        res = _run_calibration(expert_vals, non_expert_vals)
        results[d] = res

        print(f"  {d}: n_expert={res['n_expert']}  n_non_expert={res['n_non_expert']}  "
              f"ell*={res['l_star']:+.4f}  sigma*={res['sigma_star']:.4f}  "
              f"J={res['youden_J']:.4f}  95%CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]")

    payload = {
        "method":         "youden_index_cv_top_N",
        "description":    "CV top-N: expert panel for domain d = top-N candidates by mean ell on OTHER 5 domains.",
        "panel_size_N":   PANEL_SIZE,
        "candidate_pool": "cross_domain_rater_matrix.csv (29 raters: 15 ALL_7 + 14 Bonobo SPARCNET)",
        "n_candidates":   len(candidates),
        "bootstrap_n":    BOOTSTRAP_N,
        "seed":           SEED,
        "sigma_floor":    SIGMA_FLOOR,
        "domains":        DOMAINS,
        "results":        results,
        "panels":         {d: [{"canonical": n, "mean_ell_other_5": float(s)}
                                for n, s in panels[d]]
                            for d in panels},
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved → {OUT_JSON}")

    Js = [results[d]["youden_J"] for d in DOMAINS if d in results]
    print(f"\nOverall: min J = {min(Js):.4f},  mean J = {np.mean(Js):.4f}")

    # Summary table
    print()
    print(f"  {'domain':>6}  {'ell*':>9}  {'sigma*':>7}  {'J':>7}  {'95% CI':>22}  n_exp  n_non")
    print(f"  {'-'*6:>6}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*22}  {'-'*5}  {'-'*5}")
    for d in DOMAINS:
        if d not in results:
            continue
        r = results[d]
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"  {d:>6}  {r['l_star']:>+9.4f}  {r['sigma_star']:>7.4f}  "
              f"{r['youden_J']:>7.4f}  {ci:>22}  {r['n_expert']:>5}  {r['n_non_expert']:>5}")


if __name__ == "__main__":
    main()
