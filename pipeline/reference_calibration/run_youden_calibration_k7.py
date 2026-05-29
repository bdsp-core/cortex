"""K=7 Youden ell* calibration — uniform CV-top-N across spike + 6 IIIC.

Phase-9 Layer 4 port of the K=6 byte-md5-pinned `run_youden_calibration.py`
(md5 `f8bcfbd61b7a89c70e4dfec3cf8227db`) extended to all 7 K=7 tasks. The
algorithm is byte-equivalent for IIIC tasks except that the cross-domain
average now sums over 6 OTHER tasks (5 other IIIC + spike) instead of 5.

Framework (CV top-N for K=7):
  For each held-out task k:
    1. Compute each candidate rater's mean ell across the OTHER 6 tasks.
    2. Take the top-N candidates by this cross-task score as the expert
       panel for task k.
    3. Compute Youden index J on task k's ell values, separating these
       experts from all remaining sdt_fits raters.

Why CV: defining experts by ell on the held-out task (then deriving the
threshold on the same data) is circular and inflates J. Selecting experts
on the OTHER 6 tasks is non-circular; the held-out task's ell distribution
is not used to construct its own expert panel.

K=7 specifics:
  * spike held-out: cross-task average = 6 IIIC tasks → defines "spike
    expert panel" by cross-IIIC skill (psychometrically defensible —
    a rater skilled at IIIC interpretation is the standard expert proxy).
  * IIIC k held-out: cross-task average = 5 other IIIC + spike → adds
    spike skill as one weight in the cross-domain expert score.

Candidate pool: 29 raters in cross_domain_rater_matrix_k7.csv (the 29
Q2-locked cross-IIIC pool, extended with spike fits per Phase-9 Layer 3).

Why uniform CV-top-N for all 7 tasks (and not the v13 spike 70/30 split):
  - Single methodology across 7 tasks → consistent Nature manuscript story
  - The 29-candidate pool already includes raters with strong cross-IIIC
    skill; this transfers naturally to a spike expert proxy
  - Avoids the K=6 spike-special-case methodology that complicated
    reviewer narratives

Output: `youden_ell_star_k7.json` (sibling to v13's `youden_ell_star.json`;
this file IS NOT a replacement — v13 stays unchanged for audit chain).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _REPO_ROOT.parents[1]  # repo root from
                                       # pipeline/reference_calibration/

# K=7 engine inputs (Phase-9 Layer 3 outputs)
_ENGINE_INPUTS = _PROJECT_ROOT / "data" / "engine_inputs"
MATRIX_CSV_K7 = _ENGINE_INPUTS / "cross_domain_rater_matrix_k7.csv"
SDT_FITS_CSV_K7 = _ENGINE_INPUTS / "sdt_fits_k7.csv"
OUT_JSON_K7 = _REPO_ROOT / "youden_ell_star_k7.json"

DOMAINS_K7 = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
PANEL_SIZE = 14
BOOTSTRAP_N = 2000
SEED = 42

# Same SIGMA_FLOOR as K=6 (per the D2 byte-md5-pinned reference);
# raters at sigma <= lapse rate are at the optimizer boundary.
SIGMA_FLOOR = 0.025


# ── data loaders ─────────────────────────────────────────────────────────────
def _load_sdt_fits(domain: str) -> dict[str, float]:
    """Return {rater_name: ell} for converged raters with sigma > floor.

    Reads from sdt_fits_k7.csv (Phase-9 Layer 3 concat of v13 IIIC + v13
    spike sdt_fits; D2 byte-md5 invariant preserved upstream).
    """
    rater_ell: dict[str, float] = {}
    n_clipped = 0
    with open(SDT_FITS_CSV_K7, newline="") as fh:
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
        print(f"  [{domain}] excluded {n_clipped} rater(s) with sigma <= "
              f"{SIGMA_FLOOR} (optimizer boundary)")
    return rater_ell


def _load_candidates() -> list[dict[str, str]]:
    """Return the 29 Q2-locked candidates from cross_domain_rater_matrix_k7.csv."""
    out = []
    with open(MATRIX_CSV_K7, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append({
                "canonical": row["confirmed_canonical_name"].strip(),
                "sparcnet":  row["confirmed_sparcnet_name"].strip(),
            })
    return out


# ── Youden index on empirical CDFs (byte-equivalent to K=6 reference) ───────
def _youden_one(expert_vals: np.ndarray,
                non_expert_vals: np.ndarray) -> tuple[float, float]:
    """Return (ell*, J) where J = max_ell [F_nonexpert(ell) - F_expert(ell)]."""
    e = np.sort(expert_vals)
    ne = np.sort(non_expert_vals)
    if e.size == 0 or ne.size == 0:
        return float("nan"), float("nan")
    grid = np.sort(np.unique(np.concatenate([e, ne])))
    F_e = np.searchsorted(e, grid, side="right") / e.size
    F_ne = np.searchsorted(ne, grid, side="right") / ne.size
    diff = F_ne - F_e
    idx = int(np.argmax(diff))
    return float(grid[idx]), float(diff[idx])


def _run_calibration(expert_vals: np.ndarray,
                     non_expert_vals: np.ndarray,
                     bootstrap_n: int = BOOTSTRAP_N,
                     seed: int = SEED) -> dict:
    """Compute Youden ell*, sigma*, J + 95% bootstrap CI."""
    l_star, J = _youden_one(expert_vals, non_expert_vals)
    if not np.isfinite(J):
        return {"l_star": float("nan"), "sigma_star": float("nan"),
                "youden_J": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_expert": int(expert_vals.size),
                "n_non_expert": int(non_expert_vals.size)}
    rng = np.random.default_rng(seed)
    boots = np.empty(bootstrap_n, dtype=np.float64)
    for b in range(bootstrap_n):
        e_b = rng.choice(expert_vals, size=expert_vals.size, replace=True)
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


# ── CV expert panel selection (K=7) ─────────────────────────────────────────
def _cv_expert_panel(domain: str,
                     fits_by_domain: dict[str, dict[str, float]],
                     candidates: list[dict[str, str]],
                     N: int = PANEL_SIZE) -> tuple[set[str], list[tuple]]:
    """Top-N candidates by mean ell across OTHER 6 tasks (K=7 extension)."""
    other_doms = [d for d in DOMAINS_K7 if d != domain]
    scored = []
    for c in candidates:
        sn = c["sparcnet"]
        other_ells = [fits_by_domain[d][sn] for d in other_doms
                      if sn in fits_by_domain[d]]
        if other_ells:
            scored.append((sn, c["canonical"], float(np.mean(other_ells))))
    scored.sort(key=lambda x: -x[2])
    return {s[0] for s in scored[:N]}, [(s[1], s[2]) for s in scored[:N]]


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    candidates = _load_candidates()
    fits_by_domain = {d: _load_sdt_fits(d) for d in DOMAINS_K7}

    print(f"\nCandidate pool: {len(candidates)} raters from "
          f"cross_domain_rater_matrix_k7.csv")
    print(f"Panel size (CV top-N, OTHER-6-task average): {PANEL_SIZE}\n")

    results: dict[str, dict] = {}
    panels: dict[str, list[tuple[str, float]]] = {}

    for d in DOMAINS_K7:
        panel_set, panel_with_scores = _cv_expert_panel(
            d, fits_by_domain, candidates, PANEL_SIZE)
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
            print(f"  [{d}] SKIP: insufficient data "
                  f"(n_exp={expert_vals.size}, n_non={non_expert_vals.size})")
            continue

        res = _run_calibration(expert_vals, non_expert_vals)
        results[d] = res

        print(f"  {d:>5}: n_expert={res['n_expert']}  "
              f"n_non_expert={res['n_non_expert']}  "
              f"ell*={res['l_star']:+.4f}  sigma*={res['sigma_star']:.4f}  "
              f"J={res['youden_J']:.4f}  "
              f"95%CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]")

    payload = {
        "method":         "youden_index_cv_top_N_k7",
        "description":    ("CV top-N (K=7): expert panel for task k = top-N "
                           "candidates by mean ell on OTHER 6 tasks (5 IIIC "
                           "+ spike for IIIC; 6 IIIC for spike)."),
        "phase":          9,
        "K":              7,
        "panel_size_N":   PANEL_SIZE,
        "candidate_pool": ("cross_domain_rater_matrix_k7.csv (29 raters: "
                           "29 Q2-locked + spike fits joined by canonical name)"),
        "n_candidates":   len(candidates),
        "bootstrap_n":    BOOTSTRAP_N,
        "seed":           SEED,
        "sigma_floor":    SIGMA_FLOOR,
        "domains":        DOMAINS_K7,
        "results":        results,
        "panels":         {d: [{"canonical": n, "mean_ell_other_6": float(s)}
                                for n, s in panels[d]]
                            for d in panels},
    }
    with open(OUT_JSON_K7, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved → {OUT_JSON_K7}")

    Js = [results[d]["youden_J"] for d in DOMAINS_K7 if d in results]
    print(f"\nOverall: min J = {min(Js):.4f},  mean J = {np.mean(Js):.4f}")

    # Summary table
    print()
    print(f"  {'task':>6}  {'ell*':>9}  {'sigma*':>7}  {'J':>7}  "
          f"{'95% CI':>22}  n_exp  n_non")
    print(f"  {'-'*6:>6}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*22}  "
          f"{'-'*5}  {'-'*5}")
    for d in DOMAINS_K7:
        if d not in results:
            continue
        r = results[d]
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"  {d:>6}  {r['l_star']:>+9.4f}  {r['sigma_star']:>7.4f}  "
              f"{r['youden_J']:>7.4f}  {ci:>22}  "
              f"{r['n_expert']:>5}  {r['n_non_expert']:>5}")


if __name__ == "__main__":
    main()
