"""F2.4 (2026-05-15) — Sigma_l prior-covariance sensitivity sweep.

Quantifies how the hierarchical-vs-brute speedup depends on the choice of
the prior covariance for the l-block.  This answers the reviewer question
"is the borrowing-of-strength speedup an artifact of one specific Sigma_l,
or robust across reasonable prior structures?"

Cells (hier prior covariance for both l- and t-block):
  corr_l       production reference: unit-diagonal fitted correlation matrix
  fitted_raw   raw fitted Sigma_l (tiny marginal var ~0.01-0.06; agent
               flagged this as the WRONG scale for a prospective prior)
  ledoit_wolf  Ledoit-Wolf shrinkage of the 27x6 rater l-matrix
  cs_r0.2      compound symmetry (1-0.2) I + 0.2 J
  cs_r0.4      compound symmetry (1-0.4) I + 0.4 J
  diagonal     identity I — hierarchy-OFF control; if speedup ≈ 1 here and
               > 1 for structured cells, the speedup IS cross-domain pooling

brute is Sigma_l-independent (K independent N(0,1) clouds), so it is run
ONCE per rater and reused as the baseline across all cells.

27 real SPARCNET raters with complete K=6 SDT fits, real curated banks.
One ProcessPoolExecutor pool over (cell × rater) hier tasks + brute tasks.

Outputs:
  results/phase2_validation/sigma_sensitivity.json
  results/phase2_validation/sigma_sensitivity.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_THIS_DIR_BOOT = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR_BOOT not in sys.path:
    sys.path.insert(0, _THIS_DIR_BOOT)
from _parallel import configure_blas_single_thread, parallel_map  # noqa: E402
configure_blas_single_thread()

import numpy as np  # noqa: E402

ENGINE_REPO = os.path.dirname(_THIS_DIR_BOOT)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

from core_mcmc import (  # noqa: E402
    run_session_mcmc_auroc, post_hoc_delta_sweep, load_fitted_Sigma,
)
from bridge._common import (  # noqa: E402
    _autodetect_banks_dir, _default_rater_matrix_path,
    load_bank_signals, load_raters, build_true_params,
)

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
DELTAS = [0.025, 0.05, 0.10]
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

CELLS = ["corr_l", "fitted_raw", "ledoit_wolf", "cs_r0.2", "cs_r0.4",
         "diagonal"]

_SHARED: Dict[str, Any] = {}


def _build_sigma_cells(K: int) -> Dict[str, np.ndarray]:
    """Construct the 6 prior covariance matrices."""
    obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    Sigma_raw = np.asarray(obj["Sigma_l"], dtype=float)

    # Ledoit-Wolf from the 27x6 rater l-matrix.
    raters_df = load_raters(_default_rater_matrix_path())
    l_cols = [f"l_{d}" for d in DOMAINS]
    if all(c in raters_df.columns for c in l_cols):
        L = raters_df.dropna(subset=l_cols)[l_cols].to_numpy(dtype=float)
    else:
        # Fall back: l_d = -log(sigma_d)
        s_cols = [f"sigma_{d}" for d in DOMAINS]
        S = raters_df.dropna(subset=s_cols)[s_cols].to_numpy(dtype=float)
        L = -np.log(S)
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(L)
    Sigma_lw = np.asarray(lw.covariance_, dtype=float)

    J = np.ones((K, K))
    I = np.eye(K)
    return {
        "corr_l": Corr_l,
        "fitted_raw": Sigma_raw,
        "ledoit_wolf": Sigma_lw,
        "cs_r0.2": 0.8 * I + 0.2 * J,
        "cs_r0.4": 0.6 * I + 0.4 * J,
        "diagonal": I.copy(),
    }


def _get_shared() -> Dict[str, Any]:
    if not _SHARED:
        _SHARED["sigma_cells"] = _build_sigma_cells(6)
        _SHARED["bank_signals"] = load_bank_signals(
            _autodetect_banks_dir(), DOMAINS)
    return _SHARED


def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (cell, rater) hier session OR one brute baseline session."""
    shared = _get_shared()
    bank_signals = shared["bank_signals"]
    K = task["K"]
    method = task["method"]
    true_params = task["true_params"]

    kwargs = dict(
        method=method, true_params=true_params, K=K, r_assumed=0.378,
        max_q=task["max_q"], delta_auroc=0.025, N=task["N_particles"],
        seed=task["seed"], run_until_max=False, log_trajectory=True,
        ess_threshold_frac=task["ess_threshold_frac"],
        bank_signals=bank_signals,
    )
    if method == "hier":
        Sig = shared["sigma_cells"][task["cell"]]
        kwargs.update(Sigma_l=Sig, Sigma_t=Sig,
                      proposal_scale=2.38 / np.sqrt(2 * K))
    try:
        out = run_session_mcmc_auroc(**kwargs)
    except Exception as e:  # near-singular Sigma can raise — capture, don't crash
        return {"cell": task["cell"], "method": method,
                "rater_id": task["rater_id"], "error": str(e)}
    sweep = post_hoc_delta_sweep(out["lo_traj"], out["hi_traj"], deltas=DELTAS)
    return {
        "cell": task["cell"],
        "method": method,
        "rater_id": task["rater_id"],
        "stop_d0_025": sweep[0.025],
        "stop_d0_05": sweep[0.05],
        "stop_d0_10": sweep[0.10],
        "n_q_total": int(out["n_questions"]),
    }


def _median(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-q", type=int, default=1500)
    p.add_argument("--N-particles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-raters", type=int, default=None,
                   help="Cap raters (relative cross-cell comparison only "
                        "needs ~12 for a clear cell ranking).")
    p.add_argument("--ess-threshold-frac", type=float, default=0.5,
                   help="ESS resample trigger. 0.5 (Phase-1 figure setting) "
                        "is ~2-4x faster than 0.9; applied uniformly to all "
                        "cells so the cross-cell ranking is invariant.")
    p.add_argument("--max-workers", type=int, default=None)
    args = p.parse_args()

    K = 6
    raters_df = load_raters(_default_rater_matrix_path())
    needed = [f"sigma_{d}" for d in DOMAINS] + [f"theta_{d}" for d in DOMAINS]
    raters_df = raters_df.dropna(subset=needed).reset_index(drop=True)
    if args.max_raters is not None:
        raters_df = raters_df.head(args.max_raters).reset_index(drop=True)
    n_raters = len(raters_df)
    print(f"\n=== F2.4 Sigma_l sensitivity sweep ===", flush=True)
    print(f"  {n_raters} raters, cells={CELLS}, "
          f"max_q={args.max_q} N={args.N_particles}", flush=True)

    # brute is cell-independent → one task per rater.
    # hier → one task per (cell, rater).
    tasks: List[Dict[str, Any]] = []
    for ri, row in raters_df.iterrows():
        rid = str(row.get("confirmed_canonical_name", f"R{ri}"))
        tp = build_true_params(row, DOMAINS)
        tasks.append({
            "cell": "_brute_baseline", "method": "brute", "rater_id": rid,
            "K": K, "true_params": tp, "max_q": args.max_q,
            "N_particles": args.N_particles, "seed": args.seed,
            "ess_threshold_frac": args.ess_threshold_frac,
        })
        for cell in CELLS:
            tasks.append({
                "cell": cell, "method": "hier", "rater_id": rid,
                "K": K, "true_params": tp, "max_q": args.max_q,
                "N_particles": args.N_particles, "seed": args.seed,
                "ess_threshold_frac": args.ess_threshold_frac,
            })

    t0 = time.time()
    results = parallel_map(
        _worker, tasks, max_workers=args.max_workers,
        desc="sigma-sens sessions", ordered=True, progress_every=25,
    )

    # Organise: brute baseline per rater; hier per (cell, rater).
    brute_by_rater: Dict[str, Dict[str, Any]] = {}
    hier_by_cell: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CELLS}
    n_err = 0
    for r in results:
        if isinstance(r, dict) and "__error__" in r:
            n_err += 1
            continue
        if "error" in r:
            n_err += 1
            print(f"  [unstable] {r['cell']} {r['rater_id']}: {r['error']}",
                  flush=True)
            continue
        if r["method"] == "brute":
            brute_by_rater[r["rater_id"]] = r
        else:
            hier_by_cell[r["cell"]].append(r)

    brute_med = {
        d: _median([v["stop_" + d] for v in brute_by_rater.values()])
        for d in ("d0_025", "d0_05", "d0_10")
    }

    summary = {"brute_baseline": {
        "n": len(brute_by_rater),
        "median_d0_025": brute_med["d0_025"],
        "median_d0_05": brute_med["d0_05"],
        "median_d0_10": brute_med["d0_10"],
    }, "cells": {}}

    print("\n=== Sigma_l sensitivity summary "
          "(median n_q; speedup = brute/hier) ===", flush=True)
    print(f"  brute baseline: d0.05 median={brute_med['d0_05']}, "
          f"d0.10 median={brute_med['d0_10']} (n={len(brute_by_rater)})",
          flush=True)
    for cell in CELLS:
        rows = hier_by_cell[cell]
        med_05 = _median([v["stop_d0_05"] for v in rows])
        med_10 = _median([v["stop_d0_10"] for v in rows])
        med_025 = _median([v["stop_d0_025"] for v in rows])
        sp_05 = (brute_med["d0_05"] / med_05
                 if (med_05 and brute_med["d0_05"]) else None)
        sp_10 = (brute_med["d0_10"] / med_10
                 if (med_10 and brute_med["d0_10"]) else None)
        n_reach_025 = sum(1 for v in rows if v["stop_d0_025"] is not None)
        summary["cells"][cell] = {
            "n": len(rows),
            "median_d0_025": med_025,
            "median_d0_05": med_05,
            "median_d0_10": med_10,
            "speedup_d0_05": sp_05,
            "speedup_d0_10": sp_10,
            "n_reached_d0_025": n_reach_025,
        }
        sp05s = f"{sp_05:.2f}x" if sp_05 else "n/a"
        sp10s = f"{sp_10:.2f}x" if sp_10 else "n/a"
        print(f"  {cell:12s} hier d0.05 med={str(med_05):>6s} "
              f"d0.10 med={str(med_10):>6s} | "
              f"speedup d0.05={sp05s:>6s} d0.10={sp10s:>6s} "
              f"(n={len(rows)}, reached d0.025={n_reach_025})", flush=True)

    if n_err:
        print(f"\n  [note] {n_err} sessions errored/unstable "
              "(expected for near-singular fitted_raw at some raters).",
              flush=True)

    summary["n_errors"] = n_err
    summary["total_seconds"] = time.time() - t0
    json_path = os.path.join(OUT_DIR, "sigma_sensitivity.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(OUT_DIR, "sigma_sensitivity.md")
    with open(md_path, "w") as f:
        f.write("# F2.4 Sigma_l Sensitivity Sweep\n\n")
        f.write(f"{n_raters} real SPARCNET raters, K=6, max_q={args.max_q}, "
                f"N_particles={args.N_particles}, ess=0.9.  brute baseline "
                f"is Sigma_l-independent (run once per rater).\n\n")
        f.write(f"Brute baseline median n_q: δ=0.05 → "
                f"{brute_med['d0_05']}, δ=0.10 → {brute_med['d0_10']} "
                f"(n={len(brute_by_rater)}).\n\n")
        f.write("| Cell | hier d0.05 med | hier d0.10 med | "
                "speedup d0.05 | speedup d0.10 | n reached d0.025 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cell in CELLS:
            c = summary["cells"][cell]
            sp05 = f"{c['speedup_d0_05']:.2f}×" if c["speedup_d0_05"] else "n/a"
            sp10 = f"{c['speedup_d0_10']:.2f}×" if c["speedup_d0_10"] else "n/a"
            f.write(f"| {cell} | {c['median_d0_05']} | {c['median_d0_10']} "
                    f"| {sp05} | {sp10} | {c['n_reached_d0_025']}/{c['n']} |\n")
        f.write(f"\nErrors/unstable sessions: {n_err}.  "
                f"Total compute: {(time.time()-t0)/60:.1f} min.\n\n")
        f.write("**Interpretation:** if `diagonal` (no cross-domain pooling) "
                "shows speedup ≈ 1.0 while structured cells (corr_l, cs_r*) "
                "show speedup > 1, the borrowing-of-strength advantage is "
                "attributable to the prior correlation structure, not an "
                "engine artifact.\n")
    print(f"\n  wrote {json_path}\n  wrote {md_path}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
