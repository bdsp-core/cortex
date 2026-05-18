"""F2.2b (2026-05-15) — coverage calibration mini-sweep.

Re-runs the AUROC CI coverage validation across a 2×2 grid:

    N_particles      ∈ {500, 1000}
    ess_threshold    ∈ {0.5, 0.9}

The F2.2 baseline (N=500, ess=0.5) showed 95% empirical coverage = 0.933
(passes the ±0.03 band) but mild under-coverage at the 80/90% levels.  The
agent's Inference Calibration Fix Plan hypothesised that (a) too few
particles in the 12-D K=6 state and (b) too-infrequent MH rejuvenation
narrow the posterior.  This sweep tests both knobs jointly.

All 4 cells share the SAME synthetic raters (same truth seed) so the
comparison isolates the engine config, not sampling noise across cells.

One ProcessPoolExecutor pool runs all 4 cells × N_RATERS tasks.

Outputs:
  results/phase2_validation/coverage_sweep.json
  results/phase2_validation/coverage_sweep.md
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
    load_fitted_Sigma, make_state_hier, sample_prior_hier_K,
    choose_item, update, ess, resample_and_rejuvenate, simulate_response,
)
from auroc import auroc_from_l, auroc_quantiles_from_particles_hier  # noqa: E402
from bridge._common import (  # noqa: E402
    _autodetect_banks_dir, load_bank_signals,
)

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

CI_LEVELS = {
    "50%": (0.25, 0.75),
    "80%": (0.10, 0.90),
    "90%": (0.05, 0.95),
    "95%": (0.025, 0.975),
}

# 2×2 grid
GRID = [
    {"N_particles": 500, "ess_threshold_frac": 0.5},
    {"N_particles": 500, "ess_threshold_frac": 0.9},
    {"N_particles": 1000, "ess_threshold_frac": 0.5},
    {"N_particles": 1000, "ess_threshold_frac": 0.9},
]

_SHARED: Dict[str, Any] = {}


def _get_shared() -> Dict[str, Any]:
    if not _SHARED:
        obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
        _SHARED["Corr_l"] = np.asarray(obj["Corr_l"], dtype=float)
        _SHARED["bank_signals"] = load_bank_signals(
            _autodetect_banks_dir(), DOMAINS)
    return _SHARED


def _run_session_fixed_budget(*, true_params, K, max_q, N, seed,
                               Sigma_l, Sigma_t, bank_signals,
                               ess_threshold_frac, n_mh_steps=15):
    rng = np.random.default_rng(seed)
    state = make_state_hier(N, K, r_assumed=0.378, rng=rng,
                            Sigma_l=Sigma_l, Sigma_t=Sigma_t)
    proposal_scale = 2.38 / np.sqrt(2 * K)
    for _ in range(max_q):
        k, s = choose_item(state, bank_signals)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update(state, k, s, y)
        if ess(state["w"]) < ess_threshold_frac * N:
            resample_and_rejuvenate(state, rng, n_mh_steps, proposal_scale)
    return state


def _sweep_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (cell, rater) → per-domain coverage indicators for that cell."""
    shared = _get_shared()
    Corr_l = shared["Corr_l"]
    bank_signals = shared["bank_signals"]
    K = task["K"]
    true_params = task["true_params"]
    l_true = np.asarray(task["l_true"], dtype=float)
    true_auroc = auroc_from_l(l_true)

    state = _run_session_fixed_budget(
        true_params=true_params, K=K, max_q=task["max_q"],
        N=task["N_particles"], seed=task["seed"],
        Sigma_l=Corr_l, Sigma_t=Corr_l, bank_signals=bank_signals,
        ess_threshold_frac=task["ess_threshold_frac"],
    )
    alphas = []
    for (a_lo, a_hi) in CI_LEVELS.values():
        alphas.extend([a_lo, a_hi])
    qs = auroc_quantiles_from_particles_hier(state, alphas=alphas)

    covered = {}  # level -> list[int] per domain
    for i, lvl in enumerate(CI_LEVELS):
        lo = qs[:, 2 * i]
        hi = qs[:, 2 * i + 1]
        covered[lvl] = [
            int(lo[k] <= true_auroc[k] <= hi[k]) for k in range(K)
        ]
    return {
        "cell": task["cell"],
        "rater_idx": task["rater_idx"],
        "covered": covered,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-raters", type=int, default=120,
                   help="Synthetic raters per cell (default 120).")
    p.add_argument("--max-q", type=int, default=200,
                   help="Fixed question budget (default 200).")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Truth + session seed base (default 42).")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Parallel worker processes (default: 14 / cpu-2).")
    args = p.parse_args()

    K = 6
    obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)

    print(f"\n=== F2.2b coverage calibration sweep ===", flush=True)
    print(f"  grid = {GRID}", flush=True)
    print(f"  n_raters/cell={args.n_raters} max_q={args.max_q}", flush=True)

    # Shared synthetic raters (same truth across cells).
    truth_rng = np.random.default_rng(args.seed_base + 100000)
    t_true_all, l_true_all = sample_prior_hier_K(
        N=args.n_raters, K=K, r=0.378, rng=truth_rng,
        Sigma_l=Corr_l, Sigma_t=Corr_l,
    )

    tasks: List[Dict[str, Any]] = []
    for ci, cell in enumerate(GRID):
        cell_name = f"N{cell['N_particles']}_ess{cell['ess_threshold_frac']}"
        for ri in range(args.n_raters):
            true_params = []
            for k in range(K):
                true_params.append(float(t_true_all[ri][k]))
                true_params.append(float(l_true_all[ri][k]))
            tasks.append({
                "cell": cell_name,
                "rater_idx": ri,
                "K": K,
                "true_params": true_params,
                "l_true": l_true_all[ri].tolist(),
                "max_q": args.max_q,
                "N_particles": cell["N_particles"],
                "ess_threshold_frac": cell["ess_threshold_frac"],
                # session seed identical across cells for the same rater
                "seed": args.seed_base + ri,
            })

    t0 = time.time()
    results = parallel_map(
        _sweep_worker, tasks,
        max_workers=args.max_workers,
        desc="sweep sessions",
        ordered=True,
        progress_every=40,
    )

    # Aggregate per cell.
    cells = sorted({f"N{c['N_particles']}_ess{c['ess_threshold_frac']}"
                    for c in GRID})
    summary: Dict[str, Any] = {}
    for cell_name in cells:
        cell_rows = [r for r in results
                     if isinstance(r, dict) and r.get("cell") == cell_name]
        per_level = {}
        for lvl in CI_LEVELS:
            nominal = float(lvl.rstrip("%")) / 100.0
            all_flags = []
            for r in cell_rows:
                all_flags.extend(r["covered"][lvl])
            emp = float(np.mean(all_flags)) if all_flags else float("nan")
            per_level[lvl] = {
                "nominal": nominal,
                "empirical": emp,
                "delta": emp - nominal,
                "pass": abs(emp - nominal) <= 0.03,
            }
        all_pass = all(per_level[l]["pass"] for l in ("80%", "90%", "95%"))
        summary[cell_name] = {
            "per_level": per_level,
            "n_raters": len(cell_rows),
            "pass_80_90_95": all_pass,
        }

    print("\n=== Sweep summary ===", flush=True)
    for cell_name, s in summary.items():
        flags = " ".join(
            f"{lvl}:{s['per_level'][lvl]['empirical']:.3f}"
            f"({s['per_level'][lvl]['delta']:+.3f})"
            for lvl in CI_LEVELS
        )
        verdict = "ALL-PASS(80/90/95)" if s["pass_80_90_95"] else "partial"
        print(f"  {cell_name:18s} {flags}  → {verdict}", flush=True)

    json_path = os.path.join(OUT_DIR, "coverage_sweep.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {"n_raters": args.n_raters, "max_q": args.max_q,
                       "grid": GRID, "K": K},
            "summary": summary,
            "total_seconds": time.time() - t0,
        }, f, indent=2)

    md_path = os.path.join(OUT_DIR, "coverage_sweep.md")
    with open(md_path, "w") as f:
        f.write("# F2.2b Coverage Calibration Sweep\n\n")
        f.write(f"N_raters/cell = {args.n_raters}, max_q = {args.max_q}, "
                f"K = {K}.  Same synthetic raters across all 4 cells.\n\n")
        f.write("| Cell | 50% | 80% | 90% | 95% | 80/90/95 all pass |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cell_name, s in summary.items():
            cells_str = " | ".join(
                f"{s['per_level'][lvl]['empirical']:.3f} "
                f"({s['per_level'][lvl]['delta']:+.3f})"
                for lvl in CI_LEVELS
            )
            f.write(f"| {cell_name} | {cells_str} | "
                    f"{'✅' if s['pass_80_90_95'] else '⚠️'} |\n")
        f.write(f"\nTotal compute: {(time.time()-t0)/60:.1f} min "
                f"({len(tasks)} sessions, parallel).\n")
    print(f"\n  wrote {json_path}", flush=True)
    print(f"  wrote {md_path}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
