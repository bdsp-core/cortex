"""Phase 7 sub-step 2 — Tier-2 OC synthetic ℓ-grid simulation study.

Faithful port from the methodology reference (`scripts/run_phase4_
simstudy.py` in the sibling methodology repo; the "F4.2 Synthetic ℓ-grid
simulation study (Paper-1 Results centerpiece)"). Path-only edits required
by the unified-repo layout — see PHASE7_AUDIT.md §sub-2 + commit message.
Renamed `run_tier2_oc_simstudy.py` to match the plan's terminology
("Tier-2 OC simulator"). The reference K=6 result documented in the
methodology repo is the carried evidence; this port adds **K=7** to the
K_GRID so the unified-repo deliverable covers the production deployment dim.

Design (preserved verbatim from the reference, K=7 added):
  • true ℓ grid (homogeneous across all K domains):
      L_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
      → true AUROC ≈ [0.62, 0.69, 0.77, 0.84, 0.89, 0.91]
  • K ∈ {2, 4, 6, 7, 8} — **K=7 added (Phase 7 sub-2)**
  • Σ_l prior condition (affects ONLY `hier`):
      - "empirical"   : fitted Corr_l (K=6) / matched-mean CS for K≠6
      - "independent" : I_K  (pooling disabled → hier ≈ brute sanity)
      - "cs0.7"       : 0.7·J + 0.3·I  (strong compound-symmetry pool)
  • methods: random, brute, hier
  • δ: a single run at the tightest δ with run_until_max + log_trajectory;
    post_hoc_delta_sweep reports stop steps at {0.025, 0.05, 0.10}.
    Not reaching δ within the per-method/per-K cap is a legitimate
    operating characteristic → right-censored (honest OC).
  • N=1000 particles (production; matches cert_config v13 / Phase-3.5).

Sessions per replicate: random+brute 6·5·2 = 60; hier 6·5·3 = 90 → 150
(K_GRID grew from 4 entries to 5 ⇒ +30 sessions/rep over methodology).
Bootstrap CIs computed at plot time (separate, this runner emits raw
per-session rows).

Output: results/phase2_validation/tier2_oc_simstudy_rows.json
        (plan §"Phase 7" places sub-2 OC tables under
         `results/phase2_validation/`).

Usage:
  # pilot at K=7 only (this sub-step's deliverable, ~minutes)
  python scripts/run_tier2_oc_simstudy.py --n-reps 1 --k-grid 7
  # paper-grade full grid (multi-hour)
  python scripts/run_tier2_oc_simstudy.py --n-reps 25
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
from _parallel import (  # noqa: E402
    configure_blas_single_thread, parallel_map, count_errors,
)
configure_blas_single_thread()

import numpy as np  # noqa: E402

# UNIFIED-LAYOUT PATH EDIT (vs methodology reference): in the methodology
# repo, the engine modules (`core_mcmc`, `auroc`) live at the repo root;
# in the unified repo, they live in `engine/` (Phase-2 layout decision —
# see `tests/conftest.py:39-45` for the same path setup pattern). We
# also keep `Sigma_l_fitted.npy` resolution at the repo root (where the
# real file lives; the `engine/` symlink also resolves).
_REPO = os.path.dirname(_THIS_DIR_BOOT)
ENGINE_REPO = _REPO
for _p in (ENGINE_REPO, os.path.join(ENGINE_REPO, "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_mcmc import (  # noqa: E402
    run_session_mcmc_auroc, post_hoc_delta_sweep, load_fitted_Sigma,
)
from auroc import auroc_from_l  # noqa: E402

# UNIFIED-LAYOUT PATH EDIT: plan §"Phase 7" sub-2 places Tier-2 OC
# tables under `results/phase2_validation/` (the methodology
# reference wrote `results/phase4_simstudy/`).
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

L_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
# K_GRID: K=7 added (Phase 7 sub-2; production deployment dim =
# 6 IIIC + combined_spike + other). Methodology reference was {2,4,6,8}.
K_GRID = [2, 4, 6, 7, 8]
SIGMA_L_CONDS = ["empirical", "independent", "cs0.7"]
METHODS = ["random", "brute", "hier"]
DELTAS = [0.025, 0.05, 0.10]

# Per-method × per-K question caps. K=7 caps interpolated linearly
# between K=6 and K=8 (the methodology reference cells flanking it).
MAX_Q_BY_METHOD_K = {
    "hier":   {2: 1500, 4: 2500, 6: 4000, 7: 5000, 8: 6000},
    "brute":  {2: 1500, 4: 2500, 6: 4000, 7: 5000, 8: 6000},
    "random": {2: 8000, 4: 14000, 6: 20000, 7: 24000, 8: 28000},
}
N_PARTICLES = 1000
TIGHTEST_DELTA = 0.025


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


# ── lazy per-worker shared read-only cache (spawn-safe) ───────────────
_SHARED: Dict[str, Any] = {}


def _get_corr_l6() -> np.ndarray:
    if "Corr_l6" not in _SHARED:
        _SHARED["Corr_l6"] = np.asarray(
            load_fitted_Sigma(
                os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))["Corr_l"],
            dtype=float)
    return _SHARED["Corr_l6"]


def _sigma_l(cond: str, K: int) -> np.ndarray:
    """Hier prior covariance for a given condition and K."""
    if cond == "independent":
        return np.eye(K)
    if cond == "cs0.7":
        return 0.7 * np.ones((K, K)) + 0.3 * np.eye(K)
    if cond == "empirical":
        C6 = _get_corr_l6()
        if K == 6:
            return C6
        # matched-mean compound-symmetry approximation for K≠6
        mean_off = float((C6.sum() - 6) / (6 * 6 - 6))
        return mean_off * np.ones((K, K)) + (1.0 - mean_off) * np.eye(K)
    raise ValueError(f"unknown Σ_l condition {cond!r}")


def _sim_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (ℓ, K, method, Σ_l-cond, rep) Mode-A session → summary row."""
    K = task["K"]
    method = task["method"]
    l_val = task["l_val"]
    cond = task["sigma_l_cond"]
    seed = task["seed"]

    # Homogeneous skill ℓ across all K domains; criterion t≡0.
    true_params: List[float] = []
    for _ in range(K):
        true_params.append(0.0)        # t_k
        true_params.append(l_val)      # l_k
    true_auroc = float(auroc_from_l(np.array([l_val]))[0])

    t0 = time.time()
    kwargs = dict(
        method=method, true_params=true_params, K=K, r_assumed=0.378,
        max_q=task["max_q"], delta_auroc=TIGHTEST_DELTA, N=N_PARTICLES,
        seed=seed, run_until_max=True, log_trajectory=True,
        ess_threshold_frac=0.5,
    )
    if method == "hier":
        Sig = _sigma_l(cond, K)
        kwargs.update(Sigma_l=Sig, Sigma_t=Sig,
                      proposal_scale=2.38 / np.sqrt(2 * K))
    out = run_session_mcmc_auroc(**kwargs)
    dt = time.time() - t0

    sweep = post_hoc_delta_sweep(out["lo_traj"], out["hi_traj"],
                                 deltas=DELTAS)
    row = {
        "l_val": l_val,
        "true_auroc": true_auroc,
        "K": K,
        "method": method,
        "sigma_l_cond": cond if method == "hier" else "na",
        "seed": seed,
        "max_q": task["max_q"],
        "n_q_total": int(out["n_questions"]),
        "duration_s": dt,
    }
    for d in DELTAS:
        key = f"d{str(d).replace('.', '_')}"
        stop = sweep[d]
        # post_hoc_delta_sweep returns None iff HW<δ was never reached
        # within max_q; otherwise a valid 1-indexed stop step (which may
        # legitimately equal max_q).  None ⇒ right-censored (honest OC).
        reached = stop is not None
        row[f"stop_{key}"] = int(stop) if reached else None
        row[f"reached_{key}"] = bool(reached)
    return row


def build_tasks(n_reps: int, seed_base: int,
                k_grid: List[int]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for rep in range(n_reps):
        seed = seed_base + rep
        for l_val in L_GRID:
            for K in k_grid:
                # random/brute: Σ_l-agnostic → run ONCE (cond label "na")
                for method in ("random", "brute"):
                    tasks.append({
                        "l_val": l_val, "K": K, "method": method,
                        "sigma_l_cond": "na", "seed": seed,
                        "max_q": MAX_Q_BY_METHOD_K[method][K],
                    })
                # hier: crossed with all Σ_l conditions
                for cond in SIGMA_L_CONDS:
                    tasks.append({
                        "l_val": l_val, "K": K, "method": "hier",
                        "sigma_l_cond": cond, "seed": seed,
                        "max_q": MAX_Q_BY_METHOD_K["hier"][K],
                    })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=25,
                    help="Replicate seeds per cell. Default 25 "
                         "(paper-grade; pilot uses --n-reps 1).")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--max-workers", type=int, default=10,
                    help="Parallel workers. Default 10.")
    ap.add_argument("--k-grid", type=int, nargs="+", default=K_GRID,
                    help="K values to run (default: full {2,4,6,7,8}; "
                         "the Phase 7 sub-2 K=7 pilot uses --k-grid 7).")
    ap.add_argument("--out", type=str, default=os.path.join(
        OUT_DIR, "tier2_oc_simstudy_rows.json"))
    cli = ap.parse_args()

    # Validate every requested K has cap entries
    for K in cli.k_grid:
        for m in ("random", "brute", "hier"):
            if K not in MAX_Q_BY_METHOD_K[m]:
                raise SystemExit(
                    f"K={K} not in MAX_Q_BY_METHOD_K[{m!r}]: "
                    f"{sorted(MAX_Q_BY_METHOD_K[m].keys())}")

    tasks = build_tasks(cli.n_reps, cli.seed_base, cli.k_grid)
    print(f"=== Tier-2 OC simstudy: {cli.n_reps} reps × K_grid={cli.k_grid} "
          f"→ {len(tasks)} sessions ===", flush=True)
    print(f"  ℓ-grid={L_GRID}  Σ_l={SIGMA_L_CONDS}", flush=True)

    t_total = time.time()
    results = parallel_map(
        _sim_worker, tasks,
        max_workers=cli.max_workers,
        desc="sim sessions",
        ordered=True,
        progress_every=20,
    )
    n_err = count_errors(results)
    rows = [r for r in results
            if not (isinstance(r, dict) and "__error__" in r)]
    wall = time.time() - t_total

    payload = {
        "rows": rows,
        "config": {
            "L_GRID": L_GRID, "K_GRID": list(cli.k_grid),
            "SIGMA_L_CONDS": SIGMA_L_CONDS, "METHODS": METHODS,
            "DELTAS": DELTAS, "N_PARTICLES": N_PARTICLES,
            "TIGHTEST_DELTA": TIGHTEST_DELTA,
            "MAX_Q_BY_METHOD_K": MAX_Q_BY_METHOD_K,
            "n_reps": cli.n_reps, "seed_base": cli.seed_base,
            "n_sessions": len(tasks), "n_errors": n_err,
            "wall_s": wall,
            "true_auroc_grid": [
                float(auroc_from_l(np.array([l]))[0]) for l in L_GRID],
        },
    }
    with open(cli.out, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    print(f"\n  → wrote {len(rows)} rows ({n_err} errors) to {cli.out}",
          flush=True)
    print(f"  wall={wall:.1f}s "
          f"({wall / max(1, len(tasks)):.2f}s/session amortized)",
          flush=True)


if __name__ == "__main__":
    main()
