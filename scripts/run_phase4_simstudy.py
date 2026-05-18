"""F4.2 — Synthetic ℓ-grid simulation study (Paper 1 Results centerpiece).

Phase-1 (F-rig) characterised the ablation ladder on 27 *fixed* real
SPARCNET raters — scattered points, not a controlled grid.  This study
fills the controlled operating-characteristic *surface*: questions-to-δ
as a smooth function of the examinee's true skill ℓ, the number of
domains K, and the assumed cross-domain prior structure Σ_l, with the
data-generating process exactly known and bootstrap CIs over replicate
seeds.

Design
------
  • true ℓ grid (homogeneous across all K domains — one skill level per
    synthetic examinee; the standard psychometric OC design.  Criterion
    t≡0: AUROC is independent of t, see auroc.py):
        L_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
        → true AUROC ≈ [0.62, 0.69, 0.77, 0.84, 0.89, 0.91]
  • K ∈ {2, 4, 6, 8}
  • Σ_l prior condition (affects ONLY `hier`; random/brute use an
    independent N(0,1) prior by construction so they are run ONCE per
    (ℓ, K, rep), not crossed with Σ_l — avoids 3× identical compute):
        - "empirical"   : fitted Corr_l (K=6) / matched-mean CS for K≠6
        - "independent" : I_K  (pooling disabled → hier ≈ brute sanity)
        - "cs0.7"       : 0.7·J + 0.3·I  (strong compound-symmetry pool)
  • methods: random, brute, hier
  • δ: a single run at the tightest δ with run_until_max + log_trajectory;
    post_hoc_delta_sweep reports stop steps at {0.025, 0.05, 0.10}.
    Not reaching a δ within the per-method/per-K cap is a legitimate
    operating characteristic and is recorded as right-censored
    (stop_d* = None, reached_d* = False), exactly as Phase-1 framed it.
  • N=1000 particles (production; matches Phase-1 F-rig / cert_config v11).
  • Engine is the F4.1 build (n_subsample threaded through).  NOTE: with
    the default 11-point SIGNAL_GRID (no item bank), n=11 < n_coarse so
    coarse-to-fine is *inert* (exact full grid) — F4.1's speedup is a
    real-bank property exercised by the F4.3 OC campaign, not here.

Sessions per replicate: random+brute 6·4·2 = 48; hier 6·4·3 = 72 → 120.
Bootstrap CIs are computed at plot time (F4.4) from the raw rows so the
expensive run is reusable; the runner only emits raw per-session rows.

Output: results/phase4_simstudy/simstudy_rows.json
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

ENGINE_REPO = os.path.dirname(_THIS_DIR_BOOT)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

from core_mcmc import (  # noqa: E402
    run_session_mcmc_auroc, post_hoc_delta_sweep, load_fitted_Sigma,
)
from auroc import auroc_from_l  # noqa: E402

OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase4_simstudy")
os.makedirs(OUT_DIR, exist_ok=True)

L_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
K_GRID = [2, 4, 6, 8]
SIGMA_L_CONDS = ["empirical", "independent", "cs0.7"]
METHODS = ["random", "brute", "hier"]
DELTAS = [0.025, 0.05, 0.10]

# Per-method × per-K question caps.  hier/brute reach δ=0.025 with a
# moderate budget (F-rig K=6 median ≈1.2k); low-ℓ (wide AUROC CI) needs
# headroom.  random has no adaptivity/pooling so it needs a far larger
# budget but is ~25× cheaper per question, so the large cap is
# affordable.  Non-reaching within the cap → right-censored (honest OC).
MAX_Q_BY_METHOD_K = {
    "hier":   {2: 1500, 4: 2500, 6: 4000, 8: 6000},
    "brute":  {2: 1500, 4: 2500, 6: 4000, 8: 6000},
    "random": {2: 8000, 4: 14000, 6: 20000, 8: 28000},
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


def build_tasks(n_reps: int, seed_base: int) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for rep in range(n_reps):
        seed = seed_base + rep
        for l_val in L_GRID:
            for K in K_GRID:
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
                         "(paper-grade; pilot used --n-reps 1).")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--max-workers", type=int, default=10,
                    help="Parallel workers. Default 10 (leaves ~6 "
                         "logical cores for the OS during the long run).")
    ap.add_argument("--out", type=str,
                    default=os.path.join(OUT_DIR, "simstudy_rows.json"))
    cli = ap.parse_args()

    tasks = build_tasks(cli.n_reps, cli.seed_base)
    print(f"=== F4.2 simulation study: {cli.n_reps} reps → "
          f"{len(tasks)} sessions ===", flush=True)
    print(f"  ℓ-grid={L_GRID}  K={K_GRID}  Σ_l={SIGMA_L_CONDS}", flush=True)

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
            "L_GRID": L_GRID, "K_GRID": K_GRID,
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
