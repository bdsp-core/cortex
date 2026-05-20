"""Phase-1 v2 demonstration experiments (paper-grade-ish).

Larger sample than v1 (which had 4 raters × 2 seeds = 16 ExpA sessions).
v2 uses all 27 SPARCNET raters that have complete K=6 SDT fits + a beefier
K-scaling sweep.

Experiment A v2 — K=6 SPARCNET real raters
  - All 27 raters with complete (σ_d, θ_d) for all 6 SPARCNET domains.
  - 2 methods × 1 seed = 54 sessions.
  - N=600 particles, max_q=1500.
  - run_until_max=False; post-hoc reports stops at {0.025, 0.05, 0.10}.

Experiment B v2 — K-scaling synthetic
  - K ∈ {2, 4, 6, 8}; 5 synthetic raters per K; 2 methods × 1 seed.
  - max_q scaled with K; same particle count as ExpA.

Output: results/phase1_figures/expA_real_k6_v2.json,
        results/phase1_figures/expB_kscaling_v2.json,
        results/phase1_figures/example_trajectory_v2.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

# CRITICAL: single-thread BLAS before numpy import.
_THIS_DIR_BOOT = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR_BOOT not in sys.path:
    sys.path.insert(0, _THIS_DIR_BOOT)
from _parallel import configure_blas_single_thread, parallel_map  # noqa: E402
configure_blas_single_thread()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
# Unified-repo layout (Phase 2): the validated engine lives in engine/
# with flat bare-name imports. Add both the repo root (for Sigma_l_fitted.npy
# / cert_config.yaml resolution) and engine/ (for core_mcmc/auroc imports).
# Matches scripts/run_tier2_oc_simstudy.py:70.
for _p in (ENGINE_REPO, os.path.join(ENGINE_REPO, "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_mcmc import (  # noqa: E402
    run_session_mcmc_auroc, post_hoc_delta_sweep, load_fitted_Sigma,
)
from auroc import auroc_from_l  # noqa: E402
from bridge._common import (  # noqa: E402
    _autodetect_banks_dir, _default_rater_matrix_path,
    load_bank_signals, load_raters, build_true_params,
)


OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase1_figures")
os.makedirs(OUT_DIR, exist_ok=True)

DELTAS = [0.025, 0.05, 0.10]
DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]


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


def _get_shared() -> Dict[str, Any]:
    if not _SHARED:
        _SHARED["Corr_l"] = np.asarray(
            load_fitted_Sigma(
                os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))["Corr_l"],
            dtype=float,
        )
        _SHARED["bank_signals"] = load_bank_signals(
            _autodetect_banks_dir(), DOMAINS)
    return _SHARED


def _expA_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (rater, method) Mode-A session → summary row (+ traj if MBW)."""
    shared = _get_shared()
    Corr_l = shared["Corr_l"]
    bank_signals = shared["bank_signals"]
    K = task["K"]
    method = task["method"]
    true_params = task["true_params"]
    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    t0 = time.time()
    kwargs = dict(
        method=method, true_params=true_params, K=K, r_assumed=0.378,
        max_q=task["max_q"], delta_auroc=0.025, N=task["N_particles"],
        seed=task["seed"], run_until_max=False, log_trajectory=True,
        ess_threshold_frac=0.5, bank_signals=bank_signals,
    )
    if method == "hier":
        kwargs.update(Sigma_l=Corr_l, Sigma_t=Corr_l,
                      proposal_scale=2.38 / np.sqrt(2 * K))
    out = run_session_mcmc_auroc(**kwargs)
    dt = time.time() - t0
    sweep = post_hoc_delta_sweep(out["lo_traj"], out["hi_traj"], deltas=DELTAS)
    hw_final_max = float(((out["final_hi"] - out["final_lo"]) / 2.0).max())
    row = {
        "rater_id": task["rater_id"],
        "method": method,
        "seed": task["seed"],
        "K": K,
        "n_q_total": int(out["n_questions"]),
        "hw_max_final": hw_final_max,
        "stop_d0_025": sweep[0.025],
        "stop_d0_05": sweep[0.05],
        "stop_d0_10": sweep[0.10],
        "true_auroc": true_auroc.tolist(),
        "duration_s": dt,
    }
    traj = None
    if task["rater_id"] == "M. Brandon Westover" and task["seed"] == 0:
        hw_traj = ((out["hi_traj"] - out["lo_traj"]) / 2.0).max(axis=1)
        traj = {
            "method": method,
            "hw_max_per_q": hw_traj.tolist(),
            "stop_d0_025": sweep[0.025],
            "stop_d0_05": sweep[0.05],
            "stop_d0_10": sweep[0.10],
        }
    return {"row": row, "traj": traj}


def experiment_A_real_k6_v2(max_workers=None):
    """All 27 SPARCNET raters with complete K=6 SDT fits (parallel)."""
    print("\n=== Experiment A v2: K=6 SPARCNET real raters (all 27) ===",
          flush=True)
    K = 6
    raters_df = load_raters(_default_rater_matrix_path())
    needed_cols = [f"sigma_{d}" for d in DOMAINS] + [f"theta_{d}" for d in DOMAINS]
    raters_df = raters_df.dropna(subset=needed_cols).reset_index(drop=True)
    print(f"  loaded {len(raters_df)} raters with all 6 SDT fits", flush=True)

    # Rigorous config (2026-05-15, user-approved).  Per-method max_q caps:
    # random has no adaptivity/pooling so it needs a far larger budget to
    # reach the tight δ=0.025 (probed: up to ~8k q), but it is ~25× cheaper
    # per question (no EV) so a huge cap is affordable.  hier/brute that
    # haven't reached δ=0.025 by ~4k q have a plateaued AUROC CI and are
    # reported as "did not reach within N_max" (a legitimate operating
    # characteristic) rather than burning ~60 min/session running to 15k.
    MAX_Q_BY_METHOD = {"random": 15000, "brute": 4000, "hier": 4000}
    N_particles = 1000          # production (cert_config v11; Phase-2-validated)
    seeds = [0, 1, 2, 3, 4]     # 5 seeds → bootstrap IQR per cell

    tasks: List[Dict[str, Any]] = []
    for ri, row in raters_df.iterrows():
        # FIX: the matrix column is `confirmed_canonical_name`.
        rater_full = str(
            row.get("confirmed_canonical_name",
                    row.get("canonical_name", f"R{ri}"))
        )
        true_params = build_true_params(row, DOMAINS)
        for method in ["hier", "brute", "random"]:
            for seed in seeds:
                tasks.append({
                    "rater_id": rater_full,
                    "K": K,
                    "method": method,
                    "seed": seed,
                    "true_params": true_params,
                    "max_q": MAX_Q_BY_METHOD[method],
                    "N_particles": N_particles,
                })

    results = parallel_map(
        _expA_worker, tasks,
        max_workers=max_workers,
        desc="ExpA sessions",
        ordered=True,
        progress_every=10,
    )

    rows: List[Dict[str, Any]] = []
    traj_for_panel_c = None
    for r in results:
        if isinstance(r, dict) and "__error__" in r:
            print(f"  [ERROR] ExpA task {r.get('__task_index__')}: "
                  f"{r['__error__']}", file=sys.stderr, flush=True)
            continue
        rows.append(r["row"])
        if r["traj"] is not None:
            if traj_for_panel_c is None:
                traj_for_panel_c = {"rater_id": "M. Brandon Westover",
                                     "trajectories": {}}
            traj_for_panel_c["trajectories"][r["traj"]["method"]] = {
                "hw_max_per_q": r["traj"]["hw_max_per_q"],
                "stop_d0_025": r["traj"]["stop_d0_025"],
                "stop_d0_05": r["traj"]["stop_d0_05"],
                "stop_d0_10": r["traj"]["stop_d0_10"],
            }

    with open(os.path.join(OUT_DIR, "expA_real_k6_v2.json"), "w") as f:
        json.dump(_to_jsonable({"rows": rows, "config": {
            "K": K, "max_q_by_method": MAX_Q_BY_METHOD,
            "N_particles": N_particles,
            "seeds": seeds, "deltas": DELTAS, "domains": DOMAINS,
        }}), f, indent=2)
    if traj_for_panel_c is None:
        print("  [WARN] MBW not found; run scripts/run_mbw_trajectory.py "
              "separately for panel (c).", flush=True)
    else:
        with open(os.path.join(OUT_DIR, "example_trajectory_v2.json"), "w") as f:
            json.dump(_to_jsonable(traj_for_panel_c), f, indent=2)
    print(f"  → wrote {len(rows)} rows to expA_real_k6_v2.json", flush=True)


_KS_B = [2, 4, 6, 7, 8]
_N_RATERS_PER_K_B = 5
# Rigorous per-method × per-K caps.  Panel (b) reports δ=0.05; random
# needs a far larger budget to reach it (no adaptivity/pooling) but is
# ~25× cheaper per question, so the big random cap is affordable.
# K=7 added in Phase 7 sub-7.4 (production deployment dim).  K=7 caps are
# interpolated linearly between K=6 and K=8 (same convention as
# scripts/run_tier2_oc_simstudy.py:MAX_Q_BY_METHOD_K).
_MAX_Q_BY_METHOD_K = {
    "hier":   {2: 400, 4: 700, 6: 1000, 7: 1200, 8: 1400},
    "brute":  {2: 400, 4: 700, 6: 1000, 7: 1200, 8: 1400},
    "random": {2: 2000, 4: 3500, 6: 5000, 7: 6000, 8: 7000},
}
_N_PARTICLES_B = 1000           # production (match ExpA / cert_config v11)
_SEEDS_B = [0, 1, 2, 3, 4]      # 5 seeds (same rigor as ExpA)
_AUROC_DELTA_B = 0.05


def _corr_l_for_K(K: int) -> np.ndarray:
    """Empirical Corr_l for K=6; compound-symmetry approx (matched mean
    off-diagonal) for K≠6."""
    Corr_l_6 = _get_shared()["Corr_l"]
    if K == 6:
        return Corr_l_6
    mean_off = float((Corr_l_6.sum() - 6) / (6 * 6 - 6))
    return mean_off * np.ones((K, K)) + (1.0 - mean_off) * np.eye(K)


def _expB_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (K, rater_idx, method) K-scaling Mode-A session → summary row."""
    K = task["K"]
    method = task["method"]
    true_params = task["true_params"]
    Corr_l_K = _corr_l_for_K(K)
    t0 = time.time()
    kwargs = dict(
        method=method, true_params=true_params, K=K, r_assumed=0.378,
        max_q=task["max_q"], delta_auroc=_AUROC_DELTA_B,
        N=task["N_particles"], seed=task["seed"],
        run_until_max=False, log_trajectory=True,
        ess_threshold_frac=0.5,
    )
    if method == "hier":
        kwargs.update(Sigma_l=Corr_l_K, Sigma_t=Corr_l_K,
                      proposal_scale=2.38 / np.sqrt(2 * K))
    out = run_session_mcmc_auroc(**kwargs)
    dt = time.time() - t0
    sweep = post_hoc_delta_sweep(out["lo_traj"], out["hi_traj"], deltas=DELTAS)
    return {
        "K": K, "rater_idx": task["rater_idx"], "method": method,
        "seed": task["seed"],
        "stop_d0_025": sweep[0.025],
        "stop_d0_05": sweep[0.05],
        "stop_d0_10": sweep[0.10],
        "n_q_total": int(out["n_questions"]),
        "duration_s": dt,
    }


def experiment_B_k_scaling_v2(max_workers=None):
    """K-scaling synthetic with broader sampling (parallel)."""
    print("\n=== Experiment B v2: K-scaling synthetic (5 raters per K) ===",
          flush=True)
    seeds = _SEEDS_B
    tasks: List[Dict[str, Any]] = []
    for K in _KS_B:
        for ri in range(_N_RATERS_PER_K_B):
            rng = np.random.default_rng(100 + 10 * K + ri)
            l_true = rng.normal(0.4, 0.3, size=K)
            t_true = rng.normal(0.0, 0.5, size=K)
            true_params = []
            for k in range(K):
                true_params.append(float(t_true[k]))
                true_params.append(float(l_true[k]))
            for method in ["hier", "brute", "random"]:
                for seed in seeds:
                    tasks.append({
                        "K": K, "rater_idx": ri, "method": method,
                        "seed": seed, "true_params": true_params,
                        "max_q": _MAX_Q_BY_METHOD_K[method][K],
                        "N_particles": _N_PARTICLES_B,
                    })

    results = parallel_map(
        _expB_worker, tasks,
        max_workers=max_workers,
        desc="ExpB sessions",
        ordered=True,
        progress_every=10,
    )
    rows = [r for r in results
            if not (isinstance(r, dict) and "__error__" in r)]

    with open(os.path.join(OUT_DIR, "expB_kscaling_v2.json"), "w") as f:
        json.dump(_to_jsonable({"rows": rows, "config": {
            "Ks": _KS_B, "n_raters_per_K": _N_RATERS_PER_K_B, "seeds": seeds,
            "max_q_by_method_k": _MAX_Q_BY_METHOD_K,
            "N_particles": _N_PARTICLES_B,
            "auroc_delta": _AUROC_DELTA_B, "deltas": DELTAS,
        }}), f, indent=2)
    print(f"  → wrote {len(rows)} rows to expB_kscaling_v2.json", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=None,
                    help="Parallel worker processes (default: 14 / cpu-2).")
    cli = ap.parse_args()
    t_total = time.time()
    experiment_A_real_k6_v2(max_workers=cli.max_workers)
    experiment_B_k_scaling_v2(max_workers=cli.max_workers)
    print(f"\nTotal: {time.time() - t_total:.1f}s", flush=True)
