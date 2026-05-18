"""Phase-1 demonstration experiments for Multi-AUROC Precision Protocol.

Runs two focused experiments and saves JSON-formatted results for the
publication figure script.

Experiment A — K=6 SPARCNET real raters (borrowing-of-strength demonstration)
  - Uses Corr_l from Sigma_l_fitted.npy.
  - 4 real raters × 2 methods × 2 seeds = 16 sessions.
  - δ_run = 0.025 with run_until_max=False (session stops early at the
    tightest δ reached; post-hoc sweep reports stops at all loose δs from
    the same trajectory).

Experiment B — K-scaling synthetic (borrowing-of-strength scales with K)
  - K ∈ {2, 4, 6, 8}, 3 synthetic raters per K, 2 methods, 1 seed = 24 sessions.
  - δ_run = 0.05; max_q tuned per K.

Output: results/phase1_figures/expA_real_k6.json, expB_kscaling.json,
        results/phase1_figures/example_trajectory.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

from core_mcmc import (
    run_session_mcmc_auroc, post_hoc_delta_sweep, load_fitted_Sigma,
)
from auroc import auroc_from_l
from bridge._common import (
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


def experiment_A_real_k6():
    """Real K=6 SPARCNET raters with fitted Corr_l prior."""
    print("\n=== Experiment A: K=6 SPARCNET real raters ===")
    Corr_l = np.asarray(
        load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))["Corr_l"],
        dtype=float,
    )
    K = 6
    bank_signals = load_bank_signals(_autodetect_banks_dir(), DOMAINS)
    raters_df = load_raters(_default_rater_matrix_path())

    # Take first 4 raters with all 6 SDT fits (drop rows with NaN)
    needed_cols = [f"sigma_{d}" for d in DOMAINS] + [f"theta_{d}" for d in DOMAINS]
    raters_df = raters_df.dropna(subset=needed_cols).head(4).reset_index(drop=True)
    print(f"  selected {len(raters_df)} raters with all 6 SDT fits", flush=True)

    max_q = 800
    N_particles = 500
    seeds = [0, 1]
    rows: List[Dict[str, Any]] = []
    traj_for_panel_c = None  # capture one trajectory per method for panel (c)

    for ri, row in raters_df.iterrows():
        rater_id = str(row.get("canonical_name", row.get("rater_id", f"R{ri}")))
        true_params = build_true_params(row, DOMAINS)
        true_l = np.array([true_params[2 * k + 1] for k in range(K)])
        true_auroc = auroc_from_l(true_l)

        for method in ["hier", "brute"]:
            for seed in seeds:
                t0 = time.time()
                kwargs = dict(
                    method=method, true_params=true_params, K=K, r_assumed=0.378,
                    max_q=max_q, delta_auroc=0.025, N=N_particles, seed=seed,
                    run_until_max=False,   # stop early at δ=0.025; post-hoc finds loose δ from trajectory
                    log_trajectory=True,
                    ess_threshold_frac=0.5,
                    bank_signals=bank_signals,
                )
                if method == "hier":
                    kwargs.update(Sigma_l=Corr_l, Sigma_t=Corr_l,
                                  proposal_scale=2.38 / np.sqrt(2 * K))
                out = run_session_mcmc_auroc(**kwargs)
                dt = time.time() - t0

                sweep = post_hoc_delta_sweep(out["lo_traj"], out["hi_traj"],
                                              deltas=DELTAS)
                hw_final_max = float(((out["final_hi"] - out["final_lo"]) / 2.0).max())
                rows.append({
                    "rater_id": rater_id,
                    "method": method,
                    "seed": seed,
                    "K": K,
                    "n_q_total": int(out["n_questions"]),
                    "hw_max_final": hw_final_max,
                    "stop_d0_025": sweep[0.025],
                    "stop_d0_05": sweep[0.05],
                    "stop_d0_10": sweep[0.10],
                    "true_auroc": true_auroc.tolist(),
                    "duration_s": dt,
                })
                # Capture the first rater/seed=0 pair's trajectory for panel (c)
                if ri == 0 and seed == 0:
                    hw_traj = ((out["hi_traj"] - out["lo_traj"]) / 2.0).max(axis=1)
                    if traj_for_panel_c is None:
                        traj_for_panel_c = {"rater_id": rater_id, "trajectories": {}}
                    traj_for_panel_c["trajectories"][method] = {
                        "hw_max_per_q": hw_traj.tolist(),
                        "stop_d0_025": sweep[0.025],
                        "stop_d0_05": sweep[0.05],
                        "stop_d0_10": sweep[0.10],
                    }

                print(f"    A | {rater_id} {method} seed={seed}: "
                      f"sweep={sweep} hw_max={hw_final_max:.4f} ({dt:.1f}s)",
                      flush=True)

    # Save
    with open(os.path.join(OUT_DIR, "expA_real_k6.json"), "w") as f:
        json.dump(_to_jsonable({"rows": rows, "config": {
            "K": K, "max_q": max_q, "N_particles": N_particles,
            "seeds": seeds, "deltas": DELTAS, "domains": DOMAINS,
        }}), f, indent=2)
    with open(os.path.join(OUT_DIR, "example_trajectory.json"), "w") as f:
        json.dump(_to_jsonable(traj_for_panel_c), f, indent=2)
    print(f"  → wrote {len(rows)} rows to expA_real_k6.json")


def experiment_B_k_scaling():
    """Synthetic K-scaling: K in {2, 4, 6, 8}, synthetic clearly-pass raters."""
    print("\n=== Experiment B: K-scaling synthetic ===")
    Corr_l_6 = np.asarray(
        load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))["Corr_l"],
        dtype=float,
    )
    rows: List[Dict[str, Any]] = []
    Ks = [2, 4, 6, 8]
    n_raters_per_K = 3
    seeds = [0]
    # max_q scales with K (more domains need more questions to reach δ)
    max_q_per_K = {2: 250, 4: 400, 6: 550, 8: 750}
    N_particles = 500
    auroc_delta = 0.05

    for K in Ks:
        max_q = max_q_per_K[K]
        # Build a K-dimensional Corr_l: use the empirical Corr_l_6 for K=6;
        # for K≠6, use a compound-symmetry approximation with the same mean
        # off-diagonal correlation as Corr_l_6 (~0.30 empirically).
        if K == 6:
            Corr_l_K = Corr_l_6
        else:
            mean_off = float((Corr_l_6.sum() - 6) / (6 * 6 - 6))
            Corr_l_K = mean_off * np.ones((K, K)) + (1.0 - mean_off) * np.eye(K)
        for ri in range(n_raters_per_K):
            rng = np.random.default_rng(100 + 10 * K + ri)
            l_true = rng.normal(0.4, 0.3, size=K)
            t_true = rng.normal(0.0, 0.5, size=K)
            true_params = []
            for k in range(K):
                true_params.append(float(t_true[k]))
                true_params.append(float(l_true[k]))
            for method in ["hier", "brute"]:
                for seed in seeds:
                    t0 = time.time()
                    kwargs = dict(
                        method=method, true_params=true_params, K=K,
                        r_assumed=0.378,
                        max_q=max_q, delta_auroc=auroc_delta,
                        N=N_particles, seed=seed,
                        run_until_max=False, log_trajectory=True,
                        ess_threshold_frac=0.5,
                    )
                    if method == "hier":
                        kwargs.update(Sigma_l=Corr_l_K, Sigma_t=Corr_l_K,
                                      proposal_scale=2.38 / np.sqrt(2 * K))
                    out = run_session_mcmc_auroc(**kwargs)
                    dt = time.time() - t0
                    sweep = post_hoc_delta_sweep(out["lo_traj"], out["hi_traj"],
                                                  deltas=DELTAS)
                    rows.append({
                        "K": K, "rater_idx": ri, "method": method, "seed": seed,
                        "stop_d0_025": sweep[0.025],
                        "stop_d0_05": sweep[0.05],
                        "stop_d0_10": sweep[0.10],
                        "n_q_total": int(out["n_questions"]),
                        "duration_s": dt,
                    })
                    print(f"    B | K={K} R{ri} {method} seed={seed}: "
                          f"sweep={sweep} ({dt:.1f}s)", flush=True)

    with open(os.path.join(OUT_DIR, "expB_kscaling.json"), "w") as f:
        json.dump(_to_jsonable({"rows": rows, "config": {
            "Ks": Ks, "n_raters_per_K": n_raters_per_K, "seeds": seeds,
            "max_q": max_q, "N_particles": N_particles,
            "auroc_delta": auroc_delta, "deltas": DELTAS,
        }}), f, indent=2)
    print(f"  → wrote {len(rows)} rows to expB_kscaling.json")


if __name__ == "__main__":
    t_total = time.time()
    experiment_A_real_k6()
    experiment_B_k_scaling()
    print(f"\nTotal: {time.time() - t_total:.1f}s")
