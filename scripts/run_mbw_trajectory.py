"""Capture M. Brandon Westover's per-question HW trajectory for panel (c).

Run AFTER the v2 ExpA experiment completes.  Performs a single hier + single
brute session for MBW at the v2 parameters (N=600, max_q=1500, δ=0.025) and
writes results/phase1_figures/example_trajectory_v2.json with the
hw_max-per-question arrays needed for the example trajectory panel.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

from core_mcmc import run_session_mcmc_auroc, post_hoc_delta_sweep, load_fitted_Sigma
from bridge._common import (
    _autodetect_banks_dir, _default_rater_matrix_path,
    load_bank_signals, load_raters, build_true_params,
)


DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
DELTAS = [0.025, 0.05, 0.10]
OUT_PATH = os.path.join(ENGINE_REPO, "results", "phase1_figures",
                        "example_trajectory_v2.json")


def main():
    Corr_l = np.asarray(
        load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))["Corr_l"],
        dtype=float,
    )
    K = 6
    bank_signals = load_bank_signals(_autodetect_banks_dir(), DOMAINS)
    raters_df = load_raters(_default_rater_matrix_path())
    needed = [f"sigma_{d}" for d in DOMAINS] + [f"theta_{d}" for d in DOMAINS]
    raters_df = raters_df.dropna(subset=needed).reset_index(drop=True)

    mbw = raters_df[raters_df["confirmed_canonical_name"]
                    == "M. Brandon Westover"]
    if mbw.empty:
        print("ERROR: MBW not found in matrix", file=sys.stderr)
        sys.exit(1)
    row = mbw.iloc[0]
    true_params = build_true_params(row, DOMAINS)
    print(f"Loaded MBW with K={K} domains, true_l = "
          f"{[round(float(true_params[2*k+1]), 3) for k in range(K)]}", flush=True)

    out = {"rater_id": "M. Brandon Westover", "trajectories": {}}
    for method in ["hier", "brute"]:
        t0 = time.time()
        kwargs = dict(
            method=method, true_params=true_params, K=K, r_assumed=0.378,
            max_q=1500, delta_auroc=0.025, N=600, seed=0,
            run_until_max=False, log_trajectory=True,
            ess_threshold_frac=0.5,
            bank_signals=bank_signals,
        )
        if method == "hier":
            kwargs.update(Sigma_l=Corr_l, Sigma_t=Corr_l,
                          proposal_scale=2.38 / np.sqrt(2 * K))
        res = run_session_mcmc_auroc(**kwargs)
        dt = time.time() - t0
        sweep = post_hoc_delta_sweep(res["lo_traj"], res["hi_traj"], deltas=DELTAS)
        hw_traj = ((res["hi_traj"] - res["lo_traj"]) / 2.0).max(axis=1)
        out["trajectories"][method] = {
            "hw_max_per_q": hw_traj.tolist(),
            "stop_d0_025": sweep[0.025],
            "stop_d0_05": sweep[0.05],
            "stop_d0_10": sweep[0.10],
        }
        print(f"  MBW {method}: sweep={sweep} ({dt:.1f}s)", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
