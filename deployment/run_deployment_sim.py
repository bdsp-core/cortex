"""Run the multi-task adaptive test simulator on a tiered synthetic panel.

Generates ~200 candidates across 4 tiers (expert, experienced, novice, crowd),
simulates each through the full test, and persists results to
data/deployment_prior/sim/.

Outputs:
  candidates.csv     per-candidate true theta + decisions + trials per task
  trajectories.npz   posterior trajectories (mu, sd, p_pass per trial per cand)
  summary.json       config and aggregate stats

─────────────────────────────────────────────────────────────────────────
UNIFIED-MERGE PROVENANCE (Phase 4.1, 2026-05-18). FAITHFUL, PATH-ONLY
port of the PI repo's `scripts/run_deployment_sim.py`. Substantive
changes vs the PI source, all non-numeric:
  • the hardcoded `ROOT`/`sys.path.insert`/`import simulate_test as st`/
    `OUT` block → an `engine_paths` shim + package import of the ported
    `deployment.simulate_test` (zero absolute paths; plan Phase 4 step 2).
  • `main(seed=0)` → `main(seed=0, out_dir=None)` — a single ADDITIVE
    testability hook (default = the PI `data/deployment_prior/sim` dir,
    identical semantics) so the Phase-4.1 port-fidelity gate can write a
    repro dir WITHOUT clobbering the carried regression baseline.
The candidate-generation logic and all numerics are byte-faithful.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── path shim (mirrors deployment/simulate_test.py): self-locate repo
#    root, engine/ + root on sys.path, import the ported simulator. ──
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_DEPLOY_DIR)
for _p in (os.path.join(_REPO, "engine"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import engine_paths  # noqa: E402
from deployment import simulate_test as st   # noqa: E402

_DEFAULT_OUT = Path(engine_paths.DEPLOYMENT_PRIOR) / "sim"

# Phase 4.4-A: K/DIM/task-list are NOT module constants — they are
# derived inside main() from the frozen artifact (st.deployment_task_
# names()), so this driver scales with whatever K the deployment_prior
# declares (6 now, 7 after the Phase-4.6 re-freeze).


# ────────────────────── tiered synthetic candidate population ──────────────────

# Per-tier mean and within-tier dispersion. Skills are correlated across tasks
# (we sample one "skill anomaly" per candidate and add it to all tasks).
TIERS = {
    "expert":      dict(n=60,  mu_ell=1.6, sd_within=0.30, mu_t=0.0, sd_t=0.20),
    "experienced": dict(n=40,  mu_ell=1.1, sd_within=0.30, mu_t=0.0, sd_t=0.30),
    "novice":      dict(n=40,  mu_ell=0.3, sd_within=0.35, mu_t=0.0, sd_t=0.40),
    "crowd":       dict(n=60,  mu_ell=-0.3, sd_within=0.45, mu_t=0.0, sd_t=0.60),
}


def sample_candidate_theta(tier_cfg, r_ell, rng, k_tasks):
    """Sample one candidate θ from the tier-conditioned distribution.

    Skill ℓ_k = tier_mean + shared anomaly + task-specific noise, with
    shared/noise variances chosen so the cross-task correlation is r_ell.
    Bias t_k drawn independently across tasks. K-agnostic (4.4-A):
    `k_tasks` from the frozen artifact; the rng draw COUNT scales with
    it (k_tasks=6 reproduces the prior stream exactly)."""
    mu_l = tier_cfg["mu_ell"]; sd_l = tier_cfg["sd_within"]
    mu_t = tier_cfg["mu_t"]; sd_t = tier_cfg["sd_t"]
    # Skill structure: ℓ_k = μ + sd * (sqrt(r) Z_shared + sqrt(1-r) Z_k)
    Z_shared = rng.standard_normal()
    Z_k = rng.standard_normal(k_tasks)
    ells = mu_l + sd_l * (np.sqrt(r_ell) * Z_shared + np.sqrt(1 - r_ell) * Z_k)
    ts = mu_t + sd_t * rng.standard_normal(k_tasks)
    theta = np.zeros(2 * k_tasks)
    for k in range(k_tasks):
        theta[2 * k]     = float(np.clip(ts[k],  -3.0, 3.0))
        theta[2 * k + 1] = float(np.clip(ells[k], -3.0, 3.0))
    return theta


def main(seed=0, out_dir=None):
    OUT = Path(out_dir) if out_dir is not None else _DEFAULT_OUT
    OUT.mkdir(parents=True, exist_ok=True)
    # Post-Kong: per-task IIIC AUROC is 0.93-0.96 so per-task ℓ*_k are
    # defensible. Use them rather than the old uniform 0.62 fallback.
    Sigma, bank_by_task, ell_star = st.load_deployment(uniform_ell_star=None)
    tasks = st.deployment_task_names()       # AUTHORITATIVE (4.4-A)
    k_tasks = len(tasks)
    cfg = st.TestConfig.from_yaml()   # Phase 4.2: shipping-contract YAML
    rng = np.random.default_rng(seed)
    r_ell = float(np.mean([Sigma[1, 3], Sigma[1, 5], Sigma[3, 5]]))
    print(f"r_ell from prior = {r_ell:.3f}  (K={k_tasks})", flush=True)
    print(f"per-task ℓ* = {dict(zip(tasks, np.round(ell_star, 3)))}", flush=True)

    t_start = time.perf_counter()
    rows = []
    traj_mu = []      # list of (T, DIM) arrays
    traj_sd = []
    traj_p  = []
    cand_id = 0
    for tier, tcfg in TIERS.items():
        print(f"\n=== {tier} (n={tcfg['n']}) ===", flush=True)
        for j in range(tcfg["n"]):
            theta = sample_candidate_theta(tcfg, r_ell, rng, k_tasks)
            state = st.simulate_candidate(theta, Sigma, ell_star,
                                            bank_by_task, cfg, rng)
            row = {"cand_id": cand_id, "tier": tier}
            for ki, t in enumerate(tasks):
                row[f"true_t_{t}"]  = theta[2 * ki]
                row[f"true_ell_{t}"] = theta[2 * ki + 1]
                row[f"hat_t_{t}"]   = state.mu[2 * ki]
                row[f"hat_ell_{t}"] = state.mu[2 * ki + 1]
                row[f"sd_t_{t}"]    = state.sd_traj[-1][2 * ki]
                row[f"sd_ell_{t}"]  = state.sd_traj[-1][2 * ki + 1]
                row[f"n_trials_{t}"] = int(state.n_per_task[ki])
                row[f"decision_{t}"] = state.decision[ki]
            row["n_trials_total"] = int(state.n_per_task.sum())
            rows.append(row)
            traj_mu.append(np.array(state.mu_traj))
            traj_sd.append(np.array(state.sd_traj))
            traj_p .append(np.array(state.p_pass_traj))
            cand_id += 1
            if (j + 1) % 20 == 0:
                wall = time.perf_counter() - t_start
                print(f"  ...{j+1}/{tcfg['n']} ({wall:.0f}s)", flush=True)

    cands = pd.DataFrame(rows)
    cands.to_csv(OUT / "candidates.csv", index=False)
    print(f"\nwrote {OUT/'candidates.csv'}  ({len(cands)} candidates)", flush=True)

    # Trajectories: dump as object arrays (variable lengths)
    np.savez(OUT / "trajectories.npz",
             mu_traj=np.array(traj_mu, dtype=object),
             sd_traj=np.array(traj_sd, dtype=object),
             p_pass_traj=np.array(traj_p, dtype=object),
             allow_pickle=True)
    print(f"wrote {OUT/'trajectories.npz'}", flush=True)

    summary = {
        "seed": seed,
        "r_ell": r_ell,
        "ell_star": ell_star.tolist(),
        "cfg": dict(N_min=cfg.N_min, N_max=cfg.N_max,
                     N_min_per_task=cfg.N_min_per_task,
                     N_max_per_task=cfg.N_max_per_task,
                     pass_p=cfg.pass_p, fail_p=cfg.fail_p),
        "tiers": TIERS,
        "n_candidates": len(cands),
        "wall_time_s": time.perf_counter() - t_start,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT/'summary.json'}", flush=True)

    # Print quick verdict summary by tier
    print("\nVerdict summary by tier and task (pass/fail/refer):")
    for tier in TIERS:
        sub = cands[cands.tier == tier]
        line = f"  {tier:>12}: "
        for t in tasks:
            decs = sub[f"decision_{t}"].value_counts().to_dict()
            line += f"{t}={decs.get('pass',0):2d}p/{decs.get('fail',0):2d}f/{decs.get('refer',0):2d}r  "
        print(line)


if __name__ == "__main__":
    main()
