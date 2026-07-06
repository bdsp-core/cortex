"""C9 — shadow-mode replay: what would the trainer have served? (SAP Tier C)

For each real task1 rater stream (≥50 reads): run the trainer's belief filter
(assumed dynamics, D19 hardening) over the ACTUAL reads, and at every step
score the placement the platform actually delivered vs the best item the
trainer's noise-aware skill-mode rule (D15, `expected_skill_weight`) would
have chosen from the bank's real 19,332-segment domain1 pool.

Outputs (per read, aggregated per user):
  w_served   E[learning weight] of the item actually shown
  w_opt      E[learning weight] of the trainer's counterfactual pick
  uplift     w_opt / w_served   (the trainer's headroom on real streams)
plus the served-difficulty placement gap |μ_served − m*| in σ̂ units.

This quantifies, with zero human risk, how far real serving was from
trainer-optimal placement — the core exhibit for the shadow rung of the
staged pilot (§3D) and the basis of the pilot instrumentation list.

Run:  python3 -m studies.study_extset_shadow [--smoke]
Out:  figures/data_extset_shadow.npz
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from training.bank_adapter import BankAdapter
from training.bridge_conventions import SKILL_MODE_MULTIPLIER, engine_to_plan
from training.extset_adapter import load_task1
from training.learner_sim import LearnerParams
from training.trainer_policy import expected_skill_weight
from training.training_filter import TaskFilter

N_WORKERS = 40
MIN_READS = 50
SEED = 20260612

_T1 = _POOL_S = _POOL_SD = None


def assumed_params():
    return LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.5,
                         q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")


def _shadow_user(u):
    idx = np.where(_T1["user"] == u)[0]
    s = _T1["s_loo"][idx]
    sd = _T1["s_sd"][idx]
    y, ystar = _T1["y"][idx], _T1["gold"][idx]
    rng = np.random.default_rng(SEED + int(u) % 99991)
    filt = TaskFilter(0.3 * rng.standard_normal(600),
                      0.3 * rng.standard_normal(600), assumed_params(),
                      ess_frac=0.5, seed=int(u) % 99991,
                      p_static=0.3, smear_w=True)
    w_served, w_opt, gap = [], [], []
    for i in range(len(idx)):
        mt, ml = filt.mean()
        sig_hat, t_hat = engine_to_plan(mt, ml)
        side = 1 if (i % 2 == 0) else -1          # mirror-paired (D15)
        ws = expected_skill_weight(s[i], sd[i], sig_hat, t_hat, side=side)
        wo = expected_skill_weight(_POOL_S, _POOL_SD, sig_hat, t_hat,
                                   side=side).max()
        w_served.append(float(ws))
        w_opt.append(float(wo))
        gap.append(abs((s[i] - t_hat) / sig_hat) - SKILL_MODE_MULTIPLIER)
        filt.step(float(s[i]), int(y[i]), int(ystar[i]),
                  s_sd=float(sd[i]), feedback=True)
    w_served, w_opt = np.array(w_served), np.array(w_opt)
    return (u, len(idx), float(np.median(w_served)), float(np.median(w_opt)),
            float(np.median(w_opt / np.maximum(w_served, 1e-12))),
            float(np.mean(np.abs(gap))))


def main(smoke=False):
    global _T1, _POOL_S, _POOL_SD
    _T1 = load_task1()
    ad = BankAdapter()
    pool = ad.task_pool(0)                        # bank domain1 pool
    _POOL_S, _POOL_SD = pool.s_mean, pool.s_sd
    counts = pd.Series(_T1["user"]).value_counts()
    users = counts[counts >= MIN_READS].index.to_numpy()
    if smoke:
        users = users[:24]
    print(f"shadowing {len(users)} raters (≥{MIN_READS} reads) against the "
          f"{len(_POOL_S):,}-item bank pool", flush=True)
    with mp.Pool(min(N_WORKERS, len(users))) as p:
        rows = p.map(_shadow_user, users)
    df = pd.DataFrame(rows, columns=["user", "n", "w_served", "w_opt",
                                     "uplift", "gap"])
    print(df[["w_served", "w_opt", "uplift", "gap"]].describe()
          .round(3).to_string(), flush=True)
    print(f"\nmedian per-user expected-learning-weight uplift: "
          f"×{df['uplift'].median():.2f}  (IQR {df['uplift'].quantile(.25):.2f}"
          f"–{df['uplift'].quantile(.75):.2f})", flush=True)
    if not smoke:
        np.savez("figures/data_extset_shadow.npz", table=df.to_numpy(),
                 cols=np.array(df.columns, dtype=object))
        print("saved figures/data_extset_shadow.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
