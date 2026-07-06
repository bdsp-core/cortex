"""M12/V4 — SBC of the TRAINING filter in closed loop (rung 4).

Simulation-based calibration: truth (θ0, ℓ0) drawn from the filter's own
initial belief N(0, 0.3²)², learner follows EXACTLY the assumed dynamics
(soft R–W, assumed rates, shared σ_∞), items chosen by the live tier2 policy.
Sequential/adaptive selection preserves posterior calibration (selection
depends only on past observations), so under a perfectly-specified model the
PIT statistic u = P_w(particle ≤ truth) is U(0,1) at every trial index.
Deviations measure the SMC approximation error of the bootstrap filter itself
(resampling/impoverishment under real-bank s_sd physics) — the training-side
analog of M11's F27.

Reports: PIT KS distance + central-interval coverage (50/80/90/95) for ℓ and
θ at n ∈ {10, 50, 150, 300}, 200 replicates, with binomial ±2SE.

Run: python3 -m studies.study_train_sbc [--smoke]  → figures/data_train_sbc.npz
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import kstest

from training.benchmark_trainer import _select
from training.learner_sim import Learner
from studies.study_misspec import NPART, assumed_params, build_pool
from training.trainer_greedy import RewardWeights
from training.training_filter import TaskFilter

CHECK_N = (10, 50, 150, 300)
LEVELS = (0.50, 0.80, 0.90, 0.95)


def one_rep(rep, pool, *, n_max=300, p_static=0.0, smear_w=False,
            exact_kernel=False):
    rng = np.random.default_rng(9000 + rep)
    fp = assumed_params()
    # truth ~ the filter's initial belief (exact prior draw)
    th0, l0 = 0.3 * rng.standard_normal(), 0.3 * rng.standard_normal()
    lnr = Learner([np.exp(-l0)], [-th0], fp, seed=20000 + rep)
    filt = TaskFilter(0.3 * rng.standard_normal(NPART),
                      0.3 * rng.standard_normal(NPART), fp,
                      seed=30000 + rep, p_static=p_static, smear_w=smear_w,
                      exact_kernel=exact_kernel)
    state = {"k": 0, "bal": 0}
    W = RewardWeights()
    out = {}
    for k in range(n_max):
        state["k"] = k
        idx = _select("tier2", filt, pool, state, rng, W)
        s, s_sd = float(pool.s_mean[idx]), float(pool.s_sd[idx])
        y_star = int(pool.y_star[idx])
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        state["bal"] += 1 if y_star == 1 else -1
        if (k + 1) in CHECK_N:
            ell_true, th_true = -np.log(lnr.sigma[0]), -lnr.t[0]
            out[k + 1] = (float((filt.w * (filt.ell <= ell_true)).sum()),
                          float((filt.w * (filt.theta <= th_true)).sum()))
    return out


def main(smoke=False, exact_kernel=False):
    """exact_kernel (M16): run the SBC with the M15 exact conditional kernel
    (F53/D21) — the closed-loop verification that the open-loop coverage
    result carries over under live tier-2 selection. Saves to
    data_train_sbc_exact.npz; default arm unchanged."""
    pool = build_pool()
    n_rep = 12 if smoke else 200
    n_max = 50 if smoke else 300
    checks = [n for n in CHECK_N if n <= n_max]
    pit = {n: [] for n in checks}
    for rep in range(n_rep):
        for n, uv in one_rep(rep, pool, n_max=n_max,
                             exact_kernel=exact_kernel).items():
            pit[n].append(uv)
    print(f"[{'SMOKE' if smoke else 'FULL'}] {n_rep} reps"
          + (" — EXACT KERNEL (M16)" if exact_kernel else " — shipped kernel"))
    cov = np.full((len(checks), 2, len(LEVELS)), np.nan)
    ks = np.full((len(checks), 2), np.nan)
    for i, n in enumerate(checks):
        u = np.array(pit[n])                      # (rep, 2) — [ℓ, θ]
        for v, name in enumerate(("ell", "theta")):
            ks[i, v] = kstest(u[:, v], "uniform").statistic
            for q, lev in enumerate(LEVELS):
                cov[i, v, q] = float(np.mean(np.abs(u[:, v] - 0.5) < lev / 2))
            se = np.sqrt(np.array(LEVELS) * (1 - np.array(LEVELS)) / n_rep)
            print(f"  n={n:3d} {name:>5}: KS={ks[i,v]:.3f}  cov50/80/90/95 = "
                  + "/".join(f"{c:.2f}" for c in cov[i, v])
                  + f"   (±2SE {'/'.join(f'{2*s:.2f}' for s in se)})")
    if not smoke:
        out = ("figures/data_train_sbc_exact.npz" if exact_kernel
               else "figures/data_train_sbc.npz")
        np.savez(out,
                 checks=np.array(checks), levels=np.array(LEVELS),
                 coverage=cov, ks=ks, n_rep=n_rep,
                 **{f"pit_{n}": np.array(pit[n]) for n in checks})
        print(f"saved {out}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, exact_kernel="--exact" in sys.argv)
