"""C10 — empirical power analysis for the staged pilot (SAP Tier C).

Effect-size inputs are fully empirical: learners drawn from the EXTSET-fitted
population (`realpop`, F43), trained under the D19-hardened config by each
arm: tier2 (the trainer), staircase85 (the strongest conventional adaptive
control), random serving. CRN across arms.

Primary pilot endpoint (proposed): log(trials to TRUE mastery), two-sample
comparison tier2 vs staircase85. Required N per arm:
    N = 2·(z_{1−α/2} + z_{power})² · σ²_pooled / δ²
on the log scale, inflated for dropout. Censored runs (no mastery within
budget) are handled by reporting both the log-trial effect on masterers and
the mastery-rate effect (two-proportion power) — the SAP carries both.

Run:  python3 -m studies.study_pilot_power [--smoke]
Out:  figures/data_pilot_power.npz
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np
from scipy.stats import norm

from studies.study_extset_stress import fitted_link_params, register
from studies.study_misspec import BUDGET, ZOO, build_pool, run_one

N_WORKERS = 40
N_SEEDS = 60
ARMS = ("tier2", "staircase85", "random")
HARDENED = dict(p_static=0.3, evidence_z=1.28, smear_w=True)

_POOL = None
_BUDGET = None


def _job(args):
    arm, seed = args
    r = run_one("realpop", arm, seed, _POOL, budget=_BUDGET, **HARDENED)
    return arm, seed, r["n_true"], r["n_decl"]


def required_n(delta, sd, alpha, power):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(2 * (z * sd / delta) ** 2))


def main(smoke=False):
    global _POOL, _BUDGET
    best, lf, lm, nu, pop = fitted_link_params()
    register(best, lf, lm, nu, pop)
    _POOL = build_pool()
    _BUDGET = 150 if smoke else BUDGET
    seeds = range(8 if smoke else N_SEEDS)
    jobs = [(a, s) for a in ARMS for s in seeds]
    with mp.Pool(min(N_WORKERS, len(jobs))) as p:
        out = p.map(_job, jobs)

    res = {a: {"n_true": [], "n_decl": []} for a in ARMS}
    for arm, seed, nt, nd in out:
        res[arm]["n_true"].append(nt)
        res[arm]["n_decl"].append(nd)
    stats = {}
    for a in ARMS:
        nt = np.array(res[a]["n_true"], float)
        mast = nt <= _BUDGET
        ln = np.log(nt[mast])
        stats[a] = dict(rate=mast.mean(), mean_ln=ln.mean(), sd_ln=ln.std(),
                        med=np.median(nt[mast]) if mast.any() else np.nan)
        print(f"  {a:>11}: mastered {mast.sum()}/{len(nt)} "
              f"(med {stats[a]['med']:.0f} trials, "
              f"log-mean {ln.mean():.2f} ± {ln.std():.2f})", flush=True)

    grid = []
    for ctrl in ("staircase85", "random"):
        delta = stats[ctrl]["mean_ln"] - stats["tier2"]["mean_ln"]
        sd = np.sqrt(0.5 * (stats["tier2"]["sd_ln"] ** 2
                            + stats[ctrl]["sd_ln"] ** 2))
        p1, p2 = stats["tier2"]["rate"], stats[ctrl]["rate"]
        print(f"\n  tier2 vs {ctrl}: δ = {delta:+.3f} log-trials "
              f"(×{np.exp(delta):.2f}), σ = {sd:.3f}; mastery rate "
              f"{p1:.2f} vs {p2:.2f}", flush=True)
        for alpha in (0.05, 0.005):
            for power in (0.8, 0.9):
                if np.isfinite(delta) and np.isfinite(sd) and delta != 0:
                    n_t = required_n(abs(delta), sd, alpha, power)
                else:
                    n_t = -1               # log-trial endpoint undefined
                if abs(p1 - p2) > 1e-9:    # two-proportion (mastery rate)
                    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
                    pb = (p1 + p2) / 2
                    n_p = int(np.ceil(((za * np.sqrt(2 * pb * (1 - pb))
                                        + zb * np.sqrt(p1 * (1 - p1)
                                                       + p2 * (1 - p2)))
                                       / (p1 - p2)) ** 2))
                else:
                    n_p = -1
                n_use = max(n for n in (n_t, n_p) if n > 0) \
                    if max(n_t, n_p) > 0 else -1
                n_enroll = int(np.ceil(n_use / 0.85)) if n_use > 0 else -1
                grid.append((ctrl, alpha, power, n_t, n_p, n_enroll))
                print(f"    α={alpha:<6} power={power}: N/arm "
                      f"log-trials={n_t if n_t > 0 else '—'} "
                      f"mastery-rate={n_p if n_p > 0 else '—'} "
                      f"→ enroll {n_enroll if n_enroll > 0 else '—'}",
                      flush=True)

    if not smoke:
        np.savez("figures/data_pilot_power.npz",
                 grid=np.array(grid, dtype=object),
                 **{f"{a}_{k}": np.array(v) for a in ARMS
                    for k, v in res[a].items()})
        print("saved figures/data_pilot_power.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
