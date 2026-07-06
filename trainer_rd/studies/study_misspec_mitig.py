"""M12/V2c — mitigation study for finding F28 (dynamics-prior-dominated
graduation under mis-specification).

Re-runs the zoo (tier2 only) under four trainer configurations:
  shipped    — TaskFilter + D16 hybrid gate exactly as shipped
  evidence   — + raw-outcome evidence gate (trailing-40 observed correct count
               must not fall z=1.28·SD below the borderline learner's
               expectation on the same served items)
  staticmix  — + static-hypothesis mixture in the filter (p_static=0.3)
  both       — both mitigations

Questions: how much false graduation does each mitigation remove (adversarial
+ sub-cut members), and what does it cost on the well-specified/benign members
(declaration delay, non-declaration)?

Run: python3 -m studies.study_misspec_mitig [--smoke]  → figures/data_misspec_mitig.npz
"""
from __future__ import annotations

import sys

import numpy as np

from training.benchmark_trainer import T_STAR
from training.misspec_learners import accuracy_bar
from studies.study_misspec import (BUDGET, L_STAR, SIG_STAR, ZOO, build_pool, run_one)

CONFIGS = {
    "shipped":   dict(),
    "evidence":  dict(evidence_z=1.28),
    "staticmix": dict(p_static=0.3),
    "both":      dict(evidence_z=1.28, p_static=0.3),
}
# follow-up round (--extra): the F30 corrected propagate kernel alone, and the
# full hardened stack (all three mitigations)
EXTRA_CONFIGS = {
    "smearw":    dict(smear_w=True),
    "hardened3": dict(evidence_z=1.28, p_static=0.3, smear_w=True),
}
METRICS = ("n_true", "n_decl", "A_decl", "A_final", "ell_decl", "t_decl",
           "cov_ell", "cov_t", "rmse_ell", "rmse_t")


def main(smoke=False, extra=False):
    pool = build_pool()
    a_bar = accuracy_bar(pool, SIG_STAR, T_STAR)
    configs = EXTRA_CONFIGS if extra else CONFIGS
    members = ("wellspec", "static_below", "careless") if smoke else tuple(ZOO)
    seeds = list(range(3 if smoke else 30))
    budget = 150 if smoke else BUDGET
    res = {m: np.full((len(members), len(configs), len(seeds)), np.nan)
           for m in METRICS}
    print(f"A_bar = {a_bar:.4f}  [{'SMOKE' if smoke else 'FULL'}"
          f"{' EXTRA' if extra else ''}]")
    for j, (cfg, kw) in enumerate(configs.items()):
        for i, mem in enumerate(members):
            for s in seeds:
                r = run_one(mem, "tier2", s, pool, budget=budget, **kw)
                for m in METRICS:
                    res[m][i, j, s] = r[m]
            nd = res["n_decl"][i, j]
            decl = nd <= budget
            fg_perf = decl & (res["A_decl"][i, j] < a_bar)
            fg_lat = decl & ((res["ell_decl"][i, j] < L_STAR)
                             | (np.abs(res["t_decl"][i, j]) > T_STAR))
            print(f"  {cfg:>9} {mem:>14}: declared {decl.sum():2d}/{len(seeds)}"
                  f" (med {np.median(nd[decl]) if decl.any() else float('nan'):5.0f})"
                  f"  FG-perf {fg_perf.sum():2d}  FG-latent {fg_lat.sum():2d}"
                  f"  cov_ell {np.nanmean(res['cov_ell'][i, j]):.2f}")
    if not smoke:
        out = ("figures/data_misspec_mitig2.npz" if extra
               else "figures/data_misspec_mitig.npz")
        np.savez(out, members=np.array(members),
                 configs=np.array(list(configs)), a_bar=a_bar, budget=budget,
                 **{m: res[m] for m in METRICS})
        print(f"saved {out}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, extra="--extra" in sys.argv)
