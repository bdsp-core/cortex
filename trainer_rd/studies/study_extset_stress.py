"""P3 — rung-3 stress with the EXTSET-fitted real link (SAP §2.5).

Registers two zoo members built from the W1 task1 binary-frame fit
(figures/data_extset_link.npz, LOO signals):
  reallink  — fitted link shape (λ_fa, λ_miss, ν if the selected model is M3),
              zoo-standard start (SIGMA0, T0) → isolates the OBSERVATION axis
  realpop   — same link + start (σ0, t0) drawn from the fitted population
              (μ_ℓ, σ_ℓ, μ_θ, σ_θ) → full empirical-population arm

and runs the M12 campaign protocol (CRN pool, 30 seeds) under the SHIPPED and
HARDENED (D19: smear_w + p_static 0.3 + evidence gate z=1.28) trainer configs,
with the wellspec control for reference.

Run:  python3 -m studies.study_extset_stress [--smoke]
Out:  figures/data_extset_stress.npz
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np

from studies.study_misspec import (BUDGET, L_STAR, SIG_STAR, SIGMA0, T0, ZOO,
                                   build_pool, run_one, true_params)
from training.benchmark_trainer import T_STAR
from training.misspec_learners import RealLinkLearner, accuracy_bar

FIT_NPZ = "figures/data_extset_link.npz"
N_SEEDS = 30
N_WORKERS = 42
_POOL = None        # set in main() pre-fork; inherited by workers
_BUDGET = None


def _run_job(job):
    j, i, s, cfg_name, mem = job
    r = run_one(mem, "tier2", s, _POOL, budget=_BUDGET,
                **CONFIGS[cfg_name])
    r.pop("_lnr", None)
    return (j, i, s), r
METRICS = ("n_true", "n_decl", "A_decl", "A_final", "ell_decl", "t_decl",
           "cov_ell", "cov_t", "rmse_ell", "rmse_t")
CONFIGS = {
    "shipped": dict(),
    "hardened": dict(p_static=0.3, evidence_z=1.28, smear_w=True),
}


def fitted_link_params():
    """(λ_fa, λ_miss, ν|None, population) from the t1_loo selected model."""
    d = np.load(FIT_NPZ, allow_pickle=True)
    summary = {row[0]: row[1:] for row in d["summary"]}
    best = summary["t1_loo"][0]
    x = d[f"t1_loo_{best}_x"]
    lam = lambda v: 0.49 / (1.0 + np.exp(-np.clip(v, -40, 40)))
    if best == "M0":
        lf = lm = 0.025
    elif best == "M1":
        lf = lm = lam(x[4])
    else:
        lf, lm = lam(x[4]), lam(x[5])
    nu = 1.5 + np.exp(x[6]) if best == "M3" else None
    pop = dict(mu_l=x[0], sd_l=x[1], mu_t=x[2], sd_t=x[3])
    return best, float(lf), float(lm), nu, pop


def register(best, lf, lm, nu, pop):
    ZOO["reallink"] = lambda seed: RealLinkLearner(
        [SIGMA0], [T0], true_params(), lam_fa=lf, lam_miss=lm, nu=nu,
        seed=seed)

    def _realpop(seed):
        rng = np.random.default_rng(7_000_000 + seed)
        ell0 = pop["mu_l"] + pop["sd_l"] * rng.standard_normal()
        th0 = pop["mu_t"] + pop["sd_t"] * rng.standard_normal()
        # engine→plan bridge: σ = e^{−ℓ}, t = −θ (clipped to sane sim range)
        return RealLinkLearner([float(np.clip(np.exp(-ell0), 0.3, 3.0))],
                               [float(np.clip(-th0, -1.5, 1.5))],
                               true_params(), lam_fa=lf, lam_miss=lm, nu=nu,
                               seed=seed)
    ZOO["realpop"] = _realpop


def main(smoke=False):
    best, lf, lm, nu, pop = fitted_link_params()
    print(f"fitted link: {best}  λ_fa={lf:.4f} λ_miss={lm:.4f} "
          f"ν={nu if nu is None else round(nu, 1)}  pop={ {k: round(v, 3) for k, v in pop.items()} }",
          flush=True)
    register(best, lf, lm, nu, pop)

    global _POOL, _BUDGET
    _POOL = build_pool()
    a_bar = accuracy_bar(_POOL, SIG_STAR, T_STAR)
    seeds = list(range(3 if smoke else N_SEEDS))
    _BUDGET = 150 if smoke else BUDGET
    members = ("wellspec", "reallink") if smoke else ("wellspec", "reallink",
                                                      "realpop")
    res = {m: np.full((len(members), len(CONFIGS), len(seeds)), np.nan)
           for m in METRICS}
    print(f"A_bar = {a_bar:.4f}  [{'SMOKE' if smoke else 'FULL'}]", flush=True)
    # fan the 180 independent campaign runs over the machine (fork inherits
    # _POOL and the registered ZOO members)
    jobs = [(j, i, s, cfg_name, mem)
            for j, cfg_name in enumerate(CONFIGS)
            for i, mem in enumerate(members) for s in seeds]
    with mp.Pool(min(N_WORKERS, len(jobs))) as workers:
        for (j, i, s), r in workers.imap_unordered(_run_job, jobs):
            for m in METRICS:
                res[m][i, j, s] = r[m]
    for j, (cfg_name, cfg) in enumerate(CONFIGS.items()):
        for i, mem in enumerate(members):
            nd = res["n_decl"][i, j]
            decl = nd <= _BUDGET
            fg_perf = decl & (res["A_decl"][i, j] < a_bar)
            fg_lat = decl & ((res["ell_decl"][i, j] < L_STAR)
                             | (np.abs(res["t_decl"][i, j]) > T_STAR))
            print(f"  {cfg_name:>8} {mem:>9}: declared {decl.sum():2d}/"
                  f"{len(seeds)}"
                  f" (med {np.median(nd[decl]) if decl.any() else np.nan:5.0f})"
                  f"  FG-perf {fg_perf.sum():2d}  FG-latent {fg_lat.sum():2d}"
                  f"  med A_decl "
                  f"{np.nanmedian(res['A_decl'][i, j]) if decl.any() else np.nan:.3f}"
                  f"  cov_ell {np.nanmean(res['cov_ell'][i, j]):.2f}",
                  flush=True)

    if not smoke:
        np.savez("figures/data_extset_stress.npz",
                 link=np.array([best, lf, lm,
                                np.nan if nu is None else nu], dtype=object),
                 pop=np.array([pop[k] for k in
                               ("mu_l", "sd_l", "mu_t", "sd_t")]),
                 members=np.array(members), configs=np.array(list(CONFIGS)),
                 a_bar=a_bar, budget=_BUDGET, **{m: res[m] for m in METRICS})
        print("saved figures/data_extset_stress.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
