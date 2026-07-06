"""B8 — K=7 re-certification replication (M12 F32 carry-over).

F32 showed the REAL ENGINE's K=1 re-cert catches 75/75 trainer false
graduations. Replication at deployment scale: full K=7 adaptive eval
(shipped v14 instrument, joint hierarchical prior — the F34-caveat-ii
configuration) re-certifying learners at their TRAINER-DECLARATION state.

Per seed × config: train the EXTSET-fitted `realpop` learner on domain3
(`run_one(..., stop_at_decl=True)` → true latent state at declaration);
embed that state in a K=7 examinee whose other six domains are zero-bias
fitted-population draws; run the full eval; record the domain3 verdict vs
the declaration's FG-latent status.

Run:  python3 -m studies.study_recert_k7 [--smoke]
Out:  figures/data_recert_k7.npz
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np

from engine.instrument_v15 import instrument
from studies.study_extset_stress import (CONFIGS, fitted_link_params,
                                         register)
from studies.study_misspec import (BUDGET, L_STAR, TASK, build_pool, run_one)
from training.bank_adapter import BankAdapter
from training.benchmark_trainer import T_STAR
from training.pipeline_demo import _run_eval

N_WORKERS = 40
N_SEEDS = 30
MAX_Q = 420
POOL_SIZE = 800
SEED0 = 20260612

_POOL = _BANK = _INS = _POP = None


def _train_job(args):
    cfg_name, seed = args
    r = run_one("realpop", "tier2", seed, _POOL, budget=BUDGET,
                stop_at_decl=True, **CONFIGS[cfg_name])
    lnr = r.pop("_lnr", None)
    if r["n_decl"] > BUDGET or lnr is None:
        return cfg_name, seed, None
    ell = float(-np.log(lnr.sigma[0]))
    th = float(-lnr.t[0])
    fg = (ell < L_STAR) or (abs(lnr.t[0]) > T_STAR)
    return cfg_name, seed, (ell, th, fg, r["n_decl"])


def _eval_job(args):
    cfg_name, seed, ell, th, fg = args
    rng = np.random.default_rng(SEED0 + seed)
    sub = _INS.for_tasks(range(7))
    true_l = _POP["mu_l"] + _POP["sd_l"] * rng.standard_normal(7)
    true_t = np.zeros(7)                       # zero-bias others (primary)
    true_l[TASK], true_t[TASK] = ell, th
    _, verdicts, _, _, n_q = _run_eval(
        true_t, true_l, _BANK, tasks=range(7),
        session_id=f"recert-{cfg_name}-{seed}", n_part=sub["n_particles"],
        max_q=MAX_Q, seed=SEED0 + seed, pool_size=POOL_SIZE,
        ell_star_vec=sub["ell_star"], Sigma_l=sub["Sigma_l"],
        Sigma_t=sub["Sigma_t"])
    return cfg_name, seed, fg, verdicts[TASK], n_q


def main(smoke=False):
    global _POOL, _BANK, _INS, _POP
    best, lf, lm, nu, pop = fitted_link_params()
    register(best, lf, lm, nu, pop)
    _POOL = build_pool()
    _BANK = BankAdapter()
    _INS = instrument("v14")
    _POP = pop
    seeds = range(4 if smoke else N_SEEDS)

    jobs = [(c, s) for c in CONFIGS for s in seeds]
    with mp.Pool(min(N_WORKERS, len(jobs))) as p:
        trained = [t for t in p.map(_train_job, jobs) if t[2] is not None]
    print(f"declaration states: {len(trained)}/{len(jobs)}", flush=True)

    ejobs = [(c, s, st[0], st[1], st[2]) for c, s, st in trained]
    with mp.Pool(min(N_WORKERS, len(ejobs))) as p:
        out = p.map(_eval_job, ejobs)

    res = {}
    for c, s, fg, verdict, n_q in out:
        res.setdefault((c, fg), []).append(verdict)
    for (c, fg), v in sorted(res.items()):
        v = np.array(v)
        passed = (v == "PASS").sum()
        print(f"  {c:>8} {'FG' if fg else 'genuine':>8} (n={len(v)}): "
              f"K=7 re-cert PASS {passed}/{len(v)}  "
              f"verdicts {dict(zip(*np.unique(v, return_counts=True)))}",
              flush=True)
    if not smoke:
        np.savez("figures/data_recert_k7.npz",
                 rows=np.array(out, dtype=object))
        print("saved figures/data_recert_k7.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
