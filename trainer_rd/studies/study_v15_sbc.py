"""M11 — SBC-style coverage study: 600 vs 1200 particles (spec item 5 check).

The spec's calibration table found n_particles=600 mildly under-covers the
95% interval (0.932) while 1200 is nominal at every level (K=6 Mode-A
harness). This is the scratch K=7 + corr_t analogue — the run the spec
itself lists as "would close the last gap":

  truth (theta, ell) ~ the hierarchical prior N(0, blockdiag(Corr_l, Corr_t))
  → full K=7 adaptive eval (v15 cuts, AD6 stopping — the instrument's actual
    operating posterior; Bayesian coverage is stopping-rule-invariant)
  → empirical coverage of equal-tailed 50/80/90/95% credible intervals on
    ell_k (7 intervals per rater), at n_particles 600 vs 1200.

Run:  python3 -m studies.study_v15_sbc [--pilot]
Out:  figures/data_v15_sbc.npz  (+ printed coverage table)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from engine.instrument_v15 import instrument

TASKS = tuple(range(7))
MAX_Q = 420
N_RATERS = 100
LEVELS = (0.50, 0.80, 0.90, 0.95)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "data_v15_sbc.npz")

_BANK = None


def _wq(x, w, q):
    """Weighted quantile (sorted interpolation)."""
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(np.interp(q, cw / cw[-1], x[o]))


def run_session(job):
    global _BANK
    n_part, rater = job
    import engine.core_mcmc_general as eng
    from training.bank_adapter import BankAdapter
    from training.pipeline_demo import _run_eval
    if _BANK is None:
        _BANK = BankAdapter()
    ins = instrument("v15")
    sub = ins.for_tasks(TASKS)
    rng = np.random.default_rng(777_000 + rater)
    # truth from the SAME hierarchical prior the engine assumes (SBC premise)
    t_true, l_true = eng.sample_prior_hier_K(
        1, 7, 0.378, rng, Sigma_l=sub["Sigma_l"], Sigma_t=sub["Sigma_t"])
    t_true, l_true = t_true[0], l_true[0]
    state, verdicts, _, _, n_q = _run_eval(
        t_true, l_true, _BANK, tasks=TASKS, session_id=f"sbc-{rater}",
        n_part=n_part, max_q=MAX_Q, seed=900_000 + rater,
        ell_star_vec=sub["ell_star"], Sigma_l=sub["Sigma_l"],
        Sigma_t=sub["Sigma_t"])
    w = state["w"] / state["w"].sum()
    cover = {}
    for lev in LEVELS:
        a = (1.0 - lev) / 2.0
        hits = []
        for k in range(7):
            lo = _wq(state["l"][:, k], w, a)
            hi = _wq(state["l"][:, k], w, 1.0 - a)
            hits.append(bool(lo <= l_true[k] <= hi))
        cover[str(lev)] = hits
    return {"n_part": n_part, "rater": rater, "cover": cover, "n_q": n_q}


def main():
    pilot = "--pilot" in sys.argv
    n_raters = 6 if pilot else N_RATERS
    jobs = [(np_, r) for np_ in (600, 1200) for r in range(n_raters)]
    n_workers = int(os.environ.get("SBC_WORKERS", min(40, os.cpu_count() - 4)))
    print(f"{len(jobs)} SBC sessions on {n_workers} workers "
          f"({'PILOT' if pilot else 'FULL'})", flush=True)
    results, t0 = [], time.time()
    with mp.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_session, jobs), 1):
            results.append(res)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
                np.savez(OUT, results_json=json.dumps(results))
    np.savez(OUT, results_json=json.dumps(results))

    print(f"\n{'n_part':>7} " + " ".join(f"{int(100*l):>5}%" for l in LEVELS)
          + f" {'n_int':>6}")
    for np_ in (600, 1200):
        rows = [r for r in results if r["n_part"] == np_]
        line = f"{np_:>7} "
        n_int = 0
        for lev in LEVELS:
            hits = np.concatenate([r["cover"][str(lev)] for r in rows])
            n_int = hits.size
            line += f"{hits.mean():>6.3f}"
        print(line + f" {n_int:>6}")


if __name__ == "__main__":
    main()
