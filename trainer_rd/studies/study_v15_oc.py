"""M11 — v15 operating-characteristic study (spec reproduction, scratch harness).

Reproduces the DIRECTION and approximate magnitude of the v15 claim tables on
the scratch engine + real 89k bank: per-task PASS resolution at true skill
ell = ell* + offset, intrinsic error rates, and the realistic-bias regime.

Arms (instrument_v15):
  v14        — shipped: v14 cuts, Corr_l both prior blocks, 600 particles
  v15        — staged : v15 cuts, Corr_t t-block,           1200 particles
  v15cuts600 — A1 ablation: v15 cuts ONLY (Corr_l t-block, 600 particles)

Cells: offsets +0.3/+0.6/+1.0 zero-bias (resolution), −0.3 zero-bias
(false-PASS), +0.6 realistic-bias (theta ~ N(mu_t, Sigma_t), spec item 9).
false-FAIL = FAIL verdicts among above-bar raters (+0.6, +1.0 cells).

Session = full K=7 adaptive eval (pipeline_demo._run_eval), max_q=420 — the
spec's "bank-limited ~420 q" regime (claim row: v15 48/77/87%).

Run:  python3 -m studies.study_v15_oc [--pilot]      (~25-40 min full, 40 workers)
Out:  figures/data_v15_oc.npz  (+ printed table)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from engine.instrument_v15 import instrument, draw_examinee_theta, N_PARTICLES_FROZEN

TASKS = tuple(range(7))
MAX_Q = 420
POOL_SIZE = 800
N_RATERS = 24
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "data_v15_oc.npz")

CELLS = [
    ("v14", 0.3, "zero"), ("v14", 0.6, "zero"), ("v14", 1.0, "zero"),
    ("v15", 0.3, "zero"), ("v15", 0.6, "zero"), ("v15", 1.0, "zero"),
    ("v15cuts600", 0.6, "zero"),
    ("v14", -0.3, "zero"), ("v15", -0.3, "zero"),
    ("v14", 0.6, "realistic"), ("v15", 0.6, "realistic"),
]

_BANK = None     # per-worker singleton (fork-safe lazy init)


def _get_instrument(arm):
    if arm == "v15cuts600":
        ins = instrument("v15")
        ins.Sigma_t_prior = ins.Sigma_l_prior.copy()   # undo corr_t swap
        ins.n_particles = N_PARTICLES_FROZEN
        return ins
    return instrument(arm)


def run_session(job):
    """One rater through one full K=7 eval. Returns dict of verdict codes."""
    global _BANK
    arm, offset, bias_mode, rater = job
    from training.bank_adapter import BankAdapter
    from training.pipeline_demo import _run_eval
    if _BANK is None:
        _BANK = BankAdapter()
    ins = _get_instrument(arm)
    sub = ins.for_tasks(TASKS)
    cell_seed = abs(hash((arm, round(offset * 10), bias_mode))) % (2 ** 20)
    seed = cell_seed * 1000 + rater
    rng = np.random.default_rng(seed)
    true_l = np.array(sub["ell_star"]) + offset
    true_t = draw_examinee_theta(rng, 1, zero_bias=(bias_mode == "zero"))[0]
    t0 = time.time()
    _, verdicts, _, _, n_q = _run_eval(
        true_t, true_l, _BANK, tasks=TASKS, session_id=f"{arm}-{rater}",
        n_part=sub["n_particles"], max_q=MAX_Q, seed=seed,
        pool_size=POOL_SIZE, ell_star_vec=sub["ell_star"],
        Sigma_l=sub["Sigma_l"], Sigma_t=sub["Sigma_t"])
    return {"arm": arm, "offset": offset, "bias": bias_mode, "rater": rater,
            "verdicts": verdicts, "n_q": n_q, "wall_s": time.time() - t0}


def aggregate(results):
    """Per-cell PASS/FAIL/REFER rates over rater x task verdicts."""
    cells = {}
    for r in results:
        key = (r["arm"], r["offset"], r["bias"])
        cells.setdefault(key, []).extend(r["verdicts"])
    table = {}
    for key, v in sorted(cells.items()):
        v = np.array(v)
        n = len(v)
        table[key] = {"n": n,
                      "PASS": float((v == "PASS").mean()),
                      "FAIL": float((v == "FAIL").mean()),
                      "REFER": float(1.0 - (v == "PASS").mean()
                                     - (v == "FAIL").mean())}
    return table


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(c - h, 0.0), min(c + h, 1.0)


def main():
    pilot = "--pilot" in sys.argv
    n_raters = 2 if pilot else N_RATERS
    jobs = [(arm, off, bias, r) for (arm, off, bias) in CELLS
            for r in range(n_raters)]
    n_workers = min(40, os.cpu_count() - 4)
    print(f"{len(jobs)} sessions on {n_workers} workers "
          f"({'PILOT' if pilot else 'FULL'})", flush=True)
    results, t0 = [], time.time()
    with mp.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_session, jobs), 1):
            results.append(res)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
                np.savez(OUT, results_json=json.dumps(results))
    np.savez(OUT, results_json=json.dumps(results))

    table = aggregate(results)
    print(f"\n{'arm':>11} {'offset':>7} {'bias':>10} {'n':>4} "
          f"{'PASS%':>6} {'FAIL%':>6} {'REFER%':>7}  95% CI(PASS)")
    for (arm, off, bias), row in table.items():
        lo, hi = wilson_ci(row["PASS"], row["n"])
        print(f"{arm:>11} {off:>+7.1f} {bias:>10} {row['n']:>4} "
              f"{100*row['PASS']:>6.1f} {100*row['FAIL']:>6.1f} "
              f"{100*row['REFER']:>7.1f}  [{100*lo:.1f},{100*hi:.1f}]")
    # intrinsic error summary
    for arm in ("v14", "v15"):
        fp = table.get((arm, -0.3, "zero"), {"PASS": np.nan})["PASS"]
        ff_cells = [table[k] for k in table
                    if k[0] == arm and k[1] >= 0.6 and k[2] == "zero"]
        ff = (np.mean([c["FAIL"] for c in ff_cells]) if ff_cells else np.nan)
        print(f"{arm}: false-PASS(below-bar -0.3) = {100*fp:.2f}%   "
              f"false-FAIL(above-bar) = {100*ff:.2f}%")
    return results, table


if __name__ == "__main__":
    main()
