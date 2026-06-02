"""Bundle-size sweep — how many IIIC questions/domain (download) + MAX_Q (test
length) does alpha=0.05 need to RESOLVE (PASS/FAIL, not REFER) clear candidates?

Download cost (measured): IIIC = 1.69 MB/seg (dominant), spike = 0.10 MB/seg
(~free) => download ~= 10.1*B + 5 MB for B IIIC segs/domain. MAX_Q is test
length (free re: download). This sweep finds the minimum (B, MAX_Q) that gives
majority/significant PASS/FAIL for CLEAR candidates at the NEJM AI alpha=0.05.

Reuses the OC harness core (run_oc_validation) for the production-bank session
build; CPU only, single-thread BLAS.

    .venv/bin/python sim_v1_3_5/run_bank_size_sweep.py --pilot
    .venv/bin/python sim_v1_3_5/run_bank_size_sweep.py --n-per-cell 12 --procs 46
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_v1_3_5"))
import run_oc_validation as ocv  # noqa: E402  (reuse _init_worker/_session_inputs/_W)

RESULTS = REPO / "results" / "sim_v1_3_5"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ALPHA = 0.05                          # the NEJM AI paper-grade threshold
B_IIIC = (50, 100, 150)              # IIIC segs/domain bundled (download lever)
MAX_QS = (300, 500, 700)            # questions asked (test-length lever, free)
OFFSETS = (-0.8, 0.8)               # CLEAR fail / clear pass (known-groups bulk)
N_PARTICLES = 600
MB_PER_IIIC = 1.69                  # measured from data/eeg_bank.h5
SHIP = dict(n_min=20, R_star=0.30, Z=2.0)


def _download_mb(b):
    return 6 * b * MB_PER_IIIC + 5.0   # 6 IIIC domains; spike ~free (~5 MB total)


def _run_one(task):
    from session_controller import CortexSession
    from cortex_policy import AD6Policy
    from core_mcmc import simulate_response
    K = ocv._W["K"]; codes = ocv._W["codes"]; ell = ocv._W["ell_star"]
    fetch = ocv._W["fetch"]
    segs = fetch.sample_session_segments(ocv._W["manifest"], task["sid"],
                                         per_task=task["b"])
    inp, man = ocv._session_inputs(segs)
    true_l = ell + task["offset"]; true_t = np.zeros(K)
    rng_n = np.random.default_rng(task["seed"] + 101)
    true_sig = {}
    for c in codes:
        sm = man.get(f"s_mean_{c}"); sd = man.get(f"s_sd_{c}")
        if sm is None:
            continue
        for sid in man.index[sm.notna()]:
            true_sig[(int(sid), c)] = float(sm[sid]) + rng_n.normal(0, float(sd[sid]))
    rng_ans = np.random.default_rng(task["seed"] + 202)

    def y_source(k, seg_id, s):
        return simulate_response(true_sig.get((int(seg_id), codes[k]), s),
                                 true_t[k], true_l[k], rng_ans)

    policy = AD6Policy(list(ell), list(np.diag(np.asarray(inp.Corr_l, float))),
                       n_min=SHIP["n_min"], R_star=SHIP["R_star"],
                       alpha=ALPHA, Z=SHIP["Z"])
    sess = CortexSession(inp, session_id=task["sid"], seed=task["seed"],
                         max_questions=task["max_q"], n_particles=N_PARTICLES,
                         policy=policy, ess_threshold_frac=ocv.ESS_FRAC,
                         n_mh_steps=ocv.N_MH, max_consecutive_same_domain=12)
    res = sess.run(y_source)
    verdicts = res.verdicts or ["PENDING"] * K
    npt = (policy._last_diag or {}).get("n_per_task", [0] * K)
    resolved = sum(1 for v in verdicts if v in ("PASS", "FAIL"))
    row = {"b": task["b"], "max_q": task["max_q"], "offset": task["offset"],
           "seed": task["seed"], "n_questions": res.n_questions,
           "resolved": resolved, "stop_reason": res.stop_reason,
           "min_npt": int(min(npt)), "max_npt": int(max(npt))}
    return row


def build_tasks(npc, seed_base=99000):
    tasks, i = [], 0
    for b in B_IIIC:
        for mq in MAX_QS:
            for off in OFFSETS:
                for rep in range(npc):
                    tasks.append({"b": b, "max_q": mq, "offset": off, "rep": rep,
                                  "seed": seed_base + i,
                                  "sid": f"b{b}_q{mq}_o{off:+.1f}_n{rep:03d}"})
                    i += 1
    return tasks


def analyze(rows):
    by = {}
    for r in rows:
        by.setdefault((r["b"], r["max_q"]), []).append(r)
    print(f"\n=== alpha=0.05 RESOLUTION of CLEAR candidates by bundle size + MAX_Q ===")
    print("resolved% = mean fraction of the 7 domains reaching PASS/FAIL (not REFER)")
    print(f"  {'IIIC/dom':>8} {'~MB':>6} {'MAX_Q':>6} {'resolved%':>10} "
          f"{'all7%':>6} {'q_med':>6} {'min_npt_med':>11}")
    for b in B_IIIC:
        for mq in MAX_QS:
            rs = by.get((b, mq), [])
            if not rs:
                continue
            resf = 100 * st.mean(r["resolved"] / 7 for r in rs)
            all7 = 100 * st.mean(1 if r["resolved"] == 7 else 0 for r in rs)
            qmed = st.median(r["n_questions"] for r in rs)
            mnpt = st.median(r["min_npt"] for r in rs)  # questions on the WORST-served domain
            print(f"  {b:>8} {_download_mb(b):>5.0f} {mq:>6} {resf:>9.1f}% "
                  f"{all7:>5.1f}% {int(qmed):>6} {int(mnpt):>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=12)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    npc = 1 if args.pilot else args.n_per_cell
    procs = 1 if args.pilot else args.procs
    tasks = build_tasks(npc)
    print(f"running {len(tasks)} sessions ({len(B_IIIC)} B x {len(MAX_QS)} MAX_Q "
          f"x {len(OFFSETS)} offsets x {npc} reps) on {procs} procs, N={N_PARTICLES}...",
          flush=True)
    t0 = time.time()
    if procs == 1:
        ocv._init_worker()
        rows = [_run_one(t) for t in tasks]
    else:
        with Pool(procs, initializer=ocv._init_worker) as pool:
            rows = pool.map(_run_one, tasks)
    print(f"compute done in {time.time()-t0:.1f}s ({1000*(time.time()-t0)/max(len(tasks),1):.0f} ms/session)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "bank_size_sweep_rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    analyze(rows)


if __name__ == "__main__":
    main()
