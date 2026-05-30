"""sim_v1_3_5/run_consec_sweep.py — consecutive-same-domain cap sweep.

Question: does capping how many questions in a row land on one task break up
the long single-domain runs (e.g. 99 seizure in a row) WITHOUT degrading the
all_resolved rate or blowing up session length?

Long IIIC runs happen with HETEROGENEOUS skill: when one IIIC task is
borderline (lingers) while the others are clear (resolve fast), the engine
pours every remaining question into the lingering task. So this sweep builds
heterogeneous raters: one "lingering" IIIC task at a near-cut-score level, the
rest (and spike) clearly below it. We vary which IIIC task lingers.

Sweep max_consec in {0(off), 5, 8, 12, 20} at the SHIP AD6 params
(N_MIN=20, ALPHA=0.25, R_STAR=0.30), K=7. Each (lingering task, rep) uses a
FIXED rater seed shared across all caps, so off-vs-cap is the same rater.

Metric of interest: max_iiic_run (longest consecutive IIIC single-task run,
excluding the intentionally-uncapped spike block).

    .venv/bin/python sim_v1_3_5/run_consec_sweep.py            # full
    .venv/bin/python sim_v1_3_5/run_consec_sweep.py --pilot    # quick
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SIM_LIB = REPO / "sim_v1_2_0"
if str(SIM_LIB) not in sys.path:
    sys.path.insert(0, str(SIM_LIB))
RESULTS = REPO / "results" / "sim_v1_3_5"

CAPS = [0, 5, 8, 12, 20]            # 0 = off (current behavior)
IIIC_TASKS = [1, 2, 3, 4, 5, 6]     # sz, lpd, gpd, lrda, grda, iic (k=0 is spike)
CLEAR = -1.2                        # clear FAIL -> fast resolve
BORDER = 0.4                        # near the cut scores -> lingers
SHIP = dict(n_min=20, alpha=0.25, r_star=0.30)

_W = {}


def _init_worker():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(SIM_LIB))
    from _lib import build_engine_inputs
    _W["inputs"] = build_engine_inputs(7)


def _run_one(task: dict) -> dict:
    sys.path.insert(0, str(SIM_LIB))
    from _lib import run_session, SimParams
    K = 7
    params = SimParams(
        n_min=SHIP["n_min"], alpha=SHIP["alpha"], r_star=SHIP["r_star"],
        n_particles=task["n_particles"], max_questions=300,
        max_consec=task["max_consec"])
    true_t = np.zeros(K)
    true_l = np.full(K, CLEAR)
    true_l[task["lingering_task"]] = BORDER     # the one borderline task
    row = run_session(_W["inputs"], true_t, true_l, params,
                      task["session_id"], seed=task["seed"])
    row["lingering_task"] = task["lingering_task"]
    return row


def build_tasks(n_per_task: int, n_particles: int, seed_base: int = 7000):
    tasks = []
    for d in IIIC_TASKS:
        for rep in range(n_per_task):
            seed = seed_base + d * 1000 + rep        # shared across caps
            for cap in CAPS:
                tasks.append({
                    "max_consec": cap, "lingering_task": d, "seed": seed,
                    "n_particles": n_particles,
                    "session_id": f"C{cap}_d{d}_n{rep:02d}",
                })
    return tasks


def _p95(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def summarize(rows):
    import statistics as st
    by_cap = {}
    for r in rows:
        by_cap.setdefault(r["max_consec"], []).append(r)
    print("\n=== consecutive-same-domain cap sweep (N_MIN=20, ALPHA=0.25, "
          f"{len(rows)} heterogeneous-rater sessions) ===")
    print("iiic_run = longest consecutive IIIC single-task run (the target);  "
          "q = n_questions;  allres% = all_resolved rate")
    hdr = (f"{'cap':>5} {'n':>4} {'iiic_med':>8} {'iiic_p95':>8} {'iiic_max':>8} "
           f"{'q_med':>6} {'q_p95':>6} {'allres%':>8} "
           f"{'pass':>5} {'fail':>5} {'refer':>6}")
    print(hdr); print("-" * len(hdr))
    for cap in CAPS:
        rs = by_cap.get(cap, [])
        if not rs:
            continue
        ir = [r["max_iiic_run"] for r in rs]
        nq = [r["n_questions"] for r in rs]
        allres = 100.0 * sum(1 for r in rs if r["stop_reason"] == "all_resolved") / len(rs)
        npass = st.mean(r["n_pass"] for r in rs)
        nfail = st.mean(r["n_fail"] for r in rs)
        nref = st.mean(r["n_refer_borderline"] + r["n_refer_uninformative"] for r in rs)
        print(f"{cap if cap else 'off':>5} {len(rs):>4} {int(st.median(ir)):>8} "
              f"{int(_p95(ir)):>8} {int(max(ir)):>8} {int(st.median(nq)):>6} "
              f"{int(_p95(nq)):>6} {allres:>7.1f}% "
              f"{npass:>5.2f} {nfail:>5.2f} {nref:>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-task", type=int, default=8)
    ap.add_argument("--n-particles", type=int, default=300)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    npt = 2 if args.pilot else args.n_per_task
    tasks = build_tasks(npt, args.n_particles)
    print(f"running {len(tasks)} sessions ({len(CAPS)} caps x {len(IIIC_TASKS)} "
          f"lingering-tasks x {npt} reps) on {args.procs} procs...", flush=True)
    with Pool(args.procs, initializer=_init_worker) as pool:
        rows = pool.map(_run_one, tasks)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "consec_sweep_rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    summarize(rows)


if __name__ == "__main__":
    main()
