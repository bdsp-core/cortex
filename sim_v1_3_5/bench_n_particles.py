"""Step-2 latency probe — per-question wall-clock of the LIVE K=7 CortexSession
at N ∈ {600, 1200, 2400} particles, on the real bank.

This measures the binding constraint for raising N: the GUI must stay
responsive (a human answers in seconds, so per-question engine compute up to
~1–2 s is invisible). Runs one simulated mid-skill rater to a fixed budget per N
(NoStopPolicy so it doesn't stop early), timing each trial.

    python sim_v1_3_5/bench_n_particles.py --budget 80 --ns 600,1200,2400
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "engine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=80,
                    help="questions per session (fixed; NoStop)")
    ap.add_argument("--ns", type=str, default="600,1200,2400")
    ap.add_argument("--offset", type=float, default=0.3,
                    help="true ℓ = ℓ* + offset (mid-skill rater)")
    args = ap.parse_args()
    Ns = [int(x) for x in args.ns.split(",")]

    import cortex_engine_inputs_k7 as ein
    from cortex_policy import NoStopPolicy
    from cortex_policy_k7 import load_ell_star_k7
    from session_controller import CortexSession
    from core_mcmc import simulate_response

    inp = ein.build_k7_engine_inputs()
    K = len(inp.task_codes)
    ell_star = np.asarray(load_ell_star_k7(list(inp.task_codes)), float)
    nseg = len(inp.all_seg_ids)
    print(f"bank: {nseg} segs, K={K}; budget={args.budget} q/session; "
          f"offset={args.offset:+.2f}", flush=True)

    true_l = ell_star + args.offset
    true_t = np.zeros(K)

    print(f"\n{'N':>6} {'q_done':>7} {'ms/q mean':>10} {'ms/q p50':>9} "
          f"{'ms/q p95':>9} {'select%':>8} {'total_s':>8}")
    for N in Ns:
        rng_ans = np.random.default_rng(7)

        def y_source(k, seg_id, s):
            return simulate_response(s, true_t[k], true_l[k], rng_ans)

        q_ms, sel_ms = [], []
        last = {"t": None}

        def on_trial(tel):
            now = time.perf_counter()
            if last["t"] is not None:
                q_ms.append((now - last["t"]) * 1000.0)
            last["t"] = now
            sel_ms.append(float(tel.get("select_ms", 0.0)))

        sess = CortexSession(inp, session_id=f"bench_N{N}", seed=0,
                             max_questions=args.budget, n_particles=N,
                             policy=NoStopPolicy(), ess_threshold_frac=0.5,
                             n_mh_steps=15)
        last["t"] = time.perf_counter()
        t0 = time.perf_counter()
        res = sess.run(y_source, on_trial=on_trial)
        total = time.perf_counter() - t0
        nq = res.n_questions
        mean_ms = st.mean(q_ms) if q_ms else float("nan")
        p50 = st.median(q_ms) if q_ms else float("nan")
        p95 = (sorted(q_ms)[int(0.95 * len(q_ms))] if len(q_ms) > 3
               else float("nan"))
        sel_frac = 100 * (st.mean(sel_ms) / mean_ms) if q_ms else float("nan")
        print(f"{N:>6} {nq:>7} {mean_ms:>10.1f} {p50:>9.1f} {p95:>9.1f} "
              f"{sel_frac:>7.1f}% {total:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
