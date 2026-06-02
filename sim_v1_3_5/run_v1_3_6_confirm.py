"""Confirmation: the ACTUAL v1.3.6 live bank (fixed 700 segs in data/eeg_bank.h5,
engine selects from all of them — exactly the internal app) + MAX_Q=500 +
alpha=0.05. Measures per-domain PASS/FAIL resolution for CLEAR candidates, to
confirm the ~92% the resampled bank-size sweep predicted holds on the FIXED draw.
"""
from __future__ import annotations
import argparse, csv, os, statistics as st, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "engine"):
    sys.path.insert(0, str(p))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ALPHA, MAX_Q, N_PART = 0.05, 500, 600
OFFSETS = (-0.8, 0.8)
_W = {}


def _init_worker():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    import cortex_engine_inputs_k7 as ein
    from cortex_policy_k7 import load_ell_star_k7
    inp = ein.build_k7_engine_inputs(verbose=False)   # the live 700-seg bank
    codes = list(inp.task_codes)
    _W.update(inp=inp, codes=codes, K=len(codes),
              ell_star=np.asarray(load_ell_star_k7(codes), float))


def _run_one(task):
    from session_controller import CortexSession
    from cortex_policy import AD6Policy
    from core_mcmc import simulate_response
    inp = _W["inp"]; codes = _W["codes"]; K = _W["K"]; ell = _W["ell_star"]
    man = inp.manifest
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

    def y(k, seg, s):
        return simulate_response(true_sig.get((int(seg), codes[k]), s),
                                 true_t[k], true_l[k], rng_ans)
    policy = AD6Policy(list(ell), list(np.diag(np.asarray(inp.Corr_l, float))),
                       n_min=20, R_star=0.30, alpha=ALPHA, Z=2.0)
    sess = CortexSession(inp, session_id=task["sid"], seed=task["seed"],
                         max_questions=MAX_Q, n_particles=N_PART, policy=policy,
                         ess_threshold_frac=0.5, n_mh_steps=15,
                         max_consecutive_same_domain=12)
    res = sess.run(y)
    v = res.verdicts or ["PENDING"] * K
    return {"offset": task["offset"], "n_questions": res.n_questions,
            "resolved": sum(1 for x in v if x in ("PASS", "FAIL")),
            "stop_reason": res.stop_reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    tasks = [{"offset": o, "seed": 70000 + i, "sid": f"confirm_o{o:+.1f}_n{i:03d}"}
             for o in OFFSETS for i in range(args.n)]
    print(f"running {len(tasks)} sessions on the LIVE 700-seg bank "
          f"(alpha={ALPHA}, MAX_Q={MAX_Q}, N={N_PART}) on {args.procs} procs...", flush=True)
    t0 = time.time()
    with Pool(args.procs, initializer=_init_worker) as pool:
        rows = pool.map(_run_one, tasks)
    print(f"done in {time.time()-t0:.0f}s")
    (REPO / "results" / "sim_v1_3_5").mkdir(parents=True, exist_ok=True)
    with open(REPO / "results/sim_v1_3_5/v1_3_6_confirm_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    resf = 100 * st.mean(r["resolved"] / 7 for r in rows)
    all7 = 100 * st.mean(1 if r["resolved"] == 7 else 0 for r in rows)
    qmed = st.median(r["n_questions"] for r in rows)
    print(f"\n=== LIVE v1.3.6 bank | per-domain resolved={resf:.1f}%  all7={all7:.1f}%  "
          f"q_med={int(qmed)} (clear candidates, alpha=0.05, MAX_Q=500) ===")


if __name__ == "__main__":
    main()
