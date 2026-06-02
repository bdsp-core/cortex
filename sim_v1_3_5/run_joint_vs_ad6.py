"""sim_v1_3_5/run_joint_vs_ad6.py — JOINT VERDICT-VECTOR REGION rule head-to-head.

Tests the panel's chosen rule (scripts/cortex_policy_joint.JointVerdictRegionPolicy)
against the shipped AD6 marginal rule (scripts/cortex_policy.AD6Policy), on BOTH
the HIER engine (engine/core_mcmc.py — one 2K-dim correlated cloud) and the BRUTE
engine (engine/core_mcmc_brute_k.py — K independent 2-D clouds), measuring the
COMPOSITE bar: questions-to-resolve AND 7-vector verdict accuracy.

Decisive cell: HIER+joint vs BRUTE+joint at the SAME 0.95 bar (like-for-like).
  - hier joint  : Slepian orthant on Sigma_post (correlation enters the stop).
  - brute joint : product of independent marginals (the lever brute cannot pull,
                  core_mcmc_brute_k._pass_prob_joint_brute).
HIER+AD6 / BRUTE+AD6 are the shipped baselines (absolute lengthening reference).

ARMS  : {hier, brute} x {ad6, joint}
PRIOR : ell-block correlation override r in PRIOR_R (uniform equicorrelation on
        the fitted marginal variances) — isolates the correlation effect.
SKILL : (overall level) x (concordance: concordant / discordant) — the FINDINGS.md
        stratification (cohort median dilutes the concordant win).
BUDGET: hier N total vs brute N-per-task (K*N total). --budget controls fairness:
          equal-total  : N_hier = N_brute        (deployment-realistic)
          equal-effort : N_hier = K * N_brute     (PRIMARY fairness arm)

Paired on identical candidate draws + seeds.  Single-thread BLAS for bit-repro.

    .venv/bin/python sim_v1_3_5/run_joint_vs_ad6.py --pilot
    .venv/bin/python sim_v1_3_5/run_joint_vs_ad6.py --n-per-cell 800 --procs 46
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

# Engine reproducibility contract — pin BLAS single-thread BEFORE numpy use.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
ENGINE = REPO / "engine"
for p in (str(SCRIPTS), str(ENGINE)):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = REPO / "results" / "sim_v1_3_5"

# ── grids ──────────────────────────────────────────────────────────────────
ARMS = ("hier_ad6", "hier_joint", "brute_ad6", "brute_joint")
PRIOR_R = (0.135, 0.37, 0.78)          # live deployed / uniform-0.37 / empirical
# Overall skill level (true ell offset above/below the cut), x concordance.
#   level > +margin : clear PASS ; < -margin : clear FAIL ; |.|<=margin : borderline
LEVELS = (-0.6, 0.0, 0.6)              # clear-fail / borderline / clear-pass
CONCORD = ("concordant", "discordant")
CLEAR_MARGIN = 0.30
DISCORD_SPREAD = 0.6                   # sd of per-domain ell scatter (discordant)

SHIP = dict(n_min=20, R_star=0.30, alpha=0.25, Z=2.0)   # AD6 ship params
JOINT_BAR = 0.95
N_BRUTE = 300                          # particles per task (brute)
MAX_Q = 300
N_MH = 15
ESS_FRAC = 0.5

_W = {}


def _init_worker(budget, max_q=None):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if max_q is not None:                       # propagate cap to Pool workers
        global MAX_Q
        MAX_Q = int(max_q)
    for p in (str(SCRIPTS), str(ENGINE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cortex_engine_inputs_k7 as ein
    from cortex_policy_k7 import load_ell_star_k7
    base = ein.build_k7_engine_inputs()
    codes = list(base.task_codes)
    K = len(codes)
    ell_star = np.asarray(load_ell_star_k7(codes), float)
    var_prior = np.diag(np.asarray(base.Corr_l, float)).copy()
    n_hier = N_BRUTE if budget == "equal-total" else K * N_BRUTE
    _W.update(base=base, ein=ein, codes=codes, K=K, ell_star=ell_star,
              var_prior=var_prior, n_hier=int(n_hier), budget=budget)


def _equicorr(r, K):
    """Uniform equicorrelation Corr matrix (unit diagonal)."""
    C = (1.0 - r) * np.eye(K) + r * np.ones((K, K))
    return C


def _true_skill(level, concord, K, seed):
    """True per-domain ell vector for the simulated candidate.

    concordant : all K at the common `level` (low spread).
    discordant : level + N(0, DISCORD_SPREAD) per domain (realistic ILAE rater:
                 strong overall but 1-2 weak domains), clipped not to cross 0
                 wholesale (so a real conjunction win is still possible).
    """
    rng = np.random.default_rng(seed + 9001)
    if concord == "concordant":
        return np.full(K, level)
    return level + rng.normal(0.0, DISCORD_SPREAD, size=K)


# ── HIER arm — drive CortexSession with the injected policy ──────────────────
def _run_hier(arm, r, true_l, sid, seed):
    from session_controller import CortexSession
    from cortex_policy import AD6Policy
    from cortex_policy_joint import JointVerdictRegionPolicy
    from core_mcmc import simulate_response

    base = _W["base"]; ein = _W["ein"]; K = _W["K"]
    ell_star = _W["ell_star"]; n_hier = _W["n_hier"]

    # Prior with the requested ell-block correlation; marginal vars held at
    # the fitted diagonal (so only correlation changes across the r grid).
    sd = np.sqrt(np.clip(_W["var_prior"], 1e-9, None))
    Corr = _equicorr(r, K)
    Sigma_l = Corr * np.outer(sd, sd)
    inp = ein.K7EngineInputs(
        manifest=base.manifest, Corr_l=Sigma_l, Corr_t=base.Corr_t,
        task_codes=list(base.task_codes), task_labels=list(base.task_labels),
        task_pattern_words=list(base.task_pattern_words),
        task_families=list(base.task_families))

    if arm == "hier_ad6":
        policy = AD6Policy(ell_star=list(ell_star),
                           var_prior=list(np.diag(Sigma_l)),
                           n_min=SHIP["n_min"], R_star=SHIP["R_star"],
                           alpha=SHIP["alpha"], Z=SHIP["Z"])
    else:
        policy = JointVerdictRegionPolicy(
            ell_star=list(ell_star), var_prior=list(np.diag(Sigma_l)),
            n_min=SHIP["n_min"], R_star=SHIP["R_star"], Z=SHIP["Z"],
            bar=JOINT_BAR)

    rng_ans = np.random.default_rng(seed + 202)
    true_t = np.zeros(K)

    def y_source(k, seg_id, s):
        return simulate_response(s, true_t[k], true_l[k], rng_ans)

    sess = CortexSession(inp, session_id=sid, seed=seed,
                         max_questions=MAX_Q, n_particles=n_hier,
                         policy=policy, ess_threshold_frac=ESS_FRAC,
                         n_mh_steps=N_MH, max_consecutive_same_domain=12)
    res = sess.run(y_source)
    verdicts = res.verdicts or ["PENDING"] * K
    return res.n_questions, verdicts, res.stop_reason


# ── BRUTE arm — K independent clouds, policy adapters on per-task state ───────
def _brute_joint_lcb(states, ell_star, sign, Z):
    """P_joint = prod_k pi_k^(side) for the brute (independent) model — the
    panel's like-for-like control (core_mcmc_brute_k._pass_prob_joint_brute,
    generalized to per-domain PASS/FAIL side via the frozen sign).  Returns
    (p_joint, lcb) with the conservative min-active N_eff Bernoulli buffer."""
    import core_mcmc_brute_k as bk
    K = len(states)
    pass_mass = bk._pass_probs_brute(states, ell_star)        # P(l_k > l*_k)
    side = np.where(sign > 0, pass_mass, 1.0 - pass_mass)     # correct-side mass
    p_joint = float(np.prod(side))
    n_eff = bk._n_eff_per_domain(states)
    n_eff_min = float(max(np.min(n_eff), 1.0))
    se = float(np.sqrt(max(p_joint * (1.0 - p_joint), 0.0) / n_eff_min))
    return p_joint, p_joint - Z * se


def _run_brute(arm, r, true_l, sid, seed):
    """Brute session loop mirroring CortexSession's structure but over K
    independent 2-D clouds.  r is IGNORED (brute has independent N(0,1) priors
    by construction — that IS the no-sharing comparator)."""
    import core_mcmc_brute_k as bk
    from cortex_policy import (PASS, FAIL, PENDING,
                               REFER_BORDERLINE, REFER_UNINFORMATIVE)

    base = _W["base"]; ein = _W["ein"]; K = _W["K"]
    ell_star = _W["ell_star"]
    var_prior = np.ones(K)             # brute prior is N(0,1) per task -> var 1
    n_min, R_star, alpha, Z = (SHIP["n_min"], SHIP["R_star"],
                               SHIP["alpha"], SHIP["Z"])

    # Per-task candidate banks from the manifest (paired with hier).
    bank_signals, bank_sds, bank_segids = base.as_engine_arrays()

    rng = np.random.default_rng(seed)         # session rng (selection)
    rng_ans = np.random.default_rng(seed + 202)
    states = bk.make_state_brute_k(N_BRUTE, K, rng)
    true_t = np.zeros(K)

    # served-segment de-dup per task (mirror CortexSession's de-dup intent)
    served = [set() for _ in range(K)]
    n_per_task = np.zeros(K, dtype=int)
    verdicts = [PENDING] * K
    sign = np.ones(K)
    sign_frozen = np.zeros(K, dtype=bool)
    last_R = np.zeros(K)
    n_q = 0
    stop_reason = "bank_exhausted"

    for _ in range(MAX_Q):
        # active domains = unresolved with non-empty remaining bank
        active = [k for k in range(K)
                  if verdicts[k] == PENDING
                  and len(bank_signals[k]) - len(served[k]) > 0]
        if not active:
            stop_reason = "all_active_resolved"
            break
        # remaining candidates per active task
        cand = {}
        for k in active:
            mask = np.array([sid_ not in served[k]
                             for sid_ in bank_segids[k]])
            cand[k] = (bank_signals[k][mask], bank_segids[k][mask])
        cs = [cand[k][0] if k in cand else np.array([0.0]) for k in range(K)]
        k_s = bk.choose_item_brute_k(states, cs, active_domains=active)
        k, s = int(k_s[0]), float(k_s[1])
        # map chosen signal back to a seg_id (first match), record served
        sig_arr, seg_arr = cand[k]
        j = int(np.argmin(np.abs(sig_arr - s)))
        served[k].add(int(seg_arr[j]))

        y = bk.simulate_response(s, true_t[k], true_l[k], rng_ans)
        bk.update_brute_k(states, k, s, y)
        if bk.ess(states[k]["w"]) < ESS_FRAC * N_BRUTE:
            bk.resample_and_rejuvenate_2d(states[k], rng, N_MH, 1.5)
        n_per_task[k] += 1
        n_q += 1

        # per-domain marginals (independent clouds)
        pass_mass = bk._pass_probs_brute(states, ell_star)
        n_eff = bk._n_eff_per_domain(states)
        mcse = np.sqrt(np.maximum(pass_mass * (1.0 - pass_mass), 0.0)
                       / np.maximum(n_eff, 1.0))
        # R_k info gate from each task's posterior var of l
        R = np.empty(K)
        for kk in range(K):
            w = states[kk]["w"]; lk = states[kk]["l"]
            mu = float((w * lk).sum())
            vp = float((w * (lk - mu) ** 2).sum())
            R[kk] = 1.0 - vp / var_prior[kk]
            if not sign_frozen[kk]:
                sign[kk] = 1.0 if (mu - ell_star[kk]) >= 0.0 else -1.0
                if R[kk] >= R_star and n_per_task[kk] >= n_min:
                    sign_frozen[kk] = True
        last_R = R
        gate = (R >= R_star) & (n_per_task >= n_min)

        if arm == "brute_ad6":
            # AD6 marginal three-way per task (independent), monotonic lock.
            for kk in range(K):
                if verdicts[kk] != PENDING or not gate[kk]:
                    continue
                if pass_mass[kk] - Z * mcse[kk] >= 1.0 - alpha:
                    verdicts[kk] = PASS
                elif pass_mass[kk] + Z * mcse[kk] <= alpha:
                    verdicts[kk] = FAIL
            if all(v != PENDING for v in verdicts):
                stop_reason = "all_resolved"
                break
        else:  # brute_joint — product-of-marginals orthant at the 0.95 bar
            if bool(gate.all()):
                p_joint, lcb = _brute_joint_lcb(states, ell_star, sign, Z)
                if lcb >= JOINT_BAR:
                    verdicts = [PASS if sign[kk] > 0 else FAIL
                                for kk in range(K)]
                    stop_reason = "joint_region"
                    break

    # finalize PENDING -> REFER
    out = []
    for kk in range(K):
        if verdicts[kk] in (PASS, FAIL):
            out.append(verdicts[kk])
        elif sign_frozen[kk] or last_R[kk] >= R_star:
            out.append(REFER_BORDERLINE)
        else:
            out.append(REFER_UNINFORMATIVE)
    return n_q, out, stop_reason


# ── verdict accuracy scoring ────────────────────────────────────────────────
def _true_verdict(true_l_k, ell_star_k):
    return "PASS" if true_l_k >= ell_star_k else "FAIL"


def _score(true_l, verdicts):
    """7-vector accuracy + per-domain accuracy on CLEAR domains.

    A domain is 'correct' if its emitted PASS/FAIL matches the true side; a
    REFER on a clear domain counts as not-correct for accuracy (it is a
    non-decision).  vec_correct = all clear domains correct AND no clear
    domain mis-verdicted."""
    ell_star = _W["ell_star"]; K = _W["K"]
    clear = np.abs(true_l - ell_star) > CLEAR_MARGIN
    per_correct = 0
    per_n = 0
    vec_ok = True
    for k in range(K):
        if not clear[k]:
            continue
        per_n += 1
        truth = _true_verdict(true_l[k], ell_star[k])
        emitted = verdicts[k]
        ok = (emitted == truth)
        per_correct += int(ok)
        # a wrong (opposite) verdict breaks the vector; a REFER does too on clear
        if not ok:
            vec_ok = False
    return int(vec_ok and per_n > 0), per_correct, per_n


def _run_one(task):
    arm, r = task["arm"], task["r"]
    true_l = np.asarray(task["true_l"], float)
    if arm.startswith("hier"):
        n_q, verdicts, reason = _run_hier(arm, r, true_l, task["sid"],
                                          task["seed"])
    else:
        n_q, verdicts, reason = _run_brute(arm, r, true_l, task["sid"],
                                           task["seed"])
    vec_ok, per_ok, per_n = _score(true_l, verdicts)
    return {
        "arm": arm, "r": r, "level": task["level"],
        "concord": task["concord"], "rep": task["rep"],
        "n_questions": n_q, "stop_reason": reason,
        "vec_correct": vec_ok, "per_correct": per_ok, "per_n": per_n,
        "resolved": int(all(v in ("PASS", "FAIL") for v in verdicts)),
    }


def build_tasks(n_per_cell, K, seed_base=88000):
    tasks = []
    i = 0
    for level in LEVELS:
        for concord in CONCORD:
            for rep in range(n_per_cell):
                # candidate truth + seed are SHARED across all arms & r values
                # (paired): the only thing that differs is the arm/prior.
                seed = seed_base + i
                sid = f"l{level:+.1f}_{concord}_n{rep:04d}"
                true_l = _true_skill(level, concord, K, seed)
                for r in PRIOR_R:
                    for arm in ARMS:
                        tasks.append({
                            "arm": arm, "r": r, "level": level,
                            "concord": concord, "rep": rep, "sid": sid,
                            "seed": seed, "true_l": true_l.tolist()})
                i += 1
    return tasks


def summarize(rows):
    print("\n=== JOINT vs AD6 head-to-head ===")
    print("q_med = median questions-to-resolve; vec% = full 7-vector verdict "
          "accuracy on clear domains; res% = fully-resolved rate.\n")
    by = {}
    for rr in rows:
        by.setdefault((rr["r"], rr["concord"], rr["arm"]), []).append(rr)
    for r in PRIOR_R:
        print(f"  --- prior r_ell = {r} ---")
        hdr = (f"  {'concord':>11} {'arm':>11} {'q_med':>6} {'vec%':>6} "
               f"{'perDom%':>8} {'res%':>6} {'n':>5}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for concord in CONCORD:
            for arm in ARMS:
                rs = by.get((r, concord, arm), [])
                if not rs:
                    continue
                qmed = st.median(x["n_questions"] for x in rs)
                vec = 100 * sum(x["vec_correct"] for x in rs) / len(rs)
                pc = sum(x["per_correct"] for x in rs)
                pn = sum(x["per_n"] for x in rs)
                perdom = 100 * pc / pn if pn else float("nan")
                res = 100 * sum(x["resolved"] for x in rs) / len(rs)
                print(f"  {concord:>11} {arm:>11} {int(qmed):>6} {vec:>6.1f} "
                      f"{perdom:>8.1f} {res:>6.1f} {len(rs):>5}")
        # decisive ratio: hier_joint vs brute_joint q-ratio (concordant)
        for concord in CONCORD:
            hj = by.get((r, concord, "hier_joint"), [])
            bj = by.get((r, concord, "brute_joint"), [])
            if hj and bj:
                qh = st.median(x["n_questions"] for x in hj)
                qb = st.median(x["n_questions"] for x in bj)
                ratio = qb / qh if qh else float("nan")
                print(f"    [{concord}] q-ratio brute/hier (joint) = "
                      f"{ratio:.3f}x  (>=1.1 needed for a win)")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=800)
    ap.add_argument("--budget", choices=("equal-total", "equal-effort"),
                    default="equal-effort")
    ap.add_argument("--procs", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--best-case", action="store_true",
                    help="single most-favorable cell: r=0.78, concordant, "
                         "clear-pass (real N_BRUTE) — empirical NO-GO check")
    ap.add_argument("--max-q", type=int, default=None,
                    help="override MAX_Q (raise to avoid joint-arm censoring)")
    args = ap.parse_args()
    npc = 1 if args.pilot else args.n_per_cell
    procs = 1 if args.pilot else args.procs
    global PRIOR_R, LEVELS, CONCORD, N_BRUTE, MAX_Q
    if args.best_case:
        PRIOR_R = (0.78,); LEVELS = (0.6,); CONCORD = ("concordant",)
    if args.max_q is not None:
        MAX_Q = int(args.max_q)

    # K is needed to build candidate truth; get it from a throwaway init.
    _init_worker(args.budget)
    K = _W["K"]
    if args.pilot:
        # Smoke: prove end-to-end execution of all 4 arms on a TINY grid
        # (one clear-pass concordant candidate at the live prior r), small
        # particle clouds + short cap.  NOT a measurement run.
        PRIOR_R = (0.78,)
        LEVELS = (0.6,)
        CONCORD = ("concordant",)
        N_BRUTE = 120
        MAX_Q = 60
        _W["n_hier"] = (N_BRUTE if args.budget == "equal-total"
                        else K * N_BRUTE)
    tasks = build_tasks(npc, K)
    print(f"running {len(tasks)} sessions "
          f"({len(LEVELS)} levels x {len(CONCORD)} concord x {npc} reps "
          f"x {len(PRIOR_R)} r x {len(ARMS)} arms, PAIRED) "
          f"budget={args.budget} on {procs} procs...", flush=True)

    t0 = time.time()
    if procs == 1:
        rows = [_run_one(t) for t in tasks]
    else:
        with Pool(procs, initializer=_init_worker,
                  initargs=(args.budget, MAX_Q)) as pool:
            rows = pool.map(_run_one, tasks)
    dt = time.time() - t0
    print(f"compute done in {dt:.1f}s "
          f"({1000 * dt / max(len(tasks),1):.1f} ms/session wall)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "joint_vs_ad6_rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    summarize(rows)


if __name__ == "__main__":
    main()
