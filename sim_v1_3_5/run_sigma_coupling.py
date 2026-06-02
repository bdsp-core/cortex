"""T2/T3/T4 — does cross-domain correlation flow into the marginal pi_k, and
does a covariance-aware readout buy anything, on the LIVE hier SMC engine?

Design (synthesis plan):
  * Paired A/B on the SAME hier engine + identical (seed, bank, answer) streams:
      ON  = live fitted base.Corr_l (off-diag mean 0.135)
      OFF = np.diag(np.diag(base.Corr_l)) = I_7 (cross-domain prior corr killed,
            marginal variances unchanged). Only the 21 off-diagonals differ.
  * CompositePolicy runs AD6 (AUTHORITATIVE, drives the stop) AND
    JointVerdictRegionPolicy (SHADOW, logged but never acted) on the identical
    post-update state -> T3 (covariance-aware would-stop) + T4 (Sigma_post
    off-diagonal trajectory) ride free on the ON+OFF sessions.
  * Metrics: T2 = paired Delta n_q (OFF-ON) + matched-early-trial Delta pi_k;
    T3 = AD6 stop trial vs shadow joint would-stop trial; T4 = post_corr_mean
    vs questions answered (ON), with OFF as the ~0 control.

    .venv/bin/python sim_v1_3_5/run_sigma_coupling.py --pilot
    .venv/bin/python sim_v1_3_5/run_sigma_coupling.py --n-per-cell 100 --procs 46
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
SCRIPTS = REPO / "scripts"
ENGINE = REPO / "engine"
for p in (str(SCRIPTS), str(ENGINE)):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = REPO / "results" / "sim_v1_3_5"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ARMS = ("ON", "OFF")
LEVELS = (-0.6, 0.0, 0.6)                # clear-fail / borderline / clear-pass
CONCORD = ("concordant", "discordant")
DISCORD_SPREAD = 0.6
CLEAR_MARGIN = 0.30
SHIP = dict(n_min=20, R_star=0.30, alpha=0.25, Z=2.0)
JOINT_BAR = 0.95
N_PARTICLES = 300
MAX_Q = 300
N_MH = 15
ESS_FRAC = 0.5

_W = {}


def _init_worker():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for p in (str(SCRIPTS), str(ENGINE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cortex_engine_inputs_k7 as ein
    from cortex_policy_k7 import load_ell_star_k7
    base = ein.build_k7_engine_inputs()
    codes = list(base.task_codes)
    K = len(codes)
    ell_star = np.asarray(load_ell_star_k7(codes), float)
    corr_on = np.asarray(base.Corr_l, float)
    corr_off = np.diag(np.diag(corr_on))           # kill cross-domain prior corr
    _W.update(base=base, ein=ein, codes=codes, K=K, ell_star=ell_star,
              corr_on=corr_on, corr_off=corr_off)


def _true_skill(level, concord, K, seed):
    rng = np.random.default_rng(seed + 9001)
    if concord == "concordant":
        return np.full(K, level)
    return level + rng.normal(0.0, DISCORD_SPREAD, size=K)


class CompositePolicy:
    """AD6 (authoritative) + JointVerdictRegionPolicy (shadow). Logs one row per
    trial. Only AD6's StopDecision drives the session loop."""

    def __init__(self, ell_star, var_prior, meta):
        from cortex_policy import AD6Policy
        from cortex_policy_joint import JointVerdictRegionPolicy
        self.ad6 = AD6Policy(list(ell_star), list(var_prior),
                             n_min=SHIP["n_min"], R_star=SHIP["R_star"],
                             alpha=SHIP["alpha"], Z=SHIP["Z"])
        self.joint = JointVerdictRegionPolicy(
            list(ell_star), list(var_prior), n_min=SHIP["n_min"],
            R_star=SHIP["R_star"], Z=SHIP["Z"], bar=JOINT_BAR)
        self.ell_star = np.asarray(ell_star, float)
        self.meta = meta
        self.rows = []
        self._t1_max_dev = 0.0          # T1 in-session drift guard

    def reset(self, K):
        self.ad6.reset(K)
        self.joint.reset(K)

    def __call__(self, state, tel, n_per_task, K):
        dec = self.ad6(state, tel, n_per_task, K)            # AUTHORITATIVE
        sh = self.joint(state, tel, n_per_task, K)           # SHADOW (discarded)
        da, ds = dec.diagnostics, sh.diagnostics
        # T1 in-session drift guard: AD6's pi_k must equal an independent
        # per-column recompute (proves no cross-column term in the live engine).
        w = state["w"]; wn = w / w.sum(); L = state["l"]
        pi_indep = [float((wn * (L[:, k] > self.ell_star[k])).sum())
                    for k in range(K)]
        self._t1_max_dev = max(self._t1_max_dev,
                               float(np.max(np.abs(np.asarray(da["pi"]) - pi_indep))))
        row = {**self.meta, "trial": tel["trial_index"], "task_k": tel["task_k"],
               "seg_id": tel["seg_id"], "post_corr": float(ds["post_corr_mean"]),
               "ad6_stop": int(dec.stop), "joint_lcb": float(ds["joint_lcb"]),
               "joint_stop": int(sh.stop),
               "n_resolved": int(sum(1 for v in da["verdicts"]
                                     if v in ("PASS", "FAIL")))}
        for k in range(K):
            row[f"pi{k}"] = float(da["pi"][k])
        self.rows.append(row)
        return dec

    def finalize_verdicts(self):
        return self.ad6.finalize_verdicts()


def _run_one(task):
    from session_controller import CortexSession
    from core_mcmc import simulate_response
    base = _W["base"]; ein = _W["ein"]; K = _W["K"]; ell_star = _W["ell_star"]
    arm = task["arm"]
    Corr = _W["corr_on"] if arm == "ON" else _W["corr_off"]
    var_prior = list(np.diag(Corr))                  # = ones(K) for both arms
    inp = ein.K7EngineInputs(
        manifest=base.manifest, Corr_l=Corr, Corr_t=base.Corr_t,
        task_codes=list(base.task_codes), task_labels=list(base.task_labels),
        task_pattern_words=list(base.task_pattern_words),
        task_families=list(base.task_families))
    true_l = _true_skill(task["level"], task["concord"], K, task["seed"])
    true_t = np.zeros(K)
    meta = {"arm": arm, "level": task["level"], "concord": task["concord"],
            "seed": task["seed"]}
    policy = CompositePolicy(ell_star, var_prior, meta)
    rng_ans = np.random.default_rng(task["seed"] + 202)

    def y_source(k, seg_id, s):
        return simulate_response(s, true_t[k], true_l[k], rng_ans)

    sess = CortexSession(inp, session_id=task["sid"], seed=task["seed"],
                         max_questions=MAX_Q, n_particles=N_PARTICLES,
                         policy=policy, ess_threshold_frac=ESS_FRAC,
                         n_mh_steps=N_MH, max_consecutive_same_domain=12)
    res = sess.run(y_source)
    verdicts = res.verdicts or ["PENDING"] * K
    rows = policy.rows
    n_joint = next((r["trial"] + 1 for r in rows if r["joint_stop"]), None)
    summary = {"arm": arm, "level": task["level"], "concord": task["concord"],
               "seed": task["seed"], "n_ad6": res.n_questions,
               "stop_reason": res.stop_reason, "n_joint_shadow": n_joint,
               "terminal_post_corr": rows[-1]["post_corr"] if rows else float("nan"),
               "t1_max_dev": policy._t1_max_dev,
               "verdicts": "|".join(verdicts), "true_l": "|".join(f"{x:.3f}" for x in true_l)}
    return summary, rows


def build_tasks(npc, seed_base=55000):
    tasks = []
    i = 0
    for level in LEVELS:
        for concord in CONCORD:
            for rep in range(npc):
                seed = seed_base + i
                sid = f"l{level:+.1f}_{concord[:4]}_n{rep:04d}"
                for arm in ARMS:                       # paired: same seed/sid
                    tasks.append({"arm": arm, "level": level, "concord": concord,
                                  "rep": rep, "seed": seed, "sid": sid})
                i += 1
    return tasks


def _wilcoxon(deltas):
    from scipy.stats import wilcoxon
    d = [x for x in deltas if x == x]
    nz = [x for x in d if x != 0]
    if len(nz) < 5:
        return (float("nan"), float("nan"))
    try:
        s, p = wilcoxon(nz)
        return (float(st.median(d)), float(p))
    except Exception:
        return (float(st.median(d)), float("nan"))


def analyze(summaries, trials):
    K = 7
    print("\n================ ANALYSIS ================")
    # index summaries by (level, concord, seed, arm)
    sidx = {}
    for s in summaries:
        sidx[(s["level"], s["concord"], s["seed"], s["arm"])] = s
    # group trials by (arm, level, concord, seed) -> ordered list
    tidx = {}
    for r in trials:
        tidx.setdefault((r["arm"], r["level"], r["concord"], r["seed"]), []).append(r)
    for v in tidx.values():
        v.sort(key=lambda r: r["trial"])

    # ---- T1 in-session drift guard (max |pi_AD6 - pi_indep| over ALL trials) ----
    t1 = max((s["t1_max_dev"] for s in summaries), default=0.0)
    print(f"\n[T1] live-engine drift guard: max|pi_AD6 - pi_indep| over all "
          f"trials = {t1:.2e}  -> {'PASS (marginal-only)' if t1 < 1e-9 else 'LEAK'}")

    # ---- T2: paired Delta n_q (OFF - ON) and matched-early-trial Delta pi ----
    print("\n[T2] correlation ON vs OFF — does killing cross-domain prior corr "
          "change behavior?")
    print(f"  {'cell':>22} {'med n_ON':>9} {'med n_OFF':>9} {'med Δn(OFF-ON)':>15} "
          f"{'wilcoxon p':>11} {'verdict-agree%':>14} {'medTmatch':>10} {'med max|Δπ|@match':>17}")
    for level in LEVELS:
        for concord in CONCORD:
            keys = [(level, concord, s["seed"]) for s in summaries
                    if s["level"] == level and s["concord"] == concord and s["arm"] == "ON"]
            seeds = sorted({k[2] for k in keys})
            dn, agree, tmatch, dpimatch = [], [], [], []
            non_on, non_off = [], []
            for sd in seeds:
                on = sidx.get((level, concord, sd, "ON"))
                off = sidx.get((level, concord, sd, "OFF"))
                if not on or not off:
                    continue
                non_on.append(on["n_ad6"]); non_off.append(off["n_ad6"])
                dn.append(off["n_ad6"] - on["n_ad6"])
                agree.append(1.0 if on["verdicts"] == off["verdicts"] else 0.0)
                # matched early trials: walk until seg_id diverges
                ron = tidx.get(("ON", level, concord, sd), [])
                roff = tidx.get(("OFF", level, concord, sd), [])
                m = 0; mx = 0.0
                for a, b in zip(ron, roff):
                    if a["seg_id"] != b["seg_id"]:
                        break
                    m += 1
                    mx = max(mx, max(abs(a[f"pi{k}"] - b[f"pi{k}"]) for k in range(K)))
                tmatch.append(m); dpimatch.append(mx)
            cell = f"{level:+.1f}/{concord[:4]}"
            mdn, p = _wilcoxon(dn)
            print(f"  {cell:>22} {int(st.median(non_on)):>9} {int(st.median(non_off)):>9} "
                  f"{mdn:>15.1f} {p:>11.3g} {100*st.mean(agree):>13.1f}% "
                  f"{int(st.median(tmatch)):>10} {st.median(dpimatch):>17.4f}")

    # ---- T3: shadow joint would-stop vs AD6 stop (ON arm) ----
    print("\n[T3] covariance-aware shadow vs marginal AD6 (ON arm) — what the "
          "marginal leaves on the table")
    on_sum = [s for s in summaries if s["arm"] == "ON"]
    fired = [s for s in on_sum if s["n_joint_shadow"] is not None]
    print(f"  joint shadow would-stop fired in {len(fired)}/{len(on_sum)} "
          f"({100*len(fired)/max(len(on_sum),1):.1f}%) of ON sessions")
    if fired:
        dstop = [s["n_ad6"] - s["n_joint_shadow"] for s in fired]
        earlier = sum(1 for d in dstop if d > 0)
        print(f"  among those: median Δstop (n_AD6 - n_joint) = {st.median(dstop):+.1f} "
              f"questions; joint earlier in {earlier}/{len(fired)}")

    # ---- T4: Sigma_post off-diagonal trajectory (ON) + OFF control ----
    print("\n[T4] posterior cross-domain correlation (post_corr_mean) vs questions")
    bins = [(1, 10), (11, 30), (31, 60), (61, 120), (121, 300)]
    print(f"  {'q-range':>10} {'ON median post_corr':>22} {'OFF median (control~0)':>24}")
    for lo, hi in bins:
        on_v = [r["post_corr"] for r in trials
                if r["arm"] == "ON" and lo <= r["trial"] + 1 <= hi]
        off_v = [r["post_corr"] for r in trials
                 if r["arm"] == "OFF" and lo <= r["trial"] + 1 <= hi]
        on_m = st.median(on_v) if on_v else float("nan")
        off_m = st.median(off_v) if off_v else float("nan")
        print(f"  {f'{lo}-{hi}':>10} {on_m:>22.4f} {off_m:>24.4f}")
    on_term = [s["terminal_post_corr"] for s in summaries if s["arm"] == "ON"]
    print(f"  ON terminal post_corr (at stop): median "
          f"{st.median([x for x in on_term if x==x]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    npc = 2 if args.pilot else args.n_per_cell
    procs = 1 if args.pilot else args.procs
    tasks = build_tasks(npc)
    print(f"running {len(tasks)} sessions ({len(LEVELS)} levels x {len(CONCORD)} "
          f"concord x {npc} reps x {len(ARMS)} arms, PAIRED) on {procs} procs...",
          flush=True)
    t0 = time.time()
    if procs == 1:
        _init_worker()
        out = [_run_one(t) for t in tasks]
    else:
        with Pool(procs, initializer=_init_worker) as pool:
            out = pool.map(_run_one, tasks)
    dt = time.time() - t0
    print(f"compute done in {dt:.1f}s ({1000*dt/max(len(tasks),1):.1f} ms/session)")
    summaries = [s for s, _ in out]
    trials = [r for _, rows in out for r in rows]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "sigma_coupling_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    print(f"wrote {RESULTS / 'sigma_coupling_summary.csv'} ({len(summaries)} rows, "
          f"{len(trials)} trial rows)")
    analyze(summaries, trials)


if __name__ == "__main__":
    main()
