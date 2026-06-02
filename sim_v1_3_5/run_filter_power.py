"""sim_v1_3_5/run_filter_power.py — statistical power of each n_raters filter.

Properly-powered, PAIRED, deployment-faithful comparison of the statistical
power of every n_raters>=X bank-quality filter category.

Categories: n_raters in {1,2,3,5,8,10} (no expert filter — the sweep showed it
buys nothing). For each category we measure the test's ability to correctly
RESOLVE a candidate whose true skill is clearly above/below the cut (statistical
power), plus accuracy (false PASS/FAIL on clear cases), efficiency (session
length) and skill-estimate error.

Deployment-faithful: each session's bank is drawn by the REAL production sampler
`cortex_session_bank_fetch.sample_session_segments` (per_task=60, stratified by
s_mean quantile => ~350-seg session bank), seeded by session_id.

PAIRED across categories: a given (noise, level, rep) uses the SAME session_id
seed + engine/noise seeds for ALL six categories, so the only thing that differs
is which pool the session bank is drawn from -> McNemar paired tests between any
two categories, far more sensitive than unpaired proportions.

Two noise models: A (engine's view, no estimation noise — efficiency ceiling),
B (realistic: true difficulty = s_mean + N(0,s_sd), rater answers to true,
engine sees s_mean — the model power/accuracy is read from). AD6 ship params +
v1.3.5 consecutive cap=12.

    .venv/bin/python sim_v1_3_5/run_filter_power.py --pilot
    .venv/bin/python sim_v1_3_5/run_filter_power.py --n-per-level 220 --procs 46
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SIM_LIB = REPO / "sim_v1_2_0"
for p in (str(SCRIPTS), str(SIM_LIB)):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = REPO / "results" / "sim_v1_3_5"
SIGNALS_CSV = REPO / "data" / "labels" / "segment_signals.csv"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

TIER = ["n_expert", "n_experienced", "n_borderline", "n_novice",
        "n_other", "n_unknown", "n_untiered"]
VOTE2WORD = {"votes_seizure": "seizure", "votes_lpd": "lpd", "votes_gpd": "gpd",
             "votes_lrda": "lrda", "votes_grda": "grda", "votes_other": "other"}

CATEGORIES = (1, 2, 3, 5, 8, 10)             # n_raters >= X
NOISE = ("A", "B")
SKILL_LEVELS = (-0.5, 0.0, 0.4, 0.8, 1.2)    # below / below / borderline / above / above
CLEAR_MARGIN = 0.30
SHIP = dict(n_min=20, alpha=0.25, r_star=0.30, z=2.0)
PER_TASK = 60
N_PARTICLES, MAX_Q = 300, 300

_W = {}


def _init_worker():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for p in (str(SCRIPTS), str(SIM_LIB)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cortex_engine_inputs_k7 as ein
    import cortex_session_bank_fetch as fetch
    from cortex_policy_k7 import load_ell_star_k7
    base = ein.build_k7_engine_inputs()
    df = pd.read_csv(SIGNALS_CSV).set_index("seg_id")
    df["_iiic_n"] = df[TIER].fillna(0).sum(axis=1).astype(int)
    df["_spike_n"] = (df.get("votes_spike_yes", 0).fillna(0)
                      + df.get("votes_spike_no", 0).fillna(0)).astype(int)
    is_spike = df["s_mean_spike"].notna()
    fam = np.where(is_spike, "spike", "iiic")
    votes = df[list(VOTE2WORD)].fillna(0).to_numpy()
    plur = np.array(list(VOTE2WORD.values()))[votes.argmax(axis=1)]
    pc = np.where(is_spike, "spike", plur)
    df["_family"] = fam
    df["_pc"] = pc
    codes = list(base.task_codes)
    is_ii = df["s_mean_sz"].notna()

    # One in-memory MANIFEST per category, in the schema the deployment sampler
    # consumes (family, pattern_class, per-task stratify keys). Vectorised build.
    manifests = {}
    pool_sizes = {}
    sm_cols = {c: f"s_mean_{c}" for c in codes if f"s_mean_{c}" in df.columns}
    for thr in CATEGORIES:
        ii_ids = df.index[is_ii & (df["_iiic_n"] >= thr)]
        sp_ids = df.index[is_spike & (df["_spike_n"] >= thr)]
        pool_sizes[thr] = (int(len(ii_ids)), int(len(sp_ids)))
        segs = []
        sub = df.loc[ii_ids]
        pcv = sub["_pc"].to_numpy()
        smv = {c: sub[col].to_numpy() for c, col in sm_cols.items()}
        ids = sub.index.to_numpy()
        for i in range(len(ids)):
            e = {"seg_id": int(ids[i]), "family": "iiic", "pattern_class": pcv[i]}
            for c, arr in smv.items():
                v = arr[i]
                if v == v:  # not NaN
                    e[f"s_mean_{c}"] = float(v)
            segs.append(e)
        spv = df.loc[sp_ids, "s_mean_spike"].to_numpy()
        sids = sp_ids.to_numpy()
        for i in range(len(sids)):
            segs.append({"seg_id": int(sids[i]), "family": "spike",
                         "pattern_class": "spike", "s_mean": float(spv[i])})
        manifests[thr] = {"segments": segs}
    _W.update(base=base, df=df, ein=ein, fetch=fetch,
              ell_star=np.asarray(load_ell_star_k7(codes), float),
              codes=codes, K=len(codes), manifests=manifests, pool_sizes=pool_sizes)


def _session_inputs(seg_ids):
    df = _W["df"]; ein = _W["ein"]; base = _W["base"]; codes = _W["codes"]
    cols = [f"s_mean_{c}" for c in codes] + [f"s_sd_{c}" for c in codes]
    man = df.loc[seg_ids, cols].copy()
    man["family"] = df.loc[seg_ids, "_family"].values
    man["pattern_class"] = df.loc[seg_ids, "_pc"].values
    inp = ein.K7EngineInputs(
        manifest=man, Corr_l=base.Corr_l, Corr_t=base.Corr_t,
        task_codes=list(base.task_codes), task_labels=list(base.task_labels),
        task_pattern_words=list(base.task_pattern_words),
        task_families=list(base.task_families))
    return inp, man


def _run_one(task):
    from session_controller import CortexSession
    from cortex_policy import AD6Policy
    from core_mcmc import simulate_response
    K = _W["K"]; codes = _W["codes"]; ell = _W["ell_star"]
    fetch = _W["fetch"]; manifest = _W["manifests"][task["cat"]]

    sampled = fetch.sample_session_segments(manifest, task["sid"], per_task=PER_TASK)
    seg_ids = [int(s["seg_id"]) for s in sampled]
    inp, man = _session_inputs(seg_ids)

    true_l = np.full(K, task["level"]); true_t = np.zeros(K)
    true_sig = {}
    if task["noise"] == "B":
        rng_n = np.random.default_rng(task["seed"] + 101)
        for c in codes:
            sm = man[f"s_mean_{c}"]; sd = man[f"s_sd_{c}"]
            for sid in man.index[sm.notna()]:
                true_sig[(int(sid), c)] = float(sm[sid]) + rng_n.normal(0, float(sd[sid]))
    rng_ans = np.random.default_rng(task["seed"] + 202)

    def y_source(k, seg_id, s):
        s_true = true_sig.get((int(seg_id), codes[k]), s) if task["noise"] == "B" else s
        return simulate_response(s_true, true_t[k], true_l[k], rng_ans)

    policy = AD6Policy(ell_star=list(ell),
                       var_prior=list(np.diag(np.asarray(inp.Corr_l, float))),
                       n_min=SHIP["n_min"], R_star=SHIP["r_star"],
                       alpha=SHIP["alpha"], Z=SHIP["z"])
    sess = CortexSession(inp, session_id=task["sid"], seed=task["seed"],
                         max_questions=MAX_Q, n_particles=N_PARTICLES,
                         policy=policy, max_consecutive_same_domain=12)
    res = sess.run(y_source)

    verdicts = res.verdicts or ["PENDING"] * K
    lhat = res.final_l_mean if res.final_l_mean is not None else np.full(K, np.nan)
    fpass = fpass_n = ffail = ffail_n = 0
    skerr = []
    for k in range(K):
        d = true_l[k] - ell[k]
        if abs(d) <= CLEAR_MARGIN:
            pass
        elif d < 0:
            fpass_n += 1
            if verdicts[k] == "PASS":
                fpass += 1
        else:
            ffail_n += 1
            if verdicts[k] == "FAIL":
                ffail += 1
        if k < len(lhat) and np.isfinite(lhat[k]):
            skerr.append(abs(float(lhat[k]) - true_l[k]))
    return {
        "cat": task["cat"], "noise": task["noise"], "level": task["level"],
        "rep": task["rep"],
        "n_questions": res.n_questions, "stop_reason": res.stop_reason,
        "all_resolved": int(res.stop_reason == "all_resolved"),
        "clear": int(abs(task["level"]) > 1e-9 and abs(task["level"] - 0.4) > 1e-9),
        "false_pass": fpass, "false_pass_n": fpass_n,
        "false_fail": ffail, "false_fail_n": ffail_n,
        "skill_err": float(np.mean(skerr)) if skerr else float("nan"),
    }


def build_tasks(n_per_level, seed_base=77000):
    tasks = []
    i = 0
    for noise in NOISE:
        for lvl in SKILL_LEVELS:
            for rep in range(n_per_level):
                sid = f"{noise}_l{lvl:+.1f}_n{rep:04d}"   # category-independent => paired
                seed = seed_base + i
                for cat in CATEGORIES:
                    tasks.append({"cat": cat, "noise": noise, "level": lvl,
                                  "rep": rep, "sid": sid, "seed": seed})
                i += 1
    return tasks


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * (c - h), 100 * (c + h))


def summarize(rows):
    print(f"\n=== STATISTICAL POWER by n_raters filter ({len(rows)} sessions, "
          f"deployment sampler per_task={PER_TASK}) ===")
    print("clearRes% = resolution among CLEAR raters (true power); "
          "borderline (skill 0,0.4) excluded. CI = Wilson 95%.")
    by = {}
    for r in rows:
        by.setdefault((r["cat"], r["noise"]), []).append(r)
    for noise in NOISE:
        print(f"\n  --- Noise {noise} "
              f"({'realistic s_mean+N(0,s_sd)' if noise=='B' else 'engine view, no est. noise'}) ---")
        hdr = (f"{'n>=':>4} {'iiic':>7} {'spike':>6} {'clearRes% [95% CI]':>22} "
               f"{'fPASS%':>7} {'fFAIL%':>7} {'q_med':>6} {'skErr':>6} {'nClear':>7}")
        print(hdr); print("  " + "-" * (len(hdr)))
        for cat in CATEGORIES:
            rs = by[(cat, noise)]
            cl = [r for r in rs if r["clear"]]
            k = sum(r["all_resolved"] for r in cl); n = len(cl)
            p, lo, hi = _wilson(k, n)
            fp = sum(r["false_pass"] for r in rs); fpn = sum(r["false_pass_n"] for r in rs)
            ff = sum(r["false_fail"] for r in rs); ffn = sum(r["false_fail_n"] for r in rs)
            fpr = 100 * fp / fpn if fpn else 0.0
            ffr = 100 * ff / ffn if ffn else 0.0
            qmed = st.median(r["n_questions"] for r in cl)
            ske = st.mean(r["skill_err"] for r in cl if r["skill_err"] == r["skill_err"])
            pool_i, pool_s = _W.get("pool_sizes", {}).get(cat, (0, 0)) if _W.get("pool_sizes") else (0, 0)
            ci = f"{p:5.1f} [{lo:4.1f},{hi:4.1f}]"
            print(f"  {cat:>4} {pool_i:>7} {pool_s:>6} {ci:>22} {fpr:>6.1f}% {ffr:>6.1f}% "
                  f"{int(qmed):>6} {ske:>6.2f} {n:>7}")


def paired_tests(rows):
    """McNemar between every pair of categories on clear-rater resolution
    (matched session_id + engine seed), Model B (the realistic one)."""
    idx = {}
    for r in rows:
        if r["noise"] != "B" or not r["clear"]:
            continue
        idx.setdefault((r["level"], r["rep"]), {})[r["cat"]] = r["all_resolved"]
    print("\n=== PAIRED McNemar, clear-rater resolution, Noise B "
          "(catA vs catB; '+' = higher-n wins) ===")
    print("  pair        only_lo  only_hi  chi2_cc   verdict")
    print("  " + "-" * 52)
    for a, b in combinations(CATEGORIES, 2):     # a < b
        lo_only = hi_only = 0
        for key, d in idx.items():
            if a in d and b in d:
                if d[a] and not d[b]:
                    lo_only += 1
                elif d[b] and not d[a]:
                    hi_only += 1
        disc = lo_only + hi_only
        chi2 = ((abs(hi_only - lo_only) - 1) ** 2 / disc) if disc else 0.0
        if chi2 > 6.63:
            sig = "** p<0.01"
        elif chi2 > 3.84:
            sig = "*  p<0.05"
        else:
            sig = "   n.s."
        favor = ""
        if chi2 > 3.84:
            favor = f" favors >={b}" if hi_only > lo_only else f" favors >={a}"
        print(f"  >={a:<2} vs >={b:<2}   {lo_only:>7} {hi_only:>7} {chi2:>8.2f}   {sig}{favor}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-level", type=int, default=220)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    npl = 3 if args.pilot else args.n_per_level
    tasks = build_tasks(npl)
    print(f"running {len(tasks)} sessions ({len(CATEGORIES)} cats x {len(NOISE)} "
          f"noise x {len(SKILL_LEVELS)} levels x {npl} reps, PAIRED) on {args.procs} procs...",
          flush=True)
    t0 = time.time()
    with Pool(args.procs, initializer=_init_worker) as pool:
        rows = pool.map(_run_one, tasks)
    dt = time.time() - t0
    print(f"compute done in {dt:.1f}s ({1000*dt/len(tasks):.1f} ms/session wall)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "filter_power_rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    summarize(rows)
    paired_tests(rows)


if __name__ == "__main__":
    main()
