"""sim_v1_3_5/run_bank_quality_sweep.py — bank-composition quality sweep.

Question: what (n_min_raters, min_n_expert) filter maximizes the QUESTION POOL
without degrading the test's accuracy / statistical power?

Design (CPU only; runs on signals, no EEG):
  * For each filter, build the candidate POOL from segment_signals.csv
    (IIIC: sum of rater tiers >= min_raters AND n_expert >= min_expert;
     spike: spike votes >= min_raters).
  * Each synthetic session draws a FIXED-size bank (~300 IIIC + 50 spike) from
    that pool, so the FILTER varies the pool QUALITY the sample is drawn from
    while the session size is held constant.
  * Run synthetic test-takers of known true skill through full adaptive
    sessions (AD6 ship params + the v1.3.5 consecutive cap), under two noise
    models:
      A (engine's view): the rater answers to the bank's s_mean (treated as
        truth). Measures efficiency / power only.
      B (realistic): each segment's TRUE difficulty = s_mean + N(0, s_sd)
        (the bank value is a noisy estimate); the rater answers to the TRUE
        difficulty, the engine works from s_mean. Measures ACCURACY too.
  * Metrics per filter x noise: pool size, all_resolved rate, false-PASS and
    false-FAIL rates (verdict wrong vs the rater's true skill relative to the
    cut score, on CLEAR cases only), median session length, skill error.

Recommendation logic: accuracy-first — hold false-PASS/FAIL + resolution at or
above the n_raters>=5 baseline, then maximize pool size.

    .venv/bin/python sim_v1_3_5/run_bank_quality_sweep.py --pilot
    .venv/bin/python sim_v1_3_5/run_bank_quality_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
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

# Wide grid: explore BOTH directions from the n_raters>=5 baseline — looser
# (bigger pool, possibly less power) AND stricter (smaller pool, possibly MORE
# power than the baseline). min_expert>=3 forces a high-expertise consensus.
FILTERS = [(mr, me) for mr in (1, 2, 3, 5, 8, 10) for me in (0, 1, 2, 3)]
NOISE = ("A", "B")
SKILL_LEVELS = (-0.5, 0.0, 0.4, 0.8, 1.2)   # below / below / borderline / above / above
CLEAR_MARGIN = 0.30                          # |true_l - ell*| > this = a "clear" case
SHIP = dict(n_min=20, alpha=0.25, r_star=0.30, z=2.0)
N_IIIC, N_SPIKE = 300, 50                    # fixed per-session bank size
N_PARTICLES, MAX_Q = 300, 300

_W = {}


def _init_worker():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    for p in (str(SCRIPTS), str(SIM_LIB)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cortex_engine_inputs_k7 as ein
    from cortex_policy_k7 import load_ell_star_k7
    base = ein.build_k7_engine_inputs()
    df = pd.read_csv(SIGNALS_CSV).set_index("seg_id")
    df["_iiic_n"] = df[TIER].fillna(0).sum(axis=1).astype(int)
    df["_spike_n"] = (df.get("votes_spike_yes", 0).fillna(0)
                      + df.get("votes_spike_no", 0).fillna(0)).astype(int)
    df["_nexp"] = df["n_expert"].fillna(0).astype(int)
    # precompute family + pattern_class
    is_spike = df["s_mean_spike"].notna()
    fam = np.where(is_spike, "spike", "iiic")
    votes = df[list(VOTE2WORD)].fillna(0).to_numpy()
    plur = np.array(list(VOTE2WORD.values()))[votes.argmax(axis=1)]
    pc = np.where(is_spike, "spike", plur)
    df["_family"] = fam
    df["_pc"] = pc
    _W.update(base=base, df=df,
              ell_star=np.asarray(load_ell_star_k7(base.task_codes), float),
              codes=list(base.task_codes), K=len(base.task_codes),
              ein=ein)


def _pool(min_raters, min_expert):
    df = _W["df"]
    iiic = df.index[df["s_mean_sz"].notna() & (df["_iiic_n"] >= min_raters)
                    & (df["_nexp"] >= min_expert)].to_numpy()
    spike = df.index[df["s_mean_spike"].notna()
                     & (df["_spike_n"] >= min_raters)].to_numpy()
    return iiic, spike


def _session_inputs(iiic_ids, spike_ids, rng):
    df = _W["df"]; ein = _W["ein"]; base = _W["base"]; codes = _W["codes"]
    ii = rng.choice(iiic_ids, size=min(N_IIIC, len(iiic_ids)), replace=False)
    sp = (rng.choice(spike_ids, size=min(N_SPIKE, len(spike_ids)), replace=False)
          if len(spike_ids) else np.array([], int))
    segs = np.concatenate([ii, sp]).astype(int)
    cols = [f"s_mean_{c}" for c in codes] + [f"s_sd_{c}" for c in codes]
    man = df.loc[segs, cols].copy()
    man["family"] = df.loc[segs, "_family"].values
    man["pattern_class"] = df.loc[segs, "_pc"].values
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
    iiic_ids, spike_ids = _pool(task["min_raters"], task["min_expert"])
    seed = task["seed"]
    rng_bank = np.random.default_rng(seed)
    inp, man = _session_inputs(iiic_ids, spike_ids, rng_bank)
    true_l = np.full(K, task["level"]); true_t = np.zeros(K)

    # Model B: per-segment true difficulty = s_mean + N(0, s_sd)
    true_sig = {}
    if task["noise"] == "B":
        rng_n = np.random.default_rng(seed + 101)
        for c in codes:
            sm = man[f"s_mean_{c}"]; sd = man[f"s_sd_{c}"]
            for sid in man.index[sm.notna()]:
                true_sig[(int(sid), c)] = float(sm[sid]) + rng_n.normal(0, float(sd[sid]))

    rng_ans = np.random.default_rng(seed + 202)

    def y_source(k, seg_id, s):
        s_true = true_sig.get((int(seg_id), codes[k]), s) if task["noise"] == "B" else s
        return simulate_response(s_true, true_t[k], true_l[k], rng_ans)

    policy = AD6Policy(ell_star=list(ell),
                       var_prior=list(np.diag(np.asarray(inp.Corr_l, float))),
                       n_min=SHIP["n_min"], R_star=SHIP["r_star"],
                       alpha=SHIP["alpha"], Z=SHIP["z"])
    sess = CortexSession(inp, session_id=task["sid"], seed=seed,
                         max_questions=MAX_Q, n_particles=N_PARTICLES,
                         policy=policy, max_consecutive_same_domain=12)
    res = sess.run(y_source)

    verdicts = res.verdicts or ["PENDING"] * K
    lhat = res.final_l_mean if res.final_l_mean is not None else np.full(K, np.nan)
    # accuracy on CLEAR cases (per task)
    fpass = fpass_n = ffail = ffail_n = 0
    skerr = []
    for k in range(K):
        d = true_l[k] - ell[k]
        if abs(d) <= CLEAR_MARGIN:
            pass  # borderline → REFER acceptable, skip accuracy
        elif d < 0:                       # clearly BELOW cut → correct = FAIL
            fpass_n += 1
            if verdicts[k] == "PASS":
                fpass += 1
        else:                             # clearly ABOVE cut → correct = PASS
            ffail_n += 1
            if verdicts[k] == "FAIL":
                ffail += 1
        if k < len(lhat) and np.isfinite(lhat[k]):
            skerr.append(abs(float(lhat[k]) - true_l[k]))
    return {
        "min_raters": task["min_raters"], "min_expert": task["min_expert"],
        "noise": task["noise"], "level": task["level"],
        "pool_iiic": int(len(iiic_ids)), "pool_spike": int(len(spike_ids)),
        "n_questions": res.n_questions, "stop_reason": res.stop_reason,
        "all_resolved": int(res.stop_reason == "all_resolved"),
        "false_pass": fpass, "false_pass_n": fpass_n,
        "false_fail": ffail, "false_fail_n": ffail_n,
        "skill_err": float(np.mean(skerr)) if skerr else float("nan"),
    }


def build_tasks(n_per_level, seed_base=9000):
    tasks = []
    i = 0
    for (mr, me) in FILTERS:
        for noise in NOISE:
            for lvl in SKILL_LEVELS:
                for rep in range(n_per_level):
                    tasks.append({
                        "min_raters": mr, "min_expert": me, "noise": noise,
                        "level": lvl, "seed": seed_base + i,
                        "sid": f"r{mr}e{me}_{noise}_l{lvl:+.1f}_n{rep:02d}"})
                    i += 1
    return tasks


def summarize(rows):
    import statistics as st
    print("\n=== bank-composition quality sweep (AD6 ship params, "
          f"{len(rows)} sessions) ===")
    print("pool = filtered question pool size; fPASS/fFAIL = false verdicts on "
          "CLEAR raters; q = median session length")
    hdr = (f"{'minR':>4} {'minExp':>6} {'noise':>5} {'pool':>7} {'allres%':>8} "
           f"{'fPASS%':>7} {'fFAIL%':>7} {'q_med':>6} {'skErr':>6} {'n':>4}")
    print(hdr); print("-" * len(hdr))
    agg = {}
    for r in rows:
        key = (r["min_raters"], r["min_expert"], r["noise"])
        agg.setdefault(key, []).append(r)
    out = []
    for (mr, me) in FILTERS:
        for noise in NOISE:
            rs = agg.get((mr, me, noise), [])
            if not rs:
                continue
            pool = rs[0]["pool_iiic"] + rs[0]["pool_spike"]
            allres = 100.0 * sum(r["all_resolved"] for r in rs) / len(rs)
            fp = sum(r["false_pass"] for r in rs); fpn = sum(r["false_pass_n"] for r in rs)
            ff = sum(r["false_fail"] for r in rs); ffn = sum(r["false_fail_n"] for r in rs)
            fpr = 100.0 * fp / fpn if fpn else 0.0
            ffr = 100.0 * ff / ffn if ffn else 0.0
            qmed = st.median(r["n_questions"] for r in rs)
            ske = st.mean(r["skill_err"] for r in rs if r["skill_err"] == r["skill_err"])
            print(f"{mr:>4} {me:>6} {noise:>5} {pool:>7} {allres:>7.1f}% "
                  f"{fpr:>6.1f}% {ffr:>6.1f}% {int(qmed):>6} {ske:>6.2f} {len(rs):>4}")
            out.append(dict(min_raters=mr, min_expert=me, noise=noise, pool=pool,
                            allres=allres, fpr=fpr, ffr=ffr, qmed=qmed, skerr=ske))
    return out


def _power_key(s):
    # Statistical-power ordering (Model B), best first: resolve more raters,
    # make fewer clear-case errors, estimate skill tighter, finish faster.
    err = max(s["fpr"], s["ffr"])
    return (-s["allres"], err, s["skerr"], s["qmed"])


def recommend(summ):
    b = [s for s in summ if s["noise"] == "B"]
    base = next((s for s in b if s["min_raters"] == 5 and s["min_expert"] == 0), None)
    if not base:
        print("\n(no baseline cell found)"); return

    # 1) Rank EVERY filter by statistical power — does anything beat n>=5,e>=0?
    ranked = sorted(b, key=_power_key)
    print("\n=== STATISTICAL POWER ranking (Model B; best first) ===")
    print("power = resolve-rate first, then clear-case error, skill error, "
          "session length. <- baseline marks current n>=5,e>=0.")
    h = (f"{'rank':>4} {'minR':>4} {'minExp':>6} {'pool':>7} {'allres%':>8} "
         f"{'maxErr%':>7} {'skErr':>6} {'q_med':>6}")
    print(h); print("-" * len(h))
    for i, s in enumerate(ranked, 1):
        tag = "  <- baseline" if s is base else ""
        mark = "  BEATS base" if _power_key(s) < _power_key(base) and s is not base else ""
        print(f"{i:>4} {s['min_raters']:>4} {s['min_expert']:>6} {s['pool']:>7} "
              f"{s['allres']:>7.1f}% {max(s['fpr'],s['ffr']):>6.1f}% "
              f"{s['skerr']:>6.2f} {int(s['qmed']):>6}{tag}{mark}")

    best = ranked[0]
    print(f"\nHighest-power filter: n_raters>={best['min_raters']}, "
          f"n_expert>={best['min_expert']} (pool={best['pool']}, "
          f"{best['pool']/base['pool']:.2f}x baseline).")
    if best is base:
        print("  -> n_raters>=5 IS already the power optimum; nothing stricter beats it.")
    else:
        print(f"  -> beats the n_raters>=5 baseline on power "
              f"(allres {best['allres']:.1f}% vs {base['allres']:.1f}%, "
              f"maxErr {max(best['fpr'],best['ffr']):.1f}% vs "
              f"{max(base['fpr'],base['ffr']):.1f}%).")

    # 2) Accuracy-first-then-SIZE: largest pool whose power >= the BEST filter's
    #    (within tolerance). This is the deployable choice.
    tol_res, tol_err = 1.5, 1.0
    cand = [s for s in b
            if s["allres"] >= best["allres"] - tol_res
            and max(s["fpr"], s["ffr"]) <= max(best["fpr"], best["ffr"]) + tol_err]
    cand.sort(key=lambda s: -s["pool"])
    print("\n=== ACCURACY-FIRST then SIZE (largest pool holding power "
          "at the ceiling) ===")
    if cand:
        w = cand[0]
        print(f"  -> n_raters>={w['min_raters']}, n_expert>={w['min_expert']}: "
              f"pool={w['pool']} ({w['pool']/base['pool']:.2f}x baseline) "
              f"allres={w['allres']:.1f}% maxErr={max(w['fpr'],w['ffr']):.1f}% "
              f"skErr={w['skerr']:.2f}")
    else:
        print("  -> only the ceiling filter itself holds peak power.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-level", type=int, default=10)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    npl = 2 if args.pilot else args.n_per_level
    tasks = build_tasks(npl)
    print(f"running {len(tasks)} sessions ({len(FILTERS)} filters x {len(NOISE)} "
          f"noise x {len(SKILL_LEVELS)} levels x {npl} reps) on {args.procs} procs...",
          flush=True)
    with Pool(args.procs, initializer=_init_worker) as pool:
        rows = pool.map(_run_one, tasks)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "bank_quality_rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    summ = summarize(rows)
    recommend(summ)


if __name__ == "__main__":
    main()
