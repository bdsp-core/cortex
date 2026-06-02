"""OC re-validation gate — does alpha=0.05 give acceptable operating
characteristics on the FROZEN 35k production bank? (v1.3.6 paper-grade config.)

Runs synthetic candidates of KNOWN true skill (offset from each domain's cut
ell*) through the live engine (CortexSession + AD6Policy), drawing the session
bank from the production MANIFEST via the real deployment sampler. Reports, per
{alpha, per_task}: per-domain P(PASS/FAIL/REFER) vs skill offset, clear-case
false-PASS/FAIL, REFER rate, session length, and the per-domain resolution at a
clear-pass candidate (the grda-oscillation check, AD6_RESOLUTION.md:133-140).

Model B (realistic): true difficulty = s_mean + N(0, s_sd); rater answers to the
true value; engine sees s_mean. CPU only, single-thread BLAS for reproducibility
(the engine is NumPy/SciPy; GPU is calibration-only and must not enter here).

    .venv/bin/python sim_v1_3_5/run_oc_validation.py --pilot
    .venv/bin/python sim_v1_3_5/run_oc_validation.py --n-per-cell 100 --procs 46
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
ENGINE = REPO / "engine"
for p in (str(SCRIPTS), str(ENGINE)):
    if p not in sys.path:
        sys.path.insert(0, p)
RESULTS = REPO / "results" / "sim_v1_3_5"
MANIFEST_PATH = REPO / "data" / "production_bank" / "MANIFEST.json"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ALPHAS = (0.05, 0.25)                 # paper-grade vs current internal
PER_TASKS = (60, 200)                 # current internal budget vs production-scale
OFFSETS = (-1.0, -0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 1.0)   # true ell - cut
CLEAR_MARGIN = 0.30
SHIP = dict(n_min=20, R_star=0.30, Z=2.0)   # alpha varies per arm
N_PARTICLES = 600                     # production (session_controller.py:72)
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
    import cortex_session_bank_fetch as fetch
    from cortex_policy_k7 import load_ell_star_k7
    base = ein.build_k7_engine_inputs()
    codes = list(base.task_codes)
    ell_star = np.asarray(load_ell_star_k7(codes), float)
    manifest = json.loads(MANIFEST_PATH.read_text())
    _W.update(base=base, ein=ein, fetch=fetch, codes=codes, K=len(codes),
              ell_star=ell_star, manifest=manifest)


def _session_inputs(segs):
    """Build a K7EngineInputs from sampled production-manifest seg dicts."""
    base = _W["base"]; ein = _W["ein"]; codes = _W["codes"]
    rows = {}
    fam, pcl = {}, {}
    for s in segs:
        sid = int(s["seg_id"])
        r = {}
        if s["family"] == "spike":
            r["s_mean_spike"] = s.get("s_mean"); r["s_sd_spike"] = s.get("s_sd")
        else:
            for c in codes:
                if c == "spike":
                    continue
                r[f"s_mean_{c}"] = s.get(f"s_mean_{c}")
                r[f"s_sd_{c}"] = s.get(f"s_sd_{c}")
        rows[sid] = r
        fam[sid] = s["family"]; pcl[sid] = s.get("pattern_class", "")
    cols = [f"s_mean_{c}" for c in codes] + [f"s_sd_{c}" for c in codes]
    man = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols)
    man["family"] = pd.Series(fam)
    man["pattern_class"] = pd.Series(pcl)
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
    K = _W["K"]; codes = _W["codes"]; ell = _W["ell_star"]; fetch = _W["fetch"]
    segs = fetch.sample_session_segments(_W["manifest"], task["sid"],
                                         per_task=task["per_task"])
    inp, man = _session_inputs(segs)

    true_l = ell + task["offset"]           # offset from each domain's cut
    true_t = np.zeros(K)
    # Model B truth: s_mean + N(0, s_sd) per (seg, code)
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
        s_true = true_sig.get((int(seg_id), codes[k]), s)
        return simulate_response(s_true, true_t[k], true_l[k], rng_ans)

    policy = AD6Policy(list(ell), list(np.diag(np.asarray(inp.Corr_l, float))),
                       n_min=SHIP["n_min"], R_star=SHIP["R_star"],
                       alpha=task["alpha"], Z=SHIP["Z"])
    sess = CortexSession(inp, session_id=task["sid"], seed=task["seed"],
                         max_questions=MAX_Q, n_particles=N_PARTICLES,
                         policy=policy, ess_threshold_frac=ESS_FRAC,
                         n_mh_steps=N_MH, max_consecutive_same_domain=12)
    res = sess.run(y_source)
    verdicts = res.verdicts or ["PENDING"] * K

    def bucket(v):
        if v == "PASS":
            return "PASS"
        if v == "FAIL":
            return "FAIL"
        return "REFER"
    row = {"alpha": task["alpha"], "per_task": task["per_task"],
           "offset": task["offset"], "seed": task["seed"],
           "n_questions": res.n_questions, "stop_reason": res.stop_reason}
    for k, c in enumerate(codes):
        row[f"v_{c}"] = bucket(verdicts[k])
    return row


def build_tasks(npc, seed_base=88000):
    tasks = []
    i = 0
    for alpha in ALPHAS:
        for per_task in PER_TASKS:
            for offset in OFFSETS:
                for rep in range(npc):
                    tasks.append({"alpha": alpha, "per_task": per_task,
                                  "offset": offset, "rep": rep,
                                  "seed": seed_base + i,
                                  "sid": f"a{alpha}_pt{per_task}_o{offset:+.2f}_n{rep:04d}"})
                    i += 1
    return tasks


def analyze(rows):
    codes = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    by = {}
    for r in rows:
        by.setdefault((r["alpha"], r["per_task"]), []).append(r)

    for (alpha, pt), rs in sorted(by.items()):
        print(f"\n================ alpha={alpha}  per_task={pt} "
              f"({len(rs)} sessions) ================")
        # OC: collapse over the 7 domains -> P(PASS/FAIL/REFER) by offset
        print("OC (all domains pooled): P(verdict) by true-skill offset; "
              "median session length; cap/exhaust%")
        print(f"  {'offset':>7} {'PASS%':>6} {'FAIL%':>6} {'REFER%':>6} "
              f"{'q_med':>6} {'capped%':>8}")
        for off in OFFSETS:
            sub = [r for r in rs if r["offset"] == off]
            if not sub:
                continue
            verds = [r[f"v_{c}"] for r in sub for c in codes]
            n = len(verds)
            pp = 100 * sum(v == "PASS" for v in verds) / n
            pf = 100 * sum(v == "FAIL" for v in verds) / n
            pr = 100 * sum(v == "REFER" for v in verds) / n
            qmed = st.median(r["n_questions"] for r in sub)
            capped = 100 * sum(r["stop_reason"] != "all_resolved" for r in sub) / len(sub)
            print(f"  {off:>+7.2f} {pp:>5.1f}% {pf:>5.1f}% {pr:>5.1f}% "
                  f"{int(qmed):>6} {capped:>7.1f}%")
        # clear-case error rates
        below = [r for r in rs if r["offset"] <= -CLEAR_MARGIN]
        above = [r for r in rs if r["offset"] >= CLEAR_MARGIN]
        fp = [r[f"v_{c}"] for r in below for c in codes]
        ff = [r[f"v_{c}"] for r in above for c in codes]
        fpr = 100 * sum(v == "PASS" for v in fp) / max(len(fp), 1)
        ffr = 100 * sum(v == "FAIL" for v in ff) / max(len(ff), 1)
        print(f"  clear-case errors: false-PASS={fpr:.2f}%  false-FAIL={ffr:.2f}%")
        # per-domain resolution at clear-pass offset +0.6 (grda check)
        cp = [r for r in rs if abs(r["offset"] - 0.6) < 1e-9]
        if cp:
            print("  per-domain @ offset +0.6 (clear-pass) — PASS% / REFER% "
                  "(grda-oscillation check):")
            line = []
            for c in codes:
                vs = [r[f"v_{c}"] for r in cp]
                line.append(f"{c}:{100*sum(v=='PASS' for v in vs)/len(vs):.0f}/"
                            f"{100*sum(v=='REFER' for v in vs)/len(vs):.0f}")
            print("    " + "  ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    if not MANIFEST_PATH.exists():
        sys.exit(f"production MANIFEST not found at {MANIFEST_PATH}")
    npc = 1 if args.pilot else args.n_per_cell
    procs = 1 if args.pilot else args.procs
    tasks = build_tasks(npc)
    print(f"running {len(tasks)} sessions ({len(ALPHAS)} alpha x {len(PER_TASKS)} "
          f"per_task x {len(OFFSETS)} offsets x {npc} reps) on {procs} procs, "
          f"N={N_PARTICLES}...", flush=True)
    t0 = time.time()
    if procs == 1:
        _init_worker()
        rows = [_run_one(t) for t in tasks]
    else:
        with Pool(procs, initializer=_init_worker) as pool:
            rows = pool.map(_run_one, tasks)
    dt = time.time() - t0
    print(f"compute done in {dt:.1f}s ({1000*dt/max(len(tasks),1):.1f} ms/session)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "oc_validation_rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    analyze(rows)


if __name__ == "__main__":
    main()
