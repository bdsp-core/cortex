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
_ELL_STAR_JSON = None   # set in main(); per-pattern DECISION ℓ* override path
_RATER_BIAS = "zero"    # set in main(); "zero" (true θ=0) | "population" (θ~N(μ_t,Σ_t))


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
    # Optional per-pattern DECISION ℓ* override (calibration-study path; true
    # rater skill stays defined by the SHIPPED ell_star). _ELL_STAR_JSON is a
    # module global set in main() before the pool forks (inherited by workers).
    decision_ell = np.array(ell_star, dtype=float)
    if _ELL_STAR_JSON:
        ov = json.loads(Path(_ELL_STAR_JSON).read_text())
        decision_ell = np.array([float(ov[c]) for c in codes], dtype=float)
    manifest = json.loads(MANIFEST_PATH.read_text())
    # SANDBOX (Step-4 EB): load the empirical-Bayes θ-prior if present.
    eb = None
    eb_path = REPO / "sandbox_bias_prior" / "eb_prior.json"
    if eb_path.exists():
        ebj = json.loads(eb_path.read_text())
        assert list(ebj["domains"]) == codes
        eb = {"mu_t": np.asarray(ebj["mu_t"], float),
              "Sigma_t": np.asarray(ebj["Sigma_t"], float)}
    _W.update(base=base, ein=ein, fetch=fetch, codes=codes, K=len(codes),
              ell_star=ell_star, decision_ell=decision_ell, manifest=manifest,
              eb=eb)


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

    # The rater's TRUE skill is defined relative to the ORIGINAL (calibrated)
    # cut; the ell-offset lever lowers only the DECISION cut used by AD6 (the
    # §11 "recalibrate ℓ* to achievable expert performance" demonstration).
    true_l = ell + task["offset"]           # offset from each domain's ORIGINAL cut
    # Decision cut = per-pattern override (if any, from --ell-star-json) plus a
    # uniform ell-offset. Defaults to the shipped ell (override absent, offset 0).
    decision_ell = _W["decision_ell"] + task.get("ell_offset", 0.0)
    # True examinee bias θ. "zero" (shipped-OC default) = unbiased; "population"
    # draws θ ~ N(μ_t, Σ_t) — the realistic-bias scenario the EB prior assumes
    # (Step-4 sandbox: fairly tests the EB θ-prior under matched truth).
    if _RATER_BIAS == "population" and _W.get("eb") is not None:
        rng_t = np.random.default_rng(task["seed"] + 303)
        true_t = rng_t.multivariate_normal(_W["eb"]["mu_t"], _W["eb"]["Sigma_t"])
    else:
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

    policy = AD6Policy(list(decision_ell),
                       list(np.diag(np.asarray(inp.Corr_l, float))),
                       n_min=SHIP["n_min"], R_star=SHIP["R_star"],
                       alpha=task["alpha"], Z=SHIP["Z"])
    # SANDBOX (Step-4): "empirical_bayes" → full-EB θ-prior via CortexSession
    # overrides (Σ_t + μ_t); the engine's bias_prior validation stays corr_l|corr_t.
    bp = task.get("bias_prior", "corr_l")
    sig_t_ov = t_mean_ov = None
    cs_bias = bp
    if bp == "empirical_bayes":
        eb = _W["eb"]
        if eb is None:
            raise RuntimeError("empirical_bayes requested but eb_prior.json absent "
                               "(run sandbox_bias_prior/build_eb_prior.py)")
        sig_t_ov, t_mean_ov, cs_bias = eb["Sigma_t"], eb["mu_t"], "corr_l"
    sess = CortexSession(inp, session_id=task["sid"], seed=task["seed"],
                         max_questions=MAX_Q, n_particles=N_PARTICLES,
                         policy=policy, ess_threshold_frac=ESS_FRAC,
                         n_mh_steps=N_MH, max_consecutive_same_domain=12,
                         selection_objective=task.get("selection_objective",
                                                      "variance"),
                         bias_prior=cs_bias, sigma_t_override=sig_t_ov,
                         t_prior_mean=t_mean_ov)
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


def build_tasks(npc, seed_base=88000, selection_objective="variance",
                ell_offset=0.0, bias_prior="corr_l"):
    tasks = []
    i = 0
    for alpha in ALPHAS:
        for per_task in PER_TASKS:
            for offset in OFFSETS:
                for rep in range(npc):
                    tasks.append({"alpha": alpha, "per_task": per_task,
                                  "offset": offset, "rep": rep,
                                  "seed": seed_base + i,
                                  "selection_objective": selection_objective,
                                  "ell_offset": ell_offset,
                                  "bias_prior": bias_prior,
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
    # Scope flags (added for the engine-improvement plan): restrict the sweep
    # to a subset of the alpha / per_task grid to control compute. Defaults
    # reproduce the original full sweep. e.g. --alphas 0.05 --per-tasks 60
    ap.add_argument("--alphas", type=str, default=None,
                    help="comma-separated alpha values (default: full grid)")
    ap.add_argument("--per-tasks", type=str, default=None,
                    help="comma-separated per_task values (default: full grid)")
    ap.add_argument("--n-particles", type=int, default=None,
                    help="override SMC particle count N (default: 600, shipped)")
    ap.add_argument("--selection-objective", type=str, default="variance",
                    choices=("variance", "ell_variance", "decision"),
                    help="item-selection objective: 'variance' (shipped "
                         "A-optimal), 'ell_variance' (ℓ-only trace), or "
                         "'decision' (Step-3 indicator-variance)")
    ap.add_argument("--ell-offset", type=float, default=0.0,
                    help="shift ALL decision cut-scores ℓ* by this amount "
                         "(negative lowers the bar; the §11 recalibration "
                         "lever). True rater skill is unchanged.")
    ap.add_argument("--ell-star-json", type=str, default=None,
                    help="path to a per-pattern DECISION ℓ* override (JSON "
                         "{task_code: ℓ*}); true rater skill stays on the "
                         "shipped ℓ*. The calibration-study recalibration path.")
    ap.add_argument("--bias-prior", type=str, default="corr_l",
                    choices=("corr_l", "corr_t", "empirical_bayes"),
                    help="θ-block prior: 'corr_l' (shipped), 'corr_t' (fitted "
                         "bias corr), or 'empirical_bayes' (SANDBOX: Σ_t + μ_t)")
    ap.add_argument("--rater-bias", type=str, default="zero",
                    choices=("zero", "population"),
                    help="simulated examinee TRUE bias: 'zero' (shipped-OC "
                         "default) or 'population' θ~N(μ_t,Σ_t) (EB-fair test)")
    ap.add_argument("--max-q", type=int, default=None,
                    help="override MAX_Q (default 300; live production = 500)")
    args = ap.parse_args()
    global _ELL_STAR_JSON, _RATER_BIAS
    _ELL_STAR_JSON = args.ell_star_json
    _RATER_BIAS = args.rater_bias
    if not MANIFEST_PATH.exists():
        sys.exit(f"production MANIFEST not found at {MANIFEST_PATH}")
    global ALPHAS, PER_TASKS, N_PARTICLES, MAX_Q
    if args.alphas:
        ALPHAS = tuple(float(a) for a in args.alphas.split(","))
    if args.per_tasks:
        PER_TASKS = tuple(int(p) for p in args.per_tasks.split(","))
    if args.n_particles:
        N_PARTICLES = int(args.n_particles)
    if args.max_q:
        MAX_Q = int(args.max_q)
    npc = 1 if args.pilot else args.n_per_cell
    procs = 1 if args.pilot else args.procs
    tasks = build_tasks(npc, selection_objective=args.selection_objective,
                        ell_offset=args.ell_offset, bias_prior=args.bias_prior)
    print(f"running {len(tasks)} sessions ({len(ALPHAS)} alpha x {len(PER_TASKS)} "
          f"per_task x {len(OFFSETS)} offsets x {npc} reps) on {procs} procs, "
          f"N={N_PARTICLES}, objective={args.selection_objective}, "
          f"ell_offset={args.ell_offset}, bias_prior={args.bias_prior}, "
          f"rater_bias={args.rater_bias}, "
          f"ell_star_json={args.ell_star_json}...", flush=True)
    t0 = time.time()
    total = len(tasks)
    rows = []

    def _report(done):
        el = time.time() - t0
        rate = el / max(done, 1)
        eta = rate * (total - done)
        print(f"  progress {done}/{total} ({100*done/total:.0f}%)  "
              f"elapsed {el:.0f}s  eta {eta:.0f}s  "
              f"({1000*rate:.0f} ms/session avg)", flush=True)

    # ~20 progress+ETA updates over the run. imap_unordered yields results as
    # workers finish; the ETA is optimistic early (fast clear-skill sessions
    # complete before slow borderline ones) and tightens as the run proceeds.
    # Row ORDER is not preserved, but analyze()/the CSV are keyed by row fields
    # (alpha, per_task, offset), so order is irrelevant.
    step = max(1, total // 20)
    if procs == 1:
        _init_worker()
        for i, t in enumerate(tasks, 1):
            rows.append(_run_one(t))
            if i % step == 0 or i == total:
                _report(i)
    else:
        with Pool(procs, initializer=_init_worker) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_one, tasks), 1):
                rows.append(row)
                if i % step == 0 or i == total:
                    _report(i)
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
