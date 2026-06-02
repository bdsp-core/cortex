"""sim_v1_3_5/run_bank_5v8_headtohead.py — n_raters>=5 vs >=8 bank head-to-head.

Focused, high-power follow-up to run_bank_quality_sweep.py. The sweep flagged the
>=8 power gain over >=5 as "directionally clear but soft (within ~4% sampling
noise)". This script settles it with (a) many more reps, (b) a PAIRED design, and
(c) the REAL deployment session-bank sampler.

Deployment-faithful design:
  * Each session draws its bank via the ACTUAL production sampler
    `cortex_session_bank_fetch.sample_session_segments` (per_task=60 stratified
    by s_mean quantile => ~350-seg session bank), the exact path the cloud test
    serves. NOT the old sweep's 300+50 uniform draw.
  * Two pools, built on signals only (no EEG needed for the engine math):
      bank "5": IIIC sum-of-tiers >= 5 ; spike votes >= 5
      bank "8": IIIC sum-of-tiers >= 8 ; spike votes >= 8
    (Renderability spec/shape filtering is orthogonal to signal quality, so the
    signal pools isolate the bank-QUALITY effect symmetrically.)
  * PAIRED: the session_id seed (drives the stratified draw) and the engine/noise
    seeds are identical across the two banks for a given (noise, level, rep). The
    only thing that differs is which pool the bank is drawn from -> a matched-pair
    comparison (McNemar on resolution; paired deltas on q / skill error).
  * Two noise models (A = engine's view; B = realistic s_mean+N(0,s_sd) truth),
    AD6 ship params + the v1.3.5 consecutive cap=12.

    .venv/bin/python sim_v1_3_5/run_bank_5v8_headtohead.py --pilot
    .venv/bin/python sim_v1_3_5/run_bank_5v8_headtohead.py --n-per-level 300
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
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

BANKS = (5, 8)                               # n_raters >= threshold
NOISE = ("A", "B")
SKILL_LEVELS = (-0.5, 0.0, 0.4, 0.8, 1.2)    # below / below / borderline / above / above
CLEAR_MARGIN = 0.30                          # |true_l - ell*| > this = a "clear" case
SHIP = dict(n_min=20, alpha=0.25, r_star=0.30, z=2.0)
PER_TASK = 60                                # deployment v1.2.0 internal-test value
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
    # Build the two in-memory MANIFESTS in the exact schema the deployment
    # sampler consumes: each seg carries family, pattern_class, the per-task
    # stratify keys (s_mean for spike; s_mean_<code> for IIIC).
    manifests = {}
    for thr in BANKS:
        segs = []
        iiic_ids = df.index[df["s_mean_sz"].notna() & (df["_iiic_n"] >= thr)]
        spike_ids = df.index[df["s_mean_spike"].notna() & (df["_spike_n"] >= thr)]
        for sid in iiic_ids:
            row = df.loc[sid]
            entry = {"seg_id": int(sid), "family": "iiic", "pattern_class": row["_pc"]}
            for c in codes:
                v = row.get(f"s_mean_{c}")
                if pd.notna(v):
                    entry[f"s_mean_{c}"] = float(v)
            segs.append(entry)
        for sid in spike_ids:
            row = df.loc[sid]
            segs.append({"seg_id": int(sid), "family": "spike",
                         "pattern_class": "spike", "s_mean": float(row["s_mean_spike"])})
        manifests[thr] = {"segments": segs}
    _W.update(base=base, df=df, ein=ein, fetch=fetch,
              ell_star=np.asarray(load_ell_star_k7(codes), float),
              codes=codes, K=len(codes), manifests=manifests,
              pool_sizes={thr: (int((df["s_mean_sz"].notna() & (df["_iiic_n"] >= thr)).sum()),
                                int((df["s_mean_spike"].notna() & (df["_spike_n"] >= thr)).sum()))
                          for thr in BANKS})


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
    fetch = _W["fetch"]; manifest = _W["manifests"][task["bank"]]

    # Deployment-faithful session-bank draw (paired across banks via session_id).
    sampled = fetch.sample_session_segments(manifest, task["sid"], per_task=PER_TASK)
    seg_ids = [int(s["seg_id"]) for s in sampled]
    inp, man = _session_inputs(seg_ids)
    n_iiic = sum(1 for s in sampled if s["family"] == "iiic")
    n_spike = len(sampled) - n_iiic

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
        "bank": task["bank"], "noise": task["noise"], "level": task["level"],
        "rep": task["rep"], "pair_key": f"{task['noise']}_l{task['level']:+.1f}_n{task['rep']:04d}",
        "n_iiic": n_iiic, "n_spike": n_spike,
        "n_questions": res.n_questions, "stop_reason": res.stop_reason,
        "all_resolved": int(res.stop_reason == "all_resolved"),
        "clear": int(abs(task["level"] - 0.0) > 1e-9 and abs(task["level"] - 0.4) > 1e-9),
        "is_borderline": int(abs(task["level"] - 0.4) <= 1e-9),
        "false_pass": fpass, "false_pass_n": fpass_n,
        "false_fail": ffail, "false_fail_n": ffail_n,
        "skill_err": float(np.mean(skerr)) if skerr else float("nan"),
    }


def build_tasks(n_per_level, seed_base=42000):
    """PAIRED: same (noise, level, rep) -> same session_id + engine seed for BOTH
    banks. Only the bank (pool) differs."""
    tasks = []
    i = 0
    for noise in NOISE:
        for lvl in SKILL_LEVELS:
            for rep in range(n_per_level):
                sid = f"{noise}_l{lvl:+.1f}_n{rep:04d}"     # bank-independent => paired draw
                seed = seed_base + i
                for bank in BANKS:
                    tasks.append({"bank": bank, "noise": noise, "level": lvl,
                                  "rep": rep, "sid": sid, "seed": seed})
                i += 1
    return tasks


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * (c - h), 100 * (c + h))


def _power_clear(rows):
    """Resolution rate among CLEAR raters (skill far from cut) — the power metric.
    Borderline (skill 0.4 and 0.0) excluded: REFER is the correct outcome there."""
    cl = [r for r in rows if r["clear"]]
    k = sum(r["all_resolved"] for r in cl)
    return k, len(cl)


def summarize(rows):
    print(f"\n=== n_raters>=5 vs >=8 bank head-to-head ({len(rows)} sessions, "
          f"deployment sampler per_task={PER_TASK}) ===")
    by = {}
    for r in rows:
        by.setdefault((r["bank"], r["noise"]), []).append(r)
    hdr = (f"{'bank':>5} {'noise':>5} {'clearRes%':>20} {'fPASS%':>7} {'fFAIL%':>7} "
           f"{'q_med':>6} {'skErr':>6} {'nClear':>7}")
    print(hdr); print("-" * len(hdr))
    for bank in BANKS:
        for noise in NOISE:
            rs = by[(bank, noise)]
            k, n = _power_clear(rs)
            p, lo, hi = _wilson(k, n)
            fp = sum(r["false_pass"] for r in rs); fpn = sum(r["false_pass_n"] for r in rs)
            ff = sum(r["false_fail"] for r in rs); ffn = sum(r["false_fail_n"] for r in rs)
            fpr = 100 * fp / fpn if fpn else 0.0
            ffr = 100 * ff / ffn if ffn else 0.0
            cl = [r for r in rs if r["clear"]]
            qmed = st.median(r["n_questions"] for r in cl)
            ske = st.mean(r["skill_err"] for r in cl if r["skill_err"] == r["skill_err"])
            ci = f"{p:5.1f} [{lo:4.1f},{hi:4.1f}]"
            print(f"{bank:>5} {noise:>5} {ci:>20} {fpr:>6.1f}% {ffr:>6.1f}% "
                  f"{int(qmed):>6} {ske:>6.2f} {n:>7}")


def paired_test(rows):
    """Matched-pair comparison: for each (noise, level, rep) pair, compare bank 8
    vs bank 5 on the SAME draw seed. McNemar on clear-rater resolution; paired
    deltas on session length + skill error."""
    idx = {}
    for r in rows:
        idx.setdefault((r["noise"], r["level"], r["rep"]), {})[r["bank"]] = r
    print("\n=== PAIRED test, bank 8 vs bank 5 (matched session_id + engine seed) ===")
    for noise in NOISE:
        # McNemar on clear-rater resolution
        b8_win = b5_win = 0           # discordant pairs (one resolves, other doesn't)
        dq = []; dsk = []             # paired deltas (8 minus 5), clear raters
        n_pairs = 0
        for (nz, lvl, rep), d in idx.items():
            if nz != noise or 5 not in d or 8 not in d:
                continue
            r5, r8 = d[5], d[8]
            if r5["clear"]:
                n_pairs += 1
                if r8["all_resolved"] and not r5["all_resolved"]:
                    b8_win += 1
                elif r5["all_resolved"] and not r8["all_resolved"]:
                    b5_win += 1
                dq.append(r8["n_questions"] - r5["n_questions"])
                if r8["skill_err"] == r8["skill_err"] and r5["skill_err"] == r5["skill_err"]:
                    dsk.append(r8["skill_err"] - r5["skill_err"])
        disc = b8_win + b5_win
        # McNemar exact-ish: chi2 with continuity correction
        chi2 = ((abs(b8_win - b5_win) - 1) ** 2 / disc) if disc else 0.0
        # z for the difference in resolution proportions among discordant pairs
        mean_dq = st.mean(dq) if dq else float("nan")
        mean_dsk = st.mean(dsk) if dsk else float("nan")
        print(f"\n  Noise {noise}: {n_pairs} clear pairs")
        print(f"    resolution discordant pairs: bank8-only={b8_win}  bank5-only={b5_win}  "
              f"(McNemar chi2_cc={chi2:.2f}, "
              f"{'p<0.05 favoring 8' if chi2 > 3.84 and b8_win > b5_win else 'p<0.05 favoring 5' if chi2 > 3.84 else 'n.s.'})")
        print(f"    paired delta session length (q8 - q5), clear: {mean_dq:+.2f} "
              f"(negative = bank 8 finishes faster)")
        print(f"    paired delta skill error (8 - 5), clear:      {mean_dsk:+.3f} "
              f"(negative = bank 8 more accurate)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-per-level", type=int, default=300)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    npl = 3 if args.pilot else args.n_per_level
    tasks = build_tasks(npl)
    print(f"running {len(tasks)} sessions ({len(BANKS)} banks x {len(NOISE)} noise x "
          f"{len(SKILL_LEVELS)} levels x {npl} reps, PAIRED) on {args.procs} procs...",
          flush=True)
    with Pool(args.procs, initializer=_init_worker) as pool:
        rows = pool.map(_run_one, tasks)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "bank_5v8_rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    summarize(rows)
    paired_test(rows)


if __name__ == "__main__":
    main()
