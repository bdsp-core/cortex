"""M21 (F72/F75) — learner-switch monitor validation + onboarding placement.

Motivated by the 2026-07-02 two-tester sandbox sessions: two different
humans ran back-to-back (89.5 s apart) on ONE belief state. The switch was
invisible to every shipped monitor (e-gate quiet, no GapAnchor under the
4 h threshold, prequential surprise flat because near-chance beliefs make
almost everything unsurprising), while windowed discrimination flipped
(d′ −0.15→+1.15 on domain3, 0.0→−0.73 on domain2) and log-RT shifted 0.6
nats. This study validates the ConsistencyMonitor (training/consistency.py)
and the F75 conservative-quantile onboarding placement.

Arms
  A  false-alarm calibration: stationary well-specified learners, fresh and
     two-session monitors → GLR percentiles ⇒ pins DEFAULT_THRESHOLD.
  B  detection power: switch at trial 40 (guesser→competent = the real
     testers' profile; the reverse; a mild σ change) → P(flag ≤ +W), latency.
  C  replay of the REAL two-tester logs (plug-in nulls from logged beliefs).
  D  pollution cost: competent learner on a fresh state vs a state polluted
     by 40 guesser trials (trainability + placement recovery).
  E  F75 placement: novice (σ0 = 2.4) under skill_sigma_z ∈ {0, 1}:
     early served accuracy, true-σ trajectory, ℓ̂ RMSE.

Run:  python3 -m studies.study_learner_switch [--smoke]
Writes figures/data_learner_switch.npz.

Harness note: single task (bank task 2 = domain3), TaskModePolicy serving
(no cert probes/e-gate — they are orthogonal to the monitor's null and cost
runtime). The sandbox interleaves K=2, giving each per-task monitor FEWER
trials per session than this harness; calibrating on more-informative
sessions makes the pinned threshold conservative.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import norm

from sandbox import config as SC
from training.bank_adapter import BankAdapter
from training.bridge_conventions import engine_to_plan
from training.consistency import ConsistencyMonitor
from training.learner_sim import Learner, LearnerParams
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import (ModeThresholds, RetentionScheduler,
                                     TaskModePolicy)

TASK = 2                       # bank index (domain3 — the task tester 2 could do)
POOL_SEED = 42
POOL_N = 4000
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figures")


def make_filter(seed):
    rng = np.random.default_rng(seed)
    th0 = SC.PRIOR_SD * rng.standard_normal(SC.N_PARTICLES)
    el0 = SC.PRIOR_SD * rng.standard_normal(SC.N_PARTICLES)
    return SigmaInfMixtureFilter(
        th0, el0, SC.assumed_params(TASK), ell_inf_mean=SC.EXPERT_ELL[TASK],
        tau=SC.MIX_TAU, J=SC.MIX_J, p_static_stratum=SC.P_STATIC_STRATUM,
        ell_star=SC.ELL_STAR[TASK], seed=seed + 7)


def make_policy(seed, z=0.0):
    mp = TaskModePolicy(0, SC.ELL_STAR[TASK], SC.SIGMA_STAR[TASK],
                        ModeThresholds(meanskill_gate=False, skill_sigma_z=z))
    return mp


def trunc_draw(rng, s_mean, s_sd, y_star):
    """Label-consistent s_real (mirrors sandbox.stimulus M21 draw)."""
    if s_sd <= 0:
        return float(s_mean)
    lo = norm.cdf((0.0 - s_mean) / s_sd)
    u = rng.uniform()
    u_t = lo + u * (1.0 - lo) if y_star == 1 else u * lo
    u_t = min(max(u_t, 1e-12), 1 - 1e-12)
    return float(s_mean + s_sd * norm.ppf(u_t))


class Harness:
    """One task, real-bank pool, serve-once, full mode policy + mixture."""

    def __init__(self, cands_all, seed, z=0.0):
        self.cands = cands_all
        self.rng = np.random.default_rng(seed)
        self.filt = make_filter(seed + 1)
        self.mp = make_policy(seed + 2, z=z)
        self.retention = RetentionScheduler(first_interval=86_400.0)
        self.monitor = ConsistencyMonitor()
        self.now = 1_800_000_000.0
        self.avail = np.ones(len(cands_all), dtype=bool)
        self.rows = []            # per-trial dicts

    def run(self, learner, n_trials, rt_mu=None, rt_sd=0.35):
        for _ in range(n_trials):
            self.now += 30.0
            mode, est = self.mp.choose_mode(self.filt, self.retention,
                                            self.now)
            cands = self.cands.subset(np.where(self.avail)[0])
            sel = self.mp.select(mode, est, cands, self.retention, self.now,
                                 self.rng, filt=self.filt)
            if sel is None:
                break
            idx, _ = sel
            s_mean = float(cands.s_mean[idx])
            s_sd = float(cands.s_sd[idx])
            y_star = int(cands.y_star[idx])
            seg = int(cands.seg_id[idx])
            self.avail[np.where(self.avail)[0][idx]] = False
            s_real = trunc_draw(self.rng, s_mean, s_sd, y_star)
            p_pred = self.filt.predictive_p(s_mean, s_sd)
            y = learner.step(s_real, 0, y_star)
            rt = (float(np.exp(rt_mu + rt_sd * self.rng.standard_normal()))
                  if rt_mu is not None else None)
            res = self.monitor.update(s_mean, s_sd, y, p_pred, rt_ms=rt)
            self.filt.step(s_mean, y, y_star, s_sd=s_sd)
            self.mp.note_posterior(self.filt)
            sig_hat, t_hat = engine_to_plan(*self.filt.mean())
            self.rows.append({
                "seg": seg, "s_mean": s_mean, "s_sd": s_sd, "y": y,
                "y_star": y_star, "correct": int(y == y_star),
                "glr": res["glr"], "rt_z": res["rt_z"],
                "flag": int(res["flag"]),
                "sig_hat": sig_hat, "t_hat": t_hat,
                "sd_l": self.filt.sd()[1],
                "train": self.filt.trainability(SC.ELL_STAR[TASK]),
                "sig_true": float(learner.sigma[0]),
                "t_true": float(learner.t[0])})
        return self.rows


def wellspec_learner(seed):
    rng = np.random.default_rng(seed)
    ell0 = SC.PRIOR_SD * rng.standard_normal()
    t0 = SC.PRIOR_SD * rng.standard_normal()
    return Learner([float(np.exp(-ell0))], [t0], SC.assumed_params(TASK),
                   seed=seed + 3)


def guesser(seed, sigma0=3.2, t0=0.2):
    p = LearnerParams(alpha_t=0.0, alpha_sigma=0.0, sigma_inf=1.0,
                      q_t=0.0, q_sigma=0.0, rule="static")
    return Learner([sigma0], [t0], p, seed=seed)


def competent(seed, sigma0=1.05, t0=-0.38):
    return Learner([sigma0], [t0], SC.assumed_params(TASK), seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    reps_a = 30 if args.smoke else 200
    reps_b = 8 if args.smoke else 40
    reps_d = 6 if args.smoke else 30
    reps_e = 4 if args.smoke else 24
    t_start = time.time()

    bank = BankAdapter()
    cands_full = bank.candidates(TASK, feedback_safe=True)
    rng = np.random.default_rng(POOL_SEED)
    keep = rng.choice(len(cands_full), size=POOL_N, replace=False)
    cands_all = cands_full.subset(keep)
    out = {}

    # ── A: false-alarm calibration ──
    print(f"[A] calibration: {reps_a} stationary two-session learners")
    max_glr_1, max_glr_2 = [], []
    for r in range(reps_a):
        h = Harness(cands_all, seed=1000 + r)
        lnr = wellspec_learner(4000 + r)
        rows = h.run(lnr, 40, rt_mu=8.6)
        max_glr_1.append(max(x["glr"] for x in rows))
        rows2 = h.run(lnr, 40, rt_mu=8.6)      # second session, same person
        max_glr_2.append(max(x["glr"] for x in rows2))
    m1, m2 = np.array(max_glr_1), np.array(max_glr_2)
    both = np.maximum(m1, m2)
    out["A_max_glr_s1"] = m1
    out["A_max_glr_s2"] = m2
    print(f"    session-1 max-GLR pct: 95={np.percentile(m1, 95):.2f} "
          f"99={np.percentile(m1, 99):.2f} max={m1.max():.2f}")
    print(f"    two-session max-GLR pct: 95={np.percentile(both, 95):.2f} "
          f"99={np.percentile(both, 99):.2f} max={both.max():.2f}")
    thr = float(np.percentile(both, 99)) if not args.smoke else 9.5
    print(f"    ⇒ pinned threshold (99th pct, two-session): {thr:.2f}")
    out["A_threshold"] = np.array([thr])

    # ── B: detection power ──
    profiles = {
        "guess->competent": (lambda s: guesser(s), lambda s: competent(s)),
        "competent->guess": (lambda s: competent(s), lambda s: guesser(s)),
        "mild sigma 1.6->1.1": (lambda s: competent(s, sigma0=1.6, t0=0.1),
                                lambda s: competent(s, sigma0=1.1, t0=0.1)),
    }
    for name, (mk_a, mk_b) in profiles.items():
        lat, hits = [], 0
        for r in range(reps_b):
            h = Harness(cands_all, seed=2000 + r)
            h.monitor.threshold = thr
            h.run(mk_a(5000 + r), 40, rt_mu=8.4)
            n0 = len(h.rows)
            rows = h.run(mk_b(6000 + r), 40, rt_mu=9.0)
            post = h.rows[n0:]
            f = next((i for i, x in enumerate(post) if x["glr"] > thr), None)
            if f is not None:
                hits += 1
                lat.append(f + 1)
        print(f"[B] {name}: P(flag ≤ +40) = {hits}/{reps_b}"
              f"  median latency = "
              f"{np.median(lat) if lat else float('nan'):.0f} trials")
        out[f"B_lat_{name.split()[0][:12]}"] = np.array(lat, dtype=float)
        out[f"B_hits_{name.split()[0][:12]}"] = np.array([hits, reps_b],
                                                         dtype=float)

    # ── C: replay of the real two-tester logs ──
    lap = 0.025
    log_path = os.path.join(os.path.dirname(FIGDIR), "sandbox", "logs",
                            "trials.jsonl")
    if os.path.exists(log_path):
        with open(log_path) as fh:
            trials = [json.loads(x) for x in fh]
        print("[C] real-log replay (plug-in null from logged beliefs):")
        for task in sorted({t["task"] for t in trials}):
            T = [t for t in trials if t["task"] == task]
            mon = ConsistencyMonitor(threshold=thr)
            first_flag, boundary = None, None
            for i, t in enumerate(T):
                if i == 0:
                    continue
                b = T[i - 1]["belief"]
                att = np.sqrt(b["sig_hat"] ** 2 + t["s_sd"] ** 2)
                pp = lap + (1 - 2 * lap) * norm.cdf(
                    (t["s_mean"] - b["t_hat"]) / att)
                res = mon.update(t["s_mean"], t["s_sd"], t["y"], pp,
                                 rt_ms=t["rt_ms"])
                if boundary is None and t["session"] == 2:
                    boundary = i
                if res["flag"] and first_flag is None:
                    first_flag = (i, t["session"], t["global_trial"],
                                  mon.glr, mon.rt_z)
            name = T[0]["task_name"]
            if first_flag:
                i, sess, g, glr, rtz = first_flag
                after = (i - boundary if boundary is not None
                         and i >= boundary else None)
                print(f"    {name}: FLAG at trial-in-task {i} (session "
                      f"{sess}, global {g}), glr={glr:.1f}, rt_z={rtz:.1f}"
                      + (f" — {after} task-trials after the tester switch"
                         if after is not None else ""))
            else:
                print(f"    {name}: no flag (final glr={mon.glr:.1f}, "
                      f"rt_z={mon.rt_z:.1f})")
    else:
        print("[C] skipped (no real logs)")

    # ── D: pollution cost (fresh vs guesser-polluted state) ──
    tr_fresh, tr_poll, acc_fresh, acc_poll = [], [], [], []
    for r in range(reps_d):
        hf = Harness(cands_all, seed=3000 + r)
        rows_f = hf.run(competent(7000 + r), 40, rt_mu=9.0)
        hp = Harness(cands_all, seed=3000 + r)
        hp.run(guesser(7500 + r), 40, rt_mu=8.4)
        rows_p = hp.run(competent(7000 + r), 40, rt_mu=9.0)[-40:]
        tr_fresh.append(rows_f[-1]["train"])
        tr_poll.append(rows_p[-1]["train"])
        acc_fresh.append(np.mean([x["correct"] for x in rows_f]))
        acc_poll.append(np.mean([x["correct"] for x in rows_p]))
    print(f"[D] competent learner after 40 trials — trainability: "
          f"fresh {np.mean(tr_fresh):.2f} vs polluted {np.mean(tr_poll):.2f};"
          f" served acc: fresh {np.mean(acc_fresh):.2f} vs polluted "
          f"{np.mean(acc_poll):.2f}")
    out["D"] = np.array([np.mean(tr_fresh), np.mean(tr_poll),
                         np.mean(acc_fresh), np.mean(acc_poll)])

    # ── E: F75 onboarding placement ──
    for z in (0.0, 1.0):
        acc40, sig_end, rmse = [], [], []
        for r in range(reps_e):
            h = Harness(cands_all, seed=8000 + r, z=z)
            lnr = competent(9000 + r, sigma0=2.4, t0=0.3)
            rows = h.run(lnr, 200, rt_mu=8.8)
            acc40.append(np.mean([x["correct"] for x in rows[:40]]))
            sig_end.append(rows[-1]["sig_true"])
            rmse.append(np.sqrt(np.mean(
                [(np.log(x["sig_hat"]) - np.log(x["sig_true"])) ** 2
                 for x in rows])))
        print(f"[E] z={z:.0f}: first-40 served acc {np.mean(acc40):.3f}  "
              f"final true σ {np.mean(sig_end):.3f}  "
              f"log-σ̂ RMSE {np.mean(rmse):.3f}")
        out[f"E_z{int(z)}"] = np.array([np.mean(acc40), np.mean(sig_end),
                                        np.mean(rmse),
                                        np.std(acc40) / np.sqrt(len(acc40)),
                                        np.std(sig_end) / np.sqrt(len(sig_end))])

    os.makedirs(FIGDIR, exist_ok=True)
    np.savez(os.path.join(FIGDIR, "data_learner_switch.npz"), **out)
    print(f"done in {time.time() - t_start:.0f}s → "
          f"figures/data_learner_switch.npz")


if __name__ == "__main__":
    main()
