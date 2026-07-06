"""M22 (F77–F79) — validation of the participant-pilot gates.

Motivated by the four-profile pilot (USER-A..D, 2026-07-02, analysis in
docs/M22_PARTICIPANT_GATES.md): three of four participants had their
WEAKEST task absorb 60–85% of trials while its trainability collapsed
(worst-first allocation conflates "far from cut" with "worth training"),
USER-D kept being served a flagged task for 10 more trials after the
consistency monitor fired, and three of four showed a late-session
accuracy collapse (0.9 → 0.4–0.5) that the engine wrote into the
persistent skill belief.

Arms
  G1  allocation: {legacy, trainability_floor=0.25} × learner pairs
      (trainable-near-cut + static-below  /  both-trainable control):
      trials to good-task declaration, trials sunk into the untrainable
      task, false graduations.
  G2  flag-pause: broken-behavior responder on one task; {pause on, off}:
      post-flag trials on the broken task, end-state damage, good-task
      progress.
  G3  fatigue guard: (a) false-alarm rate on stationary robots across
      δ grid; (b) detection on FatigueLearner (λ ramps 0.025→0.30 within
      a session); (c) REPLAY of the four real participants (trigger for
      B s2 / C s1 / D s1 expected, none for A).

Run:  python3 -m studies.study_participant_gates [--smoke]
Writes figures/data_participant_gates.npz.
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
from training.consistency import ConsistencyMonitor
from training.learner_sim import Learner, LearnerParams
from training.misspec_learners import FatigueLearner
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import (ModeThresholds, RetentionScheduler,
                                     TrainerPolicy)

TASKS = (1, 2)                   # domain2, domain3 — the sandbox scope
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figures")


class _Remap:
    def __init__(self, bank, tasks):
        self.bank, self.tasks = bank, tasks

    def candidates(self, task, **kw):
        return self.bank.candidates(self.tasks[task], **kw)


def make_filters(seed):
    rng = np.random.default_rng(seed)
    filts = {}
    for t in TASKS:
        th0 = SC.PRIOR_SD * rng.standard_normal(SC.N_PARTICLES)
        el0 = SC.PRIOR_SD * rng.standard_normal(SC.N_PARTICLES)
        filts[t] = SigmaInfMixtureFilter(
            th0, el0, SC.assumed_params(t), ell_inf_mean=SC.EXPERT_ELL[t],
            tau=SC.MIX_TAU, J=SC.MIX_J,
            p_static_stratum=SC.P_STATIC_STRATUM,
            ell_star=SC.ELL_STAR[t], seed=seed + 31 * t)
    return filts


def trunc_draw(rng, s_mean, s_sd, y_star):
    if s_sd <= 0:
        return float(s_mean)
    lo = norm.cdf((0.0 - s_mean) / s_sd)
    u = rng.uniform()
    u_t = lo + u * (1.0 - lo) if y_star == 1 else u * lo
    return float(s_mean + s_sd * norm.ppf(min(max(u_t, 1e-12), 1 - 1e-12)))


class Harness2:
    """Headless two-task sandbox-equivalent: full TrainerPolicy over the
    real bank, per-task mixtures + monitors, optional Gate-2 pause and
    Gate-3 fatigue tracking."""

    def __init__(self, bank, seed, *, floor=None, pause=False,
                 fatigue_delta=None, fatigue_w=12):
        self.rng = np.random.default_rng(seed)
        self.filters = make_filters(seed + 1)
        self.pol = TrainerPolicy(
            [self.filters[t] for t in TASKS],
            [SC.ELL_STAR[t] for t in TASKS],
            [SC.SIGMA_STAR[t] for t in TASKS],
            _Remap(bank, list(TASKS)),
            thresholds=ModeThresholds(meanskill_gate=False,
                                      skill_sigma_z=1.0),
            seed=seed + 5, probe_every=SC.PROBE_EVERY, finish_first=True,
            trainability_floor=floor, finish_sd_tol=SC.FINISH_SD_TOL)
        self.monitors = {t: ConsistencyMonitor() for t in TASKS}
        self.pause = pause
        self.fatigue_delta, self.fatigue_w = fatigue_delta, fatigue_w
        self.now = 1_800_000_000.0
        self.rows = []

    def run_session(self, respond, n_trials=40):
        """respond(task, s_real) -> y ; learners update themselves."""
        self.pol.suspended = set()
        fatigue = []
        served = 0
        while served < n_trials:
            self.now += 30.0
            ch = self.pol.step(now=self.now)
            if ch is None:
                break
            t = TASKS[ch["task"]]
            s_real = trunc_draw(self.rng, ch["s"], ch["s_sd"], ch["y_star"])
            p_pred = self.filters[t].predictive_p(ch["s"], ch["s_sd"])
            y = int(respond(t, s_real, ch["y_star"]))
            res = self.monitors[t].update(ch["s"], ch["s_sd"], y, p_pred)
            self.pol.record(ch, y)
            served += 1
            correct = int(y == ch["y_star"])
            fatigue.append((correct,
                            p_pred if ch["y_star"] == 1 else 1 - p_pred))
            self.rows.append({
                "task": t, "correct": correct, "flag": int(res["flag"]),
                "glr": res["glr"],
                "train": {tt: self.filters[tt].trainability(SC.ELL_STAR[tt])
                          for tt in TASKS},
                "mastered": {tt: bool(self.pol.mode_policies[
                    list(TASKS).index(tt)].is_mastered(self.filters[tt]))
                    for tt in TASKS}})
            if (self.pause and res["flag"]
                    and ch["task"] not in self.pol.suspended):
                self.pol.suspended.add(ch["task"])
                self.rows[-1]["paused"] = t
            if (self.fatigue_delta is not None
                    and len(fatigue) >= 2 * self.fatigue_w):
                base = fatigue[:self.fatigue_w]
                w = fatigue[-self.fatigue_w:]
                deficit = ((np.mean([p for _, p in w])
                            - np.mean([c for c, _ in w]))
                           - (np.mean([p for _, p in base])
                              - np.mean([c for c, _ in base])))
                if deficit > self.fatigue_delta:
                    self.rows[-1]["fatigue_break"] = True
                    break
        return served


def mk_trainable(task, seed, frac=1.05):
    """Trainable learner: starts frac·σ*, ceiling COMFORTABLY below the bar
    (σ_∞ = 0.72·σ* ⇒ margin/floor ≈ 2 — declarable; 0.85·σ* would be the
    F67 narrow-margin regime where declaration needs hundreds of trials
    and would confound the allocation metric)."""
    p = LearnerParams(alpha_t=0.097, alpha_sigma=0.06,
                      sigma_inf=0.72 * SC.SIGMA_STAR[task],
                      q_t=0.03, q_sigma=0.015, rho=0.6, rule="soft")
    return Learner([frac * SC.SIGMA_STAR[task]], [0.15], p, seed=seed)


def mk_static_below(task, seed, frac=1.6):
    p = LearnerParams(alpha_t=0.0, alpha_sigma=0.0, sigma_inf=1.0,
                      q_t=0.0, q_sigma=0.0, rule="static")
    return Learner([frac * SC.SIGMA_STAR[task]], [0.1], p, seed=seed)


def mk_guesser(task, seed, frac=8.0):
    """Near-chance responder (σ = 8·σ*): the PARTICIPANT regime — USER-D's
    domain3 and the M21 testers. Trainability collapses fast under this
    behavior (real data: 0.05 by ~35 trials), which is where the Gate-1
    discount actually bites; the 1.6·σ* static learner above collapses
    slowly (still ~65% accurate) and the discount is correctly gentle."""
    return mk_static_below(task, seed, frac=frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    reps1 = 6 if args.smoke else 30
    reps2 = 6 if args.smoke else 30
    reps3 = 10 if args.smoke else 100
    t0 = time.time()
    bank = BankAdapter()
    out = {}

    # ── G1: allocation ──
    for pair_name, mk2 in (("static_pair", mk_static_below),
                           ("guess_pair", mk_guesser),
                           ("both_trainable", mk_trainable)):
        for floor in (None, 0.25):
            decl, sunk, fg = [], [], 0
            for r in range(reps1):
                h = Harness2(bank, 100 + r, floor=floor)
                lnrs = {TASKS[0]: mk_trainable(TASKS[0], 500 + r),
                        TASKS[1]: mk2(TASKS[1], 700 + r)}

                def respond(t, s_real, y_star):
                    return lnrs[t].step(s_real, 0, y_star)

                first_decl, n2 = None, 0
                for sess in range(5):
                    h.run_session(respond, 40)
                for i, row in enumerate(h.rows):
                    if row["task"] == TASKS[1]:
                        n2 += 1
                    if first_decl is None and row["mastered"][TASKS[0]]:
                        first_decl = i + 1
                decl.append(first_decl if first_decl else np.nan)
                sunk.append(n2)
                if (pair_name == "static_pair"
                        and h.rows[-1]["mastered"][TASKS[1]]):
                    fg += 1
            lab = f"{pair_name}/floor={floor}"
            d = np.array(decl, dtype=float)
            n_ok = int(np.sum(~np.isnan(d)))
            print(f"[G1] {lab}: declared {n_ok}/{reps1} "
                  f"(median among declared "
                  f"{np.nanmedian(d) if n_ok else float('nan'):.0f}); "
                  f"weak-task trials {np.mean(sunk):.1f}/200; FG={fg}")
            out[f"G1_{pair_name}_{floor}"] = np.array(
                [n_ok, np.nanmedian(d) if n_ok else np.nan,
                 np.mean(sunk), fg], dtype=float)

    # ── G2: flag-pause ──
    for pause in (False, True):
        post_flag, good_train, bad_train = [], [], []
        for r in range(reps2):
            h = Harness2(bank, 200 + r, floor=0.25, pause=pause)
            good = mk_trainable(TASKS[0], 800 + r)
            rng = np.random.default_rng(900 + r)

            def respond(t, s_real, y_star):
                if t == TASKS[0]:
                    return good.step(s_real, 0, y_star)
                # broken behavior on task 2: anti-answer 85% of trials
                ideal = int(s_real > 0)
                return 1 - ideal if rng.random() < 0.85 else ideal

            h.run_session(respond, 40)
            flags = [i for i, row in enumerate(h.rows)
                     if row["flag"] and row["task"] == TASKS[1]]
            if flags:
                post = sum(1 for row in h.rows[flags[0] + 1:]
                           if row["task"] == TASKS[1])
                post_flag.append(post)
            good_train.append(h.rows[-1]["train"][TASKS[0]])
            bad_train.append(h.rows[-1]["train"][TASKS[1]])
        print(f"[G2] pause={pause}: flagged {len(post_flag)}/{reps2}; "
              f"post-flag broken-task trials mean "
              f"{np.mean(post_flag) if post_flag else float('nan'):.1f}; "
              f"end trainability good {np.mean(good_train):.2f} / "
              f"broken {np.mean(bad_train):.2f}")
        out[f"G2_pause{int(pause)}"] = np.array(
            [len(post_flag), np.mean(post_flag) if post_flag else np.nan,
             np.mean(good_train), np.mean(bad_train)])

    # ── G3a: fatigue-guard false alarms on stationary robots ──
    deltas = (0.20, 0.25, 0.30, 0.35)
    fa = {d: 0 for d in deltas}
    for d in deltas:
        n_break = 0
        for r in range(reps3):
            h = Harness2(bank, 300 + r, fatigue_delta=d)
            lnrs = {t: mk_trainable(t, 1000 + r + t, frac=1.15)
                    for t in TASKS}

            def respond(t, s_real, y_star):
                return lnrs[t].step(s_real, 0, y_star)

            h.run_session(respond, 40)
            if any(row.get("fatigue_break") for row in h.rows):
                n_break += 1
        fa[d] = n_break
        print(f"[G3a] δ={d:.2f}: false-break rate {n_break}/{reps3} "
              f"stationary sessions")
    out["G3a"] = np.array([[d, fa[d], reps3] for d in deltas])

    # ── G3b: detection on a within-session fatigue collapse ──
    for d in (0.25, 0.30):
        caught, when = 0, []
        for r in range(12 if args.smoke else 40):
            h = Harness2(bank, 400 + r, fatigue_delta=d)
            lnrs = {t: FatigueLearner([1.15 * SC.SIGMA_STAR[t]], [0.1],
                                      LearnerParams(alpha_t=0.097,
                                                    alpha_sigma=0.05,
                                                    sigma_inf=0.85 *
                                                    SC.SIGMA_STAR[t],
                                                    q_t=0.03, q_sigma=0.015,
                                                    rule="soft"),
                                      lam_max=0.30, n_ramp=25,
                                      seed=1100 + r + t)
                    for t in TASKS}

            def respond(t, s_real, y_star):
                return lnrs[t].step(s_real, 0, y_star)

            h.run_session(respond, 40)
            br = next((i for i, row in enumerate(h.rows)
                       if row.get("fatigue_break")), None)
            if br is not None:
                caught += 1
                when.append(br + 1)
        n = 12 if args.smoke else 40
        print(f"[G3b] δ={d:.2f}: fatigue collapse caught {caught}/{n}, "
              f"median at trial "
              f"{np.median(when) if when else float('nan'):.0f}")
        out[f"G3b_{d}"] = np.array([caught, n,
                                    np.median(when) if when else np.nan])

    # ── G3c: replay the four real participants ──
    lap = 0.025
    sbx_dir = os.path.join(os.path.dirname(FIGDIR), "sandbox")
    print("[G3c] real-participant replay (differenced deficit maxima):")
    for u in ("USER-A", "USER-B", "USER-C", "USER-D"):
        path = os.path.join(sbx_dir, f"logs-{u}", "trials.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            trials = [json.loads(x) for x in fh]
        for sess in sorted({t["session"] for t in trials}):
            T = [t for t in trials if t["session"] == sess]
            if len(T) < 14:
                continue
            pairs, prev_belief = [], {}
            for t in T:
                b = prev_belief.get(t["task"])
                if b is not None:
                    att = np.sqrt(b["sig_hat"] ** 2 + t["s_sd"] ** 2)
                    p1 = lap + (1 - 2 * lap) * norm.cdf(
                        (t["s_mean"] - b["t_hat"]) / att)
                    pc = p1 if t["y_star"] == 1 else 1 - p1
                    pairs.append((t["correct"], pc))
                prev_belief[t["task"]] = t["belief"]
            if len(pairs) < 24:
                continue
            base = (np.mean([p for _, p in pairs[:12]])
                    - np.mean([c for c, _ in pairs[:12]]))
            defs = [(np.mean([p for _, p in pairs[i - 12:i]])
                     - np.mean([c for c, _ in pairs[i - 12:i]])) - base
                    for i in range(24, len(pairs) + 1)]
            mx = max(defs) if defs else float("nan")
            first = next((i + 24 for i, d in enumerate(defs) if d > 0.30),
                         None)
            print(f"    {u} s{sess}: max differenced deficit {mx:+.2f}"
                  + (f" → guard(0.30) at pair {first}" if first else
                     "  (no trigger at δ=0.30)"))

    os.makedirs(FIGDIR, exist_ok=True)
    np.savez(os.path.join(FIGDIR, "data_participant_gates.npz"), **out)
    print(f"done in {time.time() - t0:.0f}s → "
          f"figures/data_participant_gates.npz")


if __name__ == "__main__":
    main()
