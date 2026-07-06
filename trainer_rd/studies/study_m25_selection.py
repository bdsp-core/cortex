"""M25 — selection successors study (F86/F87): EP-v2 lifecycle-coupled
allocation and item-level tier-3 rollout placement, at the M23 shipping
config (Gate 4 + sd_floor 0.33) over the real bank.

The two queued M24 successors (docs/M24_SELECTION.md §5):

  F86 EP-v2 ALLOCATION (arm Q2): `progress_lifecycle` routes ever-declared
     tasks to the finish-first measurement lane (refinish_threshold) so
     the F85-ii post-boundary RE-POLISH loop cannot starve a recovering
     task, plus the `explore_every` exploration floor. Endpoint:
     BOTH-DECLARED (the F85-ii censoring metric), per-task firsts, sunk
     trials, post-declaration service split. Arms {gate1, ep(v1),
     epv2nf (lifecycle only), epv2 (lifecycle + floor)} × the three M24
     rosters. b-trap keeps the sunk/starvation reading (task1 is a true
     plateau — both-declared does not apply).

  F87 ROLLOUT PLACEMENT (arm R): `rollout_placement` — the non-myopic
     F84 successor. H-step CRN Monte-Carlo rollout value at the ITEM
     level (mixture-honest dynamics: per-stratum ceilings, static mass
     frozen — F84-ii), inside the F84-v6 safety devices (mirror window,
     finishing switch). Scenarios {wellspec, jump, contact, slow} ×
     rules {z1 (shipped), prog (F84 reference), roll}; herding arm H2.

  GUARDRAILS (arm S2): FG zoo (static_below ℓ*−0.15, careless) +
     well-spec 2-task lateness for {m23 shipped, epv2 shipping candidate,
     epv2+roll experimental}.

Run:  python3 -m studies.study_m25_selection [--smoke] [--arms q2,r,s2]
Out:  figures/data_m25_selection.npz  (arm keys merged across runs)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import norm

from sandbox import config as C
from training.bank_adapter import BankAdapter
from training.learner_sim import Learner, LearnerParams
from training.trainer_policy import ModeThresholds, TrainerPolicy
from studies.study_m24_selection import (BANK_S_SD, LAPSE, MULT, RHO, _Remap,
                                         first_declared, fresh_mixture,
                                         run_sessions)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(FIG, "data_m25_selection.npz")

EXPLORE_EVERY = 8
REFINISH = 0.50


def make_policy(bank, tasks, seed, *, rule, alloc="gate1"):
    """rule: 'z0'|'z1'|'prog'|'roll'; alloc: 'legacy'|'gate1'|'ep'|
    'epv2nf' (lifecycle only) | 'epv2' (lifecycle + exploration floor)."""
    filters = [fresh_mixture(t, seed + 1 + 31 * i)
               for i, t in enumerate(tasks)]
    th = ModeThresholds(meanskill_gate=False,
                        sd_floor=C.SD_FLOOR,
                        skill_sigma_z=(0.0 if rule in ("z0", "prog", "roll")
                                       else 1.0),
                        progress_placement=(rule == "prog"),
                        rollout_placement=(rule == "roll"))
    pol = TrainerPolicy(
        filters, [C.ELL_STAR[t] for t in tasks],
        [C.SIGMA_STAR[t] for t in tasks], _Remap(bank, list(tasks)),
        thresholds=th, seed=seed + 5, probe_every=C.PROBE_EVERY,
        finish_first=True,
        trainability_floor=(C.TRAINABILITY_FLOOR if alloc == "gate1"
                            else None),
        finish_sd_tol=C.FINISH_SD_TOL,
        progress_alloc=alloc in ("ep", "epv2", "epv2nf"),
        progress_s_sd=BANK_S_SD,
        progress_lifecycle=alloc in ("epv2", "epv2nf"),
        refinish_threshold=REFINISH,
        explore_every=(EXPLORE_EVERY if alloc == "epv2" else 0))
    return pol, filters


# ── the three M24 rosters (arm Q constructors, verbatim physics) ──
def roster_btrap(tasks, seed):
    """task0 trainable near-bar; task1 static plateau 1.6σ* (USER-B)."""
    p0 = LearnerParams(alpha_t=0.097, alpha_sigma=0.06,
                       sigma_inf=0.72 * C.SIGMA_STAR[tasks[0]],
                       q_t=0.03, q_sigma=0.015, rho=0.6, rule="soft")
    l0 = Learner([1.05 * C.SIGMA_STAR[tasks[0]]], [0.15], p0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    sig1 = 1.6 * C.SIGMA_STAR[tasks[1]]

    def respond(k, s):
        if k == 0:
            return l0.step(s, 0, y_star=int(s > 0), feedback=True)
        pr = LAPSE + (1 - 2 * LAPSE) * norm.cdf((s - 0.1) / sig1)
        return int(rng.random() < pr)
    return respond


def roster_rates(tasks, seed):
    ls = []
    for i, t in enumerate(tasks):
        p = LearnerParams(alpha_t=0.097,
                          alpha_sigma=(0.12 if i == 0 else 0.03),
                          sigma_inf=0.72 * C.SIGMA_STAR[t],
                          q_t=0.03, q_sigma=0.015, rho=0.6, rule="soft")
        ls.append(Learner([1.3], [0.1], p, seed=seed + i))
    return lambda k, s: ls[k].step(s, 0, y_star=int(s > 0), feedback=True)


def roster_jumper(tasks, seed):
    p0 = LearnerParams(alpha_t=0.097, alpha_sigma=0.06,
                       sigma_inf=0.72 * C.SIGMA_STAR[tasks[0]],
                       q_t=0.03, q_sigma=0.015, rho=0.6, rule="soft")
    l0 = Learner([1.2], [0.1], p0, seed=seed)
    st = {"n": 0}
    rng = np.random.default_rng(seed + 1)

    def respond(k, s):
        if k == 0:
            return l0.step(s, 0, y_star=int(s > 0), feedback=True)
        st["n"] += 1
        sig = 3.5 if st["n"] <= 40 else 0.55
        pr = LAPSE + (1 - 2 * LAPSE) * norm.cdf((s - 0.05) / sig)
        return int(rng.random() < pr)
    return respond


def both_declared(rows):
    """First trial index at which EACH task has (ever) declared — the
    F85-ii completion metric (max of per-task first declarations)."""
    a, b = first_declared(rows, 0), first_declared(rows, 1)
    if a is None or b is None:
        return None
    return max(a, b)


def post_decl_service(rows, served_to=1):
    """Trials/session served to `served_to` AFTER task0 first declares —
    the F85-ii starvation metric (~12/session under EP-v1)."""
    a = first_declared(rows, 0)
    if a is None:
        return float("nan")
    tail = rows[a:]
    n_sess = len(set(r["session"] for r in tail)) or 1
    return sum(1 for r in tail if r["k"] == served_to) / n_sess


# ── arm Q2: EP-v2 allocation, BOTH-declared endpoint ──
def arm_Q2(bank, n_seeds, smoke):
    print("\n== ARM Q2: EP-v2 allocation (K=2, BOTH-declared endpoint) ==")
    tasks = [1, 2]
    n_sess = 8 if not smoke else 3
    cap = n_sess * 40
    rosters = {"b-trap": roster_btrap, "rates": roster_rates,
               "jumper": roster_jumper}
    out = {}
    for rname, mk in rosters.items():
        for alloc in ("gate1", "ep", "epv2nf", "epv2"):
            d0, d1, db, sunk, svc = [], [], [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(500 + sd)
                pol, filters = make_policy(bank, tasks, 600 + 37 * sd,
                                           rule="z1", alloc=alloc)
                rows = run_sessions(pol, filters, tasks,
                                    mk(tasks, 7000 + sd), n_sess, rng)
                a = first_declared(rows, 0)
                b = first_declared(rows, 1)
                d0.append(a if a is not None else cap + 1)
                d1.append(b if b is not None else cap + 1)
                bd = both_declared(rows)
                db.append(bd if bd is not None else cap + 1)
                sunk.append(sum(1 for r in rows if r["k"] == 1))
                svc.append(post_decl_service(rows))
            out[(rname, alloc)] = dict(
                d0=float(np.median(d0)), d1=float(np.median(d1)),
                dboth=float(np.median(db)),
                rboth=float(np.mean([d <= cap for d in db])),
                r0=float(np.mean([d <= cap for d in d0])),
                sunk=float(np.mean(sunk)),
                svc=float(np.nanmean(svc)))
            print(f"  {rname:<7} [{alloc:>6}]: task0 {np.median(d0):5.0f} "
                  f"task1 {np.median(d1):5.0f}  BOTH "
                  f"{np.median(db):5.0f} (rate "
                  f"{np.mean([d <= cap for d in db]):.0%})  "
                  f"task1-trials {np.mean(sunk):5.1f}  "
                  f"post-decl svc/sess {np.nanmean(svc):4.1f}")
    return out


# ── arm R: item-level rollout placement (K=1, arm-P scenarios) ──
def arm_R(bank, n_seeds, smoke):
    print("\n== ARM R: rollout placement OC (K=1, real bank, M23 stack) ==")
    task = 1
    n_sess = 6 if not smoke else 3
    cap = n_sess * 40

    def scen_wellspec(seed):
        lnr = Learner([1.0], [0.1], C.assumed_params(task), seed=seed)
        return (lambda k, s: lnr.step(s, 0, y_star=int(s > 0), feedback=True),
                lambda k: (float(lnr.sigma[0]), float(lnr.t[0])))

    def scen_jump(seed):
        st = {"n": 0}
        rng = np.random.default_rng(seed)

        def respond(k, s):
            st["n"] += 1
            sig = 3.5 if st["n"] <= 80 else 0.55
            p = LAPSE + (1 - 2 * LAPSE) * norm.cdf((s - 0.05) / sig)
            return int(rng.random() < p)
        return respond, lambda k: (3.5 if st["n"] <= 80 else 0.55, 0.05)

    def scen_contact(seed):
        st = {"n": 0}
        rng = np.random.default_rng(seed)

        def respond(k, s):
            st["n"] += 1
            sig = 5.0 if st["n"] <= 60 else 0.9
            p = LAPSE + (1 - 2 * LAPSE) * norm.cdf((s - 0.0) / sig)
            return int(rng.random() < p)
        return respond, lambda k: (5.0 if st["n"] <= 60 else 0.9, 0.0)

    def scen_slow(seed):
        p = LearnerParams(alpha_t=0.097, alpha_sigma=0.03,
                          sigma_inf=0.62, q_t=0.03, q_sigma=0.015,
                          rho=0.6, rule="soft")
        lnr = Learner([1.6], [0.2], p, seed=seed)
        return (lambda k, s: lnr.step(s, 0, y_star=int(s > 0), feedback=True),
                lambda k: (float(lnr.sigma[0]), float(lnr.t[0])))

    scens = {"wellspec": scen_wellspec, "jump": scen_jump,
             "contact": scen_contact, "slow": scen_slow}
    out = {}
    for sname, mk in scens.items():
        for rule in ("z1", "roll"):
            decl, wtr, acc = [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(100 + sd)
                pol, filters = make_policy(bank, [task], 200 + 37 * sd,
                                           rule=rule)
                respond, truth = mk(9000 + sd)
                rows = run_sessions(pol, filters, [task], respond, n_sess,
                                    rng, record_truth=truth)
                d = first_declared(rows, 0)
                decl.append(d if d is not None else len(rows) + 1)
                sk = [r for r in rows if r["mode"] == "skill"]
                wtr.append(np.mean([r["w_true"] for r in sk]) if sk else 0.0)
                acc.append(np.mean([r["correct"] for r in rows]))
            out[(sname, rule)] = dict(
                decl_med=float(np.median(decl)),
                decl_rate=float(np.mean([d <= cap for d in decl])),
                w_true=float(np.mean(wtr)), acc=float(np.mean(acc)))
            print(f"  {sname:<9} [{rule:>4}]: declared "
                  f"{np.median(decl):5.0f} (rate "
                  f"{np.mean([d <= cap for d in decl]):.0%})  "
                  f"E[w_true|skill] {np.mean(wtr):.2f}  "
                  f"acc {np.mean(acc):.2f}")
    # H2: criterion-herding safety under free-er rollout placement
    print("  -- H2: herding safety (live R-W criterion) --")
    n_h = max(3, n_seeds // 2)
    for rule in ("z1", "roll"):
        t_end, t_err = [], []
        for sd in range(n_h):
            rng = np.random.default_rng(300 + sd)
            p = LearnerParams(alpha_t=0.2, alpha_sigma=0.04,
                              sigma_inf=0.60, q_t=0.04, q_sigma=0.02,
                              rho=0.6, rule="soft")
            lnr = Learner([1.2], [0.5], p, seed=4000 + sd)
            pol, filters = make_policy(bank, [task], 400 + 37 * sd,
                                       rule=rule)
            rows = run_sessions(
                pol, filters, [task],
                lambda k, s: lnr.step(s, 0, y_star=int(s > 0), feedback=True),
                8 if not smoke else 3, rng,
                record_truth=lambda k: (float(lnr.sigma[0]),
                                        float(lnr.t[0])))
            ts = np.array([r["t_true"] for r in rows])
            t_end.append(abs(ts[-10:]).mean())
            from training.bridge_conventions import engine_to_plan
            _, t_hat = engine_to_plan(*filters[0].mean())
            t_err.append(abs(t_hat - ts[-1]))
        out[("herding", rule)] = dict(t_end=float(np.mean(t_end)),
                                      t_err=float(np.mean(t_err)))
        print(f"  [{rule:>4}]: |t_true| end {np.mean(t_end):.3f}  "
              f"|t_hat-t| end {np.mean(t_err):.3f}")
    return out


# ── arm S2: guardrails at the candidate shipping configs ──
def arm_S2(bank, n_seeds, smoke):
    print("\n== ARM S2: guardrails ==")
    tasks = [1, 2]
    n_sess = 6 if not smoke else 3
    sig_below = {i: float(np.exp(-(C.ELL_STAR[t] - 0.15)))
                 for i, t in enumerate(tasks)}
    ARMS = {"m23": ("z1", "gate1"),
            "epv2": ("z1", "epv2"),           # shipping candidate
            "epv2roll": ("roll", "epv2")}     # experimental
    out = {}
    for zoo in ("static_below", "careless"):
        for arm, (rule_, alloc_) in ARMS.items():
            fg = 0
            for sd in range(n_seeds):
                rng = np.random.default_rng(800 + sd)
                pol, filters = make_policy(
                    bank, tasks, 900 + 37 * sd, rule=rule_, alloc=alloc_)
                rz = np.random.default_rng(1800 + sd)

                def respond(k, s):
                    if zoo == "static_below":
                        pr = LAPSE + (1 - 2 * LAPSE) * norm.cdf(
                            (s - 0.1) / sig_below[k])
                    else:
                        pr = 0.35 + 0.30 * norm.cdf(
                            (s - 0.1) / float(C.SIGMA_STAR[tasks[k]]))
                    return int(rz.random() < pr)
                rows = run_sessions(pol, filters, tasks, respond, n_sess,
                                    rng)
                fg += int(any(any(r["mastered"]) for r in rows))
            out[(zoo, arm)] = fg
            print(f"  {zoo:<13} [{arm:>8}]: FG {fg}/{n_seeds}")
    for arm, (rule_, alloc_) in ARMS.items():
        firsts = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(1100 + sd)
            pol, filters = make_policy(
                bank, tasks, 1200 + 37 * sd, rule=rule_, alloc=alloc_)
            lnrs = [Learner([1.0], [0.1], C.assumed_params(t),
                            seed=2200 + sd + i)
                    for i, t in enumerate(tasks)]
            rows = run_sessions(
                pol, filters, tasks,
                lambda k, s: lnrs[k].step(s, 0, y_star=int(s > 0),
                                          feedback=True),
                8 if not smoke else 4, rng)
            f = next((i for i, r in enumerate(rows) if any(r["mastered"])),
                     None)
            firsts.append(f if f is not None else len(rows) + 1)
        out[("lateness", arm)] = float(np.median(firsts))
        print(f"  lateness (well-spec pair) [{arm:>8}]: first declaration "
              f"median {np.median(firsts):.0f}, rate "
              f"{np.mean([f <= 320 for f in firsts]):.0%}")
    return out


def _merge_save(new_arms):
    """Merge arm results into the npz (arms run separately share one file)."""
    os.makedirs(FIG, exist_ok=True)
    old = {}
    if os.path.exists(OUT):
        with np.load(OUT, allow_pickle=False) as z:
            old = {k: str(z[k][0]) for k in z.files}
    for name, res in new_arms.items():
        enc = {f"{a}|{b}": v for (a, b), v in res.items()} \
            if res and isinstance(next(iter(res)), tuple) else res
        old[name] = json.dumps(enc)
    np.savez(OUT, **{k: np.array([v]) for k, v in old.items()})
    print(f"saved {OUT} (arms: {sorted(old)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="q2,r,s2")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    bank = BankAdapter()
    n = args.seeds or (20 if not args.smoke else 4)
    arms = {}
    todo = [a.strip() for a in args.arms.split(",")]
    if "q2" in todo:
        arms["armQ2"] = arm_Q2(bank, n, args.smoke)
    if "r" in todo:
        arms["armR"] = arm_R(bank, 20 if not args.smoke else 3, args.smoke)
    if "s2" in todo:
        arms["armS2"] = arm_S2(bank, n, args.smoke)
    _merge_save(arms)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
