"""M26 — evidence-adaptive DECLARED-task boundary hazard (F89): the queued
F86 successor one level down (docs/M25_SELECTION_V2.md §6 item 1).

MECHANISM. Gate 4 re-widens EVERY task at every session boundary; M25
measured its cost on a declared task at ~28 trials/session of
re-confirmation, forever — the binding constraint that made every
allocator-level fix (F85/F86) inert on the BOTH-declared endpoint. The
fix under test: a task that has declared and then RE-CONFIRMED across n
boundaries pays a hazard scaled by the Beta posterior-mean factor
c0/(c0+n) (training.mixture_filter.declared_hazard_scale — linear-in-ρ
updates make the plug-in exact; geometric λ^n is the aggressive
ablation; a floor keeps a permanent fraction of the pinned hazard).
Never-declared tasks always pay the full pinned hazard (the F80
breakout machinery is untouched by construction), and a D33 revocation
resets the count.

This is a BELIEF-MODEL change (it weakens the M23 safety machinery on
the declared side), so it carries its own FG guardrail campaign — the
new load-bearing scenario is TRUE REGRESSION: a genuinely-mastered,
repeatedly-confirmed learner whose regime re-draws DOWNWARD at a
boundary, exactly the event the decayed hazard is slower to re-open.

Arms (all at the M23 shipping stack — Gate 4 + sd_floor 0.33, z1
placement, Gate-1 allocation — over the real bank, K=2, D33-style
confirmation lifecycle at session opens):

  A  ENDPOINT (the M25 arm-Q2 rosters): {rates, jumper, b-trap} ×
     hazard {fixed, c5 (c0=5, no floor), c5f25, c2f25, g5f25} —
     BOTH-declared trial + rate, per-task firsts, post-declaration
     service split, and the re-polish tax measured directly (trials
     to the declared task after its first declaration, per session).

  G  REGRESSION GUARDRAIL: task0 masters (well-spec fast), confirms
     across boundaries (decay engages), then re-draws DOWN at the
     session-5 boundary to margin {deep ℓ*−0.405, sharp ℓ*−0.15};
     task1 well-spec (service competition). Metrics: wrongly-satisfied
     session-opens after regression (would-be D33 re-confirms), trials
     to sustained gate loss, end-state gate, post-regression service.
     Hazard {fixed, c2f25, g5f25, c2f0 (floor priced)}.

  F  FG ZOO (the F88 scenarios): {static_below ℓ*−0.15, careless} ×
     hazard {fixed, c2f25, g5f25} — declared FG, CONFIRMED FG (X4
     semantics), and post-FG gate persistence at opens.

Run:  python3 -m studies.study_m26_hazard [--smoke] [--arms a,g,f]
Out:  figures/data_m26_hazard.npz  (arm keys merged across runs)
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
from training.learner_sim import Learner
from training.mixture_filter import AdaptiveBoundaryHazard
from studies.study_m24_selection import (LAPSE, first_declared, trunc_draw)
from studies.study_m25_selection import (make_policy, roster_btrap,
                                         roster_jumper, roster_rates)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(FIG, "data_m26_hazard.npz")

# hazard arm definitions (None = fixed pinned hazard, the M23/M25 baseline)
HAZ = {"fixed": None,
       "c5": dict(strength=5.0, floor=0.0),
       "c5f25": dict(strength=5.0, floor=0.25),
       "c2f25": dict(strength=2.0, floor=0.25),
       "c2f0": dict(strength=2.0, floor=0.0),
       "g5f25": dict(geometric=0.5, strength=5.0, floor=0.25)}


def make_hazard(name, n_tasks):
    kw = HAZ[name]
    return None if kw is None else AdaptiveBoundaryHazard(n_tasks, **kw)


def run_sessions_h(pol, filters, tasks, respond, n_sessions, rng, *,
                   hazard=None, trials_per=40, session_hook=None):
    """The M24 harness loop + the D33-style provisional→confirm lifecycle
    at session opens (the F88/X4 layer, sandbox ordering: confirm BEFORE
    the boundary shift) + the M26 per-task adaptive Gate-4 hazard.
    hazard=None reproduces the M25 harness exactly (fixed pinned hazard).
    Returns rows + lifecycle + per-open gate states and applied scales."""
    K = len(tasks)
    rows, was = [], [False] * K
    provisional, confirmed = {}, {}
    revoked, open_gate, scales = 0, [], []
    for s_i in range(n_sessions):
        if session_hook is not None:
            session_hook(s_i)
        if s_i > 0:
            g = [bool(pol.mode_policies[i].is_mastered(filters[i]))
                 for i in range(K)]
            open_gate.append(g)
            # M26 hazard credit at the open (BEFORE the boundary): π-based
            # (see AdaptiveBoundaryHazard.credit_from_pi — the sd-floor
            # conjunct is hazard-injected and cannot be the evidence unit)
            if hazard is not None:
                for i in range(K):
                    pi, _ = filters[i].pass_mass(C.ELL_STAR[tasks[i]])
                    hazard.credit_from_pi(i, pi)
            # D33-style confirm layer — METRIC ONLY here: in this anchor-
            # less harness strict-gate revocations are mostly noise (F88
            # X4: honest confirmation 6/20 within 240 trials), so the
            # hazard reset rides credit_from_pi's π-collapse rule instead;
            # the SANDBOX (anchored confirm checks) does reset on REVOKED.
            for i in list(provisional):
                if g[i]:
                    confirmed[i] = len(rows)
                else:
                    revoked += 1
                del provisional[i]
            eps_b, lam_b, ths_b = C.BOUNDARY_JUMP
            sc_row = []
            for i, f in enumerate(filters):
                sc = hazard.scale(i) if hazard is not None else 1.0
                f.boundary_shift(eps_b * sc, lam_b, ths_b,
                                 w_share=C.BOUNDARY_WSHARE * sc)
                if hazard is not None:
                    hazard.on_boundary(i)
                pol.mode_policies[i].note_boundary()
                sc_row.append(sc)
            scales.append(sc_row)
            # NB: a gate that held straight through the boundary is NOT
            # credited — pending debt is paid only by evidence accumulated
            # AFTER the boundary (π at the next open / an in-session
            # re-demonstration), never by the shock happening to be small
            was = [bool(pol.mode_policies[i].is_mastered(filters[i]))
                   for i in range(K)]
        pol.suspended = set()
        served = 0
        now = 1_800_000_000.0 + s_i * 86_400.0
        while served < trials_per:
            now += 30.0
            ch = pol.step(now=now)
            if ch is None:
                break
            k = ch["task"]
            s_real = trunc_draw(rng, ch["s"], ch["s_sd"], ch["y_star"])
            y = int(respond(k, s_real))
            pol.record(ch, y)
            served += 1
            mast = [bool(pol.mode_policies[i].is_mastered(filters[i]))
                    for i in range(K)]
            rows.append({"k": k, "session": s_i, "mastered": mast})
            for i in range(K):
                if mast[i] and not was[i]:
                    if i not in confirmed and i not in provisional:
                        provisional[i] = len(rows) - 1
                    if hazard is not None:
                        hazard.on_gate_pass(i)
                was[i] = mast[i]
    return dict(rows=rows, confirmed=confirmed,
                provisional=dict(provisional), revoked=revoked,
                open_gate=open_gate, scales=scales)


def both_declared(rows):
    a, b = first_declared(rows, 0), first_declared(rows, 1)
    return None if a is None or b is None else max(a, b)


def _svc_after(rows, decl_task, count_task):
    """Trials/session served to count_task after decl_task first declares."""
    a = first_declared(rows, decl_task)
    if a is None:
        return float("nan")
    tail = rows[a:]
    n_sess = len(set(r["session"] for r in tail)) or 1
    return sum(1 for r in tail if r["k"] == count_task) / n_sess


# ── arm A: endpoint on the M25 rosters ──
def arm_A(bank, n_seeds, smoke):
    print("\n== ARM A: BOTH-declared endpoint under the adaptive hazard ==")
    tasks = [1, 2]
    n_sess = 8 if not smoke else 3
    cap = n_sess * 40
    rosters = {"rates": roster_rates, "jumper": roster_jumper,
               "b-trap": roster_btrap}
    arms = {"rates": ("fixed", "c5", "c5f25", "c2f25", "g5f25"),
            "jumper": ("fixed", "c5", "c5f25", "c2f25", "g5f25"),
            "b-trap": ("fixed", "c2f25", "g5f25")}
    out = {}
    for rname, mk in rosters.items():
        for hname in arms[rname]:
            d0, d1, db, sunk, svc1, tax0, scl = [], [], [], [], [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(500 + sd)
                pol, filters = make_policy(bank, tasks, 600 + 37 * sd,
                                           rule="z1", alloc="gate1")
                res = run_sessions_h(pol, filters, tasks,
                                     mk(tasks, 7000 + sd), n_sess, rng,
                                     hazard=make_hazard(hname, 2))
                rows = res["rows"]
                a, b = first_declared(rows, 0), first_declared(rows, 1)
                d0.append(a if a is not None else cap + 1)
                d1.append(b if b is not None else cap + 1)
                bd = both_declared(rows)
                db.append(bd if bd is not None else cap + 1)
                sunk.append(sum(1 for r in rows if r["k"] == 1))
                svc1.append(_svc_after(rows, 0, 1))
                tax0.append(_svc_after(rows, 0, 0))
                if res["scales"]:
                    scl.append(float(np.mean([s[0] for s in
                                              res["scales"][-2:]])))
            out[(rname, hname)] = dict(
                d0=float(np.median(d0)), d1=float(np.median(d1)),
                dboth=float(np.median(db)),
                rboth=float(np.mean([d <= cap for d in db])),
                sunk=float(np.mean(sunk)),
                svc1=float(np.nanmean(svc1)),
                tax0=float(np.nanmean(tax0)),
                end_scale=float(np.mean(scl)) if scl else 1.0)
            print(f"  {rname:<7} [{hname:>6}]: task0 {np.median(d0):5.0f} "
                  f"task1 {np.median(d1):5.0f}  BOTH {np.median(db):5.0f} "
                  f"(rate {np.mean([d <= cap for d in db]):.0%})  "
                  f"tax0/sess {np.nanmean(tax0):4.1f}  "
                  f"svc1/sess {np.nanmean(svc1):4.1f}  "
                  f"end-scale {out[(rname, hname)]['end_scale']:.2f}")
    return out


# ── arm G: true-regression guardrail ──
def arm_G(bank, n_seeds, smoke):
    print("\n== ARM G: regression-after-confirmation guardrail ==")
    tasks = [1, 2]
    n_sess = 10 if not smoke else 4
    reg_s = 5 if not smoke else 2
    margins = {"deep": 0.405, "sharp": 0.15}
    out = {}
    for mname, m in margins.items():
        sig_reg = float(np.exp(-(C.ELL_STAR[tasks[0]] - m)))
        for hname in ("fixed", "c5f25", "c2f25", "g5f25", "c2f0"):
            wrong, det, end_ok, svc0, nconf = [], [], [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(300 + sd)
                pol, filters = make_policy(bank, tasks, 400 + 37 * sd,
                                           rule="z1", alloc="gate1")
                lnrs = [Learner([1.0], [0.1], C.assumed_params(t),
                                seed=4400 + sd + 7 * i)
                        for i, t in enumerate(tasks)]
                rz = np.random.default_rng(5400 + sd)
                st = {"reg": False}

                def respond(k, s):
                    if k == 1:
                        return lnrs[1].step(s, 0, y_star=int(s > 0),
                                            feedback=True)
                    if st["reg"]:
                        pr = LAPSE + (1 - 2 * LAPSE) * norm.cdf(
                            (s - 0.1) / sig_reg)
                        return int(rz.random() < pr)
                    return lnrs[0].step(s, 0, y_star=int(s > 0),
                                        feedback=True)

                haz = make_hazard(hname, 2)
                res = run_sessions_h(
                    pol, filters, tasks, respond, n_sess, rng, hazard=haz,
                    session_hook=lambda s_i: st.__setitem__(
                        "reg", st["reg"] or s_i >= reg_s))
                rows = res["rows"]
                reg_row = next((j for j, r in enumerate(rows)
                                if r["session"] >= reg_s), len(rows))
                # session-opens after regression where the gate is still
                # (wrongly) satisfied — a D33 check there would confirm
                post_opens = res["open_gate"][reg_s:]
                wrong.append(sum(1 for g in post_opens if g[0]))
                last_true = max((j for j, r in enumerate(rows)
                                 if r["mastered"][0]), default=-1)
                det.append(max(last_true - reg_row + 1, 0)
                           if last_true >= reg_row else 0)
                end_ok.append(bool(rows[-1]["mastered"][0]))
                tail = rows[reg_row:]
                ns = len(set(r["session"] for r in tail)) or 1
                svc0.append(sum(1 for r in tail if r["k"] == 0) / ns)
                nconf.append(haz.n[0] if haz is not None else 0)
            out[(mname, hname)] = dict(
                wrong=float(np.mean(wrong)),
                det=float(np.median(det)),
                end_bad=float(np.mean(end_ok)),
                svc0=float(np.mean(svc0)),
                nconf=float(np.mean(nconf)))
            print(f"  {mname:<5} [{hname:>6}]: wrongly-satisfied opens "
                  f"{np.mean(wrong):4.2f}/{n_sess - reg_s - 1}  "
                  f"last-true +{np.median(det):4.0f} trials  "
                  f"end-gate-still-on {np.mean(end_ok):.0%}  "
                  f"svc0/sess {np.mean(svc0):4.1f}  "
                  f"n_conf@end {np.mean(nconf):.1f}")
    return out


# ── arm F: the F88 FG zoo ──
def arm_F(bank, n_seeds, smoke):
    print("\n== ARM F: FG zoo (static sharp-margin, careless) ==")
    tasks = [1, 2]
    n_sess = 6 if not smoke else 3
    sig_b = {i: float(np.exp(-(C.ELL_STAR[t] - 0.15)))
             for i, t in enumerate(tasks)}
    out = {}
    for zoo in ("static15", "careless"):
        for hname in ("fixed", "c2f25", "g5f25"):
            fg_d, fg_c, persist = 0, 0, []
            for sd in range(n_seeds):
                rng = np.random.default_rng(800 + sd)
                pol, filters = make_policy(bank, tasks, 900 + 37 * sd,
                                           rule="z1", alloc="gate1")
                rz = np.random.default_rng(1800 + sd)

                def respond(k, s):
                    if zoo == "static15":
                        pr = LAPSE + (1 - 2 * LAPSE) * norm.cdf(
                            (s - 0.1) / sig_b[k])
                    else:
                        pr = 0.35 + 0.30 * norm.cdf(
                            (s - 0.1) / float(C.SIGMA_STAR[tasks[k]]))
                    return int(rz.random() < pr)

                res = run_sessions_h(pol, filters, tasks, respond, n_sess,
                                     rng, hazard=make_hazard(hname, 2))
                rows = res["rows"]
                d = next((j for j, r in enumerate(rows)
                          if any(r["mastered"])), None)
                fg_d += int(d is not None)
                fg_c += int(bool(res["confirmed"]))
                if d is not None:
                    s_d = rows[d]["session"]
                    post = res["open_gate"][s_d:]
                    if post:
                        persist.append(np.mean([any(g) for g in post]))
            out[(zoo, hname)] = dict(
                fg_declared=fg_d, fg_confirmed=fg_c,
                persist=float(np.mean(persist)) if persist else 0.0)
            print(f"  {zoo:<9} [{hname:>6}]: FG declared {fg_d}/{n_seeds}  "
                  f"CONFIRMED {fg_c}/{n_seeds}  post-FG open-gate "
                  f"{out[(zoo, hname)]['persist']:.2f}")
    return out


def _merge_save(new_arms):
    os.makedirs(FIG, exist_ok=True)
    old = {}
    if os.path.exists(OUT):
        with np.load(OUT, allow_pickle=False) as z:
            old = {k: str(z[k][0]) for k in z.files}
    for name, res in new_arms.items():
        enc = {f"{a}|{b}": v for (a, b), v in res.items()}
        old[name] = json.dumps(enc)
    np.savez(OUT, **{k: np.array([v]) for k, v in old.items()})
    print(f"saved {OUT} (arms: {sorted(old)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="a,g,f")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    bank = BankAdapter()
    n = args.seeds or (20 if not args.smoke else 3)
    arms = {}
    todo = [a.strip() for a in args.arms.split(",")]
    if "a" in todo:
        arms["armA"] = arm_A(bank, n, args.smoke)
    if "g" in todo:
        arms["armG"] = arm_G(bank, n, args.smoke)
    if "f" in todo:
        arms["armF"] = arm_F(bank, n, args.smoke)
    _merge_save(arms)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
