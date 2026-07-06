"""M27 — terminal-confirmation semantics (F90): the primary F86-line
successor (docs/M26_HAZARD.md §6 item 1; the design M25 named and D42
re-queued).

MECHANISM. A D33-CONFIRMED task leaves the training rotation for good:
the Gate-4 hazard keeps its FULL pinned rate on the belief (nothing in
the M23 safety machinery is weakened — the D42 lesson), but the
scheduler stops converting the boundary re-widening into a
~28-trials/session re-polish tax. The retention layer owns the task
(due-driven review at SM-2 day cadence + `maint_probes` at-bar
certification probes per session), and every terminal-task outcome feeds
an anytime-valid stale-mastery e-gate (Ville, α: P(ever wrongly revoking
an at/above-bar learner) ≤ α per task); a fired gate returns the task to
the training rotation.

HARNESS NOTE (recorded honestly). This study's lifecycle harness serves
an 8-trial info-optimal ANCHOR BLOCK to each provisional task before its
confirm check — the F88/M26 harnesses omitted the sandbox's gap-anchor
machinery, which made honest confirmation artificially rare (6/20 in
240 trials) and would starve the mechanism under test. All arms
(including base) run the same anchored harness; baselines are re-pinned
here rather than compared to the M25/M26 no-anchor numbers.

Arms (M23 shipping stack, fixed hazard, z1/gate1, real bank, K=2,
day-cadence retention):

  T  ENDPOINT: rosters {rates, jumper, b-trap} × {base, term0 (terminal,
     no maintenance), term3 (3 maintenance probes/session)} — BOTH-
     declared + BOTH-CONFIRMED (+rates), the post-confirmation tax
     (trials/session to the confirmed task), service to the second task.

  R  REGRESSION GUARDRAIL: task0 masters + confirms + goes terminal,
     then re-draws DOWN at the session-5 boundary to {deep ℓ*−0.405,
     sharp ℓ*−0.15}; task1 well-spec. Metrics: sessions wrongly
     terminal, stale-gate revocation lag, end-state, evidence
     trials/session reaching the regressed task.

  Z  FG ZOO: {static ℓ*−0.15, careless} × {base, term3} — declared FG,
     CONFIRMED FG, and wrongly-TERMINAL-at-end (the lock-in metric: a
     falsely-confirmed static task's stale gate faces the F88
     information bound, so lock-in is expected — priced, with the D18
     re-cert backstop as the architectural answer).

Run:  python3 -m studies.study_m27_terminal [--smoke] [--arms t,r,z]
Out:  figures/data_m27_terminal.npz  (arm keys merged across runs)
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
from training.bridge_conventions import engine_to_plan
from training.trainer_policy import (ModeThresholds, RetentionScheduler,
                                     TrainerPolicy, cert_probe_score)
from studies.study_m24_selection import (BANK_S_SD, LAPSE, _Remap,
                                         first_declared, fresh_mixture,
                                         trunc_draw)
from studies.study_m25_selection import (roster_btrap, roster_jumper,
                                         roster_rates)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(FIG, "data_m27_terminal.npz")

N_ANCHOR = 8


def make_policy_t(bank, tasks, seed, *, terminal=False, maint=0):
    filters = [fresh_mixture(t, seed + 1 + 31 * i)
               for i, t in enumerate(tasks)]
    th = ModeThresholds(meanskill_gate=False, sd_floor=C.SD_FLOOR,
                        skill_sigma_z=1.0)
    pol = TrainerPolicy(
        filters, [C.ELL_STAR[t] for t in tasks],
        [C.SIGMA_STAR[t] for t in tasks], _Remap(bank, list(tasks)),
        thresholds=th, seed=seed + 5, probe_every=C.PROBE_EVERY,
        finish_first=True, trainability_floor=C.TRAINABILITY_FLOOR,
        finish_sd_tol=C.FINISH_SD_TOL,
        retention=RetentionScheduler(first_interval=86_400.0),
        terminal_confirmation=terminal, maint_probes=maint,
        stale_alpha=C.STALE_ALPHA)
    return pol, filters


def _anchor_block(pol, filters, tasks, i, respond, rng, n_probes):
    """The harness stand-in for the sandbox gap-anchor block: n info-
    optimal at-bar probes served to task i, filter stepped directly."""
    served = 0
    want = 1
    for _ in range(n_probes):
        excl = pol.exclude | pol._served
        pool = pol.bank.candidates(i, exclude_segids=excl,
                                   feedback_safe=True,
                                   min_margin=pol.min_margin)
        if len(pool) == 0:
            break
        _, t_hat = engine_to_plan(*filters[i].mean())
        m = np.where(pool.y_star == want)[0]
        if m.size == 0:
            m = np.arange(len(pool))
        sc = cert_probe_score(pool.s_mean[m], pool.s_sd[m],
                              pol.sigma_stars[i], t_hat)
        j = int(m[np.argmax(sc)])
        s_real = trunc_draw(rng, float(pool.s_mean[j]),
                            float(pool.s_sd[j]), int(pool.y_star[j]))
        y = int(respond(i, s_real))
        filters[i].step(float(pool.s_mean[j]), y, int(pool.y_star[j]),
                        s_sd=float(pool.s_sd[j]))
        pol.exclude.add(int(pool.seg_id[j]))
        want = 1 - want
        served += 1
    return served


def run_sessions_t(pol, filters, tasks, respond, n_sessions, rng, *,
                   terminal=False, trials_per=40, session_hook=None):
    """M26 lifecycle harness + anchored D33 confirms + F90 terminal
    semantics. Fixed pinned hazard (the D42-shipped config)."""
    K = len(tasks)
    rows, was = [], [False] * K
    provisional, confirmed, revocations = {}, {}, []
    term_state = []                       # per-open terminal membership
    for s_i in range(n_sessions):
        if session_hook is not None:
            session_hook(s_i)
        budget = trials_per
        if s_i > 0:
            # anchor blocks for tasks awaiting confirmation (sandbox
            # open_gap_phase analog), then the strict confirm check
            for i in list(provisional):
                budget -= _anchor_block(pol, filters, tasks, i, respond,
                                        rng, min(N_ANCHOR, budget))
            for i in list(provisional):
                if pol.mode_policies[i].is_mastered(filters[i]):
                    confirmed[i] = len(rows)
                    if terminal:
                        pol.mark_terminal(
                            i, now=1_800_000_000.0 + s_i * 86_400.0)
                del provisional[i]
            for i, f in enumerate(filters):
                f.boundary_shift(*C.BOUNDARY_JUMP,
                                 w_share=C.BOUNDARY_WSHARE)
                pol.mode_policies[i].note_boundary()
            was = [bool(pol.mode_policies[i].is_mastered(filters[i]))
                   for i in range(K)]
        term_state.append(sorted(pol.terminal))
        pol.note_session_open()
        pol.suspended = set()
        served = 0
        now = 1_800_000_000.0 + s_i * 86_400.0
        while served < budget:
            now += 30.0
            ch = pol.step(now=now)
            if ch is None:
                break
            k = ch["task"]
            s_real = trunc_draw(rng, ch["s"], ch["s_sd"], ch["y_star"])
            y = int(respond(k, s_real))
            pol.record(ch, y)
            served += 1
            while pol.terminal_events:
                ev = pol.terminal_events.pop(0)
                revocations.append({"task": ev["task"], "row": len(rows),
                                    "session": s_i})
            mast = [bool(pol.mode_policies[i].is_mastered(filters[i]))
                    for i in range(K)]
            rows.append({"k": k, "mode": ch["mode"], "session": s_i,
                         "mastered": mast,
                         "maint": bool(ch["info"].get("maintenance"))})
            for i in range(K):
                if mast[i] and not was[i]:
                    if i not in confirmed and i not in provisional:
                        provisional[i] = len(rows) - 1
                was[i] = mast[i]
    return dict(rows=rows, confirmed=confirmed, revocations=revocations,
                terminal=sorted(pol.terminal), term_state=term_state)


def both_declared(rows):
    a, b = first_declared(rows, 0), first_declared(rows, 1)
    return None if a is None or b is None else max(a, b)


def _svc_per_sess(rows, start_row, k):
    tail = rows[start_row:]
    ns = len(set(r["session"] for r in tail)) or 1
    return sum(1 for r in tail if r["k"] == k) / ns


ARMS_T = {"base": dict(terminal=False, maint=0),
          "term0": dict(terminal=True, maint=0),
          "term3": dict(terminal=True, maint=3)}


# ── arm T: endpoint ──
def arm_T(bank, n_seeds, smoke):
    print("\n== ARM T: endpoint under terminal confirmation ==")
    tasks = [1, 2]
    n_sess = 8 if not smoke else 3
    cap = n_sess * 40
    rosters = {"rates": roster_rates, "jumper": roster_jumper,
               "b-trap": roster_btrap}
    out = {}
    for rname, mk in rosters.items():
        for aname, kw in ARMS_T.items():
            db, dcf, tax0, svc1, nconf = [], [], [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(500 + sd)
                pol, filters = make_policy_t(bank, tasks, 600 + 37 * sd,
                                             terminal=kw["terminal"],
                                             maint=kw["maint"])
                res = run_sessions_t(pol, filters, tasks,
                                     mk(tasks, 7000 + sd), n_sess, rng,
                                     terminal=kw["terminal"])
                rows = res["rows"]
                bd = both_declared(rows)
                db.append(bd if bd is not None else cap + 1)
                cf = res["confirmed"]
                dcf.append(max(cf.values()) if len(cf) == 2 else cap + 1)
                nconf.append(len(cf))
                if 0 in cf:
                    tax0.append(_svc_per_sess(rows, cf[0], 0))
                    svc1.append(_svc_per_sess(rows, cf[0], 1))
            out[(rname, aname)] = dict(
                dboth=float(np.median(db)),
                rboth=float(np.mean([d <= cap for d in db])),
                dconf=float(np.median(dcf)),
                rconf=float(np.mean([d <= cap for d in dcf])),
                nconf=float(np.mean(nconf)),
                tax0=float(np.mean(tax0)) if tax0 else float("nan"),
                svc1=float(np.mean(svc1)) if svc1 else float("nan"))
            o = out[(rname, aname)]
            print(f"  {rname:<7} [{aname:>5}]: BOTH decl {o['dboth']:5.0f} "
                  f"({o['rboth']:.0%})  BOTH conf {o['dconf']:5.0f} "
                  f"({o['rconf']:.0%})  post-conf tax0/sess {o['tax0']:4.1f} "
                  f" svc1/sess {o['svc1']:4.1f}")
    return out


# ── arm R: regression guardrail ──
def arm_R(bank, n_seeds, smoke):
    print("\n== ARM R: regression after terminal confirmation ==")
    tasks = [1, 2]
    n_sess = 10 if not smoke else 4
    reg_s = 5 if not smoke else 2
    margins = {"deep": 0.405, "sharp": 0.15}
    out = {}
    for mname, m in margins.items():
        sig_reg = float(np.exp(-(C.ELL_STAR[tasks[0]] - m)))
        for aname, kw in ARMS_T.items():
            wrong, rev_lag, end_term, ev_sess = [], [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(300 + sd)
                pol, filters = make_policy_t(bank, tasks, 400 + 37 * sd,
                                             terminal=kw["terminal"],
                                             maint=kw["maint"])
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

                res = run_sessions_t(
                    pol, filters, tasks, respond, n_sess, rng,
                    terminal=kw["terminal"],
                    session_hook=lambda s_i: st.__setitem__(
                        "reg", st["reg"] or s_i >= reg_s))
                rows = res["rows"]
                reg_row = next((j for j, r in enumerate(rows)
                                if r["session"] >= reg_s), len(rows))
                if kw["terminal"]:
                    # sessions (after regression) STILL terminal at open
                    wrong.append(sum(1 for s_j, ts in
                                     enumerate(res["term_state"])
                                     if s_j > reg_s and 0 in ts))
                    rv = [r for r in res["revocations"]
                          if r["task"] == 0 and r["row"] >= reg_row]
                    rev_lag.append(rv[0]["row"] - reg_row if rv
                                   else len(rows) - reg_row + 1)
                    end_term.append(0 in res["terminal"])
                else:
                    wrong.append(sum(1 for j in range(reg_s + 1, n_sess)
                                     if any(r["mastered"][0] for r in rows
                                            if r["session"] == j)))
                    lt = max((j for j, r in enumerate(rows)
                              if r["mastered"][0]), default=-1)
                    rev_lag.append(max(lt - reg_row + 1, 0))
                    end_term.append(bool(rows[-1]["mastered"][0]))
                ev_sess.append(_svc_per_sess(rows, reg_row, 0))
            out[(mname, aname)] = dict(
                wrong=float(np.mean(wrong)),
                rev_lag=float(np.median(rev_lag)),
                end_bad=float(np.mean(end_term)),
                ev=float(np.mean(ev_sess)))
            o = out[(mname, aname)]
            print(f"  {mname:<5} [{aname:>5}]: wrongly-on sessions "
                  f"{o['wrong']:4.2f}/{n_sess - reg_s - 1}  "
                  f"revoke/last-true lag {o['rev_lag']:5.0f}  "
                  f"end-still-on {o['end_bad']:.0%}  "
                  f"task0 evidence/sess {o['ev']:4.1f}")
    return out


# ── arm Z: FG zoo ──
def arm_Z(bank, n_seeds, smoke):
    print("\n== ARM Z: FG zoo under terminal confirmation ==")
    tasks = [1, 2]
    n_sess = 6 if not smoke else 3
    sig_b = {i: float(np.exp(-(C.ELL_STAR[t] - 0.15)))
             for i, t in enumerate(tasks)}
    out = {}
    for zoo in ("static15", "careless"):
        for aname in ("base", "term3"):
            kw = ARMS_T[aname]
            fg_d, fg_c, locked = 0, 0, 0
            for sd in range(n_seeds):
                rng = np.random.default_rng(800 + sd)
                pol, filters = make_policy_t(bank, tasks, 900 + 37 * sd,
                                             terminal=kw["terminal"],
                                             maint=kw["maint"])
                rz = np.random.default_rng(1800 + sd)

                def respond(k, s):
                    if zoo == "static15":
                        pr = LAPSE + (1 - 2 * LAPSE) * norm.cdf(
                            (s - 0.1) / sig_b[k])
                    else:
                        pr = 0.35 + 0.30 * norm.cdf(
                            (s - 0.1) / float(C.SIGMA_STAR[tasks[k]]))
                    return int(rz.random() < pr)

                res = run_sessions_t(pol, filters, tasks, respond, n_sess,
                                     rng, terminal=kw["terminal"])
                rows = res["rows"]
                fg_d += int(any(any(r["mastered"]) for r in rows))
                fg_c += int(bool(res["confirmed"]))
                locked += int(bool(res["terminal"]))
            out[(zoo, aname)] = dict(fg_declared=fg_d, fg_confirmed=fg_c,
                                     locked=locked)
            print(f"  {zoo:<9} [{aname:>5}]: FG declared {fg_d}/{n_seeds}  "
                  f"CONFIRMED {fg_c}/{n_seeds}  wrongly-terminal@end "
                  f"{locked}/{n_seeds}")
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
    ap.add_argument("--arms", default="t,r,z")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    bank = BankAdapter()
    n = args.seeds or (20 if not args.smoke else 3)
    arms = {}
    todo = [a.strip() for a in args.arms.split(",")]
    if "t" in todo:
        arms["armT"] = arm_T(bank, n, args.smoke)
    if "r" in todo:
        arms["armR"] = arm_R(bank, n, args.smoke)
    if "z" in todo:
        arms["armZ"] = arm_Z(bank, n, args.smoke)
    _merge_save(arms)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
