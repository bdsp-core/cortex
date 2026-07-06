"""M24 — expected-progress selection study (F84/F85): validate posterior
expected-progress PLACEMENT (skill mode) and expected-progress-per-trial
ALLOCATION against the shipped M22/M23 rules, at the M23 shipping config
(boundary_shift Gate 4 + sd_floor 0.33) over the real bank.

Motivation (docs/M24_SELECTION.md): the seven-profile audit shows the
pipeline delivered a mean per-trial training value w_true = 0.48 of ideal;
the F80-lag sessions fell to 0.15–0.17 (A-s3/E-s3 over-easy against a
stale-high σ̂; D-s1/E-s1 over-hard). Skill placement ranks by plug-in E[w]
— three stacked approximations (point estimate, no gap-weighting, no
mixture structure) that `expected_progress_score` removes; allocation
ranks by worst-first deficiency that `expected_progress_rate` replaces
with bar-referenced posterior progress per trial.

Arms:
  P  placement OC (K=1, real bank, M23 stack): scenarios {well-spec,
     E-like jump, pre-contact→contact, slow-wide} × rules {legacy z=0,
     F75 z=1 (sandbox-shipped), F84 progress}: declared trial, delivered
     TRUE training value E[w_true], served-accuracy contract, t̂ error.
  H  criterion-herding safety (the device the progress path bypasses is
     the M10 mirror pairing): 300-trial single-task runs with live R–W
     criterion dynamics; |t_true| end/max + t̂ tracking error.
  Q  allocation OC (K=2, real bank, M23 stack): rosters {near-bar +
     plateau (the USER-B trap), asymmetric-rate pair, trainable +
     pre-contact-jumper} × rules {legacy worst-first, D36 Gate-1
     discount, F85 EP}: per-task declaration, plateau-sunk trials,
     near-bar starvation, FG.
  S  guardrails: FG zoo (static_below ℓ*−0.15, careless) + well-spec
     declaration lateness at the REAL 2-task rate (the arm-G lesson), for
     {M23 shipped, M24 SHIPPING = EP allocation only, M24x experimental =
     + progress placement}. The placement flag is a recorded NEGATIVE
     result (F84: six variants; training-value gains never convert to
     faster declarations under the M23 hazard-widened belief — see
     docs/M24_SELECTION.md) and stays default-off.

Run:  python3 -m studies.study_m24_selection [--smoke]
Out:  figures/data_m24_selection.npz
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
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import (ModeThresholds, TrainerPolicy)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
LAPSE = 0.025
MULT = 1.0772256
RHO = 0.5
BANK_S_SD = 0.057


class _Remap:
    def __init__(self, bank, tasks):
        self.bank, self.tasks = bank, tasks

    def candidates(self, task, **kw):
        return self.bank.candidates(self.tasks[task], **kw)


def fresh_mixture(task, seed):
    rng = np.random.default_rng(seed)
    th0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
    el0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
    return SigmaInfMixtureFilter(
        th0, el0, C.assumed_params(task), ell_inf_mean=C.EXPERT_ELL[task],
        tau=C.MIX_TAU, J=C.MIX_J, p_static_stratum=C.P_STATIC_STRATUM,
        ell_star=C.ELL_STAR[task], seed=seed + 31 * task)


def make_policy(bank, tasks, seed, *, rule, alloc="gate1"):
    """rule: 'z0' | 'z1' | 'prog' (placement); alloc: 'legacy'|'gate1'|'ep'."""
    filters = [fresh_mixture(t, seed + 1 + 31 * i)
               for i, t in enumerate(tasks)]
    th = ModeThresholds(meanskill_gate=False,
                        sd_floor=C.SD_FLOOR,
                        skill_sigma_z=(1.0 if rule == "z1" else 0.0),
                        progress_placement=(rule == "prog"))
    pol = TrainerPolicy(
        filters, [C.ELL_STAR[t] for t in tasks],
        [C.SIGMA_STAR[t] for t in tasks], _Remap(bank, list(tasks)),
        thresholds=th, seed=seed + 5, probe_every=C.PROBE_EVERY,
        finish_first=True,
        trainability_floor=(C.TRAINABILITY_FLOOR if alloc == "gate1"
                            else None),
        finish_sd_tol=C.FINISH_SD_TOL,
        progress_alloc=(alloc == "ep"), progress_s_sd=BANK_S_SD)
    return pol, filters


def trunc_draw(rng, s_mean, s_sd, y_star):
    if s_sd <= 0:
        return float(s_mean)
    lo = norm.cdf((0.0 - s_mean) / s_sd)
    u = rng.uniform()
    u_t = lo + u * (1.0 - lo) if y_star == 1 else u * lo
    return float(s_mean + s_sd * norm.ppf(min(max(u_t, 1e-12), 1 - 1e-12)))


def run_sessions(pol, filters, tasks, respond_state, n_sessions, rng,
                 *, trials_per=40, record_truth=None):
    """Serve n_sessions×trials_per through the REAL TrainerPolicy with M23
    boundary shifts between sessions. respond_state(t_idx, s_real) -> y and
    updates the true learner. record_truth(t_idx) -> (sigma_true, t_true)."""
    rows = []
    for s_i in range(n_sessions):
        if s_i > 0:
            for f in filters:
                f.boundary_shift(*C.BOUNDARY_JUMP,
                                 w_share=C.BOUNDARY_WSHARE)
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
            y = int(respond_state(k, s_real))
            pol.record(ch, y)
            served += 1
            row = {"k": k, "mode": ch["mode"], "s": ch["s"],
                   "s_real": s_real, "y": y, "y_star": ch["y_star"],
                   "correct": int(y == ch["y_star"]), "session": s_i,
                   "mastered": [bool(pol.mode_policies[i].is_mastered(
                       filters[i])) for i in range(len(tasks))]}
            if record_truth is not None:
                sig_t, t_t = record_truth(k)
                d = abs(s_real - t_t) / max(sig_t, 1e-9)
                row["w_true"] = float(np.exp(-((d - MULT) ** 2)
                                             / (2 * RHO ** 2)))
                row["sig_true"], row["t_true"] = sig_t, t_t
            rows.append(row)
    return rows


def first_declared(rows, k):
    return next((i for i, r in enumerate(rows) if r["mastered"][k]), None)


# ── arm P: placement (K=1) ──
def arm_P(bank, n_seeds, smoke):
    print("\n== ARM P: placement OC (K=1, real bank, M23 stack) ==")
    task = 1
    n_sess = 6 if not smoke else 3

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
        for rule in ("z0", "z1", "prog"):
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
                decl_rate=float(np.mean([d <= n_sess * 40 for d in decl])),
                w_true=float(np.mean(wtr)), acc=float(np.mean(acc)))
            print(f"  {sname:<9} [{rule:>4}]: declared "
                  f"{np.median(decl):5.0f} (rate "
                  f"{np.mean([d <= n_sess * 40 for d in decl]):.0%})  "
                  f"E[w_true|skill] {np.mean(wtr):.2f}  acc {np.mean(acc):.2f}")
    return out


# ── arm H: criterion herding ──
def arm_H(bank, n_seeds, smoke):
    print("\n== ARM H: criterion-herding safety (live R–W criterion) ==")
    task = 1
    n_sess = 8 if not smoke else 3
    out = {}
    for rule in ("z1", "prog"):
        t_end, t_max, t_err = [], [], []
        for sd in range(n_seeds):
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
                n_sess, rng,
                record_truth=lambda k: (float(lnr.sigma[0]),
                                        float(lnr.t[0])))
            ts = np.array([r["t_true"] for r in rows])
            half = len(ts) // 2
            t_end.append(abs(ts[-10:]).mean())
            t_max.append(abs(ts[half:]).max())
            from training.bridge_conventions import engine_to_plan
            _, t_hat = engine_to_plan(*filters[0].mean())
            t_err.append(abs(t_hat - ts[-1]))
        out[rule] = dict(t_end=float(np.mean(t_end)),
                         t_max=float(np.mean(t_max)),
                         t_err=float(np.mean(t_err)))
        print(f"  [{rule:>4}]: |t_true| end {np.mean(t_end):.3f}  "
              f"max(2nd half) {np.mean(t_max):.3f}  "
              f"|t̂−t| end {np.mean(t_err):.3f}")
    return out


# ── arm Q: allocation (K=2) ──
def arm_Q(bank, n_seeds, smoke):
    print("\n== ARM Q: allocation OC (K=2, real bank, M23 stack) ==")
    tasks = [1, 2]
    n_sess = 6 if not smoke else 3

    def roster_btrap(seed):
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

    def roster_rates(seed):
        ls = []
        for i, t in enumerate(tasks):
            p = LearnerParams(alpha_t=0.097,
                              alpha_sigma=(0.12 if i == 0 else 0.03),
                              sigma_inf=0.72 * C.SIGMA_STAR[t],
                              q_t=0.03, q_sigma=0.015, rho=0.6, rule="soft")
            ls.append(Learner([1.3], [0.1], p, seed=seed + i))
        return lambda k, s: ls[k].step(s, 0, y_star=int(s > 0),
                                       feedback=True)

    def roster_jumper(seed):
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

    rosters = {"b-trap": roster_btrap, "rates": roster_rates,
               "jumper": roster_jumper}
    out = {}
    for rname, mk in rosters.items():
        for alloc in ("legacy", "gate1", "ep"):
            d0, d1, sunk = [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(500 + sd)
                pol, filters = make_policy(bank, tasks, 600 + 37 * sd,
                                           rule="z1", alloc=alloc)
                rows = run_sessions(pol, filters, tasks, mk(7000 + sd),
                                    n_sess, rng)
                a = first_declared(rows, 0)
                b = first_declared(rows, 1)
                d0.append(a if a is not None else len(rows) + 1)
                d1.append(b if b is not None else len(rows) + 1)
                sunk.append(sum(1 for r in rows if r["k"] == 1))
            out[(rname, alloc)] = dict(
                d0=float(np.median(d0)), d1=float(np.median(d1)),
                r0=float(np.mean([d <= n_sess * 40 for d in d0])),
                sunk=float(np.mean(sunk)))
            print(f"  {rname:<7} [{alloc:>6}]: task0 declared "
                  f"{np.median(d0):5.0f} (rate "
                  f"{np.mean([d <= n_sess * 40 for d in d0]):.0%})  "
                  f"task1 trials {np.mean(sunk):5.1f}  task1 declared "
                  f"{np.median(d1):5.0f}")
    return out


# ── arm S: guardrails ──
def arm_S(bank, n_seeds, smoke):
    print("\n== ARM S: guardrails at the shipping config ==")
    tasks = [1, 2]
    n_sess = 6 if not smoke else 3
    sig_below = {i: float(np.exp(-(C.ELL_STAR[t] - 0.15)))
                 for i, t in enumerate(tasks)}
    ARMS = {"m23": ("z1", "gate1"),          # M23 shipped
            "m24": ("z1", "ep"),              # M24 SHIPPING config (F85 only)
            "m24x": ("prog", "ep")}           # experimental (F84 flag on)
    out = {}
    for zoo in ("static_below", "careless"):
        for arm, (rule_, alloc_) in ARMS.items():
            fg = 0
            for sd in range(n_seeds):
                rng = np.random.default_rng(800 + sd)
                pol, filters = make_policy(
                    bank, tasks, 900 + 37 * sd,
                    rule=rule_, alloc=alloc_)
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
            print(f"  {zoo:<13} [{arm}]: FG {fg}/{n_seeds}")
    # declaration lateness, well-spec pair, 2-task rate
    for arm, (rule_, alloc_) in ARMS.items():
        firsts = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(1100 + sd)
            pol, filters = make_policy(
                bank, tasks, 1200 + 37 * sd,
                rule=rule_, alloc=alloc_)
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
        print(f"  lateness (well-spec pair) [{arm}]: first declaration "
              f"median {np.median(firsts):.0f}, rate "
              f"{np.mean([f <= 320 for f in firsts]):.0%}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    bank = BankAdapter()
    n = 20 if not args.smoke else 4
    resP = arm_P(bank, n, args.smoke)
    resH = arm_H(bank, 12 if not args.smoke else 3, args.smoke)
    resQ = arm_Q(bank, n, args.smoke)
    resS = arm_S(bank, n, args.smoke)
    os.makedirs(FIG, exist_ok=True)
    np.savez(os.path.join(FIG, "data_m24_selection.npz"),
             armP=np.array([json.dumps({f"{a}|{b}": v for (a, b), v
                                        in resP.items()})]),
             armH=np.array([json.dumps(resH)]),
             armQ=np.array([json.dumps({f"{a}|{b}": v for (a, b), v
                                        in resQ.items()})]),
             armS=np.array([json.dumps({f"{a}|{b}": v for (a, b), v
                                        in resS.items()})]))
    print(f"\nsaved figures/data_m24_selection.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
