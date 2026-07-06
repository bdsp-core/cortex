"""M25 — sharp-margin false-graduation study (F88): the M23/M24 open item.

The shipped stack (Gate 4 hazards, sd_floor 0.33, probes/e-gate) declares
~37% (22/60) of STATIC learners parked at ℓ*−0.15 within 240 two-task
trials. At that margin the per-trial accuracy contrast is tiny (at-bar
probes: a0 − p_true ≈ 0.03 ⇒ the e-gate needs O(δ⁻²) ≈ 10³ probes — the
Bernoulli information bound, not gate inefficiency), so the fix cannot be
"more evidence per trial"; it has to remove the NON-EVIDENCE channels that
let π transiently clear the gate.

Arms:
  X1 INSTRUMENTATION — static_below at margins {0.10, 0.15, 0.25} on the
     shipped config: at every false declaration capture (session index,
     trials-since-boundary, trainability, π, sd_ℓ, e-gate e). Hypothesis
     to adjudicate: FGs ride the Gate-4 boundary hazard (fixed-share
     re-arms above-cut strata; the state shock scatters particles across
     the bar) rather than accumulating steadily from evidence.
  X2 MITIGATIONS at ℓ*−0.15 — declaration conjuncts, all opt-in:
     refr12  = boundary_refractory 12 (no declaration within 12 served
               trials of a boundary shift),
     tr50    = min_trainability 0.50 (a declaration entails a clearing
               ceiling),
     both    = refr12 + tr50,
     probe2  = probe_every 5 → 2 (densified e-gate evidence),
     Careless zoo re-checked on {base, both}.
  X3 LATENESS — well-specified pair under every X2 arm (D28: FG-safety ≥
     lateness; the conjuncts must not starve honest declarations).

Run:  python3 -m studies.study_sharp_margin [--smoke] [--seeds N]
Out:  figures/data_sharp_margin.npz
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
from training.trainer_policy import ModeThresholds, TrainerPolicy
from studies.study_m24_selection import (BANK_S_SD, LAPSE, _Remap,
                                         fresh_mixture, trunc_draw)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

TASKS = [1, 2]


def make_policy_x(bank, seed, *, refractory=0, min_train=0.0,
                  probe_every=None):
    """The M23 SHIPPED config (z1 placement, Gate-1 allocation) + the F88
    declaration conjuncts under test."""
    filters = [fresh_mixture(t, seed + 1 + 31 * i)
               for i, t in enumerate(TASKS)]
    th = ModeThresholds(meanskill_gate=False, sd_floor=C.SD_FLOOR,
                        skill_sigma_z=1.0,
                        boundary_refractory=refractory,
                        min_trainability=min_train)
    pol = TrainerPolicy(
        filters, [C.ELL_STAR[t] for t in TASKS],
        [C.SIGMA_STAR[t] for t in TASKS], _Remap(bank, list(TASKS)),
        thresholds=th, seed=seed + 5,
        probe_every=(probe_every or C.PROBE_EVERY),
        finish_first=True, trainability_floor=C.TRAINABILITY_FLOOR,
        finish_sd_tol=C.FINISH_SD_TOL)
    return pol, filters


def run_sessions_x(pol, filters, respond, n_sessions, rng, *,
                   trials_per=40, lifecycle=False):
    """The M24 harness loop + boundary refractory signalling + per-
    declaration diagnostics capture. lifecycle=True adds the D33
    provisional→confirm layer at session opens (arm X4): a task that
    declared during a session is only CONFIRMED if the gate still holds
    at the NEXT session open, BEFORE the boundary shift (the sandbox
    ordering); else the provisional is revoked."""
    rows, decls = [], []
    was = [False] * len(TASKS)
    sess_start = 0
    provisional, confirmed = {}, {}
    for s_i in range(n_sessions):
        if lifecycle and s_i > 0:
            for i in list(provisional):
                if pol.mode_policies[i].is_mastered(filters[i]):
                    confirmed[i] = len(rows)
                del provisional[i]
        if s_i > 0:
            for i, f in enumerate(filters):
                f.boundary_shift(*C.BOUNDARY_JUMP,
                                 w_share=C.BOUNDARY_WSHARE)
                pol.mode_policies[i].note_boundary()
        pol.suspended = set()
        served = 0
        sess_start = len(rows)
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
                    for i in range(len(TASKS))]
            rows.append({"k": k, "session": s_i, "mastered": mast})
            for i in range(len(TASKS)):
                if mast[i] and not was[i]:
                    pi, mcse = filters[i].pass_mass(C.ELL_STAR[TASKS[i]])
                    decls.append({
                        "task": i, "trial": len(rows) - 1, "session": s_i,
                        "tsb": len(rows) - 1 - sess_start,
                        "trainability": float(filters[i].trainability(
                            C.ELL_STAR[TASKS[i]])),
                        "pi": float(pi),
                        "sd_l": float(filters[i].sd()[1]),
                        "egate": float(pol.mode_policies[i].egate.e_now
                                       if pol.mode_policies[i].egate
                                       else 0.0)})
                    if lifecycle and i not in confirmed:
                        provisional[i] = len(rows) - 1
                was[i] = mast[i]
    if lifecycle:
        return rows, decls, confirmed, provisional
    return rows, decls


def static_responder(margin, seed):
    sig = {i: float(np.exp(-(C.ELL_STAR[t] - margin)))
           for i, t in enumerate(TASKS)}
    rz = np.random.default_rng(seed)

    def respond(k, s):
        pr = LAPSE + (1 - 2 * LAPSE) * norm.cdf((s - 0.1) / sig[k])
        return int(rz.random() < pr)
    return respond


def careless_responder(seed):
    rz = np.random.default_rng(seed)

    def respond(k, s):
        pr = 0.35 + 0.30 * norm.cdf((s - 0.1) / float(C.SIGMA_STAR[TASKS[k]]))
        return int(rz.random() < pr)
    return respond


def arm_X1(bank, n_seeds, smoke):
    print("\n== ARM X1: instrumentation (static_below, shipped config) ==")
    n_sess = 6 if not smoke else 3
    out = {}
    for margin in (0.10, 0.15, 0.25):
        fg, alld = 0, []
        for sd in range(n_seeds):
            rng = np.random.default_rng(3000 + sd)
            pol, filters = make_policy_x(bank, 3100 + 37 * sd)
            rows, decls = run_sessions_x(
                pol, filters, static_responder(margin, 3800 + sd),
                n_sess, rng)
            fg += int(bool(decls))
            alld.extend(decls)
        d = {"fg": fg, "n": n_seeds, "n_decl": len(alld)}
        if alld:
            d.update(
                tsb_med=float(np.median([x["tsb"] for x in alld])),
                tsb_le12=float(np.mean([x["tsb"] <= 12 for x in alld])),
                sess1plus=float(np.mean([x["session"] >= 1 for x in alld])),
                train_med=float(np.median([x["trainability"]
                                           for x in alld])),
                train_lo=float(np.quantile([x["trainability"]
                                            for x in alld], 0.1)),
                pi_med=float(np.median([x["pi"] for x in alld])),
                egate_med=float(np.median([x["egate"] for x in alld])))
            print(f"  margin {margin:.2f}: FG {fg}/{n_seeds} "
                  f"({len(alld)} decls)  tsb med {d['tsb_med']:.0f} "
                  f"(<=12: {d['tsb_le12']:.0%})  post-boundary sess: "
                  f"{d['sess1plus']:.0%}  trainability med "
                  f"{d['train_med']:.2f} (q10 {d['train_lo']:.2f})  "
                  f"pi {d['pi_med']:.3f}  egate {d['egate_med']:.2f}")
        else:
            print(f"  margin {margin:.2f}: FG {fg}/{n_seeds}")
        out[f"{margin:.2f}"] = d
    return out


X2_ARMS = {
    "base":   dict(),
    "refr12": dict(refractory=12),
    "tr50":   dict(min_train=0.50),
    "both":   dict(refractory=12, min_train=0.50),
    "probe2": dict(probe_every=2),
}


def arm_X2(bank, n_seeds, smoke):
    print("\n== ARM X2: mitigation conjuncts at margin 0.15 ==")
    n_sess = 6 if not smoke else 3
    out = {}
    for arm, kw in X2_ARMS.items():
        fg = 0
        for sd in range(n_seeds):
            rng = np.random.default_rng(3000 + sd)
            pol, filters = make_policy_x(bank, 3100 + 37 * sd, **kw)
            _, decls = run_sessions_x(
                pol, filters, static_responder(0.15, 3800 + sd),
                n_sess, rng)
            fg += int(bool(decls))
        out[("static15", arm)] = fg
        print(f"  static ell*-0.15 [{arm:>7}]: FG {fg}/{n_seeds}")
    for arm in ("base", "both"):
        fg = 0
        for sd in range(n_seeds):
            rng = np.random.default_rng(4000 + sd)
            pol, filters = make_policy_x(bank, 4100 + 37 * sd,
                                         **X2_ARMS[arm])
            _, decls = run_sessions_x(
                pol, filters, careless_responder(4800 + sd), n_sess, rng)
            fg += int(bool(decls))
        out[("careless", arm)] = fg
        print(f"  careless        [{arm:>7}]: FG {fg}/{n_seeds}")
    return out


def arm_X3(bank, n_seeds, smoke):
    print("\n== ARM X3: well-spec lateness under the conjuncts ==")
    n_sess = 8 if not smoke else 4
    cap = n_sess * 40
    out = {}
    for arm, kw in X2_ARMS.items():
        firsts, boths = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(5000 + sd)
            pol, filters = make_policy_x(bank, 5100 + 37 * sd, **kw)
            lnrs = [Learner([1.0], [0.1], C.assumed_params(t),
                            seed=5900 + sd + i)
                    for i, t in enumerate(TASKS)]
            rows, decls = run_sessions_x(
                pol, filters,
                lambda k, s: lnrs[k].step(s, 0, y_star=int(s > 0),
                                          feedback=True),
                n_sess, rng)
            f = min([d["trial"] for d in decls], default=cap + 1)
            per_task = {}
            for d in decls:
                per_task.setdefault(d["task"], d["trial"])
            b = (max(per_task.values()) if len(per_task) == len(TASKS)
                 else cap + 1)
            firsts.append(f)
            boths.append(b)
        out[arm] = dict(
            first=float(np.median(firsts)),
            rate=float(np.mean([f <= cap for f in firsts])),
            both=float(np.median(boths)),
            rboth=float(np.mean([b <= cap for b in boths])))
        print(f"  [{arm:>7}]: first {np.median(firsts):5.0f} (rate "
              f"{np.mean([f <= cap for f in firsts]):.0%})  BOTH "
              f"{np.median(boths):5.0f} (rate "
              f"{np.mean([b <= cap for b in boths]):.0%})")
    return out


def arm_X4(bank, n_seeds, smoke):
    """The D33 lifecycle as the sharp-margin filter (X1's implication):
    FG measured at the CONFIRMED level — a provisional declaration must
    survive the next session-open gate check (production semantics) —
    plus the well-spec confirmation cost."""
    print("\n== ARM X4: D33 confirmation lifecycle at margin 0.15 ==")
    n_sess = 6 if not smoke else 3
    out = {}
    for scen in ("static15", "careless", "wellspec"):
        fg_decl = fg_conf = pend = 0
        conf_both, conf_any = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(3000 + sd)
            pol, filters = make_policy_x(bank, 3100 + 37 * sd)
            if scen == "static15":
                resp = static_responder(0.15, 3800 + sd)
            elif scen == "careless":
                resp = careless_responder(4800 + sd)
            else:
                lnrs = [Learner([1.0], [0.1], C.assumed_params(t),
                                seed=5900 + sd + i)
                        for i, t in enumerate(TASKS)]
                resp = lambda k, s: lnrs[k].step(s, 0, y_star=int(s > 0),
                                                 feedback=True)
            rows, decls, confirmed, provisional = run_sessions_x(
                pol, filters, resp, n_sess, rng, lifecycle=True)
            fg_decl += int(bool(decls))
            fg_conf += int(bool(confirmed))
            pend += int(bool(provisional) and not confirmed)
            if scen == "wellspec":
                conf_any.append(min(confirmed.values(),
                                    default=n_sess * 40 + 1))
                conf_both.append(max(confirmed.values())
                                 if len(confirmed) == len(TASKS)
                                 else n_sess * 40 + 1)
        out[scen] = dict(decl=fg_decl, conf=fg_conf, pend=pend, n=n_seeds)
        line = (f"  {scen:<9}: declared {fg_decl}/{n_seeds}  CONFIRMED "
                f"{fg_conf}/{n_seeds}  (censored-provisional {pend})")
        if scen == "wellspec":
            cap = n_sess * 40
            out[scen].update(
                first_conf=float(np.median(conf_any)),
                rate_conf=float(np.mean([c <= cap for c in conf_any])),
                both_conf=float(np.median(conf_both)),
                rboth_conf=float(np.mean([c <= cap for c in conf_both])))
            line += (f"  first-CONF med {np.median(conf_any):.0f} (rate "
                     f"{np.mean([c <= cap for c in conf_any]):.0%}), "
                     f"both-CONF {np.median(conf_both):.0f} (rate "
                     f"{np.mean([c <= cap for c in conf_both]):.0%})")
        print(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--arms", default="x1,x2,x3,x4")
    args = ap.parse_args()
    t0 = time.time()
    bank = BankAdapter()
    n = args.seeds or (20 if not args.smoke else 4)
    todo = [a.strip() for a in args.arms.split(",")]
    path = os.path.join(FIG, "data_sharp_margin.npz")
    old = {}
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            old = {k: str(z[k][0]) for k in z.files}
    if "x1" in todo:
        old["armX1"] = json.dumps(arm_X1(bank, n, args.smoke))
    if "x2" in todo:
        old["armX2"] = json.dumps({f"{a}|{b}": v for (a, b), v
                                   in arm_X2(bank, n, args.smoke).items()})
    if "x3" in todo:
        old["armX3"] = json.dumps(arm_X3(bank, n, args.smoke))
    if "x4" in todo:
        old["armX4"] = json.dumps(arm_X4(bank, n, args.smoke))
    os.makedirs(FIG, exist_ok=True)
    np.savez(path, **{k: np.array([v]) for k, v in old.items()})
    print(f"\nsaved figures/data_sharp_margin.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
