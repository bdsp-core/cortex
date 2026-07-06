"""M27 — contact-triggered onboarding (F91): the M23-leftover protocol
response to the F83 contact e-process (docs/M25_SELECTION_V2.md §6
item 3; the M24 audit's remaining known value sink — pre-contact
sessions delivered w_true 0.15–0.17).

Two separable components, priced separately (arms):

  DEMO ramp: until a task has EVER certified contact, serve the easiest
     available label-alternating items instead of adaptive placement.
     With EXOGENOUS contact timing (the only honest in-silico model —
     assuming demonstrations accelerate concept acquisition would build
     the conclusion into the simulation), the ramp's measurable effect
     is on the BELIEF (what evidence the guessing phase writes into it)
     and the served-item stream; its pedagogical value is explicitly a
     field-check question.

  HANDOFF shift: at the first-ever certification, the task's belief
     takes one Gate-4 boundary_shift — contact IS the regime shift the
     M23 hazard models (USER-E's d′ 0.8→3.0 breakout), triggered by an
     anytime-valid e-process instead of waiting for a sitting boundary.
     Reuses validated machinery; α=0.05/session false-trigger rate is
     the F83-validated bound (0 false contacts in 400 null sessions).

Arms {base, shift, demo, demoshift} × scenarios {contact60 (no contact
for 60 trials, then engaged σ=0.9 — the M25 arm-R censored-at-30%
scenario), wellspec (no-harm check: certification in ~10 trials bounds
the demo cost), careless (weak-contact responder — certifies, then the
usual guards own it; FG must stay 0)}. K=1, M23 stack, real bank.

Run:  python3 -m studies.study_m27_onboard [--smoke]
Out:  figures/data_m27_onboard.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import norm

from sandbox import config as C
from sandbox.contact import ContactMonitor
from training.bank_adapter import BankAdapter
from training.bridge_conventions import engine_to_plan
from training.learner_sim import Learner
from training.trainer_policy import ModeThresholds, TrainerPolicy
from studies.study_m24_selection import (LAPSE, _Remap, first_declared,
                                         fresh_mixture, trunc_draw)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(FIG, "data_m27_onboard.npz")

TASK = 1
ARMS = {"base": (False, False), "shift": (False, True),
        "demo": (True, False), "demoshift": (True, True)}


def make_policy_o(bank, seed):
    filters = [fresh_mixture(TASK, seed + 1)]
    th = ModeThresholds(meanskill_gate=False, sd_floor=C.SD_FLOOR,
                        skill_sigma_z=1.0)
    pol = TrainerPolicy(
        filters, [C.ELL_STAR[TASK]], [C.SIGMA_STAR[TASK]],
        _Remap(bank, [TASK]), thresholds=th, seed=seed + 5,
        probe_every=C.PROBE_EVERY, finish_first=True,
        trainability_floor=C.TRAINABILITY_FLOOR,
        finish_sd_tol=C.FINISH_SD_TOL)
    return pol, filters


def _demo_pick(pol, flip):
    pool = pol.bank.candidates(0, exclude_segids=pol.exclude | pol._served,
                               feedback_safe=True,
                               min_margin=pol.min_margin)
    if len(pool) == 0:
        return None
    want = 1 if flip > 0 else 0
    m = np.where(pool.y_star == want)[0]
    if m.size == 0:
        m = np.arange(len(pool))
    sign = 1.0 if want == 1 else -1.0
    sc = sign * pool.s_mean[m] - pool.s_sd[m]
    j = int(m[np.argmax(sc)])
    return (int(pool.seg_id[j]), float(pool.s_mean[j]),
            float(pool.s_sd[j]), int(pool.y_star[j]))


def run_sessions_o(pol, filters, respond, n_sessions, rng, *,
                   demo=False, handoff=False, trials_per=40):
    """K=1 lifecycle harness mirroring the sandbox F91 protocol: per-
    session ContactMonitor (shadow in base), demo ramp while never-
    certified, one Gate-4 shift at first-ever certification."""
    rows, certified_at, flip = [], None, 1
    for s_i in range(n_sessions):
        if s_i > 0:
            filters[0].boundary_shift(*C.BOUNDARY_JUMP,
                                      w_share=C.BOUNDARY_WSHARE)
            pol.mode_policies[0].note_boundary()
        pol.suspended = set()
        cm = ContactMonitor(alpha=C.CONTACT_ALPHA)
        served = 0
        now = 1_800_000_000.0 + s_i * 86_400.0
        while served < trials_per:
            now += 30.0
            is_demo = demo and certified_at is None
            if is_demo:
                pk = _demo_pick(pol, flip)
                if pk is None:
                    break
                flip *= -1
                seg, s, s_sd, y_star = pk
                s_real = trunc_draw(rng, s, s_sd, y_star)
                y = int(respond(s_real))
                filters[0].step(s, y, y_star, s_sd=s_sd)
                pol.exclude.add(seg)
            else:
                ch = pol.step(now=now)
                if ch is None:
                    break
                s, s_sd, y_star = ch["s"], ch["s_sd"], ch["y_star"]
                s_real = trunc_draw(rng, s, s_sd, y_star)
                y = int(respond(s_real))
                pol.record(ch, y)
            served += 1
            cont = cm.update(int(y == y_star))
            if cont["contact"] and certified_at is None:
                certified_at = len(rows)
                # handoff fires only on the poisoning signature (the F80
                # mechanism it exists to cure) — an early healthy
                # certification must not pay a widening tax
                if (handoff and filters[0].trainability(C.ELL_STAR[TASK])
                        < 0.5):
                    filters[0].boundary_shift(*C.BOUNDARY_JUMP,
                                              w_share=C.BOUNDARY_WSHARE)
                    pol.mode_policies[0].note_boundary()
            rows.append({"session": s_i, "demo": is_demo,
                         "correct": int(y == y_star),
                         "mastered": [bool(pol.mode_policies[0]
                                           .is_mastered(filters[0]))]})
    return rows, certified_at


def scen_contact60(seed):
    st = {"n": 0}
    rz = np.random.default_rng(seed)

    def respond(s):
        st["n"] += 1
        sig = 5.0 if st["n"] <= 60 else 0.9
        p = LAPSE + (1 - 2 * LAPSE) * norm.cdf((s - 0.0) / sig)
        return int(rz.random() < p)
    return respond, 60


def scen_wellspec(seed):
    lnr = Learner([1.0], [0.1], C.assumed_params(TASK), seed=seed)
    return (lambda s: lnr.step(s, 0, y_star=int(s > 0), feedback=True), 0)


def scen_careless(seed):
    rz = np.random.default_rng(seed)

    def respond(s):
        p = 0.35 + 0.30 * norm.cdf((s - 0.1) / float(C.SIGMA_STAR[TASK]))
        return int(rz.random() < p)
    return respond, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    bank = BankAdapter()
    n_seeds = args.seeds or (20 if not args.smoke else 3)
    n_sess = 6 if not args.smoke else 3
    cap = n_sess * 40
    scens = {"contact60": scen_contact60, "wellspec": scen_wellspec,
             "careless": scen_careless}
    out = {}
    for sname, mk in scens.items():
        for aname, (demo, handoff) in ARMS.items():
            decl, cert, ndemo, acc = [], [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(100 + sd)
                pol, filters = make_policy_o(bank, 200 + 37 * sd)
                respond, _ = mk(9000 + sd)
                rows, cat = run_sessions_o(pol, filters, respond, n_sess,
                                           rng, demo=demo, handoff=handoff)
                d = first_declared(rows, 0)
                decl.append(d if d is not None else cap + 1)
                cert.append(cat if cat is not None else cap + 1)
                ndemo.append(sum(1 for r in rows if r["demo"]))
                acc.append(np.mean([r["correct"] for r in rows]))
            out[(sname, aname)] = dict(
                decl=float(np.median(decl)),
                rate=float(np.mean([d <= cap for d in decl])),
                cert=float(np.median(cert)),
                ndemo=float(np.mean(ndemo)),
                acc=float(np.mean(acc)))
            o = out[(sname, aname)]
            print(f"  {sname:<9} [{aname:>9}]: declared {o['decl']:5.0f} "
                  f"(rate {o['rate']:.0%})  certified @{o['cert']:5.0f}  "
                  f"demo trials {o['ndemo']:5.1f}  acc {o['acc']:.2f}")
    os.makedirs(FIG, exist_ok=True)
    old = {}
    if os.path.exists(OUT):
        with np.load(OUT, allow_pickle=False) as z:
            old = {k: str(z[k][0]) for k in z.files}
    old["armO"] = json.dumps({f"{a}|{b}": v for (a, b), v in out.items()})
    np.savez(OUT, **{k: np.array([v]) for k, v in old.items()})
    print(f"saved {OUT}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
