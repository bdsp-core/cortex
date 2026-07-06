"""M20 — sandbox session CLI (the manual tester's entry point).

USAGE
  python3 -m sandbox.session                 # run one interactive session
  python3 -m sandbox.session --trials 20     # shorter sitting
  python3 -m sandbox.session --reset         # archive state+logs, start fresh
  python3 -m sandbox.session --status        # print state, no trials

  Robot self-test (used by the verification suite; also handy for demos):
  python3 -m sandbox.session --auto sigma=1.3,t=0.6,seed=3 --fake-gap 86400

Each trial shows a noisy trace; answer whether the marked window contains an
UPWARD deflection (the "target"): press 1/y = present, 0/n = absent,
q = quit (state is saved; the session resumes cleanly next time).
Veridical feedback follows every answer (that is the learning protocol).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from sandbox import config as C
from sandbox import state_io
from sandbox.protocol import Session
from training.bridge_conventions import engine_to_plan
from training.learner_sim import Learner, LearnerParams


def human_responder(meta):
    print(f"\n[{meta['task_name']}]  trial — is there an UPWARD deflection "
          f"in the marked window?")
    print("  " + meta["trace"])
    print("  " + meta["marker"])
    t0 = time.monotonic()
    while True:
        ans = input("  present? [1/y = yes, 0/n = no, q = quit] ").strip().lower()
        rt_ms = (time.monotonic() - t0) * 1000.0
        if ans in ("q", "quit"):
            return 0, rt_ms, True
        if ans in ("1", "y", "yes"):
            return 1, rt_ms, False
        if ans in ("0", "n", "no"):
            return 0, rt_ms, False
        print("  (unrecognized — 1/y, 0/n, or q)")


class Robot:
    """Simulated tester: a learner_sim.Learner per task, responding at the
    trial's latent s_real and learning from feedback. Robot state persists
    across sessions in sandbox/state/robot.npz so multi-session self-tests
    are meaningful."""

    def __init__(self, sigma0=1.3, t0=0.6, seed=1):
        import os
        self.PATH = C.STATE_DIR + "/robot.npz"   # after any set_user()
        self.subs = {}
        if os.path.exists(self.PATH):
            z = np.load(self.PATH)
            for t in C.TASKS:
                self.subs[t] = self._mk(float(z[f"s{t}"]), float(z[f"t{t}"]),
                                        seed + t)
        else:
            for t in C.TASKS:
                self.subs[t] = self._mk(sigma0, t0 * (1 if t % 2 else -1),
                                        seed + t)

    def _mk(self, sigma, t, seed):
        p = LearnerParams(alpha_t=0.2, alpha_sigma=0.06,
                          sigma_inf=0.82 * min(C.SIGMA_STAR.values()),
                          q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
        return Learner([sigma], [t], p, seed=seed)

    def responder(self, meta):
        t = meta["task"]
        lnr = self.subs[t]
        y = lnr.respond(meta["s_real"], 0)
        self._pending = (t, meta["s_real"], y)
        # M23 (F82): simulated RT must be physically plausible for an
        # ENGAGED responder (measured engaged windows are ≥ 1059 ms; the
        # RT-floor gate treats sustained sub-500 ms runs as non-perceptual)
        return int(y), 1200.0 + 800.0 * np.random.random(), False

    def learn(self, task, s_real, y_star):
        # replay the transition with the known label (respond() above only
        # sampled the answer; step() would re-sample — apply learning here)
        lnr = self.subs[task]
        p = lnr.p_yes(s_real, 0)
        delta = p - y_star
        lnr.t[0] += lnr.p.alpha_t * delta + lnr.rng.normal(0, lnr.p.q_t)
        w = lnr.skill_weight(s_real, 0)
        g = -w * (np.log(lnr.sigma[0]) - np.log(lnr.sigma_inf[0]))
        lnr.sigma[0] = float(np.exp(np.log(lnr.sigma[0])
                                    + lnr.p.alpha_sigma * g
                                    + lnr.rng.normal(0, lnr.p.q_sigma)))

    def save(self):
        np.savez(self.PATH, **{f"s{t}": self.subs[t].sigma[0]
                               for t in C.TASKS},
                 **{f"t{t}": self.subs[t].t[0] for t in C.TASKS})


def print_status():
    st = state_io.load_state()
    if st is None:
        print("sandbox: fresh (no state yet) — run a session to begin")
        return
    filters, gates, anchors, meta = st
    print(f"sandbox status — sessions completed: {meta['session_no']}, "
          f"trials: {meta['global_trial']}")
    for t in C.TASKS:
        sig_hat, t_hat = engine_to_plan(*filters[t].mean())
        pi, _ = filters[t].pass_mass(C.ELL_STAR[t])
        status = ("CONFIRMED" if str(t) in meta["confirmed"] else
                  "provisional" if str(t) in meta["provisional"] else
                  "training")
        print(f"  {C.TASK_NAMES[t]:>9}: ℓ̂ {filters[t].mean()[1]:+.2f} "
              f"(cut {C.ELL_STAR[t]:.2f}, π {pi:.2f})  t̂ {t_hat:+.2f}  "
              f"trainability {filters[t].trainability(C.ELL_STAR[t]):.2f}  "
              f"[{status}]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--auto", type=str, default=None,
                    help="robot mode: sigma=1.3,t=0.6,seed=1")
    ap.add_argument("--fake-gap", type=float, default=None,
                    help="robot mode: pretend this many seconds passed")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--user", type=str, default=None,
                    help="tester profile name — belief state and logs are "
                         "PER PERSON (M21/F72); every tester should pass "
                         "their own name every time")
    args = ap.parse_args(argv)
    if args.user:
        C.set_user(args.user)
    print(f"[profile: {C.USER or 'default'}]")
    if args.reset:
        dst, moved = state_io.archive_state()
        print(f"state archived to {dst} ({', '.join(moved) or 'nothing'})")
        return
    if args.status:
        print_status()
        return

    if args.auto:
        kv = dict(p.split("=") for p in args.auto.split(","))
        robot = Robot(sigma0=float(kv.get("sigma", 1.3)),
                      t0=float(kv.get("t", 0.6)),
                      seed=int(kv.get("seed", 1)))
        # fake clock: continue from persisted end + fake gap
        st = state_io.load_state()
        base = (float(st[3]["last_session_end"]) + (args.fake_gap or 0.0)
                if st and st[3]["last_session_end"] else time.time())
        clock = {"now": base}

        def now_fn():
            clock["now"] += 30.0          # ~30 s per trial
            return clock["now"]

        sess = Session(now_fn=now_fn)

        def responder(meta):
            return robot.responder(meta)

        # teach the robot after each logged trial via a thin shim
        orig = sess._one_trial

        def shim(task, mode, seg_id, s_mean, s_sd, y_star, _r):
            rec, y, q = orig(task, mode, seg_id, s_mean, s_sd, y_star,
                             responder)
            if not q:
                robot.learn(task, rec["s_real"], y_star)
            return rec, y, q

        sess._one_trial = shim
        n, events = sess.run(responder, max_trials=args.trials)
        robot.save()
        print(f"[robot] session {sess.session_no}: {n} trials; "
              f"events: {events}")
        print_status()
        return

    # ── interactive human session ──
    print(__doc__)
    sess = Session()

    def responder(meta):
        y, rt, q = human_responder(meta)
        return y, rt, q

    # feedback printer: wrap _one_trial to show correctness after logging
    orig = sess._one_trial

    def shim(task, mode, seg_id, s_mean, s_sd, y_star, _r):
        rec, y, q = orig(task, mode, seg_id, s_mean, s_sd, y_star, responder)
        if not q:
            mark = "✓ correct" if rec["correct"] else "✗ wrong"
            print(f"  → {mark}  (target was "
                  f"{'PRESENT' if y_star else 'ABSENT'})")
        return rec, y, q

    sess._one_trial = shim
    n, events = sess.run(responder, max_trials=args.trials)
    print(f"\nsession {sess.session_no} complete — {n} trials logged.")
    for e in events:
        print("  event:", e)
    print_status()
    print("\nrun `python3 -m sandbox.report` for the full analytics report.")


if __name__ == "__main__":
    main()
