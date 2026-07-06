"""M26 smoke test — evidence-adaptive declared-task boundary hazard (F89).

Checks:
  1–3   declared_hazard_scale math: n=0 is exactly 1.0 (bit-identical to
        the pinned M23 hazard), Beta posterior-mean form c0/(c0+n) +
        monotonicity, floor + geometric ablation.
  4–7   AdaptiveBoundaryHazard lifecycle: never-declared tasks always
        scale 1.0 (the F80 breakout machinery untouched); a boundary's
        pending debt is credited at most ONCE; revocation resets the
        count to full hazard; π-credit rule (≥1−α credits, indifference
        band is inert, collapse resets).
  8     json round-trip (sandbox meta persistence, no pickle).
  9     bit-identity of the ×1.0 path: boundary_shift(ε·1.0, …, ρ·1.0)
        equals boundary_shift(ε, …, ρ) exactly on a real mixture.
  10–11 a decayed shift injects less: fewer shocked particles (matched
        rng streams — the ε-subset property) and a 4×-smaller fixed-share
        weight movement toward the prior at scale 0.25.
  12–13 sandbox wiring: config knobs present with HAZARD_ADAPT default
        False; a session under HAZARD_ADAPT=True with a seeded lifecycle
        applies the scaled Gate 4 (hazard_scale event at the Beta value)
        and persists the lifecycle in meta.

Run: python3 -m tests.test_m26
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

N_CHECK = 0


def ok(cond, msg):
    global N_CHECK
    N_CHECK += 1
    assert cond, f"CHECK {N_CHECK} FAILED: {msg}"
    print(f"  ok: {msg}")


# ── isolate all sandbox paths BEFORE anything touches them ──
import sandbox.config as C

TMP = tempfile.mkdtemp(prefix="m26_test_")
C.STATE_DIR = os.path.join(TMP, "state")
C.LOG_DIR = os.path.join(TMP, "logs")
C.FIG_DIR = os.path.join(C.LOG_DIR, "figures")
C.TRIALS_JSONL = os.path.join(C.LOG_DIR, "trials.jsonl")
C.SESSIONS_JSONL = os.path.join(C.LOG_DIR, "sessions.jsonl")
import sandbox.state_io as sio

sio.STATE_NPZ = os.path.join(C.STATE_DIR, "state.npz")
sio.STATE_JSON = os.path.join(C.STATE_DIR, "state.json")
# M27: isolate the hazard mechanics under test from the D45 onboarding
# ramp (demo-first serving leaves an 8-trial belief's π below the
# indifference band, which correctly collapse-resets the artificially
# seeded lifecycle) — the ramp has its own pins in tests/test_m27.
C.CONTACT_ONBOARD = False

from sandbox.protocol import Session
from training.learner_sim import Learner, LearnerParams
from training.mixture_filter import (AdaptiveBoundaryHazard,
                                     SigmaInfMixtureFilter,
                                     declared_hazard_scale)


def main():
    # ── 1–3 scale math ──
    ok(declared_hazard_scale(0) == 1.0
       and declared_hazard_scale(0, geometric=0.5) == 1.0,
       "n=0 scale is exactly 1.0 in both forms (bit-identical M23 hazard)")
    s = [declared_hazard_scale(n, strength=2.0) for n in range(5)]
    ok(all(abs(s[n] - 2.0 / (2.0 + n)) < 1e-12 for n in range(5))
       and all(s[i + 1] < s[i] for i in range(4)),
       f"Beta posterior-mean form c0/(c0+n), monotone ({np.round(s, 3)})")
    ok(declared_hazard_scale(50, strength=2.0, floor=0.25) == 0.25
       and abs(declared_hazard_scale(2, geometric=0.5, floor=0.1) - 0.25)
       < 1e-12,
       "floor respected; geometric ablation λ^n")

    # ── 4–7 lifecycle semantics ──
    h = AdaptiveBoundaryHazard(2, strength=2.0)
    for _ in range(5):
        h.on_boundary(0)
    ok(h.scale(0) == 1.0 and not h.pending[0],
       "never-declared task: boundaries leave scale at 1.0 (F80 untouched)")
    h.on_gate_pass(0)
    h.on_boundary(0)
    h.on_gate_pass(0)
    h.on_gate_pass(0)          # same boundary — must not double-credit
    ok(h.n[0] == 1 and h.scale(0) == 2.0 / 3.0,
       "one boundary's debt credits exactly once (n=1, scale 2/3)")
    h.on_revoked(0)
    ok(h.n[0] == 0 and h.scale(0) == 1.0 and h.ever[0],
       "revocation resets to the full pinned hazard; `ever` survives")
    h2 = AdaptiveBoundaryHazard(1, strength=2.0)
    h2.credit_from_pi(0, 0.99)           # not ever-declared: inert
    h2.on_gate_pass(0)
    h2.on_boundary(0)
    h2.credit_from_pi(0, 0.80)           # indifference band: inert
    n_mid = h2.n[0]
    h2.credit_from_pi(0, 0.97)           # credit
    h2.on_boundary(0)
    h2.credit_from_pi(0, 0.30)           # collapse: reset
    ok(n_mid == 0 and h2.n[0] == 0 and not h2.pending[0]
       and h2.scale(0) == 1.0,
       "π-credit: ≥1−α credits, band is inert, collapse resets")

    # ── 8 json round-trip ──
    h3 = AdaptiveBoundaryHazard.from_json(
        {"ever": [True, False], "n": [3, 0], "pending": [True, False]},
        2, strength=5.0, floor=0.25)
    ok(h3.to_json() == {"ever": [True, False], "n": [3, 0],
                        "pending": [True, False]}
       and abs(h3.scale(0) - 0.625) < 1e-12 and h3.scale(1) == 1.0,
       "json round-trip; scale(n=3, c0=5) = 0.625")

    # ── 9 bit-identity of the ×1.0 path ──
    def mkmix(seed):
        rng = np.random.default_rng(seed)
        th0 = 0.45 * rng.standard_normal(200)
        el0 = 0.45 * rng.standard_normal(200)
        p = LearnerParams(alpha_t=0.097, alpha_sigma=0.06, sigma_inf=0.6,
                          q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
        return SigmaInfMixtureFilter(th0, el0, p, ell_inf_mean=0.7,
                                     ell_star=0.4, p_static_stratum=0.15,
                                     seed=seed)

    fa, fb = mkmix(7), mkmix(7)
    for f in (fa, fb):
        for i in range(12):
            f.step(0.4 * (-1) ** i, i % 2, i % 2, s_sd=0.5)
    fa.boundary_shift(0.10, 0.75, 0.33, w_share=0.10)
    fb.boundary_shift(0.10 * 1.0, 0.75, 0.33, w_share=0.10 * 1.0)
    ok(np.array_equal(fa.log_w, fb.log_w)
       and all(np.array_equal(a.ell, b.ell)
               and np.array_equal(a.theta, b.theta)
               for a, b in zip(fa.strata, fb.strata)),
       "scale ×1.0 boundary_shift is bit-identical to the pinned call")

    # ── 10–11 a decayed shift injects less ──
    fc, fd = mkmix(11), mkmix(11)
    for f in (fc, fd):
        for i in range(12):
            f.step(0.4 * (-1) ** i, i % 2, i % 2, s_sd=0.5)
    ell_c0 = [a.ell.copy() for a in fc.strata]
    ell_d0 = [a.ell.copy() for a in fd.strata]
    w_pre = fc.weights().copy()
    fc.boundary_shift(0.10, 0.75, 0.33, w_share=0.10)
    fd.boundary_shift(0.10 * 0.25, 0.75, 0.33, w_share=0.10 * 0.25)
    moved_c = sum(int((a.ell != e0).sum())
                  for a, e0 in zip(fc.strata, ell_c0))
    moved_d = sum(int((a.ell != e0).sum())
                  for a, e0 in zip(fd.strata, ell_d0))
    ok(moved_d < moved_c,
       f"scale 0.25 shocks fewer particles ({moved_d} < {moved_c}, "
       f"matched rng streams)")
    mv_c = float(np.abs(fc.weights() - w_pre).sum())
    mv_d = float(np.abs(fd.weights() - w_pre).sum())
    ok(abs(mv_d - 0.25 * mv_c) < 1e-9,
       f"fixed-share weight movement is exactly linear in the scale "
       f"({mv_d:.4f} = 0.25×{mv_c:.4f})")

    # ── 12–13 sandbox wiring ──
    ok(hasattr(C, "HAZARD_ADAPT") and C.HAZARD_ADAPT is False
       and C.HAZARD_STRENGTH > 0 and 0.0 <= C.HAZARD_FLOOR < 1.0,
       "config knobs present; HAZARD_ADAPT default False (bit-identical)")

    class Robot:
        def __init__(self, seed=5):
            p = LearnerParams(alpha_t=0.25, alpha_sigma=0.10,
                              sigma_inf=0.55, q_t=0.03, q_sigma=0.015,
                              rho=0.6)
            self.subs = {t: Learner([1.1], [0.4], p, seed=seed + t)
                         for t in C.TASKS}

        def responder(self, meta):
            y = self.subs[meta["task"]].respond(meta["s_real"], 0)
            return int(y), 900.0, False

    clock = {"now": 1_800_000_000.0}

    def now_fn():
        clock["now"] += 30.0
        return clock["now"]

    robot = Robot()
    s1 = Session(now_fn=now_fn, seed=11)
    s1.run(robot.responder, max_trials=8)
    with open(sio.STATE_JSON) as fh:
        meta = json.load(fh)
    ok(meta.get("hazard_lifecycle", {}).get("ever") == [False, False],
       "fresh lifecycle persisted in meta (no declarations yet)")
    # seed an ever-declared lifecycle and run one boundary under the flag
    meta["hazard_lifecycle"] = {"ever": [True, False], "n": [3, 0],
                                "pending": [False, False]}
    meta["ever_declared"] = [str(C.TASKS[0])]
    with open(sio.STATE_JSON, "w") as fh:
        json.dump(meta, fh)
    clock["now"] += 3600.0               # same sitting < MIN_GAP_S
    C.HAZARD_ADAPT = True
    try:
        s2 = Session(now_fn=now_fn, seed=13)
        s2.run(robot.responder, max_trials=6)
        evs = [e for e in s2.events if e["event"] == "hazard_scale"]
        want = declared_hazard_scale(3, strength=C.HAZARD_STRENGTH,
                                     floor=C.HAZARD_FLOOR)
        ok(len(evs) == 1 and evs[0]["task"] == C.TASKS[0]
           and abs(evs[0]["scale"] - round(want, 3)) < 1e-9,
           f"HAZARD_ADAPT session applies the scaled Gate 4 to the "
           f"declared task only (scale {evs[0]['scale']})")
    finally:
        C.HAZARD_ADAPT = False

    print(f"\nM26 SMOKE TEST PASSED — {N_CHECK} checks.")


if __name__ == "__main__":
    main()
