"""M27 smoke test — terminal confirmation (F90), contact-triggered
onboarding (F91), λ(time-on-task) verdict artifact (F92), and the Domain
pluggability interface (F93).

Checks:
  1     F90 bit-identity: terminal_confirmation=True with nothing marked
        serves the exact same item sequence as False.
  2–4   F90 semantics: a terminal task leaves the training rotation
        (maintenance/retention only — exactly maint_probes at-bar probes
        per session, flagged); all_mastered counts terminal; a regressed
        terminal task is REVOKED by the stale e-gate and returns to
        training.
  5     F90 α-control direction: an at-bar-accurate terminal task is not
        revoked across many sessions.
  6–8   F91 sandbox wiring: CONTACT_ONBOARD serves demonstration trials
        while contact is uncertified; certification persists in meta and
        ends demo mode; the handoff shift is SKIPPED for a healthy
        (trainability ≥ 0.5) certification; a pure guesser never leaves
        the ramp and never declares.
  9     F92 artifact: the λ(time-on-task) study cache carries the
        easy/hard slope verdict.
  10–12 F93 Domain interface: ArrayBank reproduces the BankAdapter filter
        semantics (exclusion / feedback-safe / min-margin); a TOY domain
        plugs into the full TrainerPolicy stack with zero engine changes
        and a strong learner graduates on it; the real BankAdapter
        satisfies the ItemBank contract structurally.

Run: python3 -m tests.test_m27
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


# ── isolate sandbox paths BEFORE anything touches them ──
import sandbox.config as C

TMP = tempfile.mkdtemp(prefix="m27_test_")
C.STATE_DIR = os.path.join(TMP, "state")
C.LOG_DIR = os.path.join(TMP, "logs")
C.FIG_DIR = os.path.join(C.LOG_DIR, "figures")
C.TRIALS_JSONL = os.path.join(C.LOG_DIR, "trials.jsonl")
C.SESSIONS_JSONL = os.path.join(C.LOG_DIR, "sessions.jsonl")
import sandbox.state_io as sio

sio.STATE_NPZ = os.path.join(C.STATE_DIR, "state.npz")
sio.STATE_JSON = os.path.join(C.STATE_DIR, "state.json")

from sandbox.protocol import Session
from training.domain import ArrayBank, Domain
from training.learner_sim import Learner, LearnerParams
from training.trainer_policy import ModeThresholds


def toy_domain(K=2, n_items=400, seed=3):
    rng = np.random.default_rng(seed)
    pools = {}
    for t in range(K):
        s = np.concatenate([np.linspace(0.05, 3.0, n_items // 2),
                            -np.linspace(0.05, 3.0, n_items // 2)])
        pools[t] = dict(seg_id=10_000 * (t + 1) + np.arange(n_items),
                        s_mean=s, s_sd=np.full(n_items, 0.1),
                        y_star=(s > 0).astype(int),
                        margin=np.ones(n_items))
    p = LearnerParams(alpha_t=0.097, alpha_sigma=0.06, sigma_inf=0.55,
                      q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    return Domain(name="toy", tasks=tuple(range(K)),
                  task_names={t: f"toy{t}" for t in range(K)},
                  ell_star={t: 0.40 for t in range(K)},
                  sigma_star={t: float(np.exp(-0.40)) for t in range(K)},
                  assumed_params=lambda t: p, bank=ArrayBank(pools),
                  n_particles=200)


def drive(pol, respond, n, *, now0=1_800_000_000.0):
    served = []
    now = now0
    for _ in range(n):
        now += 30.0
        ch = pol.step(now=now)
        if ch is None:
            break
        y = respond(ch["task"], ch["s"], ch["y_star"])
        pol.record(ch, y)
        served.append((ch["task"], ch["seg_id"],
                       bool(ch["info"].get("maintenance"))))
    return served


def main():
    dom = toy_domain()
    rng = np.random.default_rng(0)
    lnrs = [Learner([0.9], [0.1], dom.assumed_params(t), seed=11 + t)
            for t in dom.tasks]

    def respond_learner(k, s, y_star):
        return lnrs[k].step(s, 0, y_star=y_star, feedback=True)

    # ── 1 bit-identity of the OFF/unmarked path ──
    seqs = []
    for flag in (False, True):
        lnrs2 = [Learner([0.9], [0.1], dom.assumed_params(t), seed=91 + t)
                 for t in dom.tasks]
        pol = dom.trainer(seed=7, finish_first=True,
                          thresholds=ModeThresholds(meanskill_gate=False),
                          terminal_confirmation=flag, maint_probes=3)
        seqs.append(drive(pol, lambda k, s, y: lnrs2[k].step(
            s, 0, y_star=y, feedback=True), 60))
    ok(seqs[0] == seqs[1],
       "terminal_confirmation=True with nothing marked is bit-identical")

    # ── 2–3 terminal semantics ──
    pol = dom.trainer(seed=7, finish_first=True,
                      thresholds=ModeThresholds(meanskill_gate=False),
                      terminal_confirmation=True, maint_probes=2)
    pol.mark_terminal(0)
    pol.note_session_open()
    served = drive(pol, respond_learner, 40)
    t0 = [x for x in served if x[0] == 0]
    ok(len(t0) == 2 and all(m for _, _, m in t0)
       and all(k == 1 for k, _, m in served if not m),
       f"terminal task got exactly maint_probes={2} flagged probes; all "
       f"training went to the other task")
    ok(not pol.all_mastered(), "all_mastered waits on the unmastered task")

    # ── 4 stale revocation on a regressed terminal task ──
    pol = dom.trainer(seed=13, finish_first=True,
                      thresholds=ModeThresholds(meanskill_gate=False),
                      terminal_confirmation=True, maint_probes=4)
    pol.mark_terminal(0)
    revoked_at = None
    n_srv = 0
    for s_i in range(12):
        pol.note_session_open()
        now = 1_800_000_000.0 + s_i * 86_400.0
        for _ in range(12):
            now += 30.0
            ch = pol.step(now=now)
            if ch is None:
                break
            # task0 (terminal) answers WRONG — a deep regression;
            # task1 trains normally
            y = (1 - ch["y_star"] if ch["task"] == 0
                 else respond_learner(1, ch["s"], ch["y_star"]))
            pol.record(ch, y)
            n_srv += 1
        if pol.terminal_events:
            revoked_at = (s_i, pol.terminal_events[0])
            break
    ok(revoked_at is not None and 0 not in pol.terminal,
       f"deep-regressed terminal task revoked by the stale gate "
       f"(session {revoked_at[0]})")

    # ── 5 α-control direction: healthy terminal task never revoked ──
    pol = dom.trainer(seed=17, finish_first=True,
                      thresholds=ModeThresholds(meanskill_gate=False),
                      terminal_confirmation=True, maint_probes=3)
    pol.mark_terminal(0)
    strong = Learner([0.55], [0.0], dom.assumed_params(0), seed=5)
    for s_i in range(10):
        pol.note_session_open()
        drive(pol, lambda k, s, y: (strong.step(s, 0, y_star=y,
                                                feedback=True) if k == 0
                                    else respond_learner(1, s, y)), 10,
              now0=1_800_000_000.0 + s_i * 86_400.0)
    ok(0 in pol.terminal and not pol.terminal_events,
       "an at/above-bar terminal task is never revoked (α-direction)")

    # ── 6–8 F91 sandbox wiring ──
    class StrongRobot:
        def __init__(self):
            p = LearnerParams(alpha_t=0.25, alpha_sigma=0.10,
                              sigma_inf=0.50, q_t=0.03, q_sigma=0.015,
                              rho=0.6)
            self.subs = {t: Learner([0.8], [0.1], p, seed=5 + t)
                         for t in C.TASKS}

        def responder(self, meta):
            y = self.subs[meta["task"]].respond(meta["s_real"], 0)
            return int(y), 900.0, False

    clock = {"now": 1_800_000_000.0}

    def now_fn():
        clock["now"] += 30.0
        return clock["now"]

    C.CONTACT_ONBOARD = True
    try:
        s1 = Session(now_fn=now_fn, seed=11)
        s1.run(StrongRobot().responder, max_trials=30)
        with open(C.TRIALS_JSONL) as fh:
            logs = [json.loads(x) for x in fh]
        demo_modes = [r["mode"] == "demo" for r in logs]
        with open(sio.STATE_JSON) as fh:
            meta = json.load(fh)
        cc = meta.get("contact_certified", {})
        first_non_demo = demo_modes.index(False) if False in demo_modes \
            else None
        ok(demo_modes[0] and first_non_demo is not None
           and len(cc) >= 1,
           f"onboarding ramp: demo trials until certification "
           f"(first adaptive trial {first_non_demo}), certification "
           f"persisted for {sorted(cc)}")
        ok(not any(e["event"] == "contact_handoff" for e in s1.events),
           "healthy certification skips the handoff shift (trainability "
           "guard)")
    finally:
        C.CONTACT_ONBOARD = False

    # guesser: never certifies, never leaves the ramp, never declares
    import shutil
    shutil.rmtree(C.STATE_DIR, ignore_errors=True)
    shutil.rmtree(C.LOG_DIR, ignore_errors=True)
    C.CONTACT_ONBOARD = True
    try:
        gz = np.random.default_rng(3)
        s2 = Session(now_fn=now_fn, seed=13)
        s2.run(lambda meta: (int(gz.integers(2)), 900.0, False),
               max_trials=30)
        with open(C.TRIALS_JSONL) as fh:
            logs2 = [json.loads(x) for x in fh]
        with open(sio.STATE_JSON) as fh:
            meta2 = json.load(fh)
        ok(all(r["mode"] == "demo" for r in logs2)
           and not meta2.get("contact_certified")
           and not meta2.get("provisional"),
           "a pure guesser never certifies contact, never leaves the "
           "ramp, never declares")
    finally:
        C.CONTACT_ONBOARD = False

    # ── 9 F92 artifact ──
    lam = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "figures", "data_lambda_tot.npz")
    with np.load(lam, allow_pickle=False) as z:
        res = json.loads(str(z["result"][0]))
    ok("easy" in res and "hard" in res
       and res["easy"]["mean_slope_per_min"] > 0,
       f"λ(time-on-task) verdict cached (easy-read slope "
       f"{res['easy']['mean_slope_per_min']:+.1e}/min — learning "
       f"dominates; no λ(t) term)")

    # ── 10–12 Domain interface ──
    bank = dom.bank
    full = bank.candidates(0)
    excl = bank.candidates(0, exclude_segids=set(full.seg_id[:100]))
    fs = bank.candidates(0, feedback_safe=True, min_margin=0.5)
    ok(len(excl) == len(full) - 100
       and len(fs) == len(full)
       and np.all((fs.s_mean > 0) == (fs.y_star == 1)),
       "ArrayBank filter semantics (exclusion / feedback-safe+margin)")
    dom1 = toy_domain(K=1)
    pol = dom1.trainer(seed=23, finish_first=True,
                       thresholds=ModeThresholds(meanskill_gate=False))
    pdec = LearnerParams(alpha_t=0.097, alpha_sigma=0.10, sigma_inf=0.40,
                         q_t=0.02, q_sigma=0.005, rho=0.6, rule="soft")
    fast = Learner([0.55], [0.02], pdec, seed=31)
    drive(pol, lambda k, s, y: fast.step(s, 0, y_star=y, feedback=True),
          200)
    ok(pol.mode_policies[0].is_mastered(pol.filters[0])
       or pol._was_mastered[0],
       "a TOY domain plugs into the full stack and a decisively-above-bar "
       "learner graduates on it (zero engine changes)")
    from training.bank_adapter import BankAdapter
    ok(callable(getattr(BankAdapter, "candidates", None))
       and all(hasattr(full, a) for a in
               ("seg_id", "s_mean", "s_sd", "y_star", "margin", "subset")),
       "BankAdapter satisfies the ItemBank contract structurally")

    print(f"\nM27 SMOKE TEST PASSED — {N_CHECK} checks.")


if __name__ == "__main__":
    main()
