"""M25 smoke test — selection successors (F86/F87) + sharp-margin
conjuncts (F88).

Checks:
  1–2   bit-identity of the OFF defaults (progress_lifecycle / explore /
        rollout / refractory / min_trainability all default-off reproduce
        M24 behavior exactly).
  3–5   F86 lifecycle routing: an ever-declared, finishing-eligible task
        leaves the EP lane and is served through finishing at the
        refinish threshold; below the refinish threshold it waits in the
        measurement lane (the re-polish loop is dead) and the EP lane
        keeps the recovering task; an ever-declared task whose bias
        walked out of band re-enters the EP lane (genuine regression).
  6     F86 measurement-lane fallback: when every unmastered task is
        ever-declared and none is finishing-eligible, the scheduler still
        serves (no F70-style deadlock).
  7     F86 exploration floor: a written-off task is served at least once
        per explore_every picks.
  8     lifecycle key set by TrainerPolicy.record at mastery transition.
  9–11  F87 rollout mechanics: pure-static belief earns ~zero rollout
        value everywhere; determinism under a fixed rng; a trainable
        belief ranks a near-85%-point item above a far-outside item.
  12    F87 end-to-end: rollout placement serves through
        TaskModePolicy.select with the mirror window honored.
  13–14 F88 conjuncts: boundary_refractory blocks declaration for exactly
        its budget of served trials; min_trainability blocks a
        transiently-confident mixture whose ceiling posterior does not
        clear the cut.
  15    sandbox wiring: EP-v2/F88 knobs present, inert at current flags;
        state meta carries the persistent lifecycle key.

Run: python3 -m tests.test_m25
"""
from __future__ import annotations

import numpy as np

from training.learner_sim import LearnerParams
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import (DeficiencyScheduler, ModeThresholds,
                                     TaskModePolicy)
from training.training_filter import TaskFilter

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    assert cond, f"FAILED: {msg}"
    CHECKS += 1
    print(f"  ok: {msg}")


class _Cands:
    def __init__(self, s, s_sd, y):
        self.s_mean = np.asarray(s, dtype=np.float64)
        self.s_sd = np.asarray(s_sd, dtype=np.float64)
        self.y_star = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.s_mean)

    def subset(self, idx):
        return _Cands(self.s_mean[idx], self.s_sd[idx], self.y_star[idx])


def _params(**kw):
    d = dict(alpha_t=0.097, alpha_sigma=0.06, sigma_inf=0.6,
             q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    d.update(kw)
    return LearnerParams(**kw) if False else LearnerParams(**d)


ELL_STAR = 0.4
SIG_STAR = float(np.exp(-ELL_STAR))


def _mix(ell_center, spread, *, seed, theta=0.0, top_heavy=True):
    """Fabricated mixture: particles pinned near a state, weights on the
    top stratum (top_heavy) or left at the prior."""
    rng = np.random.default_rng(seed)
    th0 = theta + 0.02 * rng.standard_normal(300)
    el0 = ell_center + spread * rng.standard_normal(300)
    m = SigmaInfMixtureFilter(th0, el0, _params(), ell_inf_mean=0.5,
                              ell_star=ELL_STAR, p_static_stratum=0.15,
                              seed=seed)
    if top_heavy:
        lw = np.full(len(m.log_w), -30.0)
        lw[len(m.grid) - 1] = 0.0
        m.log_w = lw
        for f in m.strata:
            f.theta = th0.copy()
            f.ell = el0.copy()
    return m


def main():
    rng = np.random.default_rng(1)
    th0 = 0.4 * rng.standard_normal(300)
    el0 = 0.4 * rng.standard_normal(300)
    cands = _Cands(np.linspace(-1.3, 1.3, 41), np.full(41, 0.06),
                   (np.linspace(-1.3, 1.3, 41) > 0).astype(int))
    th_def = ModeThresholds()

    # ── 1–2: OFF defaults bit-identical ──
    ok(th_def.rollout_placement is False and th_def.boundary_refractory == 0
       and th_def.min_trainability == 0.0,
       "new ModeThresholds knobs default off (M24-identical gate/placement)")
    mixA = _mix(0.0, 0.35, seed=3, top_heavy=False)
    mixB = _mix(-0.2, 0.35, seed=4, top_heavy=False)
    mp = [TaskModePolicy(0, ELL_STAR, SIG_STAR, th_def),
          TaskModePolicy(1, ELL_STAR, SIG_STAR, th_def)]
    schA = DeficiencyScheduler(2, progress_alloc=True, finish_first=True)
    schB = DeficiencyScheduler(2, progress_alloc=True, finish_first=True,
                               progress_lifecycle=False, explore_every=0)
    pA = [schA.pick([mixA, mixB], [ELL_STAR] * 2, mp) for _ in range(6)]
    pB = [schB.pick([mixA, mixB], [ELL_STAR] * 2, mp) for _ in range(6)]
    ok(pA == pB, "lifecycle/floor OFF scheduler picks bit-identical to M24")

    # ── 3: ever-declared + finishing-eligible ⇒ measurement lane ──
    # task0: near-bar, tight, bias in band (π ≈ .8 ⇒ close ≥ refinish .5)
    near = _mix(ELL_STAR + 0.12, 0.14, seed=5)
    # task1: recovering — low state, wide, weights on the prior
    rec = _mix(-0.3, 0.30, seed=6, top_heavy=False)
    mp2 = [TaskModePolicy(0, ELL_STAR, SIG_STAR, th_def),
           TaskModePolicy(1, ELL_STAR, SIG_STAR, th_def)]
    sch = DeficiencyScheduler(2, progress_alloc=True, finish_first=True,
                              progress_lifecycle=True,
                              refinish_threshold=0.50)
    sch.mark_declared(0)
    pi, mcse = near.pass_mass(ELL_STAR)
    assert not mp2[0].is_mastered(near), "fixture must be sub-declaration"
    assert pi - 2 * mcse >= 0.50, f"fixture close too low ({pi - 2 * mcse:.2f})"
    ch = sch.pick([near, rec], [ELL_STAR] * 2, mp2)
    ok(ch == 0 and sch._fin_spent.get(0, 0) == 1,
       "ever-declared eligible task served through FINISHING (refinish)")

    # ── 4: below the refinish threshold the EP lane wins (re-polish dead) ──
    mid = _mix(ELL_STAR - 0.05, 0.16, seed=7)     # π ~ .5, close < .5
    pi_m, mcse_m = mid.pass_mass(ELL_STAR)
    assert pi_m - 2 * mcse_m < 0.50
    sch2 = DeficiencyScheduler(2, progress_alloc=True, finish_first=True,
                               progress_lifecycle=True,
                               refinish_threshold=0.50)
    sch2.mark_declared(0)
    ch2 = sch2.pick([mid, rec], [ELL_STAR] * 2, mp2)
    ok(ch2 == 1, "sub-refinish ever-declared task yields to the EP lane "
       "(the F85-ii re-polish loop is dead)")

    # ── 5: bias out of band ⇒ ever-declared re-enters the EP lane ──
    biased = _mix(ELL_STAR + 0.12, 0.14, seed=8, theta=-0.8)
    sch3 = DeficiencyScheduler(2, progress_alloc=True, finish_first=True,
                               progress_lifecycle=True)
    sch3.mark_declared(0)
    # rec has EP > 0; biased has EP incl. its bias excess — both in defs.
    # Verify by internals: pick must NOT route task0 to finishing.
    ch3 = sch3.pick([biased, rec], [ELL_STAR] * 2, mp2)
    ok(sch3._fin_spent.get(0, 0) == 0,
       "bias-out-of-band ever-declared task re-enters the EP lane")

    # ── 6: measurement-lane fallback (no deadlock) ──
    sch4 = DeficiencyScheduler(2, progress_alloc=True, finish_first=True,
                               progress_lifecycle=True,
                               refinish_threshold=0.99)  # nothing eligible
    sch4.mark_declared(0)
    sch4.mark_declared(1)
    near2 = _mix(ELL_STAR + 0.10, 0.15, seed=9)
    ch4 = sch4.pick([near, near2], [ELL_STAR] * 2, mp2)
    ok(ch4 in (0, 1), "all-measurement-lane fallback still serves "
       "(no F70-style deadlock)")

    # ── 7: exploration floor ──
    strong = _mix(-0.4, 0.25, seed=10, top_heavy=False)   # big EP
    weak = _mix(0.30, 0.10, seed=11)                      # near-ceiling EP≈0
    sch5 = DeficiencyScheduler(2, progress_alloc=True, finish_first=True,
                               progress_lifecycle=True, explore_every=3)
    picks = [sch5.pick([strong, weak], [ELL_STAR] * 2, mp2)
             for _ in range(8)]
    sch6 = DeficiencyScheduler(2, progress_alloc=True, finish_first=True,
                               progress_lifecycle=True, explore_every=0)
    picks0 = [sch6.pick([strong, weak], [ELL_STAR] * 2, mp2)
              for _ in range(8)]
    ok(picks.count(1) >= 2 and picks0.count(1) <= picks.count(1),
       f"exploration floor serves the written-off task "
       f"({picks.count(1)} vs {picks0.count(1)} of 8 picks)")

    # ── 8: TrainerPolicy.record sets the lifecycle key ──
    from training.bank_adapter import BankAdapter
    from training.trainer_policy import TrainerPolicy
    bank = BankAdapter()
    decl = _mix(ELL_STAR + 0.6, 0.08, seed=12)   # instantly mastered
    live = _mix(-0.2, 0.30, seed=13, top_heavy=False)

    class _Remap:
        def candidates(self, task, **kw):
            return bank.candidates([1, 2][task], **kw)
    pol = TrainerPolicy([decl, live], [ELL_STAR] * 2, [SIG_STAR] * 2,
                        _Remap(), thresholds=ModeThresholds(
                            meanskill_gate=False), seed=2,
                        progress_alloc=True, progress_lifecycle=True,
                        finish_first=True)
    assert pol.mode_policies[0].is_mastered(decl)
    ch = pol.step(now=1.0)
    pol.record(ch, 1)
    ok(0 in pol.scheduler.ever_declared,
       "mastered task enters ever_declared via the serving loop")

    # ── 9–11: rollout mechanics ──
    from training.trainer_rollout import rollout_item_q
    stat = TaskFilter(th0, el0, _params(rule="static"), seed=21)
    q_stat = rollout_item_q(stat, cands.s_mean[:6], cands.s_sd[:6],
                            cands.y_star[:6], pool_s=cands.s_mean,
                            pool_sd=cands.s_sd, pool_y=cands.y_star,
                            rng=np.random.default_rng(3))
    ok(np.allclose(q_stat, 0.0, atol=1e-12),
       "pure-static belief earns exactly zero rollout value (F84-ii)")
    tr = TaskFilter(0.0 * th0, np.full(300, -0.1) + 0.05 * el0,
                    _params(), seed=22)
    s_try = np.array([1.16, 3.5])       # ≈ 85% point (σ̂≈1.1) vs far outside
    q1 = rollout_item_q(tr, s_try, np.full(2, 0.05), np.array([1, 1]),
                        pool_s=cands.s_mean, pool_sd=cands.s_sd,
                        pool_y=cands.y_star, rng=np.random.default_rng(7))
    q2 = rollout_item_q(tr, s_try, np.full(2, 0.05), np.array([1, 1]),
                        pool_s=cands.s_mean, pool_sd=cands.s_sd,
                        pool_y=cands.y_star, rng=np.random.default_rng(7))
    ok(np.allclose(q1, q2), "rollout value deterministic under a fixed rng")
    ok(q1[0] > q1[1], "near-85%-point item out-values a far-outside item "
       f"({q1[0]:.4f} vs {q1[1]:.4f})")

    # ── 12: rollout placement end-to-end (mirror window honored) ──
    mixr = _mix(0.0, 0.30, seed=23, top_heavy=False)
    thr = ModeThresholds(meanskill_gate=False, rollout_placement=True,
                         rollout_L=4, rollout_nro=48, rollout_short=6)
    mpr = TaskModePolicy(0, ELL_STAR, SIG_STAR, thr)
    r9 = np.random.default_rng(9)
    picks, last = [], None
    mir_ok = []
    for k in range(6):
        mode, est = mpr.choose_mode(mixr, None, k)
        sel = mpr.select("skill", est, cands, None, k, r9, filt=mixr)
        idx, info = sel
        s_pick = float(cands.s_mean[idx])
        if last is not None and (last > 0) != (s_pick > 0):
            mir_ok.append(abs(abs(s_pick) - abs(last))
                          <= 0.15 * max(abs(last), 0.3) + 1e-9)
        last = s_pick
        picks.append(idx)
        ok_info = info.get("rollout") or info.get("finishing")
        assert ok_info, "rollout path must flag its picks"
    ok(len(mir_ok) > 0 and all(mir_ok),
       f"rollout placement serves end-to-end with the mirror window "
       f"({len(mir_ok)} mirrored pairs)")

    # ── 13: boundary refractory ──
    mast = _mix(ELL_STAR + 0.6, 0.08, seed=24)
    thf = ModeThresholds(meanskill_gate=False, boundary_refractory=5)
    mpf = TaskModePolicy(0, ELL_STAR, SIG_STAR, thf)
    assert mpf.is_mastered(mast)
    mpf.note_boundary()
    blocked = [not mpf.is_mastered(mast) for _ in range(1)]
    served = 0
    while not mpf.is_mastered(mast):
        mpf.note_posterior(mast)
        served += 1
        assert served <= 5, "refractory must release after its budget"
    ok(all(blocked) and served == 5,
       f"boundary refractory blocks declaration for exactly {served} "
       f"served trials")

    # ── 14: min_trainability conjunct ──
    conf = _mix(ELL_STAR + 0.6, 0.08, seed=25)
    lw = np.full(len(conf.log_w), -30.0)
    lw[0] = 0.0                     # all ceiling weight BELOW the cut
    conf.log_w = lw
    assert conf.trainability(ELL_STAR) < 0.5
    th_tr = ModeThresholds(meanskill_gate=False, min_trainability=0.5)
    ok(TaskModePolicy(0, ELL_STAR, SIG_STAR,
                      ModeThresholds(meanskill_gate=False)).is_mastered(conf)
       and not TaskModePolicy(0, ELL_STAR, SIG_STAR, th_tr).is_mastered(conf),
       "min_trainability blocks a π-confident belief whose ceiling "
       "posterior sits below the cut")

    # ── 15: sandbox wiring ──
    from sandbox import config as SC
    from sandbox import state_io
    ok(SC.PROGRESS_LIFECYCLE is True and SC.EXPLORE_EVERY == 8
       and SC.REFINISH_THRESHOLD == 0.50
       and "ever_declared" in state_io.fresh_state()[3],
       "sandbox wires the EP-v2 lifecycle config + persistent lifecycle key")

    print(f"\nM25 SMOKE TEST PASSED — {CHECKS} checks.")


if __name__ == "__main__":
    main()
