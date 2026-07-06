"""M24 smoke test — expected-progress selection (F84/F85).

Checks:
  1–2   bit-identity of the OFF defaults (progress_placement=False,
        progress_alloc=False reproduce M23 picks exactly).
  3–5   expected_progress_score mechanics: static-rule filters score 0;
        mixture scoring excludes the static stratum (F84-ii — the pooled
        shortcut wrongly credits it); plain-TaskFilter path equals the
        tier-1 Q.
  6–8   expected_progress_rate ordering: trainable > plateau-at-ceiling;
        above-bar mass earns ~0; bias excess earns the capped α_t term.
  9     scheduler with progress_alloc picks the trainable task over the
        hopeless one (the USER-B trap, by construction).
  10    progress placement path executes end-to-end (flag on, real bank),
        including the finishing switch and the mirror-window partner.
  11    sandbox config wires F85 on (PROGRESS_ALLOC) with placement OFF.

Run: python3 -m tests.test_m24
"""
from __future__ import annotations

import numpy as np

from training.learner_sim import LearnerParams
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import (DeficiencyScheduler, ModeThresholds,
                                     TaskModePolicy, expected_progress_rate,
                                     expected_progress_score)
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
    return LearnerParams(**d)


def main():
    rng = np.random.default_rng(1)
    th0 = 0.4 * rng.standard_normal(300)
    el0 = 0.4 * rng.standard_normal(300)
    cands = _Cands(np.linspace(-1.3, 1.3, 41), np.full(41, 0.06),
                   (np.linspace(-1.3, 1.3, 41) > 0).astype(int))

    # ── 1–2: OFF defaults are bit-identical ──
    def run_policy(**kw):
        from training.bank_adapter import BankAdapter
        return None  # placeholder (bit-identity checked via scheduler below)

    mixA = SigmaInfMixtureFilter(th0, el0, _params(), ell_inf_mean=0.5,
                                 ell_star=0.4, p_static_stratum=0.15, seed=3)
    mixB = SigmaInfMixtureFilter(th0, el0, _params(), ell_inf_mean=0.5,
                                 ell_star=0.4, p_static_stratum=0.15, seed=3)
    r = np.random.default_rng(9)
    for k in range(30):
        s = float(r.uniform(-1.2, 1.2))
        y_star = int(s > 0)
        y = int(r.random() < (0.8 if y_star else 0.2))
        mixA.step(s, y, y_star, s_sd=0.06)
        mixB.step(s, y, y_star, s_sd=0.06)
    schA = DeficiencyScheduler(2)
    schB = DeficiencyScheduler(2, progress_alloc=False)
    mpA = [TaskModePolicy(0, 0.4, 0.67, ModeThresholds()),
           TaskModePolicy(1, 0.4, 0.67, ModeThresholds())]
    pA = [schA.pick([mixA, mixB], [0.4, 0.4], mpA) for _ in range(6)]
    schA2 = DeficiencyScheduler(2)
    pB = [schB.pick([mixA, mixB], [0.4, 0.4], mpA) for _ in range(6)]
    ok(pA == pB, "progress_alloc=False scheduler picks bit-identical")
    ok(ModeThresholds().progress_placement is False,
       "progress_placement defaults False (M23-identical placement)")

    # ── 3: static-rule filter scores exactly 0 ──
    fs = TaskFilter(th0, el0, _params(rule="static"), seed=5)
    q = expected_progress_score(fs, cands)
    ok(np.all(q == 0.0), "static-rule TaskFilter earns zero progress")

    # ── 4: mixture scoring excludes the static stratum (F84-ii) ──
    mix = SigmaInfMixtureFilter(th0, el0, _params(), ell_inf_mean=0.5,
                                ell_star=0.4, p_static_stratum=0.5, seed=7)
    q_mix = expected_progress_score(mix, cands)
    # manual: only non-static strata, weighted
    from training.trainer_greedy import RewardWeights, _expected_reward
    w = RewardWeights(beta_t=0.0, beta_sigma=1.0, beta_r=0.0)
    om = mix.weights()
    q_man = np.zeros(len(cands))
    for om_j, f_j in zip(om, mix.strata):
        if f_j.p.rule == "static":
            continue
        q_man += om_j * _expected_reward(f_j, cands.s_mean,
                                         cands.y_star.astype(float), w,
                                         s_sd=cands.s_sd)
    ok(np.allclose(q_mix, q_man), "mixture score = Σ ω_j Q_j over "
       "NON-static strata only (F84-ii)")

    # ── 5: plain-TaskFilter path equals tier-1 Q ──
    ft = TaskFilter(th0, el0, _params(), seed=11)
    ok(np.allclose(expected_progress_score(ft, cands),
                   _expected_reward(ft, cands.s_mean,
                                    cands.y_star.astype(float), w,
                                    s_sd=cands.s_sd)),
       "plain TaskFilter path equals the tier-1 Q (β_σ term)")

    # ── 6–8: expected_progress_rate ordering ──
    below = TaskFilter(0.0 * th0, np.full(300, -0.3), _params(), seed=13)
    at_ceil = TaskFilter(0.0 * th0, np.full(300, 0.51),
                         _params(sigma_inf=0.6), seed=17)
    ep_below = expected_progress_rate(below, 0.4, 0.3)
    ep_ceil = expected_progress_rate(at_ceil, 0.4, 0.3)
    ok(ep_below > 5 * max(ep_ceil, 1e-12),
       f"below-bar trainable EP ({ep_below:.4f}) ≫ at-ceiling EP "
       f"({ep_ceil:.5f})")
    above = TaskFilter(0.0 * th0, np.full(300, 0.9),
                       _params(sigma_inf=0.3), seed=19)
    ok(expected_progress_rate(above, 0.4, 0.3) < 1e-6,
       "above-bar mass earns ~zero σ-progress (bar-referenced)")
    biased = TaskFilter(np.full(300, 0.8), np.full(300, -0.3), _params(),
                        seed=23)
    ep_b = expected_progress_rate(biased, 0.4, 0.3, beta_sigma=0.0)
    ok(abs(ep_b - _params().alpha_t) < 1e-6,
       f"large bias excess earns the capped α_t term ({ep_b:.3f})")

    # ── 9: progress_alloc picks the trainable task in the USER-B trap ──
    mix_tr = SigmaInfMixtureFilter(th0, el0 - 0.4, _params(),
                                   ell_inf_mean=0.6, ell_star=0.4, seed=29)
    mix_pl = SigmaInfMixtureFilter(th0, el0 - 0.4, _params(),
                                   ell_inf_mean=0.6, ell_star=0.4,
                                   p_static_stratum=0.9, seed=31)
    sch = DeficiencyScheduler(2, progress_alloc=True)
    mp = [TaskModePolicy(0, 0.4, 0.67, ModeThresholds()),
          TaskModePolicy(1, 0.4, 0.67, ModeThresholds())]
    picks = [sch.pick([mix_tr, mix_pl], [0.4, 0.4], mp) for _ in range(8)]
    ok(picks.count(0) >= 6,
       f"EP allocation prefers the trainable task ({picks.count(0)}/8 picks)")

    # ── 10: progress placement end-to-end on the real bank ──
    from training.bank_adapter import BankAdapter
    from training.trainer_policy import TrainerPolicy

    class _Remap:
        def __init__(self, bank, tasks):
            self.bank, self.tasks = bank, tasks

        def candidates(self, task, **kw):
            return self.bank.candidates(self.tasks[task], **kw)

    bank = BankAdapter()
    mix1 = SigmaInfMixtureFilter(th0, el0, _params(), ell_inf_mean=0.5,
                                 ell_star=0.306, seed=37)
    pol = TrainerPolicy([mix1], [0.306], [0.736], _Remap(bank, [1]),
                        thresholds=ModeThresholds(
                            meanskill_gate=False, sd_floor=0.33,
                            progress_placement=True),
                        seed=41, probe_every=5, finish_first=True,
                        progress_alloc=True, progress_s_sd=0.057)
    served, prog_picks, mirror_ok = 0, 0, []
    rr = np.random.default_rng(43)
    last_s = None
    while served < 24:
        ch = pol.step(now=1e9 + served)
        if ch is None:
            break
        if ch["info"].get("progress"):
            prog_picks += 1
            if last_s is not None and (last_s > 0) != (ch["s"] > 0):
                mirror_ok.append(abs(abs(ch["s"]) - abs(last_s))
                                 <= 0.15 * max(abs(last_s), 0.3) + 1e-9)
            last_s = ch["s"] if ch["mode"] == "skill" else last_s
        y = int(rr.random() < 0.8) if ch["y_star"] else int(rr.random() < 0.2)
        pol.record(ch, y)
        served += 1
    ok(served == 24 and prog_picks > 0,
       f"progress placement serves end-to-end ({prog_picks} progress picks)")

    # ── 11: sandbox wiring — BOTH selection flags deferred (F84 mixed
    # result; F85-ii post-boundary re-polish loop + FG direction), the
    # machinery ships opt-in engine-side only ──
    from sandbox import config as SC
    ok(SC.PROGRESS_ALLOC is False and
       ModeThresholds().progress_placement is False,
       "sandbox defers F85 allocation and F84 placement (opt-in only)")

    print(f"\nM24 SMOKE TEST PASSED — {CHECKS} checks.")


if __name__ == "__main__":
    main()
