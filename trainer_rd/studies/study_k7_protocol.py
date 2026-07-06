"""M19-4 — K=7 OPEN-PROTOCOL capstone: the closest simulation to the product.

Everything the real tester would experience, end-to-end: 7 tasks (v15 cuts),
the full TrainerPolicy orchestration (deficiency interleave D3, retention
mode D17, certification probes + e-gates D23, derived 0-bias band F60,
mean-skill off D30), per-task σ∞-mixtures (D22) with the F62 anchored
assumed rates (D29), OPEN-ENDED 40-trial sessions (D28) with real calendar
gaps, per-task power-law TRUE forgetting, and MixtureGapAnchor re-anchoring
at session opens (D27/M19-3: top-deficiency task gets an evidence probe
block; all other stale tasks get the prior-tilted transform).

Learner realism: per-task heterogeneous rates (anchored × LogN jitter),
per-task ceilings ~ N(v15 expert, 0.2) — so some tasks are genuinely
NON-TRAINABLE (ceiling < cut; any declaration there is a false graduation),
biased sub-skill starts, person-level forgetting stability.

Arms (paired seeds): fullstack (with MixtureGapAnchor) vs identity (beliefs
carried across gaps unchanged — the pre-D27 behavior).

Metrics (D28 order): FG declarations (incl. non-trainable tasks), stale-
mastered session-opens, sessions to all-TRAINABLE-tasks declared, trials +
anchor overhead.

Run:  python3 -m studies.study_k7_protocol [--smoke]
      → figures/data_k7_protocol.npz
"""
from __future__ import annotations

import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from training.bank_adapter import (ANCHORED_ALPHA_SIGMA, ANCHORED_ALPHA_T,
                                   BankAdapter, ELL_STAR_V15, SIGMA_STAR_V15)
from training.gap_anchor import MixtureGapAnchor
from training.learner_sim import Learner, LearnerParams
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import (ModeThresholds, RetentionScheduler,
                                     TrainerPolicy, derived_t_star)
from engine.instrument_v15 import instrument

DAY = 86_400.0
GAPS_D = (0.5, 1.0, 2.0, 4.0, 7.0, 14.0)
K = 7
V15 = instrument("v15")


def make_person(seed):
    r = np.random.default_rng(500 + seed)
    alpha_t = ANCHORED_ALPHA_T * np.exp(0.35 * r.standard_normal(K))
    alpha_s = ANCHORED_ALPHA_SIGMA * np.exp(0.35 * r.standard_normal(K))
    ell_inf = V15.expert_ell + 0.2 * r.standard_normal(K)
    sigma0 = np.exp(np.log(1.4) + 0.15 * r.standard_normal(K))
    t0 = r.choice([-1, 1], K) * (0.55 + 0.15 * r.standard_normal(K))
    beta = 0.7 * (1 + 0.2 * r.standard_normal())
    tau_f = 3.0 * DAY
    return dict(alpha_t=alpha_t, alpha_s=alpha_s, ell_inf=ell_inf,
                sigma0=sigma0, t0=t0, beta=beta, tau_f=tau_f)


class MultiLearner:
    """K independent per-task learners sharing one person (rates differ per
    task; learner_sim.Learner broadcasts a single α, so wrap K of them).

    CONSOLIDATION (M19/F70 lesson): with a FIXED forgetting stability
    (τ_f = 3 d), the open protocol's equilibrium skill sits BELOW the v15
    cuts — 40-trial sessions with multi-day gaps can never certify anyone
    (verified: 0/106 declared in both arms). Real skill practice
    CONSOLIDATES (§3A LT2: 'stability S growing under successful spaced
    retrieval'); modeled here as per-task stability growth with practice,
        τ_f,k = τ_f0 · (1 + n_k / 40)^1.5      (n_k = trials on task k),
    i.e. each ~session of practice multiplies stability (SM-2-flavored;
    the exponent is a placeholder the pilot's GapAnchor S-estimates will
    MEASURE — a named pilot endpoint, not a fitted fact)."""

    def __init__(self, person, seed, *, consolidation=True):
        self.subs = []
        self.n_k = np.zeros(K)
        self.consolidation = consolidation
        for k in range(K):
            p = LearnerParams(alpha_t=float(person["alpha_t"][k]),
                              alpha_sigma=float(person["alpha_s"][k]),
                              sigma_inf=float(np.exp(-person["ell_inf"][k])),
                              q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
            self.subs.append(Learner([float(person["sigma0"][k])],
                                     [float(person["t0"][k])], p,
                                     seed=seed + 17 * k))

    def step(self, s, task, y_star, feedback=True):
        self.n_k[task] += 1
        return self.subs[task].step(s, 0, y_star, feedback=feedback)

    def ell(self, k):
        return float(-np.log(self.subs[k].sigma[0]))

    def t(self, k):
        return float(self.subs[k].t[0])

    def forget(self, dt, person, ell_b, t_b):
        for k in range(K):
            tau = person["tau_f"]
            if self.consolidation:
                tau = tau * (1.0 + self.n_k[k] / 40.0) ** 1.5
            r_true = (1.0 + dt / tau) ** (-person["beta"])
            e = self.ell(k)
            self.subs[k].sigma[0] = float(
                np.exp(-(ell_b[k] + r_true * (e - ell_b[k]))))
            self.subs[k].t[0] = t_b[k] + r_true * (self.subs[k].t[0] - t_b[k])


def run_one(job):
    arm, seed, n_sessions, per_session = job
    person = make_person(seed)
    lnr = MultiLearner(person, 800 + seed)
    ell_b = -np.log(person["sigma0"])
    t_b = person["t0"].copy()
    rng = np.random.default_rng(700 + seed)
    bank = BankAdapter()
    fp = [LearnerParams(alpha_t=ANCHORED_ALPHA_T,
                        alpha_sigma=ANCHORED_ALPHA_SIGMA,
                        sigma_inf=float(V15.sigma_inf[k]), q_t=0.04,
                        q_sigma=0.02, rho=0.6, rule="soft") for k in range(K)]
    filters = [SigmaInfMixtureFilter(0.45 * rng.standard_normal(300),
                                     0.45 * rng.standard_normal(300), fp[k],
                                     ell_inf_mean=float(V15.expert_ell[k]),
                                     tau=0.30, J=7, p_static_stratum=0.15,
                                     ell_star=float(ELL_STAR_V15[k]),
                                     seed=seed + 31 * k)
               for k in range(K)]
    pol = TrainerPolicy(filters, list(ELL_STAR_V15), list(SIGMA_STAR_V15),
                        bank, thresholds=ModeThresholds(meanskill_gate=False),
                        seed=seed + 5, probe_every=5, finish_first=True,
                        retention=RetentionScheduler())
    anchors = [MixtureGapAnchor() for _ in range(K)]
    bands = [derived_t_star(ANCHORED_ALPHA_T, 0.04, float(SIGMA_STAR_V15[k]))
             for k in range(K)]
    trainable = [person["ell_inf"][k] > ELL_STAR_V15[k] for k in range(K)]
    declared_at = [None] * K
    fg = [False] * K
    stale_opens = 0
    n_anchor_trials = 0
    trial_ctr = 0
    gaps = rng.choice(GAPS_D, size=n_sessions - 1)
    for sess in range(n_sessions):
        if sess > 0:
            dt = float(gaps[sess - 1]) * DAY
            lnr.forget(dt, person, ell_b, t_b)
            # stale-mastered check BEFORE any re-anchoring
            for k in range(K):
                if (declared_at[k] is not None
                        and pol.mode_policies[k].is_mastered(filters[k])
                        and lnr.ell(k) < ELL_STAR_V15[k]):
                    stale_opens += 1
            if arm == "fullstack":
                # MEASURE-OR-LEAVE-ALONE (F70 lesson): prior-only transforms
                # of unprobed tasks are HARMFUL once the learner has
                # consolidated (the prior assumes more forgetting than
                # reality; verified: prior-transforming 6/7 tasks per
                # session gave 0/106 declarations vs identity's 9/106).
                # Only the task that receives the anchor PROBES is
                # transformed — its evidence resolves the retention
                # hypothesis; every other belief is left alone.
                defs = [(pol.scheduler.deficiency(filters[k],
                                                  ELL_STAR_V15[k]), k)
                        for k in range(K) if declared_at[k] is None]
                top = max(defs)[1] if defs else 0
                n_a = anchors[top].open_session(
                    filters[top], dt, theta_base=float(-t_b[top]),
                    ell_base=float(ell_b[top]), seed=seed + 900 + sess + top)
                pool_k = bank.candidates(top, feedback_safe=True)
                for j in range(n_a):
                    idx = anchors[top].pick_probe(
                        pool_k, float(SIGMA_STAR_V15[top]))
                    s = float(pool_k.s_mean[idx])
                    ssd = float(pool_k.s_sd[idx])
                    ys = int(pool_k.y_star[idx])
                    y = lnr.step(s + ssd * rng.standard_normal(), top, ys)
                    anchors[top].anchor_step(s, y, ys, s_sd=ssd)
                    n_anchor_trials += 1
                anchors[top].close_anchor(seed=seed + 990 + sess + top)
                # person-level stability still pools from probed gaps
                S_pool = float(np.mean([a.S for a in anchors]))
                for a in anchors:
                    a.S = S_pool
        for j in range(per_session):
            ch = pol.step(now=trial_ctr)
            if ch is None:
                break
            k = ch["task"]
            s_real = ch["s"] + ch["s_sd"] * rng.standard_normal()
            y = lnr.step(s_real, k, ch["y_star"])
            pol.record(ch, y)
            trial_ctr += 1
            for kk in range(K):
                if (declared_at[kk] is None
                        and pol.mode_policies[kk].is_mastered(filters[kk])):
                    declared_at[kk] = trial_ctr
                    if (lnr.ell(kk) < ELL_STAR_V15[kk]
                            or abs(lnr.t(kk)) > bands[kk] + 0.1
                            or not trainable[kk]):
                        fg[kk] = True
        if all(declared_at[k] is not None for k in range(K) if trainable[k]):
            break
    n_train_declared = sum(1 for k in range(K)
                           if trainable[k] and declared_at[k] is not None)
    return dict(arm=arm, seed=seed, fg=sum(fg),
                fg_nontrainable=sum(1 for k in range(K)
                                    if fg[k] and not trainable[k]),
                n_trainable=sum(trainable),
                declared_trainable=n_train_declared,
                sessions_used=sess + 1, trials=trial_ctr,
                anchor_trials=n_anchor_trials, stale_opens=stale_opens)


def main(smoke=False):
    n_seeds = 3 if smoke else 16
    n_sessions = 6 if smoke else 40
    jobs = [(arm, s, n_sessions, 40) for arm in ("fullstack", "identity")
            for s in range(n_seeds)]
    with Pool(3 if smoke else 32) as pool:
        rows = pool.map(run_one, jobs)
    print(f"{'arm':>10} {'FG(total)':>9} {'FG(nontrain)':>12} "
          f"{'declared/trainable':>18} {'med sessions':>12} "
          f"{'stale-opens':>11} {'anchor trials':>13}")
    for arm in ("fullstack", "identity"):
        rs = [r for r in rows if r["arm"] == arm]
        print(f"{arm:>10} {sum(r['fg'] for r in rs):>9} "
              f"{sum(r['fg_nontrainable'] for r in rs):>12} "
              f"{sum(r['declared_trainable'] for r in rs):>8}/"
              f"{sum(r['n_trainable'] for r in rs):<9} "
              f"{int(np.median([r['sessions_used'] for r in rs])):>12} "
              f"{sum(r['stale_opens'] for r in rs):>11} "
              f"{int(np.mean([r['anchor_trials'] for r in rs])):>13}")
    if not smoke:
        np.savez("figures/data_k7_protocol.npz",
                 rows=np.array([(r["arm"] == "fullstack", r["seed"], r["fg"],
                                 r["fg_nontrainable"], r["n_trainable"],
                                 r["declared_trainable"], r["sessions_used"],
                                 r["trials"], r["anchor_trials"],
                                 r["stale_opens"]) for r in rows],
                                dtype=float))
        print("saved figures/data_k7_protocol.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
