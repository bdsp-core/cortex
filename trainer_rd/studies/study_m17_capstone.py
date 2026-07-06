"""M17 capstone — misspec-zoo graduation OC under the NEW DEFAULT STACK.

The F29 question re-asked after two hardening generations: what is the
graduation operating characteristic of the CURRENT default trainer —
exact kernel (D26) + σ_∞-mixture (D22, v15 expert anchor) + certification
probes & e-gate (D23) + derived 0-bias band (M17/OQ6) + v15 cuts (D25) —
across the deviation zoo? Comparator: the point filter with the same kernel
(the F28 baseline under current defaults).

Zoo (one named deviation each; misspec_learners):
  wellspec       assumed model, reachable ceiling            → should declare
  static_below   frozen at 1.25·σ*_v15                       → must NOT
  careless       λ=0.35, max accuracy 0.65                   → must NOT
  anti           sign-flipped criterion learning             → must NOT
  powerlaw       non-exponential learning curve              → should declare
  t3link         heavy-tail response link                    → should declare
  asym_fitted    EXTSET-fitted lapse λ_fa=.0024/λ_miss=.0095 → should declare
                 (the M17-6 λ-sensitivity arm: the filter still assumes .025)

Metrics (priorities FG ≥ lateness): declared rate, FG-latent (declared while
true state below bar/band), FG-perf (declared while pool-accuracy < the
designed acceptance bar), median lateness vs true crossing, e-gate fires,
trainability. 30 seeds × 400 trials, real domain3 pool, Pool(42).

Run:  python3 -m studies.study_m17_capstone [--smoke]
      → figures/data_m17_capstone.npz
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np

from training.bank_adapter import (BankAdapter, TaskCandidates,
                                   ELL_STAR_V15, SIGMA_STAR_V15)
from training.benchmark_trainer import _select
from training.bridge_conventions import engine_to_plan
from training.learner_sim import Learner, LearnerParams
from training.misspec_learners import (AntiLearner, AsymmetricLapseLearner,
                                       HeavyTailLinkLearner, PowerLawLearner,
                                       accuracy_bar, careless_learner,
                                       pool_accuracy, static_below_cut)
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_greedy import RewardWeights
from training.trainer_policy import (EProcessGate, ModeThresholds,
                                     TaskModePolicy, at_bar_accuracy,
                                     cert_probe_score, derived_t_star)
from training.training_filter import TaskFilter

TASK = 2
ELL_STAR, SIG_STAR = ELL_STAR_V15[TASK], SIGMA_STAR_V15[TASK]
ELL_INF_V15_D3 = 0.663            # v15 expert ceiling, domain3
PROBE_EVERY = 5
_POOL = None                      # per-worker cache


def _pool():
    global _POOL
    if _POOL is None:
        ad = BankAdapter()
        full = ad.candidates(TASK, feedback_safe=True)
        sub = np.random.default_rng(0).choice(len(full), size=600,
                                              replace=False)
        _POOL = TaskCandidates(TASK, full.seg_id[sub], full.s_mean[sub],
                               full.s_sd[sub], full.y_star[sub],
                               full.margin[sub], full.coherent[sub])
    return _POOL


def make_learner(kind, seed):
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06,
                       sigma_inf=0.82 * SIG_STAR, q_t=0.04, q_sigma=0.02,
                       rho=0.6, rule="soft")
    if kind == "wellspec":
        return Learner([1.5], [0.6], tp, seed=seed)
    if kind == "static_below":
        return static_below_cut(SIG_STAR, seed=seed)
    if kind == "careless":
        return careless_learner([1.5], [0.6], tp, seed=seed)
    if kind == "anti":
        return AntiLearner([0.9 * SIG_STAR], [0.4], tp, seed=seed)
    if kind == "powerlaw":
        return PowerLawLearner([1.5], [0.6], tp, seed=seed)
    if kind == "t3link":
        return HeavyTailLinkLearner([1.5], [0.6], tp, seed=seed)
    if kind == "asym_fitted":
        return AsymmetricLapseLearner([1.5], [0.6], tp, lam_fa=0.0024,
                                      lam_miss=0.0095, seed=seed)
    raise ValueError(kind)


def run_one(job):
    kind, arm, seed, budget = job
    pool = _pool()
    lnr = make_learner(kind, 800_000 + seed)
    rng = np.random.default_rng(700_000 + seed)
    # M18 (D29): assumed rates = the F62 anchored priors (fp < tp, F17-safe)
    from training.bank_adapter import ANCHORED_ALPHA_SIGMA, ANCHORED_ALPHA_T
    fp = LearnerParams(alpha_t=ANCHORED_ALPHA_T,
                       alpha_sigma=ANCHORED_ALPHA_SIGMA,
                       sigma_inf=0.82 * SIG_STAR, q_t=0.04, q_sigma=0.02,
                       rho=0.6, rule="soft")
    th0 = 0.45 * rng.standard_normal(400)
    el0 = 0.45 * rng.standard_normal(400)
    if arm == "mixstack":
        filt = SigmaInfMixtureFilter(th0, el0, fp,
                                     ell_inf_mean=ELL_INF_V15_D3, tau=0.30,
                                     J=7, p_static_stratum=0.15,
                                     ell_star=ELL_STAR, seed=seed)
        # M18 (D30/F64): the mean-skill branch reads the strata-dragged
        # mean before weights resolve — mixture graduation disables it
        # (the H1 leak: 11/11 static_below FGs were viaMS in the ablation)
        mp_th = ModeThresholds(meanskill_gate=False)
    else:                                          # point comparator (legacy)
        filt = TaskFilter(th0, el0, fp, seed=seed)
        mp_th = ModeThresholds()
    mp = TaskModePolicy(0, ELL_STAR, SIG_STAR, mp_th)
    gate = EProcessGate(alpha=0.05) if arm == "mixstack" else None
    if gate is not None:
        mp.egate = gate
    W = RewardWeights()
    state = {"k": 0, "bal": 0}
    flip, n_true = 1, None
    band = derived_t_star(fp.alpha_t, fp.q_t, SIG_STAR)
    out = None
    for k in range(budget):
        state["k"] = k
        sig_hat, t_hat = engine_to_plan(*filt.mean())
        if gate is not None and k % PROBE_EVERY == PROBE_EVERY - 1:
            want = 1 if flip > 0 else 0
            flip *= -1
            m = np.where(pool.y_star == want)[0]
            sc = cert_probe_score(pool.s_mean[m], pool.s_sd[m], SIG_STAR,
                                  t_hat)
            idx = int(m[np.argmax(sc)])
            probe = True
        else:
            idx = _select("tier2", filt, pool, state, rng, W)
            probe = False
        s, s_sd = float(pool.s_mean[idx]), float(pool.s_sd[idx])
        y_star = int(pool.y_star[idx])
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        mp.note_posterior(filt)
        state["bal"] += 1 if y_star == 1 else -1
        if probe:
            gate.update(int(y == y_star),
                        at_bar_accuracy(s, s_sd, SIG_STAR, t_hat, y_star))
        true_l = float(-np.log(lnr.sigma[0]))
        if n_true is None and true_l > ELL_STAR and abs(float(lnr.t[0])) <= band:
            n_true = k + 1
        if out is None and mp.is_mastered(filt):
            out = {"declared": True, "n_decl": k + 1, "n_true_at": n_true}
    acc = pool_accuracy(lnr, pool)
    a_bar = accuracy_bar(pool, SIG_STAR, 0.0)
    true_l = float(-np.log(lnr.sigma[0]))
    if out is None:
        out = {"declared": False, "n_decl": budget, "n_true_at": n_true}
    out.update(kind=kind, arm=arm, seed=seed,
               fg_perf=bool(out["declared"] and acc < a_bar),
               fg_latent=bool(out["declared"]
                              and (true_l < ELL_STAR
                                   or abs(float(lnr.t[0])) > band + 0.1)),
               egate_fired=bool(gate.fired) if gate else False,
               trainability=(filt.trainability(ELL_STAR)
                             if hasattr(filt, "trainability") else np.nan))
    return out


ZOO = ("wellspec", "static_below", "careless", "anti", "powerlaw",
       "t3link", "asym_fitted")


def main(smoke=False):
    n_seeds = 4 if smoke else 30
    budget = 200 if smoke else 400
    jobs = [(kind, arm, s, budget) for kind in ZOO
            for arm in ("mixstack", "point") for s in range(n_seeds)]
    with Pool(6 if smoke else 42) as pool:
        rows = pool.map(run_one, jobs)
    print(f"{'zoo member':>13} {'arm':>9} {'declared':>9} {'FG-lat':>7} "
          f"{'FG-perf':>8} {'med n_decl':>10} {'med lateness':>12} "
          f"{'egate':>6} {'med train':>9}")
    for kind in ZOO:
        for arm in ("mixstack", "point"):
            rs = [r for r in rows if r["kind"] == kind and r["arm"] == arm]
            dec = [r for r in rs if r["declared"]]
            late = [r["n_decl"] - r["n_true_at"] for r in dec
                    if r["n_true_at"] is not None]
            print(f"{kind:>13} {arm:>9} {len(dec):>6}/{n_seeds:<2} "
                  f"{sum(r['fg_latent'] for r in rs):>7} "
                  f"{sum(r['fg_perf'] for r in rs):>8} "
                  f"{int(np.median([r['n_decl'] for r in dec])) if dec else -1:>10} "
                  f"{int(np.median(late)) if late else -1:>12} "
                  f"{sum(r['egate_fired'] for r in rs):>6} "
                  f"{np.nanmedian([r['trainability'] for r in rs]):>9.2f}")
    if not smoke:
        np.savez("figures/data_m17_capstone.npz",
                 rows=np.array([(ZOO.index(r["kind"]),
                                 r["arm"] == "mixstack", r["seed"],
                                 r["declared"], r["n_decl"],
                                 -1 if r["n_true_at"] is None else r["n_true_at"],
                                 r["fg_latent"], r["fg_perf"],
                                 r["egate_fired"],
                                 np.nan_to_num(r["trainability"], nan=-1.0))
                                for r in rows], dtype=float))
        print("saved figures/data_m17_capstone.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
