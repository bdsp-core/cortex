"""M15 steps 4–5 validation — σ_∞ mixture + certification probes + e-gate.

The F28/F29 pathology: the shipped filter certifies its own σ_∞ prior
(static sub-cut learner declared 30/30, flat OC). This study measures whether
the M15 stack fixes graduation self-assessment at the BELIEF level:

  arms (all serve via the benchmark tier-2 rule, D16 hybrid gate declares):
    shipped   TaskFilter, shipped kernel               (F28 baseline)
    exact     TaskFilter(exact_kernel=True)            (MA-1 alone)
    mix       SigmaInfMixtureFilter, exact kernel      (MA-2)
    mix+probe mix + 1-in-5 certification probes scored by cert_probe_score
              + EProcessGate refutation monitor as a declaration conjunct

  learners (misspec zoo, domain3 cut ℓ*=0.534, σ*=0.586):
    wellspec      soft R–W, ceiling 0.82σ* (ℓ_∞≈0.73 > ℓ*) — MUST graduate
    static_below  frozen at σ=1.25σ* (ℓ=0.311 < ℓ*) — must NEVER graduate
    careless      λ=0.35 (max accuracy 0.65 < any bar) — must NEVER graduate

Parameter provenance: mixture grid centered at the D10 expert ceiling
(domain3 expert_ℓ=0.765), τ=0.30, J=7, static stratum 0.15, probe_every=5,
e-gate α=0.05 — justifications in mixture_filter.py / trainer_policy.py.

Run:  python3 -m studies.study_audit_hardening [--smoke]
"""
from __future__ import annotations

import sys

import numpy as np

from training.bank_adapter import BankAdapter, ELL_STAR, SIGMA_STAR, TaskCandidates
from training.bridge_conventions import LAPSE_RATE, engine_to_plan
from training.learner_sim import Learner, LearnerParams
from training.training_filter import TaskFilter
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import (ModeThresholds, TaskModePolicy,
                                     cert_probe_score, EProcessGate,
                                     at_bar_accuracy)
from training.benchmark_trainer import _select
from training.trainer_greedy import RewardWeights
from scipy.stats import norm

TASK = 2                                   # domain3
ELL_INF_MEAN_D3 = 0.765                    # §2B expert ceiling, v14
T_STAR = 0.30
PROBE_EVERY = 5


def _make_learner(kind, sig_star, seed):
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.82 * sig_star,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    if kind == "wellspec":
        return Learner([1.5], [0.5], tp, seed=seed)
    if kind == "static_below":
        from training.misspec_learners import static_below_cut
        return static_below_cut(sig_star, seed=seed)
    if kind == "careless":
        from training.misspec_learners import careless_learner
        return careless_learner([1.5], [0.5], tp, seed=seed)
    raise ValueError(kind)


def _make_filter(arm, fp, rng, ell_star, seed):
    th0 = 0.3 * rng.standard_normal(400)
    el0 = 0.3 * rng.standard_normal(400)
    if arm == "shipped":                 # pre-M17 kernel (explicit since D26)
        return TaskFilter(th0, el0, fp, seed=seed, exact_kernel=False)
    if arm == "exact":
        return TaskFilter(th0, el0, fp, seed=seed, exact_kernel=True)
    return SigmaInfMixtureFilter(th0, el0, fp, ell_inf_mean=ELL_INF_MEAN_D3,
                                 tau=0.30, J=7, p_static_stratum=0.15,
                                 ell_star=ell_star, seed=seed,
                                 exact_kernel=True)


# at-bar probe reference: canonical implementation now lives in
# trainer_policy.at_bar_accuracy (M16 wiring dedup; numerically identical)


def run_one(arm, kind, pool, *, budget=400, seed=0):
    ell_star, sig_star = ELL_STAR[TASK], SIGMA_STAR[TASK]
    lnr = _make_learner(kind, sig_star, seed)
    fp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.82 * sig_star,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    rng = np.random.default_rng(1000 + seed)
    filt = _make_filter(arm, fp, rng, ell_star, seed)
    mp = TaskModePolicy(0, ell_star, sig_star, ModeThresholds())
    gate = EProcessGate(alpha=0.05) if arm == "mix+probe" else None
    greedyW = RewardWeights()
    state = {"k": 0, "bal": 0}
    probe_flip = 1
    for k in range(budget):
        state["k"] = k
        sig_hat, t_hat = engine_to_plan(*filt.mean())
        if gate is not None and k % PROBE_EVERY == PROBE_EVERY - 1:
            # certification probe: Fisher-optimal for ℓ at the CUT state
            want = 1 if probe_flip > 0 else 0
            probe_flip *= -1
            m = np.where(pool.y_star == want)[0]
            sc = cert_probe_score(pool.s_mean[m], pool.s_sd[m], sig_star,
                                  t_hat, lapse=LAPSE_RATE)
            idx = int(m[np.argmax(sc)])
            is_probe = True
        else:
            idx = _select("tier2", filt, pool, state, rng, greedyW)
            is_probe = False
        s, s_sd = float(pool.s_mean[idx]), float(pool.s_sd[idx])
        y_star = int(pool.y_star[idx])
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        state["bal"] += 1 if y_star == 1 else -1
        mp.note_posterior(filt)
        if is_probe:
            a0 = at_bar_accuracy(s, s_sd, sig_star, t_hat, y_star)
            gate.update(int(y == y_star), a0)
        if mp.is_mastered(filt) and (gate is None or not gate.fired):
            true_l = float(-np.log(lnr.sigma[0]))
            return {"declared": True, "n": k + 1,
                    "fg": true_l < ell_star or abs(float(lnr.t[0])) > T_STAR,
                    "trainability": (filt.trainability(ell_star)
                                     if hasattr(filt, "trainability") else np.nan),
                    "egate_fired": bool(gate.fired) if gate else False}
    return {"declared": False, "n": budget, "fg": False,
            "trainability": (filt.trainability(ell_star)
                             if hasattr(filt, "trainability") else np.nan),
            "egate_fired": bool(gate.fired) if gate else False}


def main(n_seeds=20, budget=400):
    ad = BankAdapter()
    full = ad.candidates(TASK, feedback_safe=True)
    sub = np.random.default_rng(0).choice(len(full), size=600, replace=False)
    pool = TaskCandidates(TASK, full.seg_id[sub], full.s_mean[sub],
                          full.s_sd[sub], full.y_star[sub], full.margin[sub],
                          full.coherent[sub])
    print(f"{'learner':>13} {'arm':>10} {'declared':>9} {'med n_decl':>10} "
          f"{'FG':>4} {'med train.':>10} {'egate':>6}")
    for kind in ("wellspec", "static_below", "careless"):
        for arm in ("shipped", "exact", "mix", "mix+probe"):
            rows = [run_one(arm, kind, pool, budget=budget, seed=s)
                    for s in range(n_seeds)]
            dec = [r for r in rows if r["declared"]]
            med_n = int(np.median([r["n"] for r in dec])) if dec else -1
            tr = np.nanmedian([r["trainability"] for r in rows])
            eg = sum(r["egate_fired"] for r in rows)
            print(f"{kind:>13} {arm:>10} {len(dec):>6}/{n_seeds:<2} "
                  f"{med_n:>10} {sum(r['fg'] for r in rows):>4} "
                  f"{tr:>10.2f} {eg:>6}")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    main(n_seeds=(4 if smoke else 20), budget=(200 if smoke else 400))
