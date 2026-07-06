"""M12/V2 — rung-3 mis-specification campaign (PROJECT_MEMORY §3D).

Runs the SHIPPED trainer (tier2, D15 selection + D16 hybrid gate at default
ModeThresholds) — with staircase85 and random as graceful-degradation
comparators — against every member of the mis-specified learner zoo
(misspec_learners.py), plus a population-heterogeneity arm. The filter always
assumes the Phase-2 model (soft R–W, α_t=0.2, α_σ=0.06, fixed σ_∞, λ=0.025);
the LEARNER deviates. Common random numbers: same seed ⇒ same pool, same
stimulus noise stream across policies and zoo members.

Per-run metrics:
  n_true    first trial the TRUE state crosses mastery (σ ≤ σ*, |t| ≤ t*);
            budget+1 if never (adversarial members: by construction never)
  n_decl    first trial the shipped D16 hybrid gate DECLARES mastery
            (TaskModePolicy.is_mastered on the filter); budget+1 if never
  A_decl    pool accuracy (true response fn, s_sd smeared) at declaration
  false_grad  declared AND A_decl < A_bar(σ*, t*) — outside the designed
            acceptance envelope (the headline safety metric)
  rmse_ell/rmse_t   posterior-mean tracking error vs true latent, trials ≥ 50
  cov_ell/cov_t     90% PIT coverage: u = P_w(particle ≤ truth) ∈ (.05,.95),
            trials ≥ 30 (for link-misspec members the latent is a
            pseudo-parameter — tracking numbers are descriptive there)

Usage:  python3 -m studies.study_misspec --smoke   (tiny: 3 members × 3 seeds)
        python3 -m studies.study_misspec           (full campaign → figures/data_misspec_campaign.npz)
"""
from __future__ import annotations

import sys

import numpy as np

from training.bank_adapter import BankAdapter, ELL_STAR, SIGMA_STAR, TaskCandidates
from training.benchmark_trainer import _select, T_STAR
from training.learner_sim import Learner, LearnerParams
from training.misspec_learners import (AntiLearner, AsymmetricLapseLearner,
                              DriftingCeilingLearner, FatigueLearner,
                              HeavyTailLinkLearner, MomentumCriterionLearner,
                              PlateauLearner, PowerLawLearner, accuracy_bar,
                              careless_learner, static_below_cut)
from training.trainer_greedy import RewardWeights
from training.trainer_policy import ModeThresholds, TaskModePolicy
from training.training_filter import TaskFilter

TASK = 2                       # domain3 — the M10 normative benchmark task
BUDGET = 400
NPART = 400
SIG_STAR, L_STAR = SIGMA_STAR[TASK], ELL_STAR[TASK]
SIG_INF = 0.82 * SIG_STAR      # reachable ceiling (benchmark convention)
SIGMA0, T0 = 1.5, 0.8


def assumed_params():
    """The filter's (fixed, Phase-2/D7) model — NEVER varies in this study."""
    return LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=SIG_INF,
                         q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")


# learner params handed to zoo members (their own dynamics reinterpret them)
def true_params(rule="soft"):
    return LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=SIG_INF,
                         q_t=0.04, q_sigma=0.02, rho=0.6, rule=rule)


ZOO = {
    # axis: control (= rung-2 well-specified reference, same harness)
    "wellspec":      lambda seed: Learner([SIGMA0], [T0], true_params(), seed=seed),
    # axis: dynamics family
    "powerlaw":      lambda seed: PowerLawLearner([SIGMA0], [T0], true_params(),
                                                  beta=0.7, u0=10.0, seed=seed),
    "plateau":       lambda seed: PlateauLearner([SIGMA0], [T0], true_params(),
                                                 u_steps=(10.0, 25.0), seed=seed),
    "drift_ceiling": lambda seed: DriftingCeilingLearner(
        [SIGMA0], [T0], true_params(), drift_per_trial=3.7e-4,
        ceil_cap=0.97 * SIG_STAR, seed=seed),
    "momentum":      lambda seed: MomentumCriterionLearner(
        [SIGMA0], [T0], true_params(), kappa=0.15, seed=seed),
    "hard_rule":     lambda seed: Learner([SIGMA0], [T0], true_params("hard"),
                                          seed=seed),
    # axis: observation model
    "t3link":        lambda seed: HeavyTailLinkLearner([SIGMA0], [T0],
                                                       true_params(), seed=seed),
    "asym_lapse":    lambda seed: AsymmetricLapseLearner(
        [SIGMA0], [T0], true_params(), lam_fa=0.08, lam_miss=0.01, seed=seed),
    "fatigue":       lambda seed: FatigueLearner([SIGMA0], [T0], true_params(),
                                                 lam_max=0.12, n_ramp=300,
                                                 seed=seed),
    # axis: adversarial (must NEVER graduate)
    "anti":          lambda seed: AntiLearner([1.25 * SIG_STAR], [0.3],
                                              true_params(), seed=seed),
    "careless":      lambda seed: careless_learner([SIGMA0], [T0],
                                                   true_params(), seed=seed),
    "static_below":  lambda seed: static_below_cut(SIG_STAR, seed=seed),
}
NEVER_GRADUATE = ("anti", "careless", "static_below")
POLICIES = ("tier2", "staircase85", "random")


def build_pool(pool_size=600):
    ad = BankAdapter()
    full = ad.candidates(TASK, feedback_safe=True)
    sub = np.random.default_rng(0).choice(len(full), size=pool_size,
                                          replace=False)
    return TaskCandidates(TASK, full.seg_id[sub], full.s_mean[sub],
                          full.s_sd[sub], full.y_star[sub], full.margin[sub],
                          full.coherent[sub])


# ── fast pool accuracy: p_yes on a dense s grid, GH by interpolation ──────
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(21)


def fast_pool_accuracy(lnr, pool, *, grid_n=601, lo=-9.0, hi=9.0):
    grid = np.linspace(lo, hi, grid_n)
    pg = np.array([lnr.p_yes(s, 0) for s in grid])
    nodes = pool.s_mean[:, None] + np.sqrt(2.0) * pool.s_sd[:, None] * _GH_X[None, :]
    p1 = (np.interp(nodes, grid, pg) * (_GH_W / np.sqrt(np.pi))[None, :]).sum(axis=1)
    p_correct = np.where(pool.y_star == 1, p1, 1.0 - p1)
    return float(p_correct.mean())


def _borderline_p_correct(s, s_sd, y_star):
    """P(correct) on a served item for the borderline assumed learner
    (probit, σ=σ*, t=0, λ): the evidence-gate null."""
    from scipy.stats import norm
    from training.bridge_conventions import LAPSE_RATE
    p1 = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * norm.cdf(
        s / np.sqrt(SIG_STAR ** 2 + s_sd ** 2))
    return p1 if y_star == 1 else 1.0 - p1


def run_one(member, policy, seed, pool, *, budget=BUDGET, p_static=0.0,
            evidence_z=None, ev_window=40, ev_min=30, fp_override=None,
            smear_w=False, stop_at_decl=False):
    """p_static / evidence_z: M12 mitigations (F28). evidence_z blocks
    graduation while the trailing-window OBSERVED correct count falls more
    than z·SD below what the borderline acceptable learner (σ*, t=0) would
    produce on the SAME served items — raw-outcome evidence the dynamics
    prior cannot fake."""
    lnr = ZOO[member](seed)
    fp = fp_override if fp_override is not None else assumed_params()
    rng = np.random.default_rng(1000 + seed)        # CRN: matches benchmark
    filt = TaskFilter(0.0 + 0.3 * rng.standard_normal(NPART),
                      0.0 + 0.3 * rng.standard_normal(NPART), fp, seed=seed,
                      p_static=p_static, smear_w=smear_w)
    gate = TaskModePolicy(TASK, L_STAR, SIG_STAR, ModeThresholds())
    greedyW = RewardWeights()
    state = {"k": 0, "bal": 0}
    n_true = n_decl = budget + 1
    A_decl = ell_decl = t_decl = np.nan
    pit_l, pit_t, err_l, err_t = [], [], [], []
    ev_p, ev_y = [], []                       # evidence-gate ledger
    for k in range(budget):
        state["k"] = k
        idx = _select(policy, filt, pool, state, rng, greedyW)
        s = float(pool.s_mean[idx])
        s_sd = float(pool.s_sd[idx])
        y_star = int(pool.y_star[idx])
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        gate.note_posterior(filt)
        state["bal"] += 1 if y_star == 1 else -1
        # tracking diagnostics vs the true latent state
        ell_true, th_true = -np.log(lnr.sigma[0]), -lnr.t[0]
        if k >= 30:
            pit_l.append(float((filt.w * (filt.ell <= ell_true)).sum()))
            pit_t.append(float((filt.w * (filt.theta <= th_true)).sum()))
        if k >= 50:
            mt, ml = filt.mean()
            err_l.append(ml - ell_true)
            err_t.append(mt - th_true)
        if n_true > budget and lnr.sigma[0] <= SIG_STAR and abs(lnr.t[0]) <= T_STAR:
            n_true = k + 1
        if evidence_z is not None:
            ev_p.append(_borderline_p_correct(s, s_sd, y_star))
            ev_y.append(1.0 if y == y_star else 0.0)
        if n_decl > budget and gate.is_mastered(filt):
            if evidence_z is not None:
                if len(ev_p) < ev_min:
                    continue
                pw = np.array(ev_p[-ev_window:])
                yw = np.array(ev_y[-ev_window:])
                if yw.sum() < pw.sum() - evidence_z * np.sqrt(
                        (pw * (1 - pw)).sum()):
                    continue                  # blocked: observed evidence worse
                                              # than borderline performance
            n_decl = k + 1
            A_decl = fast_pool_accuracy(lnr, pool)
            ell_decl, t_decl = float(-np.log(lnr.sigma[0])), float(lnr.t[0])
            if stop_at_decl:
                break
    A_final = fast_pool_accuracy(lnr, pool)
    pl, pt = np.array(pit_l), np.array(pit_t)
    el, et = np.array(err_l), np.array(err_t)
    out_lnr = {"_lnr": lnr} if stop_at_decl else {}
    return dict(n_true=n_true, n_decl=n_decl, A_decl=A_decl, A_final=A_final,
                ell_decl=ell_decl, t_decl=t_decl, **out_lnr,
                cov_ell=float(np.mean((pl > 0.05) & (pl < 0.95))),
                cov_t=float(np.mean((pt > 0.05) & (pt < 0.95))),
                rmse_ell=float(np.sqrt(np.mean(el ** 2))),
                rmse_t=float(np.sqrt(np.mean(et ** 2))))


# ── heterogeneity arm: population draws vs the FIXED filter ───────────────

def draw_population(n, seed=2026):
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n):
        draws.append(dict(
            alpha_t=float(np.exp(rng.uniform(np.log(0.05), np.log(0.6)))),
            alpha_sigma=float(np.exp(rng.uniform(np.log(0.015), np.log(0.18)))),
            sigma0=float(rng.uniform(1.2, 2.0)),
            t0=float(rng.uniform(-1.2, 1.2)),
            inf_frac=float(rng.uniform(0.70, 0.95)),
            lapse=float(rng.uniform(0.01, 0.06)),
            rule="soft" if rng.random() < 0.5 else "hard"))
    return draws


def run_het(d, seed, pool, *, budget=BUDGET):
    p = LearnerParams(alpha_t=d["alpha_t"], alpha_sigma=d["alpha_sigma"],
                      sigma_inf=d["inf_frac"] * SIG_STAR, q_t=0.04,
                      q_sigma=0.02, rho=0.6, lapse=d["lapse"], rule=d["rule"])
    ZOO["_het"] = lambda s: Learner([d["sigma0"]], [d["t0"]], p, seed=s)
    try:
        return run_one("_het", "tier2", seed, pool, budget=budget)
    finally:
        del ZOO["_het"]


def main(smoke=False):
    pool = build_pool()
    a_bar = accuracy_bar(pool, SIG_STAR, T_STAR)
    members = ("wellspec", "static_below", "careless") if smoke else tuple(ZOO)
    policies = ("tier2",) if smoke else POLICIES
    seeds = range(3) if smoke else range(30)
    print(f"A_bar(σ*={SIG_STAR:.3f}, t*={T_STAR}) = {a_bar:.4f}   "
          f"[{'SMOKE' if smoke else 'FULL'}]")
    metrics = ("n_true", "n_decl", "A_decl", "A_final", "ell_decl", "t_decl",
               "cov_ell", "cov_t", "rmse_ell", "rmse_t")
    res = {m: np.full((len(members), len(policies), len(list(seeds))), np.nan)
           for m in metrics}
    seeds = list(seeds)
    for i, mem in enumerate(members):
        for j, pol in enumerate(policies):
            for s in seeds:
                r = run_one(mem, pol, s, pool,
                            budget=120 if smoke else BUDGET)
                for m in metrics:
                    res[m][i, j, s] = r[m]
            bud = 120 if smoke else BUDGET
            nd = res["n_decl"][i, j]
            decl = nd <= bud
            fg = decl & (res["A_decl"][i, j] < a_bar)
            fg_lat = decl & ((res["ell_decl"][i, j] < L_STAR)
                             | (np.abs(res["t_decl"][i, j]) > T_STAR))
            print(f"  {mem:>14} {pol:>12}: declared {decl.sum()}/{len(seeds)} "
                  f"(med n_decl {np.median(nd[decl]) if decl.any() else float('nan'):.0f}), "
                  f"FG-perf {fg.sum()}, FG-latent {fg_lat.sum()}, med A_decl "
                  f"{np.nanmedian(res['A_decl'][i, j]):.3f}, "
                  f"cov_ell {np.nanmean(res['cov_ell'][i, j]):.2f}")
    if smoke:
        return
    # heterogeneity arm
    draws = draw_population(120)
    het = {m: np.full(len(draws), np.nan) for m in metrics}
    for n, d in enumerate(draws):
        r = run_het(d, 5000 + n, pool)
        for m in metrics:
            het[m][n] = r[m]
    decl = het["n_decl"] <= BUDGET
    fg = decl & (het["A_decl"] < a_bar)
    print(f"  heterogeneity (n=120): declared {decl.sum()} "
          f"(med {np.median(het['n_decl'][decl]):.0f}), FALSE-GRAD {fg.sum()}")
    np.savez("figures/data_misspec_campaign.npz",
             members=np.array(members), policies=np.array(policies),
             a_bar=a_bar, budget=BUDGET,
             **{f"zoo_{m}": res[m] for m in metrics},
             **{f"het_{m}": het[m] for m in metrics},
             het_params=np.array([[d["alpha_t"], d["alpha_sigma"], d["sigma0"],
                                   d["t0"], d["inf_frac"], d["lapse"],
                                   1.0 if d["rule"] == "hard" else 0.0]
                                  for d in draws]))
    print("saved figures/data_misspec_campaign.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
