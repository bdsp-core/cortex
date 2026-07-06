"""M18-3 — static_below early-declaration hardening under v15.

F63's residual: the σ∞-mixture stack falsely declared static_below 11/30
under v15 (vs 6–7/20 at v14) — the wider ceiling-to-cut margin halves the
prior mass below cut, slowing evidence resolution past some declaration
times. Two candidate causes, three candidate fixes, all ablated here with
GATE-BRANCH ATTRIBUTION at each declaration:

  H1 (mean-skill leak): the D16 mean-skill branch (built for v14 domain1's
     sub-floor margin, F19) fires on the mixture's strata-dragged mean ℓ̂
     before the stratum weights resolve. Under v15 every margin exceeds
     1.5× the floor, so the branch's raison d'être is gone.
     → arm ablation `meanskill_gate=False`.
  H2 (prior mass): τ=0.30 under v15 puts only ~0.12 prior mass below cut.
     → arm τ=0.45 (D29's ANCHORED_CEILING_TAU, bounded by EXTSET σ_ℓ 0.434).
  H3 (probe density): more cert probes per trial resolve strata faster.
     → arm probe_every 5→4.

Assumed filter rates = the F62 ANCHORED priors (α_t .097, α_σ .047) while
true learners keep the M17 zoo rates — the realistic fp<tp misspec (F17-safe
direction). Members: static_below (target), wellspec + powerlaw (lateness /
censoring guards). 30 seeds × 400 trials, Pool(42).

Decision rule (D28 priorities): pick the cheapest arm that minimizes
static_below FG without losing wellspec/powerlaw declarations; lateness
breaks ties.

Run:  python3 -m studies.study_m18_hardening [--smoke]
      → figures/data_m18_hardening.npz
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np

from training.bank_adapter import (ANCHORED_ALPHA_SIGMA, ANCHORED_ALPHA_T,
                                   BankAdapter, ELL_STAR_V15, SIGMA_STAR_V15,
                                   TaskCandidates)
from training.benchmark_trainer import _select
from training.bridge_conventions import engine_to_plan
from training.learner_sim import Learner, LearnerParams
from training.misspec_learners import PowerLawLearner, static_below_cut
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_greedy import RewardWeights
from training.trainer_policy import (EProcessGate, ModeThresholds,
                                     TaskModePolicy, at_bar_accuracy,
                                     cert_probe_score)

TASK = 2
ELL_STAR, SIG_STAR = ELL_STAR_V15[TASK], SIGMA_STAR_V15[TASK]
ELL_INF_V15_D3 = 0.663
_POOL = None

ARMS = {
    "base":            dict(tau=0.30, meanskill=True, probe_every=5),
    "tau45":           dict(tau=0.45, meanskill=True, probe_every=5),
    "noms":            dict(tau=0.30, meanskill=False, probe_every=5),
    "tau45+noms":      dict(tau=0.45, meanskill=False, probe_every=5),
    "tau45+noms+pe4":  dict(tau=0.45, meanskill=False, probe_every=4),
}


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
    if kind == "powerlaw":
        return PowerLawLearner([1.5], [0.6], tp, seed=seed)
    raise ValueError(kind)


def run_one(job):
    kind, arm_name, seed, budget = job
    cfg = ARMS[arm_name]
    pool = _pool()
    lnr = make_learner(kind, 800_000 + seed)
    rng = np.random.default_rng(700_000 + seed)
    # D29: assumed rates = the F62 anchored priors (fp < tp — F17-safe)
    fp = LearnerParams(alpha_t=ANCHORED_ALPHA_T,
                       alpha_sigma=ANCHORED_ALPHA_SIGMA,
                       sigma_inf=0.82 * SIG_STAR, q_t=0.04, q_sigma=0.02,
                       rho=0.6, rule="soft")
    th0 = 0.45 * rng.standard_normal(400)
    el0 = 0.45 * rng.standard_normal(400)
    filt = SigmaInfMixtureFilter(th0, el0, fp, ell_inf_mean=ELL_INF_V15_D3,
                                 tau=cfg["tau"], J=7, p_static_stratum=0.15,
                                 ell_star=ELL_STAR, seed=seed)
    mp = TaskModePolicy(0, ELL_STAR, SIG_STAR,
                        ModeThresholds(meanskill_gate=cfg["meanskill"]))
    gate = EProcessGate(alpha=0.05)
    mp.egate = gate
    W = RewardWeights()
    state = {"k": 0, "bal": 0}
    flip, n_true, out = 1, None, None
    pe = cfg["probe_every"]
    for k in range(budget):
        state["k"] = k
        sig_hat, t_hat = engine_to_plan(*filt.mean())
        if k % pe == pe - 1:
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
        if n_true is None and true_l > ELL_STAR:
            n_true = k + 1
        if out is None and mp.is_mastered(filt):
            # gate-branch attribution (H1 diagnostic)
            via_pm = filt.is_mastered(ELL_STAR, sd_floor=mp.th.sd_floor,
                                      alpha=mp.th.alpha, Z=mp.th.Z)
            via_ms = mp._meanskill_ok(filt)
            out = {"declared": True, "n_decl": k + 1, "n_true_at": n_true,
                   "via_passmass": bool(via_pm), "via_meanskill": bool(via_ms)}
    true_l = float(-np.log(lnr.sigma[0]))
    if out is None:
        out = {"declared": False, "n_decl": budget, "n_true_at": n_true,
               "via_passmass": False, "via_meanskill": False}
    out.update(kind=kind, arm=arm_name, seed=seed,
               fg=bool(out["declared"] and true_l < ELL_STAR),
               trainability=filt.trainability(ELL_STAR))
    return out


MEMBERS = ("static_below", "wellspec", "powerlaw")


def main(smoke=False):
    n_seeds = 4 if smoke else 30
    budget = 200 if smoke else 400
    jobs = [(kind, arm, s, budget) for kind in MEMBERS for arm in ARMS
            for s in range(n_seeds)]
    with Pool(6 if smoke else 42) as pool:
        rows = pool.map(run_one, jobs)
    print(f"{'member':>13} {'arm':>15} {'declared':>9} {'FG':>4} "
          f"{'med n_decl':>10} {'lateness':>9} {'viaPM':>6} {'viaMS':>6}")
    for kind in MEMBERS:
        for arm in ARMS:
            rs = [r for r in rows if r["kind"] == kind and r["arm"] == arm]
            dec = [r for r in rs if r["declared"]]
            late = [r["n_decl"] - r["n_true_at"] for r in dec
                    if r["n_true_at"] is not None]
            print(f"{kind:>13} {arm:>15} {len(dec):>6}/{n_seeds:<2} "
                  f"{sum(r['fg'] for r in rs):>4} "
                  f"{int(np.median([r['n_decl'] for r in dec])) if dec else -1:>10} "
                  f"{int(np.median(late)) if late else -1:>9} "
                  f"{sum(r['via_passmass'] for r in dec):>6} "
                  f"{sum(r['via_meanskill'] for r in dec):>6}")
    if not smoke:
        np.savez("figures/data_m18_hardening.npz",
                 rows=np.array([(MEMBERS.index(r["kind"]),
                                 list(ARMS).index(r["arm"]), r["seed"],
                                 r["declared"], r["n_decl"],
                                 -1 if r["n_true_at"] is None else r["n_true_at"],
                                 r["fg"], r["via_passmass"],
                                 r["via_meanskill"], r["trainability"])
                                for r in rows], dtype=float))
        print("saved figures/data_m18_hardening.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
