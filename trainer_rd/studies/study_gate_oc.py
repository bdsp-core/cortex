"""M12/V3 — operating characteristic of trainer graduation (rung 4).

Extends the F19 sandbox (two learner positions) to a full OC curve: a
well-specified soft learner whose TRUE skill ceiling sits Δ above/below the
certification cut, ℓ_∞ = ℓ* + Δ, trained by tier2 until the gate declares
mastery (or budget 400). Δ < 0 ⇒ mastery unreachable: P(declare) is the
gate's FALSE-PASS rate. Δ > 0 ⇒ P(declare) is its POWER, with time-to-declare.

Arms:
  shipped — filter at the population prior σ_∞ = 0.82σ* (production reality:
            the individual ceiling is unknown), shipped D16 gate
  oracle  — filter told the TRUE individual ceiling (the F19 sandbox premise;
            isolates gate logic from filter misspecification)
  hardened— shipped prior + M12 mitigations (p_static=0.3, evidence z=1.28)

Run: python3 -m studies.study_gate_oc [--smoke]  → figures/data_gate_oc.npz
"""
from __future__ import annotations

import sys

import numpy as np

from training.benchmark_trainer import T_STAR
from training.learner_sim import Learner, LearnerParams
from studies.study_misspec import (BUDGET, L_STAR, SIG_STAR, ZOO, assumed_params,
                           build_pool, run_one)

DELTAS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30)
ARMS = ("shipped", "oracle", "hardened")
SIGMA0, T0 = 1.5, 0.4


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(c - h, 0.0), min(c + h, 1.0))


def true_params_delta(delta):
    return LearnerParams(alpha_t=0.2, alpha_sigma=0.06,
                         sigma_inf=float(np.exp(-(L_STAR + delta))),
                         q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")


def main(smoke=False, arms=ARMS, out="figures/data_gate_oc.npz"):
    pool = build_pool()
    deltas = (-0.10, 0.10) if smoke else DELTAS
    seeds = list(range(4 if smoke else 40))
    budget = 200 if smoke else BUDGET
    P_decl = np.full((len(arms), len(deltas)), np.nan)
    CI = np.full((len(arms), len(deltas), 2), np.nan)
    MED_N = np.full((len(arms), len(deltas)), np.nan)
    FG_LAT = np.full((len(arms), len(deltas)), np.nan)
    for a, arm in enumerate(arms):
        for d, delta in enumerate(deltas):
            tp = true_params_delta(delta)
            ZOO["_oc"] = lambda s, tp=tp: Learner([SIGMA0], [T0], tp, seed=s)
            kw = {}
            if arm == "oracle":
                fo = assumed_params()
                fo.sigma_inf = tp.sigma_inf
                kw["fp_override"] = fo
            elif arm == "hardened":
                kw.update(p_static=0.3, evidence_z=1.28)
            elif arm == "hardened3":
                kw.update(p_static=0.3, evidence_z=1.28, smear_w=True)
            nd, fg = [], []
            for s in seeds:
                r = run_one("_oc", "tier2", s, pool, budget=budget, **kw)
                nd.append(r["n_decl"])
                fg.append(r["n_decl"] <= budget
                          and (r["ell_decl"] < L_STAR
                               or abs(r["t_decl"]) > T_STAR))
            del ZOO["_oc"]
            nd = np.array(nd)
            k = int((nd <= budget).sum())
            P_decl[a, d] = k / len(seeds)
            CI[a, d] = wilson(k, len(seeds))
            MED_N[a, d] = float(np.median(nd[nd <= budget])) if k else np.nan
            FG_LAT[a, d] = float(np.mean(fg))
            print(f"  {arm:>8} Δ={delta:+.2f}: P(declare)={P_decl[a,d]:.2f} "
                  f"[{CI[a,d,0]:.2f},{CI[a,d,1]:.2f}]  med_n={MED_N[a,d]:.0f}  "
                  f"FG-latent={FG_LAT[a,d]:.2f}")
    if not smoke:
        np.savez(out, arms=np.array(arms), deltas=np.array(deltas),
                 p_decl=P_decl, ci=CI, med_n=MED_N, fg_lat=FG_LAT,
                 budget=budget, n_seeds=len(seeds))
        print(f"saved {out}")


if __name__ == "__main__":
    if "--hardened3" in sys.argv:
        main(arms=("hardened3",), out="figures/data_gate_oc2.npz")
    else:
        main(smoke="--smoke" in sys.argv)
