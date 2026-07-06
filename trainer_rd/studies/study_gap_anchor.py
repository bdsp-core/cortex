"""M17 (D27) — GapAnchor validation: multi-session protocol with real gaps.

Protocol per the user-ratified product shape: open-ended training, 40-trial
sessions whenever convenient. Simulated here as 8 sessions with gaps drawn
from {0.5, 1, 3, 7, 21} days. TRUE forgetting is a POWER LAW
    r_true(Δt) = (1 + Δt/τ_f)^(−β),   τ_f = 3 d, β = 0.7 (±20% per person),
applied to the true state toward personal baselines — deliberately OFF the
anchor's 5-point grid and OFF its exponential prior tilt, so the study
measures the measurement-based design under honest misspecification.

Arms (same learner, same gap draws, CRN):
  identity   pre-M17 default: belief carried across the gap unchanged
  widen      propagate_gap(widen=0.15) — variance-only hedge (F18 residual)
  anchor     GapAnchor: retention-hypothesis strata + adaptive probe block

Metrics (M17 priorities FG ≥ lateness > speed > retention):
  post-gap tracking error |t̂−t|, |ℓ̂−ℓ| over the first 10 trials of each
  returning session; false 'still-mastered' calls at session open (the gate
  believes a decayed learner is above bar — the F18/F34 staleness failure);
  anchor overhead (probe trials); learned S vs the true curve.

Run:  python3 -m studies.study_gap_anchor [--smoke]
      → figures/data_gap_anchor.npz
"""
from __future__ import annotations

import sys

import numpy as np

from training.bank_adapter import (BankAdapter, TaskCandidates,
                                   ELL_STAR_V15, SIGMA_STAR_V15)
from training.benchmark_trainer import _select
from training.bridge_conventions import engine_to_plan
from training.gap_anchor import GapAnchor
from training.learner_sim import Learner, LearnerParams
from training.trainer_greedy import RewardWeights
from training.trainer_policy import ModeThresholds, TaskModePolicy
from training.training_filter import TaskFilter

TASK = 2
DAY = 86_400.0
GAPS_D = (0.5, 1.0, 3.0, 7.0, 21.0)


def build_pool(seed=0, size=600):
    ad = BankAdapter()
    full = ad.candidates(TASK, feedback_safe=True)
    sub = np.random.default_rng(seed).choice(len(full), size=size,
                                             replace=False)
    return TaskCandidates(TASK, full.seg_id[sub], full.s_mean[sub],
                          full.s_sd[sub], full.y_star[sub], full.margin[sub],
                          full.coherent[sub])


def run_one(arm, pool, *, seed=0, n_sessions=8, n_per_session=40):
    ell_star, sig_star = ELL_STAR_V15[TASK], SIGMA_STAR_V15[TASK]
    r = np.random.default_rng(40_000 + seed)
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.82 * sig_star,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    sigma0, t0 = 1.5, float(r.choice([-1, 1]) * 0.65)
    lnr = Learner([sigma0], [t0], tp, seed=41_000 + seed)
    ell_b, t_b = -np.log(sigma0), t0                    # personal baselines
    beta = 0.7 * (1 + 0.2 * r.standard_normal())        # true power law
    tau_f = 3.0 * DAY
    filt = TaskFilter(0.45 * r.standard_normal(400),
                      0.45 * r.standard_normal(400), tp, seed=42_000 + seed)
    mp = TaskModePolicy(0, ell_star, sig_star, ModeThresholds())
    anchor = GapAnchor() if arm == "anchor" else None
    gaps = r.choice(GAPS_D, size=n_sessions - 1)
    state = {"k": 0, "bal": 0}
    W = RewardWeights()
    err_t, err_l, stale_calls, n_probe = [], [], 0, 0
    for sess in range(n_sessions):
        if sess > 0:
            dt = float(gaps[sess - 1]) * DAY
            # TRUE forgetting (power law toward baselines, plan coords)
            r_true = (1.0 + dt / tau_f) ** (-beta)
            ell_now = -np.log(lnr.sigma[0])
            lnr.sigma[0] = float(np.exp(-(ell_b + r_true * (ell_now - ell_b))))
            lnr.t[0] = t_b + r_true * (lnr.t[0] - t_b)
            n_anchor = 0
            if arm == "widen":
                filt.propagate_gap(dt, widen=0.15)
            elif arm == "anchor":
                n_anchor = anchor.open_session(filt, dt, theta_base=-t_b,
                                               ell_base=ell_b,
                                               seed=43_000 + seed + sess)
            # staleness check at session open: does the gate think a
            # (possibly decayed) learner is still mastered?
            gate_filter = filt if arm != "anchor" else None
            if gate_filter is not None and mp.is_mastered(gate_filter):
                true_ok = (-np.log(lnr.sigma[0]) > ell_star)
                stale_calls += (not true_ok)
        for k in range(n_per_session):
            state["k"] = state.get("k", 0) + 1
            in_anchor = (arm == "anchor" and sess > 0 and k < n_anchor
                         and anchor._strata is not None)
            if in_anchor:
                idx = anchor.pick_probe(pool, sig_star)
                n_probe += 1
            else:
                idx = _select("tier2", filt, pool, state, r, W)
            s, s_sd = float(pool.s_mean[idx]), float(pool.s_sd[idx])
            y_star = int(pool.y_star[idx])
            s_real = s + s_sd * r.standard_normal()
            y = lnr.step(s_real, 0, y_star, feedback=True)
            if in_anchor:
                anchor.anchor_step(s, y, y_star, s_sd=s_sd)
                if k == n_anchor - 1:
                    filt, _ = anchor.close_anchor(seed=44_000 + seed + sess)
            else:
                filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
            mp.note_posterior(filt)
            state["bal"] += 1 if y_star == 1 else -1
            if sess > 0 and k < 10:
                if (arm == "anchor" and anchor._strata is not None):
                    mt, ml = anchor.mean()
                else:
                    mt, ml = filt.mean()
                sig_hat, t_hat = engine_to_plan(mt, ml)
                err_t.append(abs(t_hat - float(lnr.t[0])))
                err_l.append(abs(ml - float(-np.log(lnr.sigma[0]))))
    S_learned = anchor.S / DAY if anchor is not None else np.nan
    return {"err_t": float(np.mean(err_t)), "err_l": float(np.mean(err_l)),
            "stale": stale_calls, "n_probe": n_probe, "S_days": S_learned}


def main(smoke=False):
    n_seeds = 4 if smoke else 20
    pool = build_pool()
    print(f"{'arm':>9} {'|t̂−t| post-gap':>15} {'|ℓ̂−ℓ| post-gap':>15} "
          f"{'stale-mastered':>14} {'probes':>7} {'S_days med':>10}")
    out = {}
    for arm in ("identity", "widen", "anchor"):
        rows = [run_one(arm, pool, seed=s) for s in range(n_seeds)]
        out[arm] = rows
        print(f"{arm:>9} {np.mean([r['err_t'] for r in rows]):15.3f} "
              f"{np.mean([r['err_l'] for r in rows]):15.3f} "
              f"{sum(r['stale'] for r in rows):>14} "
              f"{int(np.mean([r['n_probe'] for r in rows])):>7} "
              f"{np.nanmedian([r['S_days'] for r in rows]):>10.1f}")
    if not smoke:
        np.savez("figures/data_gap_anchor.npz",
                 **{f"{a}_{k}": np.array([r[k] for r in rows])
                    for a, rows in out.items()
                    for k in ("err_t", "err_l", "stale", "n_probe")})
        print("saved figures/data_gap_anchor.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
