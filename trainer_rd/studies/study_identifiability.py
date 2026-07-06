"""Step 4 — identifiability gate (PROJECT_MEMORY.md F11/F13; plan caveat §6.3).

The risk: when BOTH σ_k and t_k move during training, a drop in error rate is
explained either by σ shrinking (sharper discrimination) or by |t| → 0 (less
bias). If the filter cannot disentangle them, the trainer's mode decisions and
mastery verdicts are corrupted. This campaign verifies joint recoverability and
its robustness to ±2× misspecified learning rates.

Diagnostics per (cell, replicate):
  * terminal coverage  — is the true terminal (σ,t) inside the filter's 95%
    credible box?
  * trajectory RMSE    — mean |posterior-mean − truth| over the run;
  * error confusion    — across replicates, correlation between the σ-error and
    t-error of the posterior mean. High |corr| ⇒ the filter trades one for the
    other (the confusion failure mode).
  * within-posterior coupling — mean |corr(θ, ℓ)| inside the particle cloud.

Item stream drives BOTH parameters: interleave bias-mode trials (near t̂, both
label signs, balanced — moves t) and skill-mode trials (|s−t̂|≈1.077·σ̂ — moves σ).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.bridge_conventions import SKILL_MODE_MULTIPLIER, engine_to_plan
from training.learner_sim import Learner, LearnerParams
from training.training_filter import TaskFilter


def _wquantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return float(np.interp(q, cw, v[o]))


@dataclass
class CellResult:
    alpha_t: float
    tau_sigma: float
    sigma0: float
    t0: float
    rate_mult: float           # filter's misspecification factor (×true rates)
    coverage: float            # terminal 95%-box hit rate
    rmse_sigma: float
    rmse_t: float
    err_corr: float            # across-replicate corr(σ-err, t-err)
    cloud_coupling: float      # mean |corr(θ,ℓ)| in the posterior


def run_cell(alpha_t, tau_sigma, sigma0, t0, *, rate_mult=1.0, sigma_inf=0.5,
             n_reps=30, n_trials=120, Npart=600, prior_sd=0.25, seed0=0):
    alpha_sigma = 1.0 / tau_sigma
    true_params = dict(alpha_t=alpha_t, alpha_sigma=alpha_sigma,
                       sigma_inf=sigma_inf, q_t=0.05, q_sigma=0.03,
                       rho=0.5, rule="soft")
    # filter's assumed rates are misspecified by rate_mult (F13)
    filt_params = dict(true_params)
    filt_params["alpha_t"] = alpha_t * rate_mult
    filt_params["alpha_sigma"] = alpha_sigma * rate_mult

    sig_err, t_err = [], []
    cover, rmse_s, rmse_t, coupling = [], [], [], []
    for r in range(n_reps):
        lnr = Learner([sigma0], [t0], LearnerParams(**true_params), seed=seed0 + r)
        rng = np.random.default_rng(10_000 + seed0 + r)
        th0 = -t0 + prior_sd * rng.standard_normal(Npart)
        el0 = -np.log(sigma0) + prior_sd * rng.standard_normal(Npart)
        filt = TaskFilter(th0, el0, LearnerParams(**filt_params),
                          ess_frac=0.5, seed=50_000 + seed0 + r)
        run_s, run_t = [], []
        for k in range(n_trials):
            mt, ml = filt.mean()
            sig_hat, t_hat = engine_to_plan(mt, ml)
            if k % 2 == 0:                         # bias-mode: near t̂, sign alt
                s = t_hat + (0.3 if (k // 2) % 2 == 0 else -0.3) * sig_hat
            else:                                  # skill-mode: 85% point, sign alt
                s = t_hat + (1 if (k // 2) % 2 == 0 else -1) * SKILL_MODE_MULTIPLIER * sig_hat
            y_star = int(s > 0.0)
            y = lnr.step(s, 0, y_star=y_star, feedback=True)
            filt.step(s, y, y_star, feedback=True)
            sig_m, t_m = filt.plan_mean()
            run_s.append((sig_m - lnr.sigma[0]) ** 2)
            run_t.append((t_m - lnr.t[0]) ** 2)
        # terminal coverage
        sig_lo = np.exp(-_wquantile(filt.ell, filt.w, 0.975))
        sig_hi = np.exp(-_wquantile(filt.ell, filt.w, 0.025))
        t_lo = -_wquantile(filt.theta, filt.w, 0.975)
        t_hi = -_wquantile(filt.theta, filt.w, 0.025)
        cover.append(int(sig_lo <= lnr.sigma[0] <= sig_hi and
                         t_lo <= lnr.t[0] <= t_hi))
        sig_m, t_m = filt.plan_mean()
        sig_err.append(sig_m - lnr.sigma[0])
        t_err.append(t_m - lnr.t[0])
        rmse_s.append(np.sqrt(np.mean(run_s)))
        rmse_t.append(np.sqrt(np.mean(run_t)))
        c = np.corrcoef(filt.theta, filt.ell)[0, 1]
        coupling.append(abs(c) if np.isfinite(c) else 0.0)

    sig_err, t_err = np.array(sig_err), np.array(t_err)
    if sig_err.std() > 1e-9 and t_err.std() > 1e-9:
        ec = float(np.corrcoef(sig_err, t_err)[0, 1])
    else:
        ec = 0.0
    return CellResult(alpha_t, tau_sigma, sigma0, t0, rate_mult,
                      float(np.mean(cover)), float(np.mean(rmse_s)),
                      float(np.mean(rmse_t)), ec, float(np.mean(coupling)))


def run_campaign(*, rate_mults=(0.5, 1.0, 2.0), n_reps=30, n_trials=120, seed0=0):
    grid = [
        dict(alpha_t=0.10, tau_sigma=40, sigma0=1.4, t0=0.6),
        dict(alpha_t=0.20, tau_sigma=60, sigma0=1.2, t0=-0.8),
        dict(alpha_t=0.15, tau_sigma=30, sigma0=1.6, t0=0.3),
    ]
    results = []
    for cell in grid:
        for rm in rate_mults:
            results.append(run_cell(**cell, rate_mult=rm, n_reps=n_reps,
                                    n_trials=n_trials, seed0=seed0))
    return results


if __name__ == "__main__":
    res = run_campaign(n_reps=20)
    print(f"{'α_t':>5} {'τ_σ':>5} {'σ0':>4} {'t0':>5} {'×rate':>6} "
          f"{'cover':>6} {'rmseσ':>6} {'rmseT':>6} {'errC':>6} {'cpl':>5}")
    for r in res:
        print(f"{r.alpha_t:5.2f} {r.tau_sigma:5.0f} {r.sigma0:4.1f} {r.t0:5.1f} "
              f"{r.rate_mult:6.1f} {r.coverage:6.2f} {r.rmse_sigma:6.3f} "
              f"{r.rmse_t:6.3f} {r.err_corr:6.2f} {r.cloud_coupling:5.2f}")
