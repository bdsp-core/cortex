"""Step 3 smoke test (INTEGRATION_PLAN.md). Run: python3 -m tests.test_step3

  (a) tracking: filter follows a Step-2 drifting learner — bounded mean RMSE +
      95% credible coverage of true (σ,t) in a calibrated band over replicates;
  (b) static correctness: with the static rule (no propagation), the bootstrap
      filter's posterior matches a fine-grid numerical posterior on the same
      data + prior (moments within tolerance) — i.e. the Bayes step is right;
  (c) variance floor (F5): steady_state_sd measured + recorded;
  (d) real careless rater (acc 0.23): filter certifies NO task (F14/R8).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from training.bank_adapter import BankAdapter, TASK_CODES, ELL_STAR, SIGMA_INF
from training.bridge_conventions import LAPSE_RATE, engine_to_plan
from training.learner_sim import Learner, LearnerParams
from training.training_filter import TaskFilter, steady_state_sd

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


def wquantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o]); cw /= cw[-1]
    return float(np.interp(q, cw, v[o]))


# ── (a) tracking + coverage ──
n_sims, n_trials, Npart = 80, 100, 600
prior_sd = 0.25                       # eval-seed-like posterior width (engine ℓ,θ)
inside_sig = inside_t = total = 0
rmse_sig, rmse_t = [], []
for r in range(n_sims):
    params = LearnerParams(alpha_t=0.1, alpha_sigma=0.03, sigma_inf=0.6,
                           q_t=0.05, q_sigma=0.03, rho=0.5, rule="soft")
    lnr = Learner([1.4], [0.5], params, seed=r)
    rng = np.random.default_rng(1000 + r)
    th0 = -lnr.t[0] + prior_sd * rng.standard_normal(Npart)
    el0 = -np.log(lnr.sigma[0]) + prior_sd * rng.standard_normal(Npart)
    filt = TaskFilter(th0, el0, params, ess_frac=0.5, seed=5000 + r)
    sign = 1
    for k in range(n_trials):
        # closed-loop skill-mode placement around the filter's current estimate
        mt, ml = filt.mean()
        sig_hat, t_hat = engine_to_plan(mt, ml)
        s = t_hat + sign * 1.0772 * sig_hat
        sign *= -1
        y_star = int(s > 0.0)
        y = lnr.step(s, 0, y_star=y_star, feedback=True)
        filt.step(s, y, y_star, feedback=True)
        if k >= 20:                         # post burn-in
            sig_lo = np.exp(-wquantile(filt.ell, filt.w, 0.975))
            sig_hi = np.exp(-wquantile(filt.ell, filt.w, 0.025))
            t_lo = -wquantile(filt.theta, filt.w, 0.975)
            t_hi = -wquantile(filt.theta, filt.w, 0.025)
            inside_sig += int(sig_lo <= lnr.sigma[0] <= sig_hi)
            inside_t += int(t_lo <= lnr.t[0] <= t_hi)
            total += 1
            sig_m, t_m = filt.plan_mean()
            rmse_sig.append((sig_m - lnr.sigma[0]) ** 2)
            rmse_t.append((t_m - lnr.t[0]) ** 2)
cov_sig, cov_t = inside_sig / total, inside_t / total
rms_sig, rms_t = np.sqrt(np.mean(rmse_sig)), np.sqrt(np.mean(rmse_t))
print(f"  coverage σ={cov_sig:.3f} t={cov_t:.3f} | RMSE σ={rms_sig:.3f} t={rms_t:.3f}")
ok(0.85 <= cov_sig <= 0.99 and 0.85 <= cov_t <= 0.99,
   f"95% credible coverage calibrated (σ {cov_sig:.2f}, t {cov_t:.2f} ∈ [.85,.99])")
ok(rms_sig < 0.25 and rms_t < 0.30, f"tracking RMSE bounded (σ {rms_sig:.2f}, t {rms_t:.2f})")

# ── (b) static correctness vs grid posterior ──
rng = np.random.default_rng(0)
data = []
true_t, true_l = 0.3, 0.6
for _ in range(40):
    s = float(rng.uniform(-2, 2))
    z = np.exp(true_l) * (s + true_t)
    p = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * norm.cdf(z)
    data.append((s, int(rng.random() < p)))
# bootstrap filter, static rule, N(0,1)^2 prior
Np = 4000
th0 = rng.standard_normal(Np); el0 = rng.standard_normal(Np)
filt = TaskFilter(th0, el0, LearnerParams(rule="static"), ess_frac=0.5, seed=1)
for s, y in data:
    filt.step(s, y, y_star=y)          # static ⇒ no propagation
fm_t, fm_l = filt.mean()
fs_t, fs_l = filt.sd()
# grid reference posterior under the same N(0,1)^2 prior
g = np.linspace(-3, 3, 241)
TH, EL = np.meshgrid(g, g, indexing="ij")
logp = -0.5 * (TH ** 2 + EL ** 2)
for s, y in data:
    z = np.exp(EL) * (s + TH)
    p = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * norm.cdf(z)
    logp += np.log(p if y == 1 else (1 - p))
P = np.exp(logp - logp.max()); P /= P.sum()
gm_t = float((P * TH).sum()); gm_l = float((P * EL).sum())
gs_t = float(np.sqrt((P * (TH - gm_t) ** 2).sum()))
gs_l = float(np.sqrt((P * (EL - gm_l) ** 2).sum()))
print(f"  filter (θ,ℓ) mean=({fm_t:.3f},{fm_l:.3f}) grid=({gm_t:.3f},{gm_l:.3f})")
ok(abs(fm_t - gm_t) < 0.06 and abs(fm_l - gm_l) < 0.06,
   "static-filter posterior MEAN matches grid (|Δ|<0.06)")
ok(abs(fs_t - gs_t) < 0.06 and abs(fs_l - gs_l) < 0.06,
   "static-filter posterior SD matches grid (|Δ|<0.06)")

# ── (c) variance floor ──
params = LearnerParams(q_t=0.05, q_sigma=0.03, rule="soft")
floor_t, floor_l = steady_state_sd(params, n_particles=1500, n_trials=300, seed=2)
print(f"  steady-state SD floor: θ={floor_t:.3f}, ℓ={floor_l:.3f}")
ok(floor_l > 0 and floor_t > 0 and floor_l < 1.0,
   f"variance floor measured (ℓ-SD floor {floor_l:.3f}); mastery sd_floor must exceed it")

# ── (d) real careless rater certifies nothing ──
ad = BankAdapter()
ses = "859ca73f-ba36-4670-9bfb-2072e252c595_trials_anonymized.csv"
t = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                             "sessions", ses))
filt_by_task = {}
for k in range(7):
    params = LearnerParams(alpha_t=0.1, alpha_sigma=0.03, sigma_inf=SIGMA_INF[k],
                           q_t=0.05, q_sigma=0.03, rule="soft")
    filt_by_task[k] = TaskFilter(
        np.zeros(600), np.zeros(600), params, ess_frac=0.5, seed=7000 + k)
code_idx = {c: i for i, c in enumerate(TASK_CODES)}
mastered = []
for _, row in t.iterrows():
    k = code_idx[row["task_code"]]
    pool = ad.task_pool(k)
    hit = np.where(pool.seg_id == int(row["seg_id"]))[0]
    if hit.size == 0:
        continue
    y_star = int(pool.y_star[hit[0]])
    s, s_sd = float(pool.s_mean[hit[0]]), float(pool.s_sd[hit[0]])
    filt_by_task[k].step(s, int(row["response_y"]), y_star, s_sd=s_sd, feedback=True)
ell_recovered = [filt_by_task[k].mean()[1] for k in range(7)]
for k in range(7):
    if filt_by_task[k].is_mastered(ELL_STAR[k], sd_floor=max(floor_l * 1.5, 0.15)):
        mastered.append(TASK_CODES[k])
print(f"  careless rater recovered ℓ: {[round(e,2) for e in ell_recovered]}")
ok(len(mastered) == 0, f"careless rater (acc 0.23) certifies NO task (mastered={mastered})")
ok(np.mean(ell_recovered) < np.mean(ELL_STAR),
   "recovered mean skill below the mean cut-score (correctly judged sub-threshold)")

print(f"\nSTEP 3 SMOKE TEST PASSED — {checks} checks "
      f"(cov σ={cov_sig:.2f}/t={cov_t:.2f}, ℓ-floor {floor_l:.3f}).")
