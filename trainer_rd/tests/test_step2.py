"""Step 2 smoke test (INTEGRATION_PLAN.md). Run: python3 -m tests.test_step2

  (a) w=f=1: fitted exponential to mean log-σ trajectory recovers τ_σ=1/α_σ
      within 10% over many sims (skill relaxation toward σ_∞);
  (b) balanced veridical feedback drives |t| toward 0 from a biased start;
  (c) hard and soft rules share expected dynamics (Eq. hard-unbiased) —
      mean trajectories agree within MC error;
  (d) static rule (or zero rates+noise) ⇒ state bit-stable;
  (e) σ relaxes toward σ_∞ and stays (does not overshoot to 0);
  (f) engine-coord view is the exact bridge of the plan-coord state.
"""
import numpy as np

from training.bridge_conventions import SKILL_MODE_MULTIPLIER, plan_to_engine
from training.learner_sim import Learner, LearnerParams

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


# ── (a) τ_σ recovery under ideal skill-training (w=f=1, no criterion move) ──
# Place every trial exactly at |s−t|/σ = m so skill_weight ≈ 1; recompute s
# each trial as the state drifts. Use the soft rule but feed y*=p-consistent so
# the criterion barely moves; we only care about the σ relaxation here.
alpha_sigma = 0.03
tau_true = 1.0 / alpha_sigma
sig0, sig_inf = 1.6, 0.6
n_trials, n_sims = 120, 400
logsig_curves = np.zeros((n_sims, n_trials + 1))
for r in range(n_sims):
    lp = LearnerParams(alpha_t=0.0, alpha_sigma=alpha_sigma, sigma_inf=sig_inf,
                       q_t=0.0, q_sigma=0.01, rho=0.5, rule="soft")
    lnr = Learner([sig0], [0.0], lp, seed=r)
    logsig_curves[r, 0] = np.log(lnr.sigma[0])
    for k in range(n_trials):
        s = lnr.t[0] + SKILL_MODE_MULTIPLIER * lnr.sigma[0]   # |s−t|/σ = m → w≈1
        lnr.step(s, 0, y_star=1, feedback=True)
        logsig_curves[r, k + 1] = np.log(lnr.sigma[0])
mean_curve = logsig_curves.mean(axis=0)
# fit log(σ_k − σ_∞ offset): (logσ_k − logσ_∞) = (logσ_0 − logσ_∞)·exp(−rate·k)
y = mean_curve - np.log(sig_inf)
kk = np.arange(n_trials + 1)
mask = y > 1e-3
rate = -np.polyfit(kk[mask], np.log(y[mask]), 1)[0]
tau_fit = 1.0 / rate
ok(abs(tau_fit - tau_true) / tau_true < 0.10,
   f"τ_σ recovery: fit {tau_fit:.1f} vs true {tau_true:.1f} (<10%)")

# ── (b) balanced feedback drives |t| → 0 ──
# Start biased (t=0.8); present signals symmetric about the true boundary s=0
# with veridical labels y*=1[s>0]; soft criterion update should reduce |t|.
lp = LearnerParams(alpha_t=0.1, alpha_sigma=0.0, sigma_inf=1.0,
                   q_t=0.0, q_sigma=0.0, rule="soft")
lnr = Learner([1.0], [0.8], lp, seed=1)
t_start = lnr.t[0]
rng = np.random.default_rng(0)
for k in range(400):
    s = float(rng.uniform(-2.0, 2.0))
    y_star = int(s > 0.0)
    lnr.step(s, 0, y_star=y_star, feedback=True)
t_end = lnr.t[0]
ok(abs(t_end) < abs(t_start) and abs(t_end) < 0.15,
   f"balanced feedback: |t| {abs(t_start):.2f} → {abs(t_end):.3f} (toward 0)")

# ── (c) hard vs soft share expected dynamics ──
def mean_traj(rule, n_sims=600):
    curves_t = np.zeros((n_sims, 81))
    for r in range(n_sims):
        lp = LearnerParams(alpha_t=0.08, alpha_sigma=0.0, sigma_inf=1.0,
                           q_t=0.0, q_sigma=0.0, rule=rule)
        lnr = Learner([1.0], [0.6], lp, seed=10_000 + r)
        rng = np.random.default_rng(20_000 + r)
        curves_t[r, 0] = lnr.t[0]
        for k in range(80):
            s = float(rng.uniform(-2, 2)); y_star = int(s > 0)
            lnr.step(s, 0, y_star=y_star, feedback=True)
            curves_t[r, k + 1] = lnr.t[0]
    return curves_t.mean(axis=0), curves_t.std(axis=0) / np.sqrt(n_sims)

m_soft, se_soft = mean_traj("soft")
m_hard, se_hard = mean_traj("hard")
max_z = np.max(np.abs(m_soft - m_hard) / np.sqrt(se_soft ** 2 + se_hard ** 2 + 1e-12))
ok(max_z < 4.0,
   f"hard vs soft mean t-trajectory agree (max |Δ|/SE = {max_z:.2f} < 4)")

# ── (d) static stability ──
lp = LearnerParams(rule="static")
lnr = Learner([1.3, 0.7], [0.4, -0.2], lp, seed=3)
s0, t0 = lnr.state()
for k in range(50):
    lnr.step(0.5, k % 2, y_star=1, feedback=True)
s1, t1 = lnr.state()
ok(np.array_equal(s0, s1) and np.array_equal(t0, t1),
   "static rule: (σ, t) unchanged over 50 trials")
# zero rates + zero noise on soft rule is likewise frozen
lp2 = LearnerParams(alpha_t=0.0, alpha_sigma=0.0, q_t=0.0, q_sigma=0.0, rule="soft")
lnr2 = Learner([1.0], [0.5], lp2, seed=4)
a0 = lnr2.state()
for k in range(50):
    lnr2.step(0.7, 0, y_star=0, feedback=True)
a1 = lnr2.state()
ok(np.array_equal(a0[0], a1[0]) and np.array_equal(a0[1], a1[1]),
   "zero rates + zero noise: state frozen")

# ── (e) σ relaxes to σ_∞ and does not collapse to 0 ──
lp = LearnerParams(alpha_t=0.0, alpha_sigma=0.05, sigma_inf=0.55,
                   q_t=0.0, q_sigma=0.0, rule="soft")
lnr = Learner([1.8], [0.0], lp, seed=5)
for k in range(500):
    s = lnr.t[0] + SKILL_MODE_MULTIPLIER * lnr.sigma[0]
    lnr.step(s, 0, y_star=1, feedback=True)
ok(abs(lnr.sigma[0] - 0.55) < 0.02,
   f"σ relaxes to σ_∞=0.55 (ended {lnr.sigma[0]:.3f}), no overshoot to 0")
# no-feedback ⇒ no skill change (f=0)
lp = LearnerParams(alpha_t=0.0, alpha_sigma=0.05, sigma_inf=0.55,
                   q_t=0.0, q_sigma=0.0, rule="soft")
lnr = Learner([1.8], [0.0], lp, seed=6)
for k in range(50):
    s = lnr.t[0] + SKILL_MODE_MULTIPLIER * lnr.sigma[0]
    lnr.step(s, 0, y_star=1, feedback=False)
ok(abs(lnr.sigma[0] - 1.8) < 1e-12, "no feedback (f=0) ⇒ skill frozen")

# ── (f) engine-coord view ──
lnr = Learner([1.2, 0.8], [0.3, -0.5], LearnerParams(rule="static"), seed=7)
th, el = lnr.engine_state()
th_b, el_b = plan_to_engine(np.array([1.2, 0.8]), np.array([0.3, -0.5]))
ok(np.allclose(th, th_b) and np.allclose(el, el_b),
   "engine_state() == bridge plan_to_engine (θ=−t, ℓ=−logσ)")

print(f"\nSTEP 2 SMOKE TEST PASSED — {checks} checks "
      f"(τ_σ fit {tau_fit:.1f}/{tau_true:.0f}, hard≈soft z={max_z:.1f}).")
