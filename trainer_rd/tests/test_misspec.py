"""M12/V1 smoke — mis-specified learner zoo sanity (axis-by-axis).

Each check verifies the ONE deviation a zoo member is supposed to embody,
plus the pool-accuracy ground-truth machinery (GH smearing vs closed form).
Run: python3 -m tests.test_misspec
"""
import numpy as np
from scipy.stats import norm
from types import SimpleNamespace

from training.bridge_conventions import LAPSE_RATE
from training.learner_sim import Learner, LearnerParams
from training.misspec_learners import (AntiLearner, AsymmetricLapseLearner,
                              DriftingCeilingLearner, FatigueLearner,
                              HeavyTailLinkLearner, MomentumCriterionLearner,
                              PlateauLearner, PowerLawLearner, accuracy_bar,
                              careless_learner, pool_accuracy, smeared_p_yes,
                              static_below_cut)

PASS = 0


def ok(cond, msg):
    global PASS
    assert cond, f"FAIL: {msg}"
    PASS += 1
    print(f"  ok: {msg}")


P = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.5, q_t=0.0,
                  q_sigma=0.0, rho=0.6, rule="soft")          # noise off → deterministic curves
P_N = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.5, q_t=0.04,
                    q_sigma=0.02, rho=0.6, rule="soft")

pool = SimpleNamespace(s_mean=np.linspace(-2.5, 2.5, 41),
                       s_sd=np.full(41, 0.8),
                       y_star=(np.linspace(-2.5, 2.5, 41) > 0).astype(int))

# ── ground-truth machinery ──
base = Learner([0.7], [0.0], P, seed=1)
gh = smeared_p_yes(base, pool.s_mean, pool.s_sd)
closed = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * norm.cdf(
    pool.s_mean / np.sqrt(0.7 ** 2 + 0.8 ** 2))
ok(np.max(np.abs(gh - closed)) < 1e-6,
   f"GH smearing matches probit closed form (max err {np.max(np.abs(gh - closed)):.1e})")
abar = accuracy_bar(pool, 0.7)
ok(0.6 < abar < 0.95, f"accuracy bar sensible (A_bar={abar:.3f})")
ok(abs(pool_accuracy(base, pool) - abar) < 1e-6,
   "borderline assumed learner sits exactly AT the bar (A == A_bar)")

# ── power-law: early ≥ exponential progress, late slower (heavy tail) ──
def run_sigma(lnr, n=300):
    for k in range(n):
        side = 1 if k % 2 == 0 else -1
        s = side * 1.077 * lnr.sigma[0]            # ~85% placement, w≈1
        lnr.step(s, 0, int(s > 0), feedback=True)
    return float(lnr.sigma[0])

pl_120 = PowerLawLearner([1.5], [0.0], P, seed=2)
ex_120 = Learner([1.5], [0.0], P, seed=2)
s_pl, s_ex = run_sigma(pl_120, 300), run_sigma(ex_120, 300)
ok(s_ex < 0.55, f"exponential learner converged by 300 weighted trials (σ={s_ex:.3f})")
ok(s_pl > s_ex + 0.05,
   f"power-law tail converges SLOWER than exponential (σ_pl={s_pl:.3f} vs σ_exp={s_ex:.3f})")
ok(pl_120.sigma[0] > P.sigma_inf, "power-law σ stays above the floor")

# ── plateau: discrete levels, monotone, ends at floor ──
plat = PlateauLearner([1.5], [0.0], P, seed=3)
trace = []
for k in range(300):
    side = 1 if k % 2 == 0 else -1
    s = side * 1.077 * plat.sigma[0]
    plat.step(s, 0, int(s > 0), feedback=True)
    trace.append(plat.sigma[0])
lv = sorted(set(np.round(trace, 6)))
ok(len(lv) <= 3, f"plateau learner visits ≤3 discrete σ levels ({len(lv)})")
ok(abs(trace[-1] - 0.5) < 1e-9, f"plateau learner ends at the floor (σ={trace[-1]:.3f})")

# ── drifting ceiling ──
dc = DriftingCeilingLearner([1.5], [0.0], P_N, drift_per_trial=3.7e-4,
                            ceil_cap=0.62, seed=4)
inf0 = dc.sigma_inf[0]
for k in range(400):
    dc.step(0.5, 0, 1, feedback=True)
ok(dc.sigma_inf[0] > inf0 * 1.1, f"ceiling drifted up ({inf0:.3f}→{dc.sigma_inf[0]:.3f})")
ok(dc.sigma_inf[0] <= 0.62 + 1e-9, "ceiling respects the cap")

# ── momentum criterion: sluggish first response, same long-run attractor ──
mo = MomentumCriterionLearner([0.7], [0.5], P, kappa=0.15, seed=5)
bs = Learner([0.7], [0.5], P, seed=5)
m0, b0 = mo.t[0], bs.t[0]
mo.step(0.0, 0, 1, feedback=True)        # corrective trial (y*=1 at boundary)
bs.step(0.0, 0, 1, feedback=True)
ok(abs(mo.t[0] - m0) < abs(bs.t[0] - b0) * 0.5,
   f"momentum transient sluggish (|Δt| {abs(mo.t[0]-m0):.4f} vs base {abs(bs.t[0]-b0):.4f})")
for k in range(600):
    s = (1 if k % 2 == 0 else -1) * 0.8
    mo.step(s, 0, int(s > 0), feedback=True)
ok(abs(mo.t[0]) < 0.15, f"momentum learner still calibrates under balanced stream (|t|={abs(mo.t[0]):.3f})")

# ── heavy-tail link: slope-matched at 0, fatter error tail ──
ht = HeavyTailLinkLearner([1.0], [0.0], P, seed=6)
pb = Learner([1.0], [0.0], P, seed=6)
ok(abs(ht.p_yes(0.05, 0) - pb.p_yes(0.05, 0)) < 2e-3,
   "t3 link slope-matched at the boundary")
ok(ht.p_yes(3.0, 0) < pb.p_yes(3.0, 0) - 0.02,
   f"t3 link errs more on easy items (p={ht.p_yes(3.0,0):.3f} vs probit {pb.p_yes(3.0,0):.3f})")

# ── asymmetric lapse: floor/ceiling asymmetry ──
al = AsymmetricLapseLearner([1.0], [0.0], P, lam_fa=0.08, lam_miss=0.01, seed=7)
ok(abs(al.p_yes(-30.0, 0) - 0.08) < 1e-6 and abs(al.p_yes(30.0, 0) - 0.99) < 1e-6,
   "asymmetric lapse floors at λ_fa=0.08 / ceilings at 1−λ_miss=0.99")

# ── fatigue: accuracy on easy items decays with time-on-task ──
fa = FatigueLearner([1.0], [0.0], P_N, lam_max=0.12, n_ramp=300, seed=8)
p_early = fa.p_yes(3.0, 0)
for _ in range(300):
    fa.step(0.5, 0, 1, feedback=True)
ok(fa.p_yes(3.0, 0) < p_early - 0.05,
   f"fatigue degrades easy-item accuracy ({p_early:.3f}→{fa.p_yes(3.0,0):.3f})")

# ── adversarial ──
an = AntiLearner([1.0], [0.3], P_N, seed=9)
for k in range(150):
    s = (1 if k % 2 == 0 else -1) * 0.9
    an.step(s, 0, int(s > 0), feedback=True)
ok(abs(an.t[0]) > 0.6, f"anti-learner criterion diverges (|t|={abs(an.t[0]):.2f})")
ok(abs(an.t[0]) <= 4.0, "anti-learner clipped (finite)")

cl = careless_learner([1.0], [0.2], P_N, seed=10)
ok(pool_accuracy(cl, pool) < accuracy_bar(pool, 0.7) - 0.05,
   f"careless learner can never reach the bar (A={pool_accuracy(cl, pool):.3f} < {accuracy_bar(pool,0.7):.3f})")

st = static_below_cut(0.7, seed=11)
s0, t0 = st.sigma[0], st.t[0]
for _ in range(50):
    st.step(0.5, 0, 1, feedback=True)
ok(st.sigma[0] == s0 and st.t[0] == t0, "static-below-cut never moves")

print(f"\nMISSPEC ZOO SMOKE TEST PASSED — {PASS} checks.")
