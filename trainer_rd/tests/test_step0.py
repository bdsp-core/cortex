"""Step 0 smoke test (INTEGRATION_PLAN.md). Run: python3 -m tests.test_step0

Verifies:
  1. lapse-rate consistency between bridge and engine;
  2. round-trip parameter conversions (engine↔plan) to 1e-12;
  3. P(yes) agreement: plan form == engine form == engine's _p_response_yes;
  4. sign-convention direction (the F3 bug class): ∂P/∂t < 0, ∂P/∂θ > 0;
  5. difficulty multipliers: exact inverse of accuracy, and the four named
     constants from PROJECT_MEMORY.md §2;
  6. AUROC: value at ℓ=0, bridge==vendored auroc module, monotonicity;
  7. weighted AUROC quantiles sane (degenerate cloud, ordering);
  8. the engine end-to-end with the vendored auroc stub (tiny hier session).
"""
import numpy as np
from scipy.stats import norm

import engine.auroc as auroc_mod
import training.bridge_conventions as bc
import engine.core_mcmc_general as eng

rng = np.random.default_rng(0)
checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


# 1 ── lapse consistency
ok(bc.LAPSE_RATE == eng.LAPSE_RATE == 0.025, "LAPSE_RATE identical (0.025) in bridge and engine")

# 2 ── round-trip conversions
theta = rng.normal(size=1000) * 2.0
ell = rng.normal(size=1000) * 2.0
sigma, t = bc.engine_to_plan(theta, ell)
theta2, ell2 = bc.plan_to_engine(sigma, t)
ok(np.allclose(theta, theta2, atol=1e-12) and np.allclose(ell, ell2, atol=1e-12),
   "engine→plan→engine round-trip exact to 1e-12")
sig0 = np.exp(rng.normal(size=1000))
t0 = rng.normal(size=1000) * 2.0
th, el = bc.plan_to_engine(sig0, t0)
sig1, t1 = bc.engine_to_plan(th, el)
ok(np.allclose(sig0, sig1, atol=1e-12) and np.allclose(t0, t1, atol=1e-12),
   "plan→engine→plan round-trip exact to 1e-12")

# 3 ── P(yes) agreement across the three formulations
s = rng.uniform(-3, 3, size=1000)
p_plan = bc.p_yes_plan(s, sigma, t)
p_engine = bc.p_yes_engine(s, theta, ell)
z = np.exp(ell) * (s + theta)
p_eng_native = eng._p_response_yes(z)
ok(np.allclose(p_plan, p_engine, atol=1e-12), "p_yes plan-form == engine-form (1e-12)")
ok(np.allclose(p_plan, p_eng_native, atol=1e-12),
   "p_yes bridge == core_mcmc_general._p_response_yes (1e-12)")

# 4 ── sign-convention directions (F3 bug class)
ok(bc.p_yes_plan(0.5, 1.0, 0.6) < bc.p_yes_plan(0.5, 1.0, 0.4),
   "plan: raising criterion t LOWERS P(yes)")
ok(bc.p_yes_engine(0.5, 0.6, 0.0) > bc.p_yes_engine(0.5, 0.4, 0.0),
   "engine: raising bias θ RAISES P(yes)")

# 5 ── difficulty multipliers
for a in np.linspace(0.55, 0.95, 9):
    m = bc.difficulty_multiplier(a)
    ok_val = abs(bc.accuracy_at_multiplier(m) - a) < 1e-9
    assert ok_val, f"FAIL: multiplier inverse at a={a}"
checks += 1
print("  ok: accuracy_at_multiplier(difficulty_multiplier(a)) == a (9-point grid, 1e-9)")

ok(abs(bc.difficulty_multiplier(0.85, lapse=0.0) - 1.0364333894937898) < 1e-9,
   "no-lapse 85% point = 1.03643 (the plan's mistaken 1.04)")
ok(abs(bc.difficulty_multiplier(0.85) - 1.11895838) < 1e-6,
   "lapse-corrected 85% point = 1.118958")
ok(abs(bc.SKILL_MODE_MULTIPLIER - 1.0772256204820363) < 1e-9,
   "SKILL_MODE_MULTIPLIER (D6, Wilson optimum Φ(1) @ λ=0.025) = 1.0772256")
ok(abs(bc.difficulty_multiplier(bc.WILSON_OPT_ACC, lapse=0.0) - 1.0) < 1e-9,
   "no-lapse Wilson optimum = exactly 1.0")
# the plan's 1.04 under the real lapse model gives ~83.3%, not 85%
ok(abs(bc.accuracy_at_multiplier(1.04) - 0.8332885471855677) < 1e-9,
   "accuracy at the plan's 1.04 under λ=0.025 is 83.33% (F2 confirmed)")
try:
    bc.difficulty_multiplier(0.99)
    raise AssertionError("FAIL: out-of-band target accuracy must raise")
except ValueError:
    ok(True, "target accuracy outside (λ, 1−λ) raises ValueError")

# 6 ── AUROC
ok(abs(float(bc.auroc_from_ell(0.0)) - norm.cdf(1.0)) < 1e-12,
   "AUROC(ℓ=0) = Φ(1) ≈ 0.841345")
grid = np.linspace(-2, 2, 41)
ok(np.allclose(bc.auroc_from_ell(grid), auroc_mod.auroc_from_l(grid), atol=1e-12),
   "bridge auroc_from_ell == vendored auroc.auroc_from_l")
ok(np.all(np.diff(bc.auroc_from_ell(grid)) > 0), "AUROC strictly increasing in ℓ")
ok(np.allclose(bc.auroc_from_sigma(np.exp(-grid)), bc.auroc_from_ell(grid), atol=1e-12),
   "auroc_from_sigma(exp(−ℓ)) == auroc_from_ell(ℓ)")
ok(np.allclose(bc.sigma_star_from_ell_star([0.325, 0.534]),
               np.exp([-0.325, -0.534]), atol=1e-15),
   "σ* = exp(−ℓ*) mastery mapping")

# 7 ── weighted AUROC quantiles
N, K = 400, 3
state_deg = {"l": np.full((N, K), 0.3), "w": np.full(N, 1.0 / N)}
q = auroc_mod.auroc_quantiles_from_particles_hier(state_deg, alphas=(0.025, 0.975))
ok(q.shape == (K, 2) and np.allclose(q, float(auroc_mod.auroc_from_l(0.3)), atol=1e-12),
   "degenerate cloud: all quantiles equal the point AUROC")
state_rnd = {"l": rng.normal(size=(N, K)), "w": rng.uniform(0.1, 1.0, size=N)}
state_rnd["w"] /= state_rnd["w"].sum()
q = auroc_mod.auroc_quantiles_from_particles_hier(state_rnd, alphas=(0.025, 0.975))
ok(np.all(q[:, 0] < q[:, 1]) and np.all((q > 0.0) & (q < 1.0)),
   "random cloud: lo < hi quantiles, all within (0, 1)")

# 8 ── engine end-to-end with the vendored stub (tiny hier session)
res = eng.run_session_mcmc_auroc(
    "hier", true_params=[0.2, 0.5, -0.3, 0.8], K=2, r_assumed=0.3,
    max_q=60, delta_auroc=0.06, N=300, seed=42)
ok(res["n_questions"] >= 10 and res["final_lo"].shape == (2,)
   and np.all(res["final_lo"] <= res["final_hi"]),
   f"tiny hier session ran ({res['n_questions']} questions, "
   f"stopped_early={res['stopped_early']})")
ok(res["n_rejuvenations"] >= 1 and 0.0 < res["mean_acceptance_rate"] < 1.0,
   f"resample+MH rejuvenation exercised ({res['n_rejuvenations']} events, "
   f"mean accept {res['mean_acceptance_rate']:.2f})")

print(f"\nSTEP 0 SMOKE TEST PASSED — {checks} checks.")
