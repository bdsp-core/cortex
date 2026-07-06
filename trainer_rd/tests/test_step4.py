"""Step 4 — identifiability HARD GATE (INTEGRATION_PLAN.md). Do not pass on failure.
Run: python3 -m tests.test_step4

Pre-registered pass criteria (PROJECT_MEMORY.md F11/F13):
  1. IDENTIFIABILITY: across-replicate corr(σ-error, t-error) ≤ 0.5 in EVERY cell
     (σ and t are not traded for one another — the core disentanglement claim);
  2. low posterior coupling: mean |corr(θ,ℓ)| ≤ 0.6 in every cell;
  3. correct-spec coverage: at the true learning rates, terminal 95%-box coverage
     ≥ 0.88 on average (MC slack on the 0.90 target) and ≥ 0.80 per cell;
  4. bounded estimation: trajectory RMSE < 0.30 (σ and t) in EVERY cell, incl.
     ±2× misspecified rates — no divergence;
  5. graceful misspecification: at 2× rates coverage stays ≥ 0.55 (degraded but
     not collapsed) and RMSE no worse than 1.6× the correct-spec RMSE.
"""
import numpy as np

from studies.study_identifiability import run_campaign

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


print("  running identifiability campaign (3 cells × 3 rate-mults × 30 reps)...")
res = run_campaign(rate_mults=(0.5, 1.0, 2.0), n_reps=30, n_trials=120, seed0=0)

by_mult = {rm: [r for r in res if r.rate_mult == rm] for rm in (0.5, 1.0, 2.0)}

# 1. identifiability — the headline
max_errc = max(abs(r.err_corr) for r in res)
ok(max_errc <= 0.5,
   f"σ/t error-confusion |corr| ≤ 0.5 in every cell (max {max_errc:.2f})")

# 2. posterior coupling
max_cpl = max(r.cloud_coupling for r in res)
ok(max_cpl <= 0.6, f"posterior |corr(θ,ℓ)| ≤ 0.6 in every cell (max {max_cpl:.2f})")

# 3. correct-spec coverage
cov1 = [r.coverage for r in by_mult[1.0]]
ok(np.mean(cov1) >= 0.88 and min(cov1) >= 0.80,
   f"correct-spec coverage mean {np.mean(cov1):.2f} (≥0.88), min {min(cov1):.2f} (≥0.80)")

# 4. bounded estimation everywhere
max_rmse = max(max(r.rmse_sigma, r.rmse_t) for r in res)
ok(max_rmse < 0.30, f"RMSE < 0.30 in every cell incl. misspecified (max {max_rmse:.3f})")

# 5. graceful misspecification at 2×
cov2 = [r.coverage for r in by_mult[2.0]]
rmse1 = np.mean([max(r.rmse_sigma, r.rmse_t) for r in by_mult[1.0]])
rmse2 = np.mean([max(r.rmse_sigma, r.rmse_t) for r in by_mult[2.0]])
ok(min(cov2) >= 0.55, f"2× rates: coverage ≥ 0.55 every cell (min {min(cov2):.2f})")
ok(rmse2 <= 1.6 * rmse1,
   f"2× rates: RMSE {rmse2:.3f} ≤ 1.6× correct-spec {rmse1:.3f} (no divergence)")

# informational: 0.5× over-covers (conservative) — confirms the asymmetry
cov05 = np.mean([r.coverage for r in by_mult[0.5]])
print(f"  [info] coverage by rate-mult: 0.5×={cov05:.2f}  1×={np.mean(cov1):.2f}  "
      f"2×={np.mean(cov2):.2f}  (under-estimating rates is the safe direction)")

print(f"\nSTEP 4 IDENTIFIABILITY GATE PASSED — {checks} checks "
      f"(max err-corr {max_errc:.2f}, correct-spec coverage {np.mean(cov1):.2f}).")
