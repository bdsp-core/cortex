"""Step 7 smoke test (INTEGRATION_PLAN.md) — benchmark PILOT.
Run: python3 -m tests.test_step7

Under PRODUCTION-FAITHFUL physics (M10/F20: the learner experiences
s_real ~ N(s_mean, s_sd²); the real bank's s_sd ≈ 0.6–1.0 is comparable to σ)
the campaign orderings changed from the old s_sd=0 regime: naive nearest-target
placement (staircase85) is blurred into ineffectiveness, criterion placement
(measure_opt) is incidentally smeared INTO the productive band, and the
noise-aware tier2 (max expected learning weight / attenuated criterion info)
dominates. Asserted pilot claims (full 30-seed paired campaign, M10):

  1. tier2 always reaches mastery (no censoring) on the well-specified learner;
  2. tier2 ≪ random (null baseline);
  3. tier2 ≪ staircase85 — noise-aware placement is the headline win under
     real-bank stimulus noise (paired diff ≈ −25 to −38, CIs exclude 0);
  4. tier2 comparable to measure_opt when well-specified (the old "wrong
     objective" gap is an s_sd=0 artifact; tier2's edge over it returns under
     rate misspecification);
  5. tier2 more ROBUST than staircase under the hard-rule learner (lower mean
     OR lower variance).
"""
import numpy as np

from training.benchmark_trainer import run_benchmark

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


print("  running benchmark pilot (5 policies × 2 rules × 8 seeds)...")
# pool_size ≥ 1000: noise-aware selection needs item supply (production
# feedback-safe pools are 15k–65k; at 300 items tier2 is artificially starved
# of low-s_sd near-target candidates — measured M10 pool-size sensitivity)
res = run_benchmark(task=2, n_seeds=8, rules=("soft", "hard"),
                    rate_mults=(1.0,), budget=300, pool_size=1000)


def stat(rule, pol):
    return res[(rule, 1.0, pol)]            # (mean, sd, censored)


for rule in ("soft", "hard"):
    row = {p: stat(rule, p) for p in ("tier2", "tier1", "staircase85",
                                      "measure_opt", "random")}
    print(f"  [{rule}] " + "  ".join(f"{p}={row[p][0]:.0f}±{row[p][1]:.0f}"
                                     f"(c{row[p][2]})" for p in row))

t2_soft = stat("soft", "tier2")
mo_soft = stat("soft", "measure_opt")
rnd_soft = stat("soft", "random")
sc_soft = stat("soft", "staircase85")

ok(t2_soft[2] == 0, f"tier2 always reaches mastery (0 censored, mean {t2_soft[0]:.0f})")
ok(t2_soft[0] < 0.5 * rnd_soft[0],
   f"tier2 {t2_soft[0]:.0f} ≪ random {rnd_soft[0]:.0f}")
ok(t2_soft[0] < 0.85 * sc_soft[0],
   f"tier2 {t2_soft[0]:.0f} ≪ staircase {sc_soft[0]:.0f} (noise-aware placement wins)")
ok(t2_soft[0] <= 1.25 * mo_soft[0],
   f"tier2 {t2_soft[0]:.0f} ≤ 1.25× measure_opt {mo_soft[0]:.0f} (comparable when well-specified)")

t2_hard = stat("hard", "tier2")
sc_hard = stat("hard", "staircase85")
ok(t2_hard[0] <= sc_hard[0] or t2_hard[1] < sc_hard[1],
   f"tier2 more robust than staircase under hard rule "
   f"(tier2 {t2_hard[0]:.0f}±{t2_hard[1]:.0f} vs staircase {sc_hard[0]:.0f}±{sc_hard[1]:.0f})")

print(f"\nSTEP 7 SMOKE TEST PASSED — {checks} checks "
      f"(tier2 {t2_soft[0]:.0f} vs measure_opt {mo_soft[0]:.0f} vs random {rnd_soft[0]:.0f}).")
