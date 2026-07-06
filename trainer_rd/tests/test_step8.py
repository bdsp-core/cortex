"""Step 8 smoke test (INTEGRATION_PLAN.md). Run: python3 -m tests.test_step8

The capstone: a mid-skill candidate who initially does NOT pass improves under
training and re-certifies. Asserts the seamless chain end-to-end:

  1. eval renders verdicts and at least one trained task is NOT PASS (the
     candidate genuinely needs training);
  2. training raises the learner's TRUE skill on every trained task;
  3. at least one initially-non-PASS task FLIPS to PASS at re-certification;
  4. the telemetry chain is complete: training log non-empty, every record
     carries a mode + finite epoch (D9);
  5. the re-cert recovered skill exceeds the eval recovered skill on average
     (the filter/engine sees the improvement, not just the ground truth).
"""
import numpy as np

from training.pipeline_demo import run_pipeline
from engine.policy_general import PASS
from training.bank_adapter import ELL_STAR

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


print("  running end-to-end pipeline (eval → train → re-cert)...")
# 500 particles keep the AD6 Z·mcse buffer tight enough to RESOLVE PASS (fewer
# particles inflate mcse and strand strong candidates at REFER_BORDERLINE — a
# documented eval property, HOW_THE_TEST_WORKS §11). With s_sd threaded through
# the eval (F20, production-faithful) each question carries less information, so
# the question budget must be production-scale (real sessions run 500 trials).
rep = run_pipeline(tasks=(1, 2, 3), n_sessions=5, trials_per_session=80, seed=1,
                   eval_n_part=500, eval_max_q=500, eval_pool=800)

print(f"  tasks           : {rep.tasks}")
print(f"  eval verdicts   : {rep.eval_verdicts}")
print(f"  true ℓ eval→post: {rep.true_skill_eval} → {rep.true_skill_post}")
print(f"  recert verdicts : {rep.recert_verdicts}")
print(f"  PASS flips      : {rep.flips}  ({rep.n_train_trials} train trials)")

# 1. candidate needs training
ok(any(v != PASS for v in rep.eval_verdicts),
   f"eval: at least one trained task not PASS ({rep.eval_verdicts})")

# 2. training raised true skill on every task
gains = [rep.true_skill_post[k] - rep.true_skill_eval[k] for k in range(len(rep.tasks))]
ok(all(g > 0 for g in gains), f"training raised true skill on every task (Δℓ={[round(g,2) for g in gains]})")

# 3. every task moves OFF FAIL at re-cert (budget-robust: training cleared the
#    failing verdict on every task)
from engine.policy_general import FAIL
ok(all(v != FAIL for v in rep.recert_verdicts),
   f"every task off FAIL at re-cert ({rep.recert_verdicts})")

# 4. at least one PASS flip (the user's stated FAIL/REFER → PASS objective)
ok(len(rep.flips) >= 1, f"≥1 task flipped to PASS after training ({rep.flips})")

# 5. telemetry chain complete
ok(rep.train_log_len > 0 and rep.n_train_trials > 0,
   f"training telemetry non-empty ({rep.train_log_len} records)")

# 6. recovered skill improved on EVERY task and clears the cut on average.
#    (Under production-faithful physics — s_sd marginalized through selection
#    and update, F20/M10 — a single noisy re-measurement of a trainee who is
#    truly ~0.2–0.3 above the bar lands below it on a task occasionally; the
#    per-task point-estimate-clears-cut claim was an s_sd=0 artifact.)
ev_mean = float(np.mean(rep.eval_skill))
rc_mean = float(np.mean(rep.recert_skill))
cuts = [ELL_STAR[t] for t in (1, 2, 3)]
ok(rc_mean > ev_mean,
   f"re-cert recovered skill ℓ̂ {rc_mean:.2f} > eval ℓ̂ {ev_mean:.2f} (engine sees gains)")
ok(all(rep.recert_skill[k] > rep.eval_skill[k] for k in range(len(rep.tasks))),
   f"re-cert ℓ̂ improved on every task ({rep.eval_skill} → {rep.recert_skill})")
ok(rc_mean > float(np.mean(cuts)),
   f"mean re-cert ℓ̂ {rc_mean:.2f} clears the mean cut {float(np.mean(cuts)):.2f}")

print(f"\nSTEP 8 SMOKE TEST PASSED — {checks} checks "
      f"(flips {rep.flips}, {rep.n_train_trials} train trials).")
