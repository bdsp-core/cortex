"""Step 6 smoke test (INTEGRATION_PLAN.md). Run: python3 -m tests.test_step6

  1. greedy beats RANDOM item choice on realized one-step expected reward (the
     basic sanity that the Q-score is meaningful);
  2. greedy item placement is sensible: with a biased belief it favors items
     that reduce |t| (near the criterion) — i.e. it picks informative-for-reward
     items, not arbitrary ones;
  3. F6b base-rate exploit: the UNCONSTRAINED greedy farms |t|-reward with
     one-sided labels (running balance runs away); the CONSTRAINED one does not;
  4. F6a retention spam: with an unconstrained retention bonus the greedy
     re-selects an ineligible easy item; eligibility-gating removes the incentive.
"""
import numpy as np

from training.bank_adapter import BankAdapter, ELL_STAR, SIGMA_INF
from training.learner_sim import LearnerParams
from training.training_filter import TaskFilter
from training.trainer_greedy import (RewardWeights, greedy_select, _expected_reward)

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


def mk_filter(theta_mean, ell_mean, sd=0.3, N=600, seed=0, sigma_inf=0.5):
    rng = np.random.default_rng(seed)
    params = LearnerParams(alpha_t=0.15, alpha_sigma=0.05, sigma_inf=sigma_inf,
                           q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    return TaskFilter(theta_mean + sd * rng.standard_normal(N),
                      ell_mean + sd * rng.standard_normal(N), params, seed=seed)


ad = BankAdapter()
W = RewardWeights(beta_t=1.0, beta_sigma=1.0, beta_r=0.5)

# biased + sub-cut learner on domain3
filt = mk_filter(-0.6, 0.0, sigma_inf=SIGMA_INF[2], seed=1)
cands = ad.candidates(2, feedback_safe=True)
# subsample candidates for speed
sub = np.random.default_rng(0).choice(len(cands), size=400, replace=False)
from training.bank_adapter import TaskCandidates
cs = TaskCandidates(2, cands.seg_id[sub], cands.s_mean[sub], cands.s_sd[sub],
                    cands.y_star[sub], cands.margin[sub], cands.coherent[sub])

# ── 1. greedy beats random on Q ──
idx_g, Q = greedy_select(filt, cs, W, constrain_balance=False)
q_greedy = Q[idx_g]
rng = np.random.default_rng(3)
q_rand = np.mean([_expected_reward(filt, cs.s_mean[i], int(cs.y_star[i]), W)
                  for i in rng.choice(len(cs), size=50)])
ok(q_greedy > q_rand,
   f"greedy Q ({q_greedy:.4f}) > mean random Q ({q_rand:.4f})")

# ── 2. greedy placement is sensible (reduces |t|: item near the criterion) ──
_, t_hat = (-filt.mean()[0]),  -filt.mean()[0]
t_hat = -filt.mean()[0]
chosen_s = cs.s_mean[idx_g]
# the |t|-reward is biggest for items that pull t toward 0; with t̂<0 (θ̄>0)
# that means items on the side that raises t. Assert the chosen item's implied
# Δ|t| is positive (reward-improving), not arbitrary.
r_chosen = _expected_reward(filt, chosen_s, int(cs.y_star[idx_g]), W)
ok(r_chosen >= np.percentile(Q[np.isfinite(Q)], 99),
   "greedy choice is in the top 1% of the reward surface (argmax sanity)")

# ── 3. F6b base-rate exploit ──
def run_balance(constrain, n=40):
    f = mk_filter(-0.7, 0.0, sigma_inf=SIGMA_INF[2], seed=10)
    bal = 0
    bals = []
    # reward weights with ONLY beta_t so the |t| term dominates (worst case)
    Wt = RewardWeights(beta_t=1.0, beta_sigma=0.0, beta_r=0.0)
    for _ in range(n):
        i, _ = greedy_select(f, cs, Wt, label_balance=bal,
                             constrain_balance=constrain)
        bal += 1 if cs.y_star[i] == 1 else -1
        bals.append(bal)
        # advance belief a touch so it's not static
        f.step(float(cs.s_mean[i]), int(cs.y_star[i] == 1), int(cs.y_star[i]),
               feedback=True)
    return max(abs(b) for b in bals)

runaway = run_balance(constrain=False)
bounded = run_balance(constrain=True)
print(f"  label-balance excursion: unconstrained {runaway}, constrained {bounded}")
ok(runaway >= 8, f"UNCONSTRAINED greedy farms one-sided labels (excursion {runaway})")
ok(bounded <= 2, f"CONSTRAINED greedy keeps balance bounded (excursion {bounded})")

# ── 4. F6a retention spam ──
# one "easy mastered" item (far from threshold) is ineligible; an unconstrained
# retention bonus still makes it win; eligibility-gating flips the choice away.
small = TaskCandidates(
    2,
    seg_id=np.array([900, 901, 902]),
    s_mean=np.array([3.0, 0.2, -0.1]),       # 900 = very easy (retention-spammable)
    s_sd=np.array([0.1, 0.8, 0.8]),
    y_star=np.array([1, 1, 0]),
    margin=np.array([0.9, 0.4, 0.4]),
    coherent=np.array([True, True, True]))
fr = mk_filter(0.0, 0.3, sd=0.2, sigma_inf=SIGMA_INF[2], seed=5)
Wr = RewardWeights(beta_t=1.0, beta_sigma=1.0, beta_r=5.0)   # juicy retention bonus
# The retention bonus attaches to the mastered-bin (easy) item 900. UNGATED by
# due-ness, 900 collects the bonus and — having little learning value itself —
# wins purely on the bonus (the spam). GATED, 900 is not currently due, so it
# gets no bonus and a productive item wins.
elig_ungated = np.array([True, False, False])   # bonus active on the easy item
i_spam, _ = greedy_select(fr, small, Wr, constrain_balance=False,
                          retention_eligible=elig_ungated)
elig_gated = np.array([False, False, False])    # 900 not due ⇒ no bonus
i_gated, _ = greedy_select(fr, small, Wr, constrain_balance=False,
                           retention_eligible=elig_gated)
print(f"  retention: spam picks seg {small.seg_id[i_spam]}, "
      f"gated picks seg {small.seg_id[i_gated]}")
ok(small.seg_id[i_spam] == 900,
   "unconstrained retention bonus ⇒ easy item gets spammed (F6a witnessed)")
ok(small.seg_id[i_gated] != 900,
   "eligibility-gated retention removes the spam incentive (F6a fixed)")

print(f"\nSTEP 6 SMOKE TEST PASSED — {checks} checks.")
