"""M11 smoke test — Tier-3 MC rollout policy. Run: python3 -m tests.test_tier3

Verifies:
  1. unit: _RolloutPool nearest-target lookup exact on a toy bank;
  2. unit: _pi0_targets mode gate (bias when |t̂|>t*, else boundary-centered
     85%-point) and side/label alternation;
  3. selection: valid index, deterministic under a fixed rng, label-balance
     constraint honored (F6b);
  4. lookahead sanity: with a clean separable choice (one productive item vs
     junk), tier3 picks the productive item;
  5. integration: graduates a biased+sub-skill learner on the real bank
     within budget, at < 100 ms/trial.
"""
import time

import numpy as np

from training.bank_adapter import BankAdapter, TaskCandidates
from training.bridge_conventions import SKILL_MODE_MULTIPLIER
from training.learner_sim import LearnerParams
from training.training_filter import TaskFilter
from training.trainer_greedy import RewardWeights
from training.trainer_rollout import _RolloutPool, _pi0_targets, tier3_select

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


rng = np.random.default_rng(0)

# ── 1. rollout pool nearest lookup ──
s = np.array([-2.0, -1.0, -0.2, 0.4, 1.3, 2.2])
y = np.array([0, 0, 0, 1, 1, 1])
pool = _RolloutPool(s, np.zeros(6), y, cap=10, rng=rng)
got_s, _ = pool.nearest(np.array([0.5, -1.6, 2.0]), np.array([1, 0, 1]))
ok(np.allclose(got_s, [0.4, -2.0, 2.2]),
   "nearest-target lookup exact (label-respecting)")

# ── 2. π0 mode gate ──
N = 50
w = np.full((2, N), 1.0 / N)
# rollout 0: biased belief (t̂=+0.8); rollout 1: unbiased, σ̂=1
th = np.stack([np.full(N, -0.8), np.zeros(N)])    # θ = −t
el = np.zeros((2, N))                              # ℓ=0 ⇒ σ̂=1
tg, lab = _pi0_targets(th, el, w, step_parity=0, t_star=0.3)
ok(abs(tg[0] - 0.8) < 1e-9 and lab[0] == 1,
   "bias mode targets the criterion t̂ (|t̂|>t*)")
ok(abs(tg[1] - SKILL_MODE_MULTIPLIER) < 1e-9 and lab[1] == 1,
   "skill mode is boundary-centered 85%-point (side +)")
tg2, lab2 = _pi0_targets(th, el, w, step_parity=1, t_star=0.3)
ok(abs(tg2[1] + SKILL_MODE_MULTIPLIER) < 1e-9 and lab2[1] == 0,
   "side/label alternate with parity")

# ── 3. selection mechanics on a synthetic pool ──
ns = 40
sp = np.linspace(-2, 2, ns)
cands = TaskCandidates(0, np.arange(ns), sp, np.full(ns, 0.3),
                       (sp > 0).astype(int), np.ones(ns), np.ones(ns, bool))
params = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.55,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
filt = TaskFilter(0.3 * rng.standard_normal(300),
                  0.3 * rng.standard_normal(300), params, seed=1)
kw = dict(weights=RewardWeights(beta_r=0.0), t_star=0.30, H=4, L=6)
i1 = tier3_select(filt, cands, rng=np.random.default_rng(7), **kw)
i2 = tier3_select(filt, cands, rng=np.random.default_rng(7), **kw)
ok(0 <= i1 < ns and i1 == i2, "valid index, deterministic under fixed rng")
i_neg = tier3_select(filt, cands, rng=np.random.default_rng(7),
                     label_balance=2, **kw)
i_pos = tier3_select(filt, cands, rng=np.random.default_rng(7),
                     label_balance=-2, **kw)
ok(cands.y_star[i_neg] == 0 and cands.y_star[i_pos] == 1,
   "label-balance constraint honored (F6b)")

# ── 4. separable lookahead choice ──
# learner at σ̂=1, unbiased: one precise item AT the 85%-point vs junk at ±3σ.
s2 = np.array([SKILL_MODE_MULTIPLIER, 3.0, -3.0, 3.5])
c2 = TaskCandidates(0, np.arange(4), s2, np.full(4, 0.05),
                    (s2 > 0).astype(int), np.ones(4), np.ones(4, bool))
picks = [tier3_select(filt, c2, rng=np.random.default_rng(100 + r), **kw)
         for r in range(5)]
ok(np.bincount(picks, minlength=4)[0] >= 4,
   f"picks the productive 85%-point item ({picks})")

# ── 5. integration: real bank, graduates within budget, cheap per trial ──
from training.benchmark_trainer import run_one
ad = BankAdapter()
full = ad.candidates(2, feedback_safe=True)
sub = np.random.default_rng(0).choice(len(full), size=600, replace=False)
pool2 = TaskCandidates(2, full.seg_id[sub], full.s_mean[sub], full.s_sd[sub],
                       full.y_star[sub], full.margin[sub], full.coherent[sub])
t0 = time.time()
n = run_one("tier3", 2, rule="soft", budget=400, seed=1, pool=pool2)
dt = time.time() - t0
ok(n < 400, f"graduates the biased+sub-skill learner (n={n} < budget)")
ok(dt / max(n, 1) < 0.1, f"cost {1000*dt/max(n,1):.0f} ms/trial < 100")

print(f"\nTIER-3 SMOKE TEST PASSED — {checks} checks.")
