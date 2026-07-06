"""Step 1 smoke test (INTEGRATION_PLAN.md). Run: python3 -m tests.test_step1

Runs a REAL mini eval session through the engine (choose_item → update →
ESS-gated rejuvenation) with AD6 verdicts, then verifies the eval→trainer
handoff:

  1. seed build from the live state (verdicts mixed PASS/FAIL/REFER);
  2. save → load round-trip: every array exactly equal, dtype preserved;
  3. variance inflation: per-task weighted SD ×1.5 exactly, means unchanged
     (both θ and ℓ, all tasks);
  4. weights normalized; task_cloud marginals match joint columns;
  5. eval-seen seg_ids complete and unique (each segment served ≤ once);
  6. D9 fields: per-trial epochs present, session_id/created_at in meta;
  7. LearnerLedger: append-only round-trip, ordering preserved;
  8. TrialRecord dict round-trip.
"""
import os

import numpy as np

import engine.core_mcmc_general as eng
from engine.policy_general import AD6Policy, PENDING
from training.training_seed import (TrainingSeed, TrialRecord, LearnerLedger,
                           build_seed_from_state, inflate_cloud)

rng = np.random.default_rng(7)
checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


# ── synthetic bank: K=3 tasks, 120 items each, unique seg_ids ──
K, N_ITEMS, N_PART = 3, 120, 500
bank = [np.sort(rng.uniform(-2.5, 2.5, size=N_ITEMS)) for _ in range(K)]
segids = [np.arange(k * 1000, k * 1000 + N_ITEMS, dtype=np.int64) for k in range(K)]

# True examinee: task0 clear pass (ℓ=1.1 ≫ .325), task1 clear fail (ℓ=−0.8),
# task2 near its cut (ℓ=0.5 vs .534 → expect borderline/slow).
true_params = [0.15, 1.1, -0.25, -0.8, 0.05, 0.5]
ELL_STAR = [0.325, 0.256, 0.534]

# ── mini eval session (mirrors the production loop) ──
state = eng.make_state_hier(N_PART, K, r_assumed=0.4, rng=rng)
policy = AD6Policy(ELL_STAR, [1.0, 1.0, 1.0])
policy.reset(K)
n_per_task = [0] * K
served, rts, epochs = [], [], []
T0 = 1_770_000_000.0                     # synthetic wall-clock base
decision = None
proposal_scale = 2.38 / np.sqrt(2 * K)

for q in range(250):
    active = [k for k in range(K)
              if policy._verdicts[k] == PENDING and len(bank[k]) > 0]
    if not active:
        break
    k, s, seg_id = eng.choose_item(state, bank, active_domains=active,
                                   bank_segids=segids)
    y = eng.simulate_response(s, true_params[2 * k], true_params[2 * k + 1], rng)
    eng.update(state, k, s, y)
    # serve each segment at most once (production invariant)
    idx = int(np.where(segids[k] == seg_id)[0][0])
    bank[k] = np.delete(bank[k], idx)
    segids[k] = np.delete(segids[k], idx)
    if eng.ess(state["w"]) < 0.5 * N_PART:
        eng.resample_and_rejuvenate(state, rng, 15, proposal_scale)
    n_per_task[k] += 1
    served.append(seg_id)
    rts.append(float(800 + 400 * rng.random()))      # synthetic RT (ms)
    epochs.append(T0 + 4.2 * q)
    decision = policy(state, {}, n_per_task, K)
    if decision.stop:
        break

final_verdicts = policy.finalize_verdicts()
n_trials = len(served)
print(f"  session: {n_trials} trials, verdicts={final_verdicts}, "
      f"n_per_task={n_per_task}")
ok(n_trials >= 3 * 20, "session ran past N_MIN on all tasks")
ok(len(set(final_verdicts)) >= 2 and all(v != PENDING for v in final_verdicts),
   "AD6 produced a mixed, fully-finalized verdict set")

# ── 1. build seed ──
seed = build_seed_from_state(
    state, session_id="eval-0001", ell_star=ELL_STAR,
    verdicts=final_verdicts, diagnostics=policy._last_diag,
    inflate=1.5, seg_ids=served, rt_ms=rts, trial_epochs=epochs,
    session_type="eval", notes="step-1 smoke")
ok(seed.K == K and seed.N == N_PART and seed.meta["n_trials"] == n_trials,
   "seed dimensions/meta match the session")

# ── 2. save → load round-trip ──
path = "/tmp/seed_step1_test.npz"
seed.save(path)
seed2 = TrainingSeed.load(path)
for name in TrainingSeed._ARRAY_FIELDS:
    a, b = getattr(seed, name), getattr(seed2, name)
    assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), \
        f"FAIL: round-trip mismatch in {name}"
checks += 1
print("  ok: save→load round-trip exact for all 13 arrays (dtype + values)")
ok(seed2.meta == seed.meta, "meta JSON round-trips identically")
os.remove(path)

# ── 3. inflation: SD ×1.5 exactly, mean unchanged ──
w = seed.w
for k in range(K):
    for raw, infl in ((seed.ell_raw[:, k], seed.ell_infl[:, k]),
                      (seed.theta_raw[:, k], seed.theta_infl[:, k])):
        mu_r = float((w * raw).sum()); mu_i = float((w * infl).sum())
        sd_r = float(np.sqrt((w * (raw - mu_r) ** 2).sum()))
        sd_i = float(np.sqrt((w * (infl - mu_i) ** 2).sum()))
        assert abs(mu_i - mu_r) < 1e-9, f"FAIL: inflation moved mean (task {k})"
        assert abs(sd_i / sd_r - 1.5) < 1e-9, f"FAIL: SD ratio != 1.5 (task {k})"
checks += 1
print("  ok: inflation — weighted SD ×1.5 (1e-9), weighted mean unchanged, θ and ℓ, all tasks")

# ── 4. weights + marginal accessor ──
ok(abs(w.sum() - 1.0) < 1e-12 and np.all(w >= 0), "weights normalized, non-negative")
th_k, el_k, w_k = seed.task_cloud(1, inflated=True)
ok(np.array_equal(th_k, seed.theta_infl[:, 1]) and
   np.array_equal(el_k, seed.ell_infl[:, 1]) and np.array_equal(w_k, w),
   "task_cloud(k) marginal == joint columns + shared weights (D2)")

# ── 5. eval-seen seg_ids ──
seen = seed.eval_seen_segids()
ok(len(seen) == n_trials and len(np.unique(seen)) == n_trials,
   "eval-seen seg_ids complete and unique (each segment served once)")
ok(set(seen.tolist()) == set(served), "seg_ids match exactly what was served")

# ── 6. D9 fields ──
ok(np.all(np.isfinite(seed.hist_t_epoch)) and
   np.all(np.diff(seed.hist_t_epoch) > 0),
   "per-trial absolute epochs present and increasing (D9)")
ok(np.all(np.isfinite(seed.hist_rt_ms)) and np.nanmin(seed.hist_rt_ms) > 0,
   "per-trial RT field populated (F10)")
ok(seed.meta["session_id"] == "eval-0001" and "T" in seed.meta["created_at"]
   and seed.meta["verdicts"] == final_verdicts
   and seed.meta["ell_star"] == ELL_STAR
   and seed.meta["ad6_diagnostics"]["n_per_task"] == n_per_task,
   "meta carries session_id, ISO created_at, verdicts, ℓ*, AD6 diagnostics")
ok(not np.any(seed.hist_post_decision),
   "post_decision flags default False (no extended-collection trials here)")

# ── 7. learner ledger ──
lpath = "/tmp/ledger_step1_test.json"
led = LearnerLedger("learner-42")
led.append_entry(seed_path="seeds/eval-0001.npz", session_id="eval-0001",
                 session_type="eval", created_at="2026-06-10T12:00:00+00:00")
led.append_entry(seed_path="seeds/train-0001.npz", session_id="train-0001",
                 session_type="training")
led.save(lpath)
led2 = LearnerLedger.load(lpath)
ok(led2.learner_id == "learner-42" and len(led2.entries) == 2
   and [e["seq"] for e in led2.entries] == [0, 1]
   and led2.entries[0]["session_id"] == "eval-0001",
   "ledger round-trips with order + seq preserved (append-only)")
led2.entries.append({"bogus": True})      # mutating the copy must not stick
ok(len(led2.entries) == 2, "entries property returns a copy (append-only enforced)")
os.remove(lpath)

# ── 8. TrialRecord round-trip ──
tr = TrialRecord(session_id="train-0001", trial_index=0, task_k=2,
                 seg_id=2042, s=0.83, s_sd=0.1, y=1, y_star=1,
                 feedback_given=True, mode="skill", rt_ms=912.0,
                 answer_changes=0, t_epoch=T0 + 9000.0)
ok(TrialRecord.from_dict(tr.to_dict()) == tr, "TrialRecord dict round-trip")

# ── bonus: inflation helper is exact on a known case ──
x = rng.normal(size=(1000, 2)); ww = np.full(1000, 1e-3)
xi = inflate_cloud(x, ww, 2.0)
ok(np.allclose((ww[:, None] * xi).sum(0), (ww[:, None] * x).sum(0), atol=1e-12),
   "inflate_cloud preserves weighted mean (2-D case)")

print(f"\nSTEP 1 SMOKE TEST PASSED — {checks} checks "
      f"({n_trials}-trial session, verdicts {final_verdicts}).")
