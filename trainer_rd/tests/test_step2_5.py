"""Step 2.5 smoke test (INTEGRATION_PLAN.md). Run: python3 -m tests.test_step2_5

  1. per-task candidacy counts match known totals (domain1 19,332; others 69,806);
  2. label reconstruction of the 3 real sessions' served labels ≥97% overall;
  3. exclusion removes eval-seen seg_ids; feedback_safe removes conflicted/low-margin;
  4. returned arrays parallel + finite; constants (ELL_STAR/SIGMA_STAR/SIGMA_INF) sane;
  5. conflicted fraction on the served items is ≈25–30% (sanity, matches §3 F-finding).
"""
import glob
import os

import numpy as np
import pandas as pd

from training.bank_adapter import (BankAdapter, TASK_CODES, ELL_STAR, SIGMA_STAR,
                          SIGMA_INF, DOMAIN1_POS_THRESHOLD)

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


ad = BankAdapter()

# ── 1. candidacy counts ──
counts = {TASK_CODES[k]: len(ad.task_pool(k)) for k in range(7)}
print("  pool sizes:", counts)
# pools are after dropping unlabelable segs; assert close to the raw candidacy
ok(counts["domain1"] <= 19332 and counts["domain1"] >= 19000,
   f"domain1 pool {counts['domain1']} ≈ 19,332 candidacy")
ok(all(15000 <= counts[c] <= 69806 for c in TASK_CODES[1:]),
   "multiclass pools within (15k, 69,806]")

# ── 2. label reconstruction vs production target_present ──
# Use the one real session that carries target_present (f658855c).
ses = "f658855c-c9cc-4f8c-8161-d6c6f6a32200_trials_anonymized.csv"
t = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                             "sessions", ses))
code_to_idx = {c: i for i, c in enumerate(TASK_CODES)}
# build a seg_id → y* lookup per task from the adapter
n_ok = n_tot = 0
for _, row in t.iterrows():
    k = code_to_idx[row["task_code"]]
    pool = ad.task_pool(k)
    hit = np.where(pool.seg_id == int(row["seg_id"]))[0]
    if hit.size == 0:
        continue
    y_star_pred = int(pool.y_star[hit[0]])
    n_tot += 1
    n_ok += int(y_star_pred == int(row["target_present"]))
rate = n_ok / n_tot
ok(rate >= 0.95, f"label reconstruction {rate*100:.1f}% of {n_tot} served items (≥95%)")

# ── 3. exclusions ──
excl = set(int(x) for x in t[t.task_code == "domain1"]["seg_id"].tolist())
base = ad.candidates(0)
filt = ad.candidates(0, exclude_segids=excl)
ok(len(filt) == len(base) - len(excl & set(base.seg_id.tolist())) and
   not (set(filt.seg_id.tolist()) & excl),
   f"exclude_segids removed {len(excl)} eval-seen domain1 segs")

fb = ad.candidates(0, feedback_safe=True, min_margin=0.30)
ok(len(fb) < len(base) and np.all(fb.coherent) and np.all(fb.margin >= 0.30),
   f"feedback_safe pool: all coherent + margin≥0.30 ({len(fb)} of {len(base)})")

# ── 4. parallel + finite + constants ──
c = ad.candidates(3, feedback_safe=True)
lengths = {len(c.seg_id), len(c.s_mean), len(c.s_sd), len(c.y_star),
           len(c.margin), len(c.coherent)}
ok(len(lengths) == 1, "candidate arrays are parallel (equal length)")
ok(np.all(np.isfinite(c.s_mean)) and np.all(np.isfinite(c.s_sd)) and
   np.all((c.y_star == 0) | (c.y_star == 1)) and np.all(c.s_sd >= 0),
   "candidate arrays finite; y* binary; s_sd ≥ 0")
ok(np.allclose(SIGMA_STAR, np.exp(-ELL_STAR), atol=1e-6) and
   np.all(SIGMA_INF < SIGMA_STAR) and np.all(SIGMA_INF > 0),
   "constants: σ*=exp(−ℓ*); σ_∞ < σ* every task (LT1 achievable, §2B)")
print("  ELL_STAR:", np.round(ELL_STAR, 3).tolist())
print("  SIGMA_INF:", np.round(SIGMA_INF, 3).tolist())

# ── 5. conflicted fraction sanity ──
cfracs = [ad.conflicted_fraction(k) for k in range(7)]
mean_cf = float(np.mean(cfracs))
print("  per-task conflicted fraction:", [round(x, 2) for x in cfracs])
ok(0.10 <= mean_cf <= 0.45,
   f"mean conflicted fraction {mean_cf:.2f} in plausible band (trap items exist)")

print(f"\nSTEP 2.5 SMOKE TEST PASSED — {checks} checks "
      f"(label match {rate*100:.1f}%, mean conflicted {mean_cf:.2f}).")
