"""EXTSET adapter smoke test (SAP §9.1). Run: python3 -m tests.test_extset

Checks: load integrity for both tasks, F40 vote==read reconciliation (raised
inside the loaders), LOO self-shift direction, gold-policy counts, signal
finiteness/monotonicity, case-split validity.
"""
import numpy as np
import pandas as pd

from training.extset_adapter import (load_task1, load_task2, case_split,
                                     expert_consensus)

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


# ── task1 ──
t1 = load_task1()
n1 = len(t1["y"])
ok(n1 == 167503, f"task1 read count {n1}")
ok(len(np.unique(t1["seg"])) == 5000 and len(np.unique(t1["user"])) == 643,
   "task1 5,000 segs / 643 raters")
ok(np.isfinite(t1["s_loo"]).all() and np.isfinite(t1["s_shipped"]).all(),
   "task1 signals all finite")
ok(set(np.unique(t1["gold"])) == {0, 1} and 0.20 < t1["gold"].mean() < 0.28,
   f"task1 source-gold prevalence {t1['gold'].mean():.3f} (read-weighted)")

# F40: removing own vote must shift the signal AWAY from the own response.
shift = t1["s_shipped"] - t1["s_loo"]          # full − LOO, shipped≈full map
signed = np.where(t1["y"] == 1, shift, -shift)
ok(signed.mean() > 0.01,
   f"LOO self-shift positive toward own response (mean {signed.mean():+.4f})")

# psychometric monotonicity on the LOO signal
q = np.quantile(t1["s_loo"], np.linspace(0, 1, 11))
rates = [t1["y"][(t1["s_loo"] >= a) & (t1["s_loo"] < b)].mean()
         for a, b in zip(q[:-1], q[1:])]
ok(all(np.diff(rates) > -0.02) and rates[-1] - rates[0] > 0.6,
   f"P(say|s_loo) decile curve monotone ({rates[0]:.3f}→{rates[-1]:.3f})")

# ── task2 ──
cons = expert_consensus(3)
ok(len(cons) == 3789, f"≥3/4 expert consensus on {len(cons)} cases")
t2 = load_task2()
n2 = len(t2["label"])
ok(n2 == 125860, f"task2 valid read count {n2}")
ok((t2["gold"] >= 0).sum() == 93185,
   f"reads on consensus-gold cases {(t2['gold'] >= 0).sum()}")
ok(t2["s_loo"].shape == (n2, 6) and np.isfinite(t2["s_loo"]).all(),
   "task2 LOO signal matrix (n,6) finite")
hit = t2["gold"] >= 0
correct = (t2["label"] == t2["gold"])[hit]
g = pd.Series(correct).groupby(t2["user"][hit]).agg(["size", "mean"])
m20 = g.loc[g["size"] >= 20, "mean"]
ok(len(m20) == 315 and 0.24 < m20.mean() < 0.27,
   f"task2 per-user clean-gold accuracy {m20.mean():.3f} (n={len(m20)} ≥20)")
ok(correct.mean() > m20.mean(),
   f"read-pooled acc {correct.mean():.3f} > per-user mean (F36 volume effect)")

# LOO self-shift per own-labeled domain, task2
own_s_full = t2["s_shipped"][np.arange(n2), t2["label"]]
own_s_loo = t2["s_loo"][np.arange(n2), t2["label"]]
ok((own_s_full - own_s_loo).mean() > 0,
   "task2 own-class shipped signal exceeds LOO (self-vote present)")

# ── case split ──
m = case_split(t2["seg"], t2["gold"])
fit_segs, hold_segs = set(t2["seg"][m]), set(t2["seg"][~m])
ok(not (fit_segs & hold_segs), "case split: no seg in both halves")
ok(0.45 < m.mean() < 0.55, f"case split read balance {m.mean():.3f}")
m1 = case_split(t1["seg"], t1["gold"])
ok(not (set(t1["seg"][m1]) & set(t1["seg"][~m1])), "task1 split valid")

# map sanity: LOO signal stays on the shipped scale
r = np.corrcoef(t1["s_shipped"], t1["s_loo"])[0, 1]
ok(r > 0.95, f"shipped vs LOO signal corr {r:.3f} (map faithful)")

print(f"\nEXTSET ADAPTER SMOKE TEST PASSED — {checks} checks.")
