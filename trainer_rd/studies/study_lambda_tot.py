"""M27 — λ(time-on-task) on the real EXTSET reads (the M22-queued
Phase-3 channel; F92).

QUESTION. Does the lapse rate grow within a sitting? The lapse channel's
signature is a within-sitting accuracy DECLINE ON EASY reads (a lapse
floors accuracy at 1−λ regardless of difficulty; a sensitivity/d′
decline hits hard reads first). The M22 fatigue guard (F79) already
covers the operational risk in serving; this study asks whether the
PHASE-3 OFFLINE MODEL needs a λ(t) term, and if so supplies its
real-data prior.

METHOD. 125,860 task2 reads → per-user sittings (gap > 30 min splits);
time-on-task = minutes since sitting start. Reads with expert-consensus
gold only. Difficulty = LOO consensus margin m_i = s_loo[gold] −
max_other (F40 signals — never the shipped bank columns). Easy = top
margin quartile, hard = bottom. Per-user linear slope of correct vs
time-on-task within each set (users with ≥30 reads in the set and ≥20
min span), then the user-weighted mean slope with a 2,000-rep user-level
bootstrap CI. The easy-set slope converts to dλ/dt ≈ −slope (accuracy on
easy reads ≈ (1−λ) · p_max with p_max ≈ 1).

Run:  python3 -m studies.study_lambda_tot [--smoke]
Out:  figures/data_lambda_tot.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from training.extset_adapter import load_task2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "data_lambda_tot.npz")

GAP_MIN = 30.0
MIN_READS = 30
MIN_SPAN_MIN = 20.0


def per_user_slopes(user, tot_min, correct, mask):
    """OLS slope of correct on time-on-task (per user, within mask)."""
    slopes, weights = [], []
    for u in np.unique(user[mask]):
        m = mask & (user == u)
        if m.sum() < MIN_READS:
            continue
        x, y = tot_min[m], correct[m].astype(float)
        if x.max() - x.min() < MIN_SPAN_MIN:
            continue
        xc = x - x.mean()
        slopes.append(float((xc * (y - y.mean())).sum() / (xc * xc).sum()))
        weights.append(m.sum())
    return np.array(slopes), np.array(weights)


def boot_ci(slopes, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(slopes, size=len(slopes))))
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    d = load_task2()
    ok = d["gold"] >= 0
    seg_ok = ok
    user, t = d["user"][seg_ok], d["t"][seg_ok]
    gold, label = d["gold"][seg_ok], d["label"][seg_ok]
    s_loo = d["s_loo"][seg_ok]
    if args.smoke:
        keep_u = np.unique(user)[:80]
        m = np.isin(user, keep_u)
        user, t, gold, label, s_loo = (user[m], t[m], gold[m], label[m],
                                       s_loo[m])
    correct = (label == gold).astype(int)
    # per-user sittings + time-on-task
    order = np.lexsort((t, user))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    u_s, t_s = user[order], t[order]
    gap = np.diff(t_s).astype("timedelta64[s]").astype(float) / 60.0
    new_sit = np.concatenate([[True], (np.diff(u_s) != 0) | (gap > GAP_MIN)])
    sit_id = np.cumsum(new_sit)
    sit_start = {}
    tot_s = np.empty(len(t_s))
    for i in range(len(t_s)):
        if sit_id[i] not in sit_start:
            sit_start[sit_id[i]] = t_s[i]
        tot_s[i] = (t_s[i] - sit_start[sit_id[i]]) \
            / np.timedelta64(1, "m")
    tot_min = tot_s[inv]
    # difficulty margin from the LOO signals
    g_sig = s_loo[np.arange(len(gold)), gold]
    other = s_loo.copy()
    other[np.arange(len(gold)), gold] = -np.inf
    margin = g_sig - other.max(axis=1)
    q1, q3 = np.percentile(margin, [25, 75])
    easy, hard = margin >= q3, margin <= q1
    res = {}
    for name, mask in (("easy", easy), ("hard", hard),
                       ("all", np.ones_like(easy, bool))):
        sl, w = per_user_slopes(user, tot_min, correct, mask)
        if len(sl) == 0:
            continue
        lo, hi = boot_ci(sl)
        res[name] = dict(n_users=int(len(sl)),
                         mean_slope_per_min=float(np.mean(sl)),
                         ci_lo=lo, ci_hi=hi,
                         acc=float(np.mean(correct[mask])),
                         dlam_per_hour=float(-np.mean(sl) * 60.0))
        print(f"  {name:<5}: users {len(sl):4d}  acc {res[name]['acc']:.3f}"
              f"  slope {np.mean(sl):+.2e}/min "
              f"[{lo:+.2e}, {hi:+.2e}]  ⇒ dλ/dt ≈ "
              f"{res[name]['dlam_per_hour']:+.4f}/h")
    # time-bin accuracy on the easy set (descriptive)
    bins = [0, 10, 20, 40, 1e9]
    prof = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = easy & (tot_min >= a) & (tot_min < b)
        if m.sum() > 200:
            prof.append((a, float(np.mean(correct[m])), int(m.sum())))
    print("  easy-set accuracy by time-on-task:",
          [(int(a), round(x, 3), n) for a, x, n in prof])
    res["profile_easy"] = prof
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, result=np.array([json.dumps(res)]))
    print(f"saved {OUT}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
