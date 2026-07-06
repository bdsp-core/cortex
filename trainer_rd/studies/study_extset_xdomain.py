"""W4 — cross-domain skill structure from the dual-contest cohort (SAP §4).

The users who labeled in BOTH contests give real per-user skill estimates
spanning domain1 (task1, binary) × domains2–7 (task2, 6-class). Outputs:
  P4   Spearman(task1 accuracy, task2 6-class accuracy), users ≥20 reads each
  - disattenuated correlation via odd/even split-half reliability
  - per-pair domain accuracy correlation matrix (7×7, one-vs-rest accuracies)
  - per-tier skill summaries (population grounding for v15 priors)

Accuracy-based primary (model-free); the ℓ̂-based variant joins W2's replay
output when present (figures/data_extset_replay.npz).

Run:  python3 -m studies.study_extset_xdomain [--smoke]
Out:  figures/data_extset_xdomain.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from training.extset_adapter import load_task1, load_task2, _root

MIN_READS = 20
SEED = 20260612


def split_half_rho(df, col_user, col_correct, rng):
    """Odd/even split-half reliability of per-user accuracy (Spearman)."""
    df = df.copy()
    df["h"] = rng.integers(0, 2, len(df))
    g = df.groupby([col_user, "h"])[col_correct].agg(["size", "mean"])
    w = g["mean"].unstack()
    n = g["size"].unstack().min(axis=1)
    w = w[n >= MIN_READS // 2].dropna()
    r = spearmanr(w[0], w[1]).statistic
    return 2 * r / (1 + r), len(w)        # Spearman–Brown to full length


def main(smoke=False):
    rng = np.random.default_rng(SEED)
    u2r = pd.read_csv(_root("user_id_to_rater_id_scrubbed.csv"))
    r2u = dict(zip(u2r["new_rater_id"], u2r["extset_user_id"]))

    t1 = load_task1()
    d1 = pd.DataFrame({"u": [r2u[r] for r in t1["user"]],
                       "c": (t1["y"] == t1["gold"]).astype(float)})
    t2 = load_task2()
    hit = t2["gold"] >= 0
    d2 = pd.DataFrame({"u": t2["user"][hit],
                       "c": (t2["label"] == t2["gold"])[hit].astype(float),
                       "label": t2["label"][hit], "gold": t2["gold"][hit]})
    if smoke:
        keep = set(rng.choice(sorted(set(d1.u) & set(d2.u)), 60,
                              replace=False))
        d1, d2 = d1[d1.u.isin(keep)], d2[d2.u.isin(keep)]

    a1 = d1.groupby("u")["c"].agg(["size", "mean"])
    a2 = d2.groupby("u")["c"].agg(["size", "mean"])
    j = a1.join(a2, lsuffix="_1", rsuffix="_2").dropna()
    j = j[(j["size_1"] >= MIN_READS) & (j["size_2"] >= MIN_READS)]
    rho = spearmanr(j["mean_1"], j["mean_2"])
    print(f"P4: Spearman(domain1 acc, domains2-7 acc) = {rho.statistic:.3f} "
          f"(p={rho.pvalue:.2e}, n={len(j)})", flush=True)

    rel1, n1 = split_half_rho(d1, "u", "c", rng)
    rel2, n2 = split_half_rho(d2, "u", "c", rng)
    disatt = rho.statistic / np.sqrt(max(rel1, 1e-6) * max(rel2, 1e-6))
    print(f"  reliability: task1 {rel1:.3f} (n={n1}), task2 {rel2:.3f} "
          f"(n={n2}); disattenuated ρ = {disatt:.3f}", flush=True)

    # 7×7 one-vs-rest accuracy correlation matrix (domain1 + 6 task2 domains)
    cols = {"d1": a1["mean"]}
    for k in range(6):
        dk = d2[(d2.label == k) | (d2.gold == k)].copy()
        dk["ck"] = ((dk.label == k) == (dk.gold == k)).astype(float)
        gk = dk.groupby("u")["ck"].agg(["size", "mean"])
        cols[f"d{k+2}"] = gk.loc[gk["size"] >= MIN_READS, "mean"]
    M = pd.DataFrame(cols)
    corr = M.corr(method="spearman", min_periods=25)
    print("  7×7 skill correlation (Spearman):", flush=True)
    print(corr.round(2).to_string(), flush=True)

    # per-tier population summaries
    users_tbl = pd.read_csv(_root("extset_users_scrubbed.csv"))
    tier = users_tbl.set_index("extset_user_id")["derived_tier"]
    summ = (j.join(tier.rename("tier"))
            .groupby("tier")[["mean_1", "mean_2"]]
            .agg(["size", "mean", "std"]))
    print(summ.round(3).to_string(), flush=True)

    if not smoke:
        np.savez("figures/data_extset_xdomain.npz",
                 p4=(rho.statistic, rho.pvalue, len(j)),
                 reliability=(rel1, rel2, disatt),
                 corr=corr.to_numpy(), corr_labels=np.array(corr.columns,
                                                            dtype=object),
                 joint=j.reset_index().to_numpy())
        print("saved figures/data_extset_xdomain.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
