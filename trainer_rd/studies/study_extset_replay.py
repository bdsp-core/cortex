"""W2 — measurement-layer validation at scale (SAP §3; F34 protocol, n≈640).

Static per-domain TaskFilter replay (rule="static": pure Bayes measurement,
no dynamics; smear_w=True per D19) of every EXTSET user's real read sequence,
LOO signals (F40). One filter per (user, domain); task2 reads are one-vs-rest
observations for each of the 6 domains; task1 is the binary domain1 frame.

Endpoints:
  P2  (task2)  Spearman(mean_k ℓ̂_uk, clean-gold accuracy_u), users ≥20 reads
  P2b (task1)  Spearman(ℓ̂_u, source-gold accuracy_u), users ≥20 reads
Secondaries: per-domain ρ_k (Holm), ℓ̂ vs mean Qscore, tier known-groups AUC,
F34-pathology cell rate (high ℓ̂ + low accuracy).

Run:  python3 -m studies.study_extset_replay [--smoke]
Out:  figures/data_extset_replay.npz
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

from training.extset_adapter import load_task1, load_task2, _root
from training.learner_sim import LearnerParams
from training.training_filter import TaskFilter

N_PART = 600
SEED = 20260612
MIN_READS = 20


def replay_user(s, y, ystar, s_sd, seed):
    """Static filter over one (user, domain) sequence → (ℓ̂, sd_ℓ)."""
    rng = np.random.default_rng(seed)
    params = LearnerParams(alpha_t=0.0, alpha_sigma=0.0, q_t=0.0, q_sigma=0.0,
                           rho=0.6, rule="static")
    filt = TaskFilter(rng.standard_normal(N_PART), rng.standard_normal(N_PART),
                      params, ess_frac=0.5, seed=seed, smear_w=True)
    for i in range(len(s)):
        filt.step(float(s[i]), int(y[i]), int(ystar[i]),
                  s_sd=float(s_sd[i]), feedback=False)
    return filt.mean()[1], filt.sd()[1]


def replay_frame(user, order_key, s, y, ystar, s_sd, smoke, tag):
    """Replay every user in one (domain) frame. Returns DataFrame."""
    df = pd.DataFrame({"user": user, "key": order_key})
    rows = []
    uids = np.unique(user)
    if smoke:
        uids = np.random.default_rng(0).choice(
            uids, min(40, len(uids)), replace=False)
    t0 = time.time()
    for j, u in enumerate(uids):
        idx = np.where(user == u)[0]
        idx = idx[np.argsort(order_key[idx], kind="stable")]
        ell, sd = replay_user(s[idx], y[idx], ystar[idx], s_sd[idx],
                              seed=SEED + int(u) % 100000)
        rows.append((u, len(idx), float((y[idx] == ystar[idx]).mean()),
                     ell, sd))
        if j % 200 == 0:
            print(f"    [{tag}] {j}/{len(uids)} users ({time.time()-t0:.0f}s)",
                  flush=True)
    return pd.DataFrame(rows, columns=["user", "n", "acc", "ell", "sd"])


def holm(pvals):
    order = np.argsort(pvals)
    adj, mx = np.empty_like(pvals), 0.0
    for rank, i in enumerate(order):
        mx = max(mx, pvals[i] * (len(pvals) - rank))
        adj[i] = min(mx, 1.0)
    return adj


def main(smoke=False):
    out = {}

    print("== task1 (binary) ==", flush=True)
    t1 = load_task1()
    r1 = replay_frame(t1["user"], np.arange(len(t1["y"])), t1["s_loo"],
                      t1["y"], t1["gold"], t1["s_sd"], smoke, "t1")
    a1 = r1[r1.n >= MIN_READS]
    rho1 = spearmanr(a1["ell"], a1["acc"])
    print(f"P2b task1: Spearman(ℓ̂, acc) = {rho1.statistic:.3f} "
          f"(p={rho1.pvalue:.2e}, n={len(a1)})", flush=True)
    out["t1"] = r1

    print("== task2 (6 one-vs-rest frames) ==", flush=True)
    t2 = load_task2()
    hit = t2["gold"] >= 0
    user, seg = t2["user"][hit], t2["seg"][hit]
    tord = t2["t"][hit].astype(np.int64)
    frames = []
    for k in range(6):
        rk = replay_frame(user, tord, t2["s_loo"][hit, k],
                          (t2["label"][hit] == k).astype(int),
                          (t2["gold"][hit] == k).astype(int),
                          t2["s_sd"][hit, k], smoke, f"t2-d{k+2}")
        rk["dom"] = k
        frames.append(rk)
    r2 = pd.concat(frames)
    out["t2"] = r2

    # per-user 6-class accuracy + mean ℓ̂ across domains
    lab, gld = t2["label"][hit], t2["gold"][hit]
    acc6 = (pd.DataFrame({"u": user, "c": (lab == gld).astype(float)})
            .groupby("u")["c"].agg(["size", "mean"]))
    piv = r2.pivot_table(index="user", values="ell", aggfunc="mean")
    j = piv.join(acc6).dropna()
    a2 = j[j["size"] >= MIN_READS]
    rho2 = spearmanr(a2["ell"], a2["mean"])
    print(f"P2 task2: Spearman(mean ℓ̂, 6-class acc) = {rho2.statistic:.3f} "
          f"(p={rho2.pvalue:.2e}, n={len(a2)})", flush=True)

    # secondaries
    pks = []
    for k in range(6):
        rk = r2[(r2.dom == k) & (r2.n >= MIN_READS)]
        sp = spearmanr(rk["ell"], rk["acc"])
        pks.append((k + 2, sp.statistic, sp.pvalue, len(rk)))
    padj = holm(np.array([p for _, _, p, _ in pks]))
    for (d, rho, p, n), pa in zip(pks, padj):
        print(f"  domain{d}: ρ={rho:.3f} p_holm={pa:.2e} (n={n})", flush=True)

    qm = (pd.DataFrame({"u": user, "q": t2["qscore"][hit]})
          .groupby("u")["q"].mean())
    jq = piv.join(qm).join(acc6).dropna()
    jq = jq[jq["size"] >= MIN_READS]
    rq = spearmanr(jq["ell"], jq["q"])
    print(f"  ℓ̂ vs mean Qscore: ρ={rq.statistic:.3f} (n={len(jq)})",
          flush=True)

    # tier known-groups (task2 cohort)
    users_tbl = pd.read_csv(_root("extset_users_scrubbed.csv"))
    tier = users_tbl.set_index("extset_user_id")["derived_tier"]
    jt = piv.join(acc6).join(tier.rename("tier")).dropna()
    jt = jt[jt["size"] >= MIN_READS]
    ex, nv = jt[jt.tier == "expert"]["ell"], jt[jt.tier == "novice"]["ell"]
    if len(ex) >= 5 and len(nv) >= 5:
        r = rankdata(np.r_[ex, nv])
        auc = (r[:len(ex)].mean() - (len(ex) + 1) / 2) / len(nv)
        print(f"  tier AUC(expert>novice on ℓ̂) = {auc:.3f} "
              f"(n={len(ex)} vs {len(nv)})", flush=True)
    else:
        auc = np.nan

    # F34 pathology: top-quartile ℓ̂ with bottom-quartile accuracy, per domain
    cells = r2[r2.n >= MIN_READS]
    path = ((cells["ell"] >= cells.groupby("dom")["ell"].transform(
        lambda v: v.quantile(0.75)))
        & (cells["acc"] <= cells.groupby("dom")["acc"].transform(
            lambda v: v.quantile(0.25))))
    print(f"  pathology cells (hi-ℓ̂ × lo-acc): {path.sum()}/{len(cells)} "
          f"= {path.mean():.3f}", flush=True)

    if not smoke:
        np.savez("figures/data_extset_replay.npz",
                 t1=r1.to_numpy(), t1_cols=np.array(r1.columns, dtype=object),
                 t2=r2.to_numpy(), t2_cols=np.array(r2.columns, dtype=object),
                 p2=(rho2.statistic, rho2.pvalue, len(a2)),
                 p2b=(rho1.statistic, rho1.pvalue, len(a1)),
                 per_domain=np.array(pks), per_domain_holm=padj,
                 qscore_rho=(rq.statistic, rq.pvalue, len(jq)),
                 tier_auc=auc, pathology_rate=path.mean())
        print("saved figures/data_extset_replay.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
