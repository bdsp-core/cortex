"""A3 — specification-curve (multiverse) analysis of the validity endpoints.

Branches:
  task2 (P2): gold {consensus≥3, unanimous, source} × signal {loo, shipped}
              × min-reads {10, 20, 50} × replay read-set {all, gold-only}
              = 36 specifications
  task1 (P2b): signal {loo, shipped} × min-reads {10, 20, 50} = 6

Key economy (verified): under rule="static" the filter's propagate() is a
no-op, so y_star never affects ℓ̂ — gold policy and thresholds are pure
post-processing. Only (signal × read-set × domain) requires replay:
26 jobs, parallel over the worker pool.

Run:  python3 -m studies.study_extset_speccurve [--smoke]
Out:  figures/data_extset_speccurve.npz
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from studies.study_extset_replay import replay_user
from training.extset_adapter import (expert_consensus, load_task1,
                                     load_task2, _root)

N_WORKERS = 40
GOLDS = ("cons3", "unanimous", "source")
SIGNALS = ("loo", "shipped")
MIN_READS = (10, 20, 50)
READSETS = ("all", "goldonly")

_T1 = _T2 = _GOLDMAPS = None     # fork-inherited


def _gold_maps(t2):
    seg = pd.read_csv(_root("extset_segment_map_scrubbed.csv"))
    cls = {f"domain{k}": k - 2 for k in range(2, 8)}
    source = dict(zip(seg["extset_case_id"],
                      seg["gold_label"].map(cls)))
    out = {}
    for name, m in (("cons3", expert_consensus(3)),
                    ("unanimous", expert_consensus(4))):
        out[name] = {c: cls[v] for c, v in m.items()}
    out["source"] = {c: v for c, v in source.items() if pd.notna(v)}
    return out


def _replay_job(job):
    """One (frame, signal, readset[, domain]) replay → per-user ℓ̂ table."""
    kind, sig, readset, k = job
    if kind == "t1":
        t = _T1
        s = t["s_loo"] if sig == "loo" else t["s_shipped"]
        user, y, ystar, s_sd = t["user"], t["y"], t["gold"], t["s_sd"]
        order = np.arange(len(y))
    else:
        t = _T2
        m = np.ones(len(t["label"]), bool) if readset == "all" \
            else t["gold"] >= 0
        S = t["s_loo"] if sig == "loo" else t["s_shipped"]
        user, s, s_sd = t["user"][m], S[m, k], t["s_sd"][m, k]
        y = (t["label"][m] == k).astype(int)
        ystar = (t["gold"][m] == k).astype(int)
        order = t["t"][m].astype(np.int64)
    rows = []
    for u in np.unique(user):
        idx = np.where(user == u)[0]
        idx = idx[np.argsort(order[idx], kind="stable")]
        ell, _ = replay_user(s[idx], y[idx], ystar[idx], s_sd[idx],
                             seed=20260612 + int(u) % 100000)
        rows.append((u, ell))
    return job, dict(rows)


def main(smoke=False):
    global _T1, _T2, _GOLDMAPS
    _T1, _T2 = load_task1(), load_task2()
    _GOLDMAPS = _gold_maps(_T2)
    if smoke:
        rng = np.random.default_rng(0)
        for t, ucol in ((_T1, "user"), (_T2, "user")):
            keep = set(rng.choice(np.unique(t[ucol]), 50, replace=False))
            m = np.isin(t[ucol], list(keep))
            for key in list(t):
                if isinstance(t[key], np.ndarray) and len(t[key]) == len(m):
                    t[key] = t[key][m]

    jobs = ([("t1", sig, "all", None) for sig in SIGNALS]
            + [("t2", sig, rs, k) for sig in SIGNALS for rs in READSETS
               for k in range(6)])
    with mp.Pool(min(N_WORKERS, len(jobs))) as pool:
        replays = dict(pool.map(_replay_job, jobs))
    print(f"replays done: {len(replays)} jobs", flush=True)

    # case-level gold per policy → per-read gold idx
    case = pd.read_csv(_root("case_id_to_seg_id_scrubbed.csv"))
    seg2case = dict(zip(case["new_seg_id"], case["extset_case_id"]))
    spec = []

    # task2 branches
    lab, user2, seg2 = _T2["label"], _T2["user"], _T2["seg"]
    for gname in GOLDS:
        gm = _GOLDMAPS[gname]
        gold = np.array([gm.get(seg2case.get(s2, -1), -1) for s2 in seg2])
        ok = gold >= 0
        acc = (pd.DataFrame({"u": user2[ok],
                             "c": (lab[ok] == gold[ok]).astype(float)})
               .groupby("u")["c"].agg(["size", "mean"]))
        for sig in SIGNALS:
            for rs in READSETS:
                ell = pd.DataFrame(
                    {k: replays[("t2", sig, rs, k)] for k in range(6)}
                ).mean(axis=1)
                for mr in MIN_READS:
                    j = acc[acc["size"] >= mr].join(ell.rename("ell")
                                                    ).dropna()
                    rho = spearmanr(j["ell"], j["mean"])
                    spec.append(("t2", gname, sig, rs, mr,
                                 rho.statistic, rho.pvalue, len(j)))

    # task1 branches
    acc1 = (pd.DataFrame({"u": _T1["user"],
                          "c": (_T1["y"] == _T1["gold"]).astype(float)})
            .groupby("u")["c"].agg(["size", "mean"]))
    for sig in SIGNALS:
        ell = pd.Series(replays[("t1", sig, "all", None)])
        for mr in MIN_READS:
            j = acc1[acc1["size"] >= mr].join(ell.rename("ell")).dropna()
            rho = spearmanr(j["ell"], j["mean"])
            spec.append(("t1", "source", sig, "all", mr,
                         rho.statistic, rho.pvalue, len(j)))

    df = pd.DataFrame(spec, columns=["frame", "gold", "signal", "readset",
                                     "min_reads", "rho", "p", "n"])
    print(df.sort_values("rho").to_string(index=False), flush=True)
    print(f"\nrho range: task2 [{df[df.frame=='t2'].rho.min():.3f}, "
          f"{df[df.frame=='t2'].rho.max():.3f}]  "
          f"task1 [{df[df.frame=='t1'].rho.min():.3f}, "
          f"{df[df.frame=='t1'].rho.max():.3f}]", flush=True)
    if not smoke:
        np.savez("figures/data_extset_speccurve.npz",
                 spec=df.to_numpy(dtype=object),
                 cols=np.array(df.columns, dtype=object))
        print("saved figures/data_extset_speccurve.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
