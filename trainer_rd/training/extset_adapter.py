"""EXTSET loaders + exact leave-one-user-out consensus signals (SAP §0/§9.1).

Implements the F40 protocol: the PRIMARY analysis signal is the LOO smoothed-
probit consensus score recomputed from read-level votes; the shipped s_mean is
exposed only as `s_shipped` (sensitivity analyses). Guards:
  - F35: truth is NEVER taken from bank plurality on EXTSET segments — gold
    comes from the expert panel (task2) or the source label (task1) only.
  - F40: task1 vote columns must reconcile exactly with the aggregated reads;
    task2 vote bases are BUILT from the reads + the 4-expert panel (the bank's
    vote columns diverge from this release — F42 — and are never used).

All loaders return plain numpy arrays keyed by integer codes (row order =
read order in the source files). Run from the scratch root.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# F40 map constants — recovered by inverting the task1 vote→s_mean pipeline
# (R²=0.966; SAP §0). FIXED for all analyses; do not refit per dataset.
LOO_A, LOO_B, LOO_C, LOO_D = 0.78, 6.73, 1.102, 1.123

TASK2_CLASSES = ("domain2", "domain3", "domain4", "domain5", "domain6",
                 "domain7")
EXPERT_COLS = ("label_expert1", "label_expert2", "label_expert3",
               "label_expert4")


def consensus_signal(yes, n):
    """Smoothed-probit consensus score on the engine s-scale (F40 map)."""
    f = (np.asarray(yes, float) + LOO_A) / (np.asarray(n, float) + LOO_A + LOO_B)
    return LOO_C * norm.ppf(np.clip(f, 1e-6, 1 - 1e-6)) + LOO_D


_EXTSET_DIR = os.path.join(_ROOT, "data", "extset")


def _root(path):
    """Resolve a release file. The EXTSET release lives in `data/extset/`
    (M27 layout; was the scratch root); explicit `data/…` paths resolve
    from the scratch root unchanged."""
    if path.replace("\\", "/").startswith("data/"):
        return os.path.join(_ROOT, path)
    return os.path.join(_EXTSET_DIR, path)


def expert_consensus(min_agree=3):
    """case_id → class for cases where ≥min_agree of the 4 experts agree."""
    exp = pd.read_csv(_root("extset_expert_labels_scrubbed.csv"))
    lab = exp[list(EXPERT_COLS)].to_numpy()
    out = {}
    for cid, row in zip(exp["case_id"], lab):
        cls, cnt = Counter(row).most_common(1)[0]
        if cnt >= min_agree:
            out[int(cid)] = cls
    return out


def load_task1():
    """task1/domain1 binary reads with LOO + shipped signals and source gold.

    Returns dict of aligned arrays over the 167,503 reads:
      seg, user (rater_id), y, gold, s_loo, s_shipped, s_sd
    """
    rd = pd.read_csv(_root("extset_task1_reads.csv"))
    sg = pd.read_csv(_root("extset_task1_signals.csv")).set_index("seg_id")
    c2s = pd.read_csv(_root("case_id_to_seg_id_scrubbed.csv"))
    d1 = pd.read_csv(_root("domain1_case_source_map_scrubbed.csv")).merge(
        c2s, left_on="Case ID", right_on="extset_case_id")
    gold_map = dict(zip(d1["new_seg_id"],
                        (d1["source_or_gold_label"] == "domain1").astype(int)))

    y = (rd["value"] == "domain1").to_numpy(int)
    seg = rd["seg_id"].to_numpy()
    yes = sg.loc[seg, "votes_domain1_yes"].to_numpy(float)
    no = sg.loc[seg, "votes_domain1_no"].to_numpy(float)
    n = yes + no

    # F40 guard: votes must equal the aggregated reads exactly.
    agg = rd.assign(yv=y).groupby("seg_id")["yv"].agg(["sum", "count"])
    chk = sg.join(agg)
    if not ((chk["votes_domain1_yes"] == chk["sum"]).all()
            and (chk["votes_domain1_yes"] + chk["votes_domain1_no"]
                 == chk["count"]).all()):
        raise AssertionError("F40 guard: task1 votes != aggregated reads")

    return dict(
        seg=seg,
        user=rd["rater_id"].to_numpy(),
        y=y,
        gold=np.array([gold_map[s] for s in seg], int),
        s_loo=consensus_signal(yes - y, n - 1),
        s_shipped=sg.loc[seg, "s_mean_domain1"].to_numpy(float),
        s_sd=sg.loc[seg, "s_sd_domain1"].to_numpy(float),
    )


def load_task2(min_agree=3):
    """task2 6-class reads with per-domain LOO/shipped signal matrices.

    Returns dict over the 125,860 valid reads:
      seg, user, label (0..5 idx into TASK2_CLASSES), gold (idx or -1 where no
      ≥min_agree consensus), qscore, t (datetime64), s_loo/s_shipped/s_sd
      (n_reads × 6), and `classes`.
    """
    nov = pd.read_csv(_root("extset_novice_labels_scrubbed.csv"),
                      encoding="utf-8-sig")
    nov.columns = [c.strip() for c in nov.columns]
    nov = nov[nov["Labeling State"] == "Gold Standard"].copy()
    nov["label"] = nov["User Label"].str.strip("'")

    c2s = pd.read_csv(_root("case_id_to_seg_id_scrubbed.csv"))
    nov = nov.merge(c2s, left_on="Case ID", right_on="extset_case_id")

    sig_cols = (["seg_id"] + [f"s_mean_{c}" for c in TASK2_CLASSES]
                + [f"s_sd_{c}" for c in TASK2_CLASSES])
    sg = pd.read_csv(_root("data/segment_signals_general.csv"),
                     usecols=sig_cols, low_memory=False).set_index("seg_id")

    cls_idx = {c: i for i, c in enumerate(TASK2_CLASSES)}
    label = nov["label"].map(cls_idx).to_numpy()
    seg = nov["new_seg_id"].to_numpy()
    own = np.eye(len(TASK2_CLASSES))[label]                 # one-hot own vote

    # Vote base built FROM THE READS + the 4-expert panel — never from the
    # bank's vote columns, which diverge from this release (miss ~2.2k of
    # these reads while holding ~14.5k extra historical votes; F42).
    exp = pd.read_csv(_root("extset_expert_labels_scrubbed.csv")).merge(
        c2s, left_on="case_id", right_on="extset_case_id")
    evotes = np.zeros((len(exp), len(TASK2_CLASSES)))
    for col in EXPERT_COLS:
        evotes += np.eye(len(TASK2_CLASSES))[
            exp[col].map(cls_idx).to_numpy()]
    evotes = pd.DataFrame(evotes, index=exp["new_seg_id"].to_numpy())

    rvotes = (pd.DataFrame(own).assign(seg=seg).groupby("seg").sum())
    base = rvotes.add(evotes.loc[evotes.index.isin(rvotes.index)],
                      fill_value=0.0)
    votes = base.loc[seg].to_numpy(float)                   # (n, 6)
    n_tot = votes.sum(axis=1, keepdims=True)

    cons = expert_consensus(min_agree)
    gold = nov["Case ID"].map(cons).map(cls_idx).fillna(-1).to_numpy(int)

    return dict(
        seg=seg,
        user=nov["User ID"].to_numpy(),
        label=label,
        gold=gold,
        qscore=nov["Qscore"].to_numpy(float),
        t=pd.to_datetime(nov["Response Submitted At"], format="mixed",
                         utc=True).dt.tz_localize(None).to_numpy(),
        s_loo=consensus_signal(votes - own, n_tot - 1),
        s_shipped=sg.loc[seg, [f"s_mean_{c}" for c in TASK2_CLASSES]
                         ].to_numpy(float),
        s_sd=sg.loc[seg, [f"s_sd_{c}" for c in TASK2_CLASSES]].to_numpy(float),
        classes=TASK2_CLASSES,
    )


def case_split(seg, gold, frac=0.5, seed=20260612):
    """Deterministic 50/50 CASE-level split stratified by gold class.

    Returns boolean mask over reads: True = fit half. Never splits a case.
    """
    rng = np.random.default_rng(seed)
    segs = pd.DataFrame({"seg": seg, "gold": gold}).drop_duplicates("seg")
    fit = set()
    for _, grp in segs.groupby("gold"):
        ids = grp["seg"].to_numpy()
        rng.shuffle(ids)
        fit.update(ids[: int(round(frac * len(ids)))])
    return np.isin(seg, list(fit))
