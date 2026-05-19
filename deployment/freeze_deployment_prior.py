"""Build the frozen deployment artifact for the multi-task certification test.

What this writes (under data/deployment_prior/):

  Sigma.csv              (2K)x(2K) prior covariance (uniform r_ell, rho=0,
                         indep t) — 14x14 at K=7 (Phase 4.4-B), 12x12 at K=6
  ell_thresholds.csv     per-task ell* (Youden) and AUROC threshold
  case_bank.csv          per (task, seg_id, s_mean, s_sd) — frozen item bank
  summary.json           r_ell, n cases per task, source fit refs

The deployment Σ uses one scalar r_ell ≈ +0.45 estimated by the block
variant of the hierarchical fit. All other off-diagonals (t-t, t-ℓ) are 0.

ell*_k thresholds: per task, the Youden-index point on the empirical
distribution of single-task ℓ̂_i over the gold-rater pool, separating
"experts" from "non-experts" by group label. Experts are anyone in
{sparcnet50K, profiler_iiic, Super8, spikeed_expert, Bonobo} for the
IIIC tasks; for spike, just the Super8 / Bonobo / spikeed_expert pool.

UNIFIED-MERGE PROVENANCE (Phase 4.4-B, 2026-05-19). Faithful port of the
PI scripts/freeze_deployment_prior.py. Documented, auditable changes:
  - hardcoded absolute-ROOT -> repo-root self-location
    (Path(__file__).resolve().parents[1]); FITS/HIER_BLOCK/OUT derived
    paths byte-identical; HIER_BLOCK.relative_to(ROOT) still valid.
  - TASKS 6 -> 7: append the real "other" (the per-task loops,
    build_sigma, slot_names and the Youden block all iterate TASKS
    generically, so K=7 needs nothing else). Σ becomes 14x14.
UN-EXERCISED until Phase 4.6 (no re-freeze run per the locked 4.4
scope — 4.6 runs this on the unified corpus with v13 ℓ*).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repo root (was hardcoded)
FITS = ROOT / "data/labels/fits"
HIER_BLOCK = ROOT / "data/labels/fits_hier_block"
OUT = ROOT / "data/deployment_prior"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]
K = len(TASKS)
DIM = 2 * K
SPIKE_EXPERT_GROUPS = {"Super8", "spikeed_expert", "Bonobo"}
IIIC_EXPERT_GROUPS = {"sparcnet50K", "profiler_iiic"}


def build_sigma(r_ell: float) -> np.ndarray:
    """One-r ℓ-block prior with independent biases and ρ=0."""
    S = np.eye(DIM)
    for k in range(K):
        for kp in range(K):
            if k == kp:
                continue
            S[2 * k + 1, 2 * kp + 1] = r_ell   # ℓ_k vs ℓ_k'
            # t-t, t-ℓ, ℓ-t off-diagonals stay 0
    return S


def _primary_group(groups_json):
    import json as _json
    if pd.isna(groups_json) or groups_json == "":
        return "Other"
    try:
        gs = _json.loads(groups_json)
    except Exception:
        return "Other"
    return gs[0] if gs else "Other"


def youden_threshold(values, is_expert):
    """Find ℓ* maximizing P(expert ≥ ℓ*) - P(non-expert ≥ ℓ*).

    Returns (ell_star, j, auroc) where j is Youden index, auroc is the
    discriminability of the binary "expert" label by ℓ on this task.
    """
    ex = np.asarray(values)[np.asarray(is_expert).astype(bool)]
    nx = np.asarray(values)[~np.asarray(is_expert).astype(bool)]
    if len(ex) == 0 or len(nx) == 0:
        return float("nan"), float("nan"), float("nan")
    cands = np.sort(np.unique(values))
    best = None
    for c in cands:
        tpr = float(np.mean(ex >= c))
        fpr = float(np.mean(nx >= c))
        j = tpr - fpr
        if best is None or j > best[1]:
            best = (float(c), j, tpr, fpr)
    # AUROC via Mann-Whitney
    pairs = 0
    wins = 0
    for a in ex:
        for b in nx:
            pairs += 1
            wins += (a > b) + 0.5 * (a == b)
    auroc = float(wins / max(pairs, 1))
    return best[0], best[1], auroc


def main():
    # 1) read block fit
    with open(HIER_BLOCK / "factor_summary.json") as f:
        block_summary = json.load(f)
    block_corr = pd.read_csv(HIER_BLOCK / "correlation.csv", index_col=0)
    ell_cols = [c for c in block_corr.columns if c.startswith("l_")]
    ell_block = block_corr.loc[ell_cols, ell_cols].values
    # average all off-diagonal ℓ-ℓ entries → r_ell
    mask = ~np.eye(K, dtype=bool)
    r_ell = float(ell_block[mask].mean())
    print(f"r_ell (uniform ℓ-block correlation) = {r_ell:+.4f}", flush=True)

    # 2) build Σ
    Sigma = build_sigma(r_ell)
    slot_names = []
    for t in TASKS:
        slot_names.append(f"t_{t}")
        slot_names.append(f"l_{t}")
    pd.DataFrame(Sigma, index=slot_names, columns=slot_names).to_csv(OUT / "Sigma.csv")
    print(f"wrote {OUT/'Sigma.csv'}")

    # 3) case bank: per task, copy cases.csv (s_mean, s_sd, n_raters, pos_rate)
    bank_rows = []
    for task in TASKS:
        df = pd.read_csv(FITS / task / "cases.csv")
        df["task"] = task
        bank_rows.append(df)
    bank = pd.concat(bank_rows, ignore_index=True)
    bank = bank[["task", "seg_id", "s_mean", "s_sd", "n_raters", "pos_rate"]]
    bank.to_csv(OUT / "case_bank.csv", index=False)
    print(f"wrote {OUT/'case_bank.csv'}  ({len(bank):,} cases)")

    # 4) per-task ℓ* Youden thresholds
    thresh_rows = []
    for task in TASKS:
        r = pd.read_csv(FITS / task / "raters.csv")
        r["primary_group"] = r["groups"].apply(_primary_group)
        expert_groups = SPIKE_EXPERT_GROUPS if task == "spike" else (
            IIIC_EXPERT_GROUPS | SPIKE_EXPERT_GROUPS)
        r["is_expert"] = r["primary_group"].isin(expert_groups)
        ell_star, j, auroc = youden_threshold(r["ell_mean"].values,
                                              r["is_expert"].values)
        n_ex = int(r["is_expert"].sum())
        n_nx = int((~r["is_expert"]).sum())
        thresh_rows.append({
            "task": task,
            "ell_star": ell_star,
            "youden_j": j,
            "auroc_expert_vs_nonexpert": auroc,
            "n_experts": n_ex,
            "n_non_experts": n_nx,
        })
        print(f"  {task:>8}: ℓ* = {ell_star:+.3f}  J = {j:.3f}  "
              f"AUROC = {auroc:.3f}  ({n_ex} expert / {n_nx} non-expert)",
              flush=True)
    pd.DataFrame(thresh_rows).to_csv(OUT / "ell_thresholds.csv", index=False)
    print(f"wrote {OUT/'ell_thresholds.csv'}")

    # 5) summary
    summary = {
        "r_ell": r_ell,
        "rho": 0.0,
        "t_t_offdiag": 0.0,
        "sigma_t": 1.0,
        "sigma_ell": 1.0,
        "tasks": TASKS,
        "K": K,
        "DIM": DIM,
        "source_block_fit": str(HIER_BLOCK.relative_to(ROOT)),
        "block_summary": block_summary,
        "n_cases_per_task": {t: int((bank.task == t).sum()) for t in TASKS},
        "thresholds": {row["task"]: {
            "ell_star": row["ell_star"],
            "youden_j": row["youden_j"],
            "auroc": row["auroc_expert_vs_nonexpert"],
        } for row in thresh_rows},
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
