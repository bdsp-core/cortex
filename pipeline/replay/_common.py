"""Phase 7 sub-step 3-C — shared replay helpers.

Reused by both engine drivers:
  * deployment/replay/run_deployment_replay.py (the 7.3-B harness,
    extended in 7.3-C with parallelization + fitted-θ comparator)
  * bridge/run_multi_auroc_replay.py (the new 7.3-C Mode-A harness)

Provides:
  * `load_replay_bank_cached()` — per-worker SHARED cache for the
    100MB rater_replay_bank + summary + raters.csv + sdt_fits join
    (same pattern as scripts/run_tier2_oc_simstudy.py's _SHARED).
  * `fitted_theta_for_rater(rater_id, tasks)` — convert the rater's
    fitted (σ, θ) from sdt_fits to the engine's true_params layout
    [t1, ℓ1, t2, ℓ2, ...] using the reference convention
    ℓ = -log(σ), t = θ (see bridge/_common.build_true_params).
  * Task mapping: engine task ↔ sdt_fits domain.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent

# Engine task → sdt_fits domain. The IIIC sdt_fits uses sparcnet labels
# (sz, lpd, gpd, lrda, grda, iic); spike fits live in sdt_fits.spike.csv.
# `other` (the K=7 deployment task) maps to `iic` (the SDT-fit name for
# the IIIC 'other' class), matching the v13 cert_config _V13_TASK_KEY
# at deployment/simulate_test.py:498.
ENGINE_TASK_TO_SDT_DOMAIN = {
    "spike":   "spike",
    "seizure": "sz",
    "lpd":     "lpd",
    "gpd":     "gpd",
    "lrda":    "lrda",
    "grda":    "grda",
    "other":   "iic",
}

# Per-worker SHARED read-only cache (spawn-safe).
_SHARED: Dict[str, Any] = {}


def load_replay_bank_cached() -> Dict[str, Any]:
    """Lazy-load + cache: rater_replay_bank, summary, raters, sdt_fits
    join. ~30 s first-call cost (the 102MB gz decompress); subsequent
    calls return from `_SHARED` instantly. Used inside parallel_map
    workers (each worker process loads once at first task)."""
    if "bank" in _SHARED:
        return _SHARED
    pkl_path = _REPO / "data" / "replay" / "rater_replay_bank.indexed.pkl"
    bank_path = _REPO / "data" / "replay" / "rater_replay_bank.csv.gz"
    summary_path = _REPO / "data" / "replay" / "rater_replay_summary.csv"
    raters_path = _REPO / "data" / "labels" / "raters.csv"
    sdt_iiic_path = _REPO / "data" / "engine_inputs" / "sdt_fits.csv"
    sdt_spike_path = _REPO / "data" / "engine_inputs" / "sdt_fits.spike.csv"
    if not bank_path.is_file():
        raise SystemExit(
            f"  ✗ replay bank missing — run "
            f"`python -m pipeline.replay.build_rater_replay_bank`")

    # ── Phase 7 sub-3-C performance: prefer the pre-indexed pickle
    # (rater_id, task)-sorted multi-index, ~60× faster to load than
    # the gz CSV) when present. The .indexed.pkl is built from the
    # canonical .csv.gz by the bank-builder (Phase 7 sub-3-C). Falls
    # back to building the multi-index from the CSV if pickle absent.
    if pkl_path.is_file():
        _SHARED["bank_by_rt"] = pd.read_pickle(pkl_path)
        _SHARED["bank"] = _SHARED["bank_by_rt"].reset_index()
    else:
        bank = pd.read_csv(bank_path)
        _SHARED["bank"] = bank
        _SHARED["bank_by_rt"] = (
            bank.set_index(["rater_id", "task"]).sort_index())
    _SHARED["summary"] = pd.read_csv(summary_path)
    _SHARED["raters"] = pd.read_csv(raters_path)
    # Per-rater Y lookup cache: built lazily on first per-rater
    # call (avoids paying the full pivot cost upfront).
    _SHARED["_y_cache"] = {}

    # sdt_fits union (6 IIIC + spike); engine task names appended for
    # fast lookup. sdt_fits has DOMAIN-LOCAL rater_id; the canonical
    # link is via rater_name = raters.csv.canonical_name.
    iiic = pd.read_csv(sdt_iiic_path)
    spk = pd.read_csv(sdt_spike_path)
    sdt = pd.concat([iiic, spk], ignore_index=True)
    # Add engine-task column (inverse of ENGINE_TASK_TO_SDT_DOMAIN)
    sdt_to_engine = {v: k for k, v in ENGINE_TASK_TO_SDT_DOMAIN.items()}
    sdt["engine_task"] = sdt["domain"].map(sdt_to_engine)
    # Drop unconverged rows (sigma / theta NaN) — those raters have no
    # usable fitted-θ; Bernoulli comparator falls back to the engine's
    # prior mean (t=0, ℓ=0) for them; documented honestly in the
    # close-out doc.
    sdt = sdt.dropna(subset=["sigma", "theta"])
    _SHARED["sdt_by_name_task"] = sdt.set_index(
        ["rater_name", "engine_task"])[["sigma", "theta"]].sort_index()
    # rater_id → canonical_name lookup (for sdt_fits join via name)
    _SHARED["name_by_id"] = _SHARED["raters"].set_index(
        "rater_id")["canonical_name"].to_dict()
    return _SHARED


def rater_canonical_name(rater_id: int) -> Optional[str]:
    """canonical_name lookup (used to join sdt_fits via rater_name)."""
    sh = load_replay_bank_cached()
    name = sh["name_by_id"].get(int(rater_id))
    return str(name) if name is not None else None


def fitted_theta_for_rater(rater_id: int,
                            tasks: List[str]) -> Tuple[np.ndarray, Dict[str, bool]]:
    """Return the rater's fitted (t_k, ℓ_k) interleaved across the
    given engine task list — the engine's `true_params` layout:
    [t0, ℓ0, t1, ℓ1, ..., t_{K-1}, ℓ_{K-1}].

    For tasks without a converged SDT fit (no sigma/theta row),
    fall back to (t=0, ℓ=0) — the deployment prior mean. The
    second return value flags which tasks had a fit vs the
    fall-back, for honest reporting.

    Conversion follows `bridge/_common.build_true_params`:
        ℓ = -log(σ),  t = θ.
    """
    sh = load_replay_bank_cached()
    name = rater_canonical_name(rater_id)
    fitted: Dict[str, bool] = {t: False for t in tasks}
    out = np.zeros(2 * len(tasks))
    if name is None:
        return out, fitted
    for ki, t in enumerate(tasks):
        try:
            row = sh["sdt_by_name_task"].loc[(name, t)]
        except KeyError:
            continue
        sigma = float(row["sigma"])
        theta = float(row["theta"])
        if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(theta):
            continue
        out[2 * ki] = theta              # t
        out[2 * ki + 1] = float(-np.log(sigma))  # ℓ
        fitted[t] = True
    return out, fitted


def bank_for_rater_task(rater_id: int, task: str) -> pd.DataFrame:
    """The rater's strict-A bank for one task — engine-input cols
    `seg_id, s_mean, s_sd`. O(log N) lookup via the pre-indexed
    multi-index `bank_by_rt` (~ms; Phase 7 sub-3-C performance fix
    over the original O(5.5M) scan)."""
    sh = load_replay_bank_cached()
    bank_rt = sh["bank_by_rt"]
    key = (int(rater_id), task)
    if key not in bank_rt.index:
        return pd.DataFrame(columns=["seg_id", "s_mean", "s_sd"])
    sub = bank_rt.loc[[key]]   # always-DataFrame (even single row)
    return sub[["seg_id", "s_mean", "s_sd"]].sort_values("seg_id").reset_index(
        drop=True)


def y_lookup_for_rater(rater_id: int, tasks: List[str]) -> Dict[int, Dict[int, int]]:
    """Per-rater Y lookup table: {k → {seg_id → y}}. Per-worker
    cached after first build for the same rater (the y_table is
    static; per-(rater) computation cost is paid once per worker)."""
    sh = load_replay_bank_cached()
    cache = sh["_y_cache"]
    key = (int(rater_id), tuple(tasks))
    if key in cache:
        return cache[key]
    bank_rt = sh["bank_by_rt"]
    lookup: Dict[int, Dict[int, int]] = {ki: {} for ki in range(len(tasks))}
    for ki, t in enumerate(tasks):
        sub_key = (int(rater_id), t)
        if sub_key not in bank_rt.index:
            continue
        sub = bank_rt.loc[[sub_key], ["seg_id", "y"]]
        # vectorised over DataFrame (zip is fast on numpy arrays)
        for seg, y in zip(sub["seg_id"].astype(int).values,
                          sub["y"].astype(int).values):
            lookup[ki][int(seg)] = int(y)
    cache[key] = lookup
    return lookup
