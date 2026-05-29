"""K=6 v13 baseline fetch — for Layer-1 variant comparison gates.

The K=6 joint hierarchical fit (Phase 3.5, current v13) produced per-task
outputs at:

  * `calibration/joint/s_j_table.csv` — per-(task, seg_id) s_mean, s_sd
                                        (the un-calibrated SVI posterior)
  * `calibration/joint/s_j_table_calibrated.csv` — same + s_sd_calibrated
                                        (κ-inflated per Phase-3.5 variance
                                        calibration vs NUTS subsample)
  * `calibration/joint/joint_per_rater_params.csv` — per-(domain, rater_id)
                                        sigma, theta (from joint posterior;
                                        retained for provenance even though
                                        v13 cert_config uses two-stage fits)
  * `calibration/joint/{task}_nuts_summary.json` — NUTS validation R̂ +
                                        diagnostics (per IIIC task)

This helper exposes the K=6 baseline as a structured object the K=7
variant-selection gate consumes for s_j drift comparison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "calibration" / "joint"

# K=6 IIIC tasks (NOT including spike — spike was fit separately in
# Phase 3 via byte-verbatim two-stage and is NOT in the K=6 joint output)
K6_IIIC_TASKS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]


@dataclass
class K6Baseline:
    """K=6 v13 baseline outputs for ONE IIIC task. Used as the reference
    against which K=7 variants are compared at the Layer-1 gate."""

    task: str
    seg_ids: np.ndarray          # (S,) integer seg_ids
    s_mean: np.ndarray           # (S,) s_j posterior mean (uncalibrated)
    s_sd: np.ndarray             # (S,) s_j posterior sd (uncalibrated)
    s_sd_calibrated: np.ndarray  # (S,) κ-inflated sd (matches v13 ship state)
    nuts_rhat_s_j: float | None  # NUTS R̂ for s_j (None if missing)
    nuts_rhat_ell: float | None
    nuts_rhat_t: float | None
    n_seg: int
    # Per-rater (domain-filtered) fits from the joint posterior side-car.
    # Note: v13 cert_config consumes the byte-verbatim TWO-STAGE per-rater
    # params, NOT these. They are retained here only because reviewers may
    # want a per-rater drift comparison alongside the s_j drift.
    rater_ids: np.ndarray        # (R,) integer rater_ids
    rater_sigma: np.ndarray      # (R,) per-rater sigma from joint
    rater_theta: np.ndarray      # (R,) per-rater theta from joint
    rater_converged: np.ndarray  # (R,) bool


def load_k6_baseline(task: str) -> K6Baseline:
    """Load the K=6 v13 baseline for one IIIC task. Raises FileNotFoundError
    if expected K=6 artifacts are absent (the Layer-1 gate cannot be applied
    without them)."""
    if task not in K6_IIIC_TASKS:
        raise ValueError(
            f"task {task!r} not in K=6 IIIC set {K6_IIIC_TASKS}; "
            "spike has no K=6 baseline (Phase-3.5 fit was two-stage, not joint).")

    # ── s_j table (un-calibrated + calibrated) ────────────────────────────
    raw_path = JOINT / "s_j_table.csv"
    cal_path = JOINT / "s_j_table_calibrated.csv"
    if not raw_path.exists() or not cal_path.exists():
        raise FileNotFoundError(
            f"K=6 baseline s_j tables missing: {raw_path}, {cal_path}. "
            "Phase-3.5 joint outputs are required for the Layer-1 gate.")

    raw = pd.read_csv(raw_path)
    cal = pd.read_csv(cal_path)
    raw_t = raw[raw["task"] == task].sort_values("seg_id").reset_index(drop=True)
    cal_t = cal[cal["task"] == task].sort_values("seg_id").reset_index(drop=True)
    if len(raw_t) == 0 or len(cal_t) == 0:
        raise RuntimeError(
            f"K=6 baseline has no rows for task {task!r}")
    if not (raw_t["seg_id"].values == cal_t["seg_id"].values).all():
        raise RuntimeError(
            f"K=6 raw vs calibrated s_j tables disagree on seg_ids for task {task!r}")

    # ── per-rater joint params side-car ───────────────────────────────────
    rater_path = JOINT / "joint_per_rater_params.csv"
    if not rater_path.exists():
        raise FileNotFoundError(
            f"K=6 baseline per-rater params missing: {rater_path}")
    rdf = pd.read_csv(rater_path)
    rdf_t = rdf[rdf["domain"] == task].sort_values("rater_id").reset_index(drop=True)

    # ── NUTS validation summary (R̂; per IIIC task) ──────────────────────
    nuts_path = JOINT / f"{task}_nuts_summary.json"
    nuts_rhat_s_j = nuts_rhat_ell = nuts_rhat_t = None
    if nuts_path.exists():
        try:
            ns = json.loads(nuts_path.read_text())
            conv = ns.get("convergence", {})
            if conv.get("method") == "nuts":
                rh = conv.get("rhat_max", {})
                if isinstance(rh, dict):
                    nuts_rhat_s_j = rh.get("s_j")
                    nuts_rhat_ell = rh.get("ell")
                    nuts_rhat_t = rh.get("t")
        except (json.JSONDecodeError, KeyError):
            pass

    return K6Baseline(
        task=task,
        seg_ids=raw_t["seg_id"].astype(int).to_numpy(),
        s_mean=raw_t["s_mean"].astype(float).to_numpy(),
        s_sd=raw_t["s_sd"].astype(float).to_numpy(),
        s_sd_calibrated=cal_t["s_sd_calibrated"].astype(float).to_numpy(),
        nuts_rhat_s_j=nuts_rhat_s_j,
        nuts_rhat_ell=nuts_rhat_ell,
        nuts_rhat_t=nuts_rhat_t,
        n_seg=len(raw_t),
        rater_ids=rdf_t["rater_id"].astype(int).to_numpy(),
        rater_sigma=rdf_t["sigma"].astype(float).to_numpy(),
        rater_theta=rdf_t["theta"].astype(float).to_numpy(),
        rater_converged=rdf_t["converged"].astype(bool).to_numpy(),
    )


def s_j_drift_against_k6(k7_seg_ids: np.ndarray, k7_s_mean: np.ndarray,
                         k6: K6Baseline) -> dict:
    """Inner-join K=7 SVI s_id_mean vs K=6 baseline s_mean on seg_id; compute
    drift statistics. Returns dict with `n_overlap`, `max_abs_drift`,
    `mean_abs_drift`, `p95_abs_drift`.
    """
    k7_idx = {int(s): i for i, s in enumerate(k7_seg_ids)}
    k6_idx = {int(s): i for i, s in enumerate(k6.seg_ids)}
    common = sorted(set(k7_idx) & set(k6_idx))
    if not common:
        return {"n_overlap": 0, "max_abs_drift": float("nan"),
                "mean_abs_drift": float("nan"), "p95_abs_drift": float("nan")}
    d = np.array([k7_s_mean[k7_idx[s]] - k6.s_mean[k6_idx[s]]
                  for s in common])
    abs_d = np.abs(d)
    return {
        "n_overlap": len(common),
        "max_abs_drift": float(abs_d.max()),
        "mean_abs_drift": float(abs_d.mean()),
        "p95_abs_drift": float(np.percentile(abs_d, 95)),
    }
