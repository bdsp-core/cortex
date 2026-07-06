"""Step 2.5 — bank adapter: real 89k bank + label file → per-task candidates.

This rebuild is from the documented spec (PROJECT_MEMORY.md D11/D12/D13, §2B) and is
validated against the original's pinned smoke test `test_step2_5.py`
(candidacy counts exact; label reconstruction 98.4% vs the real session;
σ_∞ < σ* invariant). Where the spec under-determined a choice it is flagged
with "RECON:" below.

One validated bridge from the real CSVs to per-task candidate arrays;
reimplements the absent `engine_inputs_k7.as_engine_arrays()` (D13).

  * `segment_signals_general.csv` → per-(seg, task) s_mean / s_sd; a segment is
    a candidate for task k iff s_mean_k is non-NaN (domain1 19,332; others 69,806).
  * `segment_labels_general.csv` → ground-truth y* + confidence margin (D11):
      - multiclass (domain2..7): y* = (plurality == task);
        margin = vote-fraction FOR the asked task if y*=1, else 1 − that
        fraction (confidence in the binary decision for the asked task).
      - domain1: y* = (domain1_pos_frac > DOMAIN1_POS_THRESHOLD);
        RECON: margin = |pos_frac − thr| / (1 − thr), clipped to [0, 1]
        (normalizes the positive side's max distance to 1).
  * `coherent` = label agrees with the signal side (y* == (s_mean > 0));
    feedback trials exclude conflicted + low-margin items (D11).
  * `exclude_segids` removes eval-seen segments (D12).

Constants exposed (memory §2B, from cert_config_general.yaml v14):
ELL_STAR, SIGMA_STAR (= exp(−ℓ*), mastery bar), SIGMA_INF (= exp(−expert_ℓ),
the data-grounded skill ceiling, D10), TASK_CODES.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data")

TASK_CODES = ("domain1", "domain2", "domain3", "domain4",
              "domain5", "domain6", "domain7")

# cert_config_general.yaml :: ell_star_unified_v14 (verified §2B)
ELL_STAR = np.array([0.3251213139414513, 0.2559949138064387,
                     0.5337430687087749, 0.3297002787536504,
                     0.4792595265871899, 0.4864510172880300,
                     0.4418131962513707])
SIGMA_STAR = np.exp(-ELL_STAR)                      # = config sigma_star (✓ test)
_EXPERT_ELL = np.array([0.4100009781611287, 0.6289604310414190,
                        0.7646784869102404, 0.6431387721985599,
                        0.6810426571061013, 0.7403155141054790,
                        0.7138492605529919])
SIGMA_INF = np.exp(-_EXPERT_ELL)                    # skill ceiling (D10/LT1)

DOMAIN1_POS_THRESHOLD = 0.65                        # D11


def _v15_targets():
    from engine.instrument_v15 import instrument
    ins = instrument("v15")
    return ins.ell_star, ins.sigma_star, ins.sigma_inf


# M17 D25 (user-ratified): TRAINER-side mastery targets switch to the v15
# credentialed cuts (every ceiling-to-cut margin widens, e.g. domain1
# 0.085→0.168, domain3 0.231→0.357; σ_∞ < σ* holds everywhere). The v14
# arrays above remain the live EVAL instrument and the conditioning of all
# M10–M16 pinned numbers.
ELL_STAR_V15, SIGMA_STAR_V15, SIGMA_INF_V15 = _v15_targets()

# M18 D29: EMPIRICALLY-ANCHORED dynamics priors (F62 — 1,158 fits on real
# EXTSET reads, 193 users). These are the ASSUMED (filter-side) rates for
# M18+ studies; the old defaults (α_t 0.2, α_σ 0.06) over-estimate the real
# population ×2 / ×1.3 — F17's anti-conservative direction. Deflation
# caveat: between-read gaps were ignored in the fit, so these lean LOW —
# exactly the D14-safe side.
ANCHORED_ALPHA_T = 0.097
ANCHORED_ALPHA_SIGMA = 0.047
# F62/F43 population ceiling spread (upper-bounded by EXTSET σ_ℓ = 0.434).
# M18 hardening OUTCOME (F64): τ=0.45 was tested and NOT adopted as the
# mixture default — the FG gain beyond the mean-skill fix (1/30 → 0/30) did
# not justify doubling declaration lateness; τ=0.30 stays (D22), with 0.45
# recorded as the FG=0 frontier knob for PI consideration.
ANCHORED_CEILING_TAU = 0.45


@dataclass
class TaskCandidates:
    """Parallel per-task candidate arrays in the shape choose_item / the
    trainer policy consume."""
    task: int
    seg_id: np.ndarray      # (n,) int64
    s_mean: np.ndarray      # (n,) float64
    s_sd: np.ndarray        # (n,) float64
    y_star: np.ndarray      # (n,) int64 ∈ {0,1}
    margin: np.ndarray      # (n,) float64 ∈ [0,1] — label confidence
    coherent: np.ndarray    # (n,) bool — label matches signal side

    def __len__(self):
        return int(self.seg_id.shape[0])

    def subset(self, mask) -> "TaskCandidates":
        return TaskCandidates(self.task, self.seg_id[mask], self.s_mean[mask],
                              self.s_sd[mask], self.y_star[mask],
                              self.margin[mask], self.coherent[mask])


class BankAdapter:
    """Loads the two real CSVs once and exposes filtered per-task pools."""

    def __init__(self, root=_HERE):
        sig = pd.read_csv(os.path.join(root, "segment_signals_general.csv"),
                          low_memory=False)
        lab_cols = (["seg_id", "domain1_pos_frac", "n_votes", "plurality",
                     "plurality_frac"]
                    + [f"vote_{c}" for c in TASK_CODES[1:]])
        lab = pd.read_csv(os.path.join(root, "segment_labels_general.csv"),
                          usecols=lab_cols, low_memory=False)
        m = sig.merge(lab, on="seg_id", how="left")
        self._pools = [self._build_pool(m, k) for k in range(len(TASK_CODES))]

    @staticmethod
    def _build_pool(m, k) -> TaskCandidates:
        code = TASK_CODES[k]
        s_mean = m[f"s_mean_{code}"].to_numpy(dtype=np.float64)
        s_sd = m[f"s_sd_{code}"].to_numpy(dtype=np.float64)
        cand = np.isfinite(s_mean) & np.isfinite(s_sd)
        if k == 0:
            pf = m["domain1_pos_frac"].to_numpy(dtype=np.float64)
            labeled = np.isfinite(pf)
            y = (pf > DOMAIN1_POS_THRESHOLD).astype(np.int64)
            margin = np.clip(np.abs(pf - DOMAIN1_POS_THRESHOLD)
                             / (1.0 - DOMAIN1_POS_THRESHOLD), 0.0, 1.0)
        else:
            plur = m["plurality"].to_numpy(dtype=object)
            n_votes = m["n_votes"].to_numpy(dtype=np.float64)
            labeled = pd.notna(m["plurality"]).to_numpy() & (n_votes > 0)
            y = (plur == code).astype(np.int64)
            frac_task = np.where(n_votes > 0,
                                 m[f"vote_{code}"].to_numpy(dtype=np.float64)
                                 / np.where(n_votes > 0, n_votes, 1.0), 0.0)
            margin = np.where(y == 1, frac_task, 1.0 - frac_task)
        keep = cand & labeled
        seg = m["seg_id"].to_numpy(dtype=np.int64)
        coherent = (y == 1) == (s_mean > 0.0)
        return TaskCandidates(k, seg[keep], s_mean[keep], s_sd[keep],
                              y[keep], margin[keep].astype(np.float64),
                              coherent[keep])

    # ── public API ──
    def task_pool(self, task: int) -> TaskCandidates:
        """Full labeled candidate pool for a task (no exclusions)."""
        return self._pools[int(task)]

    def candidates(self, task: int, *, exclude_segids=None,
                   feedback_safe: bool = False,
                   min_margin: float = 0.30) -> TaskCandidates:
        """Filtered pool: drop eval-seen/served segs (D12) and, when
        `feedback_safe`, conflicted + low-margin items (D11)."""
        pool = self._pools[int(task)]
        mask = np.ones(len(pool), dtype=bool)
        if exclude_segids:
            excl = np.fromiter((int(x) for x in exclude_segids), dtype=np.int64)
            mask &= ~np.isin(pool.seg_id, excl)
        if feedback_safe:
            mask &= pool.coherent & (pool.margin >= float(min_margin))
        return pool.subset(mask)

    def conflicted_fraction(self, task: int) -> float:
        """Fraction of the labeled pool whose label conflicts with the signal
        side (the trap/ambiguous stratum D11 keeps out of feedback trials)."""
        pool = self._pools[int(task)]
        return float(1.0 - pool.coherent.mean()) if len(pool) else 0.0
