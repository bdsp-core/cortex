"""VERBATIM extract from
`ilae-skill-certification-test-main/src/train_val_split_and_fit.py`.

`youden_sigma_star` is copied BYTE-FOR-BYTE from lines 211-225 of the
reference (function body unchanged). The three constants below are copied
verbatim from the reference module header (LAMBDA=0.025 line 55,
SEED=42 line 56, EXPERT_TRAIN_FRAC=0.70 line 60). Nothing else from the
H5-bound reference script is imported or executed.

DO NOT EDIT the function body or the constant values.
"""
from __future__ import annotations

import numpy as np

# --- verbatim constants from train_val_split_and_fit.py ---
LAMBDA = 0.025
SEED = 42
EXPERT_TRAIN_FRAC = 0.70   # 70% train, 30% val for expert pool (wider calibration base)


def youden_sigma_star(sigma_expert: np.ndarray, sigma_non: np.ndarray) -> tuple[float, float]:
    """Find σ* = argmax [F_expert(σ) − F_nonexpert(σ)] along the σ grid
    spanned by the pooled sample. Returns (σ*, Youden J)."""
    grid = np.sort(np.concatenate([sigma_expert, sigma_non]))
    grid = grid[np.isfinite(grid)]
    grid = np.unique(grid)
    best_j, best_sigma = -np.inf, np.nan
    for s in grid:
        f_e = np.mean(sigma_expert <= s) if len(sigma_expert) else 0.0
        f_n = np.mean(sigma_non <= s) if len(sigma_non) else 0.0
        j = f_e - f_n
        if j > best_j:
            best_j = j
            best_sigma = s
    return best_sigma, best_j
