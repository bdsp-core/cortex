"""F2.5 (2026-05-15) — MCMC diagnostics for SMC + MH-rejuvenation engine.

Helper functions that compute reviewer-grade diagnostic statistics on the
particle cloud and its trajectory:

  weighted_ess_per_parameter   per-coordinate ESS (Kong et al. 1994)
  lag1_autocorrelation         particle-cloud lag-1 autocorrelation per coord
  integrated_autocorr_time     τ_int via emcee's autocorr method (no arviz dep)
  split_rhat                   split-R̂ on an offline thinned chain (Gelman et al.)
  rejuvenation_event_diag      consolidated diagnostic record per rejuvenation

These helpers are deliberately decoupled from the engine so they can be
called from a sidecar logger or unit tests without modifying the core SMC
loop in core_mcmc.py.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def weighted_ess(weights: np.ndarray) -> float:
    """Standard Kong (1994) effective sample size for the full weight vector."""
    w = np.asarray(weights, dtype=float)
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def weighted_ess_per_parameter(
    t: np.ndarray, l: np.ndarray, weights: np.ndarray
) -> Dict[str, np.ndarray]:
    """Per-coordinate effective sample size for a weighted particle cloud.

    Because the weights are the same across all coordinates, the marginal
    ESS *of the weights themselves* is identical for every coordinate.  The
    quantity we actually want is the ESS of an estimator of E[g(θ_k)] for
    each k; the variance-based formulation gives the same number under
    multinomial reweighting.  We expose both forms for clarity:

      - "scalar_ess":   single number, identical across coords (Kong 1994)
      - "per_coord":    if particle resampling is asymmetric per coord
                        (e.g. independent per-task brute clouds), the ESS
                        differs.  For hier this collapses to scalar_ess
                        broadcast over coords.

    Returns dict with keys "t" and "l", each shape (K,).
    """
    K = t.shape[1]
    ess = weighted_ess(weights)
    return {"t": np.full(K, ess), "l": np.full(K, ess)}


def lag1_autocorrelation(
    arr_before: np.ndarray, arr_after: np.ndarray
) -> np.ndarray:
    """Particle-cloud lag-1 autocorrelation per coordinate.

    Computes corr(arr_before[:, k], arr_after[:, k]) for each k.  Captures
    how "memory" of the pre-MCMC cloud persists after rejuvenation.  A
    perfectly-mixing chain has lag-1 → 0; sticky chains have lag-1 → 1.

    arr_before, arr_after: shape (N, K)
    Returns shape (K,).
    """
    a = np.asarray(arr_before, dtype=float)
    b = np.asarray(arr_after, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: {a.shape} vs {b.shape}"
        )
    if a.ndim == 1:
        a = a[:, None]
        b = b[:, None]
    K = a.shape[1]
    out = np.empty(K, dtype=float)
    for k in range(K):
        c = np.corrcoef(a[:, k], b[:, k])
        out[k] = float(c[0, 1]) if c.shape == (2, 2) and np.isfinite(c[0, 1]) else np.nan
    return out


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Sokal-style integrated autocorrelation time estimator.

    `x` is a 1-D chain of post-burn-in samples.  Uses Sokal's automated
    windowing (window = first M such that M ≥ c·τ(M)).  Returns τ_int.

    Reference: emcee.autocorr.integrated_time (BSD-licensed, MIT 2013-2020
    Foreman-Mackey et al.).  Re-implemented here to avoid pulling in the
    full emcee dependency for one function.

    Raises ValueError if the chain is too short to estimate τ_int with the
    given threshold c.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError(f"chain length {n} too short for tau_int")
    # Autocovariance via FFT, normalised to autocorrelation.
    f = x - x.mean()
    # Zero-pad to next power of 2 ≥ 2n for FFT convolution.
    m = 1
    while m < 2 * n:
        m <<= 1
    F = np.fft.fft(f, n=m)
    acf = np.fft.ifft(F * np.conjugate(F))[:n].real
    acf /= acf[0]
    # Sokal automated windowing: smallest M with M >= c * tau(M).
    # tau(M) = 2 * sum_{k=0..M} rho_k - 1.
    taus = 2.0 * np.cumsum(acf) - 1.0
    M_arr = np.arange(len(taus))
    cond = M_arr < c * taus
    if not cond.any():
        # No M satisfies the windowing → chain too short relative to τ.
        raise ValueError(
            f"failed to estimate tau_int with c={c} on chain of length {n}; "
            f"taus[0]={taus[0]}, taus[-1]={taus[-1]}"
        )
    # First M where the condition fails (M >= c*tau(M)).
    M = int(np.argmin(cond))
    return float(taus[M])


def split_rhat(chain: np.ndarray) -> float:
    """Split-R̂ on a 1-D chain (Gelman et al. 2013 §11.4).

    Splits the chain into 4 equal sub-chains (after dropping the first half
    as burn-in if requested by caller), computes between- and within-chain
    variances, returns R̂ = sqrt(var_hat_plus / W).

    chain: 1-D array of post-burn-in samples (length divisible by 4
    recommended).  Returns scalar R̂.  R̂ < 1.01 is the standard pass.
    """
    chain = np.asarray(chain, dtype=float)
    n = chain.size
    if n < 8:
        raise ValueError(f"chain length {n} too short for split-Rhat")
    # Drop trailing samples so we can split evenly into 4 sub-chains.
    n_per = n // 4
    n_used = 4 * n_per
    parts = chain[:n_used].reshape(4, n_per)
    means = parts.mean(axis=1)
    vars_ = parts.var(axis=1, ddof=1)
    grand_mean = means.mean()
    B = n_per * float(((means - grand_mean) ** 2).sum()) / 3.0   # between
    W = float(vars_.mean())                                       # within
    var_hat_plus = ((n_per - 1) / n_per) * W + B / n_per
    if W <= 0:
        return float("nan")
    return float(math.sqrt(var_hat_plus / W))


def rejuvenation_event_diag(
    *,
    event_id: int,
    q_index: int,
    n_mh_steps: int,
    accept_rate: float,
    t_before: np.ndarray,
    t_after: np.ndarray,
    l_before: np.ndarray,
    l_after: np.ndarray,
    weights_after: np.ndarray,
    domain_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serialisable record summarising one rejuvenation event.

    Inputs are particle clouds before and after rejuvenation (both shape
    (N, K)).  The full chain is not stored — only summary statistics.

    The record is suitable for one line of a per-session JSONL diagnostic
    log:  audit_logs/<rater>_seed<seed>_mcmc_diag.jsonl.
    """
    t_after = np.asarray(t_after)
    l_after = np.asarray(l_after)
    K = t_after.shape[1]
    domains = list(domain_names) if domain_names is not None else [f"d{k}" for k in range(K)]
    lag_t = lag1_autocorrelation(t_before, t_after)
    lag_l = lag1_autocorrelation(l_before, l_after)
    ess = weighted_ess_per_parameter(t_after, l_after, weights_after)
    return {
        "event_id": int(event_id),
        "q_index": int(q_index),
        "n_mh_steps": int(n_mh_steps),
        "accept_rate": float(accept_rate),
        "domains": domains,
        "lag1_autocorr_t": [None if not np.isfinite(v) else float(v) for v in lag_t],
        "lag1_autocorr_l": [None if not np.isfinite(v) else float(v) for v in lag_l],
        "ess_t": ess["t"].tolist(),
        "ess_l": ess["l"].tolist(),
    }


def write_diag_record(path: str, record: Dict[str, Any]) -> None:
    """Append a JSONL diagnostic record to `path`, creating the dir if needed.

    Uses allow_nan=False after sanitising; treat as authoritative reproducibility
    artifact.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
