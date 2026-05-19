"""Hierarchical multi-task 2PL probit fit.

Same probit-2PL likelihood as fit_2pl_probit.py:

    P(Y_ijk = 1) = Phi( exp(l_ik) * (s_jk + t_ik) )

But the per-rater parameter vector

    theta_i = (t_i1, l_i1, t_i2, l_i2, ..., t_iK, l_iK)  in  R^{2K}

is given a SHARED multivariate-normal prior across raters:

    theta_i ~ N(0, Sigma).

We fit Sigma by empirical Bayes, in two variants:

  --variant free    (Option A) unstructured Sigma, ridge-regularized for
                    stability. After fitting, also reports the factor
                    parameters (r, rho) implied by the empirical Sigma.

  --variant factor  (Option B) Sigma constrained to the factor structure
                    from the simulation slides:
                       Var(t_k) = Var(l_k) = 1
                       Cov(t_k, t_{k'}) = r   (k != k')
                       Cov(l_k, l_{k'}) = r   (k != k')
                       Cov(t_k, l_k)    = rho
                       Cov(t_k, l_{k'}) = rho * r  (k != k')
                    Only r in [0, 0.99] and rho in (-0.99, 0.99) are estimated,
                    by moment-matching to the empirical Sigma each iteration.

Tasks fit (Phase 4.4-B): spike + 6 IIIC tasks — the 5 subtypes
(seizure, lpd, gpd, lrda, grda) + the REAL "other" (= Phase-3/v13
sparcnet_iic; erratum-correct value in {other,bipd,birds}). The
PI-era *composite* 'iic' (a deterministic OR over the 5 subtypes,
which would make Sigma block-singular) is NOT this — that degenerate
OR was DELETED; "other" is an INDEPENDENT IIIC class so the
block-singularity rationale does not apply and it is a legitimate 7th
hierarchical task (IIIC_IDXS extends to 1..K-1; K=7, Sigma 14x14).

Inference: alternating block-coordinate MAP.

  Per outer iteration:
   1. For each task k, IRLS update of s_{j,k} given current theta.
   2. For each rater i, for each task k they participated in, IRLS update
      of (t_{i,k}, l_{i,k}) with the Sigma-conditional prior on slot j
      given the rater's other observed slots (one-slot-at-a-time
      Gauss-Seidel; uses precision-matrix conditional formula).
   3. Center t per task (shift-identifiability), absorb into s.
   4. Update Sigma:
        - variant=free:   pairwise-observed empirical covariance, ridged.
        - variant=factor: empirical Sigma -> moment-match (r, rho) ->
                           reconstruct Sigma with the factor structure.

Outputs to data/labels/fits_hier_<variant>/:
  <task>/cases.csv          seg_id, s_mean, n_raters, pos_rate
  <task>/raters.csv         rater_id, canonical_name, groups, expertise_level,
                            t_mean, t_sd, ell_mean, ell_sd, n_labels, pos_rate
  <task>/summary.json
  Sigma.csv                 14x14 (rows/cols: t_spike, l_spike, ..., t_grda, l_grda
                            with K=6 here, so 12x12)
  correlation.csv           same as Sigma but normalized
  factor_summary.json       r_hat, rho_hat (computed post-hoc for either variant)
  fit_summary.json          all of the above plus per-task n_labels/n_cases/n_raters

UNIFIED-MERGE PROVENANCE (Phase 4.4-B, 2026-05-19). Faithful port of the
PI scripts/fit_2pl_probit_hier.py. Documented, auditable changes vs PI:
  - hardcoded absolute-ROOT -> repo-root self-location
    (Path(__file__).resolve().parents[1]); derived paths byte-identical.
  - TASKS 6 -> 7: append the real "other".
  - extract_task_labels: + erratum-correct "other" branch
    (value in {other,bipd,birds}); spike/else logic unchanged.
  - IIIC_IDXS: was hardcoded (1,2,3,4,5); now tuple(range(1,K)) — the
    documented "index 0=spike, 1..K-1=IIIC" structure, K-correct for
    K=6 (1..5) AND K=7 (1..6). block_sigma/moment_match_block consume
    IIIC_IDXS generically, so they handle K=7 with no further change.
UN-EXERCISED until Phase 4.6 (no K=7 fit run per the locked 4.4 scope).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]   # repo root (was hardcoded)
LABELS_CSV = ROOT / "data/labels/labels.csv"
RATERS_CSV = ROOT / "data/labels/raters.csv"
SINGLE_FIT_DIR = ROOT / "data/labels/fits"

TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]
K = len(TASKS)
DIM = 2 * K   # parameter ordering: t_0, l_0, t_1, l_1, ...

DEFAULTS = {
    "min_raters_per_case": 3,
    "min_labels_per_rater": 100,
    "min_votes_per_class": 5,
    "max_outer": 25,
    "tol": 1e-4,
    "max_inner_l": 5,
    "sigma_s": 2.0,
    "ridge": 0.05,
    "S_MAX": 5.0,
    "T_MAX": 3.0,
    "L_MAX": 3.0,
}


# ───────────────────── data extraction (same as single-task) ─────────────────

def extract_task_labels(labels, task):
    if task == "spike":
        sub = labels[labels.label_type == "spike"].copy()
        sub["Y"] = pd.to_numeric(sub["value"], errors="coerce")
    else:
        sub = labels[labels.label_type == "pattern_class"].copy()
        if task == "other":
            # Phase 4.4-B: real "other" IIIC task — ERRATUM-CORRECT
            # uniform {bipd,birds}->other (AUDIT §6 / Phase-1); MUST
            # match Phase-3/v13 sparcnet_iic (unified count 204,163).
            sub["Y"] = (sub["value"].isin(
                ["other", "bipd", "birds"])).astype(int)
        else:
            sub["Y"] = (sub["value"] == task).astype(int)
    return sub[["seg_id", "rater_id", "Y"]].dropna()


def filter_iteratively(df, min_r, min_l, min_votes_per_class):
    prev_n = -1
    n = len(df)
    while n != prev_n:
        prev_n = n
        rc_total = df.groupby("rater_id").size()
        rc_pos = df.groupby("rater_id")["Y"].sum()
        rc_neg = rc_total - rc_pos
        ok_r = rc_total[(rc_total >= min_l)
                         & (rc_pos >= min_votes_per_class)
                         & (rc_neg >= min_votes_per_class)].index
        df = df[df.rater_id.isin(ok_r)]
        cc = df.groupby("seg_id").size()
        ok_c = cc[cc >= min_r].index
        df = df[df.seg_id.isin(ok_c)]
        n = len(df)
    return df


# ───────────────────── probit IRLS helpers ─────────────────────

def irls_z_w(eta, Y):
    eta_c = np.clip(eta, -6.0, 6.0)
    phi = norm.pdf(eta_c)
    Phi = norm.cdf(eta_c)
    eps = 1e-10
    Phi_c = np.clip(Phi, eps, 1 - eps)
    phi_c = np.clip(phi, eps, None)
    w = phi_c ** 2 / (Phi_c * (1 - Phi_c))
    r = (Y - Phi_c) / phi_c
    z = eta_c + r
    return z, w


# ───────────────────── conditional Gaussian helpers ─────────────────────

def cond_prior_for_slot(theta_rater, slots_obs, Sigma, slot):
    """Conditional (mean, var) of theta[slot] given theta[other slots in slots_obs].

    Uses the precision-matrix conditional formula on the marginal of Sigma
    restricted to slots_obs.
    """
    Sigma_obs = Sigma[np.ix_(slots_obs, slots_obs)]
    Lambda = np.linalg.inv(Sigma_obs)
    j = slots_obs.index(slot)
    diag = Lambda[j, j]
    others_idx = [i for i in range(len(slots_obs)) if i != j]
    others_slots = [slots_obs[i] for i in others_idx]
    theta_others = theta_rater[others_slots]
    cond_mean = -(Lambda[j, others_idx] @ theta_others) / diag
    cond_var = 1.0 / diag
    return float(cond_mean), float(cond_var)


# ───────────────────── per-task s update ─────────────────────

def update_s_task(s, theta_t, theta_l, rater_idx, case_idx, Y,
                   sigma_s, S_MAX):
    """IRLS update of s for this task, holding (t, l) fixed.

    Returns (s_new, s_sd) where s_sd is the Laplace posterior SD per case
    (1 / sqrt of observed information).
    """
    eta = np.exp(theta_l[rater_idx]) * (s[case_idx] + theta_t[rater_idx])
    z, w = irls_z_w(eta, Y)
    el = np.exp(theta_l[rater_idx])
    target = z / el - theta_t[rater_idx]
    weight = w * el ** 2
    n_case = len(s)
    num = np.bincount(case_idx, weights=weight * target, minlength=n_case)
    den = np.bincount(case_idx, weights=weight, minlength=n_case) + 1.0 / sigma_s ** 2
    s_new = num / den
    np.clip(s_new, -S_MAX, S_MAX, out=s_new)
    s_sd = 1.0 / np.sqrt(den)
    return s_new, s_sd


# ───────────────────── per-rater theta update with Σ-conditional prior ──────

def update_theta_block(theta, s_arrays, mask, rt_index, Sigma, opts):
    """Update each rater's (t_{ik}, l_{ik}) for each task k they're in.

    rt_index: {(ri, ki): (case_local_indices, Y)} — precomputed per-rater per-task.
    Returns (t_sd_mat, l_sd_mat): (n_rat, K) arrays.
    """
    n_rat = theta.shape[0]
    t_sd_mat = np.full((n_rat, K), np.nan)
    l_sd_mat = np.full((n_rat, K), np.nan)

    # Precompute observed slot list per rater
    rater_slots = [[] for _ in range(n_rat)]
    for ri in range(n_rat):
        for ki in range(K):
            if mask[ri, ki]:
                rater_slots[ri].append(2 * ki)
                rater_slots[ri].append(2 * ki + 1)

    for ri in range(n_rat):
        slots = rater_slots[ri]
        if not slots:
            continue
        for ki in range(K):
            if not mask[ri, ki]:
                continue
            key = (ri, ki)
            if key not in rt_index:
                continue
            cl, Y = rt_index[key]
            if len(cl) == 0:
                continue

            t_slot = 2 * ki
            l_slot = 2 * ki + 1
            s_local = s_arrays[TASKS[ki]][cl]

            # ── update t_{i,k} ────────────────────────
            t_i = theta[ri, t_slot]; l_i = theta[ri, l_slot]
            eta = np.exp(l_i) * (s_local + t_i)
            z, w = irls_z_w(eta, Y)
            el = np.exp(l_i)
            target = z / el - s_local  # target ≈ t_i
            weight = w * el ** 2
            cm, cv = cond_prior_for_slot(theta[ri], slots, Sigma, t_slot)
            num = (weight * target).sum() + cm / cv
            den = weight.sum() + 1.0 / cv
            t_new = float(np.clip(num / den, -opts["T_MAX"], opts["T_MAX"]))
            theta[ri, t_slot] = t_new
            t_sd_mat[ri, ki] = 1.0 / np.sqrt(den)

            # ── update l_{i,k} via 1-D Newton on log-skill ───
            u = s_local + t_new
            l_curr = theta[ri, l_slot]
            for _ in range(opts["max_inner_l"]):
                el = np.exp(l_curr)
                eta = el * u
                z, w = irls_z_w(eta, Y)
                resid = z - eta
                score = (w * eta * resid).sum()
                info = (w * eta ** 2).sum()
                cm, cv = cond_prior_for_slot(theta[ri], slots, Sigma, l_slot)
                score = score - (l_curr - cm) / cv
                info = info + 1.0 / cv
                step = float(np.clip(score / info, -0.5, 0.5))
                l_curr = l_curr + step
            l_new = float(np.clip(l_curr, -opts["L_MAX"], opts["L_MAX"]))
            theta[ri, l_slot] = l_new
            l_sd_mat[ri, ki] = 1.0 / np.sqrt(info)
    return t_sd_mat, l_sd_mat


# ───────────────────── Σ updates ─────────────────────

def empirical_sigma(theta, mask, ridge):
    """Pairwise-observed empirical covariance with marginal variances forced
    to 1.0. Returns a positive-definite, ridge-regularized matrix.
    """
    DIM_ = theta.shape[1]
    K_ = DIM_ // 2
    slot_obs = np.zeros((theta.shape[0], DIM_), dtype=bool)
    for k in range(K_):
        slot_obs[:, 2 * k]     = mask[:, k]
        slot_obs[:, 2 * k + 1] = mask[:, k]

    Sigma_new = np.eye(DIM_)
    for j in range(DIM_):
        for l in range(j + 1, DIM_):
            both = slot_obs[:, j] & slot_obs[:, l]
            if both.sum() < 3:
                continue
            cov = float((theta[both, j] * theta[both, l]).mean())
            cov = float(np.clip(cov, -0.99, 0.99))
            Sigma_new[j, l] = Sigma_new[l, j] = cov

    Sigma_reg = (1.0 - ridge) * Sigma_new + ridge * np.eye(DIM_)
    try:
        np.linalg.cholesky(Sigma_reg)
    except np.linalg.LinAlgError:
        Sigma_reg = 0.5 * Sigma_reg + 0.5 * np.eye(DIM_)
    return Sigma_reg


def factor_sigma(r, rho, K_):
    """Build (2*K_)x(2*K_) Sigma with the factor structure."""
    DIM_ = 2 * K_
    S = np.zeros((DIM_, DIM_))
    for k in range(K_):
        for kp in range(K_):
            tt = 1.0 if k == kp else r
            ll = 1.0 if k == kp else r
            tl = rho * (1.0 if k == kp else r)
            S[2*k,     2*kp]     = tt
            S[2*k + 1, 2*kp + 1] = ll
            S[2*k,     2*kp + 1] = tl
            S[2*k + 1, 2*kp]     = tl
    return S


def moment_match_factor(Sigma_emp):
    """Estimate (r, rho) from an empirical Σ by averaging the relevant entries."""
    K_ = Sigma_emp.shape[0] // 2
    rs, rhos = [], []
    for k in range(K_):
        rhos.append(Sigma_emp[2 * k, 2 * k + 1])
        for kp in range(K_):
            if k == kp:
                continue
            rs.append(Sigma_emp[2 * k,     2 * kp])         # tt
            rs.append(Sigma_emp[2 * k + 1, 2 * kp + 1])     # ll
    return float(np.mean(rs)), float(np.mean(rhos))


# ───────────────────── block-Σ (deployment-recommended) ─────────────────────
#
# TASKS layout: index 0 = spike, indices 1..5 = IIIC subtypes
# (seizure, lpd, gpd, lrda, grda). The block-Σ structure is:
#   - Var(t_k) = Var(ℓ_k) = 1   (scale convention)
#   - Cov(t_k, t_k') = 0        (biases independent across tasks)
#   - Cov(t_k, ℓ_k') = 0        (rho = 0 — fixed by recommendation)
#   - Cov(ℓ_k, ℓ_k') = r_IIIC   for k, k' both in IIIC
#   - Cov(ℓ_spike, ℓ_IIIC_k) = r_cross

SPIKE_IDX = 0
# Phase 4.4-B: was hardcoded (1,2,3,4,5). The documented model layout is
# "index 0 = spike, indices 1..K-1 = IIIC group", so derive it K-correctly
# (K=6 -> 1..5 ; K=7 -> 1..6 with the real "other"). block_sigma /
# moment_match_block consume IIIC_IDXS generically ⇒ K=7 needs nothing
# else changed.
IIIC_IDXS = tuple(range(1, K))


def block_sigma(r_iiic, r_cross, K_):
    """Build (2K_) × (2K_) block-Σ with the recommended deployment structure."""
    DIM_ = 2 * K_
    S = np.eye(DIM_)
    for k in range(K_):
        for kp in range(K_):
            if k == kp:
                continue
            l_k  = 2 * k + 1
            l_kp = 2 * kp + 1
            in_iiic_k  = (k  in IIIC_IDXS)
            in_iiic_kp = (kp in IIIC_IDXS)
            if in_iiic_k and in_iiic_kp:
                S[l_k, l_kp] = r_iiic
            else:
                S[l_k, l_kp] = r_cross
            # t-t, t-l, l-t off-diagonals all stay 0 by construction
    return S


def moment_match_block(Sigma_emp):
    """Estimate (r_IIIC, r_cross) from empirical Σ by averaging the ℓ-ℓ entries.

    Off-block bias entries and within-task t-ℓ are not used.
    """
    r_iiic_vals, r_cross_vals = [], []
    K_ = Sigma_emp.shape[0] // 2
    for k in range(K_):
        for kp in range(K_):
            if k == kp:
                continue
            v = Sigma_emp[2 * k + 1, 2 * kp + 1]  # ℓ_k vs ℓ_k'
            in_k  = (k  in IIIC_IDXS)
            in_kp = (kp in IIIC_IDXS)
            if in_k and in_kp:
                r_iiic_vals.append(v)
            elif (k == SPIKE_IDX and in_kp) or (kp == SPIKE_IDX and in_k):
                r_cross_vals.append(v)
    return (float(np.mean(r_iiic_vals)) if r_iiic_vals else 0.0,
            float(np.mean(r_cross_vals)) if r_cross_vals else 0.0)


# ───────────────────── main fit loop ─────────────────────

def fit_hier(variant, opts, verbose=True):
    print("Reading labels...", flush=True)
    labels = pd.read_csv(LABELS_CSV, dtype={"value": "str"})
    raters_meta = pd.read_csv(RATERS_CSV)
    print(f"  labels: {len(labels):,} rows; raters: {len(raters_meta):,}", flush=True)

    # Filter and gather per task
    print("Extracting + filtering per task:")
    task_dfs = {}
    for task in TASKS:
        df = extract_task_labels(labels, task)
        df = filter_iteratively(df, opts["min_raters_per_case"],
                                 opts["min_labels_per_rater"],
                                 opts["min_votes_per_class"])
        task_dfs[task] = df
        print(f"  {task:>8}: {len(df):,} labels, {df.seg_id.nunique()} cases, "
              f"{df.rater_id.nunique()} raters", flush=True)

    # Global rater list (union)
    all_raters = set()
    for task in TASKS:
        all_raters |= set(task_dfs[task].rater_id.unique())
    rater_list = sorted(all_raters)
    rater_id_to_global = {r: i for i, r in enumerate(rater_list)}
    n_rat = len(rater_list)
    print(f"\nGlobal rater count (union across tasks): {n_rat}", flush=True)

    # Per-task structures + per-rater-per-task label indices
    task_data = {}
    case_local_to_global = {}
    rt_index = {}      # (global_rater_idx, task_idx) -> (cl, Y)
    for ki, task in enumerate(TASKS):
        df = task_dfs[task].copy()
        case_globals = sorted(df.seg_id.unique())
        case_g2l = {g: i for i, g in enumerate(case_globals)}
        df["case_local"] = df.seg_id.map(case_g2l)
        df["rater_global_idx"] = df.rater_id.map(rater_id_to_global)
        task_data[task] = df
        case_local_to_global[task] = case_globals
        # Build rt_index
        for r_gi, sub in df.groupby("rater_global_idx"):
            rt_index[(int(r_gi), ki)] = (sub["case_local"].to_numpy(),
                                          sub["Y"].to_numpy(dtype=float))

    # mask[ri, ki]: True if rater ri labeled task ki
    mask = np.zeros((n_rat, K), dtype=bool)
    for ki, task in enumerate(TASKS):
        for r_g in task_dfs[task].rater_id.unique():
            mask[rater_id_to_global[r_g], ki] = True

    # Initialize theta and s from single-task fits
    theta = np.zeros((n_rat, DIM))
    s_arrays = {}
    for ki, task in enumerate(TASKS):
        single_raters = pd.read_csv(SINGLE_FIT_DIR / task / "raters.csv")
        for _, row in single_raters.iterrows():
            ri = rater_id_to_global.get(int(row["rater_id"]))
            if ri is None: continue
            theta[ri, 2 * ki]     = row["t_mean"]
            theta[ri, 2 * ki + 1] = row["ell_mean"]
        single_cases = pd.read_csv(SINGLE_FIT_DIR / task / "cases.csv")
        seg_to_smean = dict(zip(single_cases.seg_id, single_cases.s_mean))
        s_local = np.zeros(len(case_local_to_global[task]))
        for li, seg_g in enumerate(case_local_to_global[task]):
            s_local[li] = seg_to_smean.get(seg_g, 0.0)
        s_arrays[task] = s_local

    # Σ initialization
    Sigma = np.eye(DIM)

    # Outer loop
    t_start = time.perf_counter()
    print(f"\nFitting (variant={variant})... ", flush=True)
    t_sd_mat = np.full((n_rat, K), np.nan)
    l_sd_mat = np.full((n_rat, K), np.nan)
    s_sd_arrays = {task: np.full(len(case_local_to_global[task]), np.nan)
                    for task in TASKS}
    for it in range(opts["max_outer"]):
        # 1) update each task's s
        for ki, task in enumerate(TASKS):
            df = task_data[task]
            rl = df["rater_global_idx"].to_numpy()
            cl = df["case_local"].to_numpy()
            Y = df["Y"].to_numpy(dtype=float)
            theta_t = theta[:, 2 * ki]
            theta_l = theta[:, 2 * ki + 1]
            s_arrays[task], s_sd_arrays[task] = update_s_task(
                s_arrays[task], theta_t, theta_l, rl, cl, Y,
                opts["sigma_s"], opts["S_MAX"])

        # 2) update theta per (rater, task), with Σ-conditional prior
        t_sd_mat, l_sd_mat = update_theta_block(
            theta, s_arrays, mask, rt_index, Sigma, opts)

        # 3) center t per task; absorb into s
        for ki in range(K):
            obs = mask[:, ki]
            if obs.sum() == 0: continue
            t_mean = theta[obs, 2 * ki].mean()
            theta[obs, 2 * ki] -= t_mean
            s_arrays[TASKS[ki]] += t_mean
            np.clip(s_arrays[TASKS[ki]], -opts["S_MAX"], opts["S_MAX"],
                    out=s_arrays[TASKS[ki]])

        # 4) update Σ
        Sigma_emp = empirical_sigma(theta, mask, opts["ridge"])
        if variant == "free":
            Sigma = Sigma_emp
        elif variant == "factor":
            r_hat, rho_hat = moment_match_factor(Sigma_emp)
            r_hat = float(np.clip(r_hat, 0.0, 0.99))
            rho_hat = float(np.clip(rho_hat, -0.99, 0.99))
            Sigma_f = factor_sigma(r_hat, rho_hat, K)
            Sigma = (1.0 - opts["ridge"]) * Sigma_f + opts["ridge"] * np.eye(DIM)
        elif variant == "block":
            r_iiic_hat, r_cross_hat = moment_match_block(Sigma_emp)
            r_iiic_hat  = float(np.clip(r_iiic_hat,  -0.95, 0.95))
            r_cross_hat = float(np.clip(r_cross_hat, -0.95, 0.95))
            Sigma_b = block_sigma(r_iiic_hat, r_cross_hat, K)
            Sigma = (1.0 - opts["ridge"]) * Sigma_b + opts["ridge"] * np.eye(DIM)
        else:
            raise ValueError(variant)

        if verbose and (it % 2 == 0 or it < 3):
            if variant == "block":
                r_i, r_c = moment_match_block(Sigma)
                print(f"  it={it:3d}  block r_IIIC≈{r_i:+.3f}  r_cross≈{r_c:+.3f}  "
                      f"({(time.perf_counter()-t_start):.0f}s)", flush=True)
            else:
                r_h, rho_h = moment_match_factor(Sigma)
                print(f"  it={it:3d}  factor r≈{r_h:+.3f}  rho≈{rho_h:+.3f}  "
                      f"({(time.perf_counter()-t_start):.0f}s)", flush=True)

    return (theta, Sigma, s_arrays, s_sd_arrays, mask, task_data,
            t_sd_mat, l_sd_mat, rater_list, case_local_to_global,
            raters_meta, time.perf_counter() - t_start)


# ───────────────────── output writers ─────────────────────

def write_outputs(out_dir, theta, Sigma, s_arrays, s_sd_arrays, mask, task_data,
                   t_sd_mat, l_sd_mat, rater_list, case_local_to_global,
                   raters_meta, opts, variant, wall_time):
    out_dir.mkdir(parents=True, exist_ok=True)
    rater_id_to_global = {r: i for i, r in enumerate(rater_list)}

    summaries = []
    for ki, task in enumerate(TASKS):
        df = task_data[task]
        cl = df["case_local"].to_numpy()
        rl = df["rater_global_idx"].to_numpy()
        Y = df["Y"].to_numpy(dtype=float)
        case_globals = case_local_to_global[task]
        n_local_cases = len(case_globals)
        n_raters_per_case = np.bincount(cl, minlength=n_local_cases)
        pos_per_case = np.bincount(cl, weights=Y, minlength=n_local_cases)
        cases_df = pd.DataFrame({
            "seg_id": case_globals,
            "s_mean": s_arrays[task],
            "s_sd": s_sd_arrays[task],
            "n_raters": n_raters_per_case,
            "pos_rate": pos_per_case / np.maximum(n_raters_per_case, 1),
        })

        # Raters: pull global ids, attach metadata
        rater_globals = sorted(df.rater_id.unique())
        rows = []
        for r_g in rater_globals:
            ri = rater_id_to_global[r_g]
            meta = raters_meta[raters_meta.rater_id == r_g]
            sub_mask = (rl == ri)
            rows.append({
                "rater_id": r_g,
                "canonical_name": meta["canonical_name"].iloc[0] if len(meta) else f"rater_{r_g}",
                "groups":         meta["groups"].iloc[0] if len(meta) else "",
                "expertise_level": meta["expertise_level"].iloc[0] if len(meta) else "",
                "t_mean": theta[ri, 2 * ki],
                "t_sd":   t_sd_mat[ri, ki],
                "ell_mean": theta[ri, 2 * ki + 1],
                "ell_sd":   l_sd_mat[ri, ki],
                "n_labels": int(sub_mask.sum()),
                "pos_rate": float(Y[sub_mask].mean()) if sub_mask.any() else float("nan"),
            })
        raters_df = pd.DataFrame(rows)

        (out_dir / task).mkdir(parents=True, exist_ok=True)
        cases_df.to_csv(out_dir / task / "cases.csv", index=False)
        raters_df.to_csv(out_dir / task / "raters.csv", index=False)
        summaries.append({
            "task": task,
            "n_labels_fit": int(len(Y)),
            "n_cases": int(n_local_cases),
            "n_raters": int(len(rater_globals)),
            "pos_rate": float(Y.mean()),
        })
        with open(out_dir / task / "summary.json", "w") as f:
            json.dump(summaries[-1], f, indent=2)

    # Σ + correlation
    slot_names = []
    for t in TASKS:
        slot_names.append(f"t_{t}")
        slot_names.append(f"l_{t}")
    pd.DataFrame(Sigma, index=slot_names, columns=slot_names).to_csv(out_dir / "Sigma.csv")
    sd = np.sqrt(np.diag(Sigma))
    Corr = Sigma / np.outer(sd, sd)
    pd.DataFrame(Corr, index=slot_names, columns=slot_names).to_csv(out_dir / "correlation.csv")

    r_hat, rho_hat = moment_match_factor(Sigma)
    r_iiic_hat, r_cross_hat = moment_match_block(Sigma)
    factor_summary = {
        "variant": variant,
        "r_hat": float(r_hat),
        "rho_hat": float(rho_hat),
        "r_iiic_hat": float(r_iiic_hat),
        "r_cross_hat": float(r_cross_hat),
        "K": K,
        "tasks": TASKS,
        "wall_time_s": float(wall_time),
        "max_outer": opts["max_outer"],
        "min_raters_per_case": opts["min_raters_per_case"],
        "min_labels_per_rater": opts["min_labels_per_rater"],
        "min_votes_per_class": opts["min_votes_per_class"],
        "sigma_s": opts["sigma_s"],
        "ridge": opts["ridge"],
    }
    with open(out_dir / "factor_summary.json", "w") as f:
        json.dump(factor_summary, f, indent=2)
    with open(out_dir / "fit_summary.json", "w") as f:
        json.dump({"variant": variant, "per_task": summaries,
                   "factor": factor_summary}, f, indent=2)


# ───────────────────── main ─────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["free", "factor", "block", "both", "all"], default="both")
    p.add_argument("--max-outer", type=int, default=DEFAULTS["max_outer"])
    p.add_argument("--min-labels-per-rater", type=int,
                    default=DEFAULTS["min_labels_per_rater"])
    p.add_argument("--min-raters-per-case", type=int,
                    default=DEFAULTS["min_raters_per_case"])
    p.add_argument("--out-suffix", default="",
                    help="appended to data/labels/fits_hier_<variant><suffix>/")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    opts = dict(DEFAULTS)
    opts["max_outer"] = args.max_outer
    opts["min_labels_per_rater"] = args.min_labels_per_rater
    opts["min_raters_per_case"] = args.min_raters_per_case

    if args.variant == "both":
        variants = ["free", "factor"]
    elif args.variant == "all":
        variants = ["free", "factor", "block"]
    else:
        variants = [args.variant]
    for var in variants:
        print(f"\n========== Hierarchical 2PL probit fit, variant = {var} ==========")
        (theta, Sigma, s_arrays, s_sd_arrays, mask, task_data, t_sd, l_sd,
          rater_list, case_g2l, raters_meta, wall) = fit_hier(
            var, opts, verbose=not args.quiet)
        r_hat, rho_hat = moment_match_factor(Sigma)
        print(f"\nVariant '{var}' summary:")
        print(f"  wall time = {wall:.0f}s")
        print(f"  r_hat = {r_hat:+.3f}    rho_hat = {rho_hat:+.3f}")
        print(f"  Sigma diag = {np.diag(Sigma).round(3)}")

        out_dir = ROOT / f"data/labels/fits_hier_{var}{args.out_suffix}"
        write_outputs(out_dir, theta, Sigma, s_arrays, s_sd_arrays, mask, task_data,
                      t_sd, l_sd, rater_list, case_g2l,
                      raters_meta, opts, var, wall)
        print(f"  wrote outputs to {out_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
