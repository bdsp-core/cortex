"""v15 staged instrument — opt-in config for simulations (M11).

The LIVE instrument is FROZEN (v14 cuts, Corr_l reused for both prior blocks,
600 particles) for the in-flight tiered-pilot cohort. v15 is STAGED and
consumed opt-in (v15_simulation_spec.txt, 2026-06-11):

    v15 = credentialed ell* (cert_config `ell_star_unified_v15`)
        + corr_t prior      (t-block Corr_l -> Corr_t, npy key 'Corr_t')
        + n_particles=1200  (particle count, NOT a question budget;
                             stopping rule unchanged)

This module is the ONE place simulations get a versioned instrument from:
`instrument("v14")` / `instrument("v15")`. bank_adapter's module-level
constants stay v14 (live default) and are untouched.

Also carries the realistic-bias examinee prior (spec item 9):
theta ~ N(mu_t, Sigma_t) per-domain CORRELATED (engine coords; the
"correct-sign theta ~ -1.7" regime). Zero-bias arm: theta = 0.

CAVEAT (spec): every v15 resolution claim was measured on UNBIASED
examinees; the cohort's per-examinee theta is the gating check.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_ROOT, "config", "cert_config_general.yaml")
_SIGMA_NPY = os.path.join(_ROOT, "data", "Sigma_l_fitted_k7_general.npy")

TASK_CODES = ("domain1", "domain2", "domain3", "domain4",
              "domain5", "domain6", "domain7")
K = 7

# stopping rule — UNCHANGED between v14 and v15 (spec item 5)
N_MIN, R_STAR, ALPHA, Z = 20, 0.30, 0.05, 2.0
ESS_FRAC = 0.5

N_PARTICLES_FROZEN = 600
N_PARTICLES_STAGED = 1200


def _load_block(block_name):
    import yaml
    with open(_CONFIG) as fh:
        data = yaml.safe_load(fh)
    return data[block_name]


def _task_vec(block, key):
    by_code = {v["task_code"]: float(v[key]) for v in block["tasks"].values()}
    return np.array([by_code[c] for c in TASK_CODES])


def load_fitted_cov():
    """The fitted cross-rater covariance artifact (Corr_l, Corr_t, Sigma_l,
    Sigma_t) — local anonymized copy, verified entry-identical to the spec."""
    d = np.load(_SIGMA_NPY, allow_pickle=True).item()
    return {k: np.asarray(d[k], dtype=np.float64)
            for k in ("Corr_l", "Corr_t", "Sigma_l", "Sigma_t")}


@dataclass
class Instrument:
    """One certification-instrument configuration for simulation."""
    version: str
    ell_star: np.ndarray          # (7,) cut-scores
    sigma_star: np.ndarray        # (7,) = exp(-ell*)
    expert_ell: np.ndarray        # (7,) credentialed-panel ceiling
    sigma_inf: np.ndarray         # (7,) = exp(-expert_ell), LT1 target
    nonexpert_ell: np.ndarray     # (7,) fresh-learner prior center
    Sigma_l_prior: np.ndarray     # (7,7) l-block prior covariance (Corr_l)
    Sigma_t_prior: np.ndarray     # (7,7) t-block prior covariance
    n_particles: int

    def for_tasks(self, tasks):
        """Per-task views for a K-subset eval (tasks: indices into 0..6)."""
        idx = np.asarray(tasks, dtype=int)
        return {
            "ell_star": [float(self.ell_star[t]) for t in idx],
            "Sigma_l": self.Sigma_l_prior[np.ix_(idx, idx)],
            "Sigma_t": self.Sigma_t_prior[np.ix_(idx, idx)],
            "n_particles": self.n_particles,
        }


def instrument(version: str = "v14") -> Instrument:
    """The shipped ('v14') or staged ('v15') instrument.

    v14: v14 cuts; prior reuses Corr_l for BOTH blocks; 600 particles.
    v15: v15 cuts; t-block swaps to Corr_t;        1200 particles.
    Corr_l, Sigma_l, the bank and the labels are UNCHANGED between them.
    """
    cov = load_fitted_cov()
    if version == "v14":
        from training.bank_adapter import ELL_STAR, SIGMA_STAR, _EXPERT_ELL
        v14 = _load_block("ell_star_unified_v14")
        return Instrument(
            version="v14", ell_star=ELL_STAR.copy(),
            sigma_star=SIGMA_STAR.copy(), expert_ell=_EXPERT_ELL.copy(),
            sigma_inf=np.exp(-_EXPERT_ELL),
            nonexpert_ell=_task_vec(v14, "non_expert_ell_mean"),
            Sigma_l_prior=cov["Corr_l"], Sigma_t_prior=cov["Corr_l"],
            n_particles=N_PARTICLES_FROZEN)
    if version == "v15":
        v15 = _load_block("ell_star_unified_v15")
        expert = _task_vec(v15, "expert_ell_mean")
        return Instrument(
            version="v15", ell_star=_task_vec(v15, "ell_star"),
            sigma_star=_task_vec(v15, "sigma_star"), expert_ell=expert,
            sigma_inf=np.exp(-expert),
            nonexpert_ell=_task_vec(v15, "non_expert_ell_mean"),
            Sigma_l_prior=cov["Corr_l"], Sigma_t_prior=cov["Corr_t"],
            n_particles=N_PARTICLES_STAGED)
    raise ValueError(f"unknown instrument version {version!r}")


# ───────────── realistic-bias examinee prior (spec item 9) ─────────────

MU_T_CLIP = 2.0     # mu_t censored at +2 for domain2..domain7 (spec)


def load_bias_prior():
    """(mu_t, Sigma_t) of the examinee-bias prior, engine coords."""
    block = _load_block("eb_examinee_bias_prior_v15")
    mu = np.array([float(block["mu_t"][c]) for c in TASK_CODES])
    Sigma_t = load_fitted_cov()["Sigma_t"]
    return mu, Sigma_t


def draw_examinee_theta(rng, n=1, *, zero_bias=False, tasks=None):
    """Draw n examinees' true bias vectors theta (engine coords).

    zero_bias=True reproduces the spec's operating-characteristic arms
    (theta = 0). Otherwise theta ~ N(mu_t, Sigma_t) with draws clipped at
    +MU_T_CLIP on domain2..domain7 (the prior's censored support).
    tasks: optional index subset; returns (n, len(tasks))."""
    idx = np.arange(K) if tasks is None else np.asarray(tasks, dtype=int)
    if zero_bias:
        return np.zeros((int(n), idx.size))
    mu, Sig = load_bias_prior()
    th = rng.multivariate_normal(mu, Sig, size=int(n))
    th[:, 1:] = np.minimum(th[:, 1:], MU_T_CLIP)
    return th[:, idx]
