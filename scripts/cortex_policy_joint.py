"""CORTEX termination policy — the JOINT VERDICT-VECTOR REGION rule.

This is a DROP-IN sibling of `cortex_policy.AD6Policy` (same
TerminationPolicy `__call__` / `reset` / `finalize_verdicts` contract,
same `StopDecision` output, same `state["w"]`/`state["l"]` particle
contract) implementing the panel's chosen rule.  It does NOT edit
`cortex_policy.py`.

THE RULE (panel `chosen_rule_formal`)
-------------------------------------
Per candidate, K tasks.  From the live hier SMC cloud the ell-block is
`L = state["l"]` shape (N, K) with weights `w = state["w"]` shape (N,).

Provisional verdict v_k is fixed by the posterior-median sign
    s_k = sign(mu_k - ell*_k)   (s_k = +1 PASS, -1 FAIL),
with a SOFT PER-DOMAIN FREEZE: s_k is frozen the first time domain k's
MARGINAL pass-mass clears its info gate (R_k >= R*), keeping the orthant
R(v) fixed and P_joint monotone-trackable (panel `smc_estimator`,
mirrors AD6's verdict-lock semantics without the v1.2.3 spurious early
JOINT lock).

The correct-region is the orthant
    R(v) = { ell : s_k (ell_k - ell*_k) > 0  for all k }.
P_joint = P(ell in R(v) | data) = E_post[ prod_k 1[s_k(ell_k-ell*_k)>0] ].

STOP and emit v iff, FOR EVERY domain k, the AD6 preconditions hold
(n_k >= N_MIN and R_k >= R*) AND the buffered joint test passes:
    P_hat_joint - Z * SE(P_hat_joint) >= 0.95,   Z = 2.0.

This REPLACES AD6's per-domain product-of-marginals conjunction with a
single orthant test on the full 2K-dim cloud, so Sigma_post (cross-domain
correlation) enters the STOP DECISION itself.  The 0.95 bar and Z=2.0 are
LOCKED; AD6's alpha is NOT used (the conjunction self-corrects multiplicity).

ESTIMATOR (panel `smc_estimator`, two-tier)
-------------------------------------------
PRIMARY — Rao-Blackwellized Gaussian fit.  One weighted pass over the
cloud gives mu_post (K) and Sigma_post (K x K) of the ell-block.  Evaluate
Phi_K(D mu_d ; D Sigma_d D), D = diag(s_k), via scipy's MVN CDF (K small)
with a one-factor Gauss-Hermite fallback that uses scipy.special.log_ndtr
inside the product (numerically stable to |a|~8 — never the clipped
norm.cdf, per CLAUDE.md FIX-T0.7).  This is SMOOTH (no 1/ESS granularity
wall); its only error is Gaussian-shape misfit.

CONTROL VARIATE / VALIDATION — raw weighted indicator.
    P_hat_raw = sum_i w_i prod_k 1[s_k(ell_k^(i)-ell*_k)>0]
with weighted-Bernoulli SE = sqrt(P_hat(1-P_hat)/ESS), ESS = 1/sum w_i^2
(matches cortex_policy.py:272).  Reported as a diagnostic.

The Gaussian-fit SE is taken as max(Bernoulli-SE-at-P_gauss, jackknife-ish
floor); the STOP buffer uses the Gaussian estimator P_gauss with that SE
(it is the statistic that does not suffer the 1/ESS floor).

VERDICT LOCK + REFER semantics are preserved: a domain's PROVISIONAL sign
freezes once its info gate opens; at end-of-run PENDING domains become
REFER_BORDERLINE (gate open) / REFER_UNINFORMATIVE (gate never opened),
exactly as AD6.

The BRUTE comparator is the like-for-like control: brute's K independent
clouds force P_joint = prod_k pi_k (core_mcmc_brute_k._pass_prob_joint_brute);
the harness drives that arm with this same 0.95 bar.  See
sim_v1_3_5/run_joint_vs_ad6.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
else:
    _REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Reuse verdict labels, StopDecision, the base class, and the defaults from
# the shipped policy module so the contract is byte-identical.
from cortex_policy import (  # noqa: E402
    TerminationPolicy, StopDecision,
    PASS, FAIL, PENDING, REFER_BORDERLINE, REFER_UNINFORMATIVE,
    DEFAULT_N_MIN, DEFAULT_R_STAR, DEFAULT_Z,
)

# Joint-rule LOCKED parameters (panel chosen_rule_formal).
JOINT_BAR = 0.95          # P_joint stop bar (LOCKED)
# N_MIN, R*, Z reuse the AD6 ship defaults; alpha is intentionally UNUSED.

try:
    from scipy.stats import multivariate_normal as _mvn
    _HAVE_MVN = True
except Exception:                                       # pragma: no cover
    _HAVE_MVN = False

from scipy.special import log_ndtr                       # noqa: E402
from numpy.polynomial.hermite_e import hermegauss        # noqa: E402

# One-factor Gauss-Hermite nodes (probabilists' Hermite -> N(0,1) weight).
_GH_NODES, _GH_WTS = hermegauss(33)
_GH_WTS = _GH_WTS / np.sqrt(2.0 * np.pi)                 # normalize to E_F[.]


def _orthant_prob_gauss(mu_d, Sigma_d):
    """P(X > 0 elementwise) for X ~ N(mu_d, Sigma_d), K-dim.

    Equivalently Phi_K(mu_d; Sigma_d) after the sign flip already applied
    to mu_d/Sigma_d by the caller (so the region is the positive orthant).
    Tries scipy's MVN CDF first (exact Genz for small K); falls back to a
    one-factor Gauss-Hermite reduction that is stable in the tails.
    """
    K = len(mu_d)
    sd = np.sqrt(np.clip(np.diag(Sigma_d), 1e-12, None))
    # Standardized lower bounds a_k = -mu_k/sd_k  ->  P(Z_k > a_k) jointly,
    # i.e. Phi_K(-a ; Corr).  Equivalent to CDF of -Z at -mu/sd.
    if _HAVE_MVN:
        try:
            # P(X>0) = P(-X < 0) = CDF of N(-mu_d, Sigma_d) at 0.
            rv = _mvn(mean=-mu_d, cov=Sigma_d, allow_singular=True)
            p = float(rv.cdf(np.zeros(K)))
            if np.isfinite(p):
                return float(np.clip(p, 0.0, 1.0))
        except Exception:
            pass
    # One-factor fallback: approximate the correlation by a single common
    # factor with loading sqrt(rho_bar), rho_bar = mean off-diagonal corr.
    corr = Sigma_d / np.outer(sd, sd)
    off = corr[~np.eye(K, dtype=bool)]
    rho_bar = float(np.clip(off.mean(), 0.0, 0.999)) if off.size else 0.0
    a = (-mu_d) / sd                       # P(Z_k > a_k) jointly
    sqrt_rho = np.sqrt(rho_bar)
    sqrt_1mr = np.sqrt(max(1.0 - rho_bar, 1e-9))
    acc = 0.0
    for f, wgt in zip(_GH_NODES, _GH_WTS):
        # P(Z_k > a_k | F=f) = Phi((sqrt_rho f - a_k)/sqrt_1mr)
        z = (sqrt_rho * f - a) / sqrt_1mr
        log_term = log_ndtr(z).sum()
        acc += wgt * np.exp(log_term)
    return float(np.clip(acc, 0.0, 1.0))


class JointVerdictRegionPolicy(TerminationPolicy):
    """The JOINT VERDICT-VECTOR REGION rule — drop-in for AD6Policy.

    Constructor signature mirrors AD6Policy (ell_star, var_prior, +kwargs),
    so any AD6 call site can swap the class.  `bar` (0.95) and `Z` (2.0)
    are LOCKED defaults; `alpha` is accepted-and-ignored for call-site
    compatibility with AD6Policy.
    """

    def __init__(self, ell_star, var_prior, *,
                 n_min: int = DEFAULT_N_MIN,
                 R_star: float = DEFAULT_R_STAR,
                 Z: float = DEFAULT_Z,
                 bar: float = JOINT_BAR,
                 alpha: float = None,            # accepted, UNUSED (AD6 parity)
                 ess_orthant_min: float = 50.0):
        self.ell_star = np.asarray(ell_star, dtype=float)
        self.var_prior = np.asarray(var_prior, dtype=float)
        if self.ell_star.shape != self.var_prior.shape:
            raise ValueError(
                f"ell_star ({self.ell_star.shape}) and var_prior "
                f"({self.var_prior.shape}) must align by task")
        self.n_min = int(n_min)
        self.R_star = float(R_star)
        self.Z = float(Z)
        self.bar = float(bar)
        self.ess_orthant_min = float(ess_orthant_min)
        self._verdicts: list | None = None        # final PASS/FAIL/PENDING
        self._sign: np.ndarray | None = None       # frozen provisional s_k
        self._sign_frozen: np.ndarray | None = None
        self._last_diag: dict | None = None

    @classmethod
    def from_inputs(cls, inputs, **kwargs):
        """Construct from a K7EngineInputs-like object (ell* via cert_config
        K=7 loader, Var_prior from the fitted Corr_l diagonal)."""
        from cortex_policy_k7 import load_ell_star_k7
        ell_star = load_ell_star_k7(inputs.task_codes)
        var_prior = list(np.diag(np.asarray(inputs.Corr_l, dtype=float)))
        return cls(ell_star, var_prior, **kwargs)

    def reset(self, K: int) -> None:
        if len(self.ell_star) != K:
            raise ValueError(
                f"K={K} != len(ell_star)={len(self.ell_star)} — task mismatch")
        self._verdicts = [PENDING] * K
        self._sign = np.ones(K)                 # provisional sign, +1 default
        self._sign_frozen = np.zeros(K, dtype=bool)
        self._last_diag = None

    # ── core posterior summaries ──────────────────────────────────────────
    def _marginals(self, w, L, K):
        """Per-domain mu_k, var_post_k, pi_k (marginal pass-mass), R_k."""
        mu = np.empty(K)
        var_post = np.empty(K)
        pi = np.empty(K)
        R = np.empty(K)
        for k in range(K):
            ell_k = L[:, k]
            mu_k = float((w * ell_k).sum())
            v = float((w * (ell_k - mu_k) ** 2).sum())
            mu[k] = mu_k
            var_post[k] = v
            pi[k] = float((w * (ell_k > self.ell_star[k])).sum())
            R[k] = 1.0 - v / float(self.var_prior[k])
        return mu, var_post, pi, R

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        if self._verdicts is None:
            self.reset(K)
        w = state["w"]
        wsum = float(w.sum())
        w = w / wsum if wsum > 0 else np.full_like(w, 1.0 / len(w))
        L = state["l"]                                # (N, K)
        ess = float(1.0 / float((w * w).sum()))

        mu, var_post, pi, R = self._marginals(w, L, K)

        # ── soft per-domain sign freeze ──────────────────────────────────
        # Provisional sign s_k = sign(mu_k - ell*_k); freeze the first time
        # domain k's info gate opens (R_k >= R*), keeping the orthant fixed.
        gate_open = (R >= self.R_star) & (np.asarray(n_per_task) >= self.n_min)
        for k in range(K):
            if not self._sign_frozen[k]:
                s_now = 1.0 if (mu[k] - self.ell_star[k]) >= 0.0 else -1.0
                self._sign[k] = s_now
                if gate_open[k]:
                    self._sign_frozen[k] = True
        s = self._sign

        # ── Rao-Blackwellized Gaussian-fit orthant probability (PRIMARY) ──
        d = mu - self.ell_star                         # centered means
        # Weighted ell-block covariance Sigma_post (K x K).
        dev = L - mu[None, :]                          # (N, K)
        Sigma_post = (dev * w[:, None]).T @ dev        # weighted cov, (K,K)
        Sigma_post = Sigma_post + 1e-9 * np.eye(K)     # engine jitter convention
        D = np.diag(s)
        mu_d = D @ d                                   # sign-flipped means
        Sigma_d = D @ Sigma_post @ D                   # sign-flipped cov
        p_gauss = _orthant_prob_gauss(mu_d, Sigma_d)

        # ── raw weighted indicator (CONTROL VARIATE / validation) ─────────
        inreg = np.ones(len(w), dtype=bool)
        for k in range(K):
            inreg &= (s[k] * (L[:, k] - self.ell_star[k]) > 0.0)
        p_raw = float(w[inreg].sum())
        # orthant-occupancy ESS (panel n_min-analogue gate)
        wr = w[inreg]
        ess_orthant = (float(wr.sum()) ** 2 / float((wr * wr).sum())
                       if wr.size and (wr * wr).sum() > 0 else 0.0)

        # ── SE + buffered joint test ──────────────────────────────────────
        # Gaussian-fit statistic with a weighted-Bernoulli SE evaluated at
        # p_gauss (the smooth statistic; SE is the n_min-analogue buffer).
        se_gauss = float(np.sqrt(max(p_gauss * (1.0 - p_gauss), 0.0)
                                 / max(ess, 1.0)))
        se_raw = float(np.sqrt(max(p_raw * (1.0 - p_raw), 0.0)
                               / max(ess, 1.0)))
        joint_lcb = p_gauss - self.Z * se_gauss

        all_gates_open = bool(gate_open.all())
        stop_test = (all_gates_open and joint_lcb >= self.bar)

        diag = {
            "p_joint_gauss": p_gauss,
            "p_joint_raw": p_raw,
            "joint_lcb": joint_lcb,
            "se_gauss": se_gauss,
            "se_raw": se_raw,
            "ess": ess,
            "ess_orthant": ess_orthant,
            "mu": mu.tolist(),
            "pi": pi.tolist(),
            "R": R.tolist(),
            "sign": s.tolist(),
            "sign_frozen": self._sign_frozen.tolist(),
            "gate_open": gate_open.tolist(),
            "post_corr_mean": _post_corr_mean(Sigma_post),
            "n_per_task": list(n_per_task),
        }
        self._last_diag = diag

        if stop_test:
            # Emit the full verdict vector from the frozen provisional signs.
            verdicts = [PASS if s[k] > 0 else FAIL for k in range(K)]
            self._verdicts = list(verdicts)
            diag["verdicts"] = list(verdicts)
            return StopDecision(stop=True, stop_reason="joint_region",
                                verdicts=list(verdicts), diagnostics=diag)

        # Provisional (not-yet-stopped) verdicts for live telemetry: PENDING
        # until the joint test fires (the conjunction is the unit of decision).
        provisional = [
            (PASS if s[k] > 0 else FAIL) if self._sign_frozen[k] else PENDING
            for k in range(K)
        ]
        diag["verdicts"] = list(provisional)
        return StopDecision(stop=False, stop_reason="continue",
                            verdicts=list(provisional), diagnostics=diag)

    def finalize_verdicts(self) -> list:
        """End-of-run: if the joint test fired the verdicts are already set.
        Otherwise convert each domain to its provisional sign verdict if its
        gate opened, else REFER_BORDERLINE (gate open, joint stalled) /
        REFER_UNINFORMATIVE (gate never opened), mirroring AD6."""
        if self._verdicts is None:
            return []
        # If the joint test already emitted a full PASS/FAIL vector, keep it.
        if all(v in (PASS, FAIL) for v in self._verdicts):
            return list(self._verdicts)
        R = self._last_diag["R"] if self._last_diag else [0.0] * len(self._verdicts)
        out = []
        for k in range(len(self._verdicts)):
            if self._sign_frozen is not None and self._sign_frozen[k]:
                # gate opened but the JOINT test never fired -> borderline
                out.append(REFER_BORDERLINE)
            elif R[k] >= self.R_star:
                out.append(REFER_BORDERLINE)
            else:
                out.append(REFER_UNINFORMATIVE)
        return out


def _post_corr_mean(Sigma_post):
    """Mean off-diagonal posterior correlation — for the decay diagnostic."""
    sd = np.sqrt(np.clip(np.diag(Sigma_post), 1e-12, None))
    corr = Sigma_post / np.outer(sd, sd)
    K = corr.shape[0]
    off = corr[~np.eye(K, dtype=bool)]
    return float(off.mean()) if off.size else 0.0
