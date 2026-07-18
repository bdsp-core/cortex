from __future__ import annotations

import numpy as np

from cortex_termination_policy import StopDecision, TerminationPolicy

PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING"
REFER_BORDERLINE = "REFER_BORDERLINE"
REFER_UNINFORMATIVE = "REFER_UNINFORMATIVE"

DEFAULT_N_MIN = 20
DEFAULT_R_STAR = 0.30
DEFAULT_ALPHA = 0.05
DEFAULT_Z = 2.0


class AD6Policy(TerminationPolicy):
    """AD6's monotone, cut-aware three-way verdict policy.

    For domain ``k`` the policy computes posterior mass above the supplied
    certification cut, a Monte-Carlo error guard, and posterior variance
    contraction. PASS and FAIL are locked once reached. Any unresolved domain
    is converted to a REFER reason when the session controller finalizes.

    Cut-vector loading intentionally lives outside this standalone package.
    """

    def __init__(
        self,
        ell_star,
        var_prior,
        *,
        n_min: int = DEFAULT_N_MIN,
        R_star: float = DEFAULT_R_STAR,
        alpha: float = DEFAULT_ALPHA,
        Z: float = DEFAULT_Z,
    ):
        self.ell_star = np.asarray(ell_star, dtype=float)
        self.var_prior = np.asarray(var_prior, dtype=float)
        if self.ell_star.shape != self.var_prior.shape:
            raise ValueError(
                f"ell_star ({self.ell_star.shape}) and var_prior "
                f"({self.var_prior.shape}) must align by task"
            )
        if (
            self.ell_star.ndim != 1
            or len(self.ell_star) == 0
            or np.any(~np.isfinite(self.ell_star))
            or np.any(~np.isfinite(self.var_prior))
            or np.any(self.var_prior <= 0.0)
        ):
            raise ValueError("ell_star and var_prior must be finite vectors")
        self.n_min = int(n_min)
        self.R_star = float(R_star)
        self.alpha = float(alpha)
        self.Z = float(Z)
        self._verdicts: list | None = None
        self._last_diag: dict | None = None

    @property
    def verdicts(self) -> list | None:
        return list(self._verdicts) if self._verdicts is not None else None

    @property
    def last_diagnostics(self) -> dict | None:
        return dict(self._last_diag) if self._last_diag is not None else None

    def reset(self, K: int) -> None:
        if len(self.ell_star) != K:
            raise ValueError(
                f"K={K} != len(ell_star)={len(self.ell_star)} — task mismatch"
            )
        self._verdicts = [PENDING] * K
        self._last_diag = None

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        if self._verdicts is None:
            self.reset(K)
        w = np.asarray(state["w"], dtype=float)
        wsum = float(w.sum())
        w = w / wsum if wsum > 0 else np.full_like(w, 1.0 / len(w))
        skill = np.asarray(state["l"], dtype=float)
        ess = float(1.0 / float((w * w).sum()))

        pi = np.zeros(K)
        mcse = np.zeros(K)
        contraction = np.zeros(K)
        for k in range(K):
            ell_k = skill[:, k]
            pi_k = float((w * (ell_k > self.ell_star[k])).sum())
            mu_k = float((w * ell_k).sum())
            var_post = float((w * (ell_k - mu_k) ** 2).sum())
            pi[k] = pi_k
            mcse[k] = float(
                np.sqrt(max(pi_k * (1.0 - pi_k), 0.0) / max(ess, 1.0))
            )
            contraction[k] = 1.0 - var_post / float(self.var_prior[k])

        for k in range(K):
            if self._verdicts[k] != PENDING:
                continue
            if n_per_task[k] < self.n_min or contraction[k] < self.R_star:
                continue
            if pi[k] - self.Z * mcse[k] >= 1.0 - self.alpha:
                self._verdicts[k] = PASS
            elif pi[k] + self.Z * mcse[k] <= self.alpha:
                self._verdicts[k] = FAIL

        diagnostics = {
            "pi": pi.tolist(),
            "mcse": mcse.tolist(),
            "R": contraction.tolist(),
            "ess": ess,
            "verdicts": list(self._verdicts),
            "n_per_task": list(n_per_task),
        }
        self._last_diag = diagnostics
        stop = all(verdict != PENDING for verdict in self._verdicts)
        return StopDecision(
            stop=stop,
            stop_reason="all_resolved" if stop else "continue",
            verdicts=list(self._verdicts),
            diagnostics=diagnostics,
        )

    def finalize_verdicts(self) -> list:
        if self._verdicts is None:
            return []
        out = list(self._verdicts)
        contraction = (
            self._last_diag["R"] if self._last_diag else [0.0] * len(out)
        )
        for k, verdict in enumerate(out):
            if verdict == PENDING:
                out[k] = (
                    REFER_BORDERLINE
                    if contraction[k] >= self.R_star
                    else REFER_UNINFORMATIVE
                )
        return out
