"""CORTEX termination + verdict policies — the AD6 resolution.

This module owns the live test's stopping and verdict logic. It is the
ONLY place that needs to change to swap to a different stopping rule —
the rest of CORTEX (session controller, storage, results screen) treats
the policy as an opaque object via the `TerminationPolicy` interface.

Three policies ship:

  AD6Policy          — the resolved rule (default for production).
                       Per-task three-way verdict:
                          PASS    if  n_k ≥ N_min  AND  R_k ≥ R*  AND
                                      π_k − Z·mcse_k ≥ 1−α
                          FAIL    if  n_k ≥ N_min  AND  R_k ≥ R*  AND
                                      π_k + Z·mcse_k ≤ α
                          REFER   otherwise at session end, split into
                          REFER_BORDERLINE  (gate open, π_k stayed in band)
                          REFER_UNINFORMATIVE  (gate never opened — the
                                             one-vs-rest degeneracy case)
                       Session stops when all 6 tasks are RESOLVED, the
                       MAX_QUESTIONS_DEFAULT (300, v1.1.0) cap is hit, or
                       the bank is exhausted.
                       This is the unanimous output of a four-expert
                       independent derivation panel — see
                       docs/AD6_RESOLUTION.md.

  DeltaStopPolicy    — the legacy Mode-A AUROC half-width rule
                       (max_k HW < δ). Retained for the methodology /
                       Paper-1 path; not the live-test default.

  NoStopPolicy       — never stops; runs to the bank cap. For the audit
                       and OC harness, which need full trajectories.

To swap the rule later, write a new TerminationPolicy subclass and pass
it as `policy=` to `CortexSession`. The default policy is built from
`inputs` via `AD6Policy.from_inputs(inputs)`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Frozen PyInstaller bundles: __file__ for a PYZ-loaded module does not
# resolve to a real filesystem path whose parent.parent is the data unpack
# root. sys._MEIPASS is set by the bootloader to that root; the bundled
# calibration/cert_config.yaml (and the optional cert_config.yaml fallback)
# live under it.
if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
else:
    _REPO = Path(__file__).resolve().parent.parent

# ── verdict labels ────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING"
REFER_BORDERLINE = "REFER_BORDERLINE"
REFER_UNINFORMATIVE = "REFER_UNINFORMATIVE"

# Default rule parameters — middle-of-range from the four-expert panel
# (each one is independently OC-calibratable in the harness; see
# scripts/run_oc_validation.py).
#
# v1.1.0 calibration-trial settings (2026-05-27, against the 300-IIIC
# bank). N_min is now at the panel-derived production value of 15 —
# reachable in expectation at 300/6 ≈ 50 q/task. ALPHA is the midpoint
# between the v1.0 internal-test override (0.30) and the panel-derived
# production target (0.05) — gives a calibration run before committing
# to 0.05 in v1.2.0. See docs/AD6_RESOLUTION.md "v1.1.0 strictness
# upgrade" for the rationale and OC implications.
#
# v1.2.0 calibration (2026-05-29) against the 350-seg K=7 internal bank.
# Selected by sim_v1_2_0/ — see results/sim_v1_2_0/report.md for the
# 500-session sweep. Target: median ~200 trials on the 350-seg K=7 bank
# (57% utilization, 43% selector headroom). Achieved median 188 with
# 76% all_resolved at the values below. The engine quits early
# (~110 trials) at confident-skill extremes and runs long at borderline
# skill (ℓ=+0.5 hits the 300-cap in ~60% of sessions) — correct
# adaptive behavior verified.
#
# History of (N_MIN, ALPHA):
#   v1.0   — N_MIN=6,  ALPHA=0.30  (internal-test override; 100-seg bank)
#   v1.1.0 — N_MIN=15, ALPHA=0.10  (panel-target strictness mid-point;
#                                    300-IIIC bank)
#   v1.2.0 — N_MIN=12, ALPHA=0.25  (sim_v1_2_0 calibration on 350-seg
#                                    K=7 bank)
#   v1.2.3 — N_MIN=20, ALPHA=0.25  (THIS RELEASE; harden against the
#                                    random-rater edge case observed on
#                                    Eli's v1.2.0/v1.2.2 self-test —
#                                    session f3da305d-... — where gpd
#                                    and iic spuriously PASSed via early
#                                    verdict-lock at the N_MIN=12 floor
#                                    when posterior was momentarily
#                                    elevated. PhD agent analysis:
#                                    raising N_MIN 12 → 20 ~halves the
#                                    per-task spurious-PASS rate for a
#                                    random rater (~5.5% → ~3%) while
#                                    keeping the median session length
#                                    inside the 350-seg bank. ALPHA
#                                    unchanged at 0.25 by design — keeps
#                                    the internal-test fast while still
#                                    raising the bar. Monotonic-lock is
#                                    kept as a deliberate adaptive
#                                    design choice; see
#                                    `docs/PHASE9_LOCK_AUDIT.md`.)
#   v2.0   — TBD; panel-target production strictness is ALPHA=0.05; the
#            switch depends on full-cohort OC sims at production
#            n_particles=600.
DEFAULT_N_MIN = 20        # v1.2.3: was 12 — Eli's random-rater edge-case
                          # hardening. Outside the sim_v1_2_0/ sweep grid
                          # (which covered {8,10,12,15}); extrapolated
                          # from NMIN=15, ALPHA=0.25 (median 221, p95 300,
                          # all_resolved 64%) to NMIN=20 → expected
                          # median ~245 trials on the 350-seg K=7 bank.
DEFAULT_R_STAR = 0.30     # info-gate: SD(ℓ_k) must contract ≥ ~16% from prior
DEFAULT_ALPHA = 0.05      # v1.3.6: 0.25 -> 0.05, the panel-derived PRODUCTION /
                          # NEJM AI paper-grade certification threshold (95%
                          # posterior confidence for PASS/FAIL). The earlier 0.25
                          # was the deliberate internal-test-fast override
                          # (v1.2.0-v1.3.5). alpha=0.05 is viable now that the
                          # 700-seg bank + MAX_Q=500 give clear candidates ~92%
                          # per-domain resolution (sim_v1_3_5/run_bank_size_sweep
                          # + run_v1_3_6_confirm). See cortex-nejm-ai-v1_3_6.
DEFAULT_Z = 2.0           # MC-error buffer (engine's existing Z_BUFFER)


@dataclass
class StopDecision:
    """One trial's policy output. `verdicts`/`diagnostics` are None for
    policies that do not render verdicts (NoStop, DeltaStop)."""
    stop: bool
    stop_reason: str                          # "continue" | "delta_reached" |
                                              # "all_resolved" | "bank_exhausted"
    verdicts: list | None = None              # per-task PASS/FAIL/PENDING
    diagnostics: dict | None = None           # per-task π, mcse, R, n + ESS


# ── ℓ*_k loader ───────────────────────────────────────────────────────────
def load_ell_star_iiic(task_codes, config_path=None):
    """Load the Youden ℓ*_k for the K=6 IIIC tasks from
    calibration/cert_config.yaml's `ell_star_unified_v13` block.
    Returns a list aligned to ``task_codes``.
    """
    import yaml
    path = Path(config_path) if config_path else (_REPO / "calibration"
                                                  / "cert_config.yaml")
    if not path.exists():
        # Bundle fallback: the lab distribution stages cert_config at the
        # repo root rather than under calibration/.
        alt = _REPO / "cert_config.yaml"
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(
            f"cert_config.yaml not found at {path} — AD6Policy needs the "
            "Youden ℓ*_k cut-scores.")
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    try:
        tasks = data["ell_star_unified_v13"]["tasks"]
    except KeyError as e:
        raise KeyError(f"cert_config missing ell_star_unified_v13.tasks: {e}")
    out = []
    for code in task_codes:
        key = f"sparcnet_{code}"
        if key not in tasks:
            raise KeyError(f"cert_config has no entry for {key}")
        out.append(float(tasks[key]["ell_star"]))
    return out


# ── policy base + flavors ────────────────────────────────────────────────
class TerminationPolicy:
    """Common interface. Policies may be stateful across a session — call
    `reset(K)` at run start and `finalize_verdicts()` at run end."""

    def reset(self, K: int) -> None:                # pragma: no cover
        pass

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        raise NotImplementedError

    def finalize_verdicts(self) -> list | None:
        return None


class NoStopPolicy(TerminationPolicy):
    """Never stops. Used by the audit + OC harness + bank-exhaustion tests."""

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        return StopDecision(stop=False, stop_reason="continue")


class DeltaStopPolicy(TerminationPolicy):
    """Legacy Mode-A rule: stop when max-k AUROC half-width < delta.
    Retained for back-compat with the methodology / Paper-1 path. NOT the
    live-test default — that is AD6Policy."""

    def __init__(self, delta_auroc: float = 0.15):
        self.delta_auroc = float(delta_auroc)

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        if telemetry["max_hw"] < self.delta_auroc:
            return StopDecision(stop=True, stop_reason="delta_reached")
        return StopDecision(stop=False, stop_reason="continue")


class AD6Policy(TerminationPolicy):
    """The resolved AD6 rule — per-task three-way classification with an
    information-sufficiency gate.

    For each task k after every trial:
        π_k    = Σ_i w_i · 1[ℓ_k^(i) > ℓ*_k]                     pass-mass
        mcse_k = √( π_k(1−π_k) / ESS )                            MC error
        R_k    = 1 − Var_post(ℓ_k) / Var_prior(ℓ_k)               info gate

    Task k is RESOLVED iff n_k ≥ N_min AND R_k ≥ R* AND
        PASS : π_k − Z·mcse_k ≥ 1 − α
        FAIL : π_k + Z·mcse_k ≤ α
    The information gate is mandatory and was independently derived by all
    four panel experts — it prevents the one-vs-rest degeneracy from
    certifying a task on a prior-dominated posterior.

    Session stops when every task is RESOLVED. At end of run,
    `finalize_verdicts()` converts each remaining PENDING task into either
    REFER_BORDERLINE (gate open, posterior simply did not cross) or
    REFER_UNINFORMATIVE (gate never opened — data did not constrain ℓ_k).
    """

    def __init__(self, ell_star, var_prior, *,
                 n_min: int = DEFAULT_N_MIN,
                 R_star: float = DEFAULT_R_STAR,
                 alpha: float = DEFAULT_ALPHA,
                 Z: float = DEFAULT_Z):
        self.ell_star = np.asarray(ell_star, dtype=float)
        self.var_prior = np.asarray(var_prior, dtype=float)
        if self.ell_star.shape != self.var_prior.shape:
            raise ValueError(
                f"ell_star ({self.ell_star.shape}) and var_prior "
                f"({self.var_prior.shape}) must align by task")
        self.n_min = int(n_min)
        self.R_star = float(R_star)
        self.alpha = float(alpha)
        self.Z = float(Z)
        self._verdicts: list | None = None
        self._last_diag: dict | None = None

    @classmethod
    def from_inputs(cls, inputs, **kwargs):
        """Construct from an IIICEngineInputs: ℓ*_k from cert_config v13,
        Var_prior from the engine's fitted Corr_l diagonal."""
        ell_star = load_ell_star_iiic(inputs.task_codes)
        var_prior = list(np.diag(np.asarray(inputs.Corr_l, dtype=float)))
        return cls(ell_star, var_prior, **kwargs)

    def reset(self, K: int) -> None:
        if len(self.ell_star) != K:
            raise ValueError(
                f"K={K} != len(ell_star)={len(self.ell_star)} — task mismatch")
        self._verdicts = [PENDING] * K
        self._last_diag = None

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        if self._verdicts is None:
            self.reset(K)
        w = state["w"]
        wsum = float(w.sum())
        w = w / wsum if wsum > 0 else np.full_like(w, 1.0 / len(w))
        L = state["l"]                                # shape (N, K)
        ess = float(1.0 / float((w * w).sum()))

        pi = np.zeros(K)
        mcse = np.zeros(K)
        R = np.zeros(K)
        for k in range(K):
            ell_k = L[:, k]
            pi_k = float((w * (ell_k > self.ell_star[k])).sum())
            mu_k = float((w * ell_k).sum())
            var_post = float((w * (ell_k - mu_k) ** 2).sum())
            pi[k] = pi_k
            mcse[k] = float(np.sqrt(max(pi_k * (1.0 - pi_k), 0.0)
                                    / max(ess, 1.0)))
            R[k] = 1.0 - var_post / float(self.var_prior[k])

        # Update verdicts monotonically — once PASS/FAIL, stays that way.
        for k in range(K):
            if self._verdicts[k] != PENDING:
                continue
            if n_per_task[k] < self.n_min or R[k] < self.R_star:
                continue
            if pi[k] - self.Z * mcse[k] >= 1.0 - self.alpha:
                self._verdicts[k] = PASS
            elif pi[k] + self.Z * mcse[k] <= self.alpha:
                self._verdicts[k] = FAIL

        diag = {
            "pi": pi.tolist(),
            "mcse": mcse.tolist(),
            "R": R.tolist(),
            "ess": ess,
            "verdicts": list(self._verdicts),
            "n_per_task": list(n_per_task),
        }
        self._last_diag = diag
        if all(v != PENDING for v in self._verdicts):
            return StopDecision(stop=True, stop_reason="all_resolved",
                                verdicts=list(self._verdicts),
                                diagnostics=diag)
        return StopDecision(stop=False, stop_reason="continue",
                            verdicts=list(self._verdicts), diagnostics=diag)

    def finalize_verdicts(self) -> list:
        """Convert each PENDING task into REFER_BORDERLINE (gate opened,
        posterior stayed in band) or REFER_UNINFORMATIVE (gate never
        opened — data did not constrain ℓ_k)."""
        if self._verdicts is None:
            return []
        out = list(self._verdicts)
        R = self._last_diag["R"] if self._last_diag else [0.0] * len(out)
        for k in range(len(out)):
            if out[k] == PENDING:
                out[k] = (REFER_BORDERLINE if R[k] >= self.R_star
                          else REFER_UNINFORMATIVE)
        return out


# ── default builder ──────────────────────────────────────────────────────
def default_policy_for(inputs, *, delta_auroc=None,
                       policy=None) -> TerminationPolicy:
    """Resolve the policy actually used by a CortexSession. Precedence:
      explicit `policy=` instance         → use it
      `delta_auroc` is None (production)  → AD6Policy.from_inputs(inputs)
      `delta_auroc == 0.0` (audit/tests)  → NoStopPolicy()
      `delta_auroc > 0` (legacy/methods)  → DeltaStopPolicy(delta_auroc)
    """
    if policy is not None:
        return policy
    if delta_auroc is None:
        return AD6Policy.from_inputs(inputs)
    if float(delta_auroc) == 0.0:
        return NoStopPolicy()
    return DeltaStopPolicy(float(delta_auroc))
