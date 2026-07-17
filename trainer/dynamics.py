"""Learner-dynamics parameters (G1 port of trainer_rd learner_sim.LearnerParams).

`LearnerParams` is the filter's *assumed* transition-kernel parameterization and
is SHIPPED (the runtime filter needs it). The `Learner` truth-kernel simulator
stays research-only (ported later to `trainer/sim/`). Verbatim field/default copy
of the scratch dataclass so ported-filter behavior is bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

from trainer.conventions import LAPSE_RATE

RULES = ("soft", "hard", "static")


@dataclass
class LearnerParams:
    """Per-learner dynamics parameters (one set, broadcast across tasks unless
    given as arrays). σ_∞ defaults are the data-grounded expert ceilings; the
    caller passes the per-task vector."""
    alpha_t: float = 0.15          # criterion learning rate (D7 placeholder)
    alpha_sigma: float = 0.02      # skill rate = 1/τ_σ (τ≈50 trials)
    sigma_inf: float = 0.5         # skill floor (override per task)
    q_t: float = 0.05              # criterion process-noise SD
    q_sigma: float = 0.02          # log-skill process-noise SD
    rho: float = 0.5               # 85%-weight width
    lapse: float = LAPSE_RATE
    rule: str = "soft"

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
