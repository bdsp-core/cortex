from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StopDecision:
    """One evaluation of a termination policy.

    AD6 populates ``verdicts``. PrecisionPolicy populates the additive domain
    status fields. Legacy policies may leave both result models empty.
    """

    stop: bool
    stop_reason: str
    verdicts: list | None = None
    diagnostics: dict | None = None
    domain_statuses: list | None = None
    streak_counts: list | None = None
    terminal_reasons: list | None = None


class TerminationPolicy:
    """Stateful policy interface consumed by the CORTEX session controller."""

    def reset(self, K: int) -> None:  # pragma: no cover - interface default
        pass

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        raise NotImplementedError

    def finalize_verdicts(self) -> list | None:
        return None

    @property
    def domain_statuses(self) -> list | None:
        return None

    @property
    def active_domains(self) -> list | None:
        return None
