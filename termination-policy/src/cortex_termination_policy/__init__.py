"""Shared structural contract for CORTEX stopping policies."""

from .contract import StopDecision, TerminationPolicy

__all__ = ["StopDecision", "TerminationPolicy"]
