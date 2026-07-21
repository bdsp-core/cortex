"""Unified public names for the deployed CORTEX trainer policy.

The aliases keep the production class identities intact while presenting one
consistent reviewer-facing vocabulary under ``trainer_policy``.
"""
from __future__ import annotations

from .deployed import EngineManager, EngineSession
from .policy import LETrainerPolicy


TrainerPolicy = LETrainerPolicy
TrainerSession = EngineSession
TrainerManager = EngineManager

__all__ = ["TrainerPolicy", "TrainerSession", "TrainerManager"]
