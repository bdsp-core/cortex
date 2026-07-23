"""CORTEX provisional percentile policy.

This package is deliberately independent of ``cortex_web``.  It builds and
scores immutable reporting artifacts; it has no certification, stopping, or
training-allocation behavior.
"""

from .model import DOMAINS, score_candidate, verify_artifact

__all__ = ["DOMAINS", "score_candidate", "verify_artifact"]
__version__ = "0.1.0"
