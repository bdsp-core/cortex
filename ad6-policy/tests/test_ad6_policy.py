from __future__ import annotations

import numpy as np

from ad6_policy import AD6Policy, PASS, PENDING, REFER_BORDERLINE
from cortex_termination_policy import TerminationPolicy


def test_ad6_is_standalone_and_preserves_monotone_verdict_locks() -> None:
    policy = AD6Policy([0.0], [1.0], n_min=1, R_star=0.0, alpha=0.25, Z=0.0)
    assert isinstance(policy, TerminationPolicy)
    policy.reset(1)
    state = {
        "l": np.array([[1.0], [2.0]]),
        "w": np.array([0.5, 0.5]),
    }
    assert policy(state, {}, [1], 1).verdicts == [PASS]

    reversed_state = {
        "l": np.array([[-2.0], [-1.0]]),
        "w": np.array([0.5, 0.5]),
    }
    assert policy(reversed_state, {}, [2], 1).verdicts == [PASS]


def test_unresolved_domain_finalizes_with_reason() -> None:
    policy = AD6Policy([0.0], [1.0], n_min=1, R_star=-1.0, alpha=0.05)
    policy.reset(1)
    state = {
        "l": np.array([[-1.0], [1.0]]),
        "w": np.array([0.5, 0.5]),
    }
    assert policy(state, {}, [1], 1).verdicts == [PENDING]
    assert policy.finalize_verdicts() == [REFER_BORDERLINE]
