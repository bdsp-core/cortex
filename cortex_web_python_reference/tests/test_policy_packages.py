from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [
    str(WORKSPACE / "termination-policy" / "src"),
    str(WORKSPACE / "ad6-policy" / "src"),
    str(WORKSPACE / "precision-policy" / "src"),
    str(ROOT / "engine"),
    str(ROOT / "scripts"),
]

from ad6_policy import AD6Policy as StandaloneAD6Policy  # noqa: E402
from precision_policy import PrecisionPolicy  # noqa: E402
from cortex_engine_inputs_k7 import build_k7_engine_inputs  # noqa: E402
from cortex_policy import AD6Policy  # noqa: E402
from session_controller import (  # noqa: E402
    build_cortex_session,
    make_simulated_y_source,
)


def test_policy_code_is_owned_by_the_organized_projects() -> None:
    assert Path(StandaloneAD6Policy.__module__.replace(".", "/")) == Path(
        "ad6_policy/policy"
    )
    assert Path(PrecisionPolicy.__module__.replace(".", "/")) == Path(
        "precision_policy/policy"
    )
    assert issubclass(AD6Policy, StandaloneAD6Policy)
    assert (WORKSPACE / "ad6-policy" / "src" / "ad6_policy" / "policy.py").is_file()
    assert (
        WORKSPACE
        / "precision-policy"
        / "src"
        / "precision_policy"
        / "policy.py"
    ).is_file()


def test_local_session_factory_defaults_to_ad6(monkeypatch) -> None:
    monkeypatch.delenv("CORTEX_TERMINATION_POLICY", raising=False)
    session = build_cortex_session(build_k7_engine_inputs(), session_id="ad6-default")
    assert isinstance(session.policy, AD6Policy)


def test_local_session_factory_freezes_the_full_precision_profile(monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_TERMINATION_POLICY", "precision_v1")
    session = build_cortex_session(
        build_k7_engine_inputs(), session_id="precision-profile", seed=7
    )
    assert isinstance(session.policy, PrecisionPolicy)
    assert session.N == 1200
    assert session.objective == "total_var"
    assert session.n_subsample == 128
    assert session.uncertainty_aware_subsample is True
    assert session.first_item_topn == 10
    assert session._max_consec == 5
    assert session.floor_progress_deadline is True
    assert session.per_domain_cap == 60
    assert session.policy.interval_radius_scale_by_domain is None
    assert session.policy.interval_radius_ramp_max_by_domain is None
    assert session.policy.bias_radius_scale_by_domain is None
    assert not hasattr(session.policy, "ell_star")

    with pytest.raises(ValueError, match="freezes n_particles"):
        build_cortex_session(
            build_k7_engine_inputs(),
            termination_policy="precision_v1",
            n_particles=600,
        )


def test_local_session_factory_rejects_unknown_policy(monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_TERMINATION_POLICY", "typo")
    with pytest.raises(ValueError, match="CORTEX_TERMINATION_POLICY"):
        build_cortex_session(build_k7_engine_inputs())


@pytest.mark.skipif(
    os.environ.get("CORTEX_RUN_POLICY_GOLDENS") != "1",
    reason="full 1200-particle per-engine golden sessions are opt-in",
)
@pytest.mark.parametrize(
    ("policy_name", "expected"),
    [
        (
            "ad6",
            {
                "n_questions": 300,
                "stop_reason": "resolved_or_referred",
                "outcomes": [
                    "REFER_BORDERLINE",
                    "REFER_BORDERLINE",
                    "PASS",
                    "PASS",
                    "REFER_BORDERLINE",
                    "PASS",
                    "REFER_BORDERLINE",
                ],
                "n_per_task": [60, 60, 20, 20, 60, 20, 60],
            },
        ),
        (
            "precision_v1",
            {
                "n_questions": 317,
                "stop_reason": "all_estimated_or_undeterminable",
                "outcomes": [
                    "DETERMINED",
                    "DETERMINED",
                    "DETERMINED",
                    "DETERMINED",
                    "UNDETERMINABLE_CAP",
                    "UNDETERMINABLE_CAP",
                    "DETERMINED",
                ],
                "n_per_task": [35, 59, 31, 40, 60, 60, 32],
            },
        ),
    ],
)
def test_deterministic_full_session_golden(policy_name, expected) -> None:
    inputs = build_k7_engine_inputs()
    session = build_cortex_session(
        inputs,
        termination_policy=policy_name,
        session_id=f"{policy_name}-golden",
        seed=20260718,
    )
    result = session.run(
        make_simulated_y_source(np.zeros(7), np.full(7, 0.6), seed=20260719)
    )
    outcomes = result.verdicts if policy_name == "ad6" else result.determinations
    assert {
        "n_questions": result.n_questions,
        "stop_reason": result.stop_reason,
        "outcomes": outcomes,
        "n_per_task": result.trials[-1]["n_per_task"],
    } == expected
    assert result.served_seg_ids[:10] == [
        7977,
        9213,
        3027,
        8146,
        6983,
        6289,
        1902,
        1230,
        1417,
        866,
    ]
    assert max(result.trials[-1]["n_per_task"]) <= 60
