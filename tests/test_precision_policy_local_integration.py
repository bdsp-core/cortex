from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "termination-policy" / "src"),
    str(ROOT / "ad6-policy" / "src"),
    str(ROOT / "precision-policy" / "src"),
    str(ROOT / "engine"),
    str(ROOT / "scripts"),
]

from ad6_policy import AD6Policy as StandaloneAD6Policy  # noqa: E402
from precision_policy import PrecisionPolicy  # noqa: E402
from cortex_engine_inputs_k7 import build_k7_engine_inputs  # noqa: E402
from session_controller import build_cortex_session  # noqa: E402


def test_primary_local_factory_defaults_to_packaged_ad6(monkeypatch) -> None:
    monkeypatch.delenv("CORTEX_TERMINATION_POLICY", raising=False)
    session = build_cortex_session(build_k7_engine_inputs(), session_id="local-ad6")
    assert isinstance(session.policy, StandaloneAD6Policy)


def test_primary_local_factory_connects_frozen_precision_profile(monkeypatch) -> None:
    monkeypatch.setenv("CORTEX_TERMINATION_POLICY", "precision_v1")
    session = build_cortex_session(
        build_k7_engine_inputs(), session_id="local-precision", seed=11
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
    assert session.policy.reliability_mode == "quantile_mcse"
    assert session.policy.precision_statistic == "point_centered_radius"
    assert session.policy.interval_radius_scale_by_domain is None
    assert session.policy.interval_radius_ramp_max_by_domain is None
    assert session.policy.bias_radius_scale_by_domain is None
    assert not hasattr(session.policy, "ell_star")
