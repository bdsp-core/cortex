"""Golden parity and join tests for the unchanged Precision stopping policy.

The committed fixture (tests/fixtures/precision_join_golden.json) is written
by tests/precision_cli_golden.test.ts from a direct TypeScript invocation of
the frozen policy dispatch. Here the identical wire requests go through the
real sidecar subprocess + Python client; the responses must match exactly
(both sides are float64 with shortest-round-trip JSON, so equality is exact).
"""
from __future__ import annotations

import json

import pytest

from nway_protocol.precision_stop import (
    PROTOCOL_ROOT,
    PrecisionStopClient,
    ensure_sidecar_built,
)
from nway_protocol.qualification import QualificationConfig, run_replicate

GOLDEN = PROTOCOL_ROOT / "tests" / "fixtures" / "precision_join_golden.json"


@pytest.fixture(scope="module")
def client():
    ensure_sidecar_built()
    sidecar = PrecisionStopClient()
    yield sidecar
    sidecar.close()


def test_golden_parity_with_direct_ts_evaluation(client):
    payload = json.loads(GOLDEN.read_text())
    for request, expected in zip(
        payload["requests"], payload["responses"], strict=True,
    ):
        assert client.request(request) == expected


def test_precision_join_stops_via_the_frozen_policy(client):
    config = QualificationConfig(
        replicates=1, particles=48, own_cap=60, bank_segments=24,
        selector_per_domain=2, mh_steps=1, stopping="precision",
    )
    rows = run_replicate(62_100_777, config)
    for row in rows:
        assert row.end_statuses is not None and len(row.end_statuses) == 7
        # Domain 0 is never served by the IIIC harness.
        assert row.end_statuses[0] == "UNDETERMINABLE_BANK"
        # The session ended on the policy's stop, not the own-cap safety cap.
        assert "ACTIVE" not in row.end_statuses
        assert 1 <= row.questions <= config.bank_segments
    # Determinism: the sidecar re-initializes per session.
    again = run_replicate(62_100_777, config)
    assert [row.questions for row in again] == [row.questions for row in rows]
    assert [row.end_statuses for row in again] == [row.end_statuses for row in rows]


def test_own_cap_mode_records_no_precision_statuses():
    config = QualificationConfig(
        replicates=1, particles=32, own_cap=1, bank_segments=12,
        selector_per_domain=2, mh_steps=1,
    )
    rows = run_replicate(62_100_778, config)
    assert all(row.end_statuses is None for row in rows)
    assert all(row.questions == 6 for row in rows)
