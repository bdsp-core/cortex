from __future__ import annotations

import numpy as np

from nway_protocol.bank_audit import BankSegment, audit_bank, candidate_information


def _segments() -> list[BankSegment]:
    return [
        BankSegment(
            segment_index=index,
            seg_id=10_000 + index,
            s_mean=tuple(np.linspace(-1.5, 1.5, 7) + index * 0.1),
            s_sd=tuple(np.full(7, 0.03 + index * 0.02)),
        )
        for index in range(4)
    ]


def test_candidate_information_reports_skill_bias_and_cross_domain_terms():
    metrics = candidate_information(
        _segments()[0], 2, np.zeros((1, 7)), np.zeros((1, 7)), 0.9912, 0,
    )
    assert set(metrics) == {
        "skill_information", "bias_information", "cross_skill_information",
        "cross_bias_information", "focal_skill_information", "focal_bias_information",
    }
    assert all(np.isfinite(value) and value >= 0 for value in metrics.values())
    assert metrics["cross_skill_information"] > 0


def test_bank_audit_is_nonmutating_and_ranks_each_asked_task():
    segments = _segments()
    before = list(segments)
    result = audit_bank(segments, top_per_task=2)
    assert segments == before
    assert result["status"] == "audit_only_no_bank_mutation"
    assert result["candidate_rows"] == len(segments) * 6
    assert set(result["top_by_asked_task"]) == {"1", "2", "3", "4", "5", "6"}
    assert all(len(rows) == 2 for rows in result["top_by_asked_task"].values())
