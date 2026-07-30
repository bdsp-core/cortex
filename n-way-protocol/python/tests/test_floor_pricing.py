from __future__ import annotations

import numpy as np

from nway_protocol.bank_audit import BankSegment
from nway_protocol.floor_pricing import binary_information, price_floors


SEGMENT = BankSegment(
    segment_index=0,
    seg_id=1,
    s_mean=(0.0, 0.8, -0.4, 0.2, -0.9, 0.5, -0.1),
    s_sd=(0.0, 0.05, 0.1, 0.08, 0.12, 0.06, 0.09),
)


def test_binary_information_is_positive_and_focal_only():
    t = np.zeros((1, 7))
    l = np.zeros((1, 7))
    row = binary_information(SEGMENT, 2, t, l)
    assert row["skill_information"] > 0
    assert row["bias_information"] > 0
    # Perturbing a non-asked axis must not change the binary channel.
    shifted = np.zeros((1, 7))
    shifted[0, 4] = 0.7
    assert binary_information(SEGMENT, 2, t, shifted) == row


def test_price_floors_reports_monotone_retention():
    result = price_floors([SEGMENT], lapse_grid=(0.0, 0.15, 0.20))
    table = result["by_lapse"]
    assert table["0.0"]["retention_vs_unfloored_skill"] == 1.0
    retentions = [
        table[key]["retention_vs_unfloored_skill"] for key in ("0.0", "0.15", "0.2")
    ]
    assert retentions == sorted(retentions, reverse=True)
    assert all(0 < value <= 1 for value in retentions)
    # The categorical channel must never price below the binary channel.
    assert table["0.2"]["multiple_vs_binary_skill"] > 1
