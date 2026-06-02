"""Drift-guard for the frozen v1.3.6 paper-grade CORTEX instrument.

These run in CI during the enrollment window: if the bank, cut-score config,
engine prior, or any engine param drifts from the freeze, the cohort would no
longer be collected on a bit-identical instrument — these tests catch that.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
import freeze_instrument_v1_3_6 as fr  # noqa: E402

pytestmark = pytest.mark.skipif(
    not fr.FREEZE_PATH.exists(),
    reason="no v1.3.6 instrument freeze manifest present")


def test_live_instrument_matches_freeze():
    """Bank + cut-scores + prior + engine params must match the freeze."""
    problems = fr.check_drift()
    assert not problems, "v1.3.6 INSTRUMENT DRIFT:\n  " + "\n  ".join(problems)


def test_freeze_records_paper_grade_config():
    """The freeze must pin the NEJM AI paper-grade config."""
    m = json.loads(fr.FREEZE_PATH.read_text())
    assert m["params"]["alpha"] == 0.05          # 95% certification threshold
    assert m["params"]["max_questions"] == 500
    assert m["params"]["n_particles"] == 600
    assert m["bank"]["n_iiic"] == 600 and m["bank"]["n_spike"] == 100
    assert all(c == 100 for c in m["bank"]["iiic_by_class"].values())
