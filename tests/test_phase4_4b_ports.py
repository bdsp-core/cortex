"""Phase 4.4-B — ported calibration scripts (fit_2pl_probit*, freeze).

These three scripts were ported faithfully (path-only + the locked
targeted K=7 / OR-iic-delete / erratum changes) and are UN-EXERCISED
until Phase 4.6 (no K=7 fit/freeze/sim run per the locked 4.4 scope).
The gate here is therefore static + data-integrity, NOT a sim:

  * the degenerate composite iic=OR branch is GONE; TASKS is the 7-list
    with the real "other" (not "iic"); zero absolute /Users/ paths.
  * the hier IIIC block is K-correct (IIIC_IDXS == range(1,K), so K=7
    → (1..6); block-Σ is well-posed for the independent real "other").
  * DATA-INTEGRITY: the ported extract_task_labels("other") positive
    count on the unified labels equals the Phase-3/v13 `sparcnet_iic`
    erratum-correct definition (value ∈ {other,bipd,birds}) so the
    deployment "other" Y is consistent with the v13 ℓ* it consumes at
    4.5 — and is NOT the plan's stale pre-merge 79,383 (documented).

Run: pytest tests/test_phase4_4b_ports.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
PIPE = REPO / "pipeline"
FIT = PIPE / "fit_2pl_probit.py"
HIER = PIPE / "fit_2pl_probit_hier.py"
FREEZE = REPO / "deployment" / "freeze_deployment_prior.py"
TASKS7 = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]


def test_ported_files_present_and_clean():
    for p in (FIT, HIER, FREEZE):
        assert p.is_file(), f"missing ported script {p}"
        txt = p.read_text()
        assert "/Users/" not in txt, f"absolute /Users/ path in {p.name}"
        assert "Path(__file__).resolve().parents[1]" in txt, (
            f"{p.name} ROOT not repo-root self-located")
        assert "Phase 4.4-B" in txt, f"{p.name} missing provenance header"


def test_degenerate_or_iic_deleted():
    """The composite iic = OR(seizure,lpd,gpd,lrda,grda) must be gone
    from BOTH fitters, replaced by the real erratum-correct 'other'."""
    for p in (FIT, HIER):
        txt = p.read_text()
        assert 'task == "iic"' not in txt, f"{p.name} still has iic branch"
        # the OR over the 5 subtypes (the degenerate construction)
        assert '"seizure", "lpd", "gpd", "lrda", "grda"])).astype' \
            not in txt, f"{p.name} still has the degenerate iic OR"
        assert '["other", "bipd", "birds"]' in txt, (
            f"{p.name} missing the erratum-correct other mapping")


def test_tasks_are_7_list_with_other():
    for p in (FIT, HIER, FREEZE):
        txt = p.read_text()
        assert ('TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", '
                '"grda", "other"]') in txt, f"{p.name} TASKS not 7-list"
        assert '"iic"]' not in txt, f"{p.name} still lists iic"


def test_hier_iiic_block_is_K_correct():
    """IIIC_IDXS derived as range(1,K): the documented 'idx0=spike,
    1..K-1=IIIC' structure, K=7 ⇒ (1,2,3,4,5,6). block_sigma must
    build a well-posed 14×14 with that block."""
    import numpy as np
    sys.path.insert(0, str(PIPE))
    import fit_2pl_probit_hier as h
    assert h.K == 7 and h.SPIKE_IDX == 0
    assert h.IIIC_IDXS == tuple(range(1, h.K)) == (1, 2, 3, 4, 5, 6)
    S = h.block_sigma(r_iiic=0.45, r_cross=0.30, K_=h.K)
    assert S.shape == (14, 14)
    # diagonal = 1 (scale convention); ℓ-ℓ within IIIC = r_iiic
    assert np.allclose(np.diag(S), 1.0)
    li, lj = 2 * 1 + 1, 2 * 6 + 1          # l_seizure vs l_other (IIIC)
    assert S[li, lj] == pytest.approx(0.45)
    ls, lo = 2 * 0 + 1, 2 * 6 + 1          # l_spike vs l_other (cross)
    assert S[ls, lo] == pytest.approx(0.30)


def test_y_other_data_integrity_is_v13_consistent():
    """The ported extract_task_labels('other') positive count on the
    UNIFIED labels MUST equal the Phase-3/v13 sparcnet_iic erratum-
    correct definition (so the deployment 'other' Y matches the v13 ℓ*
    consumed at 4.5) — and must NOT be the plan's stale 79,383."""
    labels_csv = REPO / "data" / "labels" / "labels.csv"
    if not labels_csv.exists():
        pytest.skip("unified labels.csv absent")
    sys.path.insert(0, str(PIPE))
    import fit_2pl_probit as f
    import fit_2pl_probit_hier as h
    L = pd.read_csv(labels_csv, low_memory=False, dtype={"value": "str"})
    pc = L[L.label_type == "pattern_class"].value.astype(str).str.strip()
    expected = int(pc.isin(["other", "bipd", "birds"]).sum())   # Phase-3
    assert expected > 0
    for mod in (f, h):
        pos = int(mod.extract_task_labels(L, "other").Y.sum())
        assert pos == expected, (
            f"{mod.__name__}: Y_other {pos} != erratum-correct "
            f"{expected} (v13 sparcnet_iic) — would mismatch the v13 "
            "ℓ* the deployment consumes at 4.5")
    assert expected != 79383, (
        "79,383 is the STALE pre-unified-merge PI count; the unified "
        "corpus (proven superset) erratum-correct count is the truth")
