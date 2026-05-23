"""Phase A — tests for scripts/cortex_engine_inputs.py.

The IIIC-only engine inputs the CORTEX internal test feeds to the SMC
particle-cloud engine. The whole module is skipped when data/eeg_bank.h5
is absent — it is gitignored (D9), so a clean checkout has no bank.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import cortex_engine_inputs as cei  # noqa: E402
from core_mcmc import make_state_hier, choose_item  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(cei.BANK_PATH),
    reason=f"eeg_bank.h5 not present at {cei.BANK_PATH}")


@pytest.fixture(scope="module")
def inputs():
    return cei.build_iiic_engine_inputs()


def test_build_returns_100_iiic_segments(inputs):
    assert len(inputs.manifest) == 100
    assert inputs.manifest.index.name == "seg_id"
    assert len(inputs.all_seg_ids) == 100


def test_six_tasks_in_canonical_order(inputs):
    assert inputs.task_codes == ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
    assert inputs.task_labels == [
        "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]


def test_no_nan_signals(inputs):
    sig_cols = ([f"s_mean_{c}" for c in inputs.task_codes]
                + [f"s_sd_{c}" for c in inputs.task_codes])
    assert not inputs.manifest[sig_cols].isna().any().any()


def test_corr_l_is_valid_prior(inputs):
    C = inputs.Corr_l
    assert C.shape == (6, 6)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(C) > 0), "Corr_l must be positive definite"


def test_as_engine_arrays_full_bank(inputs):
    bs, bsd, bseg = inputs.as_engine_arrays()
    assert len(bs) == len(bsd) == len(bseg) == 6
    for k in range(6):
        assert len(bs[k]) == len(bsd[k]) == len(bseg[k]) == 100
        assert not np.isnan(bs[k]).any()
        assert np.all(bsd[k] > 0)


def test_as_engine_arrays_subset_dedup(inputs):
    keep = inputs.all_seg_ids[:99]
    bs, bsd, bseg = inputs.as_engine_arrays(keep)
    assert all(len(a) == 99 for a in bs)
    assert list(bseg[0]) == keep


def test_as_engine_arrays_rejects_unknown_seg(inputs):
    with pytest.raises(KeyError):
        inputs.as_engine_arrays([-1])


def test_true_task_index_consistent(inputs):
    for sid in inputs.all_seg_ids:
        k = inputs.true_task_index(sid)
        assert 0 <= k < 6
        assert inputs.task_pattern_words[k] == inputs.pattern_class(sid)


def test_without_removes_segments(inputs):
    drop = inputs.all_seg_ids[:3]
    sub = inputs.without(drop)
    assert len(sub.manifest) == len(inputs.manifest) - 3
    assert not (set(drop) & set(sub.all_seg_ids))
    bs, _, _ = sub.as_engine_arrays()
    assert len(bs[0]) == len(inputs.manifest) - 3


def test_inputs_drive_choose_item(inputs):
    rng = np.random.default_rng(0)
    state = make_state_hier(400, 6, 0.378, rng,
                            Sigma_l=inputs.Corr_l, Sigma_t=inputs.Corr_l)
    bs, bsd, bseg = inputs.as_engine_arrays()
    k, s, s_sd, seg = choose_item(state, bs, bank_sds=bsd,
                                  return_sd=True, bank_segids=bseg)
    assert 0 <= k < 6
    assert seg in inputs.all_seg_ids
    assert np.isfinite(s) and s_sd >= 0.0


def test_dedup_excludes_served_segment(inputs):
    rng = np.random.default_rng(0)
    state = make_state_hier(400, 6, 0.378, rng,
                            Sigma_l=inputs.Corr_l, Sigma_t=inputs.Corr_l)
    bs, bsd, bseg = inputs.as_engine_arrays()
    _, _, _, seg = choose_item(state, bs, bank_sds=bsd,
                               return_sd=True, bank_segids=bseg)
    remaining = [x for x in inputs.all_seg_ids if x != seg]
    bs2, bsd2, bseg2 = inputs.as_engine_arrays(remaining)
    _, _, _, seg2 = choose_item(state, bs2, bank_sds=bsd2,
                                return_sd=True, bank_segids=bseg2)
    assert seg2 != seg
    assert seg not in np.concatenate(bseg2)
