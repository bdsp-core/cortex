"""Phase-9 Gate A (procurement fix, 2026-05-29) — pin the prep_k7 changes:

  - Location anchor expanded from n=4 Centaur source to ≥3 expert raters
    across ANY source (≈5.4× expansion: 5,000 → 26,750 IIIC segs).
  - Scale anchor weld as UNION of pairwise intersections (the kong:expert
    weld provides label diversity even when seg-set overlaps kong:crowd).
  - 7-tier expertise_level → 4-tier collapse {expert, experienced, novice,
    non_expert} where non_expert = borderline + other + unknown + untiered.

These pin Gate-A behavior numerically; a future PR that changes prep_k7
must intentionally update these expectations.
"""
from __future__ import annotations

import pytest

from pipeline.joint_calibration import prep_k7


# ── Module-level constants ─────────────────────────────────────────────────

def test_gate_a_expert_loc_anchor_min_is_3():
    assert prep_k7.EXPERT_LOC_ANCHOR_MIN == 3


def test_gate_a_scale_welds_include_kong_expert():
    """Gate A scale anchor = UNION of pairwise welds; must include the
    kong:expert pair to add expert-tier observations to the gauge."""
    welds = prep_k7.IIIC_SCALE_WELDS
    assert isinstance(welds, tuple)
    assert len(welds) == 2
    pair_strs = ["|".join(sorted(p)) for p in welds]
    expected_pairs = {
        "|".join(sorted(("sparcnet50K", "iiic_crowdsourcing:kong2025:crowd"))),
        "|".join(sorted(("sparcnet50K", "iiic_crowdsourcing:kong2025:expert"))),
    }
    assert set(pair_strs) == expected_pairs


def test_gate_a_tier_collapse_4tier():
    """Gate A: 7-tier raters.csv expertise_level → 4-tier."""
    assert prep_k7._canonical_tier("expert") == "expert"
    assert prep_k7._canonical_tier("experienced") == "experienced"
    assert prep_k7._canonical_tier("novice") == "novice"
    # The 4 weakly-defined tiers all collapse to non_expert:
    assert prep_k7._canonical_tier("borderline") == "non_expert"
    assert prep_k7._canonical_tier("other") == "non_expert"
    assert prep_k7._canonical_tier("unknown") == "non_expert"
    assert prep_k7._canonical_tier("untiered") == "non_expert"
    # NaN / empty / unrecognised → non_expert (defensive)
    assert prep_k7._canonical_tier("") == "non_expert"
    assert prep_k7._canonical_tier("nonsense_value") == "non_expert"


# ── End-to-end anchor counts on full corpus ───────────────────────────────

@pytest.mark.slow
def test_gate_a_iiic_loc_anchor_size():
    """IIIC tasks: loc anchor jumps 5,000 → 26,750 (5.4× expansion).
    Marked slow because it scans full labels.csv (~25 sec)."""
    td = prep_k7.load_task("iic", subsample=None, seed=42)
    n_loc = int(td.loc_anchor_seg.sum())
    # Allow some tolerance for future minor corpus changes
    assert 25_000 <= n_loc <= 28_000, (
        f"IIIC iic loc anchor count {n_loc:,} outside Gate-A range "
        f"[25,000, 28,000]; if labels.csv changed, update this test "
        f"AND verify the NUTS multi-modal fix still holds.")


@pytest.mark.slow
def test_gate_a_iiic_4_tiers_in_rater_set():
    """IIIC tasks: rater set contains exactly the 4 canonical tiers."""
    td = prep_k7.load_task("iic", subsample=None, seed=42)
    tiers = set(td.rater_expertise)
    expected = {"expert", "experienced", "novice", "non_expert"}
    assert tiers == expected, f"got {tiers}; expected {expected}"


@pytest.mark.slow
def test_gate_a_spike_anchor_unchanged():
    """Gate A is IIIC-specific; spike anchor methodology unchanged.
    Spike loc anchor = mid-difficulty quantile band; scale = sn1:sn1."""
    td = prep_k7.load_task("spike", subsample=None, seed=42)
    n_loc = int(td.loc_anchor_seg.sum())
    n_scale = int(td.scale_anchor_seg.sum())
    # Spike loc ≈ 20% of segs (mid-difficulty band 0.40-0.60 quantile)
    expected_loc = int(0.20 * td.n_seg)
    assert abs(n_loc - expected_loc) / max(expected_loc, 1) < 0.10, (
        f"spike loc anchor {n_loc:,} vs expected ~{expected_loc:,} "
        f"(20% of {td.n_seg:,} segs)")
    # Scale anchor = sn1:sn1 sub-source segs
    assert n_scale >= 10_000, (
        f"spike scale anchor {n_scale:,} < 10,000; sn1:sn1 has ~13,262 segs")
