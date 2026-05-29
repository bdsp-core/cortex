"""Phase 9 K=7 data prep — extends prep.py to spike + 6 IIIC = 7 tasks.

For one task one-vs-rest, builds dense-indexed JAX-ready arrays from the
canonical data/labels/labels.csv, plus the two split-gauge anchor masks.

Differences vs prep.py (the K=6 IIIC predecessor):

  * **TASK_TO_CLASS extended:**  `'spike': 'spike'` added; the IIIC mappings
    are unchanged (`sz→seizure, lpd→lpd, gpd→gpd, lrda→lrda, grda→grda,
    iic→other`).
  * **Per-task source filter:** spike-source datasets vs IIIC-source datasets
    are disjoint in the canonical labels.csv (spike comes from
    `sn1_combined_v2:{sn1,bonobo_only,fabio_spikeed}`; IIIC comes from
    `sparcnet50K, pd_rda_profiler, kong2025, centaur_2025_iiic,
    centaur_iiic_expert`). load_task() picks the source set by task family.
  * **Spike gauge anchors:** the IIIC gauge is `sparcnet50K ∩ kong:crowd`
    (scale) + `centaur_iiic_expert` (location). Spike has no equivalent
    cross-source weld or gold panel in the unified corpus; the spike
    task uses sn1-only scale anchor and a stratified-quantile location
    anchor (mid-difficulty segs) — see SPIKE_SCALE_WELD_SRCS,
    SPIKE_LOCATION_ANCHOR_STRATEGY below. This is the documented K=7
    extension; the gauge anchoring methodology is preserved per
    `pipeline/joint_calibration/fit_joint_iiic.py:102-117`.
  * **TaskData schema unchanged** — same dataclass shape; downstream
    `fit_joint_k7.py` consumes it the same way `fit_joint_iiic.py`
    consumes `prep.load_task` output.

The {bipd,birds}→other uniform collapse is applied to IIIC tasks only
(spike is its own pattern_class value, not subject to the collapse).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "data" / "labels" / "labels.csv"
RATERS = REPO / "data" / "labels" / "raters.csv"
SEGMENTS = REPO / "data" / "labels" / "segments.csv"

# Source-dataset namespaces are ASYMMETRIC between labels.csv and segments.csv:
#   labels.csv:  source_dataset = 'sn1_combined_v2' (collapsed)
#   segments.csv: source_dataset = 'sn1_combined_v2:{sn1,bonobo_only,fabio_spikeed}'
# For the spike task we lift the sub-source from segments.csv via seg_id join so
# the joint fit's per-source latent (δ_d, γ_d) is at the granularity matching
# the IIIC fit's per-source latent (where labels.csv already carries the
# granular source_dataset).

# Label-type per task family:
#   spike:  label_type='spike', value ∈ {'0','1'} (binary clean spike from sn1)
#   IIIC:   label_type='pattern_class', value ∈ 6-class + {bipd,birds → other}
SPIKE_LABEL_TYPE = "spike"
IIIC_LABEL_TYPE = "pattern_class"

# Source-dataset filter per task family.
# IIIC: 6 distinct datasets in labels.csv.
# Spike: only sn1_combined_v2 in labels.csv (clean binary; matches v13 un-fold
# decision per cert_config.yaml:303-310). Centaur IED panel (167,503 labels
# from centaur_2025_ied with multi-class values like ied/posts/bets/etc.) is
# NOT included — that's Paper-2 K=8 spike-variant work.
IIIC_SOURCES_LABELS = {
    "sparcnet50K", "pd_rda_profiler",
    "iiic_crowdsourcing:kong2025:crowd",
    "iiic_crowdsourcing:kong2025:expert",
    "centaur_2025_iiic", "centaur_iiic_expert",
}
SPIKE_SOURCES_LABELS = {"sn1_combined_v2"}
SPIKE_SUBSOURCES_SEGMENTS = (
    "sn1_combined_v2:sn1",
    "sn1_combined_v2:bonobo_only",
    "sn1_combined_v2:fabio_spikeed",
)

# K=7 task code -> canonical positive-class value
TASK_TO_CLASS = {
    "spike": "1",         # spike binary: value '1' positive, value '0' negative
    "sz": "seizure",
    "lpd": "lpd",
    "gpd": "gpd",
    "lrda": "lrda",
    "grda": "grda",
    "iic": "other",
}
TASKS = list(TASK_TO_CLASS)

# IIIC task -> {bipd,birds} → other uniform collapse (Phase-3 erratum fix).
# Spike is binary {'0','1'}; non-{'0','1'} values are dropped at the label
# level (the centaur_2025_ied rows with values 'ied'/'posts'/etc never reach
# here because centaur_2025_ied is NOT in SPIKE_SOURCES_LABELS).
_COLLAPSE_IIIC = {"bipd": "other", "birds": "other"}

# Tier canonicalisation (Phase-9 Gate A, 2026-05-29).
# raters.csv:expertise_level has 7 values: expert (n=174), experienced (n=163),
# novice (n=1244), borderline (n=47), other (n=465), unknown (n=740), and
# NaN/untiered (n=2473). The 7-tier scheme creates per-tier `a_skill[g]` and
# `a_bias[g]` parameters with weak likelihood support for 4 of the 7 tiers,
# adding multi-modality at no inferential value (per Phase-9 code-architect
# audit). Collapse to 4 tiers — preserves the inferentially-meaningful
# expert/experienced/novice distinction; collapses the 4 weakly-defined buckets
# into a single `non_expert` tier.
_TIER_COLLAPSE = {
    "expert": "expert",
    "experienced": "experienced",
    "novice": "novice",
    "borderline": "non_expert",
    "other": "non_expert",
    "unknown": "non_expert",
    "untiered": "non_expert",
}
EXPERT_TIER = "expert"

# Gauge anchors per task family.
#
# Phase-9 Gate A (procurement fix, 2026-05-29) — the n=4 Centaur location
# anchor was insufficient for full-corpus NUTS convergence (R̂ up to 72 on
# spike + gpd in winner_full at 1500/1500/4 vectorised chains). The fix
# expands both anchors to break the gauge-ridge-multimodality at the DATA
# level rather than the model level (preserves V_B + D2 invariant
# byte-pinned two-stage fitter + full corpus):
#
#   Location anchor: ANY segment with >= EXPERT_LOC_ANCHOR_MIN expert raters
#       across ANY source. With 174 experts in raters.csv and ≥3 threshold,
#       this yields ~26,750 IIIC anchor segments vs the prior 5,000 (5.4×
#       expansion → √5.4 ≈ 2.3× location-ridge tightening).
#   Scale anchor: UNION of pairwise welds (a segment is scale-anchor iff it
#       appears in BOTH sources for ANY weld pair). The kong:expert weld
#       adds expert-tier observations to break Kong-crowd dominance at the
#       weld geometry; per-pair sizes: sparcnet ∩ kong:crowd 6,691;
#       sparcnet ∩ kong:expert 4,488; UNION ~10K.
IIIC_SCALE_WELDS = (
    ("sparcnet50K", "iiic_crowdsourcing:kong2025:crowd"),
    ("sparcnet50K", "iiic_crowdsourcing:kong2025:expert"),
)
EXPERT_LOC_ANCHOR_MIN = 3  # ≥3 of 174 experts on segment → location anchor

# Spike: no cross-source weld available in sn1_combined_v2 (single source).
# Scale anchor = the sn1:sn1 sub-source (the largest sub-source = 13,262 segs);
# location anchor = mid-difficulty segs (computed inside load_task using
# pos_rate quantiles).
SPIKE_SCALE_ANCHOR_SUBSRC = "sn1_combined_v2:sn1"
SPIKE_LOCATION_QUANTILE = (0.40, 0.60)  # mid-difficulty band (20% of segs)


@dataclass
class TaskData:
    """Identical shape to prep.TaskData — fit_joint_k7 consumes both K=6 and
    K=7 outputs via the same dataclass interface (no schema change)."""

    task: str
    n_obs: int
    seg_idx: np.ndarray       # (N,) dense 0..S-1
    rater_idx: np.ndarray     # (N,) dense 0..R-1
    src_idx: np.ndarray       # (N,) dense 0..D-1
    y: np.ndarray             # (N,) {0,1}
    n_seg: int
    n_rater: int
    n_src: int
    scale_anchor_seg: np.ndarray   # bool (S,) — per-task scale anchor
    loc_anchor_seg: np.ndarray     # bool (S,) — per-task location anchor
    seg_ids: np.ndarray            # (S,) original unified seg_id
    rater_ids: np.ndarray          # (R,) original unified rater_id
    src_names: list                # (D,)
    rater_canonical: list          # (R,) canonical_name (R5-safe)
    rater_expertise: list          # (R,) expertise_level
    pos_rate: float


def _canon_iiic_value(v: str) -> str:
    """Apply the {bipd,birds}→other collapse for IIIC tasks."""
    v = v.strip()
    return _COLLAPSE_IIIC.get(v, v)


def _canonical_tier(raw_tier: str) -> str:
    """Map raters.csv:expertise_level to the 4-tier collapse (Phase-9 Gate A).
    Unknown/empty raw_tier defaults to 'non_expert'."""
    raw_tier = (raw_tier or "untiered").strip() or "untiered"
    return _TIER_COLLAPSE.get(raw_tier, "non_expert")


def _build_expert_rater_set(rmeta: dict) -> set:
    """Return the set of rater_ids where canonical tier == EXPERT_TIER ('expert').
    Built from the rmeta dict that load_task constructs from raters.csv."""
    return {rid for rid, (_name, tier) in rmeta.items()
            if tier == EXPERT_TIER}


def _spike_subsource_map() -> dict[int, str]:
    """Build {seg_id → sub-source} from segments.csv for the 20,521 spike segs.
    labels.csv has source_dataset='sn1_combined_v2' (collapsed); the
    sub-source granularity (sn1, bonobo_only, fabio_spikeed) lives in
    segments.csv."""
    out: dict[int, str] = {}
    with open(SEGMENTS) as f:
        for d in csv.DictReader(f):
            sd = d["source_dataset"]
            if sd in SPIKE_SUBSOURCES_SEGMENTS:
                out[int(d["seg_id"])] = sd
    return out


def _spike_location_anchor(seg_u: list, rows: list, smap: dict) -> np.ndarray:
    """Spike has no cross-source gold panel. Define the location anchor as
    segments with pos_rate in the SPIKE_LOCATION_QUANTILE band (mid-
    difficulty), so the gauge location is fit on segments whose latent s_j
    is well-constrained from the binary observations.
    """
    S = len(seg_u)
    if not rows:
        return np.zeros(S, bool)
    pos_n = np.zeros(S)
    obs_n = np.zeros(S)
    for seg, _rid, _sd, yv in rows:
        si = smap[seg]
        obs_n[si] += 1.0
        pos_n[si] += float(yv)
    pr = np.divide(pos_n, np.maximum(obs_n, 1.0))
    valid = obs_n > 0
    if valid.sum() < 10:
        return np.zeros(S, bool)
    q_lo, q_hi = SPIKE_LOCATION_QUANTILE
    lo, hi = np.quantile(pr[valid], [q_lo, q_hi])
    mask = (pr >= lo) & (pr <= hi) & valid
    return mask


def load_task(task: str, subsample: int | None = None,
              seed: int = 0) -> TaskData:
    """Build arrays for one K=7 one-vs-rest task. Subsample caps N for
    smoke tests; None = full corpus."""
    if task not in TASK_TO_CLASS:
        raise ValueError(f"task {task!r} not in {TASKS}")
    is_iiic = task != "spike"
    pos_class = TASK_TO_CLASS[task]
    sources_labels = IIIC_SOURCES_LABELS if is_iiic else SPIKE_SOURCES_LABELS
    label_type_filter = IIIC_LABEL_TYPE if is_iiic else SPIKE_LABEL_TYPE

    # Rater metadata for canonical_name + (collapsed) expertise_level.
    # Phase-9 Gate A: apply 7-tier → 4-tier collapse via _canonical_tier.
    rmeta: dict[str, tuple[str, str]] = {}
    with open(RATERS) as f:
        for d in csv.DictReader(f):
            rmeta[d["rater_id"]] = (
                (d.get("canonical_name") or f"rater_{d['rater_id']}"),
                _canonical_tier(d.get("expertise_level") or ""),
            )

    # Phase-9 Gate A: expert rater set for the location-anchor count rule.
    expert_rids = _build_expert_rater_set(rmeta)

    # For spike: pre-load the seg→subsource map (used to remap source_dataset
    # from labels.csv's collapsed 'sn1_combined_v2' to the granular
    # 'sn1_combined_v2:{sn1,bonobo_only,fabio_spikeed}' for the joint fit's
    # δ_d/γ_d per-source latent.
    subsrc_map: dict[int, str] = (
        {} if is_iiic else _spike_subsource_map()
    )

    # Stream labels.csv; filter to task sources + label_type; collect rows.
    # IIIC (Phase-9 Gate A):
    #   - Track multi-pair scale-weld source memberships (UNION semantics)
    #   - Track per-seg expert-rater counts → ≥3 → location anchor
    # Spike: track sn1:sn1 scale-anchor seg ids (via subsrc_map); location
    #   anchor via mid-difficulty quantile (_spike_location_anchor).
    iiic_weld_srcs = set()
    if is_iiic:
        for pair in IIIC_SCALE_WELDS:
            iiic_weld_srcs.update(pair)
    iiic_seg_in_src: dict[str, set] = (
        {s: set() for s in iiic_weld_srcs} if is_iiic else {}
    )
    # Per-seg expert-rater set (for ≥EXPERT_LOC_ANCHOR_MIN gate)
    iiic_seg_expert_rids: dict[str, set] = {}
    spike_scale_seg_ids: set = set()
    rows = []
    with open(LABELS) as f:
        for d in csv.DictReader(f):
            if d["label_type"] != label_type_filter:
                continue
            sd_raw = d["source_dataset"]
            if sd_raw not in sources_labels:
                continue
            seg = d["seg_id"]
            rid = d["rater_id"]
            v = d["value"].strip()

            if is_iiic:
                if sd_raw in iiic_seg_in_src:
                    iiic_seg_in_src[sd_raw].add(seg)
                # Track expert-rater membership per segment (any source)
                if rid in expert_rids:
                    iiic_seg_expert_rids.setdefault(seg, set()).add(rid)
                y = 1 if _canon_iiic_value(v) == pos_class else 0
                sd_for_src = sd_raw
            else:
                # Spike: binary value filter ({'0','1'} only); centaur IED
                # multi-class values never reach here (centaur_2025_ied
                # is NOT in SPIKE_SOURCES_LABELS).
                if v not in ("0", "1"):
                    continue
                # Remap to sub-source via segments.csv join
                seg_int = int(seg)
                sd_sub = subsrc_map.get(seg_int)
                if sd_sub is None:
                    # seg_id has spike label but no segments.csv sub-source —
                    # extremely rare; skip defensively.
                    continue
                if sd_sub == SPIKE_SCALE_ANCHOR_SUBSRC:
                    spike_scale_seg_ids.add(seg)
                y = 1 if v == pos_class else 0
                sd_for_src = sd_sub  # use granular sub-source for src_idx

            rows.append((seg, rid, sd_for_src, y))

    if subsample and len(rows) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=subsample, replace=False)
        rows = [rows[i] for i in idx]

    if not rows:
        raise RuntimeError(
            f"prep_k7.load_task({task!r}): no rows matched "
            f"label_type={label_type_filter!r} + "
            f"source_dataset∈{sources_labels} + value-filter. "
            "Check labels.csv schema.")

    # Build dense index maps over unique (seg, rater, source)
    seg_u = sorted({r[0] for r in rows}, key=int)
    rat_u = sorted({r[1] for r in rows}, key=int)
    src_u = sorted({r[2] for r in rows})
    smap = {s: i for i, s in enumerate(seg_u)}
    rmap = {r: i for i, r in enumerate(rat_u)}
    dmap = {s: i for i, s in enumerate(src_u)}

    seg_idx = np.fromiter((smap[r[0]] for r in rows), np.int32, len(rows))
    rater_idx = np.fromiter((rmap[r[1]] for r in rows), np.int32, len(rows))
    src_idx = np.fromiter((dmap[r[2]] for r in rows), np.int32, len(rows))
    y = np.fromiter((r[3] for r in rows), np.int8, len(rows))

    S = len(seg_u)
    if is_iiic:
        # Phase-9 Gate A: scale anchor = UNION of pairwise intersections.
        # Each pair contributes segs in BOTH sources of that pair; the
        # segment is scale-anchor iff it's in ANY pair's intersection.
        weld: set = set()
        for src_a, src_b in IIIC_SCALE_WELDS:
            weld |= (iiic_seg_in_src.get(src_a, set())
                     & iiic_seg_in_src.get(src_b, set()))
        scale_mask = np.array([seg_u[i] in weld for i in range(S)], bool)
        # Phase-9 Gate A: location anchor = segs with >=EXPERT_LOC_ANCHOR_MIN
        # expert raters (any source). Replaces the n=4 centaur-only anchor.
        loc_mask = np.array(
            [len(iiic_seg_expert_rids.get(seg_u[i], set()))
             >= EXPERT_LOC_ANCHOR_MIN
             for i in range(S)], bool)
    else:
        scale_mask = np.array(
            [seg_u[i] in spike_scale_seg_ids for i in range(S)], bool)
        loc_mask = _spike_location_anchor(seg_u, rows, smap)

    rc = [rmeta.get(r, (f"rater_{r}", "untiered")) for r in rat_u]

    return TaskData(
        task=task, n_obs=len(rows), seg_idx=seg_idx, rater_idx=rater_idx,
        src_idx=src_idx, y=y.astype(np.int32), n_seg=S, n_rater=len(rat_u),
        n_src=len(src_u),
        scale_anchor_seg=scale_mask, loc_anchor_seg=loc_mask,
        seg_ids=np.array([int(s) for s in seg_u]),
        rater_ids=np.array([int(r) for r in rat_u]),
        src_names=src_u,
        rater_canonical=[c[0] for c in rc],
        rater_expertise=[c[1] for c in rc],
        pos_rate=float(y.mean()),
    )
