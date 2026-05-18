"""Phase 3.5 data prep: unified IIIC observation arrays for the joint fit.

For one IIIC one-vs-rest task, builds dense-indexed JAX-ready arrays from
the canonical data/labels/labels.csv, plus the two split-gauge anchor masks:
  * SCALE anchor  = segments in BOTH sparcnet50K AND kong:crowd (the dense
                    ~6,691-seg weld; pins the multiplicative scale b).
  * LOCATION anchor = the n=4 Centaur gold's 5,000 segments (pins the
                    additive location a).  Gold is an ANCHOR, never a
                    Youden expert population (3-agent adjudication).

UNIFORM 6-CLASS COLLAPSE (correctness fix; Phase-3 erratum reconciliation):
the IIIC label space is 6-class {seizure,lpd,gpd,lrda,grda,other}.  The
Centaur-novice source records the IIC-spectrum natively as 7-class
{...,bipd,birds}; per the project AUDIT section 6 + the Phase-1 decision
the canonical mapping is {bipd,birds} -> other.  Phase-3
build_calibration_inputs.py applied `value=='other'` only, so ~30,511
Centaur-novice bipd/birds reads were mis-labelled NEGATIVE for the `iic`
task across sources.  This module applies {bipd,birds}->other UNIFORMLY to
EVERY source before one-vs-rest binarisation, so the joint fit (and the
re-derived Youden) are cross-source consistent.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "data" / "labels" / "labels.csv"
RATERS = REPO / "data" / "labels" / "raters.csv"

IIIC_SOURCES = {
    "sparcnet50K", "pd_rda_profiler",
    "iiic_crowdsourcing:kong2025:crowd", "iiic_crowdsourcing:kong2025:expert",
    "centaur_2025_iiic", "centaur_iiic_expert",
}
# reference domain name -> the canonical 6-class pattern_class value
TASK_TO_CLASS = {
    "sz": "seizure", "lpd": "lpd", "gpd": "gpd",
    "lrda": "lrda", "grda": "grda", "iic": "other",
}
# uniform 6-class collapse applied to EVERY source
_COLLAPSE = {"bipd": "other", "birds": "other"}
SCALE_WELD_SRCS = ("sparcnet50K", "iiic_crowdsourcing:kong2025:crowd")
LOCATION_ANCHOR_SRC = "centaur_iiic_expert"


@dataclass
class TaskData:
    task: str
    n_obs: int
    seg_idx: np.ndarray      # (N,) dense 0..S-1
    rater_idx: np.ndarray    # (N,) dense 0..R-1
    src_idx: np.ndarray      # (N,) dense 0..D-1
    y: np.ndarray            # (N,) {0,1}
    n_seg: int
    n_rater: int
    n_src: int
    scale_anchor_seg: np.ndarray   # bool (S,) — sparcnet50K ∩ kong:crowd
    loc_anchor_seg: np.ndarray     # bool (S,) — centaur gold 5,000
    seg_ids: np.ndarray            # (S,) original unified seg_id
    rater_ids: np.ndarray          # (R,) original unified rater_id
    src_names: list                # (D,)
    rater_canonical: list          # (R,) canonical_name (R5-safe)
    rater_expertise: list          # (R,) expertise_level
    pos_rate: float


def _canon_value(v: str) -> str:
    v = v.strip()
    return _COLLAPSE.get(v, v)


def load_task(task: str, subsample: int | None = None,
              seed: int = 0) -> TaskData:
    """Build arrays for one IIIC one-vs-rest task ('sz'|'lpd'|'gpd'|
    'lrda'|'grda'|'iic'). subsample: cap N (stratified by source) for
    smoke tests; None = full corpus."""
    if task not in TASK_TO_CLASS:
        raise ValueError(f"task {task!r} not in {list(TASK_TO_CLASS)}")
    pos_class = TASK_TO_CLASS[task]

    rmeta: dict[str, tuple[str, str]] = {}
    with open(RATERS) as f:
        for d in csv.DictReader(f):
            rmeta[d["rater_id"]] = (
                (d.get("canonical_name") or f"rater_{d['rater_id']}"),
                (d.get("expertise_level") or "").strip() or "untiered",
            )

    seg_in_src: dict[str, set] = {s: set() for s in SCALE_WELD_SRCS}
    gold_seg: set = set()
    rows = []
    with open(LABELS) as f:
        for d in csv.DictReader(f):
            if d["label_type"] != "pattern_class":
                continue
            sd = d["source_dataset"]
            if sd not in IIIC_SOURCES:
                continue
            seg = d["seg_id"]
            if sd in seg_in_src:
                seg_in_src[sd].add(seg)
            if sd == LOCATION_ANCHOR_SRC:
                gold_seg.add(seg)
            y = 1 if _canon_value(d["value"]) == pos_class else 0
            rows.append((seg, d["rater_id"], sd, y))

    if subsample and len(rows) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=subsample, replace=False)
        rows = [rows[i] for i in idx]

    weld = seg_in_src[SCALE_WELD_SRCS[0]] & seg_in_src[SCALE_WELD_SRCS[1]]

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
    scale_mask = np.array([seg_u[i] in weld for i in range(S)], bool)
    loc_mask = np.array([seg_u[i] in gold_seg for i in range(S)], bool)
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
