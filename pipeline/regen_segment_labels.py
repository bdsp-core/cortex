"""R1: regenerate data/labels/segment_labels.csv from the CURRENT canonical
labels.csv + segments.csv.

segment_labels.csv was byte-identical and STALE in both source repos (84,555
rows, built from the pre-Centaur/pre-Kong corpus) while labels/segments grew.
It is a pure derived view: build_segment_labels(labels_df, segments_df) (see
build_unified_labels.py:563) with no external/global inputs. We reuse that
EXACT function (zero logic drift) on the post-gold-ingest tables, so the
regenerated file is schema- and aggregation-faithful and includes
iiic_vote_other (the K=7 'other' task) for every segment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from build_unified_labels import build_segment_labels  # verbatim, pure

REPO = Path(__file__).resolve().parents[1]
LBL = REPO / "data" / "labels"
F_LABELS = LBL / "labels.csv"
F_SEGMENTS = LBL / "segments.csv"
F_OUT = LBL / "segment_labels.csv"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    print("=== R1: regenerate segment_labels.csv ===", flush=True)
    old_md5 = _md5(F_OUT) if F_OUT.exists() else None
    old_n = (sum(1 for _ in open(F_OUT)) - 1) if F_OUT.exists() else 0

    labels_df = pd.read_csv(F_LABELS, dtype=str, keep_default_na=False,
                            low_memory=False)
    segments_df = pd.read_csv(F_SEGMENTS, dtype=str, keep_default_na=False,
                              low_memory=False)
    n_seg = len(segments_df)
    print(f"  inputs: labels={len(labels_df):,}  segments={n_seg:,}")

    seg_labels = build_segment_labels(labels_df, segments_df)

    if len(seg_labels) != n_seg:
        raise SystemExit(
            f"ABORT: regenerated rows {len(seg_labels)} != segments {n_seg} "
            "(one row per segment expected)")
    if "iiic_vote_other" not in seg_labels.columns:
        raise SystemExit("ABORT: iiic_vote_other column missing (K=7 task)")

    seg_labels.to_csv(F_OUT, index=False)
    print(f"  STALE  : {old_n:,} rows  md5={old_md5}")
    print(f"  REGEN  : {len(seg_labels):,} rows  md5={_md5(F_OUT)}")
    print(f"  columns: {list(seg_labels.columns)}")
    # spot-check: a gold-bearing segment now reflects centaur_iiic_expert
    print("  DONE (one row per segment; schema = original "
          "build_segment_labels output)")


if __name__ == "__main__":
    main()
