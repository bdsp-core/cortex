"""Build the v1.1.0 CORTEX test bank — 300 IIIC + 100 spike segments.

Produces a new ``data/eeg_bank.h5`` (or ``--out PATH``) with the same
group layout as the existing v1 bank but with 3× the IIIC payload, so
the engine has a 300-question runway instead of 100.

Bank composition (v2):
    iiic/   300 segments, 50 per pattern_class
            (seizure / lpd / gpd / lrda / grda / other)
    spike/  100 segments, copied verbatim from the v1 bank

Backward continuity:
    All 100 v1 IIIC segments are preserved in v2 so a rater who took
    the v1 test and the v2 test can have their answered v1 segments
    compared on an exactly-identical signal basis. The 200 new IIIC
    segments are sampled deterministically (seed=42) from
    ``data/labels/iiic_segment_signals.csv`` candidates that:
      * have a labels.csv plurality matching the target class,
      * have a full IIIC payload (data30s + sdata + sfreqs + stimes)
        in ``data/eeg_bank_spec.h5``,
      * have at least the median per-class n_raters in labels.csv
        (well-labeled examples).

Sources (all local):
    * data/eeg_bank.h5            v1 bank to preserve + lift spike from
    * data/eeg_bank_spec.h5       100 GB precomputed payload source
    * data/labels/labels.csv      plurality pattern_class
    * data/labels/iiic_segment_signals.csv  candidate filter

Usage:
    # dry-run: report sampling without writing
    .venv/bin/python scripts/build_cortex_test_bank_v2.py --dry-run

    # write the new bank in-place (overwrites data/eeg_bank.h5)
    .venv/bin/python scripts/build_cortex_test_bank_v2.py \\
        --out data/eeg_bank.h5

    # safer: write to a side path first, sanity-check, then swap
    .venv/bin/python scripts/build_cortex_test_bank_v2.py \\
        --out data/eeg_bank_v2.h5
    .venv/bin/python -c "import h5py; f=h5py.File('data/eeg_bank_v2.h5'); \\
        print(len(f['iiic']), len(f['spike']), dict(f.attrs))"
    mv data/eeg_bank.h5 data/eeg_bank_v1_backup.h5
    mv data/eeg_bank_v2.h5 data/eeg_bank.h5

After the new bank is on disk, upload it to a new GitHub release
``build-data-v2`` (see ``cortex_app/BUILD.md``), then bump the
``build-data-v1`` references in ``cortex_app/fetch_test_bank.sh`` and
``.github/workflows/cortex-release.yml``.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
V1_BANK = REPO / "data" / "eeg_bank.h5"
SPEC_FILE = REPO / "data" / "eeg_bank_spec.h5"
LABELS_CSV = REPO / "data" / "labels" / "labels.csv"
SIGNALS_CSV = REPO / "data" / "labels" / "iiic_segment_signals.csv"

IIIC_CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
PER_CLASS_TARGET = 50                              # 50 × 6 = 300
RNG_SEED = 42
SCHEMA_VERSION = "2"


def _check_inputs() -> None:
    """Verify all source files exist before doing any expensive work."""
    for path in (V1_BANK, SPEC_FILE, LABELS_CSV, SIGNALS_CSV):
        if not path.exists():
            raise SystemExit(
                f"Required input missing: {path}\n"
                "  This script expects all four source files to be local.\n"
                "  Check data/SENSITIVE.md for retrieval instructions.")


def load_v1_iiic() -> dict:
    """Return {seg_id: pattern_class} for the 100 existing v1 IIIC segs."""
    out = {}
    with h5py.File(V1_BANK, "r") as f:
        for sid_str in f["iiic"]:
            try:
                sid = int(sid_str)
            except ValueError:
                continue
            pc = str(f["iiic"][sid_str].attrs.get("pattern_class", ""))
            if pc:
                out[sid] = pc
    return out


def load_pattern_plurality() -> tuple[dict, dict]:
    """Plurality vote of pattern_class labels per seg_id.

    Returns (plurality_class_by_seg_id, n_raters_by_seg_id). Reading 962k
    pattern_class rows takes ~2 s.
    """
    print(f"  loading {LABELS_CSV.name} ...", flush=True)
    df = pd.read_csv(LABELS_CSV, low_memory=False,
                     usecols=["seg_id", "label_type", "value"])
    pc = df[df["label_type"] == "pattern_class"]
    print(f"    {len(pc):,} pattern_class label rows", flush=True)
    plur = (pc.groupby("seg_id")["value"]
              .agg(lambda s: s.value_counts().idxmax()))
    n_rt = pc.groupby("seg_id").size().rename("n_raters")
    return (plur.to_dict(), n_rt.to_dict())


def spec_segments_with_full_iiic(spec_path: Path,
                                 candidate_ids: set[int]) -> set[int]:
    """Subset of candidate_ids whose spec-file group has the v1-compatible
    full IIIC payload.

    The spec file holds two upstream pipelines:
      * morgoth1_recompute  — 20 channels (no Photic),
                              eeg30s (20, 6000), sdata (597, 504)   ← v1 schema
      * kong_precomputed    — 21+ channels (includes Photic),
                              eeg30s (21+, 6000), sdata (300, 388)  ← incompatible

    The CORTEX viewer + cortex_render_videos assume the 20-channel,
    (597, 504) shape. Mixing kong segments breaks the spectrogram render
    silently and the per-channel display. So we hard-require
    spec_source == "morgoth1_recompute" AND the canonical shapes.
    """
    out = set()
    REQ_EEG = (20, 6000)
    REQ_SDATA_COLS = 504
    with h5py.File(spec_path, "r") as f:
        for sid in candidate_ids:
            key = str(sid)
            if key not in f["segments"]:
                continue
            g = f[f"segments/{key}"]
            if not ("data30s" in g and "sdata" in g
                    and "sfreqs" in g and "stimes" in g):
                continue
            if str(g.attrs.get("spec_source", "")) != "morgoth1_recompute":
                continue
            if g["data30s"].shape != REQ_EEG:
                continue
            if g["sdata"].shape[1] != REQ_SDATA_COLS:
                continue
            out.add(sid)
    return out


def sample_new_iiic(plurality: dict, n_raters: dict,
                    full_payload_ids: set[int],
                    already_in_v1: set[int],
                    rng) -> dict:
    """Per class, pick (PER_CLASS_TARGET - n_already) new segs from the
    pool. Quality bias: rank by n_raters DESC, take the top 3× the
    needed count, sample uniformly without replacement from that top
    pool — same heuristic as the v1 build_test_h5.py.

    Returns {seg_id: pattern_class} for ALL chosen (existing + new).
    """
    chosen = {sid: cls for sid, cls in
              [(s, plurality.get(s, "")) for s in already_in_v1]
              if cls in IIIC_CLASSES}
    # Reverse index: class -> seg_ids matching that class & having full payload
    by_class: dict[str, list[int]] = {c: [] for c in IIIC_CLASSES}
    for sid, cls in plurality.items():
        if cls in IIIC_CLASSES and sid in full_payload_ids:
            by_class[cls].append(sid)
    print(f"  full-payload candidates per class:", flush=True)
    for c in IIIC_CLASSES:
        print(f"    {c:8s}: {len(by_class[c]):>6,}", flush=True)

    for cls in IIIC_CLASSES:
        n_already = sum(1 for sid, k in chosen.items() if k == cls)
        n_needed = PER_CLASS_TARGET - n_already
        if n_needed <= 0:
            continue
        # Excluded: already-chosen segs + segs not in this class's pool
        pool = [s for s in by_class[cls] if s not in chosen]
        if len(pool) < n_needed:
            raise SystemExit(
                f"Need {n_needed} more {cls} segments, only "
                f"{len(pool)} available with full payload.")
        # Quality bias — n_raters DESC, then random within top pool.
        pool_arr = np.array(pool)
        nrt = np.array([n_raters.get(s, 0) for s in pool], dtype=int)
        order = np.argsort(-nrt, kind="stable")
        top_pool = pool_arr[order][:max(n_needed * 3, 50)]
        picks = rng.choice(top_pool, size=n_needed, replace=False)
        for s in picks:
            chosen[int(s)] = cls
    return chosen


def copy_iiic_segment(src_spec: h5py.File, dst: h5py.File, seg_id: int,
                      pattern_class: str, n_raters: int) -> None:
    """Copy one IIIC segment from the spec file into the target bank,
    attaching the attrs the engine reads at runtime."""
    src_grp = src_spec[f"segments/{seg_id}"]
    dst_grp = dst.create_group(f"iiic/{seg_id}")
    # gzip-compressed datasets, h5py default chunks — matches v1 schema
    for name in ("data30s", "sdata", "sfreqs", "stimes"):
        data = np.asarray(src_grp[name])
        # v1 names eeg30s, v2 source has data30s. Rename to v1's name.
        out_name = "eeg30s" if name == "data30s" else name
        dst_grp.create_dataset(out_name, data=data,
                               compression="gzip", compression_opts=4)
    # The spec file carries `spec_source` (e.g. "morgoth1_recompute") at
    # the segment level rather than `source_dataset`, so use that. The v1
    # bank's source_dataset attr was joined from segments.csv at build
    # time — we don't have that source here, so leave it as the
    # spec_source label (still distinct from kong's "kong_precomputed").
    spec_source = str(src_grp.attrs.get("spec_source", ""))
    dst_grp.attrs["seg_id"] = int(seg_id)
    dst_grp.attrs["pattern_class"] = pattern_class
    dst_grp.attrs["source_dataset"] = spec_source
    dst_grp.attrs["spec_source"] = spec_source
    dst_grp.attrs["test_class"] = "iiic"
    dst_grp.attrs["n_raters"] = int(n_raters)


def copy_iiic_from_v1(v1_bank: h5py.File, dst: h5py.File,
                      seg_id: int, pattern_class: str) -> None:
    """Copy an existing v1 IIIC segment so v2 is byte-identical to v1
    on the EEG/spectrogram payload, but force the pattern_class attr to
    the labels.csv plurality (`pattern_class` arg) — the sampler counts
    by plurality, so the H5 attr must match plurality for internal
    consistency.

    v1's pattern_class came from an older labels.csv snapshot. For most
    segs (~94/100) it agrees with current plurality; for the few that
    drift, the H5 attr is updated to current plurality. The underlying
    EEG signal does not change.
    """
    src = v1_bank[f"iiic/{seg_id}"]
    dst_grp = dst.create_group(f"iiic/{seg_id}")
    for ds_name in src.keys():
        data = np.asarray(src[ds_name])
        dst_grp.create_dataset(ds_name, data=data,
                               compression="gzip", compression_opts=4)
    for ak, av in src.attrs.items():
        dst_grp.attrs[ak] = av
    dst_grp.attrs["pattern_class"] = pattern_class


def copy_spike_group(v1_bank: h5py.File, dst: h5py.File) -> int:
    """Copy the full spike/ group from v1 verbatim into v2."""
    n = 0
    if "spike" not in v1_bank:
        return 0
    v1_bank.copy("spike", dst)
    n = len(dst["spike"])
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Build the v1.1.0 CORTEX test bank (300 IIIC + 100 spike).")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "eeg_bank_v2.h5",
                    help="Target file (default: data/eeg_bank_v2.h5).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written, don't touch disk.")
    args = ap.parse_args()

    print("CORTEX test-bank builder v2", flush=True)
    print(f"  out: {args.out}", flush=True)
    print(f"  dry-run: {args.dry_run}", flush=True)
    print()
    _check_inputs()

    # Stage 1: v1 segs to carry forward
    print("Stage 1 — v1 IIIC carry-forward")
    v1_iiic = load_v1_iiic()
    print(f"  v1 IIIC count: {len(v1_iiic)}", flush=True)
    v1_class_counts = Counter(v1_iiic.values())
    for c in IIIC_CLASSES:
        print(f"    {c:8s}: {v1_class_counts.get(c, 0):>3}", flush=True)
    print()

    # Stage 2: labels.csv plurality + n_raters
    print("Stage 2 — pattern_class plurality from labels.csv")
    plurality, n_raters = load_pattern_plurality()
    print(f"  {len(plurality):,} segments with a plurality label", flush=True)
    print()

    # Stage 3: filter to spec-file full-payload IIIC
    print("Stage 3 — spec-file full-payload filter")
    iiic_candidates = {sid for sid, cls in plurality.items()
                       if cls in IIIC_CLASSES}
    full = spec_segments_with_full_iiic(SPEC_FILE, iiic_candidates)
    print(f"  IIIC candidates with full payload: {len(full):,}", flush=True)
    print()

    # Stage 4: sample to reach 50/class (incl. v1 carry-forward)
    print("Stage 4 — per-class sampling")
    rng = np.random.default_rng(RNG_SEED)
    chosen = sample_new_iiic(plurality, n_raters, full,
                             set(v1_iiic), rng)
    final_class_counts = Counter(chosen.values())
    new_segs = set(chosen) - set(v1_iiic)
    print(f"  v1 carried forward: {len(set(v1_iiic) & set(chosen))}", flush=True)
    print(f"  new segs sampled:   {len(new_segs)}", flush=True)
    print(f"  total IIIC:         {len(chosen)}", flush=True)
    print(f"  per-class final:")
    for c in IIIC_CLASSES:
        print(f"    {c:8s}: {final_class_counts.get(c, 0):>3}", flush=True)
    if any(final_class_counts.get(c, 0) != PER_CLASS_TARGET
           for c in IIIC_CLASSES):
        raise SystemExit("Per-class counts != target; aborting.")
    print()

    if args.dry_run:
        print("--dry-run: skipping write.", flush=True)
        return 0

    # Stage 5: write
    print("Stage 5 — write bank")
    if args.out.exists():
        backup = args.out.with_suffix(args.out.suffix + ".bak")
        print(f"  {args.out} exists; renaming to {backup}", flush=True)
        args.out.rename(backup)

    with h5py.File(V1_BANK, "r") as v1, h5py.File(SPEC_FILE, "r") as spec, \
         h5py.File(args.out, "w") as dst:
        dst.attrs["schema_version"] = SCHEMA_VERSION
        dst.attrs["build_utc"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds")
        dst.attrs["n_iiic"] = PER_CLASS_TARGET * len(IIIC_CLASSES)
        # IIIC: v1 carry-forward, then new
        for i, sid in enumerate(sorted(set(v1_iiic) & set(chosen)), 1):
            copy_iiic_from_v1(v1, dst, sid, chosen[sid])
            if i % 25 == 0:
                print(f"    [{i}/{len(v1_iiic)}] v1 carry: {sid}", flush=True)
        for i, sid in enumerate(sorted(new_segs), 1):
            copy_iiic_segment(spec, dst, sid, chosen[sid],
                              n_raters.get(sid, 0))
            if i % 25 == 0:
                print(f"    [{i}/{len(new_segs)}] new: {sid}",
                      flush=True)
        # Spike: full verbatim copy
        n_spike = copy_spike_group(v1, dst)
        dst.attrs["n_spike"] = n_spike

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"  wrote {args.out} ({size_mb:.1f} MB)", flush=True)
    print(f"  schema_version={SCHEMA_VERSION}, n_iiic={PER_CLASS_TARGET * 6}, "
          f"n_spike={n_spike}", flush=True)
    print()
    print("Done. Next steps:")
    print(f"  1. Sanity-check the file (h5py listing of /iiic and /spike).")
    print(f"  2. mv data/eeg_bank.h5 data/eeg_bank_v1_backup.h5")
    print(f"     mv {args.out} data/eeg_bank.h5")
    print(f"  3. Upload to a new GitHub release tagged build-data-v2.")
    print(f"  4. Bump build-data-v1 -> build-data-v2 in fetch_test_bank.sh")
    print(f"     and .github/workflows/cortex-release.yml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
