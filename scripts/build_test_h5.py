"""Build a canonical test set of 200 H5 files: 100 IIIC + 100 spike.

Layout (one H5 per case, flat directory):
    data/test_h5/
        test_iiic_seizure_<seg_id>.h5  ×17
        test_iiic_lpd_<seg_id>.h5      ×16
        test_iiic_gpd_<seg_id>.h5      ×17
        test_iiic_lrda_<seg_id>.h5     ×16
        test_iiic_grda_<seg_id>.h5     ×17
        test_iiic_other_<seg_id>.h5    ×17
        test_spike_<seg_id>.h5         ×100

Canonical per-case schema (gzip-compressed, h5py default chunks):
    /eeg30s      (n_ch, n_samples_30s) float32   central 30-sec EEG
        attrs: fs_hz (float), channel_names (S16 array)

  For IIIC cases also:
    /sdata       (n_times, n_freqs*4) float32    10-min multitaper spec
                                                 (LL, RL, LP, RP regions stacked)
    /sfreqs      (n_freqs,) float32
    /stimes      (n_times,) float32

  Root attrs (all):
    seg_id (int), source_dataset (str), test_class (str),
    n_raters (int)
  Root attrs (IIIC only):
    pattern_class (str), spec_source (str),
    source_window_center_s (float), sdata_duration_s (float)
  Root attrs (spike only):
    spike_category (str)

Sources:
    IIIC: read from /Volumes/Extreme SSD/eeg_bank_spec.h5 (precomputed
          30-sec EEG + 10-min multitaper from precompute_iiic.py).
    Spike: read 30-sec EEG from /Volumes/Extreme SSD/eeg_bank.h5
          (morgoth1:spikes segs, fs=128, shape (20, 3840)).

Deterministic sampling (seed=42).
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/Users/mwestover/GithubRepos/ideal-test-multi")
BANK = Path("/Volumes/Extreme SSD/eeg_bank.h5")
SPEC = Path("/Volumes/Extreme SSD/eeg_bank_spec.h5")
OUT_DIR = ROOT / "data/test_h5"

IIIC_CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
IIIC_PER_CLASS = {"seizure": 17, "lpd": 16, "gpd": 17,
                   "lrda": 16, "grda": 17, "other": 17}  # sums to 100
N_SPIKE = 100
RNG_SEED = 42


def load_labels():
    print("Loading labels.csv.gz ...")
    lab = pd.read_csv(ROOT / "data/labels/labels.csv.gz", low_memory=False)
    iiic = lab[lab["label_type"] == "pattern_class"]
    iiic_plurality = (iiic.groupby("seg_id")["value"]
                       .agg(lambda s: s.value_counts().idxmax()))
    iiic_n = iiic.groupby("seg_id").size()
    sp = lab[lab["label_type"] == "spike"]
    sp_plurality = (sp.groupby("seg_id")["value"]
                      .agg(lambda s: s.value_counts().idxmax()))
    sp_n = sp.groupby("seg_id").size()
    return iiic_plurality, iiic_n, sp_plurality, sp_n


def sample_iiic_seg_ids(iiic_plurality, iiic_n, available_in_spec, rng):
    """Return {class: [seg_id...]} sampling from segs available in side file."""
    out = {}
    for cls in IIIC_CLASSES:
        cls_segs = set(iiic_plurality[iiic_plurality == cls].index.astype(int))
        avail = sorted(cls_segs & available_in_spec)
        k = IIIC_PER_CLASS[cls]
        if len(avail) < k:
            raise SystemExit(f"Need {k} {cls} segs in side file, only {len(avail)} available")
        # Sort by n_raters descending, then take a random subsample from the
        # top half so we pick well-labeled examples.
        avail_arr = np.array(avail)
        nrt = iiic_n.reindex(avail_arr).fillna(0).astype(int).values
        order = np.argsort(-nrt, kind="stable")
        top_pool = avail_arr[order][: max(k * 3, 50)]
        chosen = rng.choice(top_pool, size=k, replace=False)
        out[cls] = sorted(int(s) for s in chosen)
    return out


def sample_spike_seg_ids(sp_plurality, sp_n, available_in_bank,
                          source_lookup, rng):
    """Sample 100 spike-positive segs from morgoth1:spikes in the main bank."""
    pos = sp_plurality[sp_plurality.astype(str) == "1"].index.astype(int)
    in_bank = sorted(set(int(s) for s in pos) & available_in_bank)
    # Restrict to morgoth1:spikes (only positive-spike source in bank)
    in_bank = [s for s in in_bank if source_lookup.get(s) == "morgoth1:spikes"]
    if len(in_bank) < N_SPIKE:
        raise SystemExit(f"Need {N_SPIKE} spike segs, only {len(in_bank)} available")
    arr = np.array(in_bank)
    nrt = sp_n.reindex(arr).fillna(0).astype(int).values
    order = np.argsort(-nrt, kind="stable")
    top_pool = arr[order][: N_SPIKE * 3]
    chosen = rng.choice(top_pool, size=N_SPIKE, replace=False)
    return sorted(int(s) for s in chosen)


def write_iiic_case(out_path, seg_id, cls, n_raters, seg_row, fspec):
    """Write one IIIC test H5 from side-file entry."""
    g = fspec[f"segments/{seg_id}"]
    eeg30 = np.asarray(g["data30s"], dtype=np.float32)
    sdata = np.asarray(g["sdata"], dtype=np.float32)
    sfreqs = np.asarray(g["sfreqs"], dtype=np.float32)
    stimes = np.asarray(g["stimes"], dtype=np.float32)
    attrs = dict(g.attrs)
    fs_hz = float(attrs.get("fs_hz30s", 200.0))
    ch_names = attrs.get("channel_names30s")
    if ch_names is not None:
        ch_names = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in ch_names]
    else:
        ch_names = []

    with h5py.File(out_path, "w") as f:
        d = f.create_dataset("eeg30s", data=eeg30,
                              compression="gzip", compression_opts=4, chunks=True)
        d.attrs["fs_hz"] = fs_hz
        d.attrs["channel_names"] = np.array(ch_names, dtype="S16")
        f.create_dataset("sdata", data=sdata,
                          compression="gzip", compression_opts=4, chunks=True)
        f.create_dataset("sfreqs", data=sfreqs)
        f.create_dataset("stimes", data=stimes)
        f.attrs["seg_id"] = int(seg_id)
        f.attrs["test_class"] = "iiic"
        f.attrs["pattern_class"] = cls
        f.attrs["n_raters"] = int(n_raters)
        f.attrs["source_dataset"] = str(seg_row.get("source_dataset", ""))
        # PHI omitted: MRN/patient_id is a direct identifier — never written
        # into vendored H5 attrs. Track via seg_id only; resolve MRN locally
        # through data/labels/segments.csv when needed (sibling-repo only).
        if "spec_source" in attrs:
            f.attrs["spec_source"] = str(attrs["spec_source"])
        if "source_window_center_s" in attrs:
            f.attrs["source_window_center_s"] = float(attrs["source_window_center_s"])
        f.attrs["sdata_duration_s"] = float(stimes[-1] - stimes[0]) if len(stimes) > 1 else 0.0


def write_spike_case(out_path, seg_id, category, n_raters, seg_row, fbank):
    """Write one spike test H5 from main bank."""
    g = fbank[f"segments/{seg_id}"]
    eeg = np.asarray(g["data"], dtype=np.float32)
    attrs = dict(g.attrs)
    fs_hz = float(attrs.get("fs_hz", 128.0))
    ch_names = attrs.get("channel_names")
    if ch_names is not None:
        ch_names = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in ch_names]
    else:
        ch_names = []

    with h5py.File(out_path, "w") as f:
        d = f.create_dataset("eeg30s", data=eeg,
                              compression="gzip", compression_opts=4, chunks=True)
        d.attrs["fs_hz"] = fs_hz
        d.attrs["channel_names"] = np.array(ch_names, dtype="S16")
        f.attrs["seg_id"] = int(seg_id)
        f.attrs["test_class"] = "spike"
        f.attrs["spike_category"] = str(category)
        f.attrs["n_raters"] = int(n_raters)
        f.attrs["source_dataset"] = str(seg_row.get("source_dataset", ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--clean", action="store_true",
                    help="Wipe out-dir before building")
    args = p.parse_args()

    if args.clean and args.out.exists():
        print(f"Cleaning {args.out}")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    seg = pd.read_csv(ROOT / "data/labels/segments.csv", low_memory=False)
    seg_idx = seg.set_index("seg_id")
    source_lookup = seg.set_index("seg_id")["source_dataset"].to_dict()
    iiic_plurality, iiic_n, sp_plurality, sp_n = load_labels()

    with h5py.File(SPEC, "r") as fs:
        spec_segs = set(int(k) for k in fs.get("segments", {}).keys())
    with h5py.File(BANK, "r") as fb:
        bank_segs = set(int(k) for k in fb.get("segments", {}).keys())
    print(f"  side file has {len(spec_segs):,} IIIC segs ready")
    print(f"  main bank has {len(bank_segs):,} segs total")

    rng = np.random.default_rng(RNG_SEED)
    iiic_choice = sample_iiic_seg_ids(iiic_plurality, iiic_n, spec_segs, rng)
    spike_choice = sample_spike_seg_ids(sp_plurality, sp_n, bank_segs,
                                         source_lookup, rng)

    print("\nSelected IIIC seg_ids per class:")
    for cls in IIIC_CLASSES:
        print(f"  {cls:8s} (n={len(iiic_choice[cls])}): {iiic_choice[cls]}")
    print(f"\nSelected {len(spike_choice)} spike seg_ids (first 10): "
          f"{spike_choice[:10]}")

    n_written = 0
    print("\nWriting IIIC test H5 files...")
    with h5py.File(SPEC, "r") as fspec:
        for cls in IIIC_CLASSES:
            for sid in iiic_choice[cls]:
                out_path = args.out / f"test_iiic_{cls}_{sid}.h5"
                row = seg_idx.loc[sid] if sid in seg_idx.index else {}
                write_iiic_case(out_path, sid, cls,
                                 int(iiic_n.get(sid, 0)), row, fspec)
                n_written += 1
    print(f"  wrote {n_written} IIIC test H5 files")

    print("\nWriting spike test H5 files...")
    n0 = n_written
    with h5py.File(BANK, "r") as fbank:
        for sid in spike_choice:
            out_path = args.out / f"test_spike_{sid}.h5"
            row = seg_idx.loc[sid] if sid in seg_idx.index else {}
            write_spike_case(out_path, sid, "1",
                              int(sp_n.get(sid, 0)), row, fbank)
            n_written += 1
    print(f"  wrote {n_written - n0} spike test H5 files")

    total_size = sum(p.stat().st_size for p in args.out.glob("*.h5"))
    print(f"\nTotal: {n_written} files, {total_size/1e6:.1f} MB at {args.out}")


if __name__ == "__main__":
    main()
