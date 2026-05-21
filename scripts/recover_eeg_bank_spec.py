"""Salvage segments from a corrupted eeg_bank_spec.h5 side file.

Same pattern as recover_eeg_bank.py, but the side file has
data30s + sdata + sfreqs + stimes per group instead of just `data`.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/Users/mwestover/GithubRepos/ideal-test-multi")
SRC_DEFAULT = Path("/Volumes/Extreme SSD/eeg_bank_spec.h5")
DST_DEFAULT = Path("/Volumes/Extreme SSD/eeg_bank_spec_recovered.h5")

DATASETS = ("data30s", "sdata", "sfreqs", "stimes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=SRC_DEFAULT)
    p.add_argument("--dst", type=Path, default=DST_DEFAULT)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "data/iiic_precompute_manifest.csv")
    args = p.parse_args()

    man = pd.read_csv(args.manifest, low_memory=False, dtype={"notes": "str"})
    seg_ids = sorted(man["seg_id"].astype(int).tolist())
    print(f"Will try {len(seg_ids):,} seg_ids from manifest")

    n_ok = n_skip = n_fail = n_partial = 0
    t0 = time.perf_counter()
    with h5py.File(args.src, "r") as fsrc, h5py.File(args.dst, "w") as fdst:
        for k, v in fsrc.attrs.items():
            fdst.attrs[k] = v
        seg_dst = fdst.require_group("segments")
        for i, sid in enumerate(seg_ids, 1):
            path = f"/segments/{sid}"
            try:
                src_grp = fsrc[path]
            except KeyError:
                n_skip += 1
                continue
            except Exception:
                n_fail += 1
                continue
            try:
                # Need at least sdata + data30s (or just sdata for legacy
                # 7-seg smoke-test entries without data30s).
                if "sdata" not in src_grp:
                    n_skip += 1
                    continue
                dst_grp = seg_dst.require_group(str(sid))
                copied_any = False
                for name in DATASETS:
                    if name in src_grp:
                        arr = np.asarray(src_grp[name])
                        if name in dst_grp:
                            del dst_grp[name]
                        kwargs = {}
                        if arr.ndim >= 2:
                            kwargs = dict(compression="gzip",
                                          compression_opts=4, chunks=True)
                        dst_grp.create_dataset(name, data=arr, **kwargs)
                        copied_any = True
                for k, v in src_grp.attrs.items():
                    dst_grp.attrs[k] = v
                if copied_any:
                    if "data30s" in src_grp:
                        n_ok += 1
                    else:
                        n_partial += 1
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1
            if i % 1000 == 0:
                wall = time.perf_counter() - t0
                print(f"  [{i:6d}/{len(seg_ids):6d}]  ok={n_ok:6d} "
                      f"partial={n_partial:5d} skip={n_skip:6d} fail={n_fail:5d}  "
                      f"{wall:.0f}s", flush=True)
            if i % 5000 == 0:
                fdst.flush()
    print(f"\nDone. recovered_full={n_ok:,}  recovered_partial={n_partial:,}  "
          f"not-in-src={n_skip:,}  failed={n_fail:,}")
    print(f"  src: {args.src}  ({args.src.stat().st_size / 1e9:.3f} GB)")
    print(f"  dst: {args.dst}  ({args.dst.stat().st_size / 1e9:.3f} GB)")


if __name__ == "__main__":
    main()
