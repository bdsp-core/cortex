"""Merge all bank-only (spike-domain) segments into eeg_bank_spec.h5 so the
side file becomes ONE unified EEG bank: every segment has EEG; the 68,000
IIIC segments additionally have a 10-min spectrogram.

Per-segment schema after this pass:
    /segments/<seg_id>/
        data<N>s   (n_ch, n_samples)  the EEG window; N = round(duration_s)
                   attrs: fs_hz, channel_names
        sdata/sfreqs/stimes            present iff has_spectrogram
        group attrs:
            eeg_duration_s   float
            has_spectrogram  bool
            source_dataset   str   (for newly-merged spike segments)

Bank-only segments are copied verbatim (EEG + attrs) from eeg_bank.h5 with
has_spectrogram=False. Existing IIIC groups get has_spectrogram/eeg_duration_s
backfilled for a uniform schema.
"""
from __future__ import annotations
import os, time
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np

BANK = "/Volumes/Extreme SSD/eeg_bank.h5"
SPEC = "/Volumes/Extreme SSD/eeg_bank_spec.h5"


def main():
    t0 = time.perf_counter()
    bank = h5py.File(BANK, "r")
    spec = h5py.File(SPEC, "a")
    sg = spec.require_group("segments")

    bank_ids = set(int(k) for k in bank["segments"].keys())
    spec_ids = set(int(k) for k in sg.keys())
    to_add = sorted(bank_ids - spec_ids)
    print(f"bank={len(bank_ids):,}  spec={len(spec_ids):,}  to add={len(to_add):,}",
          flush=True)

    # 1) Backfill uniform attrs on existing spec groups.
    nb = 0
    for k in list(sg.keys()):
        g = sg[k]
        if "has_spectrogram" not in g.attrs:
            has = "sdata" in g
            g.attrs["has_spectrogram"] = bool(has)
            if "eeg_duration_s" not in g.attrs:
                # data30s => 30s; data10s already set its own
                g.attrs["eeg_duration_s"] = 30.0 if "data30s" in g else 10.0
            nb += 1
    print(f"backfilled attrs on {nb:,} existing groups", flush=True)

    # 2) Copy bank-only (spike-domain) segments in.
    added = 0
    for i, sid in enumerate(to_add, 1):
        bg = bank["segments"][str(sid)]
        if "data" not in bg:
            continue
        data = np.asarray(bg["data"], dtype=np.float32)
        a = dict(bg.attrs)
        fs = float(a.get("fs_hz", 200.0))
        dur = data.shape[1] / fs
        dsname = f"data{int(round(dur))}s"
        g = sg.require_group(str(sid))
        if dsname in g:
            del g[dsname]
        d = g.create_dataset(dsname, data=data, compression="gzip",
                              compression_opts=4, chunks=True)
        d.attrs["fs_hz"] = fs
        if "channel_names" in a:
            d.attrs["channel_names"] = a["channel_names"]
        g.attrs["eeg_duration_s"] = float(dur)
        g.attrs["has_spectrogram"] = False
        g.attrs["source_dataset"] = str(a.get("source_dataset", ""))
        added += 1
        if i % 5000 == 0:
            spec.flush()
            print(f"  [{i:6d}/{len(to_add)}] added={added:,}  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    spec.flush()
    total = len(sg.keys())
    bank.close(); spec.close()
    print(f"\nDone. added={added:,}  total groups now={total:,}  "
          f"wall={time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
