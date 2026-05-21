"""Pre-compute morgoth-style 10-min spectrograms for selected bank segments.

For each target segment, re-pulls the full 10-min source .mat from S3,
applies morgoth's compute_features filter chain (3rd-order Butterworth
0.5-70 Hz bandpass + 60 Hz iirnotch, both via filtfilt), builds the
clean 18-channel bipolar montage, computes per-channel spectrograms with
`scipy.signal.spectrogram` (4-sec window, 1-sec step, 0.5-25 Hz, PSD
density), averages within the 4 IIIC regions (LL, RL, LP, RP), and
writes the result back into the segment's group in the EEG bank H5:

    /segments/<seg_id>/
        sdata    (n_times, n_freqs*4)   float32, gzip-4   morgoth layout
        sfreqs   (n_freqs,)              float32
        stimes   (n_times,)              float32

The viewer (eeg_bank_viewer.py) already auto-detects these datasets and
displays them in place of an on-the-fly compute.

Sources supported (must be one of):
    sparcnet50K            morgoth1:* IIIC subdir, 10-min @ 200 Hz
    centaur_2025_iiic      morgoth1:* IIIC, 10-min @ 200 Hz
    morgoth1:spikes        10-min @ 200 Hz
    morgoth1:bets/bird/posts/vw/wickets   10-min @ 200 Hz
    iiic_crowdsourcing:kong2025   50-sec contest H5  (no 10-min context)

Kong segments only have 50-sec EEG available — we still compute a
spectrogram (50-sec context, fewer time bins) and store it for
completeness.

USAGE
    python3 scripts/precompute_spectrograms.py --dry-run --max 10
    python3 scripts/precompute_spectrograms.py --max 10
    python3 scripts/precompute_spectrograms.py --seg-ids 28236 53271 55715
    python3 scripts/precompute_spectrograms.py --sources sparcnet50K --max 100
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
from scipy import signal as sig

# Re-use the loading machinery from build_eeg_bank
ROOT = Path("/Users/mwestover/GithubRepos/ideal-test-multi")
sys.path.insert(0, str(ROOT / "scripts"))
import build_eeg_bank as beb  # noqa: E402

# Same constants the viewer uses
LABELS_DIR = ROOT / "data/labels"
BANK_PATH = Path("/Volumes/Extreme SSD/eeg_bank.h5")
# Spectrograms live in a SEPARATE side-file so we never need to write back
# into the main bank (which has corrupted twice from concurrent r+w).
SPEC_PATH = Path("/Volumes/Extreme SSD/eeg_bank_spec.h5")


# Channel names + bipolar map (mirror eeg_bank_viewer / morgoth's _fcn_bipolar)
CHANNELS_19 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
               'Fz', 'Cz', 'Pz',
               'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']


def apply_bipolar_clean(data, channel_names):
    ch_to_idx = {nm: i for i, nm in enumerate(channel_names) if i < data.shape[0]}
    n = data.shape[1]
    bp = np.zeros((18, n), dtype=data.dtype)
    def d(a, b):
        if a in ch_to_idx and b in ch_to_idx:
            return data[ch_to_idx[a]] - data[ch_to_idx[b]]
        return np.zeros(n, dtype=data.dtype)
    bp[0] = d('Fp1','F7'); bp[1] = d('F7','T3'); bp[2] = d('T3','T5'); bp[3] = d('T5','O1')
    bp[4] = d('Fp2','F8'); bp[5] = d('F8','T4'); bp[6] = d('T4','T6'); bp[7] = d('T6','O2')
    bp[8] = d('Fp1','F3'); bp[9] = d('F3','C3'); bp[10] = d('C3','P3'); bp[11] = d('P3','O1')
    bp[12] = d('Fp2','F4'); bp[13] = d('F4','C4'); bp[14] = d('C4','P4'); bp[15] = d('P4','O2')
    bp[16] = d('Fz','Cz'); bp[17] = d('Cz','Pz')
    return bp


def butter_band_filtfilt(x, lo, hi, fs, order=3):
    nyq = 0.5 * fs
    hi = min(hi, nyq - 1.0)   # clamp below Nyquist for low-fs sources (e.g. 128 Hz)
    b, a = sig.butter(order, [lo / nyq, hi / nyq], btype='band')
    return sig.filtfilt(b, a, x, axis=-1)


def iirnotch_filtfilt(x, freq, fs, Q=30):
    b, a = sig.iirnotch(freq, Q, fs)
    return sig.filtfilt(b, a, x, axis=-1)


def compute_spectrogram_one_channel(x, fs, window_size=4.0, step_size=1.0,
                                      NW=3, K=5, fpass=(0.0, 55.0)):
    """Multitaper spectrogram, matching morgoth's MATLAB Chronux
    `mtspecgram_jj.m` (the actual pipeline that produced `sdata` in
    morgoth's H5 files).

    Per-window:
      * detrend
      * apply each of K DPSS tapers (time-half-bandwidth = NW)
      * |FFT|^2, scaled to PSD (density)
      * mean across K tapers → reduces variance ~K× vs single-taper STFT

    Defaults: NW=3, K=5, window=4 s, step=1 s, fpass=0-55 Hz, NFFT =
    next power of 2 ≥ window samples (Chronux convention).
    """
    Nwin = int(window_size * fs)
    Nstep = int(step_size * fs)
    if Nwin > len(x):
        return None, None, None
    NFFT = 1 << int(np.ceil(np.log2(Nwin)))   # next power of 2
    # DPSS tapers: shape (K, Nwin)
    tapers = sig.windows.dpss(Nwin, NW=NW, Kmax=K)
    # Window start indices
    starts = np.arange(0, len(x) - Nwin + 1, Nstep)
    n_win = len(starts)
    if n_win == 0:
        return None, None, None
    freqs_all = np.fft.rfftfreq(NFFT, d=1.0 / fs)
    fmask = (freqs_all >= fpass[0]) & (freqs_all <= fpass[1])
    freqs = freqs_all[fmask]
    n_freq = len(freqs)
    # Pre-allocate
    S = np.empty((n_freq, n_win), dtype=np.float64)
    for i, ws in enumerate(starts):
        seg = x[ws:ws + Nwin]
        seg = sig.detrend(seg, type='linear')
        # Multitaper: shape (K, NFFT//2+1)
        tapered = seg[None, :] * tapers       # (K, Nwin)
        # FFT along last axis (samples)
        F = np.fft.rfft(tapered, n=NFFT, axis=-1)   # (K, n_freq_all)
        psd = (np.abs(F) ** 2) / fs           # density scaling (Chronux)
        S[:, i] = psd[:, fmask].mean(axis=0)  # mean across tapers
    # Midpoint times (Chronux convention: winstart + Nwin/2)
    times = (starts + Nwin // 2) / fs
    return S, freqs, times


def compute_sdata(eeg, fs, fpass_compute=(0.0, 55.0), fpass_store=(0.5, 25.0)):
    """morgoth-MATLAB-style: filter → bipolar → multitaper spectrogram per
    region (mean of 4 bipolar channels) → stack as (n_times, n_freqs*4)
    in LL, RL, LP, RP order.

      fpass_compute=(0, 55) Hz — matches Chronux's mtspecgram_jj defaults.
      fpass_store=(0.5, 25) Hz — what the viewer displays (matches the
                                   morgoth python compute_features.py output).

    Returns (sdata, sfreqs, stimes) or (None, None, None).
    """
    eeg = np.nan_to_num(eeg, nan=0.0).astype(np.float64)
    eeg = butter_band_filtfilt(eeg, 0.5, 70.0, fs, order=3)
    eeg = iirnotch_filtfilt(eeg, 60.0, fs)
    bp = apply_bipolar_clean(eeg, CHANNELS_19)
    region_chans = {'LL': [0, 1, 2, 3], 'RL': [4, 5, 6, 7],
                     'LP': [8, 9, 10, 11], 'RP': [12, 13, 14, 15]}
    sfreqs = stimes = None
    region_means = {}
    for region, idxs in region_chans.items():
        Ssum = None
        for ch in idxs:
            S, f, t = compute_spectrogram_one_channel(bp[ch], fs,
                                                      fpass=fpass_compute)
            if S is None:
                return None, None, None
            if sfreqs is None:
                sfreqs, stimes = f, t
            Ssum = S if Ssum is None else Ssum + S
        region_means[region] = Ssum / len(idxs)
    # Slice to the display range
    keep = (sfreqs >= fpass_store[0]) & (sfreqs <= fpass_store[1])
    sfreqs = sfreqs[keep]
    region_means = {k: v[keep] for k, v in region_means.items()}
    # Stack region-by-region as (n_freqs*4, n_times), then transpose to
    # morgoth's on-disk format (n_times, n_freqs*4).
    stacked = np.vstack([region_means['LL'],
                          region_means['RL'],
                          region_means['LP'],
                          region_means['RP']])   # (n_freqs*4, n_times)
    sdata = stacked.T.astype(np.float32)
    return sdata, sfreqs.astype(np.float32), stimes.astype(np.float32)


# ───────────────────── source EEG fetchers ─────────────────────

def fetch_full_eeg_for_segment(row):
    """Pull the FULL 10-min (or 50-sec for Kong) EEG for a segment.
    Returns (data (n_channels, n_samples), fs, channel_names) or None.

    Re-uses build_eeg_bank's helpers: s3fs streaming for 10-min .mats
    (with IIIC subdir fallback for sparcnet50K), local cache for Kong.
    """
    sd = str(row["source_dataset"])
    s3_uri = row.get("s3_uri")
    if pd.isna(s3_uri) or not s3_uri:
        return None
    if sd == "iiic_crowdsourcing:kong2025":
        key = row.get("contest_h5_seg_key")
        if pd.isna(key) or not key:
            return None
        data, fs, ch, _ = beb.read_kong_segment(str(s3_uri), str(key))
        return data, fs, ch
    if sd == "pd_rda_profiler":
        # The iiic-freq3 bipolar copies are only 10-sec → no useful 10-min
        # context. Fall back to the morgoth1 IIIC raw (also 10-min) via the
        # s3_uri the segments table already holds.
        pass
    # 10-min .mat — stream with the same IIIC-subdir fix as build_eeg_bank
    effective_uri = str(s3_uri)
    if sd == "sparcnet50K":
        fn = Path(effective_uri).name
        correct_sd = beb.IIIC_FILE_INDEX.get(fn)
        if correct_sd is not None:
            effective_uri = (f"s3://bdsp-opendata-credentialed/morgoth1/data/"
                             f"internal_dataset/{correct_sd}/segments_raw/{fn}")
    # Stream the WHOLE file (not just central window) — that's the change
    # vs build_eeg_bank.stream_mat_central_window.
    bucket_key = effective_uri.removeprefix("s3://")
    fs_obj = beb._get_s3fs()
    try:
        with fs_obj.open(bucket_key, mode="rb") as fobj:
            with h5py.File(fobj, "r") as hf:
                if "data" not in hf or "Fs" not in hf:
                    return None
                full = np.asarray(hf["data"], dtype=np.float64)  # (samples, ch)
                fs = float(np.asarray(hf["Fs"]).flatten()[0])
                if full.shape[0] < full.shape[1]:
                    # already channels-first
                    data = full.astype(np.float32)
                else:
                    data = full.T.astype(np.float32)
                chs = beb._decode_mat_channels(hf, hf["channels"]) if "channels" in hf else CHANNELS_19[:data.shape[0]]
                return data, fs, chs
    except Exception as e:
        print(f"  stream failed for {effective_uri}: {e}", flush=True)
        return None


# ───────────────────── main ─────────────────────

PRECOMP_SOURCES = ("sparcnet50K", "centaur_2025_iiic", "morgoth1:spikes",
                    "morgoth1:bird", "morgoth1:vw", "morgoth1:bets",
                    "morgoth1:posts", "morgoth1:wickets",
                    "iiic_crowdsourcing:kong2025")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, default=BANK_PATH,
                    help="Main bank (read-only, source of seg_ids)")
    p.add_argument("--spec", type=Path, default=SPEC_PATH,
                    help="Side file for storing computed spectrograms")
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--sources", nargs="+", default=None,
                    help=f"subset of {PRECOMP_SOURCES}")
    p.add_argument("--seg-ids", nargs="+", type=int, default=None,
                    help="explicit seg_ids; overrides --max / --sources")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                    help="recompute even if sdata already exists in the group")
    args = p.parse_args()

    beb.CACHE.mkdir(parents=True, exist_ok=True)
    beb.build_iiic_index()  # populate IIIC_FILE_INDEX for sparcnet50K fallback

    # Pick segments
    seg = pd.read_csv(LABELS_DIR / "segments.csv", low_memory=False)
    if args.seg_ids:
        seg = seg[seg["seg_id"].astype(int).isin(args.seg_ids)]
        print(f"  filtering to {len(args.seg_ids)} explicit seg_ids → "
              f"{len(seg)} found in segments.csv")
    else:
        srcs = args.sources or PRECOMP_SOURCES
        seg = seg[seg["source_dataset"].isin(srcs) & seg["s3_uri"].notna()]
        seg = seg.head(args.max)
        print(f"  filtered to {len(seg)} candidates (sources={srcs}, max={args.max})")

    # Check which already have sdata in the SIDE FILE
    existing = set()
    if args.spec.exists():
        with h5py.File(args.spec, "r") as f:
            for sid in seg["seg_id"].astype(int):
                try:
                    if f"segments/{sid}" in f and "sdata" in f[f"segments/{sid}"]:
                        existing.add(int(sid))
                except Exception:
                    pass
    if existing and not args.force:
        print(f"  already-precomputed in {args.spec.name}: {len(existing)} "
              f"(use --force to redo)")
        seg = seg[~seg["seg_id"].astype(int).isin(existing)]
    print(f"  to process: {len(seg)}")

    if args.dry_run:
        print("[--dry-run] not writing.")
        return

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    # Open ONLY the side file for writing. Bank stays untouched.
    args.spec.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.spec, "a") as fspec:
        fspec.require_group("segments")
        for i, (_, row) in enumerate(seg.iterrows(), 1):
            sid = int(row["seg_id"])
            t_seg = time.perf_counter()
            res = fetch_full_eeg_for_segment(row)
            if res is None:
                print(f"  [{i:3d}/{len(seg)}] seg {sid}: fetch failed", flush=True)
                n_fail += 1
                continue
            data, fs, ch = res
            sdata, sfreqs, stimes = compute_sdata(data, fs)
            if sdata is None:
                print(f"  [{i:3d}/{len(seg)}] seg {sid}: spectrogram compute "
                      f"failed", flush=True)
                n_fail += 1
                continue
            try:
                grp = fspec.require_group(f"segments/{sid}")
                for k in ("sdata", "sfreqs", "stimes"):
                    if k in grp:
                        del grp[k]
                grp.create_dataset("sdata", data=sdata, compression="gzip",
                                    compression_opts=4, chunks=True)
                grp.create_dataset("sfreqs", data=sfreqs)
                grp.create_dataset("stimes", data=stimes)
                grp.attrs["sdata_duration_s"] = float(stimes[-1] - stimes[0])
                grp.attrs["source_dataset"] = str(row.get("source_dataset", ""))
                n_ok += 1
                dt = time.perf_counter() - t_seg
                print(f"  [{i:3d}/{len(seg)}] seg {sid}: sdata={sdata.shape} "
                      f"({stimes[-1]-stimes[0]:.0f}s context) in {dt:.1f}s",
                      flush=True)
                fspec.flush()
            except Exception as e:
                print(f"  [{i:3d}/{len(seg)}] seg {sid}: write failed ({e}); "
                      f"continuing", flush=True)
                n_fail += 1

    print(f"\nDone. ok={n_ok}  fail={n_fail}  total wall={time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
