"""IIIC precompute: 30-sec central EEG + 10-min multitaper spectrogram.

Driven by data/iiic_precompute_manifest.csv. For each row whose
`fetch_strategy` is `morgoth1_10min` or `kong_contest_h5`, downloads the
source, extracts the central 30 sec of EEG, computes the 10-min
multitaper spectrogram (Chronux conventions: NW=3, K=5, 4-sec window,
1-sec step, 0.5-25 Hz, NFFT=next-pow2), and writes:

    /Volumes/Extreme SSD/eeg_bank_spec.h5    (side file, never touches main bank)
        /segments/<seg_id>/
            data30s    (n_ch, 6000)  float32  central 30 sec @ 200 Hz
            fs_hz30s   attr
            channel_names30s  attr
            sdata      (n_times, n_freqs*4)  float32  10-min multitaper
            sfreqs     (n_freqs,)
            stimes     (n_times,)
            spec_source     attr  "morgoth1_recompute" | "kong_precomputed"
            source_window_center_s  attr (timing in source recording)

Parallel: ProcessPoolExecutor with `--workers`. h5py writes are in the
main process only.

USAGE
    python3 scripts/precompute_iiic.py --max 10           # smoke test
    python3 scripts/precompute_iiic.py --workers 8        # full run
    python3 scripts/precompute_iiic.py --strategies morgoth1_10min
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
from scipy import signal as sig

ROOT = Path("/Users/mwestover/GithubRepos/ideal-test-multi")
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the S3FS helpers + Kong h5 cache from the bank builder
import build_eeg_bank as beb  # noqa: E402

MANIFEST_PATH = ROOT / "data/iiic_precompute_manifest.csv"
SPEC_PATH = Path("/Volumes/Extreme SSD/eeg_bank_spec.h5")

# Compute params (Chronux mtspecgram_jj.m defaults)
NW = 3
K_TAPERS = 5
WINDOW_S = 4.0
STEP_S = 1.0
FPASS_COMPUTE = (0.0, 55.0)
FPASS_STORE = (0.5, 25.0)

CHANNELS_19 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
               'Fz', 'Cz', 'Pz',
               'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']


# ─────────────────── shared signal helpers (mirrors precompute_spectrograms) ─

def _butter_band_filtfilt(x, lo, hi, fs, order=3):
    nyq = 0.5 * fs
    hi = min(hi, nyq - 1.0)
    b, a = sig.butter(order, [lo / nyq, hi / nyq], btype='band')
    return sig.filtfilt(b, a, x, axis=-1)


def _iirnotch_filtfilt(x, freq, fs, Q=30):
    b, a = sig.iirnotch(freq, Q, fs)
    return sig.filtfilt(b, a, x, axis=-1)


def _apply_bipolar_clean(data, channel_names):
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


def _mt_spectrogram(x, fs, NW_=NW, K=K_TAPERS, win_s=WINDOW_S, step_s=STEP_S,
                     fpass=FPASS_COMPUTE):
    Nwin = int(win_s * fs)
    Nstep = int(step_s * fs)
    if Nwin > len(x):
        return None, None, None
    NFFT = 1 << int(np.ceil(np.log2(Nwin)))
    tapers = sig.windows.dpss(Nwin, NW=NW_, Kmax=K)
    starts = np.arange(0, len(x) - Nwin + 1, Nstep)
    if len(starts) == 0:
        return None, None, None
    freqs_all = np.fft.rfftfreq(NFFT, d=1.0 / fs)
    fmask = (freqs_all >= fpass[0]) & (freqs_all <= fpass[1])
    freqs = freqs_all[fmask]
    S = np.empty((len(freqs), len(starts)), dtype=np.float64)
    for i, ws in enumerate(starts):
        seg = sig.detrend(x[ws:ws + Nwin], type='linear')
        tapered = seg[None, :] * tapers
        F = np.fft.rfft(tapered, n=NFFT, axis=-1)
        psd = (np.abs(F) ** 2) / fs
        S[:, i] = psd[:, fmask].mean(axis=0)
    times = (starts + Nwin // 2) / fs
    return S, freqs, times


def _compute_sdata_from_eeg(eeg_20ch, fs):
    """Filter -> bipolar -> per-region multitaper -> stack as morgoth sdata.
    Returns (sdata, sfreqs, stimes) or (None, None, None)."""
    eeg = np.nan_to_num(eeg_20ch, nan=0.0).astype(np.float64)
    eeg = _butter_band_filtfilt(eeg, 0.5, 70.0, fs, order=3)
    eeg = _iirnotch_filtfilt(eeg, 60.0, fs)
    bp = _apply_bipolar_clean(eeg, CHANNELS_19)
    region_chans = {'LL': [0,1,2,3], 'RL': [4,5,6,7],
                     'LP': [8,9,10,11], 'RP': [12,13,14,15]}
    sfreqs = stimes = None
    region_means = {}
    for region, idxs in region_chans.items():
        Ssum = None
        for ch in idxs:
            S, f, t = _mt_spectrogram(bp[ch], fs)
            if S is None:
                return None, None, None
            if sfreqs is None:
                sfreqs, stimes = f, t
            Ssum = S if Ssum is None else Ssum + S
        region_means[region] = Ssum / len(idxs)
    keep = (sfreqs >= FPASS_STORE[0]) & (sfreqs <= FPASS_STORE[1])
    sfreqs = sfreqs[keep]
    stacked = np.vstack([region_means['LL'][keep], region_means['RL'][keep],
                          region_means['LP'][keep], region_means['RP'][keep]])
    return stacked.T.astype(np.float32), sfreqs.astype(np.float32), stimes.astype(np.float32)


# ─────────────────── per-segment worker ───────────────────

def _process_one(row_dict):
    """Top-level worker for ProcessPoolExecutor. Returns a payload dict
    {seg_id, status, eeg30, fs, channel_names, sdata, sfreqs, stimes,
     spec_source, source_window_center_s, error}."""
    sid = int(row_dict["seg_id"])
    strat = row_dict["fetch_strategy"]
    if strat not in ("morgoth1_10min", "kong_contest_h5"):
        return {"seg_id": sid, "status": "skip-unreachable"}
    try:
        if strat == "kong_contest_h5":
            return _process_kong(row_dict)
        else:
            return _process_morgoth1(row_dict)
    except Exception as e:
        return {"seg_id": sid, "status": "error", "error": f"{type(e).__name__}: {e}"}


def _reset_s3fs_worker():
    """Drop this worker's s3fs handle so the next _get_s3fs() builds a
    fresh one (new boto3 session, new connection pool, new asyncio loop)."""
    try:
        if hasattr(beb._S3FS_TLS, "fs"):
            try:
                beb._S3FS_TLS.fs = None  # let it get GC'd
            except Exception:
                pass
            delattr(beb._S3FS_TLS, "fs")
    except Exception:
        pass


def _stream_morgoth_mat_with_retry(uri: str, max_attempts: int = 4):
    """Open the .mat via s3fs+h5py, returning (data_full, fs_hz, channels).
    On transient boto3/s3fs connection failures, rebuild the s3fs handle
    and retry with exponential backoff."""
    key = uri.removeprefix("s3://")
    last_err = None
    for attempt in range(max_attempts):
        try:
            fs_obj = beb._get_s3fs()
            with fs_obj.open(key, mode="rb") as fobj:
                with h5py.File(fobj, "r") as hf:
                    data_full = np.asarray(hf["data"], dtype=np.float32)
                    fs_hz = float(np.asarray(hf["Fs"]).flatten()[0])
                    chs = beb._decode_mat_channels(hf, hf["channels"]) if "channels" in hf else CHANNELS_19
            return data_full, fs_hz, chs
        except Exception as e:
            last_err = e
            name = type(e).__name__
            # boto3/aiobotocore/aiohttp transient errors → rebuild handle
            transient = name in (
                "EndpointConnectionError", "ConnectionError", "ReadTimeoutError",
                "ConnectTimeoutError", "ClientConnectionError", "ServerDisconnectedError",
                "ClientOSError", "ClientPayloadError",
            )
            if not transient or attempt == max_attempts - 1:
                raise
            _reset_s3fs_worker()
            time.sleep(min(2 ** attempt, 8))
    raise last_err if last_err else RuntimeError("retry loop exhausted")


def _process_morgoth1(row):
    """Fetch 10-min .mat from morgoth1 via s3fs streaming, extract 30 sec
    EEG, compute multitaper spectrogram. Reads the FULL file (not just the
    central window) because the spectrogram needs all 10 minutes."""
    sid = int(row["seg_id"])
    uri = str(row["fetch_s3_uri"])
    data_full, fs_hz, chs = _stream_morgoth_mat_with_retry(uri)
    # Normalise to (channels, samples)
    if data_full.shape[0] > data_full.shape[1]:
        data_full = data_full.T
    n_ch, n_samples = data_full.shape
    # Compute window indices from manifest (clamp in case file shorter)
    s0 = int(row["eeg30_sample_start"])
    s1 = int(row["eeg30_sample_end"])
    s0 = max(0, min(s0, n_samples))
    s1 = max(0, min(s1, n_samples))
    eeg30 = data_full[:, s0:s1].astype(np.float32)
    # Spectrogram on full 10-min context
    sdata, sfreqs, stimes = _compute_sdata_from_eeg(data_full, fs_hz)
    if sdata is None:
        return {"seg_id": sid, "status": "error", "error": "spectrogram compute failed"}
    return {
        "seg_id": sid,
        "status": "ok",
        "eeg30": eeg30,
        "fs": fs_hz,
        "channel_names": list(chs),
        "sdata": sdata, "sfreqs": sfreqs, "stimes": stimes,
        "spec_source": "morgoth1_recompute",
        "source_window_center_s":
            float(row["source_window_center_s"]) if pd.notna(row["source_window_center_s"]) else None,
    }


def _process_kong(row):
    """Lift the precomputed regional spectrograms from the contest H5
    (already downloaded by build_eeg_bank.py), plus extract the central
    30 sec of data_50sec.

    Contest H5 specs are (100 freq × 300 time) per region (0-20 Hz @ 0.2 Hz,
    2-sec steps over 600 s). We reshape into morgoth's sdata layout
    (n_times, n_freqs*4) — LL, RL, LP, RP.
    """
    sid = int(row["seg_id"])
    uri = str(row["fetch_s3_uri"])
    key = row["fetch_h5_key"]
    f = beb.get_kong_h5(uri)
    with beb._KONG_READ_LOCK:
        grp = f[f"/segments/{key}"]
        data_50sec = np.asarray(grp["data_50sec"], dtype=np.float32)  # (21, 10000)
        spec_LL = np.asarray(grp["spec_LL"], dtype=np.float32)  # (100, 300)
        spec_RL = np.asarray(grp["spec_RL"], dtype=np.float32)
        spec_LP = np.asarray(grp["spec_LP"], dtype=np.float32)
        spec_RP = np.asarray(grp["spec_RP"], dtype=np.float32)
    # Central 30 sec EEG: samples 2000-8000 of the 50-sec clip
    s0, s1 = int(row["eeg30_sample_start"]), int(row["eeg30_sample_end"])
    eeg30 = data_50sec[:, s0:s1]
    # Build morgoth-style sdata from the precomputed contest specs.
    # Contest format: (n_freqs=100, n_times=300), 0-20 Hz at 0.2 Hz steps,
    # 2-sec time steps over 600 s.
    n_freqs, n_times = spec_LL.shape  # (100, 300)
    sfreqs = np.linspace(0.0, 20.0, n_freqs, dtype=np.float32)
    stimes = np.arange(n_times, dtype=np.float32) * 2.0 + 1.0  # 1-sec offset, 2-sec steps
    # Slice to 0.5-25 Hz (consistent with morgoth_recompute path)
    keep = (sfreqs >= FPASS_STORE[0]) & (sfreqs <= FPASS_STORE[1])
    sfreqs_keep = sfreqs[keep]
    stacked = np.vstack([spec_LL[keep], spec_RL[keep],
                          spec_LP[keep], spec_RP[keep]])   # (n_freqs*4, n_times)
    sdata = stacked.T.astype(np.float32)
    # Kong contest channels (21): Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz, Pz,
    # Fp2, F4, C4, P4, F8, T4, T6, O2, EKG, Photic
    kong_chs = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                 'Fp2','F4','C4','P4','F8','T4','T6','O2','EKG','Photic']
    return {
        "seg_id": sid,
        "status": "ok",
        "eeg30": eeg30,
        "fs": 200.0,
        "channel_names": kong_chs,
        "sdata": sdata, "sfreqs": sfreqs_keep.astype(np.float32),
        "stimes": stimes.astype(np.float32),
        "spec_source": "kong_precomputed",
        "source_window_center_s":
            float(row["source_window_center_s"]) if pd.notna(row["source_window_center_s"]) else None,
    }


# ─────────────────── pool worker init ───────────────────

def _pool_init():
    beb._pool_init()


# ─────────────────── main ───────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--spec",     type=Path, default=SPEC_PATH)
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--max",      type=int, default=None,
                    help="Max segments to process (smoke test)")
    p.add_argument("--strategies", nargs="+",
                    default=["morgoth1_10min", "kong_contest_h5"],
                    help="Subset of fetch_strategy values to process")
    p.add_argument("--force",    action="store_true",
                    help="Recompute even if a side-file entry already exists")
    p.add_argument("--seg-ids",  nargs="+", type=int, default=None,
                    help="Restrict to these seg_ids (overrides --max).")
    args = p.parse_args()

    print(f"Loading manifest: {args.manifest}")
    man = pd.read_csv(args.manifest, low_memory=False, dtype={"notes": "str"})
    print(f"  total manifest rows: {len(man):,}")
    man = man[man["fetch_strategy"].isin(args.strategies)].copy()
    print(f"  after strategy filter: {len(man):,}")

    if args.seg_ids:
        man = man[man["seg_id"].astype(int).isin(args.seg_ids)].copy()
        print(f"  --seg-ids filter: {len(man):,}")
    elif args.max:
        man = man.head(args.max).copy()
        print(f"  --max filter: {len(man):,}")

    # Skip already-done unless --force
    args.spec.parent.mkdir(parents=True, exist_ok=True)
    already_done: set[int] = set()
    if args.spec.exists() and not args.force:
        try:
            with h5py.File(args.spec, "r") as f:
                for sid in man["seg_id"].astype(int):
                    try:
                        g = f[f"segments/{sid}"]
                        if "sdata" in g and "data30s" in g:
                            already_done.add(int(sid))
                    except Exception:
                        pass
        except Exception:
            pass
    if already_done:
        print(f"  already done in side file: {len(already_done):,} (use --force to redo)")
        man = man[~man["seg_id"].astype(int).isin(already_done)].copy()
    print(f"  to process: {len(man):,}")

    if len(man) == 0:
        print("Nothing to do.")
        return

    # Make the IIIC subdir index visible to worker processes via cache
    beb.CACHE.mkdir(parents=True, exist_ok=True)
    if not (beb.CACHE / "iiic_file_index.json").exists():
        beb.build_iiic_index()

    # Warm Kong h5 in the main process if Kong rows are present (so the file
    # is cached on disk; workers will reopen it themselves).
    if "kong_contest_h5" in args.strategies and (man["fetch_strategy"] == "kong_contest_h5").any():
        kong_uri = man[man["fetch_strategy"] == "kong_contest_h5"]["fetch_s3_uri"].iloc[0]
        print("Warming Kong contest H5 (one-time download if not cached)...")
        beb.get_kong_h5(kong_uri)
        print("  Kong H5 ready.")

    t0 = time.perf_counter()
    n_ok = n_fail = n_skip = 0
    rows = man.to_dict("records")

    print(f"\nStarting ProcessPoolExecutor ({args.workers} workers)...", flush=True)
    fspec = h5py.File(args.spec, "a")
    fspec.require_group("segments")
    REOPEN_EVERY = 100
    try:
        # NOTE: tried max_tasks_per_child=200 — on Python 3.14 + this
        # workload, the simultaneous recycle at i=N_workers*200 deadlocked
        # the pool (workers died, respawn never delivered). The retry+rebuild
        # logic in _stream_morgoth_mat_with_retry handles stale boto3 sessions
        # without needing worker recycling.
        with ProcessPoolExecutor(max_workers=args.workers,
                                  initializer=_pool_init) as pool:
            futures = [pool.submit(_process_one, r) for r in rows]
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                sid = res["seg_id"]
                status = res.get("status")
                if status == "ok":
                    try:
                        grp = fspec.require_group(f"segments/{sid}")
                        for k in ("data30s", "sdata", "sfreqs", "stimes"):
                            if k in grp:
                                del grp[k]
                        grp.create_dataset("data30s", data=res["eeg30"],
                                            compression="gzip", compression_opts=4,
                                            chunks=True)
                        grp.create_dataset("sdata", data=res["sdata"],
                                            compression="gzip", compression_opts=4,
                                            chunks=True)
                        grp.create_dataset("sfreqs", data=res["sfreqs"])
                        grp.create_dataset("stimes", data=res["stimes"])
                        grp.attrs["fs_hz30s"] = float(res["fs"])
                        grp.attrs["channel_names30s"] = np.array(
                            res["channel_names"], dtype="S16")
                        grp.attrs["spec_source"] = res["spec_source"]
                        if res.get("source_window_center_s") is not None:
                            grp.attrs["source_window_center_s"] = \
                                float(res["source_window_center_s"])
                        n_ok += 1
                    except Exception as e:
                        n_fail += 1
                        print(f"  [{i:5d}/{len(rows)}] seg {sid}: write fail {e}",
                              flush=True)
                elif status == "skip-unreachable":
                    n_skip += 1
                else:
                    n_fail += 1
                    err = res.get("error", "?")
                    if n_fail <= 20 or n_fail % 100 == 0:
                        print(f"  [{i:5d}/{len(rows)}] seg {sid}: FAIL {err}",
                              flush=True)
                if i % 50 == 0:
                    wall = time.perf_counter() - t0
                    rate = n_ok / max(wall, 1e-6)
                    print(f"  [{i:5d}/{len(rows)}]  ok={n_ok:5d}  fail={n_fail:4d}  "
                          f"skip={n_skip:3d}  {rate:.1f} seg/s  {wall:.0f}s",
                          flush=True)
                # Close + reopen the side file every N writes to limit
                # corruption blast radius on hard process death.
                if i % REOPEN_EVERY == 0:
                    try:
                        fspec.flush()
                        fspec.close()
                    except Exception:
                        pass
                    fspec = h5py.File(args.spec, "a")
                    fspec.require_group("segments")
    finally:
        try:
            fspec.flush()
            fspec.close()
        except Exception:
            pass

    wall = time.perf_counter() - t0
    print(f"\nDone. ok={n_ok}  fail={n_fail}  skip={n_skip}  "
          f"wall={wall:.0f}s  ({n_ok/wall:.1f} seg/s)")


if __name__ == "__main__":
    main()
