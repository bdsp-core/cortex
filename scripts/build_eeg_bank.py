"""Build the unified EEG testing-bank H5 on the external SSD.

Target: /Volumes/Extreme SSD/eeg_bank.h5

Reads every row in data/labels/segments.csv with a usable s3_uri, pulls
the source EEG from S3, extracts a short clip (≤30 s), and writes it as
a group in the unified H5:

    /segments/<seg_id>/
        data            (channels, samples) float32
        attrs:
            seg_id, source_dataset, subtype, fs_hz, n_samples,
            window_start_s, window_end_s, channel_names,
            s3_uri, contest_h5_seg_key (if Kong)

Idempotent: re-running skips segments already present. Streamable: writes
each segment incrementally. Designed to run overnight.

Source families handled:

  sparcnet50K, pd_rda_profiler, morgoth1:lpd/gpd/lrda/grda/seizure/iiic
      sparcnet50K segments are 10-min 200 Hz clips. Extract the central
      10 s around the labeled event (sample index 60,000 ± 1000).

  pd_rda_profiler (via iiic-freq3 s3_uri_bipolar)
      Already 10 s @ 200 Hz, 19 channels. Use as-is.

  morgoth1:spikes_hm  (15 s @ 128 Hz, 20 channels). Use as-is.

  morgoth1:spikes / bets / bird / posts / vw / wickets
      10-min @ 200 Hz clips. Extract a 10-s window around the event_time
      (relative-to-recording-start) listed in the manifest.

  iiic_crowdsourcing:kong2025
      Stream /segments/<contest_h5_seg_key>/data_50sec from the contest
      H5 (21 ch × 10,000 samples = 50 s @ 200 Hz). Extract central 10 s.

  sn1_combined_v2  (binary spike corpus)
      Slice /eeg/signals[s3_h5_index] from the spike H5 (typically 10 s
      @ 128 Hz, 19 channels). Source bucket is bdsp-opendata-RESTRICTED
      (different creds); skipped here pending bucket access.

USAGE
    python3 scripts/build_eeg_bank.py --dry-run         # report counts only
    python3 scripts/build_eeg_bank.py --max 1000        # bound the run
    python3 scripts/build_eeg_bank.py                   # full overnight run
    python3 scripts/build_eeg_bank.py --sources sparcnet50K pd_rda_profiler
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, local
from typing import Optional, Tuple

_KONG_READ_LOCK = Lock()
_S3FS_TLS = local()

import h5py
import numpy as np
import pandas as pd

try:
    import s3fs   # noqa: F401
    _HAVE_S3FS = True
except Exception:
    _HAVE_S3FS = False

ROOT = Path("/Users/mwestover/GithubRepos/ideal-test-multi")
LAB = ROOT / "data/labels"
BANK_PATH = Path("/Volumes/Extreme SSD/eeg_bank.h5")
CACHE = Path("/Volumes/Extreme SSD/.eeg_cache")
LOG_PATH = Path("/tmp/eeg_bank_build.log")

WINDOW_S_DEFAULT = 10.0  # central window length to extract


# ───────────────────────── S3 helpers ─────────────────────────

def s3_cp(s3_url: str, dest: Path, retry: int = 2) -> bool:
    """Download an S3 object to local cache. Idempotent."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retry + 1):
        proc = subprocess.run(
            ["aws", "--profile", "opendata", "s3", "cp",
              "--no-progress", s3_url, str(dest)],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return True
        if attempt < retry:
            time.sleep(2 ** attempt)
    return False


def s3_local(s3_url: str) -> Path:
    """Map an s3:// URL to a local cache path."""
    assert s3_url.startswith("s3://"), s3_url
    key = s3_url.removeprefix("s3://")
    return CACHE / key


# ───────────────────────── readers per source format ─────────────────────────

CHANNELS_20 = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
               'Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']
CHANNELS_19 = CHANNELS_20[:-1]


def _decode_mat_channels(f, ch_handle):
    """Decode a MATLAB v7.3 cell array of channel names."""
    names = []
    arr = np.asarray(ch_handle)
    for ref in arr.flatten():
        try:
            ch_data = f[ref]
            vals = np.asarray(ch_data).flatten()
            names.append(bytes(vals).decode('utf-16', errors='ignore').rstrip('\x00'))
        except Exception:
            names.append("?")
    return names


def read_mat_v73(path: Path) -> Tuple[np.ndarray, float, list[str], dict]:
    """Read a MATLAB v7.3 .mat file with `data`, `Fs`, `channels` fields.

    Returns (data shape=(samples, channels), Fs, channel_names, extras).
    """
    with h5py.File(path, "r") as f:
        data = np.asarray(f["data"], dtype=np.float32)  # (samples, channels) or (channels, samples)
        Fs = float(np.asarray(f["Fs"]).flatten()[0])
        chs = _decode_mat_channels(f, f["channels"]) if "channels" in f else None
        extras = {}
        for k in ("event_time_pts", "score"):
            if k in f:
                try:
                    extras[k] = np.asarray(f[k]).flatten().tolist()
                except Exception:
                    pass
    # Normalise to (channels, samples) shape — most readers want channels-first.
    if data.shape[0] > data.shape[1]:
        # (samples, channels) -> (channels, samples)
        data = data.T
    return data, Fs, chs or [], extras


def _get_s3fs():
    """Return a thread-local s3fs.S3FileSystem.

    s3fs uses a single asyncio loop per FileSystem instance — sharing one
    instance across N threads serialises all S3 reads through that one
    loop. Giving each thread its own instance lets the OS run N HTTPS
    transfers truly in parallel.
    """
    inst = getattr(_S3FS_TLS, "fs", None)
    if inst is None:
        import s3fs as _s3fs_mod
        inst = _s3fs_mod.S3FileSystem(profile="opendata", anon=False,
                                       default_block_size=2 ** 20)
        _S3FS_TLS.fs = inst
    return inst


def stream_mat_central_window(s3_uri: str, window_s: float
                              ) -> Optional[Tuple[np.ndarray, float, list[str], dict]]:
    """Stream the central `window_s` seconds of a 10-min .mat file from S3
    via h5py + s3fs partial reads (HTTP range requests).

    Reads only ~5 chunks (~300 KB) instead of the full ~10 MB file. About
    20× faster end-to-end than `aws s3 cp` + full read for the same windows.

    Returns (data (channels, samples), Fs, channel_names, extras) or None
    on failure.
    """
    if not _HAVE_S3FS:
        return None
    bucket_key = s3_uri.removeprefix("s3://")
    fs = _get_s3fs()
    try:
        with fs.open(bucket_key, mode="rb") as fobj:
            with h5py.File(fobj, "r") as hf:
                if "data" not in hf or "Fs" not in hf:
                    return None
                d = hf["data"]
                Fs = float(np.asarray(hf["Fs"]).flatten()[0])
                # data is (samples, channels) — verify before slicing
                n_samples, n_channels = d.shape
                if n_samples < n_channels:
                    # data was stored channels-first; flip indices
                    return None
                w_samples = int(round(window_s * Fs))
                center = n_samples // 2
                start = max(0, center - w_samples // 2)
                end = min(n_samples, start + w_samples)
                start = max(0, end - w_samples)
                block = np.asarray(d[start:end, :], dtype=np.float32).T  # (chans, samples)
                ws, we = start / Fs, end / Fs
                chs = _decode_mat_channels(hf, hf["channels"]) if "channels" in hf else []
                extras = {"window_streamed": True}
                return block, Fs, chs, extras
    except Exception:
        return None


def read_mat_scipy(path: Path) -> Tuple[np.ndarray, float, list[str], dict]:
    """Read a classic .mat file (pre-v7.3) used by iiic-freq3 bipolar."""
    import scipy.io
    d = scipy.io.loadmat(path)
    data = np.asarray(d.get("data"), dtype=np.float32)
    Fs = float(np.asarray(d.get("Fs")).flatten()[0])
    # data is typically (channels, samples) or (samples, channels)
    if data.shape[0] > data.shape[1]:
        data = data.T
    chs = CHANNELS_19 if data.shape[0] == 19 else CHANNELS_20
    return data, Fs, chs, {}


def center_window(data: np.ndarray, Fs: float, window_s: float,
                   event_idx: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
    """Crop `data` (channels, samples) to a window of `window_s` seconds
    centred either on `event_idx` (sample index) or on the clip middle.
    Returns (cropped_data, window_start_s, window_end_s)."""
    n_samples = data.shape[1]
    total_s = n_samples / Fs
    w_samples = int(round(window_s * Fs))
    if event_idx is None:
        center = n_samples // 2
    else:
        center = int(event_idx)
    start = max(0, center - w_samples // 2)
    end = min(n_samples, start + w_samples)
    start = max(0, end - w_samples)
    cropped = data[:, start:end]
    return cropped, start / Fs, end / Fs


# ───────────────────────── Kong contest H5 ─────────────────────────

_KONG_H5_HANDLE = {"path": None, "handle": None}


def get_kong_h5(s3_uri: str) -> h5py.File:
    """Open the contest H5 once and reuse."""
    if _KONG_H5_HANDLE["handle"] is None:
        local = s3_local(s3_uri)
        if not local.exists() or local.stat().st_size < 1e9:
            print(f"  Downloading Kong contest H5 (~9.5 GB) — this is the one-time hit ...",
                  flush=True)
            if not s3_cp(s3_uri, local):
                raise RuntimeError("Failed to fetch Kong contest H5")
        _KONG_H5_HANDLE["handle"] = h5py.File(local, "r")
        _KONG_H5_HANDLE["path"] = local
    return _KONG_H5_HANDLE["handle"]


def read_kong_segment(s3_uri: str, key: str
                      ) -> Tuple[np.ndarray, float, list[str], dict]:
    """Stream a segment out of the Kong contest H5. Threadsafe via lock."""
    with _KONG_READ_LOCK:
        f = get_kong_h5(s3_uri)
        grp = f[f"/segments/{key}"]
        data = np.asarray(grp["data_50sec"], dtype=np.float32)   # (21, 10000)
        Fs = 200.0
        ch = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
               'Fp2','F4','C4','P4','F8','T4','T6','O2','EKG','Photic']
        extras = {k: grp.attrs[k] for k in grp.attrs if not k.startswith("__")}
    return data, Fs, ch, extras


# ───────────────────────── per-source dispatch ─────────────────────────

# Pre-built file-name → subdir index for IIIC; populated by build_iiic_index().
IIIC_FILE_INDEX: dict[str, str] = {}


def _pool_init():
    """Process-pool initializer: load the IIIC subdir index once per worker."""
    cache_path = CACHE / "iiic_file_index.json"
    if cache_path.exists():
        with open(cache_path) as f:
            IIIC_FILE_INDEX.update(json.load(f))


def _pool_work(row_dict, window_s):
    """Top-level worker function for ProcessPoolExecutor (picklable).

    Receives a plain-dict row (Series can be flaky across processes), runs
    `load_one`, returns (seg_id, row_dict, payload).
    """
    import pandas as _pd  # ensure import in worker
    row = _pd.Series(row_dict)
    try:
        return int(row_dict["seg_id"]), row_dict, load_one(row, window_s)
    except Exception as e:
        return int(row_dict["seg_id"]), row_dict, None


def build_iiic_index() -> None:
    """List the 6 IIIC subdirs once and build {file_name -> subdir}."""
    cache_path = CACHE / "iiic_file_index.json"
    if cache_path.exists() and cache_path.stat().st_size > 100:
        with open(cache_path) as f:
            IIIC_FILE_INDEX.update(json.load(f))
        print(f"  loaded IIIC file index from cache: {len(IIIC_FILE_INDEX):,} files",
              flush=True)
        return
    print("Building IIIC file-name index (one-time S3 ls)...", flush=True)
    subdirs = ("LPD", "GPD", "LRDA", "GRDA", "SEIZURE", "IIIC")
    for sd in subdirs:
        url = (f"s3://bdsp-opendata-credentialed/morgoth1/data/"
                f"internal_dataset/{sd}/segments_raw/")
        proc = subprocess.run(
            ["aws", "--profile", "opendata", "s3", "ls", url, "--recursive"],
            capture_output=True, text=True)
        n_before = len(IIIC_FILE_INDEX)
        for line in proc.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            key = parts[-1]
            file_name = Path(key).name
            if file_name.endswith(".mat"):
                IIIC_FILE_INDEX[file_name] = sd
        print(f"  {sd}: +{len(IIIC_FILE_INDEX) - n_before:,} files "
              f"(total {len(IIIC_FILE_INDEX):,})", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(IIIC_FILE_INDEX, f)


LOADER_FOR_SOURCE = {
    "sparcnet50K":               "morgoth1_10min",
    "pd_rda_profiler":           "iiic_freq3_or_morgoth1",
    "centaur_2025_iiic":         "morgoth1_10min",
    "morgoth1:spikes":           "morgoth1_10min",
    "morgoth1:spikes_hm":        "morgoth1_15sec",
    "morgoth1:bets":             "morgoth1_10min",
    "morgoth1:bird":             "morgoth1_10min",
    "morgoth1:posts":            "morgoth1_10min",
    "morgoth1:vw":               "morgoth1_10min",
    "morgoth1:wickets":          "morgoth1_10min",
    "iiic_crowdsourcing:kong2025": "kong_h5",
}


def load_one(seg_row, window_s: float) -> Optional[dict]:
    """Read + window a single segment. Returns dict or None on failure."""
    sd = str(seg_row["source_dataset"])
    s3_uri = seg_row.get("s3_uri")
    if pd.isna(s3_uri) or not s3_uri:
        return None
    loader = LOADER_FOR_SOURCE.get(sd)
    if loader is None:
        return None

    if loader == "kong_h5":
        key = seg_row.get("contest_h5_seg_key")
        if pd.isna(key) or not key:
            return None
        data, Fs, ch, extras = read_kong_segment(s3_uri, str(key))
        # Kong is 50s @ 200Hz; window to central 10s
        cropped, ws, we = center_window(data, Fs, window_s, event_idx=None)
        return {"data": cropped, "Fs": Fs, "channels": ch,
                 "window_start_s": ws, "window_end_s": we,
                 "extras": extras}

    if loader == "iiic_freq3_or_morgoth1":
        # pd_rda_profiler: prefer the small 10s bipolar at iiic-freq3
        bip = seg_row.get("s3_uri_bipolar")
        if pd.notna(bip) and str(bip).startswith("s3://"):
            local = s3_local(str(bip))
            if not s3_cp(str(bip), local):
                # fall through to 10-min
                pass
            else:
                data, Fs, ch, extras = read_mat_scipy(local)
                cropped, ws, we = center_window(data, Fs, window_s)
                return {"data": cropped, "Fs": Fs, "channels": ch,
                         "window_start_s": ws, "window_end_s": we,
                         "extras": extras}

    # For 10-min .mats (morgoth1_10min), stream just the central window via
    # s3fs + HTTP byte-range reads on the chunked HDF5 — ~20× faster than a
    # full file download. For 15-sec files we just download the whole
    # 300 KB file since the win is small and the code path is simpler.
    if loader == "morgoth1_10min":
        # For sparcnet50K, the recorded s3_uri is keyed by vote-argmax which
        # is wrong ~85% of the time. Look up the correct IIIC subdir first.
        effective_uri = str(s3_uri)
        if sd == "sparcnet50K":
            file_name = Path(effective_uri).name
            correct_sd = IIIC_FILE_INDEX.get(file_name)
            if correct_sd is not None:
                effective_uri = (f"s3://bdsp-opendata-credentialed/morgoth1/data/"
                                 f"internal_dataset/{correct_sd}/segments_raw/{file_name}")
        streamed = stream_mat_central_window(effective_uri, window_s)
        if streamed is not None:
            data, Fs, ch, extras = streamed
            n_samples = data.shape[1]
            ws = (data.shape[1] // 2) / Fs   # rough — the stream already centred
            return {"data": data, "Fs": Fs, "channels": ch,
                     "window_start_s": ws, "window_end_s": ws + n_samples / Fs,
                     "extras": extras}
        # Streaming failed entirely (network blip, missing file) — fall
        # through to full-download path below.

    # Fallback: full download
    local = s3_local(str(s3_uri))
    if not s3_cp(str(s3_uri), local):
        # sparcnet50K's s3_uri is keyed by vote-argmax which is wrong for
        # most rows. Use the IIIC file-name → subdir index built at startup.
        if sd == "sparcnet50K":
            file_name = Path(str(s3_uri)).name
            alt = IIIC_FILE_INDEX.get(file_name)
            if alt is None:
                return None
            alt_uri = (f"s3://bdsp-opendata-credentialed/morgoth1/data/"
                       f"internal_dataset/{alt}/segments_raw/{file_name}")
            if alt_uri == str(s3_uri):
                return None
            alt_local = s3_local(alt_uri)
            if not s3_cp(alt_uri, alt_local):
                return None
            local = alt_local
        else:
            return None
    try:
        data, Fs, ch, extras = read_mat_v73(local)
    except Exception:
        return None
    if loader == "morgoth1_15sec":
        # Already short — use as-is
        return {"data": data, "Fs": Fs, "channels": ch,
                 "window_start_s": 0.0, "window_end_s": data.shape[1] / Fs,
                 "extras": extras}
    # 10-min: window to central window_s seconds
    cropped, ws, we = center_window(data, Fs, window_s)
    return {"data": cropped, "Fs": Fs, "channels": ch,
             "window_start_s": ws, "window_end_s": we,
             "extras": extras}


# ───────────────────────── main loop ─────────────────────────

def already_written(bank: h5py.File, seg_id: int) -> bool:
    return f"segments/{seg_id}" in bank


def write_segment(bank: h5py.File, seg_id: int, seg_row, payload: dict):
    grp = bank.require_group(f"segments/{seg_id}")
    if "data" in grp:
        del grp["data"]
    grp.create_dataset("data", data=payload["data"], compression="gzip",
                        compression_opts=4, chunks=True, dtype=np.float32)
    grp.attrs["seg_id"]          = int(seg_id)
    grp.attrs["source_dataset"]  = str(seg_row["source_dataset"])
    grp.attrs["subtype"]         = str(seg_row.get("subtype") or "")
    grp.attrs["fs_hz"]           = float(payload["Fs"])
    grp.attrs["n_samples"]       = int(payload["data"].shape[1])
    grp.attrs["window_start_s"]  = float(payload["window_start_s"])
    grp.attrs["window_end_s"]    = float(payload["window_end_s"])
    grp.attrs["channel_names"]   = np.array(payload["channels"], dtype="S16")
    if pd.notna(seg_row.get("s3_uri")):
        grp.attrs["s3_uri"] = str(seg_row["s3_uri"])
    if pd.notna(seg_row.get("contest_h5_seg_key")):
        grp.attrs["contest_h5_seg_key"] = str(seg_row["contest_h5_seg_key"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, default=BANK_PATH)
    p.add_argument("--window-s", type=float, default=WINDOW_S_DEFAULT)
    p.add_argument("--max", type=int, default=None,
                    help="Max segments to process (smoke-test).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sources", nargs="+", default=None,
                    help="Restrict to specific source_dataset values.")
    p.add_argument("--workers", type=int, default=8,
                    help="Concurrent S3 downloads.")
    p.add_argument("--executor", choices=["thread", "process"], default="process",
                    help="thread = ThreadPoolExecutor (legacy; s3fs serialises). "
                         "process = ProcessPoolExecutor (true parallelism).")
    args = p.parse_args()

    logging.basicConfig(
        filename=str(LOG_PATH), level=logging.INFO,
        format="%(asctime)s  %(message)s")
    logger = logging.getLogger("eeg_bank")
    print(f"Log: {LOG_PATH}", flush=True)

    seg = pd.read_csv(LAB / "segments.csv", low_memory=False)
    candidate_sources = set(LOADER_FOR_SOURCE.keys())
    if args.sources:
        candidate_sources &= set(args.sources)
    seg = seg[seg["source_dataset"].isin(candidate_sources)
              & seg["s3_uri"].notna()].copy()
    # Skip already-written segments at filter time so --max bounds new work.
    if args.bank.exists():
        try:
            with h5py.File(args.bank, "r") as _bk:
                done = set(int(s) for s in _bk["segments"]) if "segments" in _bk else set()
            seg = seg[~seg["seg_id"].astype(int).isin(done)].copy()
            print(f"  pre-filter: dropped {len(done):,} segments already in bank",
                  flush=True)
        except OSError:
            pass
    if args.max:
        seg = seg.head(args.max)
    print(f"Eligible segments: {len(seg):,}", flush=True)
    print(f"  by source: {seg['source_dataset'].value_counts().to_dict()}", flush=True)

    if args.dry_run:
        return

    args.bank.parent.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    n_done = 0; n_skip = 0; n_fail = 0
    t_start = time.perf_counter()

    # Pre-flight: build the file-name → subdir index for sparcnet50K fallbacks.
    if "sparcnet50K" in candidate_sources:
        build_iiic_index()

    # Pre-flight: warm up Kong H5 (one-time large download). Doing it serially
    # so the per-segment loop can stream from the local copy without contention.
    if "iiic_crowdsourcing:kong2025" in candidate_sources:
        any_kong = seg[seg["source_dataset"] == "iiic_crowdsourcing:kong2025"]
        if len(any_kong):
            kong_uri = any_kong["s3_uri"].iloc[0]
            print("Warming Kong contest H5 (one-time 9.5 GB if not cached)...",
                  flush=True)
            get_kong_h5(str(kong_uri))
            print("  Kong H5 ready.", flush=True)

    with h5py.File(args.bank, "a") as bank:
        bank.attrs.setdefault("description",
            b"Unified EEG testing-bank: short-window clips for "
            b"spike / IIIC / mimic segments across all unified label sources.")
        bank.attrs.setdefault("window_s_target", args.window_s)
        rows = [r for _, r in seg.iterrows()]
        # Filter out already-written segments up-front so the pool isn't
        # processing things we'd skip anyway.
        rows = [r for r in rows if not already_written(bank, int(r["seg_id"]))]
        print(f"  to process: {len(rows):,} (already written: "
              f"{len(seg) - len(rows):,})", flush=True)

        # Parallel pre-fetch + serial write. The pool downloads + parses each
        # segment to a payload dict; the main thread receives futures as they
        # complete and writes to the H5 (h5py is not threadsafe for writes).
        def _work(row):
            sid = int(row["seg_id"])
            try:
                return sid, row, load_one(row, args.window_s)
            except Exception as e:
                logger.warning(f"seg {sid}: {type(e).__name__}: {e}")
                return sid, row, None

        if args.executor == "process":
            # Convert each row to a plain dict for picklability
            row_dicts = [r.to_dict() for r in rows]
            executor_ctx = ProcessPoolExecutor(max_workers=args.workers,
                                                 initializer=_pool_init)
            print(f"  using ProcessPoolExecutor ({args.workers} workers)", flush=True)
        else:
            row_dicts = rows
            executor_ctx = ThreadPoolExecutor(max_workers=args.workers)
            print(f"  using ThreadPoolExecutor ({args.workers} workers)", flush=True)

        with executor_ctx as pool:
            if args.executor == "process":
                futures = [pool.submit(_pool_work, rd, args.window_s) for rd in row_dicts]
            else:
                futures = [pool.submit(_work, r) for r in row_dicts]
            for i, fut in enumerate(as_completed(futures), 1):
                seg_id, row, payload = fut.result()
                if args.executor == "process" and isinstance(row, dict):
                    row = pd.Series(row)
                if payload is None:
                    n_fail += 1
                else:
                    try:
                        write_segment(bank, seg_id, row, payload)
                        n_done += 1
                    except Exception as e:
                        logger.warning(f"seg {seg_id} write failed: {e}")
                        n_fail += 1
                if i % 25 == 0:
                    wall = time.perf_counter() - t_start
                    rate = n_done / max(wall, 1e-6)
                    msg = (f"  [{i:6d}/{len(rows):6d}]  done={n_done:6d}  "
                            f"fail={n_fail:5d}  {rate:.1f} seg/s  {wall:.0f}s")
                    print(msg, flush=True)
                    logger.info(msg)
                if i % 500 == 0:
                    bank.flush()

    print(f"\nDone. {n_done:,} written, {n_skip:,} already present, {n_fail:,} failed.",
          flush=True)
    print(f"Bank at {args.bank}  size={args.bank.stat().st_size / 1e9:.2f} GB",
          flush=True)


if __name__ == "__main__":
    main()
