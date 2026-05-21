"""Build a precompute manifest for the IIIC spectrogram batch.

For each segment with at least one pattern_class label from one of the 4
IIIC sources (sparcnet50K, pd_rda_profiler, centaur_2025_iiic,
iiic_crowdsourcing:kong2025), determine:

  * which S3 file to fetch (cheapest path that yields both 30-sec EEG +
    10-min context for the spectrogram)
  * file-level metadata (expected sampling rate, channel count, duration)
  * which samples in that file are the central 30-sec EEG window
  * which samples are the 10-min spectrogram window
  * traceability to the source recording (old_file_recording,
    window_center_s in the source recording's timeline)
  * any caveats / missing-data flags

Outputs:
  data/iiic_precompute_manifest.csv   — one row per segment, human-readable
  data/iiic_precompute_manifest.json  — same content, machine-readable

Three fetch strategies:

  morgoth1_10min        10-min .mat at morgoth1/<SUBTYPE>/segments_raw/
                        Used for sparcnet50K, centaur_2025_iiic, pd_rda_profiler
                        File shape: (120000, 20) @ 200 Hz, 600 sec.
                        Central 30 sec: samples 57000-63000.
                        Full file is the 10-min spectrogram window.

  kong_contest_h5       50-sec contest H5 (per-segment group inside
                        iiic_contest_eeg.h5). File shape: (21, 10000)
                        @ 200 Hz, 50 sec. Central 30 sec: samples 2000-8000.
                        WARNING: 10-min spectrogram NOT POSSIBLE from this
                        source — only 50 sec available. The contest H5
                        does, however, ship 4 precomputed 10-min regional
                        spectrograms (spec_LL/RL/LP/RP, 100 freq × 300 time
                        bins, 0.2 Hz × 2 s steps) that we could lift
                        instead — flagged in the manifest.

  unreachable           segments.csv row has no usable s3_uri (mostly the
                        centaur_2025_iiic rows that pointed at now-dead
                        contest image URLs, plus the 7,533 sparcnet50K
                        rows with zero IIIC votes → vote-argmax undefined).

For sparcnet50K we use the IIIC subdir index (built by build_eeg_bank.py)
to correct the vote-argmax-keyed paths that are wrong ~85% of the time.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/mwestover/GithubRepos/ideal-test-multi")
LABELS_DIR = ROOT / "data/labels"
CACHE = Path("/Volumes/Extreme SSD/.eeg_cache")
IIIC_INDEX_PATH = CACHE / "iiic_file_index.json"

IIIC_SOURCES = ("sparcnet50K", "pd_rda_profiler",
                "centaur_2025_iiic", "iiic_crowdsourcing:kong2025")

# Defaults for morgoth1 10-min .mat files. Verified from sample inspections.
MORGOTH1_FS = 200.0
MORGOTH1_N_CHANNELS = 20
MORGOTH1_DURATION_S = 600.0
MORGOTH1_N_SAMPLES = int(MORGOTH1_FS * MORGOTH1_DURATION_S)        # 120,000

# Defaults for Kong contest H5 per-segment data_50sec
KONG_FS = 200.0
KONG_N_CHANNELS = 21
KONG_DURATION_S = 50.0
KONG_N_SAMPLES = int(KONG_FS * KONG_DURATION_S)                     # 10,000

WIN_30S = 30.0
WIN_10MIN_S = 600.0


def load_iiic_index() -> dict[str, str]:
    if IIIC_INDEX_PATH.exists():
        with open(IIIC_INDEX_PATH) as f:
            return json.load(f)
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv",  type=Path, default=ROOT / "data/iiic_precompute_manifest.csv")
    p.add_argument("--out-json", type=Path, default=ROOT / "data/iiic_precompute_manifest.json")
    args = p.parse_args()

    print("Loading unified tables...")
    seg = pd.read_csv(LABELS_DIR / "segments.csv", low_memory=False)
    lab = pd.read_csv(LABELS_DIR / "labels.csv.gz", dtype={"value": "str"},
                       low_memory=False)

    # Restrict to segments that have at least one pattern_class label
    pc = lab[lab["label_type"] == "pattern_class"]
    plur = (pc.groupby("seg_id")["value"]
              .agg(lambda s: s.mode().iloc[0] if len(s) else None)
              .rename("iiic_plurality"))
    n_raters = pc.groupby("seg_id")["rater_id"].nunique().rename("n_iiic_raters")

    iiic_segs = seg[seg["source_dataset"].isin(IIIC_SOURCES)].copy()
    iiic_segs = iiic_segs.merge(plur,    on="seg_id", how="left")
    iiic_segs = iiic_segs.merge(n_raters, on="seg_id", how="left")
    print(f"  IIIC-source segments: {len(iiic_segs):,}")
    print(f"  with ≥1 pattern_class label: {iiic_segs['iiic_plurality'].notna().sum():,}")

    iiic_index = load_iiic_index()
    print(f"  IIIC subdir index loaded ({len(iiic_index):,} files)")

    rows: list[dict] = []
    for _, r in iiic_segs.iterrows():
        sd = str(r["source_dataset"])
        sid = int(r["seg_id"])
        s3_uri = r.get("s3_uri")
        s3_bip = r.get("s3_uri_bipolar")
        plur_v = r.get("iiic_plurality")
        sub = r.get("subtype")
        rec = r.get("old_file_recording")
        wcs = r.get("window_center_s")
        wes = r.get("window_end_s")
        wss = r.get("window_start_s")
        contest_key = r.get("contest_h5_seg_key")

        out: dict = {
            "seg_id":             sid,
            "source_dataset":     sd,
            "iiic_plurality":     plur_v if pd.notna(plur_v) else None,
            "n_iiic_raters":      int(r["n_iiic_raters"]) if pd.notna(r.get("n_iiic_raters")) else 0,
            "subtype":            sub if pd.notna(sub) else None,
            "old_file_recording": rec if pd.notna(rec) else None,
            "source_window_center_s": float(wcs) if pd.notna(wcs) else None,
            "source_window_start_s":  float(wss) if pd.notna(wss) else None,
            "source_window_end_s":    float(wes) if pd.notna(wes) else None,
            "fetch_strategy":     None,
            "fetch_s3_uri":       None,
            "fetch_h5_key":       None,
            "fetch_fs_hz":        None,
            "fetch_n_channels":   None,
            "fetch_n_samples":    None,
            "fetch_duration_s":   None,
            "eeg30_sample_start": None,
            "eeg30_sample_end":   None,
            "eeg30_t_in_source_s_start": None,
            "eeg30_t_in_source_s_end":   None,
            "spec10min_sample_start": None,
            "spec10min_sample_end":   None,
            "spec10min_t_in_source_s_start": None,
            "spec10min_t_in_source_s_end":   None,
            "spec10min_available": False,
            "kong_contest_specs_available": False,
            "notes":              [],
        }

        # ─────────── Kong-only: contest_h5 ───────────
        if sd == "iiic_crowdsourcing:kong2025":
            if pd.notna(contest_key) and pd.notna(s3_uri) and ".h5" in str(s3_uri):
                out["fetch_strategy"]   = "kong_contest_h5"
                out["fetch_s3_uri"]     = str(s3_uri)
                out["fetch_h5_key"]     = str(contest_key)
                out["fetch_fs_hz"]      = KONG_FS
                out["fetch_n_channels"] = KONG_N_CHANNELS
                out["fetch_n_samples"]  = KONG_N_SAMPLES
                out["fetch_duration_s"] = KONG_DURATION_S
                # data_50sec is centred on window_end_s of the source recording
                # (Kong's labeled 10-sec window is [window_end_s-10, window_end_s];
                # the 50-sec h5 is window_end_s±25). So sample 5000 ≈ window_end_s.
                # Central 30 sec in the 50-sec clip = samples [2000:8000]
                out["eeg30_sample_start"] = 2000
                out["eeg30_sample_end"]   = 8000
                if pd.notna(wes):
                    out["eeg30_t_in_source_s_start"] = float(wes) - 15.0
                    out["eeg30_t_in_source_s_end"]   = float(wes) + 15.0
                # 10-min spectrogram NOT POSSIBLE from contest_h5/data_50sec
                # — only 50 sec of EEG available. BUT the contest H5 also
                # contains 4 precomputed 10-min regional spectrograms
                # (spec_LL/RL/LP/RP) at 0.2 Hz × 2 s resolution.
                out["spec10min_available"] = False
                out["kong_contest_specs_available"] = True
                out["notes"].append(
                    "Kong contest H5 holds only 50-sec EEG, but ships "
                    "precomputed 4-region 10-min spectrograms (0-20 Hz, "
                    "0.2 Hz, 2-sec steps). Use those instead of recomputing.")
                rows.append(out); continue
            else:
                out["fetch_strategy"] = "unreachable"
                out["notes"].append("Kong segment lacks contest_h5_seg_key or s3_uri.")
                rows.append(out); continue

        # ─────────── morgoth1 10-min .mat sources ───────────
        # sparcnet50K, pd_rda_profiler, centaur_2025_iiic all SHOULD live in
        # morgoth1/<SUBTYPE>/segments_raw/<file>.mat for 10-min context.

        # Pick best path. Prefer corrected IIIC-subdir lookup (s3_uri's
        # vote-argmax is wrong ~85% of the time for sparcnet50K; even
        # rows with NULL s3_uri can usually be recovered via file_key
        # because the BIDS-style file_key tells us which file to find in
        # the IIIC subdir index).
        candidate_uri = None
        file_key = r.get("file_key")
        # 1. Try the recorded s3_uri (morgoth1 only)
        if pd.notna(s3_uri) and "morgoth1" in str(s3_uri):
            fn = Path(str(s3_uri)).name
            correct_sd = iiic_index.get(fn)
            if correct_sd is not None:
                candidate_uri = (f"s3://bdsp-opendata-credentialed/morgoth1/data/"
                                 f"internal_dataset/{correct_sd}/segments_raw/{fn}")
            else:
                candidate_uri = str(s3_uri)
        # 2. Fallback: use file_key (BIDS-style filename) directly
        elif pd.notna(file_key):
            fn = str(file_key)
            if not fn.endswith(".mat"):
                fn = fn + ".mat"
            correct_sd = iiic_index.get(fn)
            if correct_sd is not None:
                candidate_uri = (f"s3://bdsp-opendata-credentialed/morgoth1/data/"
                                 f"internal_dataset/{correct_sd}/segments_raw/{fn}")
                out["notes"].append(
                    "Recovered fetch path via file_key lookup in IIIC subdir "
                    "index (s3_uri was null in segments.csv — likely zero-vote "
                    "argmax).")

        if candidate_uri is not None:
            out["fetch_strategy"]   = "morgoth1_10min"
            out["fetch_s3_uri"]     = candidate_uri
            out["fetch_fs_hz"]      = MORGOTH1_FS
            out["fetch_n_channels"] = MORGOTH1_N_CHANNELS
            out["fetch_n_samples"]  = MORGOTH1_N_SAMPLES
            out["fetch_duration_s"] = MORGOTH1_DURATION_S
            # Convention: morgoth1 segments_raw .mat files contain the
            # 10-min window centred on the labeled event. So .mat sample
            # 60,000 ≈ window_center_s in the source recording.
            # Central 30 sec: samples 57,000–63,000.
            out["eeg30_sample_start"] = MORGOTH1_N_SAMPLES // 2 - int(WIN_30S * MORGOTH1_FS / 2)
            out["eeg30_sample_end"]   = MORGOTH1_N_SAMPLES // 2 + int(WIN_30S * MORGOTH1_FS / 2)
            out["spec10min_sample_start"] = 0
            out["spec10min_sample_end"]   = MORGOTH1_N_SAMPLES
            out["spec10min_available"] = True
            if pd.notna(wcs):
                out["eeg30_t_in_source_s_start"] = float(wcs) - WIN_30S / 2
                out["eeg30_t_in_source_s_end"]   = float(wcs) + WIN_30S / 2
                out["spec10min_t_in_source_s_start"] = float(wcs) - WIN_10MIN_S / 2
                out["spec10min_t_in_source_s_end"]   = float(wcs) + WIN_10MIN_S / 2
            rows.append(out); continue

        # ─────────── Unreachable ───────────
        out["fetch_strategy"] = "unreachable"
        if not pd.notna(s3_uri):
            out["notes"].append(
                "segments.csv has no s3_uri (likely zero IIIC votes → "
                "vote-argmax undefined, or broken contest image URL).")
        else:
            out["notes"].append(f"s3_uri not in morgoth1 pattern: {s3_uri}")
        rows.append(out)

    df = pd.DataFrame(rows)
    # Materialise the notes list as semicolon-joined string for CSV
    df["notes"] = df["notes"].apply(lambda L: "; ".join(L) if L else "")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    df.to_json(args.out_json, orient="records", indent=2)

    print("\n═══ summary ═══")
    print(f"  rows in manifest:     {len(df):,}")
    print(f"\nBy fetch strategy:")
    print(df["fetch_strategy"].value_counts(dropna=False).to_string())
    print(f"\nBy source_dataset × strategy:")
    print(df.groupby(["source_dataset", "fetch_strategy"]).size().unstack(fill_value=0).to_string())
    print(f"\n10-min spectrogram availability (raw EEG):")
    print(f"  available (morgoth1 10-min): {df['spec10min_available'].sum():,}")
    print(f"  Kong contest precomputed:    {df['kong_contest_specs_available'].sum():,}")
    print(f"  neither:                     {(~(df['spec10min_available'] | df['kong_contest_specs_available'])).sum():,}")
    print(f"\nWrote: {args.out_csv}  ({args.out_csv.stat().st_size/1024:.1f} KB)")
    print(f"       {args.out_json}  ({args.out_json.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
