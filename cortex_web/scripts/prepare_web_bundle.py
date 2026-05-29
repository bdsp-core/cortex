"""Convert the CORTEX IIIC test bank into a browser-optimized bundle.

Reads the same artifacts the desktop engine uses:
  data/eeg_bank.h5                          (the `iiic/` group)
  data/labels/iiic_segment_signals.csv      (per-task s_mean / s_sd)
  Sigma_l_fitted.npy                        (the K=6 fitted Corr_l prior)
  calibration/cert_config.yaml              (ell_star_unified_v13 cut-scores)

Writes (default: cortex_web/public/bundle/<version>/):
  manifest.json     engine inputs + per-segment metadata (see PLAN.md §5)
  seg/<id>.eeg      int16 LE, (nCh × nSamp), µV × EEG_SCALE
  seg/<id>.spec     uint8, sdata quantized to the fixed [-10,25] dB range
                    (only when the segment carries precomputed sdata)

The browser engine needs nothing beyond manifest.json to run the particle
filter; the .eeg / .spec blobs are pulled lazily for display.

USAGE
    python cortex_web/scripts/prepare_web_bundle.py --version v1.1
    python cortex_web/scripts/prepare_web_bundle.py --max 50   # smoke subset
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent  # ideal-test-multi/
BANK = REPO / "data" / "eeg_bank.h5"
SIGNALS = REPO / "data" / "labels" / "iiic_segment_signals.csv"
SIGMA = REPO / "Sigma_l_fitted.npy"
CERT = REPO / "calibration" / "cert_config.yaml"

# Canonical K=6 IIIC tasks (must match scripts/cortex_engine_inputs.py TASKS).
TASKS = [
    ("sz", "Seizure", "seizure"),
    ("lpd", "LPD", "lpd"),
    ("gpd", "GPD", "gpd"),
    ("lrda", "LRDA", "lrda"),
    ("grda", "GRDA", "grda"),
    ("iic", "Other", "other"),
]
TASK_CODES = [t[0] for t in TASKS]
TASK_LABELS = [t[1] for t in TASKS]
TASK_WORDS = [t[2] for t in TASKS]

EEG_SCALE = 4.0  # int16 µV × 4  →  0.25 µV resolution; ±8191 µV range
SPEC_DB_LO, SPEC_DB_HI = -10.0, 25.0  # fixed display levels → uint8 quantize


def load_ell_star() -> list[float]:
    import yaml
    data = yaml.safe_load(CERT.read_text()) or {}
    tasks = data["ell_star_unified_v13"]["tasks"]
    return [float(tasks[f"sparcnet_{c}"]["ell_star"]) for c in TASK_CODES]


def load_corr_l() -> list[list[float]]:
    obj = np.load(SIGMA, allow_pickle=True).item()
    domains = [str(d) for d in np.asarray(obj["domains"])]
    if domains != TASK_CODES:
        raise SystemExit(f"Sigma domain order {domains} != {TASK_CODES}")
    return np.asarray(obj["Corr_l"], dtype=float).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.1")
    ap.add_argument("--out", type=Path, default=REPO / "cortex_web" / "public" / "bundle")
    ap.add_argument("--max", type=int, default=None, help="cap #segments (smoke)")
    args = ap.parse_args()

    out_root = args.out / args.version
    seg_dir = out_root / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    ell_star = load_ell_star()
    corr_l = load_corr_l()

    iss = pd.read_csv(SIGNALS).set_index("seg_id")
    sig_cols = [f"s_mean_{c}" for c in TASK_CODES] + [f"s_sd_{c}" for c in TASK_CODES]
    missing = [c for c in sig_cols if c not in iss.columns]
    if missing:
        raise SystemExit(f"{SIGNALS} missing columns: {missing}")

    segments = []
    n_spec = 0
    with h5py.File(BANK, "r") as f:
        if "iiic" not in f:
            raise SystemExit(f"{BANK} has no 'iiic' group")
        seg_ids = sorted(int(s) for s in f["iiic"] if str(s).lstrip("-").isdigit())
        if args.max:
            seg_ids = seg_ids[: args.max]
        for sid in seg_ids:
            g = f["iiic"][str(sid)]
            pattern = str(g.attrs.get("pattern_class", ""))
            if pattern not in TASK_WORDS:
                print(f"  skip seg {sid}: pattern_class {pattern!r} not a task")
                continue
            if sid not in iss.index:
                print(f"  skip seg {sid}: no signal row")
                continue
            row = iss.loc[sid]
            s_mean = [float(row[f"s_mean_{c}"]) for c in TASK_CODES]
            s_sd = [float(row[f"s_sd_{c}"]) for c in TASK_CODES]
            if any(np.isnan(s_mean)) or any(np.isnan(s_sd)):
                print(f"  skip seg {sid}: NaN signal")
                continue

            # --- EEG: eeg30s (nCh, nSamp) → int16 µV×scale ---
            # nan_to_num: some banks carry NaN samples (bad channels); int16
            # cast of NaN is undefined → zero-fill (renders as flat line).
            eeg = np.nan_to_num(np.asarray(g["eeg30s"], dtype=np.float64), nan=0.0)
            fs = float(g.attrs.get("fs_hz", 200.0))
            ch_names = [
                (c.decode() if isinstance(c, bytes) else str(c))
                for c in g.attrs.get("channel_names", [])
            ]
            n_ch, n_samp = eeg.shape
            q = np.clip(np.round(eeg * EEG_SCALE), -32768, 32767).astype("<i2")
            (seg_dir / f"{sid}.eeg").write_bytes(q.tobytes())

            spec_name = ""
            if "sdata" in g:
                # sdata: (nTimes, nFreqs*4) dB-able PSD → uint8 over [-10,25] dB
                sdata = np.asarray(g["sdata"], dtype=np.float64)
                db = 10.0 * np.log10(sdata + 1e-12)
                qd = np.clip(
                    np.round((db - SPEC_DB_LO) / (SPEC_DB_HI - SPEC_DB_LO) * 255.0),
                    0, 255,
                ).astype(np.uint8)
                (seg_dir / f"{sid}.spec").write_bytes(qd.tobytes())
                spec_name = f"seg/{sid}.spec"
                n_spec += 1

            segments.append({
                "segId": sid,
                "patternClass": pattern,
                "sMean": s_mean,
                "sSd": s_sd,
                "fsHz": fs,
                "nCh": int(n_ch),
                "nSamp": int(n_samp),
                "channelNames": ch_names,
                "specShape": (list(np.asarray(g["sdata"]).shape) if "sdata" in g else None),
                "eeg": f"seg/{sid}.eeg",
                "spec": spec_name,
            })

    manifest = {
        "version": args.version,
        "eegScale": EEG_SCALE,
        "specDbRange": [SPEC_DB_LO, SPEC_DB_HI],
        "taskCodes": TASK_CODES,
        "taskLabels": TASK_LABELS,
        "taskPatternWords": TASK_WORDS,
        "ellStar": ell_star,
        "corrL": corr_l,
        "nSegments": len(segments),
        "segments": segments,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(p.stat().st_size for p in seg_dir.glob("*"))
    print(f"\nbundle {args.version}: {len(segments)} segments "
          f"({n_spec} with spectrogram), {total/1e6:.1f} MB at {out_root}")
    print(f"  manifest: {out_root / 'manifest.json'}")
    print("  upload `seg/` + manifest.json to S3+CloudFront for the SPA to fetch.")


if __name__ == "__main__":
    main()
