"""Convert the CORTEX test bank into a browser-optimized bundle.

Reads the same artifacts the desktop K=7 engine uses
(scripts/cortex_engine_inputs_k7.py):
  data/eeg_bank.h5                          h5 with /iiic/<id> and /spike/<id>
  data/labels/iiic_segment_signals.csv      per-task s_mean / s_sd
  Sigma_l_fitted_k7.npy                     K=7 fitted Corr_l prior
  calibration/cert_config.yaml              ell_star_unified_v14 cut-scores

Writes (default: cortex_web/public/bundle/<version>/):
  manifest.json     engine inputs + per-segment metadata (PLAN.md §5)
  seg/<id>.eeg      int16 LE, (nCh × nSamp), µV × EEG_SCALE
  seg/<id>.spec     uint8, sdata quantized to the fixed [-10,25] dB range
                    (only when the segment carries precomputed sdata)

The browser engine needs nothing beyond manifest.json to run the particle
filter; the .eeg / .spec blobs are pulled lazily for display.

The web is now K=7 in framework — spike sits at task 0 — but spike *segments*
are not yet included pending the SpikeViewer UI (per the deploy roadmap, step
2). With no spike items in the bank, the engine simply leaves the spike task
PENDING → REFER, while resolving the 6 IIIC tasks as before.

USAGE
    python cortex_web/scripts/prepare_web_bundle.py --version v1.5-k7
    python cortex_web/scripts/prepare_web_bundle.py --bank data/eeg_bank.h5
    python cortex_web/scripts/prepare_web_bundle.py --cert-block ell_star_unified_v15
    python cortex_web/scripts/prepare_web_bundle.py --include-spike   # once UI ready
    python cortex_web/scripts/prepare_web_bundle.py --max 50          # smoke
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
SIGMA = REPO / "Sigma_l_fitted_k7.npy"
CERT = REPO / "calibration" / "cert_config.yaml"
CERT_V15 = REPO / "calibration" / "cert_config_v15.yaml"   # opt-in (post-pilot)

# Canonical K=7 tasks — must match scripts/cortex_engine_inputs_k7.py TASKS.
# (code, label, pattern_word, test_class). test_class routes the UI: 'spike'
# uses the spike screen (binary mark-or-not), 'iiic' uses the 6-button viewer.
TASKS = [
    ("spike", "Spike",   "spike",   "spike"),
    ("sz",    "Seizure", "seizure", "iiic"),
    ("lpd",   "LPD",     "lpd",     "iiic"),
    ("gpd",   "GPD",     "gpd",     "iiic"),
    ("lrda",  "LRDA",    "lrda",    "iiic"),
    ("grda",  "GRDA",    "grda",    "iiic"),
    ("iic",   "Other",   "other",   "iiic"),
]
TASK_CODES = [t[0] for t in TASKS]
TASK_LABELS = [t[1] for t in TASKS]
TASK_WORDS = [t[2] for t in TASKS]
TASK_CLASSES = [t[3] for t in TASKS]
PATTERN_TO_TASK_IDX = {t[2]: i for i, t in enumerate(TASKS)}

EEG_SCALE = 4.0
SPEC_DB_LO, SPEC_DB_HI = -10.0, 25.0

# cert_config key naming differs by domain (combined_spike vs sparcnet_<iiic>).
_CERT_KEY = {
    "spike": "combined_spike", "sz": "sparcnet_sz", "lpd": "sparcnet_lpd",
    "gpd": "sparcnet_gpd", "lrda": "sparcnet_lrda", "grda": "sparcnet_grda",
    "iic": "sparcnet_iic",
}


def load_ell_star(cert_block: str) -> list[float]:
    """Load per-task ℓ* from cert_config.yaml. `v15` is staged in a sibling
    file (calibration/cert_config_v15.yaml) — only the live default (v14) and
    legacy v13 are present in the main config."""
    import yaml
    if cert_block == "ell_star_unified_v15":
        data = yaml.safe_load(CERT_V15.read_text())
    else:
        data = yaml.safe_load(CERT.read_text())
    if cert_block not in data:
        raise SystemExit(f"{cert_block!r} not in cert_config "
                         f"(available: {sorted(k for k in data if k.startswith('ell_star_'))})")
    tasks = data[cert_block]["tasks"]
    return [float(tasks[_CERT_KEY[c]]["ell_star"]) for c in TASK_CODES]


def load_corr_l() -> list[list[float]]:
    obj = np.load(SIGMA, allow_pickle=True).item()
    domains = [str(d) for d in np.asarray(obj["domains"])]
    if domains != TASK_CODES:
        raise SystemExit(f"Σ_l_k7 domain order {domains} != {TASK_CODES}")
    C = np.asarray(obj["Corr_l"], dtype=float)
    if C.shape != (len(TASK_CODES), len(TASK_CODES)):
        raise SystemExit(f"Σ_l_k7 Corr_l shape {C.shape} != ({len(TASK_CODES)},)*2")
    return C.tolist()


def _emit_segment(g: h5py.Group, sid: int, test_class: str, pattern: str,
                  s_mean: list[float], s_sd: list[float], seg_dir: Path,
                  ) -> tuple[dict, bool]:
    """Quantize one h5 segment's EEG (+ spectrogram if present) to disk and
    return (manifest entry, had_spectrogram)."""
    ds = g["eeg30s"]
    eeg = np.nan_to_num(np.asarray(ds, dtype=np.float64), nan=0.0)
    fs = float(ds.attrs.get("fs_hz", g.attrs.get("fs_hz", 200.0)))
    ch_names = [
        (c.decode() if isinstance(c, bytes) else str(c))
        for c in ds.attrs.get("channel_names", g.attrs.get("channel_names", []))
    ]
    n_ch, n_samp = eeg.shape
    q = np.clip(np.round(eeg * EEG_SCALE), -32768, 32767).astype("<i2")
    (seg_dir / f"{sid}.eeg").write_bytes(q.tobytes())

    spec_name = ""
    spec_shape = None
    if "sdata" in g:
        sdata = np.asarray(g["sdata"], dtype=np.float64)
        db = 10.0 * np.log10(sdata + 1e-12)
        qd = np.clip(
            np.round((db - SPEC_DB_LO) / (SPEC_DB_HI - SPEC_DB_LO) * 255.0),
            0, 255,
        ).astype(np.uint8)
        (seg_dir / f"{sid}.spec").write_bytes(qd.tobytes())
        spec_name = f"seg/{sid}.spec"
        spec_shape = list(np.asarray(g["sdata"]).shape)

    return ({
        "segId": sid,
        "testClass": test_class,
        "patternClass": pattern,
        "sMean": s_mean,
        "sSd": s_sd,
        "fsHz": fs,
        "nCh": int(n_ch),
        "nSamp": int(n_samp),
        "channelNames": ch_names,
        "specShape": spec_shape,
        "eeg": f"seg/{sid}.eeg",
        "spec": spec_name,
    }, spec_shape is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.5-k7")
    ap.add_argument("--out", type=Path, default=REPO / "cortex_web" / "public" / "bundle")
    ap.add_argument("--bank", type=Path, default=BANK,
                    help=f"source h5 bank with /iiic and /spike groups (default {BANK}).")
    ap.add_argument("--cert-block", default="ell_star_unified_v14",
                    help="cert_config block to read ℓ* from (v14 default; v15 opt-in).")
    ap.add_argument("--include-spike", action="store_true",
                    help="include /spike segments. Requires the SpikeViewer UI; "
                         "off by default until that lands (engine still runs K=7 — "
                         "spike just REFERs with no items).")
    ap.add_argument("--max", type=int, default=None, help="cap #segments (smoke)")
    args = ap.parse_args()

    out_root = args.out / args.version
    seg_dir = out_root / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    ell_star = load_ell_star(args.cert_block)
    corr_l = load_corr_l()
    iss = pd.read_csv(SIGNALS).set_index("seg_id")
    sig_cols = [f"s_mean_{c}" for c in TASK_CODES] + [f"s_sd_{c}" for c in TASK_CODES]
    missing = [c for c in sig_cols if c not in iss.columns]
    if missing:
        raise SystemExit(f"{SIGNALS} missing columns: {missing}")

    segments: list[dict] = []
    n_spec = 0
    n_skip_pattern = n_skip_signal = n_skip_nan = 0

    with h5py.File(args.bank, "r") as f:
        groups = []
        if "iiic" in f:
            groups.append(("iiic", "iiic"))
        if "spike" in f and args.include_spike:
            groups.append(("spike", "spike"))
        elif "spike" in f:
            print(f"  (skipping {len(f['spike'])} spike segments — "
                  f"pass --include-spike once the SpikeViewer is wired)")
        if not groups:
            raise SystemExit(f"{args.bank} has no /iiic or /spike group")

        for grp_name, test_class in groups:
            seg_ids = sorted(int(s) for s in f[grp_name] if str(s).lstrip("-").isdigit())
            if args.max:
                seg_ids = seg_ids[: args.max]
            for sid in seg_ids:
                g = f[grp_name][str(sid)]
                pattern = str(g.attrs.get("pattern_class", ""))
                if pattern not in PATTERN_TO_TASK_IDX:
                    n_skip_pattern += 1
                    continue
                if sid not in iss.index:
                    n_skip_signal += 1
                    continue
                row = iss.loc[sid]
                s_mean = [float(row[f"s_mean_{c}"]) for c in TASK_CODES]
                s_sd = [float(row[f"s_sd_{c}"]) for c in TASK_CODES]
                if any(np.isnan(s_mean)) or any(np.isnan(s_sd)):
                    n_skip_nan += 1
                    continue
                entry, had_spec = _emit_segment(g, sid, test_class, pattern,
                                                s_mean, s_sd, seg_dir)
                segments.append(entry)
                if had_spec:
                    n_spec += 1

    manifest = {
        "version": args.version,
        "eegScale": EEG_SCALE,
        "specDbRange": [SPEC_DB_LO, SPEC_DB_HI],
        "taskCodes": TASK_CODES,
        "taskLabels": TASK_LABELS,
        "taskPatternWords": TASK_WORDS,
        "taskClasses": TASK_CLASSES,
        "certBlock": args.cert_block,
        "ellStar": ell_star,
        "corrL": corr_l,
        "nSegments": len(segments),
        "segments": segments,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(p.stat().st_size for p in seg_dir.glob("*"))
    skips = (f"  skipped: pattern={n_skip_pattern}  no-signal={n_skip_signal}  "
             f"nan-signal={n_skip_nan}")
    print(f"\nbundle {args.version}: K={len(TASK_CODES)} "
          f"({args.cert_block}); {len(segments)} segments "
          f"({n_spec} with spectrogram), {total/1e6:.1f} MB at {out_root}")
    print(skips)
    print(f"  manifest: {out_root / 'manifest.json'}")
    print("  upload `seg/` + manifest.json to S3+CloudFront for the SPA to fetch.")


if __name__ == "__main__":
    main()
