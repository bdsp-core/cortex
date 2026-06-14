"""Convert the CORTEX K=7 test bank into a browser-optimized bundle.

K=7 = spike (task 0) + 6 IIIC (sz, lpd, gpd, lrda, grda, iic). Reads:
  data/eeg_bank.h5                          (`iiic/` group: eeg30s + sdata;
                                             `spike/` group: eeg10s, no spec)
  data/labels/iiic_segment_signals.csv      (per-task IIIC s_mean / s_sd)
                                             spike s_mean/s_sd come from the
                                             spike group attrs.
  Sigma_l_fitted_k7.npy                      (the K=7 fitted Corr_l prior)
  calibration/cert_config_v15.yaml           (ell_star_unified_v15 cut-scores)

Writes (default: cortex_web/public/bundle/<version>/):
  manifest.json     engine inputs + per-segment metadata
  seg/<id>.eeg      int16 LE, (nCh × nSamp), µV × EEG_SCALE
  seg/<id>.spec     uint8 sdata over the fixed [-10,25] dB range (IIIC only)

Per-segment sMean/sSd are length-7 vectors; the slot for a task the segment is
NOT a candidate for is written as JSON `null` (spike seg → [val,null×6]; IIIC
seg → [null, 6 IIIC vals]). The TS engine maps null → NaN and family-gates the
per-task candidate banks on it (mirrors cortex_engine_inputs_k7.as_engine_arrays).

USAGE
    python cortex_web/scripts/prepare_web_bundle.py --version v1.1-local
    python cortex_web/scripts/prepare_web_bundle.py --max 60   # smoke subset
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent  # ideal-test-multi/
BANK = REPO / "data" / "eeg_bank.h5"
SIGNALS = REPO / "data" / "labels" / "iiic_segment_signals.csv"
SIGMA_K7 = REPO / "Sigma_l_fitted_k7.npy"
CERT_V15 = REPO / "calibration" / "cert_config_v15.yaml"

# Canonical K=7 tasks (spike first) — matches scripts/cortex_engine_inputs_k7.py.
TASKS = [
    ("spike", "Spike", "spike"),
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
IIIC_CODES = TASK_CODES[1:]   # ['sz','lpd','gpd','lrda','grda','iic']
IIIC_WORDS = TASK_WORDS[1:]   # ['seizure','lpd','gpd','lrda','grda','other']

# cert_config v15 YAML key per task code (combined_spike vs sparcnet_*),
# mirrors cortex_policy_k7._KEY_FOR_CODE_K7.
KEY_FOR_CODE = {
    "spike": "combined_spike", "sz": "sparcnet_sz", "lpd": "sparcnet_lpd",
    "gpd": "sparcnet_gpd", "lrda": "sparcnet_lrda", "grda": "sparcnet_grda",
    "iic": "sparcnet_iic",
}

EEG_SCALE = 4.0  # int16 µV × 4  →  0.25 µV resolution; ±8191 µV range
SPEC_DB_LO, SPEC_DB_HI = -10.0, 25.0  # fixed display levels → uint8 quantize

# Canonical 19-channel order (eeg_bank_viewer.py:87-89 + the :1152 fallback);
# the bank omits the per-dataset channel_names attr, so we supply this order.
CHANNELS_19 = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
               "Fz", "Cz", "Pz",
               "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]


def load_ell_star() -> list[float]:
    import yaml
    data = yaml.safe_load(CERT_V15.read_text()) or {}
    tasks = data["ell_star_unified_v15"]["tasks"]
    return [float(tasks[KEY_FOR_CODE[c]]["ell_star"]) for c in TASK_CODES]


def load_corr_l() -> list[list[float]]:
    obj = np.load(SIGMA_K7, allow_pickle=True).item()
    domains = [str(d) for d in np.asarray(obj["domains"])]
    if domains != TASK_CODES:
        raise SystemExit(f"k7 Sigma domain order {domains} != {TASK_CODES}")
    return np.asarray(obj["Corr_l"], dtype=float).tolist()


def write_eeg(ds, sid: int, seg_dir: Path, fs_default: float):
    """int16 µV×scale EEG blob → seg/<id>.eeg; returns (fs, nCh, nSamp, names)."""
    eeg = np.nan_to_num(np.asarray(ds, dtype=np.float64), nan=0.0)
    fs = float(ds.attrs.get("fs_hz", fs_default))
    ch_names = [(c.decode() if isinstance(c, bytes) else str(c))
                for c in ds.attrs.get("channel_names", [])]
    n_ch, n_samp = eeg.shape
    if not ch_names:
        ch_names = CHANNELS_19[:n_ch]
    q = np.clip(np.round(eeg * EEG_SCALE), -32768, 32767).astype("<i2")
    (seg_dir / f"{sid}.eeg").write_bytes(q.tobytes())
    return fs, int(n_ch), int(n_samp), ch_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.1-local")
    ap.add_argument("--out", type=Path, default=REPO / "cortex_web" / "public" / "bundle")
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--max", type=int, default=None, help="cap #IIIC segs (smoke)")
    args = ap.parse_args()

    out_root = args.out / args.version
    seg_dir = out_root / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    ell_star = load_ell_star()
    corr_l = load_corr_l()

    iss = pd.read_csv(SIGNALS).set_index("seg_id")
    sig_cols = [f"s_mean_{c}" for c in IIIC_CODES] + [f"s_sd_{c}" for c in IIIC_CODES]
    missing = [c for c in sig_cols if c not in iss.columns]
    if missing:
        raise SystemExit(f"{SIGNALS} missing columns: {missing}")

    segments = []
    n_spec = n_spike = 0
    with h5py.File(args.bank, "r") as f:
        # ── IIIC segments first (carry the spectrogram; the tutorial example
        #    is the first IIIC seg) ────────────────────────────────────────
        iiic_ids = sorted(int(s) for s in f["iiic"]) if "iiic" in f else []
        if args.max:
            iiic_ids = iiic_ids[: args.max]
        for sid in iiic_ids:
            g = f["iiic"][str(sid)]
            pattern = str(g.attrs.get("pattern_class", ""))
            if pattern not in IIIC_WORDS or sid not in iss.index:
                continue
            row = iss.loc[sid]
            iiic_sm = [float(row[f"s_mean_{c}"]) for c in IIIC_CODES]
            iiic_sd = [float(row[f"s_sd_{c}"]) for c in IIIC_CODES]
            if any(np.isnan(iiic_sm)) or any(np.isnan(iiic_sd)):
                continue
            fs, n_ch, n_samp, ch_names = write_eeg(g["eeg30s"], sid, seg_dir, 200.0)

            spec_name, spec_time, spec_freq, spec_shape = "", None, None, None
            if "sdata" in g:
                sdata = np.asarray(g["sdata"], dtype=np.float64)
                db = 10.0 * np.log10(sdata + 1e-12)
                qd = np.clip(np.round((db - SPEC_DB_LO) / (SPEC_DB_HI - SPEC_DB_LO) * 255.0),
                             0, 255).astype(np.uint8)
                (seg_dir / f"{sid}.spec").write_bytes(qd.tobytes())
                spec_name = f"seg/{sid}.spec"
                spec_shape = list(sdata.shape)
                n_spec += 1
                if "stimes" in g:
                    st = np.asarray(g["stimes"], dtype=float)
                    if st.size:
                        spec_time = [float(st[0]), float(st[-1])]
                if "sfreqs" in g:
                    sf = np.asarray(g["sfreqs"], dtype=float)
                    if sf.size:
                        spec_freq = [float(sf[0]), float(sf[-1])]

            segments.append({
                "segId": sid, "family": "iiic", "patternClass": pattern,
                "sMean": [None] + iiic_sm, "sSd": [None] + iiic_sd,
                "fsHz": fs, "nCh": n_ch, "nSamp": n_samp, "channelNames": ch_names,
                "specShape": spec_shape, "specTime": spec_time, "specFreq": spec_freq,
                "eeg": f"seg/{sid}.eeg", "spec": spec_name,
            })

        # ── spike segments (eeg10s @128Hz, Yes/No, no spectrogram) ─────────
        spike_ids = sorted(int(s) for s in f["spike"]) if "spike" in f else []
        if args.max:
            spike_ids = spike_ids[: max(8, args.max // 4)]
        for sid in spike_ids:
            g = f["spike"][str(sid)]
            sm, sd = g.attrs.get("s_mean"), g.attrs.get("s_sd")
            if sm is None or sd is None or np.isnan(float(sm)) or np.isnan(float(sd)):
                continue
            fs, n_ch, n_samp, ch_names = write_eeg(g["eeg10s"], sid, seg_dir, 128.0)
            segments.append({
                "segId": sid, "family": "spike", "patternClass": "spike",
                "sMean": [float(sm)] + [None] * 6, "sSd": [float(sd)] + [None] * 6,
                "fsHz": fs, "nCh": n_ch, "nSamp": n_samp, "channelNames": ch_names,
                "specShape": None, "specTime": None, "specFreq": None,
                "eeg": f"seg/{sid}.eeg", "spec": "",
            })
            n_spike += 1

    manifest = {
        "version": args.version, "eegScale": EEG_SCALE,
        "specDbRange": [SPEC_DB_LO, SPEC_DB_HI],
        "taskCodes": TASK_CODES, "taskLabels": TASK_LABELS, "taskPatternWords": TASK_WORDS,
        "ellStar": ell_star, "corrL": corr_l,
        "nSegments": len(segments), "segments": segments,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(p.stat().st_size for p in seg_dir.glob("*"))
    print(f"\nbundle {args.version}: {len(segments)} segments "
          f"({n_spec} IIIC w/ spectrogram + {n_spike} spike), {total/1e6:.1f} MB at {out_root}")
    print(f"  K=7 ell* (v15): {[round(e,3) for e in ell_star]}")
    print(f"  manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
