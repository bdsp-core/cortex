"""Convert the CORTEX test bank into a browser-optimized bundle.

Reads the same artifacts the desktop K=7 engine uses
(scripts/cortex_engine_inputs_k7.py):
  data/eeg_bank.h5                          h5 with /iiic/<id> and /spike/<id>
  data/labels/iiic_segment_signals.csv      per-task s_mean / s_sd
  Sigma_l_fitted_k7.npy                     K=7 fitted Corr_l prior
  calibration/cert_config_v15.yaml          ell_star_unified_v15 cut-scores (default)

Writes (default: cortex_web/public/bundle/<version>/):
  manifest.json     engine inputs + per-segment metadata (PLAN.md §5)
  seg/<id>.eeg      int16 LE, (nCh × nSamp), µV × EEG_SCALE
  seg/<id>.spec     uint8, sdata quantized to the fixed [-10,25] dB range
                    (only when the segment carries precomputed sdata)

The browser engine needs nothing beyond manifest.json to run the particle
filter; the .eeg / .spec blobs are pulled lazily for display.

The web is K=7 — spike at task 0, served first — and spike segments ARE now
included by default (the SpikeViewer UI is wired). ℓ* defaults to the v15
credentialed-panel cut-scores. Pass --no-spike for an IIIC-only build.

USAGE
    python cortex_web/scripts/prepare_web_bundle.py --version v1.5-k7
    python cortex_web/scripts/prepare_web_bundle.py --cert-block ell_star_unified_v14  # override
    python cortex_web/scripts/prepare_web_bundle.py --no-spike        # IIIC only
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
# K=7 unified signals (spike + 6 IIIC; NaN where not applicable). The pre-K=7
# iiic_segment_signals.csv only had the 6 IIIC columns.
SIGNALS = REPO / "data" / "labels" / "segment_signals.csv"
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

# Canonical 20-channel order for IIIC clips (10–20 system + EKG). The v4-k7
# bank stores the array channels-first in this order but DOESN'T carry a
# channel_names attr — the desktop knows it implicitly. We hardcode it as the
# default so the web montage layer (which looks up bipolar pairs by name) can
# render correctly when the attr is absent. Spike clips use the same first 20.
CHANNELS_IIIC_20 = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG",
]

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


def _applicable_task_idx(test_class: str) -> list[int]:
    """Which task indices a segment of this test_class is informative about.
    IIIC segments only carry IIIC signals (tasks 1..6); spike segments only
    carry the spike signal (task 0). The engine must not draw cross-family."""
    return [i for i, c in enumerate(TASK_CLASSES) if c == test_class]


def _emit_segment(g: h5py.Group, sid: int, test_class: str, pattern: str,
                  s_mean: list[float], s_sd: list[float], seg_dir: Path,
                  ) -> tuple[dict, bool]:
    """Quantize one h5 segment's EEG (+ spectrogram if present) to disk and
    return (manifest entry, had_spectrogram). Branches on test_class:
       iiic  → /eeg30s, 200 Hz default, sdata→spec if present
       spike → /eeg10s, 128 Hz default, no spectrogram (spike clips don't carry one)
    """
    if test_class == "spike":
        ds_name, fs_default = "eeg10s", 128.0
    else:
        ds_name, fs_default = "eeg30s", 200.0
    ds = g[ds_name]
    eeg = np.nan_to_num(np.asarray(ds, dtype=np.float64), nan=0.0)
    fs = float(ds.attrs.get("fs_hz", g.attrs.get("fs_hz", fs_default)))
    ch_names = [
        (c.decode() if isinstance(c, bytes) else str(c))
        for c in ds.attrs.get("channel_names", g.attrs.get("channel_names", []))
    ]
    n_ch, n_samp = eeg.shape
    # v4-k7 bank doesn't carry channel_names on either iiic or spike groups —
    # fall back to the canonical 20-channel ordering so the bipolar montage
    # renders correctly (both schemas use the same first 20 channels).
    if not ch_names and n_ch == len(CHANNELS_IIIC_20):
        ch_names = list(CHANNELS_IIIC_20)
    q = np.clip(np.round(eeg * EEG_SCALE), -32768, 32767).astype("<i2")
    (seg_dir / f"{sid}.eeg").write_bytes(q.tobytes())

    spec_name = ""
    spec_shape = None
    spec_time = None  # [t0,t1] s — spectrogram x-axis extent (from stimes)
    spec_freq = None  # [f0,f1] Hz — spectrogram y-axis extent (from sfreqs)
    # Only IIIC clips carry a spectrogram; spike clips render EEG only.
    if test_class != "spike" and "sdata" in g:
        sdata = np.asarray(g["sdata"], dtype=np.float64)
        db = 10.0 * np.log10(sdata + 1e-12)
        qd = np.clip(
            np.round((db - SPEC_DB_LO) / (SPEC_DB_HI - SPEC_DB_LO) * 255.0),
            0, 255,
        ).astype(np.uint8)
        (seg_dir / f"{sid}.spec").write_bytes(qd.tobytes())
        spec_name = f"seg/{sid}.spec"
        spec_shape = list(sdata.shape)
        if "stimes" in g:
            st = np.asarray(g["stimes"], dtype=float)
            if st.size:
                spec_time = [float(st[0]), float(st[-1])]
        if "sfreqs" in g:
            sf = np.asarray(g["sfreqs"], dtype=float)
            if sf.size:
                spec_freq = [float(sf[0]), float(sf[-1])]

    return ({
        "segId": sid,
        "testClass": test_class,
        "patternClass": pattern,
        # Engine reads sMean/sSd only at applicable indices; the rest are
        # sentinel 0.0 to keep the array length-K but the mask is authoritative.
        "sMean": s_mean,
        "sSd": s_sd,
        "applicableTaskIdx": _applicable_task_idx(test_class),
        "fsHz": fs,
        "nCh": int(n_ch),
        "nSamp": int(n_samp),
        "channelNames": ch_names,
        "specShape": spec_shape,
        "specTime": spec_time,
        "specFreq": spec_freq,
        "eeg": f"seg/{sid}.eeg",
        "spec": spec_name,
    }, spec_shape is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.5-k7")
    ap.add_argument("--out", type=Path, default=REPO / "cortex_web" / "public" / "bundle")
    ap.add_argument("--bank", type=Path, default=BANK,
                    help=f"source h5 bank with /iiic and /spike groups (default {BANK}).")
    ap.add_argument("--cert-block", default="ell_star_unified_v15",
                    help="cert_config block to read ℓ* from (v15 default; v14 via override).")
    ap.add_argument("--include-spike", action="store_true",
                    help="(deprecated: spike is now included by default).")
    ap.add_argument("--no-spike", action="store_true",
                    help="exclude /spike segments (spike is included by default now "
                         "that the SpikeViewer UI is wired).")
    ap.add_argument("--max", type=int, default=None, help="cap #segments/group (smoke)")
    ap.add_argument("--max-spike", type=int, default=None,
                    help="separate cap for #spike segments (keeps spike-first from "
                         "dominating a small test bundle). Defaults to --max.")
    args = ap.parse_args()
    include_spike = not args.no_spike

    out_root = args.out / args.version
    seg_dir = out_root / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    ell_star = load_ell_star(args.cert_block)
    corr_l = load_corr_l()
    # K=7 signals CSV carries all 7 task columns; NaN where a segment isn't
    # informative about that task (e.g. spike segs have NaN in IIIC columns).
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
        if "spike" in f and include_spike:
            groups.append(("spike", "spike"))
        elif "spike" in f:
            print(f"  (skipping {len(f['spike'])} spike segments — --no-spike)")
        if not groups:
            raise SystemExit(f"{args.bank} has no /iiic or /spike group")

        for grp_name, test_class in groups:
            seg_ids = sorted(int(s) for s in f[grp_name] if str(s).lstrip("-").isdigit())
            cap = (args.max_spike if (test_class == "spike" and args.max_spike is not None)
                   else args.max)
            if cap:
                seg_ids = seg_ids[:cap]
            for sid in seg_ids:
                g = f[grp_name][str(sid)]
                pattern = str(g.attrs.get("pattern_class", ""))
                # The v4-k7 spike group doesn't carry pattern_class (it's
                # implicitly "spike"); fall back to the group's test_class.
                if not pattern and test_class == "spike":
                    pattern = "spike"
                if pattern not in PATTERN_TO_TASK_IDX:
                    n_skip_pattern += 1
                    continue
                if sid not in iss.index:
                    n_skip_signal += 1
                    continue
                row = iss.loc[sid]
                # NaN at inapplicable task indices is expected (spike segs have
                # NaN IIIC signals and vice versa); we replace with sentinel
                # 0.0 and rely on applicableTaskIdx as the authoritative mask.
                # NaN at an APPLICABLE index is real bad data → skip the seg.
                applicable = _applicable_task_idx(test_class)
                s_mean_raw = [float(row[f"s_mean_{c}"]) for c in TASK_CODES]
                s_sd_raw = [float(row[f"s_sd_{c}"]) for c in TASK_CODES]
                if any(np.isnan(s_mean_raw[k]) or np.isnan(s_sd_raw[k])
                       for k in applicable):
                    n_skip_nan += 1
                    continue
                s_mean = [0.0 if np.isnan(v) else v for v in s_mean_raw]
                s_sd = [0.0 if np.isnan(v) else v for v in s_sd_raw]
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
