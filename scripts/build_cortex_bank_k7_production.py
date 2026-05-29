"""Phase 9 Layer 6b — build the Nature-Medicine CORTEX production bank.

The production-scale K=7 bank that backs the actual Nature-Medicine CORTEX-
collected cohort. Per Eli's directive: "utilize the maximum amount of
questions within all the datasets for each domain/task that maintain
maximum statistical cleanliness and power. The idea is that no two tests
are the same due to our large bank of questions."

Filter criteria (n_raters ≥ 5; matches Phase-9 Gate-A "statistical
cleanliness" bound):

  spike    16,681 calibrated segs from data/labels/fits/spike/cases.csv
           (sn1_combined_v2: sn1 + bonobo_only + fabio_spikeed); EEG from
           SN1_combined_v2.h5 /eeg/signals (10s × 128Hz × 20 ch).
  IIIC     21,927 segs from data/labels/iiic_segment_signals.csv with
           n_total ≥ 5; EEG + spectrogram from data/eeg_bank_spec.h5
           (morgoth1_recompute spec_source only); 30s × 200Hz × 20 ch.
  Total    ~38,608 segments.

Bank file size estimate:
  spike   16,681 × ~102 KB = ~1.6 GB
  IIIC    21,927 × ~1.7 MB (incl. sdata) = ~35 GB
  Total   ~37 GB (gzip-compressed inside h5; effective ~22-28 GB)

Modes:
  --mode full        Build all qualifying segments (~37 GB; ~3-4 h compute).
                     Output: data/production_bank/eeg_bank_production.h5
                              + MANIFEST.json (~22 MB; per-seg sha256)
  --mode fallback    Build the offline fallback bundle: stratified 150/task
                     × 7 = 1050 segs (~300 MB; fits in CORTEX installer or
                     GitHub Release). Used when CORTEX runs offline or
                     before the first cloud fetch.
                     Output: cortex_app/cortex_offline_fallback.h5
                              + .manifest.json
  --mode dry-run     Report what would be built; no I/O.

Output schema (matches eeg_bank.h5 internal bank schema, scaled):

  /spike/<seg_id>/
    eeg10s        (20, 1281) float32 — 10s @ 128Hz from SN1
    attrs: family='spike', seg_id, source_dataset, n_raters, s_mean,
           s_sd, pos_rate, pat_age, sex
  /iiic/<seg_id>/
    eeg30s        (20, 6000) float32 — 30s @ 200Hz from morgoth1_recompute
    sdata         (597, 504) float32 — precomputed regional spectrogram
    sfreqs        (126,)     float32 — frequency axis
    stimes        (597,)     float32 — time axis
    attrs: family='iiic', seg_id, pattern_class, source_dataset, n_raters,
           s_mean_<task> + s_sd_<task> for each of 6 IIIC tasks, etc.

MANIFEST.json schema:
  {schema_version, phase, K, generated_utc, filter_criteria,
   bank_path | bank_url, n_spike, n_iiic, n_total,
   segments: [
     {seg_id, family, pattern_class | 'spike',
      source_dataset, n_raters, group_path,
      payload_sha256, ...per-task s_mean/s_sd...}, ...]}

The MANIFEST is consumed by `scripts/cortex_session_bank_fetch.py` for
per-session stratified sampling without downloading the full ~37 GB bank.

Usage:
    .venv/bin/python scripts/build_cortex_bank_k7_production.py --mode fallback
    .venv/bin/python scripts/build_cortex_bank_k7_production.py --mode full
    .venv/bin/python scripts/build_cortex_bank_k7_production.py --mode dry-run
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SN1_PHI = REPO / "data" / "SN1_combined_v2.h5"
SPEC_FILE = REPO / "data" / "eeg_bank_spec.h5"
SEGMENTS_CSV = REPO / "data" / "labels" / "segments.csv"
SEGMENT_SIGNALS_CSV = REPO / "data" / "labels" / "segment_signals.csv"  # Phase-9 K=7 Layer 2
IIIC_SEGMENT_SIGNALS_CSV = REPO / "data" / "labels" / "iiic_segment_signals.csv"  # K=6 fallback
SPIKE_CASES = REPO / "data" / "labels" / "fits" / "spike" / "cases.csv"
LABELS_CSV = REPO / "data" / "labels" / "labels.csv"

PROD_BANK_DIR = REPO / "data" / "production_bank"
PROD_BANK_PATH = PROD_BANK_DIR / "eeg_bank_production.h5"
PROD_MANIFEST_PATH = PROD_BANK_DIR / "MANIFEST.json"

FALLBACK_DIR = REPO / "cortex_app"
FALLBACK_PATH = FALLBACK_DIR / "cortex_offline_fallback.h5"
FALLBACK_MANIFEST_PATH = FALLBACK_DIR / "cortex_offline_fallback.manifest.json"

MIN_N_RATERS = 5
IIIC_CLASSES = ("seizure", "lpd", "gpd", "lrda", "grda", "other")
SPIKE_SUBSOURCES = (
    "sn1_combined_v2:sn1", "sn1_combined_v2:bonobo_only",
    "sn1_combined_v2:fabio_spikeed",
)
FALLBACK_PER_TASK = 150       # offline fallback bundle: 150/task × 7 = 1050 segs
N_STRATA = 10                 # quantile strata for stratified sampling


# ── filters + selection ─────────────────────────────────────────────────────

def _select_spike_pool(min_n_raters: int = MIN_N_RATERS) -> pd.DataFrame:
    """Return spike segments meeting quality filter.
    Columns: seg_id, n_raters, pos_rate, s_mean, s_sd, source_dataset."""
    # Spike fits + K=7 V_B s_mean / s_sd
    fits = pd.read_csv(SPIKE_CASES)
    fits = fits[fits["n_raters"] >= min_n_raters]

    # K=7 V_B s_mean / s_sd for spike (Phase-9 Layer 2 output)
    if SEGMENT_SIGNALS_CSV.exists():
        sig = pd.read_csv(
            SEGMENT_SIGNALS_CSV,
            usecols=["seg_id", "s_mean_spike", "s_sd_spike"])
        sig = sig.rename(columns={"s_mean_spike": "s_mean",
                                   "s_sd_spike": "s_sd"})
        fits = fits.merge(sig, on="seg_id", suffixes=("_v13", ""), how="left")
        # Where K=7 V_B has NaN (segments outside the IIIC pool), keep v13.
        fits["s_mean"] = fits["s_mean"].fillna(fits["s_mean_v13"])
        fits["s_sd"] = fits["s_sd"].fillna(fits["s_sd_v13"])
        fits = fits.drop(columns=["s_mean_v13", "s_sd_v13"])

    # Sub-source from segments.csv
    segs = pd.read_csv(SEGMENTS_CSV, low_memory=False,
                       usecols=["seg_id", "source_dataset", "pat_age", "sex"])
    spike_segs = segs[segs["source_dataset"].isin(SPIKE_SUBSOURCES)]
    fits = fits.merge(spike_segs, on="seg_id", how="left")
    return fits.dropna(subset=["source_dataset"]).reset_index(drop=True)


def _select_iiic_pool(min_n_raters: int = MIN_N_RATERS) -> pd.DataFrame:
    """Return IIIC segments meeting quality filter.
    Columns: seg_id, n_raters, pattern_class, source_dataset,
             s_mean_<task> + s_sd_<task> for each of 6 IIIC tasks."""
    # Use K=7 V_B segment_signals.csv if available (Phase-9 Layer 2 output);
    # else fall back to v13 iiic_segment_signals.csv (drift_mean 0.001 vs v13
    # per the variant-comparison gate, so this fallback is acceptable in dev).
    if SEGMENT_SIGNALS_CSV.exists():
        sig_path = SEGMENT_SIGNALS_CSV
    elif IIIC_SEGMENT_SIGNALS_CSV.exists():
        sig_path = IIIC_SEGMENT_SIGNALS_CSV
    else:
        raise FileNotFoundError(
            "neither segment_signals.csv (K=7) nor iiic_segment_signals.csv "
            "(K=6) found")
    sig = pd.read_csv(sig_path)

    # IIIC-task columns (always present; spike is in K=7 file but ignored here)
    iiic_tasks = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
    s_cols = []
    for t in iiic_tasks:
        s_cols.extend([f"s_mean_{t}", f"s_sd_{t}"])
    s_cols_present = [c for c in s_cols if c in sig.columns]
    if len(s_cols_present) != len(s_cols):
        missing = set(s_cols) - set(s_cols_present)
        raise RuntimeError(f"{sig_path} missing columns: {missing}")

    # n_total per segment (IIIC raters)
    tier_cols = [c for c in ("n_expert", "n_experienced", "n_novice",
                              "n_crowd", "n_other", "n_untiered",
                              "n_borderline", "n_unknown")
                 if c in sig.columns]
    if not tier_cols:
        raise RuntimeError(f"{sig_path} has no tier-count columns")
    sig["n_total"] = sig[tier_cols].sum(axis=1)

    pool = sig[sig["n_total"] >= min_n_raters].copy()
    pool = pool.rename(columns={"n_total": "n_raters"})

    # Add plurality pattern_class from labels.csv (the 6-class label)
    print(f"  Loading labels.csv plurality for {len(pool):,} IIIC segs "
          f"(this takes ~30s)...", flush=True)
    seg_set = set(pool["seg_id"].astype(int))
    plur: dict[int, str] = {}
    src_map: dict[int, str] = {}
    COLLAPSE = {"bipd": "other", "birds": "other"}
    seg_class_counts: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    seg_src_set: dict[int, set] = defaultdict(set)
    with open(LABELS_CSV) as f:
        import csv as _csv
        for d in _csv.DictReader(f):
            if d["label_type"] != "pattern_class":
                continue
            seg_id = int(d["seg_id"])
            if seg_id not in seg_set:
                continue
            v = COLLAPSE.get(d["value"].strip(), d["value"].strip())
            seg_class_counts[seg_id][v] += 1
            seg_src_set[seg_id].add(d["source_dataset"])
    for s, counts in seg_class_counts.items():
        plur[s] = max(counts.items(), key=lambda kv: kv[1])[0]
        # Source: pick the dominant source for this seg
        if seg_src_set[s]:
            src_map[s] = sorted(seg_src_set[s])[0]
    pool["pattern_class"] = pool["seg_id"].astype(int).map(plur)
    pool["source_dataset"] = pool["seg_id"].astype(int).map(src_map)
    pool = pool[pool["pattern_class"].isin(IIIC_CLASSES)].reset_index(drop=True)
    return pool


def _stratified_sample(df: pd.DataFrame, key: str, n_per_stratum: int,
                       n_strata: int = N_STRATA, seed: int = 42) -> pd.DataFrame:
    """Stratified-by-quantile sample of `df` on column `key`."""
    if len(df) == 0:
        return df.iloc[:0]
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0, 1, n_strata + 1)
    bins = np.unique(df[key].quantile(quantiles).to_numpy())
    if len(bins) < 2:
        return df.sample(n=min(n_per_stratum * n_strata, len(df)),
                         random_state=int(rng.integers(0, 2**31)))
    df = df.copy()
    df["__stratum"] = pd.cut(df[key], bins=bins, include_lowest=True,
                              labels=False).astype(int)
    picks = []
    for _, sub in df.groupby("__stratum", observed=True):
        n_take = min(n_per_stratum, len(sub))
        picks.append(sub.sample(n=n_take,
                                random_state=int(rng.integers(0, 2**31))))
    return pd.concat(picks, ignore_index=True).drop(columns=["__stratum"])


# ── per-segment writers ────────────────────────────────────────────────────

def _write_spike_seg(dst: h5py.File, sid: int, signal: np.ndarray,
                      row: pd.Series) -> str:
    """Write one spike segment to dst/spike/<sid>/; return its sha256."""
    g = dst.create_group(f"spike/{sid}")
    g.create_dataset("eeg10s", data=signal.astype(np.float32).T,  # (20, 1281)
                     compression="gzip", compression_opts=4)
    g.attrs["family"] = "spike"
    g.attrs["seg_id"] = int(sid)
    g.attrs["source_dataset"] = str(row.get("source_dataset", ""))
    g.attrs["n_raters"] = int(row["n_raters"])
    g.attrs["s_mean"] = float(row.get("s_mean", float("nan")))
    g.attrs["s_sd"] = float(row.get("s_sd", float("nan")))
    g.attrs["pos_rate"] = float(row.get("pos_rate", float("nan")))
    g.attrs["pat_age"] = (float(row.get("pat_age")) if not pd.isna(row.get("pat_age"))
                          else float("nan"))
    g.attrs["sex"] = str(row.get("sex", "")) if not pd.isna(row.get("sex")) else ""
    g.attrs["test_class"] = "spike"
    # sha256 of the eeg10s payload (canonical per-seg integrity hash)
    h = hashlib.sha256(np.ascontiguousarray(signal, dtype=np.float32).tobytes())
    return h.hexdigest()


def _write_iiic_seg(dst: h5py.File, sid: int, spec_seg: h5py.Group,
                     row: pd.Series, k7_tasks: tuple[str, ...]) -> str:
    """Write one IIIC segment to dst/iiic/<sid>/; return its sha256."""
    g = dst.create_group(f"iiic/{sid}")
    for ds_name in ("data30s", "sdata", "sfreqs", "stimes"):
        if ds_name not in spec_seg:
            continue
        data = np.asarray(spec_seg[ds_name])
        out_name = "eeg30s" if ds_name == "data30s" else ds_name
        g.create_dataset(out_name, data=data,
                         compression="gzip", compression_opts=4)
    g.attrs["family"] = "iiic"
    g.attrs["seg_id"] = int(sid)
    g.attrs["pattern_class"] = str(row.get("pattern_class", ""))
    g.attrs["source_dataset"] = str(row.get("source_dataset", ""))
    g.attrs["n_raters"] = int(row["n_raters"])
    g.attrs["test_class"] = "iiic"
    # Per-task signals (K=7 includes spike too; we include all 7 for cross-ref)
    for t in k7_tasks:
        sm_col = f"s_mean_{t}"; sd_col = f"s_sd_{t}"
        if sm_col in row and not pd.isna(row[sm_col]):
            g.attrs[sm_col] = float(row[sm_col])
        if sd_col in row and not pd.isna(row[sd_col]):
            g.attrs[sd_col] = float(row[sd_col])
    # sha256 of eeg30s as the canonical per-seg integrity hash
    eeg = np.asarray(spec_seg["data30s"])
    h = hashlib.sha256(np.ascontiguousarray(eeg, dtype=np.float32).tobytes())
    return h.hexdigest()


# ── main ───────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["full", "fallback", "dry-run"])
    ap.add_argument("--min-n-raters", type=int, default=MIN_N_RATERS)
    ap.add_argument("--fallback-per-task", type=int, default=FALLBACK_PER_TASK)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="override output h5 path")
    ap.add_argument("--manifest-out", type=Path, default=None,
                    help="override manifest path")
    args = ap.parse_args(argv)

    print("=== Phase 9 Layer 6b — production bank build ===")
    print(f"  mode: {args.mode}")
    print(f"  min n_raters: {args.min_n_raters}")

    # Required inputs check
    required = [SN1_PHI, SPEC_FILE, SEGMENTS_CSV, SPIKE_CASES, LABELS_CSV]
    for p in required:
        if not p.exists():
            print(f"ERROR: required input missing: {p}")
            return 2
    if not SEGMENT_SIGNALS_CSV.exists() and not IIIC_SEGMENT_SIGNALS_CSV.exists():
        print(f"ERROR: neither {SEGMENT_SIGNALS_CSV} nor "
              f"{IIIC_SEGMENT_SIGNALS_CSV} found")
        return 2

    print("\nStage 1 — select spike pool")
    spike_pool = _select_spike_pool(args.min_n_raters)
    print(f"  spike pool (n_raters>={args.min_n_raters}): "
          f"{len(spike_pool):,} segs")
    print(f"    s_mean range [{spike_pool['s_mean'].min():.2f}, "
          f"{spike_pool['s_mean'].max():.2f}]")

    print("\nStage 2 — select IIIC pool")
    iiic_pool = _select_iiic_pool(args.min_n_raters)
    print(f"  IIIC pool (n_raters>={args.min_n_raters}): "
          f"{len(iiic_pool):,} segs")
    cls_dist = iiic_pool["pattern_class"].value_counts().to_dict()
    for c in IIIC_CLASSES:
        print(f"    {c:>8}: {cls_dist.get(c, 0):>5,}")

    # Sampling for fallback mode
    if args.mode == "fallback":
        print(f"\nStage 2b — stratified sampling for offline fallback "
              f"({args.fallback_per_task}/task × 7 = "
              f"{args.fallback_per_task * 7} segs)")
        spike_sample = _stratified_sample(
            spike_pool, "s_mean", args.fallback_per_task // 10,
            seed=args.seed)
        print(f"  spike sampled: {len(spike_sample):,}")
        iiic_samples = []
        for cls in IIIC_CLASSES:
            sub = iiic_pool[iiic_pool["pattern_class"] == cls]
            if len(sub) == 0:
                continue
            key = f"s_mean_{cls if cls != 'other' else 'iic'}"
            if key in sub.columns:
                samp = _stratified_sample(
                    sub, key, args.fallback_per_task // 10,
                    seed=args.seed + hash(cls) % 1000)
            else:
                samp = sub.sample(
                    n=min(args.fallback_per_task, len(sub)),
                    random_state=args.seed + hash(cls) % 1000)
            iiic_samples.append(samp)
            print(f"  IIIC {cls:>8} sampled: {len(samp):,}")
        iiic_sample = pd.concat(iiic_samples, ignore_index=True)
        spike_to_write = spike_sample
        iiic_to_write = iiic_sample
    else:
        spike_to_write = spike_pool
        iiic_to_write = iiic_pool

    print(f"\n  TOTAL TO WRITE: spike={len(spike_to_write):,}  "
          f"iiic={len(iiic_to_write):,}  "
          f"= {len(spike_to_write) + len(iiic_to_write):,} segs")

    if args.mode == "dry-run":
        print("\n--mode dry-run: skipping write")
        return 0

    # Resolve output paths
    if args.mode == "fallback":
        out_path = args.out or FALLBACK_PATH
        manifest_path = args.manifest_out or FALLBACK_MANIFEST_PATH
    else:
        out_path = args.out or PROD_BANK_PATH
        manifest_path = args.manifest_out or PROD_MANIFEST_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStage 3 — write bank + manifest")
    print(f"  output:   {out_path}")
    print(f"  manifest: {manifest_path}")
    if out_path.exists():
        print(f"  WARN: {out_path} exists; will overwrite")

    k7_tasks = ("spike", "sz", "lpd", "gpd", "lrda", "grda", "iic")
    manifest_segs: list[dict] = []

    with h5py.File(SN1_PHI, "r") as sn1, h5py.File(SPEC_FILE, "r") as spec, \
         h5py.File(out_path, "w") as dst:
        dst.attrs["phase"] = 9
        dst.attrs["K"] = 7
        dst.attrs["schema_version"] = "production_v1"
        dst.attrs["build_utc"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds")

        # Spike (10s × 128Hz from SN1)
        signals = sn1["/eeg/signals"]
        n_samples_arr = sn1["/eeg/n_samples"][:]
        for i, (_, row) in enumerate(spike_to_write.iterrows()):
            sid = int(row["seg_id"])
            ns = int(n_samples_arr[sid])
            sig = signals[sid, :ns, :].astype(np.float32)  # (ns, 20)
            sha = _write_spike_seg(dst, sid, sig, row)
            manifest_segs.append({
                "seg_id": sid, "family": "spike",
                "pattern_class": "spike",
                "source_dataset": str(row.get("source_dataset", "")),
                "n_raters": int(row["n_raters"]),
                "group_path": f"spike/{sid}",
                "s_mean": float(row["s_mean"]) if not pd.isna(row["s_mean"]) else None,
                "s_sd": float(row["s_sd"]) if not pd.isna(row["s_sd"]) else None,
                "pos_rate": float(row.get("pos_rate", float("nan"))) if not pd.isna(row.get("pos_rate", float("nan"))) else None,
                "payload_sha256": sha,
            })
            if (i + 1) % 500 == 0:
                print(f"    [spike {i + 1}/{len(spike_to_write)}]", flush=True)

        # IIIC (30s × 200Hz from eeg_bank_spec.h5).
        # STRICT FILTER (Nature defensibility — Layer 6b investigation):
        #   spec_source == "morgoth1_recompute" — matches K=6 internal-bank
        #     provenance. Mixing kong_precomputed adds methodology
        #     heterogeneity that's hard to defend in Nature Methods.
        #   data30s shape == (20, 6000) — 20 EEG channels @ 30s × 200 Hz.
        #     Alt-shape segs with 21+ channels are non-standard montages
        #     the CORTEX viewer cannot render.
        # Coverage: 20,090 / 21,927 IIIC pool = 91.6%. The 1,837 dropped
        # segs (743 missing + 1,094 kong_precomputed) are documented in
        # MANIFEST.filter_criteria for reviewer transparency.
        spec_segs = spec["/segments"]
        n_skipped_missing = 0
        n_skipped_kong = 0
        n_skipped_other_spec = 0
        n_skipped_shape = 0
        for i, (_, row) in enumerate(iiic_to_write.iterrows()):
            sid = int(row["seg_id"])
            key = str(sid)
            if key not in spec_segs:
                n_skipped_missing += 1
                continue
            sg = spec_segs[key]
            spec_src = str(sg.attrs.get("spec_source", ""))
            if spec_src != "morgoth1_recompute":
                if spec_src == "kong_precomputed":
                    n_skipped_kong += 1
                else:
                    n_skipped_other_spec += 1
                continue
            if "data30s" not in sg or tuple(sg["data30s"].shape) != (20, 6000):
                n_skipped_shape += 1
                continue
            sha = _write_iiic_seg(dst, sid, sg, row, k7_tasks)
            manifest_row = {
                "seg_id": sid, "family": "iiic",
                "pattern_class": str(row.get("pattern_class", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "n_raters": int(row["n_raters"]),
                "group_path": f"iiic/{sid}",
                "payload_sha256": sha,
            }
            for t in k7_tasks:
                sm_col = f"s_mean_{t}"; sd_col = f"s_sd_{t}"
                if sm_col in row and not pd.isna(row[sm_col]):
                    manifest_row[sm_col] = float(row[sm_col])
                if sd_col in row and not pd.isna(row[sd_col]):
                    manifest_row[sd_col] = float(row[sd_col])
            manifest_segs.append(manifest_row)
            if (i + 1) % 1000 == 0:
                print(f"    [iiic {i + 1}/{len(iiic_to_write)}]", flush=True)
        n_skipped_spec_total = (n_skipped_missing + n_skipped_kong
                                 + n_skipped_other_spec + n_skipped_shape)
        if n_skipped_spec_total:
            print(f"  IIIC seg drops: {n_skipped_spec_total} "
                  f"(missing={n_skipped_missing} "
                  f"kong_precomputed={n_skipped_kong} "
                  f"other_spec_source={n_skipped_other_spec} "
                  f"bad_shape={n_skipped_shape})")

        dst.attrs["n_spike"] = sum(1 for s in manifest_segs if s["family"] == "spike")
        dst.attrs["n_iiic"] = sum(1 for s in manifest_segs if s["family"] == "iiic")

    # Write manifest
    manifest = {
        "schema_version": "production_v1",
        "phase": 9,
        "K": 7,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "bank_path": str(out_path.relative_to(REPO)),
        "filter_criteria": {
            "spike": {"source": "data/labels/fits/spike/cases.csv",
                       "min_n_raters": args.min_n_raters},
            "iiic": {"source": (
                "data/labels/segment_signals.csv (K=7 V_B SVI) → "
                "fallback iiic_segment_signals.csv (K=6 v13)"),
                     "min_n_raters": args.min_n_raters,
                     "spec_filter": "morgoth1_recompute only",
                     "shape_filter": "(20, 6000) only",
                     "n_skipped_missing_from_spec": n_skipped_missing,
                     "n_skipped_kong_precomputed": n_skipped_kong,
                     "n_skipped_other_spec_source": n_skipped_other_spec,
                     "n_skipped_bad_shape": n_skipped_shape},
        },
        "mode": args.mode,
        "n_spike": sum(1 for s in manifest_segs if s["family"] == "spike"),
        "n_iiic": sum(1 for s in manifest_segs if s["family"] == "iiic"),
        "n_total": len(manifest_segs),
        "segments": manifest_segs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print(f"  written segments: {len(manifest_segs):,}")
    print(f"    spike: {manifest['n_spike']:,}")
    print(f"    iiic:  {manifest['n_iiic']:,}")
    print(f"  bank: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  manifest: {manifest_path}  "
          f"({manifest_path.stat().st_size / 1024:.1f} KB)")
    print("=== Phase 9 Layer 6b — production bank build COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
