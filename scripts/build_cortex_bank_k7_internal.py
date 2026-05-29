"""Phase 9 Layer 6a — build the K=7 internal CORTEX test bank.

Replaces the v1.1.0 eeg_bank.h5 `spike/` group (100 uncalibrated `morgoth1:spikes`
segments) with 50 CALIBRATED SN1 spike segments selected stratified-by-s_mean
from `data/labels/fits/spike/cases.csv` (17,346 calibrated candidates → ≥5
raters filter → 16,681 candidates → 10 quantile strata × 5 segments).

Outputs:
  * `data/eeg_bank.h5` — updated. Existing `iiic/` group untouched (300 IIIC
    segs preserved); `spike/` group replaced with 50 calibrated SN1 segs.
  * `data/eeg_bank_v1.1.0_legacy.h5` — pre-Phase-9 bank backup (only created
    if no backup exists yet).

Spike segment H5 schema (NEW for K=7):
  spike/<seg_id>/
    eeg10s     shape=(20, 1281), dtype=float32, 20 channels @ 128 Hz
    attrs:
      seg_id            int — the SN1 corpus seg_id (0-20520)
      source_dataset    str — 'sn1_combined_v2:sn1' | ':bonobo_only' | ':fabio_spikeed'
      test_class        str = 'spike'
      n_raters          int — number of raters per fits/spike/cases.csv
      s_mean            float — V_B SVI s_j posterior mean
      s_sd              float — V_B SVI s_j posterior sd
      pos_rate          float — empirical positive rate from fits
      pat_age           float — from SN1 segments/pat_age (may be nan)
      sex               str   — from SN1 segments/sex (may be empty)

Why eeg10s (not eeg30s for IIIC compatibility):
  The SN1 spike paper methodology uses 10s windows (Jing et al. published
  methodology). Storing native 10s signals preserves the methodology AND
  matches what was actually used to derive Phase-9 V_B SVI s_mean / s_sd.
  Padding to 30s with zeros would introduce synthetic context; re-fetching
  30s windows from source recordings would require access to SN1's original
  data layer (not in SN1_combined_v2.h5).

Sampling design (stratified-by-s_mean):
  1. Filter calibrated spike cases to n_raters ≥ 5 (Gate-A-style quality
     bound — sufficient ratings to identify per-segment difficulty).
  2. Compute s_mean (the V_B SVI posterior mean per segment) — already in
     `pipeline/joint_calibration/s_j_table_k7_B.csv` for spike task.
  3. 10 equal-probability quantile strata on s_mean.
  4. Per stratum: 5 segments, quality-biased by n_raters (top-3× n_raters
     pool then RNG seed=42 uniform sample of 5).
  5. Result: 50 calibrated spike segments spanning the full difficulty range.

Usage:
    .venv/bin/python scripts/build_cortex_bank_k7_internal.py
        [--out PATH] [--dry-run] [--n-spike 50]
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EEG_BANK = REPO / "data" / "eeg_bank.h5"
SN1_PHI = REPO / "data" / "SN1_combined_v2.h5"
SPIKE_CASES = REPO / "data" / "labels" / "fits" / "spike" / "cases.csv"
S_J_TABLE_K7 = REPO / "calibration" / "joint" / "s_j_table_k7_B.csv"
SEGMENTS_CSV = REPO / "data" / "labels" / "segments.csv"

DEFAULT_N_SPIKE = 50
N_STRATA = 10
PER_STRATUM = DEFAULT_N_SPIKE // N_STRATA  # 5 segments per stratum
QUALITY_POOL_MULT = 3  # quality-bias: pool = top (PER_STRATUM × MULT) by n_raters
N_RATERS_MIN = 5
RNG_SEED = 42


def _stratified_spike_sample(n_target: int = DEFAULT_N_SPIKE,
                              seed: int = RNG_SEED) -> pd.DataFrame:
    """Sample N spike segments stratified by V_B SVI s_mean.

    Returns a DataFrame with columns: seg_id, s_mean, s_sd, n_raters, pos_rate.
    """
    # Load V_B SVI s_mean / s_sd for spike (the Phase-9 calibration substrate)
    sj = pd.read_csv(S_J_TABLE_K7)
    spike_sj = sj[sj["task"] == "spike"][["seg_id", "s_mean", "s_sd"]]

    # Load v13 spike cases (n_raters, pos_rate)
    cases = pd.read_csv(SPIKE_CASES)
    # Join on seg_id
    df = cases.merge(spike_sj, on="seg_id", suffixes=("_v13", ""))
    # Use V_B SVI's s_mean / s_sd for selection (Phase-9 calibration substrate)
    df = df.rename(columns={"s_mean_v13": "_legacy", "s_sd_v13": "_legacy_sd"})

    # Quality filter: n_raters >= N_RATERS_MIN
    df = df[df["n_raters"] >= N_RATERS_MIN].reset_index(drop=True)
    print(f"  candidates with n_raters>={N_RATERS_MIN}: {len(df):,}")

    # 10 quantile strata on s_mean
    quantiles = np.linspace(0, 1, N_STRATA + 1)
    bins = np.unique(df["s_mean"].quantile(quantiles).to_numpy())
    if len(bins) < N_STRATA + 1:
        print(f"  WARN: only {len(bins)-1} distinct strata (s_mean tied at edges)")
    df["stratum"] = pd.cut(df["s_mean"], bins=bins, include_lowest=True,
                            labels=False).astype(int)

    rng = np.random.default_rng(seed)
    picks = []
    per_stratum = max(1, n_target // (len(bins) - 1))
    for k, stratum_df in df.groupby("stratum"):
        # Quality-bias: take top (per_stratum × QUALITY_POOL_MULT) by n_raters,
        # then RNG-sample per_stratum from that pool.
        pool = stratum_df.sort_values("n_raters", ascending=False).head(
            per_stratum * QUALITY_POOL_MULT)
        n_take = min(per_stratum, len(pool))
        sample = pool.sample(n=n_take, random_state=rng.integers(0, 2**31))
        picks.append(sample)
        print(f"    stratum {int(k)}: pool={len(pool):>4}  picked={n_take}  "
              f"s_mean range [{stratum_df['s_mean'].min():.2f}, "
              f"{stratum_df['s_mean'].max():.2f}]")

    out = pd.concat(picks, ignore_index=True).sort_values("seg_id").reset_index(drop=True)
    return out[["seg_id", "s_mean", "s_sd", "n_raters", "pos_rate"]]


def _sn1_segment_meta() -> dict[int, dict]:
    """Map SN1 seg_id (0..20520) → {source, pat_age, sex} from segments.csv.

    The SN1 corpus segments.csv has source ∈ {sn1, bonobo_only, fabio_spikeed}
    rolled up under source_dataset = 'sn1_combined_v2:<sub>'.
    """
    segs = pd.read_csv(SEGMENTS_CSV, low_memory=False)
    sub_sources = {
        "sn1_combined_v2:sn1", "sn1_combined_v2:bonobo_only",
        "sn1_combined_v2:fabio_spikeed",
    }
    spike_segs = segs[segs["source_dataset"].isin(sub_sources)]
    out: dict[int, dict] = {}
    for _, r in spike_segs.iterrows():
        out[int(r["seg_id"])] = {
            "source_dataset": str(r["source_dataset"]),
            "pat_age": float(r.get("pat_age", np.nan) if not pd.isna(r.get("pat_age")) else np.nan),
            "sex": str(r.get("sex", "")) if not pd.isna(r.get("sex", "")) else "",
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=EEG_BANK,
                    help=f"Target file (default {EEG_BANK})")
    ap.add_argument("--n-spike", type=int, default=DEFAULT_N_SPIKE,
                    help=f"Number of calibrated spike segs (default {DEFAULT_N_SPIKE})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report sampling without writing")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args(argv)

    print("=== Phase 9 Layer 6a — CORTEX K=7 internal bank build ===")
    for required in (SN1_PHI, SPIKE_CASES, S_J_TABLE_K7, SEGMENTS_CSV, EEG_BANK):
        if not required.exists():
            print(f"ERROR: required input missing: {required}")
            return 2

    print(f"\nStage 1 — stratified spike sample (target n={args.n_spike})")
    picks = _stratified_spike_sample(n_target=args.n_spike, seed=args.seed)
    print(f"  selected: {len(picks)} segments")
    print(f"  s_mean range: [{picks['s_mean'].min():.3f}, "
          f"{picks['s_mean'].max():.3f}]")
    print(f"  n_raters range: [{picks['n_raters'].min()}, "
          f"{picks['n_raters'].max()}]")

    print(f"\nStage 2 — SN1 segment metadata lookup")
    meta = _sn1_segment_meta()
    missing_meta = [s for s in picks["seg_id"] if s not in meta]
    if missing_meta:
        print(f"  WARN: {len(missing_meta)} segs without segments.csv metadata: "
              f"{missing_meta[:5]}")
    print(f"  metadata covered: {len(picks) - len(missing_meta)} / {len(picks)}")

    if args.dry_run:
        print("\n--dry-run: skipping write")
        return 0

    print(f"\nStage 3 — backup existing bank + fetch SN1 signals + write")
    backup_path = EEG_BANK.parent / "eeg_bank_v1.1.0_legacy.h5"
    if not backup_path.exists():
        print(f"  backing up {EEG_BANK} → {backup_path}")
        shutil.copy2(EEG_BANK, backup_path)
    else:
        print(f"  backup exists: {backup_path} (not overwritten)")

    # Open SN1 in read mode + target bank in append/replace mode
    print(f"  opening SN1 source: {SN1_PHI}")
    with h5py.File(SN1_PHI, "r") as sn1, h5py.File(EEG_BANK, "a") as dst:
        # Remove existing spike/ group
        if "spike" in dst:
            print(f"  removing existing spike/ group ({len(dst['spike'])} segs)")
            del dst["spike"]
        spike_grp = dst.create_group("spike")

        # Bulk-load SN1 signals + n_samples
        signals = sn1["/eeg/signals"]
        n_samples = sn1["/eeg/n_samples"][:]

        n_written = 0
        for _, row in picks.iterrows():
            sid = int(row["seg_id"])
            ns = int(n_samples[sid])
            sig = signals[sid, :ns, :].astype(np.float32)  # (ns, 20)
            # Transpose to match CORTEX convention (channels, samples)
            sig_t = sig.T  # (20, ns)
            g = spike_grp.create_group(str(sid))
            g.create_dataset("eeg10s", data=sig_t,
                             compression="gzip", compression_opts=4)
            # Attributes (calibration + provenance)
            g.attrs["seg_id"] = sid
            g.attrs["test_class"] = "spike"
            g.attrs["s_mean"] = float(row["s_mean"])
            g.attrs["s_sd"] = float(row["s_sd"])
            g.attrs["n_raters"] = int(row["n_raters"])
            g.attrs["pos_rate"] = float(row["pos_rate"])
            m = meta.get(sid, {})
            g.attrs["source_dataset"] = m.get("source_dataset", "")
            g.attrs["pat_age"] = m.get("pat_age", float("nan"))
            g.attrs["sex"] = m.get("sex", "")
            n_written += 1
        # Top-level attrs update
        dst.attrs["n_spike"] = n_written
        dst.attrs["schema_version"] = "3"  # v1.2.0+ phase-9
        dst.attrs["k7_bank_build_utc"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds")
        print(f"  wrote {n_written} calibrated SN1 spike segments")

    print(f"\nStage 4 — verify")
    with h5py.File(EEG_BANK, "r") as f:
        print(f"  iiic count: {len(f['iiic'])}")
        print(f"  spike count: {len(f['spike'])}")
        first_spike = list(f["spike"].keys())[0]
        s = f["spike"][first_spike]
        print(f"  first spike: id={first_spike}, "
              f"shape={s['eeg10s'].shape}, dtype={s['eeg10s'].dtype}")
        print(f"  first spike attrs:")
        for k, v in s.attrs.items():
            print(f"    {k}: {v!r}")
        print(f"  top-level attrs: {dict(f.attrs)}")
    print("\n=== Phase 9 Layer 6a — CORTEX K=7 bank build COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
