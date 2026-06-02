"""Build the v1.3.6 internal CORTEX bank: 100 IIIC/pattern-class (600) + 100
spike, replacing the v1.2.x 300+50 bank.

Per the bundle-size sweep (sim_v1_3_5/run_bank_size_sweep.py): at the NEJM AI
alpha=0.05, 100 IIIC/domain + MAX_Q=500 gives ~92% per-domain PASS/FAIL for clear
candidates at ~1.0 GB. Spike segs are ~free (0.10 MB vs 1.69 MB/IIIC) -> 100.

Method: SUBSET the frozen 35k production bank (data/production_bank/
eeg_bank_production.h5) — it already holds the EEG + spectrograms and its
pattern_class vocabulary matches the engine. Selection is stratified-by-s_mean
(10 quantile strata x 10), matching cortex_session_bank_fetch.sample_session_
segments so the validated resolution holds. The engine reads per-task signals
from segment_signals.csv keyed by seg_id (cortex_engine_inputs_k7.py:222), so
every selected seg_id is checked to exist there.

Writes data/eeg_bank_v1_3_6.h5 (temp); the caller validates then swaps.

    .venv/bin/python scripts/build_internal_bank_v1_3_6.py --dry-run
    .venv/bin/python scripts/build_internal_bank_v1_3_6.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

REPO = Path(__file__).resolve().parent.parent
PROD_H5 = REPO / "data" / "production_bank" / "eeg_bank_production.h5"
PROD_MANIFEST = REPO / "data" / "production_bank" / "MANIFEST.json"
SIGNALS_CSV = REPO / "data" / "labels" / "segment_signals.csv"
OUT_H5 = REPO / "data" / "eeg_bank_v1_3_6.h5"

N_IIIC_PER_CLASS = 100
N_SPIKE = 100
N_STRATA = 10
SEED = 42
# IIIC pattern_class -> own difficulty code (for stratification by s_mean_<code>)
CLASS_TO_CODE = {"seizure": "sz", "lpd": "lpd", "gpd": "gpd",
                 "lrda": "lrda", "grda": "grda", "other": "iic"}


def _stratified(seg_ids, values, n_target, rng):
    """Stratified-by-quantile sample (mirrors sample_session_segments: 10 strata
    x n_target/10 each)."""
    seg_ids = np.asarray(seg_ids)
    values = np.asarray(values, float)
    fin = np.isfinite(values)
    seg_ids, values = seg_ids[fin], values[fin]
    if len(seg_ids) <= n_target:
        return list(seg_ids)
    bins = np.unique(np.quantile(values, np.linspace(0, 1, N_STRATA + 1)))
    per = max(1, n_target // (len(bins) - 1))
    strat = np.clip(np.digitize(values, bins[1:-1], right=False), 0, len(bins) - 2)
    out = []
    for s in range(len(bins) - 1):
        idx = np.flatnonzero(strat == s)
        if idx.size:
            out.extend(seg_ids[rng.choice(idx, size=min(per, idx.size), replace=False)])
    if len(out) < n_target:                      # top up from remainder
        rest = [i for i in seg_ids if i not in set(out)]
        if rest:
            out.extend(np.asarray(rest)[rng.choice(len(rest),
                       size=min(n_target - len(out), len(rest)), replace=False)])
    return [int(x) for x in out[:n_target]]


def select():
    segs = json.loads(PROD_MANIFEST.read_text())["segments"]
    rng = np.random.default_rng(SEED)
    chosen_iiic, chosen_spike = {}, []
    by_class = {}
    spike_ids, spike_sm = [], []
    for s in segs:
        if s["family"] == "spike":
            spike_ids.append(int(s["seg_id"])); spike_sm.append(s.get("s_mean"))
        else:
            by_class.setdefault(s["pattern_class"], []).append(s)
    for cls, code in CLASS_TO_CODE.items():
        pool = by_class.get(cls, [])
        ids = [int(s["seg_id"]) for s in pool]
        sm = [s.get(f"s_mean_{code}") for s in pool]
        pick = _stratified(ids, sm, N_IIIC_PER_CLASS, rng)
        chosen_iiic[cls] = pick
        print(f"  IIIC {cls:>8} (code {code}): {len(pick)}/{len(pool)} selected")
    chosen_spike = _stratified(spike_ids, spike_sm, N_SPIKE, rng)
    print(f"  spike: {len(chosen_spike)}/{len(spike_ids)} selected")
    return chosen_iiic, chosen_spike


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for p in (PROD_H5, PROD_MANIFEST, SIGNALS_CSV):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    print("=== selecting 100 IIIC/class + 100 spike from the frozen 35k bank ===")
    chosen_iiic, chosen_spike = select()
    iiic_ids = [sid for ids in chosen_iiic.values() for sid in ids]
    all_ids = iiic_ids + chosen_spike
    print(f"  TOTAL: {len(iiic_ids)} IIIC + {len(chosen_spike)} spike = {len(all_ids)}")

    # engine requirement: every bank seg_id must have a segment_signals.csv row
    sig_idx = set(pd.read_csv(SIGNALS_CSV, usecols=["seg_id"])["seg_id"].astype(int))
    missing = [s for s in all_ids if s not in sig_idx]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} selected segs not in "
                         f"{SIGNALS_CSV.name}: {missing[:5]} — engine would reject")
    print(f"  signals check: all {len(all_ids)} seg_ids present in segment_signals.csv")

    est_mb = len(iiic_ids) * 1.69 + len(chosen_spike) * 0.10
    print(f"  estimated bank payload ~{est_mb:.0f} MB (IIIC dominates)")
    if args.dry_run:
        print("--dry-run: no h5 written"); return

    print(f"=== copying segment groups -> {OUT_H5.name} ===")
    with h5py.File(PROD_H5, "r") as src, h5py.File(OUT_H5, "w") as dst:
        gi = dst.create_group("iiic"); gs = dst.create_group("spike")
        for sid in iiic_ids:
            src.copy(src[f"iiic/{sid}"], gi, name=str(sid))
        for sid in chosen_spike:
            src.copy(src[f"spike/{sid}"], gs, name=str(sid))
        dst.attrs["schema_version"] = "3"
        dst.attrs["build"] = "v1.3.6 internal: 100 IIIC/class + 100 spike (subset of production_v1)"
        dst.attrs["n_iiic"] = len(iiic_ids); dst.attrs["n_spike"] = len(chosen_spike)
    sz = OUT_H5.stat().st_size / 1e6
    print(f"  wrote {OUT_H5} — {sz:.0f} MB, {len(iiic_ids)} IIIC + {len(chosen_spike)} spike")
    print("  NEXT: validate with build_k7_engine_inputs, then back up + swap data/eeg_bank.h5")


if __name__ == "__main__":
    main()
