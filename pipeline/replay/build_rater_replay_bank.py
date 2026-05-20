"""Phase 7 sub-step 3-A — build the per-rater replay bank.

The strict constrained-bank ("Design A") replay (user-locked
2026-05-19) needs, per rater r and task k:

  • the seg_ids the rater scored on task k
  • the rater's real binary Y on each (seg_id, k)
  • the engine inputs `s_mean`, `s_sd` for each seg_id on task k
    (from the K=7 frozen `data/deployment_prior/case_bank.csv`)

This script produces TWO outputs at `data/replay/`:

  rater_replay_bank.csv.gz   long-form (rater_id, task, seg_id, y,
                             s_mean, s_sd) ready for the engine
                             drivers (7.3-B, 7.3-C). Gzipped (~100MB
                             uncompressed → ~20MB gz).

  rater_replay_summary.csv   per-(rater, task) summary: n_segs,
                             n_pos, sum_y, expertise_level (joined
                             from raters.csv) — fast inventory for
                             the engine drivers + tests + reporting.

Erratum-correct mapping: for `label_type='pattern_class'`,
`{bipd,birds,other}→other` (Phase-3 D5; AUDIT §6). Spike values are
typed-coerced to {0, 1} only (Phase-3 v13 build_calibration_inputs
combined_spike binary convention, also Phase-4.6-A clean-sn1 spike).

Reads only: `data/labels/labels.csv`, `data/labels/raters.csv`,
`data/deployment_prior/case_bank.csv`. Writes only `data/replay/`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent

# Mode-A K=7 task list (the 7 production tasks). Matches
# deployment_task_names() at deployment/simulate_test.py:112 (slot
# names parsed from Sigma).
TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]

# Erratum-correct `pattern_class` value → task mapping (Phase-3 D5
# / AUDIT §6). Three "other-class" raw values fold to `other`.
IIIC_MAP = {"bipd": "other", "birds": "other"}


def _build(labels_csv: Path, raters_csv: Path, case_bank_csv: Path,
           out_dir: Path) -> dict:
    """Produce the rater-replay long-form bank + summary table.

    Returns a dict of headline counts for caller reporting + tests.
    """
    t0 = time.time()
    print(f"[7.3-A] reading {labels_csv.relative_to(_REPO)}", flush=True)
    lab = pd.read_csv(labels_csv, low_memory=False)
    ratr = pd.read_csv(raters_csv)
    case_bank = pd.read_csv(case_bank_csv)
    print(f"[7.3-A] labels={len(lab):,}  raters={len(ratr):,}  "
          f"case_bank={len(case_bank):,}", flush=True)

    # ── spike: typed-coerced to {0,1}; reference-correct combined_spike
    spike = lab[lab["label_type"] == "spike"].copy()
    spike["y"] = pd.to_numeric(spike["value"], errors="coerce")
    spike = spike.dropna(subset=["y"]).astype({"y": int})
    spike = spike[spike["y"].isin([0, 1])]
    spike["task"] = "spike"

    # ── IIIC: erratum-correct {bipd,birds,other}→other; per-task Y =
    # int(value == task) (the canonical Phase-3 binary collapse).
    pc = lab[lab["label_type"] == "pattern_class"].copy()
    pc["v2"] = pc["value"].replace(IIIC_MAP)
    iiic_tasks = [t for t in TASKS if t != "spike"]
    frames = [spike[["rater_id", "seg_id", "task", "y"]]]
    for task in iiic_tasks:
        sub = pc.copy()
        sub["task"] = task
        sub["y"] = (sub["v2"] == task).astype(int)
        frames.append(sub[["rater_id", "seg_id", "task", "y"]])

    allY = pd.concat(frames, ignore_index=True)
    # Drop duplicate (rater, seg, task) rows (a rater scoring the
    # same seg twice on the same task is degenerate; keep the first).
    allY = allY.drop_duplicates(["rater_id", "seg_id", "task"],
                                keep="first")
    print(f"[7.3-A] long-form (rater,seg,task) triples: "
          f"{len(allY):,}", flush=True)

    # ── attach engine inputs (s_mean, s_sd) from case_bank
    bank_keys = case_bank[["task", "seg_id", "s_mean", "s_sd"]]
    bank_merge = allY.merge(bank_keys, on=["task", "seg_id"], how="left")
    # A scored seg may not appear in case_bank if it's not in the
    # production K=7 bank (e.g. an iiic seg that didn't make the
    # joint-fit filter). Drop those rows — the engine can't process
    # them (no s_mean signal). Report the drop count.
    n_pre = len(bank_merge)
    bank_merge = bank_merge.dropna(subset=["s_mean", "s_sd"])
    n_post = len(bank_merge)
    n_drop = n_pre - n_post
    print(f"[7.3-A] joined with case_bank → kept {n_post:,} rows "
          f"(dropped {n_drop:,} = {100*n_drop/max(1,n_pre):.2f}% "
          f"with no s_mean signal in the K=7 bank)", flush=True)

    # ── per-(rater, task) summary
    g = bank_merge.groupby(["rater_id", "task"])
    summary = g.agg(
        n_segs=("seg_id", "size"),
        n_pos=("y", "sum"),
    ).reset_index()
    # join tier (expertise_level) from raters.csv
    summary = summary.merge(
        ratr[["rater_id", "canonical_name", "expertise_level"]],
        on="rater_id", how="left")
    summary["expertise_level"] = summary["expertise_level"].fillna(
        "unknown")

    # ── persist
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "rater_replay_bank.csv.gz"
    pkl_path = out_dir / "rater_replay_bank.indexed.pkl"
    sum_path = out_dir / "rater_replay_summary.csv"
    bank_merge_out = bank_merge[
        ["rater_id", "task", "seg_id", "y", "s_mean", "s_sd"]
    ].sort_values(["rater_id", "task", "seg_id"]).reset_index(drop=True)
    bank_merge_out.to_csv(bank_path, index=False, compression="gzip")
    # Phase 7 sub-3-C: also persist a pre-indexed pickle (set_index
    # by [rater_id, task], pre-sorted) — ~60× faster to load in
    # workers than the gz CSV. Same content; consumed by
    # `pipeline/replay/_common.load_replay_bank_cached`.
    bank_merge_out.set_index(["rater_id", "task"]).to_pickle(pkl_path)
    summary.sort_values(["rater_id", "task"]).to_csv(sum_path, index=False)

    dt = time.time() - t0
    # ── headline inventory for caller + tests
    counts = {
        "n_obs_total": int(n_post),
        "n_obs_per_task": (
            bank_merge.groupby("task").size().to_dict()),
        "n_rater_task_cells": int(len(summary)),
        "n_rater_task_cells_at_floor": {
            f: int((summary["n_segs"] >= f).sum())
            for f in (1, 10, 20, 50, 100)
        },
        "n_per_candidate_at_floor_10": _per_candidate_cohort(
            summary, n_min_per_task=10),
        "n_per_candidate_at_floor_20": _per_candidate_cohort(
            summary, n_min_per_task=20),
        "n_dropped_no_signal": int(n_drop),
        "duration_s": round(dt, 2),
        "outputs": {
            "bank": str(bank_path.relative_to(_REPO)),
            "summary": str(sum_path.relative_to(_REPO)),
        },
    }
    print(f"[7.3-A] DONE in {dt:.1f}s", flush=True)
    print(f"  bank  → {bank_path.relative_to(_REPO)}", flush=True)
    print(f"  summary → {sum_path.relative_to(_REPO)}", flush=True)
    print(f"  rater×task cells at floor=10: "
          f"{counts['n_rater_task_cells_at_floor'][10]:,}", flush=True)
    print(f"  per-candidate cohort at floor=10 (all 7 tasks): "
          f"{counts['n_per_candidate_at_floor_10']['n']} raters",
          flush=True)
    return counts


def _per_candidate_cohort(summary: pd.DataFrame,
                          n_min_per_task: int) -> dict:
    """Raters with ≥ n_min_per_task on ALL 7 tasks (the headline
    per-candidate cohort). Returns count + per-tier breakdown."""
    piv = summary.pivot_table(
        index="rater_id", columns="task", values="n_segs", fill_value=0)
    if not set(TASKS).issubset(piv.columns):
        return {"n": 0, "by_tier": {}}
    has_all = (piv[TASKS] >= n_min_per_task).all(axis=1)
    qualifiers = piv[has_all].index
    if len(qualifiers) == 0:
        return {"n": 0, "by_tier": {}}
    tiers = summary[summary["rater_id"].isin(qualifiers)].drop_duplicates(
        "rater_id")[["rater_id", "expertise_level"]]
    return {
        "n": int(len(qualifiers)),
        "by_tier": tiers["expertise_level"].value_counts().to_dict(),
    }


def main() -> int:
    labels = _REPO / "data" / "labels" / "labels.csv"
    raters = _REPO / "data" / "labels" / "raters.csv"
    case_bank = _REPO / "data" / "deployment_prior" / "case_bank.csv"
    out_dir = _REPO / "data" / "replay"
    for p in (labels, raters, case_bank):
        if not p.exists():
            print(f"  ✗ MISSING: {p}", file=sys.stderr)
            return 1
    counts = _build(labels, raters, case_bank, out_dir)
    import json
    print()
    print(json.dumps(counts, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
