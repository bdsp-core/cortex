"""Phase 7 sub-step 3-B — D6 strict-A deployment replay driver.

Drives the deployment engine (`simulate_candidate` from
`deployment/simulate_test.py`) with held-out *real* rater response
sequences instead of Bernoulli draws. User-locked design (2026-05-19):

  • Q3 design = strict constrained-bank ("Design A"): the engine
    selects EV-optimal from the rater's actually-scored segments;
    Y is the rater's recorded value in `data/labels/labels.csv`;
    if the rater's personal bank exhausts on a task before the
    engine reaches a decision (n_per_task < N_min_per_task=10),
    that task's verdict is REFER (existing engine semantics).
  • Q4 coverage = BOTH per-task (~14,823 cells at floor=10) and
    per-candidate (~21 raters with ≥10/task on ALL 7).

Per-task replay (interpretation i, user-locked): the K=7 engine is
run with `bank_by_task[k_target] = rater's strict-A bank`,
`bank_by_task[k_other] = empty` for the other 6 tasks. The engine's
`select_next_case` silently skips empty banks; the 6 empty-bank
tasks naturally end as REFER (n_per_task < N_min_per_task=10 →
cannot become pass/fail). Only `k_target`'s verdict is reported in
the per-task output.

Per-candidate replay: same K=7 engine, all 7 task banks populated
from the rater's strict-A banks.

Outputs (under `results/replay/`):
  • deployment_replay_per_task.csv — one row per (rater, task)
  • deployment_replay_per_candidate.csv — one row per (rater)
  • deployment_replay_run_summary.json — config + headline aggregates

The fitted-θ Bernoulli paired comparator (Q3) is in 7.3-C. This
sub-step (7.3-B) ships the headline strict-A real-rater arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent
for _p in (str(_REPO), str(_REPO / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deployment.simulate_test import (  # noqa: E402
    load_deployment, simulate_candidate, TestConfig,
    deployment_task_names,
)

# Headline floor: deployment N_min_per_task = 10 (user Q2, 2026-05-19).
DEFAULT_FLOOR = 10
# Supplementary strata (Q2: report at multiple floors)
STRATA = (1, 10, 20, 50, 100)

REPLAY_BANK = _REPO / "data" / "replay" / "rater_replay_bank.csv.gz"
REPLAY_SUMMARY = _REPO / "data" / "replay" / "rater_replay_summary.csv"
OUT_DIR = _REPO / "results" / "replay"


def _rel(p: Path) -> str:
    """Best-effort relative-to-repo path display; falls back to str(p)
    when the path is outside the repo (e.g. --out-dir /tmp/...)."""
    try:
        return str(Path(p).resolve().relative_to(_REPO))
    except (ValueError, AttributeError):
        return str(p)


def _load_replay_bank() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load the 7.3-A artifacts + the engine's authoritative task order."""
    if not (REPLAY_BANK.is_file() and REPLAY_SUMMARY.is_file()):
        raise SystemExit(
            f"  ✗ replay bank artifacts missing — run\n"
            f"  python -m pipeline.replay.build_rater_replay_bank")
    print(f"  loading {REPLAY_BANK.relative_to(_REPO)}", flush=True)
    bank = pd.read_csv(REPLAY_BANK)
    print(f"  loading {REPLAY_SUMMARY.relative_to(_REPO)}", flush=True)
    summary = pd.read_csv(REPLAY_SUMMARY)
    tasks = deployment_task_names()
    print(f"  K=7 task order (from frozen Σ): {tasks}", flush=True)
    return bank, summary, tasks


def _bank_for_rater_task(bank: pd.DataFrame, rater_id: int,
                         task: str) -> pd.DataFrame:
    """The rater's strict-A bank for one task — exactly the segs
    the rater scored on this task, with their s_mean/s_sd from the
    K=7 frozen production bank. Returns a DataFrame with the
    columns simulate_candidate's `select_next_case` consumes
    (seg_id, s_mean, s_sd)."""
    sub = bank[(bank["rater_id"] == rater_id) & (bank["task"] == task)]
    return sub[["seg_id", "s_mean", "s_sd"]].reset_index(drop=True)


def _y_lookup_for_rater(bank: pd.DataFrame, rater_id: int,
                        tasks: List[str]) -> "_RaterYLookup":
    """Pre-build the rater's per-(task, seg_id) → y lookup as a
    dict-of-dicts (fast in the engine's inner loop)."""
    sub = bank[bank["rater_id"] == rater_id]
    lookup: Dict[int, Dict[int, int]] = {ki: {} for ki in range(len(tasks))}
    for _, row in sub.iterrows():
        task = row["task"]
        if task not in tasks:
            continue
        k = tasks.index(task)
        lookup[k][int(row["seg_id"])] = int(row["y"])
    return _RaterYLookup(lookup, rater_id)


class _RaterYLookup:
    """Callable Y-lookup for the engine's `y_source=` hook.

    Engine signature: `y_source(k, seg, s_val) -> int`. We ignore
    s_val (the engine already used it for item selection) and look
    up the rater's recorded Y on (k, seg_id). If a seg is missing
    from the lookup we raise — this is a contract violation
    (strict-A guarantees every selected seg is in the rater's bank
    because the bank IS the rater's scored-segs subset).
    """
    __slots__ = ("table", "rater_id", "n_calls", "n_pos")

    def __init__(self, table: Dict[int, Dict[int, int]], rater_id: int):
        self.table = table
        self.rater_id = rater_id
        self.n_calls = 0
        self.n_pos = 0

    def __call__(self, k: int, seg: int, s_val: float) -> int:
        y = self.table[k].get(int(seg))
        if y is None:
            raise RuntimeError(
                f"y_source contract violation: rater {self.rater_id} "
                f"has no recorded Y for (k={k}, seg={seg}) — engine "
                f"selected a seg outside the strict-A bank")
        self.n_calls += 1
        self.n_pos += y
        return int(y)


def replay_one_per_task(rater_id: int, task: str,
                        bank: pd.DataFrame,
                        Sigma_prior: np.ndarray,
                        ell_star: np.ndarray,
                        tasks: List[str], cfg: TestConfig,
                        seed: int = 0) -> Dict[str, Any]:
    """Per-task replay: K=7 engine, all-other-task banks empty;
    only `task` is driven by the rater's strict-A bank. Returns a
    single-row dict with the per-task headline + diagnostics."""
    k_target = tasks.index(task)
    rater_bank = _bank_for_rater_task(bank, rater_id, task)
    n_segs = len(rater_bank)
    bank_by_task: Dict[int, pd.DataFrame] = {}
    empty = pd.DataFrame(
        columns=["seg_id", "s_mean", "s_sd"])
    for k_idx, t_name in enumerate(tasks):
        bank_by_task[k_idx] = (
            rater_bank if t_name == task else empty)

    y_lookup = _y_lookup_for_rater(bank, rater_id, tasks)
    # true_theta not consumed when y_source is provided; pass zeros
    # so the record-keeping fields are well-defined.
    true_theta = np.zeros(Sigma_prior.shape[0])

    # Pre-mark the 6 non-target tasks as 'refer' so the engine's
    # `select_next_case` n_min_per_task constraint doesn't force it
    # to try selecting from their empty banks (the pathology found
    # in the 7.3-B smoke; documented in PHASE7_REPLAY_DESIGN.md §11).
    init_decision = ["refer"] * len(tasks)
    init_decision[k_target] = "pending"

    rng = np.random.default_rng(seed)
    state = simulate_candidate(
        true_theta, Sigma_prior, ell_star, bank_by_task, cfg,
        rng=rng, y_source=y_lookup,
        initial_decision=init_decision)

    decision = state.decision[k_target]
    n_per_task = int(state.n_per_task[k_target])
    bank_exhausted = (n_segs > 0
                      and n_per_task >= n_segs
                      and decision == "refer")
    pos_rate_real = (y_lookup.n_pos / max(1, y_lookup.n_calls))
    return {
        "rater_id": int(rater_id),
        "task": task,
        "decision": decision,
        "n_per_task": n_per_task,
        "n_scored_segs": int(n_segs),
        "bank_exhausted": bool(bank_exhausted),
        "n_engine_calls": int(y_lookup.n_calls),
        "pos_rate_real": float(pos_rate_real),
        "seed": int(seed),
    }


def replay_one_per_candidate(rater_id: int,
                             bank: pd.DataFrame,
                             Sigma_prior: np.ndarray,
                             ell_star: np.ndarray,
                             tasks: List[str], cfg: TestConfig,
                             seed: int = 0) -> Dict[str, Any]:
    """Per-candidate replay: K=7 engine, all 7 task banks populated
    from the rater's strict-A banks. Returns a single-row dict with
    the full 7-task decision array + diagnostics."""
    bank_by_task: Dict[int, pd.DataFrame] = {}
    sizes: Dict[str, int] = {}
    for k_idx, t_name in enumerate(tasks):
        rb = _bank_for_rater_task(bank, rater_id, t_name)
        bank_by_task[k_idx] = rb
        sizes[t_name] = len(rb)

    y_lookup = _y_lookup_for_rater(bank, rater_id, tasks)
    true_theta = np.zeros(Sigma_prior.shape[0])

    rng = np.random.default_rng(seed)
    state = simulate_candidate(
        true_theta, Sigma_prior, ell_star, bank_by_task, cfg,
        rng=rng, y_source=y_lookup)

    row: Dict[str, Any] = {
        "rater_id": int(rater_id),
        "seed": int(seed),
        "n_engine_calls_total": int(y_lookup.n_calls),
        "pos_rate_real_total": float(
            y_lookup.n_pos / max(1, y_lookup.n_calls)),
    }
    for k_idx, t_name in enumerate(tasks):
        row[f"decision_{t_name}"] = state.decision[k_idx]
        row[f"n_per_task_{t_name}"] = int(state.n_per_task[k_idx])
        row[f"n_scored_segs_{t_name}"] = int(sizes[t_name])
        row[f"bank_exhausted_{t_name}"] = bool(
            sizes[t_name] > 0
            and state.n_per_task[k_idx] >= sizes[t_name]
            and state.decision[k_idx] == "refer")
    # roll-up: any task refer counts as not-all-pass
    all_pass = all(state.decision[k] == "pass" for k in range(len(tasks)))
    any_fail = any(state.decision[k] == "fail" for k in range(len(tasks)))
    row["all_pass"] = bool(all_pass)
    row["any_fail"] = bool(any_fail)
    return row


def _cohorts(summary: pd.DataFrame, tasks: List[str],
             floor: int) -> Tuple[pd.DataFrame, List[int]]:
    """Compute the per-task cohort and the per-candidate cohort at
    the given floor. Returns (per_task_cells_df, per_candidate_rater_ids)."""
    per_task = summary[summary["n_segs"] >= floor].copy()
    # per-candidate: raters with ≥ floor segs on ALL 7 tasks
    piv = summary.pivot_table(
        index="rater_id", columns="task", values="n_segs",
        fill_value=0)
    has_all = piv[tasks].ge(floor).all(axis=1)
    per_cand = piv[has_all].index.tolist()
    return per_task, per_cand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                    help="Headline N_min floor for per-task cohort "
                         "(default 10 = deployment N_min_per_task).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (replay is mostly deterministic; "
                         "seed governs the engine's tie-breaking).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional per-cohort cap on cells / raters "
                         "for fast smoke; default = full cohort.")
    ap.add_argument("--out-dir", type=str,
                    default=str(OUT_DIR))
    cli = ap.parse_args()

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"=== deployment replay (strict-A, floor=N_min={cli.floor}) ===",
          flush=True)
    Sigma_prior, _, ell_star = load_deployment()
    cfg = TestConfig.from_yaml()
    print(f"  cfg: N_min={cfg.N_min} N_max={cfg.N_max} "
          f"N_min_per_task={cfg.N_min_per_task} "
          f"N_max_per_task={cfg.N_max_per_task}", flush=True)
    bank, summary, tasks = _load_replay_bank()

    per_task_cells, per_cand_raters = _cohorts(summary, tasks, cli.floor)
    if cli.limit is not None:
        per_task_cells = per_task_cells.head(cli.limit)
        per_cand_raters = per_cand_raters[:cli.limit]
    n_pt = len(per_task_cells)
    n_pc = len(per_cand_raters)
    print(f"  per-task cohort at floor={cli.floor}: {n_pt} cells",
          flush=True)
    print(f"  per-candidate cohort at floor={cli.floor}: {n_pc} raters",
          flush=True)

    # ── per-task replay
    print(f"\n=== per-task replay ({n_pt} cells) ===", flush=True)
    per_task_rows: List[Dict[str, Any]] = []
    t_pt = time.time()
    for i, (_, row) in enumerate(per_task_cells.iterrows(), 1):
        out = replay_one_per_task(
            int(row["rater_id"]), row["task"],
            bank, Sigma_prior, ell_star, tasks, cfg, seed=cli.seed)
        out["expertise_level"] = row["expertise_level"]
        per_task_rows.append(out)
        if i % 500 == 0 or i == n_pt:
            print(f"  per-task {i}/{n_pt}  "
                  f"elapsed={time.time() - t_pt:.1f}s", flush=True)
    dt_pt = time.time() - t_pt
    pt_df = pd.DataFrame(per_task_rows)
    pt_path = out_dir / "deployment_replay_per_task.csv"
    pt_df.to_csv(pt_path, index=False)
    print(f"  → {_rel(pt_path)}  ({len(pt_df)} rows, {dt_pt:.1f}s)",
          flush=True)

    # ── per-candidate replay
    print(f"\n=== per-candidate replay ({n_pc} raters) ===", flush=True)
    per_cand_rows: List[Dict[str, Any]] = []
    t_pc = time.time()
    for i, rid in enumerate(per_cand_raters, 1):
        out = replay_one_per_candidate(
            int(rid), bank, Sigma_prior, ell_star, tasks, cfg, seed=cli.seed)
        # tier
        tier = summary[summary["rater_id"] == rid]["expertise_level"].iloc[0]
        out["expertise_level"] = tier
        per_cand_rows.append(out)
        print(f"  per-candidate {i}/{n_pc}  rater={rid}  "
              f"elapsed={time.time() - t_pc:.1f}s", flush=True)
    dt_pc = time.time() - t_pc
    pc_df = pd.DataFrame(per_cand_rows)
    pc_path = out_dir / "deployment_replay_per_candidate.csv"
    pc_df.to_csv(pc_path, index=False)
    print(f"  → {_rel(pc_path)}  ({len(pc_df)} rows, {dt_pc:.1f}s)",
          flush=True)

    # ── headline aggregates
    pt_agg = pt_df.groupby("task")["decision"].value_counts().unstack(
        fill_value=0)
    pt_agg["total"] = pt_agg.sum(axis=1)
    for col in ("pass", "fail", "refer"):
        if col in pt_agg.columns:
            pt_agg[f"{col}_share"] = pt_agg[col] / pt_agg["total"]
    headline = {
        "n_per_task_cells": int(len(pt_df)),
        "n_per_candidate_raters": int(len(pc_df)),
        "floor": int(cli.floor),
        "seed": int(cli.seed),
        "wall_s": round(time.time() - t0, 1),
        "per_task_decision_distribution": _df_to_jsonable(pt_agg),
        "per_candidate_all_pass": int(pc_df["all_pass"].sum()),
        "per_candidate_any_fail": int(pc_df["any_fail"].sum()),
    }
    sum_path = out_dir / "deployment_replay_run_summary.json"
    with open(sum_path, "w") as f:
        json.dump(headline, f, indent=2, default=str)
    print(f"\n  → {_rel(sum_path)}", flush=True)
    print(f"  wall={headline['wall_s']:.1f}s", flush=True)
    return 0


def _df_to_jsonable(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx in df.index:
        row = df.loc[idx].to_dict()
        out[str(idx)] = {str(k): (float(v) if isinstance(v, float)
                                  else int(v) if isinstance(v, (int,
                                                                np.integer))
                                  else str(v))
                        for k, v in row.items()}
    return out


if __name__ == "__main__":
    sys.exit(main())
