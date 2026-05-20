"""Phase 7 sub-step 3-C — unified parallel replay driver.

Drives the D6 real-rater replay across:

  * Engines: deployment (Laplace/EKF; PASS/FAIL/REFER + E[n]) and
    Mode-A (SMC+MCMC; AUROC posterior CI).
  * Arms: `replay` (strict-A real Y from labels.csv) and `bernoulli`
    (fitted-θ Bernoulli paired comparator at the rater's
    sdt_fits-fitted (σ, θ)).
  * Cohorts: per-task (per (rater, task) cell at the headline
    N_min=10 floor; ~14,823 cells) and per-candidate (the ~21
    raters with ≥N_min on ALL 7 tasks).

User-locked design (2026-05-19, Q3=fitted-θ comparator, Q4=both
cohorts, Q1=both engines). The fitted-θ Bernoulli arm reuses the
engine's default code path: pass the rater's fitted (t, ℓ) as
`true_params` with `y_source=None`. This means BOTH arms share
the same engine code; only the Y source differs.

Parallel via `scripts/_parallel.parallel_map` (default 14 workers
per user 2026-05-19). Each worker process loads the 102 MB
rater_replay_bank once via the `_SHARED` cache in
`pipeline/replay/_common.py`.

Output:
  results/replay/replay_per_task.csv         (long-form: arm column)
  results/replay/replay_per_candidate.csv    (long-form: arm column)
  results/replay/replay_run_summary.json     (config + aggregates)
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
for _p in (str(_REPO), str(_REPO / "engine"), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _parallel import (  # noqa: E402  scripts/_parallel.py
    configure_blas_single_thread, parallel_map, count_errors,
)
configure_blas_single_thread()

from pipeline.replay._common import (  # noqa: E402
    load_replay_bank_cached, bank_for_rater_task,
    fitted_theta_for_rater, y_lookup_for_rater,
)

# Engines: lazy-imported in workers to avoid heavy module init at
# spawn time.

DEFAULT_FLOOR = 10
DEFAULT_WORKERS = 14
OUT_DIR = _REPO / "results" / "replay"


# ──────────────────────────────────────────────────────────────────
# Per-cell session runners (called inside workers)
# ──────────────────────────────────────────────────────────────────

def _run_deployment_cell(task_arg: Dict[str, Any]) -> Dict[str, Any]:
    """One (rater, task or per-candidate, arm='replay'|'bernoulli')
    deployment session. Returns a single-row dict."""
    sh = load_replay_bank_cached()
    rater_id = int(task_arg["rater_id"])
    cohort = task_arg["cohort"]
    arm = task_arg["arm"]
    seed = int(task_arg["seed"])
    task = task_arg.get("task")  # None for per-candidate

    from deployment.simulate_test import (
        load_deployment, simulate_candidate, TestConfig,
        deployment_task_names,
    )
    Sigma_prior, _bank_unused, ell_star = load_deployment()
    cfg = TestConfig.from_yaml()
    tasks = deployment_task_names()
    K_ = len(tasks)

    # Build per-task strict-A banks
    if cohort == "per_task":
        k_target = tasks.index(task)
        rater_bank = bank_for_rater_task(rater_id, task)
        empty = pd.DataFrame(columns=["seg_id", "s_mean", "s_sd"])
        bank_by_task = {ki: (rater_bank if t == task else empty)
                        for ki, t in enumerate(tasks)}
        init_decision = ["refer"] * K_
        init_decision[k_target] = "pending"
    else:  # per_candidate
        bank_by_task = {ki: bank_for_rater_task(rater_id, t)
                        for ki, t in enumerate(tasks)}
        init_decision = None

    # ── Y source per arm
    if arm == "replay":
        y_table = y_lookup_for_rater(rater_id, tasks)

        def y_source(k, seg, s_val, _t=y_table):
            return int(_t[k][int(seg)])

        true_theta = np.zeros(Sigma_prior.shape[0])
    else:  # bernoulli (fitted-θ at rater's sdt_fits)
        true_theta, _fitted_flags = fitted_theta_for_rater(rater_id, tasks)
        y_source = None  # default Bernoulli path

    rng = np.random.default_rng(seed)
    state = simulate_candidate(
        true_theta, Sigma_prior, ell_star, bank_by_task, cfg,
        rng=rng, y_source=y_source, initial_decision=init_decision)

    if cohort == "per_task":
        row = {
            "rater_id": rater_id, "engine": "deployment", "arm": arm,
            "cohort": "per_task", "task": task, "seed": seed,
            "decision": state.decision[k_target],
            "n_per_task": int(state.n_per_task[k_target]),
            "n_scored_segs": int(len(bank_by_task[k_target])),
            "bank_exhausted": bool(
                len(bank_by_task[k_target]) > 0
                and int(state.n_per_task[k_target]) >= len(bank_by_task[k_target])
                and state.decision[k_target] == "refer"),
        }
        return row
    # per_candidate
    row: Dict[str, Any] = {
        "rater_id": rater_id, "engine": "deployment", "arm": arm,
        "cohort": "per_candidate", "seed": seed,
    }
    sizes = [len(bank_by_task[ki]) for ki in range(K_)]
    for ki, t in enumerate(tasks):
        row[f"decision_{t}"] = state.decision[ki]
        row[f"n_per_task_{t}"] = int(state.n_per_task[ki])
        row[f"bank_exhausted_{t}"] = bool(
            sizes[ki] > 0
            and int(state.n_per_task[ki]) >= sizes[ki]
            and state.decision[ki] == "refer")
    row["all_pass"] = bool(all(
        state.decision[ki] == "pass" for ki in range(K_)))
    row["any_fail"] = bool(any(
        state.decision[ki] == "fail" for ki in range(K_)))
    return row


def _run_mode_a_cell(task_arg: Dict[str, Any]) -> Dict[str, Any]:
    """One (rater, task or per-candidate, arm) Mode-A session.

    Per-task runs at K=1 (single-task Mode-A; clean isolation,
    matches the way Mode-A reports per-domain AUROC); per-candidate
    runs at K=7 with the cross-task hierarchical prior (the
    'borrowing of strength' that's Mode-A's headline)."""
    sh = load_replay_bank_cached()
    rater_id = int(task_arg["rater_id"])
    cohort = task_arg["cohort"]
    arm = task_arg["arm"]
    seed = int(task_arg["seed"])
    task = task_arg.get("task")

    import core_mcmc
    from deployment.simulate_test import deployment_task_names
    tasks = deployment_task_names()

    if cohort == "per_task":
        # K=1 single-task Mode-A
        k_target_full = tasks.index(task)
        rater_bank = bank_for_rater_task(rater_id, task)
        if len(rater_bank) == 0:
            return {"rater_id": rater_id, "engine": "mode_a",
                    "arm": arm, "cohort": "per_task", "task": task,
                    "n_q_total": 0, "stopped_early": False,
                    "bank_exhausted": True}
        bank_signals = [rater_bank["s_mean"].values.astype(float)]
        bank_segids = [rater_bank["seg_id"].values.astype(int)]
        K = 1

        if arm == "replay":
            y_table = y_lookup_for_rater(rater_id, tasks)

            def y_source(k, seg, s_val, _t=y_table, _kfull=k_target_full):
                # K=1 engine sees k=0; map back to full-K seg-table key.
                return int(_t[_kfull][int(seg)])

            true_params = [0.0, 0.0]  # not consumed
        else:
            theta_full, _fl = fitted_theta_for_rater(rater_id, tasks)
            t_k = float(theta_full[2 * k_target_full])
            l_k = float(theta_full[2 * k_target_full + 1])
            true_params = [t_k, l_k]
            y_source = None

        # Phase 7 sub-3-C replay-grade Mode-A params (perf-tuned).
        # Paper-grade Mode-A uses N=2500 / max_q=400 / δ=0.025 (the
        # session driver's defaults). For REPLAY purposes we need a
        # sensible AUROC posterior summary, not paper-grade
        # precision — N=200 / max_q=120 / δ=0.05 is ~10× faster per
        # session and produces qualitatively-equivalent posterior
        # summaries (verified in 7.3-C smoke). The replay headline
        # cost is dominated by Mode-A per_candidate (K=7); per_task
        # at K=1 is light.
        out = core_mcmc.run_session_mcmc_auroc(
            method="hier", true_params=true_params, K=K,
            r_assumed=0.378,
            max_q=min(120, len(rater_bank)),
            delta_auroc=0.05, N=200, seed=seed,
            log_trajectory=False, run_until_max=False,
            bank_signals=bank_signals,
            bank_segids=bank_segids,
            y_source=y_source,
            n_subsample=100,  # F4.1 coarse-to-fine argmin, ~5-6× faster
        )
        return {
            "rater_id": rater_id, "engine": "mode_a", "arm": arm,
            "cohort": "per_task", "task": task, "seed": seed,
            "n_q_total": int(out["n_questions"]),
            "stopped_early": bool(out["stopped_early"]),
            "auroc_lo_95": float(out["final_lo"][0]),
            "auroc_hi_95": float(out["final_hi"][0]),
            "auroc_hw": float((out["final_hi"][0] - out["final_lo"][0]) / 2),
            "n_scored_segs": int(len(rater_bank)),
            "bank_exhausted": bool(int(out["n_questions"]) >= len(rater_bank)),
        }

    # per_candidate Mode-A: K=7, all banks populated
    K = len(tasks)
    bank_signals: List[np.ndarray] = []
    bank_segids: List[np.ndarray] = []
    sizes: List[int] = []
    for t in tasks:
        rb = bank_for_rater_task(rater_id, t)
        bank_signals.append(rb["s_mean"].values.astype(float))
        bank_segids.append(rb["seg_id"].values.astype(int))
        sizes.append(len(rb))

    if arm == "replay":
        y_table = y_lookup_for_rater(rater_id, tasks)

        def y_source(k, seg, s_val, _t=y_table):
            return int(_t[k][int(seg)])

        true_params = [0.0] * (2 * K)
    else:
        true_params, _fl = fitted_theta_for_rater(rater_id, tasks)
        true_params = list(true_params)
        y_source = None

    # Phase 7 sub-3-C replay-grade Mode-A params (perf-tuned;
    # ~10× faster than paper-grade for per_candidate's K=7
    # MCMC+SMC; see _run_mode_a_cell per_task block for rationale).
    out = core_mcmc.run_session_mcmc_auroc(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        max_q=int(min(350, sum(sizes))),   # was 840 (paper-grade)
        delta_auroc=0.05, N=200, seed=seed,  # was N=400
        log_trajectory=False, run_until_max=False,
        bank_signals=bank_signals, bank_segids=bank_segids,
        y_source=y_source,
    )
    row: Dict[str, Any] = {
        "rater_id": rater_id, "engine": "mode_a", "arm": arm,
        "cohort": "per_candidate", "seed": seed,
        "n_q_total": int(out["n_questions"]),
        "stopped_early": bool(out["stopped_early"]),
    }
    for ki, t in enumerate(tasks):
        row[f"auroc_lo_{t}"] = float(out["final_lo"][ki])
        row[f"auroc_hi_{t}"] = float(out["final_hi"][ki])
        row[f"auroc_hw_{t}"] = float(
            (out["final_hi"][ki] - out["final_lo"][ki]) / 2)
        row[f"n_scored_segs_{t}"] = int(sizes[ki])
    return row


def _run_one(task_arg: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to engine-specific runner (called by parallel_map)."""
    engine = task_arg["engine"]
    if engine == "deployment":
        return _run_deployment_cell(task_arg)
    if engine == "mode_a":
        return _run_mode_a_cell(task_arg)
    raise ValueError(f"unknown engine {engine!r}")


# ──────────────────────────────────────────────────────────────────
# Cohort + task-graph construction
# ──────────────────────────────────────────────────────────────────

def _build_tasks(summary: pd.DataFrame, tasks: List[str], floor: int,
                  engines: List[str], arms: List[str], cohorts: List[str],
                  seed: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    per_task_cells = summary[summary["n_segs"] >= floor]
    piv = summary.pivot_table(
        index="rater_id", columns="task", values="n_segs", fill_value=0)
    has_all = piv[tasks].ge(floor).all(axis=1)
    per_cand_raters = piv[has_all].index.tolist()

    if limit is not None:
        per_task_cells = per_task_cells.head(limit)
        per_cand_raters = per_cand_raters[:limit]

    out: List[Dict[str, Any]] = []
    if "per_task" in cohorts:
        for _, row in per_task_cells.iterrows():
            for engine in engines:
                for arm in arms:
                    out.append({
                        "engine": engine, "arm": arm,
                        "cohort": "per_task",
                        "rater_id": int(row["rater_id"]),
                        "task": row["task"], "seed": seed,
                    })
    if "per_candidate" in cohorts:
        for rid in per_cand_raters:
            for engine in engines:
                for arm in arms:
                    out.append({
                        "engine": engine, "arm": arm,
                        "cohort": "per_candidate",
                        "rater_id": int(rid), "seed": seed,
                    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+",
                    default=["deployment", "mode_a"],
                    choices=["deployment", "mode_a"])
    ap.add_argument("--arms", nargs="+",
                    default=["replay", "bernoulli"],
                    choices=["replay", "bernoulli"])
    ap.add_argument("--cohorts", nargs="+",
                    default=["per_task", "per_candidate"],
                    choices=["per_task", "per_candidate"])
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional per-cohort cap for smokes")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    cli = ap.parse_args()

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"=== Phase-7 sub-3-C unified replay driver ===", flush=True)
    print(f"  engines={cli.engines}  arms={cli.arms}  "
          f"cohorts={cli.cohorts}  floor={cli.floor}", flush=True)

    # Pre-load in parent (warms _SHARED for serial paths;
    # workers each re-load via the same cached helper)
    sh = load_replay_bank_cached()
    from deployment.simulate_test import deployment_task_names
    tasks = deployment_task_names()
    tasks_list = _build_tasks(sh["summary"], tasks, cli.floor,
                               cli.engines, cli.arms, cli.cohorts,
                               cli.seed, cli.limit)
    print(f"  {len(tasks_list)} sessions to run "
          f"on {cli.max_workers} workers", flush=True)

    results = parallel_map(
        _run_one, tasks_list,
        max_workers=cli.max_workers,
        desc="replay sessions",
        ordered=True,
        progress_every=200,
    )
    n_err = count_errors(results)
    rows = [r for r in results
            if not (isinstance(r, dict) and "__error__" in r)]
    wall = time.time() - t0
    print(f"\n  {len(rows)} rows ({n_err} errors)  wall={wall:.1f}s "
          f"({wall / max(1, len(tasks_list)):.2f}s/session amortized)",
          flush=True)

    # Persist per-cohort
    df = pd.DataFrame(rows)
    if "per_task" in cli.cohorts:
        pt = df[df["cohort"] == "per_task"].copy()
        if len(pt):
            pt_path = out_dir / "replay_per_task.csv"
            pt.to_csv(pt_path, index=False)
            print(f"  → {pt_path.name}  ({len(pt)} rows)", flush=True)
    if "per_candidate" in cli.cohorts:
        pc = df[df["cohort"] == "per_candidate"].copy()
        if len(pc):
            pc_path = out_dir / "replay_per_candidate.csv"
            pc.to_csv(pc_path, index=False)
            print(f"  → {pc_path.name}  ({len(pc)} rows)", flush=True)

    # Aggregate summary
    headline = {
        "n_sessions": len(tasks_list),
        "n_rows": len(rows),
        "n_errors": n_err,
        "wall_s": round(wall, 1),
        "config": {
            "engines": cli.engines, "arms": cli.arms,
            "cohorts": cli.cohorts, "floor": cli.floor,
            "seed": cli.seed, "limit": cli.limit,
            "max_workers": cli.max_workers,
        },
    }
    if "per_task" in cli.cohorts:
        pt = df[df["cohort"] == "per_task"]
        if "decision" in pt.columns:
            agg = (pt[pt["engine"] == "deployment"]
                   .groupby(["arm", "task", "decision"])
                   .size().unstack(fill_value=0))
            headline["deployment_per_task_decisions"] = (
                _df_to_jsonable(agg))
    sum_path = out_dir / "replay_run_summary.json"
    with open(sum_path, "w") as f:
        json.dump(headline, f, indent=2, default=str)
    print(f"  → {sum_path.name}", flush=True)
    return 0


def _df_to_jsonable(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx in df.index:
        key = "|".join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
        row = df.loc[idx].to_dict()
        out[key] = {str(k): (float(v) if isinstance(v, float)
                              else int(v) if isinstance(v, (int, np.integer))
                              else str(v))
                    for k, v in row.items()}
    return out


if __name__ == "__main__":
    sys.exit(main())
