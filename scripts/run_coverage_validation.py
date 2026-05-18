"""F2.2 (2026-05-15) — AUROC CI empirical coverage validation.

Verifies that the 50/80/90/95% credible intervals produced by Mode-A
session reach their nominal empirical coverage on synthetic raters drawn
from the production prior.  This is the critical validation: if 95% CIs
only cover the truth 75% of the time, the δ=0.05 stopping rule's "5%
absolute AUROC error" claim collapses.

Protocol
--------
- K=6, draw N_RATERS synthetic (t, l) from N(0, Sigma=Corr_l) prior.
- Run Mode-A session to FIXED max_q (no early stopping; we want the
  posterior at a known budget, not where the stopping rule would fire).
- Compute weighted quantiles of AUROC posterior per domain at
  alphas = (0.025, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.975).
- For each (CI level, domain): empirical coverage = fraction of synthetic
  raters whose true AUROC lies inside the credible interval.
- PASS criterion: empirical ∈ [nominal − 0.03, nominal + 0.03] for 95%.

Outputs:
  results/phase2_validation/coverage_results.csv     (per-rater per-domain)
  results/phase2_validation/coverage_summary.json    (per-CI-level coverage)
  results/phase2_validation/coverage_summary.md      (markdown table)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# CRITICAL: configure single-thread BLAS BEFORE importing numpy.  _parallel
# applies the env caps on import.
_THIS_DIR_BOOT = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR_BOOT not in sys.path:
    sys.path.insert(0, _THIS_DIR_BOOT)
from _parallel import configure_blas_single_thread, parallel_map  # noqa: E402
configure_blas_single_thread()

import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

from core_mcmc import (
    load_fitted_Sigma, make_state_hier, sample_prior_hier_K,
    choose_item, update, ess, resample_and_rejuvenate, simulate_response,
)
from auroc import auroc_from_l, auroc_quantiles_from_particles_hier
from bridge._common import (
    _autodetect_banks_dir, load_bank_signals,
)


DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── lazy per-worker shared read-only data cache ───────────────────────
# Under the spawn start method each worker re-imports this module; the
# first task in each worker populates this cache from disk (≈12 KB total),
# then all subsequent tasks in that worker reuse it.
_SHARED: Dict[str, Any] = {}


def _get_shared() -> Dict[str, Any]:
    if not _SHARED:
        obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
        _SHARED["Corr_l"] = np.asarray(obj["Corr_l"], dtype=float)
        _SHARED["bank_signals"] = load_bank_signals(
            _autodetect_banks_dir(), DOMAINS
        )
    return _SHARED

# CI levels evaluated.  Tail alphas come in pairs so (1-2alpha) is the
# nominal CI level.
CI_LEVELS = {
    "50%": (0.25, 0.75),
    "80%": (0.10, 0.90),
    "90%": (0.05, 0.95),
    "95%": (0.025, 0.975),
}


def auroc_quantiles_at_levels(state, levels):
    """Return dict { level_str -> (lo_K, hi_K) } of AUROC posterior quantiles."""
    alphas = []
    for lvl, (a_lo, a_hi) in levels.items():
        alphas.extend([a_lo, a_hi])
    qs = auroc_quantiles_from_particles_hier(state, alphas=alphas)
    out = {}
    for i, lvl in enumerate(levels):
        out[lvl] = (qs[:, 2 * i], qs[:, 2 * i + 1])
    return out


def run_session_fixed_budget(*, true_params, K, max_q, N, seed,
                              Sigma_l, Sigma_t, bank_signals,
                              ess_threshold_frac=0.5, n_mh_steps=15):
    """Inline Mode-A session loop that runs exactly max_q questions and
    returns the final particle state (for quantile extraction).
    """
    rng = np.random.default_rng(seed)
    state = make_state_hier(N, K, r_assumed=0.378, rng=rng,
                            Sigma_l=Sigma_l, Sigma_t=Sigma_t)
    proposal_scale = 2.38 / np.sqrt(2 * K)
    for q in range(max_q):
        k, s = choose_item(state, bank_signals)
        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update(state, k, s, y)
        if ess(state["w"]) < ess_threshold_frac * N:
            resample_and_rejuvenate(state, rng, n_mh_steps, proposal_scale)
    return state


def _coverage_worker(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Top-level worker: one synthetic rater → list of per-domain row dicts.

    Self-contained and picklable.  Shared read-only data (Corr_l, bank
    signals) is loaded once per worker process via `_get_shared()`.
    """
    shared = _get_shared()
    Corr_l = shared["Corr_l"]
    bank_signals = shared["bank_signals"]

    ri = task["rater_idx"]
    K = task["K"]
    true_params = task["true_params"]
    t_true = np.asarray(task["t_true"], dtype=float)
    l_true = np.asarray(task["l_true"], dtype=float)
    true_auroc = auroc_from_l(l_true)

    state = run_session_fixed_budget(
        true_params=true_params, K=K, max_q=task["max_q"],
        N=task["N_particles"], seed=task["seed"],
        Sigma_l=Corr_l, Sigma_t=Corr_l, bank_signals=bank_signals,
    )
    ci_set = auroc_quantiles_at_levels(state, CI_LEVELS)

    rows: List[Dict[str, Any]] = []
    for k, d in enumerate(DOMAINS):
        row = {
            "rater_idx": ri,
            "domain": d,
            "true_auroc": float(true_auroc[k]),
            "true_l": float(l_true[k]),
            "true_t": float(t_true[k]),
        }
        for lvl, (lo, hi) in ci_set.items():
            lo_v = float(lo[k]); hi_v = float(hi[k])
            row[f"ci_lo_{lvl}"] = lo_v
            row[f"ci_hi_{lvl}"] = hi_v
            row[f"covered_{lvl}"] = int(lo_v <= true_auroc[k] <= hi_v)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-raters", type=int, default=200,
                   help="Number of synthetic raters (default 200).")
    p.add_argument("--max-q", type=int, default=200,
                   help="Fixed question budget per session (default 200).")
    p.add_argument("--N-particles", type=int, default=600,
                   help="Particles per session (default 600).")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Starting RNG seed (default 42).")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Parallel worker processes (default: 14 / cpu-2).")
    args = p.parse_args()

    K = 6
    obj = load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)

    print(f"\n=== F2.2 AUROC CI coverage validation ===", flush=True)
    print(f"  N_raters={args.n_raters} max_q={args.max_q} "
          f"N_particles={args.N_particles}", flush=True)
    print(f"  Prior: Corr_l (unit-diagonal, K=6); l_prior_mean=0", flush=True)

    # Pre-draw all true (t, l) from the prior (single RNG, deterministic).
    truth_rng = np.random.default_rng(args.seed_base + 100000)
    t_true_all, l_true_all = sample_prior_hier_K(
        N=args.n_raters, K=K, r=0.378, rng=truth_rng,
        Sigma_l=Corr_l, Sigma_t=Corr_l,
    )
    print(f"  drew {args.n_raters} synthetic raters from prior; "
          f"l_true range = [{l_true_all.min():.2f}, {l_true_all.max():.2f}]",
          flush=True)

    # Build self-contained task descriptors (each carries its own seed).
    tasks: List[Dict[str, Any]] = []
    for ri in range(args.n_raters):
        t_true = t_true_all[ri]
        l_true = l_true_all[ri]
        true_params = []
        for k in range(K):
            true_params.append(float(t_true[k]))
            true_params.append(float(l_true[k]))
        tasks.append({
            "rater_idx": ri,
            "K": K,
            "true_params": true_params,
            "t_true": t_true.tolist(),
            "l_true": l_true.tolist(),
            "max_q": args.max_q,
            "N_particles": args.N_particles,
            "seed": args.seed_base + ri,
        })

    t0 = time.time()
    per_rater_rows = parallel_map(
        _coverage_worker, tasks,
        max_workers=args.max_workers,
        desc="coverage raters",
        ordered=True,
        progress_every=10,
    )
    # Flatten; surface any worker errors loudly.
    rows: List[Dict[str, Any]] = []
    n_err = 0
    for r in per_rater_rows:
        if isinstance(r, dict) and "__error__" in r:
            n_err += 1
            print(f"  [ERROR] task {r.get('__task_index__')}: "
                  f"{r['__error__']}", file=sys.stderr, flush=True)
            continue
        rows.extend(r)
    if n_err:
        print(f"  [WARN] {n_err}/{len(tasks)} raters errored; "
              f"coverage computed on the rest.", flush=True)

    # Compute empirical coverage per CI level
    print("\n=== Coverage summary ===", flush=True)
    summary = {}
    for lvl in CI_LEVELS:
        nominal = float(lvl.rstrip("%")) / 100.0
        per_domain = {}
        for d in DOMAINS:
            covers = [r[f"covered_{lvl}"] for r in rows if r["domain"] == d]
            cov = float(np.mean(covers)) if covers else float("nan")
            per_domain[d] = cov
        overall = float(np.mean([r[f"covered_{lvl}"] for r in rows]))
        summary[lvl] = {
            "nominal": nominal,
            "overall_empirical": overall,
            "per_domain": per_domain,
            "deviation_from_nominal": overall - nominal,
            "pass": abs(overall - nominal) <= 0.03,
        }
        print(f"  CI {lvl}: empirical = {overall:.3f} "
              f"(nominal {nominal:.3f}, Δ = {overall - nominal:+.3f}) "
              f"PASS={summary[lvl]['pass']}", flush=True)

    # Write outputs
    csv_path = os.path.join(OUT_DIR, "coverage_results.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {csv_path}", flush=True)

    json_path = os.path.join(OUT_DIR, "coverage_summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "n_raters": args.n_raters,
                "max_q": args.max_q,
                "N_particles": args.N_particles,
                "K": K,
                "domains": DOMAINS,
                "prior": "Corr_l unit-diagonal, zero mean",
            },
            "summary": summary,
            "total_seconds": time.time() - t0,
        }, f, indent=2)
    print(f"  wrote {json_path}", flush=True)

    md_path = os.path.join(OUT_DIR, "coverage_summary.md")
    with open(md_path, "w") as f:
        f.write("# AUROC CI Coverage Validation (F2.2)\n\n")
        f.write(f"Synthetic raters: N = {args.n_raters}, max_q = {args.max_q}, "
                f"N_particles = {args.N_particles}.  "
                f"Prior: Corr_l (unit-diagonal, K=6).\n\n")
        f.write("## Summary (overall, all 6 domains pooled)\n\n")
        f.write("| CI level | Nominal | Empirical | Δ | Status |\n")
        f.write("|---|---|---|---|---|\n")
        for lvl, s in summary.items():
            status = "✅ PASS" if s["pass"] else "⚠️ FAIL"
            f.write(f"| {lvl} | {s['nominal']:.3f} | "
                    f"{s['overall_empirical']:.3f} | "
                    f"{s['deviation_from_nominal']:+.3f} | {status} |\n")
        f.write("\n## Per-domain coverage (95% CI)\n\n")
        f.write("| Domain | Empirical 95% coverage |\n")
        f.write("|---|---|\n")
        for d, cov in summary["95%"]["per_domain"].items():
            f.write(f"| {d} | {cov:.3f} |\n")
        f.write(f"\nTotal compute time: {(time.time() - t0)/60:.1f} min.\n")
    print(f"  wrote {md_path}", flush=True)
    print(f"\nDone in {(time.time() - t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
