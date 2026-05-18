"""Multi-AUROC Precision Protocol Bridge (Mode-A) — Paper 1 production driver.

F1.3 (2026-05-15).  Replaces the Mode-B binary-cert bridge as the production
path for Paper 1 (Epilepsia / Clin Neurophysiol) submission.  Reframe rationale
in `docs/historical/state_logs/SYNTHESIS_2026_05_15.md`.

What this bridge does
---------------------
1. Loads `cert_config.yaml` (v11, mode='mode_a').
2. Loads `Sigma_l_fitted.npy` (Corr_l → hierarchical prior, unit-diagonal,
   no boundary scaling — Mode-A keeps the prior centered at 0).
3. Loads per-domain bank signals (s_probit) and the rater matrix.
4. For each (rater, method, seed) triple:
     - runs `run_session_mcmc_auroc` with `delta_auroc=auroc_delta_paper`
       (the tightest target, e.g. 0.025) and `log_trajectory=True`,
       `run_until_max=True`
     - uses `post_hoc_delta_sweep` to report n_q at all δ ∈ delta_sweep
       (e.g. {0.025, 0.05, 0.10}) from the SAME run
     - records per-domain final AUROC posterior summary (mean, 95% CI)
5. Writes per-session JSONL audit log via `bridge.audit_trail`.
6. Writes summary CSV: rater_id, method, seed, n_q_d0.025, n_q_d0.05,
   n_q_d0.10, per-domain auroc_mean/auroc_ci_low/auroc_ci_high, true_auroc_k.

Usage
-----
    # pilot (1 seed per rater, per method)
    python -m bridge.run_multi_auroc_bridge

    # paper-grade (40 seeds, hierarchical method only)
    python -m bridge.run_multi_auroc_bridge --paper-grade --method hier

    # custom rater matrix + tag
    python -m bridge.run_multi_auroc_bridge \\
        --rater-matrix /tmp/my_raters.csv --tag pilot_mode_a
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_BRIDGE_DIR)   # repo root: cert_config/Sigma
# Phase 2 (unified merge): the validated engine modules live in engine/
# with flat bare-name imports preserved. Put engine/ on sys.path; keep
# ENGINE_REPO = repo root so DEFAULT_CONFIG_PATH / sigma_l_path resolve
# unchanged (path-only; no behavior change).
for _p in (os.path.join(ENGINE_REPO, "engine"), ENGINE_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_mcmc import (                              # noqa: E402
    run_session_mcmc_auroc,
    post_hoc_delta_sweep,
    load_fitted_Sigma,
)
from auroc import auroc_from_l                      # noqa: E402

# Reuse the mode-agnostic helpers from the Mode-B bridge.
try:
    from bridge._common import (    # noqa: E402
        load_config,
        _require,
        _autodetect_banks_dir,
        _default_rater_matrix_path,
        load_bank_signals,
        load_raters,
        build_true_params,
    )
    from bridge.audit_trail import AuditTrail, session_audit_path
except ImportError:
    sys.path.insert(0, _BRIDGE_DIR)
    from _common import (           # type: ignore
        load_config,
        _require,
        _autodetect_banks_dir,
        _default_rater_matrix_path,
        load_bank_signals,
        load_raters,
        build_true_params,
    )
    from audit_trail import AuditTrail, session_audit_path  # type: ignore


DEFAULT_CONFIG_PATH = os.path.join(ENGINE_REPO, "cert_config.yaml")


# ── Mode-A prior covariance loader ────────────────────────────────────
def resolve_sigma_mode_a(cfg: Dict[str, Any]) -> Tuple[Optional[np.ndarray],
                                                       Optional[np.ndarray],
                                                       float,
                                                       str]:
    """Resolve (Sigma_l, Sigma_t, r_assumed_fallback, note) for Mode-A.

    Mode-A uses Corr_l with NO boundary-prior scaling (unlike Mode-B's
    `prior_sigma_l²·Corr_l`).  The prior is centered at zero (no boundary
    shift) and has unit marginal variance — this gives the engine a
    standardised, well-conditioned prior for the per-examinee Multi-AUROC
    measurement task.

    Returns Corr_l for both l-block and t-block when available; falls back to
    `r_assumed_fallback` compound-symmetry if `sigma_l_path` is missing.
    """
    sigma_l_path = cfg.get("sigma_l_path")
    r_fallback = float(cfg.get("r_assumed_fallback", 0.3782))
    use_corr_l = bool(cfg.get("use_corr_l", True))

    Sigma_l = None
    Sigma_t = None
    note = ""

    abs_l = (os.path.join(ENGINE_REPO, sigma_l_path)
             if sigma_l_path and not os.path.isabs(sigma_l_path)
             else sigma_l_path)
    if abs_l and os.path.exists(abs_l):
        obj_l = load_fitted_Sigma(abs_l)
        if use_corr_l:
            if "Corr_l" not in obj_l:
                raise KeyError(
                    f"use_corr_l=True but 'Corr_l' key not found in {abs_l}. "
                    "Re-run the Sigma estimation script to regenerate the file."
                )
            Sigma_l = np.asarray(obj_l["Corr_l"], dtype=float)
            note = "Corr_l (unit-diagonal); Mode-A no boundary scaling"
        else:
            Sigma_l = np.asarray(obj_l["Sigma_l"], dtype=float)
            note = "Sigma_l raw covariance (use_corr_l=False)"
        # t-block: same matrix (no separate t-data available)
        Sigma_t = Sigma_l.copy()
    else:
        note = (f"sigma_l_path={sigma_l_path!r} not found; "
                f"falling back to compound-symmetry r={r_fallback}.")
    return Sigma_l, Sigma_t, r_fallback, note


# ── single-session driver (Mode-A) ────────────────────────────────────
def run_one_session_mode_a(
    *,
    method: str,
    true_params: List[float],
    bank_signals: List[np.ndarray],
    seed: int,
    K: int,
    Sigma_l: Optional[np.ndarray],
    Sigma_t: Optional[np.ndarray],
    r_fallback: float,
    auroc_delta: float,
    alpha: float,
    n_particles: int,
    max_q: int,
    ess_threshold_frac: float,
    n_mh_steps: int,
    proposal_scale_hier: Optional[float],
    proposal_scale_brute: float,
    delta_sweep: List[float],
    log_trajectory: bool = True,
) -> Dict[str, Any]:
    """Run one Mode-A session and compute the post-hoc δ-sweep summary."""
    if method == "hier":
        ps = (proposal_scale_hier
              if proposal_scale_hier is not None
              else 2.38 / np.sqrt(2 * K))
    else:
        ps = float(proposal_scale_brute)

    out = run_session_mcmc_auroc(
        method=method,
        true_params=true_params,
        K=K,
        r_assumed=r_fallback,
        max_q=max_q,
        delta_auroc=auroc_delta,
        N=n_particles,
        seed=seed,
        run_until_max=True,                # run to max so post-hoc δ-sweep works
        log_trajectory=log_trajectory,
        alpha=alpha,
        n_mh_steps=n_mh_steps,
        proposal_scale=ps,
        ess_threshold_frac=ess_threshold_frac,
        bank_signals=bank_signals,
        Sigma_l=Sigma_l,
        Sigma_t=Sigma_t,
    )

    # Post-hoc δ-sweep — uses lo_traj/hi_traj from log_trajectory=True.
    if "lo_traj" in out and "hi_traj" in out:
        out["delta_sweep_stops"] = post_hoc_delta_sweep(
            out["lo_traj"], out["hi_traj"], deltas=delta_sweep,
        )
    else:
        out["delta_sweep_stops"] = {float(d): None for d in delta_sweep}

    return out


# ── per-domain AUROC posterior summary ────────────────────────────────
def auroc_posterior_summary(out: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract per-domain AUROC posterior mean + 95% CI from the final state.

    The session driver's `final_lo` / `final_hi` are the 95% CIs.  The mean
    is approximated as the midpoint of the CI (good when the AUROC posterior
    is roughly symmetric near the mode; for full sampling-based mean, run
    the trajectory with finer particle resolution).
    """
    lo = np.asarray(out["final_lo"])
    hi = np.asarray(out["final_hi"])
    mean = (lo + hi) / 2.0
    hw = (hi - lo) / 2.0
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "hw": hw,
            "hw_max": float(hw.max())}


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-AUROC Precision Protocol bridge (Paper 1 Mode-A).",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="Path to cert_config.yaml (default: %(default)s).")
    p.add_argument("--rater-matrix", default=None,
                   help="Path to cross_domain_rater_matrix.csv "
                        "(default: data/engine_inputs/).")
    p.add_argument("--banks-dir", default=None,
                   help="Curated banks directory (default: auto-detect).")
    p.add_argument("--audit-log-dir", default=None,
                   help="Audit-log output dir (default: audit_log_dir from config).")
    p.add_argument("--results-dir", default=None,
                   help="Where to write per-run CSVs "
                        "(default: <ENGINE_REPO>/results/mode_a_auroc/).")
    p.add_argument("--n-reps", type=int, default=None,
                   help="Number of seeds per (rater, method).  "
                        "Default = n_reps_pilot from config.")
    p.add_argument("--seed-base", type=int, default=0,
                   help="Starting seed (effective seeds = seed_base..seed_base+n_reps-1).")
    p.add_argument("--method", choices=["hier", "brute", "both"], default="both",
                   help="Which method(s) to run (default: both).")
    p.add_argument("--paper-grade", action="store_true",
                   help="Shortcut: set --n-reps to n_reps_paper from config.")
    p.add_argument("--no-audit", action="store_true",
                   help="Disable the JSONL audit trail (debugging only).")
    p.add_argument("--tag", default=None,
                   help="Free-form tag stamped into audit records under extra.tag.")
    p.add_argument("--max-raters", type=int, default=None,
                   help="Cap on number of raters (debug; default: all).")
    return p.parse_args(argv)


# ── main ──────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    cfg = load_config(args.config)
    mode = cfg.get("mode", "mode_b")
    if mode != "mode_a":
        print(f"[WARN] cert_config.yaml mode={mode!r}, expected 'mode_a'. "
              "Mode-A bridge will run anyway but uses Mode-A semantics "
              "regardless of config 'mode' key.", file=sys.stderr)

    domains: List[str] = list(_require(cfg, "domains"))
    K: int = int(_require(cfg, "K"))
    if len(domains) != K:
        raise ValueError(
            f"cert_config.yaml inconsistency: K={K} but len(domains)={len(domains)}"
        )
    auroc_delta = float(_require(cfg, "auroc_delta"))
    delta_sweep = [float(d) for d in _require(cfg, "delta_sweep")]
    alpha = float(_require(cfg, "alpha"))
    n_particles = int(_require(cfg, "n_particles"))
    max_q = int(_require(cfg, "max_q"))
    ess_threshold_frac = float(_require(cfg, "ess_threshold_frac"))
    n_mh_steps = int(_require(cfg, "n_mh_steps"))
    proposal_scale_hier = cfg.get("proposal_scale_hier")
    if proposal_scale_hier is not None:
        proposal_scale_hier = float(proposal_scale_hier)
    proposal_scale_brute = float(_require(cfg, "proposal_scale_brute"))
    lapse_rate = float(_require(cfg, "lapse_rate"))
    n_reps_pilot = int(_require(cfg, "n_reps_pilot"))
    n_reps_paper = int(_require(cfg, "n_reps_paper"))
    log_trajectory = bool(cfg.get("log_trajectory", True))

    # Resolve Sigma
    Sigma_l, Sigma_t, r_fallback, sigma_note = resolve_sigma_mode_a(cfg)
    print(f"Mode-A prior: {sigma_note}")
    if Sigma_l is not None:
        print(f"  Sigma_l shape={Sigma_l.shape}, "
              f"diag mean={np.diag(Sigma_l).mean():.3f}")

    # Resolve data paths
    banks_dir = args.banks_dir or _autodetect_banks_dir()
    rater_matrix_path = args.rater_matrix or _default_rater_matrix_path()

    bank_signals = load_bank_signals(banks_dir, domains)
    raters_df = load_raters(rater_matrix_path)
    if args.max_raters is not None:
        raters_df = raters_df.head(args.max_raters)
    n_raters = len(raters_df)
    print(f"Rater matrix: {n_raters} raters loaded from {rater_matrix_path}")

    # Resolve audit dir + results dir
    audit_log_dir = args.audit_log_dir or cfg.get("audit_log_dir", "audit_logs/")
    if not os.path.isabs(audit_log_dir):
        audit_log_dir = os.path.join(ENGINE_REPO, audit_log_dir)
    results_dir = args.results_dir or os.path.join(ENGINE_REPO,
                                                    "results", "mode_a_auroc")
    os.makedirs(audit_log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Resolve replication budget
    if args.paper_grade:
        n_reps = n_reps_paper
    elif args.n_reps is not None:
        n_reps = int(args.n_reps)
    else:
        n_reps = n_reps_pilot
    seed_base = int(args.seed_base)
    methods = (["hier", "brute"] if args.method == "both" else [args.method])
    print(f"Plan: {n_raters} raters × {len(methods)} method(s) × {n_reps} seed(s) "
          f"= {n_raters * len(methods) * n_reps} sessions.")

    # Iterate sessions
    rows: List[Dict[str, Any]] = []
    n_sessions = 0
    n_target = n_raters * len(methods) * n_reps
    for _, row in raters_df.iterrows():
        rater_id = str(row.get("rater_id", row.get("canonical_name", "unknown")))
        try:
            true_params = build_true_params(row, domains)
        except KeyError as e:
            print(f"[SKIP] rater {rater_id}: missing field {e}", file=sys.stderr)
            continue
        true_l = np.array([true_params[2 * k + 1] for k in range(K)])
        true_auroc = auroc_from_l(true_l)

        for method in methods:
            for rep in range(n_reps):
                seed = seed_base + rep
                n_sessions += 1
                t_sess = time.time()
                out = run_one_session_mode_a(
                    method=method,
                    true_params=true_params,
                    bank_signals=bank_signals,
                    seed=seed,
                    K=K,
                    Sigma_l=Sigma_l,
                    Sigma_t=Sigma_t,
                    r_fallback=r_fallback,
                    auroc_delta=auroc_delta,
                    alpha=alpha,
                    n_particles=n_particles,
                    max_q=max_q,
                    ess_threshold_frac=ess_threshold_frac,
                    n_mh_steps=n_mh_steps,
                    proposal_scale_hier=proposal_scale_hier,
                    proposal_scale_brute=proposal_scale_brute,
                    delta_sweep=delta_sweep,
                    log_trajectory=log_trajectory,
                )
                summary = auroc_posterior_summary(out)
                dt = time.time() - t_sess

                row_out: Dict[str, Any] = {
                    "rater_id": rater_id,
                    "method": method,
                    "seed": seed,
                    "auroc_delta": auroc_delta,
                    "n_q_total": int(out["n_questions"]),
                    "stopped_early": bool(out["stopped_early"]),
                    "hw_max_final": summary["hw_max"],
                    "mean_acceptance_rate": float(out["mean_acceptance_rate"]),
                    "n_rejuvenations": int(out["n_rejuvenations"]),
                    "duration_s": round(dt, 3),
                }
                # Post-hoc δ-sweep stops
                for d, stop in out["delta_sweep_stops"].items():
                    row_out[f"n_q_at_d{d:.3f}"] = stop
                # Per-domain summaries
                for k, dname in enumerate(domains):
                    row_out[f"true_auroc_{dname}"] = float(true_auroc[k])
                    row_out[f"auroc_mean_{dname}"] = float(summary["mean"][k])
                    row_out[f"auroc_ci_low_{dname}"] = float(summary["ci_low"][k])
                    row_out[f"auroc_ci_high_{dname}"] = float(summary["ci_high"][k])
                rows.append(row_out)

                # Audit trail
                if not args.no_audit:
                    audit_path = session_audit_path(audit_log_dir, rater_id, seed)
                    with AuditTrail(audit_path) as audit:
                        audit.record_session_mode_a(
                            rater_id=rater_id,
                            method=method,
                            seed=seed,
                            domains=domains,
                            true_params=true_params,
                            lapse_rate=lapse_rate,
                            session_result=out,
                            delta_sweep=delta_sweep,
                            sigma_note=sigma_note,
                            extra={"tag": args.tag} if args.tag else None,
                        )

                print(f"  [{n_sessions}/{n_target}] {rater_id} {method} "
                      f"seed={seed}: n_q={out['n_questions']} "
                      f"hw_max={summary['hw_max']:.4f} "
                      f"sweep={out['delta_sweep_stops']} ({dt:.1f}s)")

    # Write summary CSV
    summary_df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "mode_a_auroc_results.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(rows)} rows to {out_csv}")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
