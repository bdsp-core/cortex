"""Mode-B Binary Certification Bridge — Wave-1-corrected version.

Last revised 2026-05-14: integrated Wave-1 reviewer corrections (T0.5, T0.7,
T1.1, T1.3, T1.4, T1.6).  Loads parameters from `cert_config.yaml` (single
source of truth for all certified Mode-B constants — see W1-C).

Local copy of the historical bridge that lived at
`ilae-skill-certification-test-main/src/run_mode_b_cert_bridge.py`.  Reviewer
R5 flagged the cross-repo coupling (sibling `sys.path.insert` on the
multi-main repo from inside the test-main repo) as bad for reproducibility,
so the bridge now lives inside the engine repo and discovers the engine via
relative imports from the parent directory.

Wave-1 changes vs. the legacy bridge
------------------------------------
1. **Config-driven.** All constants (`K`, `domains`, `stop_thresh`, `alpha`,
   `lapse_rate`, `n_particles`, `max_q`, `n_reps_pilot`, `n_reps_paper`, ...)
   are read from `cert_config.yaml`.  No hardcoded scientific values.
2. **Unstructured prior covariance** (T1.6).  Loads `Sigma_l_fitted.npy` and
   passes `Sigma_l` to `run_session_mcmc_certification` instead of the legacy
   compound-symmetry `r_assumed`.  Falls back to `r_assumed_fallback` from
   config only if the fitted `.npy` file is missing.
3. **Virtual cert removed** (T1.1).  The legacy `VIRTUAL_PAIRS = [(SZ_IDX,
   GRDA_IDX, ...)]` block and the `SZ_GRDA_THRESH` constant are deleted.
   The joint particle posterior subsumes virtual certification via the prior
   covariance; calling the engine with `virtual_pairs=...` would only emit a
   `DeprecationWarning`.
4. **KL item selection removed** (T0.5).  The legacy `DELTA_INDIFF = 0.15`
   constant is deleted (no consumer remains; EV-optimal selection is the
   only certified criterion).
5. **Lapse rate.**  `LAPSE_RATE = 0.025` is documented in the config and
   consumed inside `core_mcmc.py` (it is a module constant on the engine
   side; the bridge logs the value in the audit trail for traceability but
   does not pass it as a kwarg).
6. **MCSE-buffered stopping** (T1.3).  Already implemented inside
   `core_mcmc.py` (with `Z_BUFFER = 2.0`).  No bridge-side buffering logic;
   the legacy bridge had none either, so nothing to remove.
7. **CLI args** for pilot vs paper-grade runs:
       --n-reps INT             number of seeds per (rater, method).  Default
                                is `n_reps_pilot` from config (=1).  For
                                paper-grade, pass `--n-reps 25` or use
                                `--paper-grade` (which sets it to
                                `n_reps_paper`).
       --seed-base INT          starting seed; effective seeds are
                                seed_base + 0..(n_reps - 1).  Default 0.
       --method {hier,brute,both}
                                which method(s) to run.  Default both.
       --paper-grade            shortcut for --n-reps n_reps_paper.
       --rater-matrix PATH      override the path to the rater matrix CSV
                                (default: in-repo
                                `data/engine_inputs/cross_domain_rater_matrix.csv`).
       --banks-dir PATH         override the curated bank directory.  Two
                                options are auto-detected if not given:
                                (a) sibling-repo
                                `ilae-skill-certification-test-main/data/curated_banks/`
                                or (b) this-repo
                                `ilae-skill-certification-test-multi-main/data/curated_banks/`
                                (whichever exists first wins).
       --config PATH            override `cert_config.yaml` path.
       --audit-log-dir PATH     override the audit log dir (default
                                `audit_logs/` from config).
       --results-dir PATH       where to write the per-run results CSVs.
                                Default: a new `results/mode_b_cert/` dir
                                under this engine repo.
       --no-audit               disable the audit trail (debugging only).
       --tag STR                free-form tag stamped into each audit
                                record under `extra.tag` for batch
                                bookkeeping.

Audit trail (W2-C, R5 §D2)
--------------------------
Each `(rater, method, seed)` triple writes one JSONL line to
`<audit_log_dir>/<rater_id>_seed<seed>.jsonl`.  See `audit_trail.py` for
the schema.  Option A (post-hoc adapter) is used: `core_mcmc.py` is NOT
modified; the bridge calls `run_session_mcmc_certification` and then
records the return dict.

Usage
-----
    # pilot (1 seed per rater, per method)
    python -m bridge.run_mode_b_cert_bridge

    # paper-grade (25 seeds, hierarchical method only)
    python -m bridge.run_mode_b_cert_bridge --paper-grade --method hier

    # custom rater matrix
    python -m bridge.run_mode_b_cert_bridge \
        --rater-matrix /tmp/my_raters.csv --tag pilot_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────
# This file lives at <ENGINE_REPO>/bridge/run_mode_b_cert_bridge.py.  The
# core engine modules live in the parent directory.  We add that to sys.path
# so `from core_mcmc import ...` works whether the bridge is invoked as
# `python -m bridge.run_mode_b_cert_bridge` or as a plain script.
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_BRIDGE_DIR)   # repo root: cert_config/Sigma
# Phase 2 (unified merge): engine modules live in engine/ (flat bare-name
# imports preserved). engine/ on sys.path; ENGINE_REPO stays repo root.
for _p in (os.path.join(ENGINE_REPO, "engine"), ENGINE_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_mcmc import load_fitted_Sigma  # noqa: E402
from engine_mode_b import run_session_mcmc_certification  # noqa: E402  (F3.1: Mode-B relocated)
from auroc import auroc_from_l                                            # noqa: E402

# Bridge-package-local imports (work both as -m bridge.X and as plain script).
# F3.2: mode-agnostic helpers now live in bridge/_common.py.
try:
    from bridge.audit_trail import AuditTrail, session_audit_path
    from bridge._common import (
        DEFAULT_CONFIG_PATH, load_config, _require,
        _autodetect_banks_dir, _default_rater_matrix_path,
        load_bank_signals, load_raters, build_true_params,
    )
except ImportError:
    sys.path.insert(0, _BRIDGE_DIR)
    from audit_trail import AuditTrail, session_audit_path  # type: ignore
    from _common import (  # type: ignore
        DEFAULT_CONFIG_PATH, load_config, _require,
        _autodetect_banks_dir, _default_rater_matrix_path,
        load_bank_signals, load_raters, build_true_params,
    )


# ── prior covariance loader ───────────────────────────────────────────
def resolve_sigma(cfg: Dict[str, Any]) -> Tuple[Optional[np.ndarray],
                                                Optional[np.ndarray],
                                                float,
                                                str]:
    """Resolve (Sigma_l, Sigma_t, r_assumed_for_logging, note).

    FIX-T1.7: when cfg['use_corr_l'] is true (the default-safe setting),
    reads the correlation matrix ('Corr_l' key) instead of the raw covariance
    ('Sigma_l' key).  The fitted Sigma_l has marginal std ~0.10-0.24, which
    reflects inter-rater skill SPREAD, not the scale of plausible examinee
    skills.  Using it as a zero-mean prior places the certification threshold
    l*_k at 2-4 prior standard deviations above zero, making P(l_k > l*_k |
    prior) < 1% for all domains — causing the MCSE-buffered FAIL condition to
    fire immediately after the first question regardless of the examinee's true
    skill.  The correlation matrix (unit marginal variance) preserves the
    empirical cross-domain correlation structure while restoring the prior
    marginal variance to 1.0 (matching the pre-T1.6 compound-symmetry
    diagonal).  Corr_l is already stored in Sigma_l_fitted.npy.

    Returns Sigma matrices loaded from cfg['sigma_l_path'] / cfg['sigma_t_path']
    if available; falls back to the compound-symmetry `r_assumed_fallback`
    only if BOTH are missing.  In that fallback case Sigma_l/Sigma_t are
    None and the engine uses its legacy CS path.
    """
    sigma_l_path = cfg.get("sigma_l_path")
    sigma_t_path = cfg.get("sigma_t_path")
    r_fallback = float(cfg.get("r_assumed_fallback", 0.3782))
    # FIX-T1.7: use_corr_l defaults to True to prevent prior-scale misspecification.
    use_corr_l = bool(cfg.get("use_corr_l", True))

    Sigma_l = None
    Sigma_t = None
    note = ""

    abs_l = (os.path.join(ENGINE_REPO, sigma_l_path)
             if sigma_l_path and not os.path.isabs(sigma_l_path)
             else sigma_l_path)
    abs_t = (os.path.join(ENGINE_REPO, sigma_t_path)
             if sigma_t_path and not os.path.isabs(sigma_t_path)
             else sigma_t_path)

    if abs_l and os.path.exists(abs_l):
        obj_l = load_fitted_Sigma(abs_l)
        if use_corr_l:
            # FIX-T1.7: prefer Corr_l (unit diagonal) over raw Sigma_l.
            if "Corr_l" not in obj_l:
                raise KeyError(
                    f"use_corr_l=True but 'Corr_l' key not found in {abs_l}. "
                    "Re-run the Sigma estimation script to regenerate the file, "
                    "or set use_corr_l: false in cert_config.yaml to use the "
                    "raw covariance (WARNING: this will cause false FAILs)."
                )
            Sigma_l = np.asarray(obj_l["Corr_l"], dtype=float)
            corr_note = "Corr_l (unit-diagonal, FIX-T1.7)"
        else:
            Sigma_l = np.asarray(obj_l["Sigma_l"], dtype=float)
            corr_note = "Sigma_l (raw covariance — WARNING: prior-scale bug)"

        # FIX-T1.8: scale l-block covariance by prior_sigma_l^2 so the boundary
        # prior N(l*, sigma^2 * Corr_l) has the correct marginal std.
        # The t-block is NOT scaled (bias parameter keeps unit marginal variance).
        prior_sigma_l = float(cfg.get("prior_sigma_l", 1.0))
        if prior_sigma_l != 1.0:
            Sigma_l = Sigma_l * (prior_sigma_l ** 2)
            corr_note += f" × sigma²={prior_sigma_l**2:.3f} (FIX-T1.8)"

        # t-block: always use unscaled Corr_l (or Sigma_t from file if present)
        base_corr_unscaled = np.asarray(
            obj_l.get("Corr_l", obj_l.get("Sigma_l")), dtype=float)
        if abs_t and os.path.exists(abs_t):
            obj_t = load_fitted_Sigma(abs_t)
            if use_corr_l:
                Sigma_t = np.asarray(
                    obj_t.get("Corr_t", obj_l.get("Corr_t", base_corr_unscaled)),
                    dtype=float,
                )
            else:
                Sigma_t = np.asarray(
                    obj_t.get("Sigma_t", obj_l.get("Sigma_t", base_corr_unscaled)),
                    dtype=float,
                )
        else:
            Sigma_t = base_corr_unscaled.copy()
            note = "sigma_t_path missing; Sigma_t = unscaled Corr_l per W1-A guidance."
        if not note:
            note = corr_note
    else:
        note = (f"sigma_l_path={sigma_l_path!r} not found; "
                f"falling back to compound-symmetry r={r_fallback}.")

    return Sigma_l, Sigma_t, r_fallback, note


# ── single-session driver ─────────────────────────────────────────────
def run_one_cert(
    *,
    method: str,
    true_params: List[float],
    bank_signals: List[np.ndarray],
    seed: int,
    K: int,
    l_star: np.ndarray,
    Sigma_l: Optional[np.ndarray],
    Sigma_t: Optional[np.ndarray],
    r_fallback: float,
    stop_thresh: float,
    alpha: float,
    n_particles: int,
    max_q: int,
    ess_threshold_frac: float,
    n_mh_steps: int,
    proposal_scale_hier: Optional[float],
    proposal_scale_brute: float,
    n_min_guard: int = 20,          # FIX-T1.7
    l_prior_mean: Optional[np.ndarray] = None,   # FIX-T1.8
    use_iut_stopping: bool = True,  # FIX-T1.9
    fisher_imin: float = 0.0,       # FIX-T1.11
) -> Dict[str, Any]:
    """Wrap `run_session_mcmc_certification` with config-driven kwargs.

    Note: the engine consumes LAPSE_RATE and Z_BUFFER as module constants
    inside core_mcmc.py / core_mcmc_brute_k.py — they are NOT parameters
    of this call.  cert_config.yaml documents them and the audit trail
    records them, but the engine reads them from its own module scope.
    """
    if method == "hier":
        ps = (proposal_scale_hier
              if proposal_scale_hier is not None
              else 2.38 / np.sqrt(2 * K))
    else:
        ps = float(proposal_scale_brute)

    return run_session_mcmc_certification(
        method                = method,
        true_params           = true_params,
        K                     = K,
        r_assumed             = r_fallback,        # only used when Sigma_l is None
        l_star                = l_star,
        max_q                 = max_q,
        stop_thresh           = stop_thresh,
        alpha                 = alpha,
        N                     = n_particles,
        seed                  = seed,
        bank_signals          = bank_signals,
        Sigma_l               = Sigma_l,           # FIX-T1.6 (W1-A)
        Sigma_t               = Sigma_t,           # FIX-T1.6 (W1-A)
        proposal_scale        = ps,
        ess_threshold_frac    = ess_threshold_frac,
        n_mh_steps            = n_mh_steps,
        n_min_guard           = n_min_guard,       # FIX-T1.7
        l_prior_mean          = l_prior_mean,      # FIX-T1.8
        use_iut_stopping      = use_iut_stopping,  # FIX-T1.9
        fisher_imin           = fisher_imin,       # FIX-T1.11
        # FIX-T1.1: virtual_pairs is intentionally not passed (default None).
        # FIX-T0.5: KL is gone; delta_indiff defaults to its (unused) value.
    )


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mode-B binary certification bridge (Wave-1 corrected).",
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
                        "(default: <ENGINE_REPO>/results/mode_b_cert/).")
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
    return p.parse_args(argv)


# ── main ──────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    cfg = load_config(args.config)
    domains: List[str] = list(_require(cfg, "domains"))
    K: int = int(_require(cfg, "K"))
    if len(domains) != K:
        raise ValueError(
            f"cert_config.yaml inconsistency: K={K} but len(domains)={len(domains)}"
        )
    stop_thresh = float(_require(cfg, "stop_thresh"))
    alpha = float(_require(cfg, "alpha"))
    n_particles = int(_require(cfg, "n_particles"))
    max_q = int(_require(cfg, "max_q"))
    ess_threshold_frac = float(_require(cfg, "ess_threshold_frac"))
    n_mh_steps = int(_require(cfg, "n_mh_steps"))
    proposal_scale_hier = cfg.get("proposal_scale_hier")  # may be null -> dynamic
    if proposal_scale_hier is not None:
        proposal_scale_hier = float(proposal_scale_hier)
    proposal_scale_brute = float(_require(cfg, "proposal_scale_brute"))
    lapse_rate = float(_require(cfg, "lapse_rate"))      # for audit log only
    n_reps_pilot = int(_require(cfg, "n_reps_pilot"))
    n_reps_paper = int(_require(cfg, "n_reps_paper"))
    # FIX-T1.7: n_min_guard prevents FAIL from firing before enough questions
    n_min_guard: int = int(cfg.get("n_min_guard", 20))

    # FIX-T1.9: IUT stopping (Berger 1982)
    use_iut_stopping: bool = bool(cfg.get("use_iut_stopping", True))

    # FIX-T1.11: Fisher info guard
    fisher_imin: float = 0.0
    if cfg.get("use_fisher_guard", False):
        eps = float(cfg.get("fisher_guard_epsilon", 0.10))
        sigma_l = float(cfg.get("prior_sigma_l", 1.0))
        fisher_imin = max(0.0, 1.0 / eps ** 2 - 1.0 / sigma_l ** 2)

    # l_star: per-domain or global
    l_star_per_domain = cfg.get("l_star_per_domain")
    if l_star_per_domain:
        l_star = np.asarray(
            [float(l_star_per_domain[d]) for d in domains], dtype=float
        )
    else:
        l_star_global = float(_require(cfg, "l_star_global"))
        l_star = np.full(K, l_star_global, dtype=float)

    # n_reps resolution
    if args.paper_grade and args.n_reps is not None:
        # paper_grade is a shortcut; explicit --n-reps wins if both supplied.
        n_reps = int(args.n_reps)
    elif args.paper_grade:
        n_reps = n_reps_paper
    elif args.n_reps is not None:
        n_reps = int(args.n_reps)
    else:
        n_reps = n_reps_pilot

    # methods
    if args.method == "both":
        methods = ["hier", "brute"]
    else:
        methods = [args.method]

    # paths
    rater_matrix_path = args.rater_matrix or _default_rater_matrix_path()
    banks_dir = args.banks_dir or _autodetect_banks_dir()

    audit_log_dir = (args.audit_log_dir
                     or os.path.join(ENGINE_REPO,
                                     str(cfg.get("audit_log_dir", "audit_logs"))))
    if not args.no_audit:
        os.makedirs(audit_log_dir, exist_ok=True)

    results_dir = (args.results_dir
                   or os.path.join(ENGINE_REPO, "results", "mode_b_cert"))
    os.makedirs(results_dir, exist_ok=True)

    # prior covariance (FIX-T1.7/T1.8: Corr_l scaled by prior_sigma_l^2)
    Sigma_l, Sigma_t, r_fallback, sigma_note = resolve_sigma(cfg)

    # FIX-T1.8: classification-boundary prior mean = l* per domain
    l_prior_mean_vec: Optional[np.ndarray] = None
    if cfg.get("prior_mean_l_star", False):
        l_prior_mean_vec = l_star.copy()

    # data
    bank_signals = load_bank_signals(banks_dir, domains)
    raters = load_raters(rater_matrix_path)

    # ── header banner ─────────────────────────────────────────────────
    print("\nCertification engine parameters (Wave-1 corrected):")
    print(f"  K={K}  domains={domains}")
    print(f"  l_star={l_star.tolist()}")
    print(f"  stop_thresh={stop_thresh}  alpha={alpha}  "
          f"lapse_rate={lapse_rate}")
    print(f"  N_particles={n_particles}  max_q={max_q}  "
          f"n_reps={n_reps} (pilot={n_reps_pilot}, paper={n_reps_paper})")
    print(f"  methods={methods}  seed_base={args.seed_base}")
    print(f"  Sigma_l: {'fitted' if Sigma_l is not None else 'CS fallback'}  "
          f"({sigma_note})")
    print(f"  rater matrix: {rater_matrix_path}")
    print(f"  audit log dir: "
          f"{'(disabled)' if args.no_audit else audit_log_dir}")
    print(f"  results dir: {results_dir}")
    print(f"\nRaters: {len(raters)}\n", flush=True)

    hdr = (f"{'Rater':<30}  {'True AUROC (sz..iic)':<44}  "
           f"{'hier Nq':>7}  {'brute Nq':>8}  {'hier dec':>8}")
    print(hdr)
    print("-" * 100)

    rows: List[Dict[str, Any]] = []

    for _, rrow in raters.iterrows():
        name = str(rrow["confirmed_canonical_name"])
        true_params = build_true_params(rrow, domains)

        true_l = np.array([true_params[2 * k + 1] for k in range(K)])
        auroc_vals = auroc_from_l(true_l)
        true_pass = (true_l > l_star)

        hier_nqs: List[int] = []
        brute_nqs: List[int] = []
        hier_dec_strs: List[str] = []

        for rep in range(n_reps):
            seed = int(args.seed_base) + rep
            for method in methods:
                # ── audit trail (one file per rater+seed) ─────────────
                audit: Optional[AuditTrail] = None
                if not args.no_audit:
                    audit_path = session_audit_path(audit_log_dir, name, seed)
                    audit = AuditTrail(audit_path, flush_each_line=True)

                try:
                    out = run_one_cert(
                        method                = method,
                        true_params           = true_params,
                        bank_signals          = bank_signals,
                        seed                  = seed,
                        K                     = K,
                        l_star                = l_star,
                        Sigma_l               = Sigma_l,
                        Sigma_t               = Sigma_t,
                        r_fallback            = r_fallback,
                        stop_thresh           = stop_thresh,
                        alpha                 = alpha,
                        n_particles           = n_particles,
                        max_q                 = max_q,
                        ess_threshold_frac    = ess_threshold_frac,
                        n_mh_steps            = n_mh_steps,
                        proposal_scale_hier   = proposal_scale_hier,
                        proposal_scale_brute  = proposal_scale_brute,
                        n_min_guard           = n_min_guard,         # FIX-T1.7
                        l_prior_mean          = l_prior_mean_vec,    # FIX-T1.8
                        use_iut_stopping      = use_iut_stopping,    # FIX-T1.9
                        fisher_imin           = fisher_imin,         # FIX-T1.11
                    )

                    if audit is not None:
                        audit.record_session(
                            rater_id       = name,
                            method         = method,
                            seed           = seed,
                            domains        = domains,
                            l_star         = l_star,
                            true_params    = true_params,
                            lapse_rate     = lapse_rate,
                            session_result = out,
                            extra          = {
                                "tag": args.tag,
                                "config_path": os.path.abspath(args.config),
                                "rater_matrix": os.path.abspath(rater_matrix_path),
                                "banks_dir": os.path.abspath(banks_dir),
                                "sigma_source": (
                                    "fitted" if Sigma_l is not None
                                    else f"compound_symmetry_r={r_fallback}"
                                ),
                            },
                        )
                finally:
                    if audit is not None:
                        audit.close()

                nq = int(out["n_questions"])
                dec = np.asarray(out["decisions"])

                rows.append({
                    "rater":          name,
                    "method":         method,
                    "seed":           seed,
                    "n_questions":    nq,
                    "stopped_early":  int(bool(out["stopped_early"])),
                    "n_undecided":    len(out["active_at_end"]),
                    **{f"decision_{d}": int(dec[di])
                       for di, d in enumerate(domains)},
                    **{f"pass_prob_{d}": float(out["pass_probs_final"][di])
                       for di, d in enumerate(domains)},
                    **{f"mcse_{d}": float(out["mcse_final"][di])
                       for di, d in enumerate(domains)},
                    **{f"true_pass_{d}": int(true_pass[di])
                       for di, d in enumerate(domains)},
                    **{f"true_auroc_{d}": float(auroc_vals[di])
                       for di, d in enumerate(domains)},
                    "stop_thresh": float(out["stop_thresh_used"]),
                    "z_buffer":    float(out.get("z_buffer", float("nan"))),
                })

                if method == "hier":
                    hier_nqs.append(nq)
                    dec_str = "".join(
                        ("P" if d == 1 else ("F" if d == -1 else "?"))
                        for d in dec
                    )
                    hier_dec_strs.append(dec_str)
                else:
                    brute_nqs.append(nq)

        h_med = float(np.median(hier_nqs)) if hier_nqs else float("nan")
        b_med = float(np.median(brute_nqs)) if brute_nqs else float("nan")
        auroc_str = " ".join(f"{a:.3f}" for a in auroc_vals)
        dec_display = hier_dec_strs[0] if hier_dec_strs else "------"
        print(f"  {name:<28}  {auroc_str:<44}  {h_med:>7.0f}  {b_med:>8.0f}"
              f"  {dec_display:>8}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    df = pd.DataFrame(rows)
    results_path = os.path.join(results_dir, "mode_b_cert_results.csv")
    df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    if not df.empty:
        summary = (
            df.groupby(["rater", "method"])["n_questions"]
            .agg(["median", "mean", "std", "min", "max"])
            .reset_index()
        )
        summary_path = os.path.join(results_dir, "mode_b_cert_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")

        for method in methods:
            mdf = df[df["method"] == method]
            if mdf.empty:
                continue
            correct_by_domain: List[float] = []
            for d in domains:
                dec_col = f"decision_{d}"
                tp_col = f"true_pass_{d}"
                decided = mdf[dec_col] != 0
                if decided.any():
                    correct = ((mdf[dec_col][decided] == 1)
                               == mdf[tp_col][decided].astype(bool))
                    correct_by_domain.append(float(correct.mean()))
            overall_acc = (float(np.mean(correct_by_domain))
                           if correct_by_domain else float("nan"))
            undecided_frac = (
                mdf[[f"decision_{d}" for d in domains]] == 0
            ).any(axis=1).mean()
            print(f"\n{method}: accuracy={overall_acc:.1%}  "
                  f"undecided={float(undecided_frac):.1%}  "
                  f"median Nq={mdf['n_questions'].median():.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
