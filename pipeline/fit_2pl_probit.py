"""Single-task batch fitter for the 2-parameter probit IRT model.

Model (per task k, all (rater i, case j) labels):

    P(Y_ij = 1) = Phi( exp(l_i) * (s_j + t_i) )

  s_j ~ N(0, sigma_s^2)     case signal strength
  t_i ~ N(0, sigma_t^2)     rater bias / threshold
  l_i ~ N(0, sigma_l^2)     rater log-skill

Inference: MAP via block-coordinate IRLS (probit Fisher scoring).
Posterior SDs via diagonal of the observed-information Hessian (Laplace
approximation, ignoring off-block cross-terms).

Filters applied iteratively until stable:
  - drop cases with < MIN_RATERS_PER_CASE raters
  - drop raters with < MIN_LABELS_PER_RATER labels in this task

Outputs per task to data/labels/fits/<task>/:
  cases.csv     seg_id, s_mean, s_sd, n_raters, pos_rate
  raters.csv    rater_id, canonical_name, group, t_mean, t_sd,
                ell_mean, ell_sd, n_labels, pos_rate
  summary.json  hyperparameters + log-posterior + iters

UNIFIED-MERGE PROVENANCE (Phase 4.4-B, 2026-05-19). Faithful port of the
PI scripts/fit_2pl_probit.py. Documented, auditable changes vs PI:
  - the PI hardcoded absolute-ROOT line -> repo-root self-location
    (Path(__file__).resolve().parents[1]; zero absolute paths). Every
    derived ROOT/"data/..." path stays byte-identical.
  - TASKS: the degenerate composite "iic" -> the REAL "other" (the 7th
    IIIC task; Sigma goes 12x12 -> 14x14 downstream).
  - extract_task_labels: the degenerate iic = OR(seizure,lpd,gpd,lrda,
    grda) branch is DELETED; "other" uses the ERRATUM-CORRECT mapping
    value in {other,bipd,birds} -- MANDATED by consistency with the
    Phase-3/v13 sparcnet_iic definition the deployment ell* consumes at
    Phase 4.5 (uniform {bipd,birds}->other; AUDIT s6 / Phase-1). On the
    UNIFIED corpus the "other" positive count is 204,163 (= Phase-3
    sparcnet_iic), NOT the plan's stale pre-merge 79,383 (PI's old
    value=='other' on a smaller corpus).
Phase 4.6-A (2026-05-19) additionally makes the "spike" branch
CLEAN-SN1-ONLY (Centaur-IED EXCLUDED) — consistent with v13
combined_spike (the Phase-3.5 spike un-fold) and with
fit_2pl_probit_hier; so the deployment spike BANK shares one
population with its v13 ℓ*. Deliberate v13-consistency correction to
the port (cf. erratum-correct "other"), not a faithful-PI deviation
bug. 4.6-A EXERCISES this script (re-fit on the unified corpus).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import log_ndtr   # numerically stable log(Phi)

ROOT = Path(__file__).resolve().parents[1]   # repo root (was hardcoded)
LABELS_CSV = ROOT / "data/labels/labels.csv"
RATERS_CSV = ROOT / "data/labels/raters.csv"
OUT_DIR = ROOT / "data/labels/fits"

TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]

DEFAULTS = {
    "min_raters_per_case": 3,
    "min_labels_per_rater": 100,
    # Drop raters whose vote distribution is near-degenerate (always-no or
    # always-yes). For such raters the MLE of log-skill ℓ is unbounded (the
    # complete-separation problem) and the Laplace SD collapses to the prior
    # — they end up in the high-SD cluster we observed in sd_diagnostic.png.
    # We require at least this many votes of *each* class (0 and 1).
    "min_votes_per_class": 5,
    "max_outer": 40,
    "tol": 1e-4,
    "max_inner_l": 5,             # 1-D Newton steps for ell per outer iter
    "init_sigma_s": 2.0,
    "init_sigma_t": 1.0,
    "init_sigma_l": 0.5,
}


# ───────────────────────── data extraction ─────────────────────────

def extract_task_labels(labels: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return long-form (seg_id, rater_id, Y) for the given binary task."""
    if task == "spike":
        sub = labels[labels.label_type == "spike"].copy()
        # Phase 4.6-A (decision 2026-05-19): CLEAN-SN1-ONLY spike, made
        # CONSISTENT with v13 `combined_spike` (the Phase-3.5 spike
        # UN-FOLD: Centaur-IED is EXCLUDED from the spike cert task) AND
        # with fit_2pl_probit_hier's spike (`pd.to_numeric`). The PI
        # port's "treat Centaur 'ied'/'spike' strings as positive" is
        # REMOVED so the deployment spike BANK shares ONE population
        # with its ℓ* (v13 combined_spike = clean sn1 binary). sn1 is
        # literal "0"/"1" → numeric; Centaur 6-class strings →
        # to_numeric NaN → dropped by the trailing .dropna(). This is a
        # deliberate v13-consistency correction to the faithful port
        # (cf. the erratum-correct "other" in 4.4-B), documented here
        # and in the provenance header.
        sub["Y"] = pd.to_numeric(sub["value"], errors="coerce")
    else:
        sub = labels[labels.label_type == "pattern_class"].copy()
        if task == "other":
            # Phase 4.4-B: the REAL "other" IIIC task (replaces the
            # DELETED degenerate iic=OR). ERRATUM-CORRECT uniform
            # {bipd,birds}->other collapse (AUDIT §6 / Phase-1) — MUST
            # match the Phase-3/v13 sparcnet_iic Y the deployment ℓ*
            # consumes at 4.5 (unified positive count 204,163).
            sub["Y"] = (sub["value"].isin(
                ["other", "bipd", "birds"])).astype(int)
        else:
            sub["Y"] = (sub["value"] == task).astype(int)
    return sub[["seg_id", "rater_id", "Y"]].dropna()


def filter_iteratively(df: pd.DataFrame, min_r: int, min_l: int,
                        min_votes_per_class: int = 5) -> pd.DataFrame:
    """Repeatedly drop low-coverage cases and raters until stable.

    Three filters interact (removing one cascade can re-trigger another):
      - raters with < min_l labels         → dropped
      - raters with < min_votes_per_class positive OR negative votes → dropped
        (avoids unanimous-vote raters whose ℓ is unidentifiable)
      - cases with < min_r raters labeling → dropped
    """
    prev_n = -1
    n = len(df)
    while n != prev_n:
        prev_n = n
        # raters with too few labels (either class) OR too few of either kind
        rc_total = df.groupby("rater_id").size()
        rc_pos = df.groupby("rater_id")["Y"].sum()
        rc_neg = rc_total - rc_pos
        ok_r = rc_total[(rc_total >= min_l)
                          & (rc_pos >= min_votes_per_class)
                          & (rc_neg >= min_votes_per_class)].index
        df = df[df.rater_id.isin(ok_r)]
        # cases with too few raters
        cc = df.groupby("seg_id").size()
        ok_c = cc[cc >= min_r].index
        df = df[df.seg_id.isin(ok_c)]
        n = len(df)
    return df


# ───────────────────────── probit IRLS pieces ─────────────────────────

def irls_z_w(eta: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probit IRLS working response z and weight w at current eta.

    Standard probit Fisher scoring. To prevent weight collapse at extreme |eta|
    (Φ → 0 or 1 → information vanishes), clip eta to [-6, 6] for the IRLS
    weight/residual computation. Beyond ±6 the probit is essentially saturated;
    capping the working point keeps subsequent Newton steps well-defined while
    the prior still pulls the underlying parameter back.
    """
    eta_c = np.clip(eta, -6.0, 6.0)
    phi = norm.pdf(eta_c)
    Phi = norm.cdf(eta_c)
    eps = 1e-10
    Phi_c = np.clip(Phi, eps, 1 - eps)
    phi_c = np.clip(phi, eps, None)
    w = phi_c ** 2 / (Phi_c * (1 - Phi_c))
    r = (Y - Phi_c) / phi_c
    z = eta_c + r
    return z, w


def neg_log_post(s, t, l, case_idx, rater_idx, Y, sigma_s, sigma_t, sigma_l):
    eta = np.exp(l[rater_idx]) * (s[case_idx] + t[rater_idx])
    # log-lik via log_ndtr for stability
    log_p = log_ndtr(eta)
    log_1mp = log_ndtr(-eta)
    nll = -(Y * log_p + (1 - Y) * log_1mp).sum()
    prior = (0.5 * (s ** 2).sum() / sigma_s ** 2
             + 0.5 * (t ** 2).sum() / sigma_t ** 2
             + 0.5 * (l ** 2).sum() / sigma_l ** 2)
    return nll + prior


# ───────────────────────── block updates ─────────────────────────

def update_s(s, t, l, case_idx, rater_idx, Y, sigma_s):
    """Probit IRLS update of s, holding t and l fixed.

    eta_ij = e^{l_i} (s_j + t_i)
    Holding t, l: derivative of eta w.r.t. s_j is e^{l_i} for any rater i
    who labeled case j. So update is per-case 1-D weighted least squares.
    """
    eta = np.exp(l[rater_idx]) * (s[case_idx] + t[rater_idx])
    z, w = irls_z_w(eta, Y)
    el = np.exp(l[rater_idx])
    # Target equation per (i,j): z_ij = el * (s_j + t_i)
    # → working response on s-scale: z/el - t_i,  weight: w * el^2
    target = z / el - t[rater_idx]
    weight = w * el ** 2
    # accumulate per-case
    n_case = len(s)
    num = np.bincount(case_idx, weights=weight * target, minlength=n_case)
    den = np.bincount(case_idx, weights=weight, minlength=n_case) + 1.0 / sigma_s ** 2
    s_new = num / den
    sd = 1.0 / np.sqrt(den)
    return s_new, sd


def update_t(s, t, l, case_idx, rater_idx, Y, sigma_t):
    """Probit IRLS update of t, holding s and l fixed."""
    eta = np.exp(l[rater_idx]) * (s[case_idx] + t[rater_idx])
    z, w = irls_z_w(eta, Y)
    el = np.exp(l[rater_idx])
    target = z / el - s[case_idx]
    weight = w * el ** 2
    n_rat = len(t)
    num = np.bincount(rater_idx, weights=weight * target, minlength=n_rat)
    den = np.bincount(rater_idx, weights=weight, minlength=n_rat) + 1.0 / sigma_t ** 2
    t_new = num / den
    sd = 1.0 / np.sqrt(den)
    return t_new, sd


def update_l(s, t, l, case_idx, rater_idx, Y, sigma_l, n_inner=5):
    """1-D Newton on each l_i. eta_ij = e^{l_i} u_ij with u_ij = s_j + t_i.

    Score: sum_j r_ij * eta_ij  - l_i / sigma_l^2
    Fisher info: sum_j w_ij * eta_ij^2 + 1/sigma_l^2
    """
    n_rat = len(l)
    u = s[case_idx] + t[rater_idx]
    l_new = l.copy()
    for _ in range(n_inner):
        el = np.exp(l_new[rater_idx])
        eta = el * u
        z, w = irls_z_w(eta, Y)
        # score per obs in eta-coord: w_ij * (z_ij - eta_ij) is the linearized residual
        # gradient of log-lik w.r.t. l_i at this point: sum_j w_ij * eta_ij * (z_ij - eta_ij)
        # (using IRLS: ∂eta/∂l = eta)
        resid = z - eta
        score_l = np.bincount(rater_idx, weights=w * eta * resid, minlength=n_rat) \
                  - l_new / sigma_l ** 2
        info_l = np.bincount(rater_idx, weights=w * eta ** 2, minlength=n_rat) \
                 + 1.0 / sigma_l ** 2
        step = score_l / info_l
        # clip step for stability
        np.clip(step, -0.5, 0.5, out=step)
        l_new = l_new + step
    sd = 1.0 / np.sqrt(info_l)
    return l_new, sd


# ───────────────────────── empirical Bayes ─────────────────────────

def update_hyperparams(s, t, l):
    """Empirical Bayes update of the prior SDs."""
    sigma_s = np.sqrt((s ** 2).mean()) if len(s) else 1.0
    sigma_t = np.sqrt((t ** 2).mean()) if len(t) else 1.0
    sigma_l = np.sqrt((l ** 2).mean()) if len(l) else 1.0
    sigma_s = max(sigma_s, 0.1); sigma_t = max(sigma_t, 0.1); sigma_l = max(sigma_l, 0.1)
    return sigma_s, sigma_t, sigma_l


# ───────────────────────── main fit loop ─────────────────────────

def fit_one_task(task, labels, raters, opts, verbose=True):
    df = extract_task_labels(labels, task)
    n_raw = len(df)
    if verbose:
        print(f"  raw labels for task '{task}': {n_raw:,}", flush=True)

    df = filter_iteratively(df, opts["min_raters_per_case"],
                             opts["min_labels_per_rater"])
    n_kept = len(df)
    if verbose:
        n_cases = df.seg_id.nunique()
        n_raters = df.rater_id.nunique()
        pos_rate = df.Y.mean()
        print(f"  after filter (min_r={opts['min_raters_per_case']}, "
              f"min_l={opts['min_labels_per_rater']}): {n_kept:,} labels, "
              f"{n_cases} cases, {n_raters} raters, pos_rate={pos_rate:.3f}",
              flush=True)
    if n_kept == 0 or df.seg_id.nunique() < 2:
        print("  nothing to fit; skipping.")
        return None

    # Compact indexing
    seg_codes, seg_uniques = pd.factorize(df.seg_id.values, sort=True)
    rat_codes, rat_uniques = pd.factorize(df.rater_id.values, sort=True)
    Y = df.Y.to_numpy(dtype=np.float64)
    case_idx = seg_codes.astype(np.int64)
    rater_idx = rat_codes.astype(np.int64)
    n_case = len(seg_uniques); n_rater = len(rat_uniques)

    # Initialize s to a probit transform of the per-case empirical pos_rate
    # (clamped well inside the saturation range so IRLS has positive weight).
    case_pos = np.bincount(case_idx, weights=Y, minlength=n_case)
    case_n = np.bincount(case_idx, minlength=n_case).astype(float)
    case_p = np.clip(case_pos / np.maximum(case_n, 1.0), 0.02, 0.98)
    s = norm.ppf(case_p).astype(float)
    s = np.clip(s, -3.0, 3.0)
    t = np.zeros(n_rater); l = np.zeros(n_rater)
    s_sd = np.full(n_case, np.nan); t_sd = np.full(n_rater, np.nan)
    l_sd = np.full(n_rater, np.nan)
    sigma_s = opts["init_sigma_s"]; sigma_t = opts["init_sigma_t"]
    sigma_l = opts["init_sigma_l"]

    # All three priors fixed (matches the slides' marginal priors). EB on σ_s
    # interacts badly with the scaling-identifiability degeneracy on unbalanced
    # tasks (very low pos_rate), so we fix it at the slides' default.
    sigma_t = 1.0
    sigma_l = 1.0
    sigma_s = 2.0

    # Hard clamps to keep parameters in their identifiable / interpretable
    # range. The priors weakly pin these too, but at large N the likelihood
    # can briefly overshoot before the prior catches up; clamping per step
    # prevents single-rater runaways.
    S_MAX, T_MAX, L_MAX = 5.0, 3.0, 3.0

    prev_obj = np.inf
    for it in range(opts["max_outer"]):
        s, s_sd = update_s(s, t, l, case_idx, rater_idx, Y, sigma_s)
        np.clip(s, -S_MAX, S_MAX, out=s)
        # Shift-identifiability: a constant added to all (s_j) and subtracted
        # from all (t_i) leaves eta unchanged. Center t by its mean.
        t_mean = t.mean()
        t = t - t_mean
        s = s + t_mean
        np.clip(s, -S_MAX, S_MAX, out=s)
        t, t_sd = update_t(s, t, l, case_idx, rater_idx, Y, sigma_t)
        np.clip(t, -T_MAX, T_MAX, out=t)
        l, l_sd = update_l(s, t, l, case_idx, rater_idx, Y, sigma_l,
                            n_inner=opts["max_inner_l"])
        np.clip(l, -L_MAX, L_MAX, out=l)
        obj = neg_log_post(s, t, l, case_idx, rater_idx, Y,
                            sigma_s, sigma_t, sigma_l)
        if verbose and (it % 5 == 0 or it < 3):
            print(f"    it={it:3d}  obj={obj:.1f}  σ_s={sigma_s:.3f} "
                  f"σ_t={sigma_t:.3f} σ_l={sigma_l:.3f}", flush=True)
        if abs(prev_obj - obj) < opts["tol"] * max(1.0, abs(obj)):
            if verbose:
                print(f"    converged at it={it}", flush=True)
            break
        prev_obj = obj

    # Build output frames
    case_stats = (df.groupby("seg_id")
                  .agg(n_raters=("Y", "size"), pos_rate=("Y", "mean"))
                  .reset_index())
    case_stats = case_stats.set_index("seg_id").loc[seg_uniques].reset_index()
    cases_df = pd.DataFrame({
        "seg_id": seg_uniques,
        "s_mean": s,
        "s_sd": s_sd,
        "n_raters": case_stats.n_raters.values,
        "pos_rate": case_stats.pos_rate.values,
    })

    rater_stats = (df.groupby("rater_id")
                   .agg(n_labels=("Y", "size"), pos_rate=("Y", "mean"))
                   .reset_index())
    rater_stats = rater_stats.set_index("rater_id").loc[rat_uniques].reset_index()
    # join in canonical info
    raters_join = raters.set_index("rater_id")[["canonical_name", "groups",
                                                  "expertise_level"]]
    rater_meta = raters_join.reindex(rat_uniques).reset_index()
    raters_df = pd.DataFrame({
        "rater_id": rat_uniques,
        "canonical_name": rater_meta.canonical_name.values,
        "groups": rater_meta.groups.values,
        "expertise_level": rater_meta.expertise_level.values,
        "t_mean": t, "t_sd": t_sd,
        "ell_mean": l, "ell_sd": l_sd,
        "n_labels": rater_stats.n_labels.values,
        "pos_rate": rater_stats.pos_rate.values,
    })

    summary = {
        "task": task,
        "n_labels_raw": int(n_raw),
        "n_labels_fit": int(n_kept),
        "n_cases": int(n_case),
        "n_raters": int(n_rater),
        "min_raters_per_case": opts["min_raters_per_case"],
        "min_labels_per_rater": opts["min_labels_per_rater"],
        "sigma_s_eb": float(sigma_s),
        "sigma_t_eb": float(sigma_t),
        "sigma_l_eb": float(sigma_l),
        "final_neg_log_post": float(obj),
        "iters": int(it + 1),
    }
    return cases_df, raters_df, summary


# ───────────────────────── main ─────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--min-raters-per-case", type=int,
                    default=DEFAULTS["min_raters_per_case"])
    p.add_argument("--min-labels-per-rater", type=int,
                    default=DEFAULTS["min_labels_per_rater"])
    p.add_argument("--max-outer", type=int, default=DEFAULTS["max_outer"])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    opts = dict(DEFAULTS)
    opts.update(dict(
        min_raters_per_case=args.min_raters_per_case,
        min_labels_per_rater=args.min_labels_per_rater,
        max_outer=args.max_outer,
    ))

    print(f"Reading labels...", flush=True)
    labels = pd.read_csv(LABELS_CSV)
    raters = pd.read_csv(RATERS_CSV)
    print(f"  labels: {len(labels):,} rows; raters: {len(raters):,}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall = []
    for task in args.tasks:
        print(f"\n=== fitting task: {task} ===")
        t0 = time.perf_counter()
        out = fit_one_task(task, labels, raters, opts,
                            verbose=not args.quiet)
        if out is None:
            continue
        cases_df, raters_df, summary = out
        summary["wall_time_s"] = round(time.perf_counter() - t0, 1)
        outdir = OUT_DIR / task
        outdir.mkdir(parents=True, exist_ok=True)
        cases_df.to_csv(outdir / "cases.csv", index=False)
        raters_df.to_csv(outdir / "raters.csv", index=False)
        with open(outdir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  → wrote {outdir} ({summary['wall_time_s']}s)", flush=True)
        overall.append(summary)

    with open(OUT_DIR / "fit_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nAll done. Summary in {OUT_DIR/'fit_summary.json'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
