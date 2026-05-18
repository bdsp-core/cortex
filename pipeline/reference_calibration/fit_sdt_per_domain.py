"""Fit per-rater SDT parameters (σ, θ) for all six SPARCNET domains.

Adapts the fitting logic from train_val_split_and_fit.py but works entirely
from the long-format prepared CSVs — no H5 file required.

Model (identical to train_val_split_and_fit.py):
    P(y=1 | c; σ, θ) = λ + (1 − 2λ) · Φ((c − θ) / σ)
    λ = 0.025 (fixed), L-BFGS-B on (θ, log σ).

Scale conversion (critical): c_mean in {domain}_fit/c.csv is on the logit
scale. Divide by 1.7 before fitting so σ lands on the same probit scale as
the spike pipeline.

Outputs per domain:
    data/prepared/{domain}_sdt_fits.csv
    Columns: rater_id, rater_name, sigma, theta, se_sigma, se_theta,
             n_trials, converged

Run:
    python3 src/fit_sdt_per_domain.py              # all 6 SPARCNET domains
    python3 src/fit_sdt_per_domain.py sparcnet_sz  # single domain
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent
PREPARED = REPO / "data" / "prepared"
OUT_DIR = PREPARED

LAMBDA = 0.025
LOGIT_TO_PROBIT = 1.0 / 1.7
MIN_FIT_TRIALS = 20

DOMAINS = [
    "sparcnet_sz",
    "sparcnet_lpd",
    "sparcnet_gpd",
    "sparcnet_lrda",
    "sparcnet_grda",
    "sparcnet_iic",
]


def probit_lapse_nll(params: np.ndarray, c: np.ndarray, y: np.ndarray) -> float:
    t, log_sigma = params
    sigma = np.exp(log_sigma)
    z = (c - t) / sigma
    p = LAMBDA + (1.0 - 2.0 * LAMBDA) * norm.cdf(z)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_rater(
    c: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, bool, int]:
    """Returns (sigma, theta, se_sigma, se_theta, converged, n_trials).
    SE via 2D numerical Hessian; delta method for se_sigma = sigma * se_log_sigma.
    Returns NaN for SEs when Hessian is non-PSD.
    """
    n = len(y)
    if n < MIN_FIT_TRIALS:
        return (np.nan, np.nan, np.nan, np.nan, False, n)
    x0 = np.array([0.5, np.log(0.5)])
    res = minimize(
        probit_lapse_nll,
        x0,
        args=(c, y),
        method="L-BFGS-B",
        bounds=[(-2.0, 2.0), (np.log(0.02), np.log(5.0))],
    )
    if not res.success:
        return (np.nan, np.nan, np.nan, np.nan, False, n)
    t_hat, log_sigma = res.x
    sigma = float(np.exp(log_sigma))
    eps = 0.03
    f00 = probit_lapse_nll([t_hat, log_sigma], c, y)
    f_tp = probit_lapse_nll([t_hat + eps, log_sigma], c, y)
    f_tm = probit_lapse_nll([t_hat - eps, log_sigma], c, y)
    f_lp = probit_lapse_nll([t_hat, log_sigma + eps], c, y)
    f_lm = probit_lapse_nll([t_hat, log_sigma - eps], c, y)
    fpp = probit_lapse_nll([t_hat + eps, log_sigma + eps], c, y)
    fpm = probit_lapse_nll([t_hat + eps, log_sigma - eps], c, y)
    fmp = probit_lapse_nll([t_hat - eps, log_sigma + eps], c, y)
    fmm = probit_lapse_nll([t_hat - eps, log_sigma - eps], c, y)
    H = np.array(
        [
            [
                (f_tp - 2.0 * f00 + f_tm) / eps**2,
                (fpp - fpm - fmp + fmm) / (4.0 * eps**2),
            ],
            [
                (fpp - fpm - fmp + fmm) / (4.0 * eps**2),
                (f_lp - 2.0 * f00 + f_lm) / eps**2,
            ],
        ]
    )
    try:
        cov = np.linalg.inv(H)
        if cov[0, 0] <= 0 or cov[1, 1] <= 0:
            raise np.linalg.LinAlgError("Non-PSD Hessian")
        se_theta = float(np.sqrt(cov[0, 0]))
        se_log_sigma = float(np.sqrt(cov[1, 1]))
        se_sigma = sigma * se_log_sigma
    except np.linalg.LinAlgError:
        se_theta = np.nan
        se_sigma = np.nan
    return (sigma, float(t_hat), se_sigma, se_theta, True, n)


def fit_domain(domain: str) -> None:
    long_csv = PREPARED / f"{domain}.csv"
    c_csv = PREPARED / f"{domain}_fit" / "c.csv"
    l_csv = PREPARED / f"{domain}_fit" / "l.csv"
    out_csv = OUT_DIR / f"{domain}_sdt_fits.csv"

    if not long_csv.exists():
        print(f"  [{domain}] SKIP — {long_csv} not found")
        return
    if not c_csv.exists():
        print(f"  [{domain}] SKIP — {c_csv} not found")
        return

    print(f"\n=== {domain} ===")

    # Load Rasch c values and convert logit → probit
    c_df = pd.read_csv(c_csv)
    case_to_c = dict(
        zip(c_df["case_id"], c_df["c_mean"].astype(float) * LOGIT_TO_PROBIT)
    )
    print(f"  {len(case_to_c):,} cases with Rasch c (probit scale)")

    # Load rater names from l.csv (strip surrounding single quotes)
    rater_names: dict[int, str] = {}
    if l_csv.exists():
        l_df = pd.read_csv(l_csv)
        for _, row in l_df.iterrows():
            name = str(row["rater_name"]).strip("'")
            rater_names[int(row["rater_id"])] = name

    # Load long-format annotations
    ann = pd.read_csv(long_csv)
    ann["c_probit"] = ann["case_id"].map(case_to_c)
    ann = ann.dropna(subset=["c_probit"])
    print(f"  {len(ann):,} annotations with valid c_probit")

    rater_ids = sorted(ann["rater_id"].unique())
    print(f"  {len(rater_ids)} raters — fitting...")

    rows = []
    for rid in rater_ids:
        sub = ann[ann["rater_id"] == rid]
        c_arr = sub["c_probit"].to_numpy(dtype=np.float64)
        y_arr = sub["Y"].to_numpy(dtype=np.float64)
        sigma, theta, se_sigma, se_theta, converged, n = fit_rater(c_arr, y_arr)
        rows.append(
            {
                "rater_id": rid,
                "rater_name": rater_names.get(rid, f"rater_{rid}"),
                "sigma": sigma,
                "theta": theta,
                "se_sigma": se_sigma,
                "se_theta": se_theta,
                "n_trials": n,
                "converged": converged,
            }
        )

    fits = pd.DataFrame(rows)
    n_conv = fits["converged"].sum()
    print(f"  {n_conv}/{len(fits)} converged")

    if n_conv > 0:
        conv = fits[fits["converged"]]
        print(
            f"  sigma: median={conv['sigma'].median():.3f}, "
            f"mean={conv['sigma'].mean():.3f}, "
            f"range=[{conv['sigma'].min():.3f}, {conv['sigma'].max():.3f}]"
        )

    fits.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv}")


def main() -> None:
    domains = sys.argv[1:] if len(sys.argv) > 1 else DOMAINS
    for d in domains:
        if d not in DOMAINS:
            print(f"Unknown domain '{d}'. Valid: {DOMAINS}")
            sys.exit(1)
        fit_domain(d)
    print("\nDone.")


if __name__ == "__main__":
    main()
