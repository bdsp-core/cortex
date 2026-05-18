"""
Estimate per-case signal c_j and per-rater bias l_i from multi-rater binary
annotations using a main-effects logistic model:

    logit P(Y_ij = 1) = mu + c_j + l_i
    c_j ~ N(0, sigma_c^2),   l_i ~ N(0, sigma_l^2)

MAP estimation via block-coordinate Newton exploiting the bipartite
(rater, case) structure. Laplace-approximation posterior SDs are taken from
the diagonal of the Hessian (ignoring c<->l cross-terms, which are weak when
each case has few annotations but each rater has many).

Hyperparameters sigma_c, sigma_l are re-estimated between outer iterations
(empirical Bayes).

Outputs per prepared dataset, under data/prepared/<name>_fit/:
  c.csv        -- case_id, c_mean, c_sd, n_annotations, pos_rate
  l.csv        -- rater_id, rater_name, l_mean, l_sd, n_annotations
  summary.json -- mu, sigma_c, sigma_l, final log-posterior, iterations
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
import json
import time

DATA_DIR = Path("data/prepared")

DATASETS = [
    "combined_spike",
    "sparcnet_sz", "sparcnet_lpd", "sparcnet_gpd",
    "sparcnet_lrda", "sparcnet_grda", "sparcnet_iic",
]


def fit_main_effects(long, n_outer=120, tol=1e-5, verbose=True):
    case_ids = long["case_id"].to_numpy(dtype=np.int64)
    rater_ids = long["rater_id"].to_numpy(dtype=np.int64)
    Y = long["Y"].to_numpy(dtype=np.float64)
    N = len(Y)
    n_case = int(case_ids.max() + 1)
    n_rater = int(rater_ids.max() + 1)

    # init
    p_mean = np.clip(Y.mean(), 1e-3, 1 - 1e-3)
    mu = float(np.log(p_mean / (1 - p_mean)))
    c = np.zeros(n_case)
    l = np.zeros(n_rater)
    sigma_c = 2.0
    sigma_l = 1.0

    prev_lp = -np.inf
    n_case_obs = np.bincount(case_ids, minlength=n_case).astype(np.float64)
    n_rater_obs = np.bincount(rater_ids, minlength=n_rater).astype(np.float64)

    MAX_STEP = 1.0   # clamp per-parameter Newton step to avoid blow-up

    for it in range(n_outer):
        # --- update c (one Newton step per c_j given current mu, l) ---
        eta = mu + c[case_ids] + l[rater_ids]
        p = expit(eta)
        w = p * (1.0 - p)
        resid = p - Y
        grad = np.bincount(case_ids, weights=resid, minlength=n_case) \
               + c / sigma_c ** 2
        hess = np.bincount(case_ids, weights=w, minlength=n_case) \
               + 1.0 / sigma_c ** 2
        step = grad / hess
        np.clip(step, -MAX_STEP, MAX_STEP, out=step)
        c = c - step

        # --- update l ---
        eta = mu + c[case_ids] + l[rater_ids]
        p = expit(eta)
        w = p * (1.0 - p)
        resid = p - Y
        grad = np.bincount(rater_ids, weights=resid, minlength=n_rater) \
               + l / sigma_l ** 2
        hess = np.bincount(rater_ids, weights=w, minlength=n_rater) \
               + 1.0 / sigma_l ** 2
        step = grad / hess
        np.clip(step, -MAX_STEP, MAX_STEP, out=step)
        l = l - step

        # --- update mu ---
        eta = mu + c[case_ids] + l[rater_ids]
        p = expit(eta)
        grad = (p - Y).sum()
        hess = (p * (1.0 - p)).sum()
        mu_step = np.clip(grad / hess, -MAX_STEP, MAX_STEP)
        mu = mu - float(mu_step)

        # --- update hyperparameters via proper EM step ---
        # posterior variance approx: 1 / (likelihood precision + prior precision)
        eta = mu + c[case_ids] + l[rater_ids]
        p = expit(eta)
        w = p * (1.0 - p)
        hess_c_lik = np.bincount(case_ids, weights=w, minlength=n_case)
        hess_l_lik = np.bincount(rater_ids, weights=w, minlength=n_rater)
        post_var_c = 1.0 / (hess_c_lik + 1.0 / sigma_c ** 2)
        post_var_l = 1.0 / (hess_l_lik + 1.0 / sigma_l ** 2)
        # E[c^2] = c_MAP^2 + post_var_c ; then sigma_c = sqrt(mean(E[c^2]))
        sigma_c_new = float(np.sqrt((c * c + post_var_c).mean()))
        sigma_l_new = float(np.sqrt((l * l + post_var_l).mean()))
        # light damping on sigma updates
        sigma_c = 0.5 * sigma_c + 0.5 * max(sigma_c_new, 0.1)
        sigma_l = 0.5 * sigma_l + 0.5 * max(sigma_l_new, 0.05)

        # log-posterior (for monitoring)
        eta = mu + c[case_ids] + l[rater_ids]
        log_p = -np.logaddexp(0.0, -eta)
        log_q = -np.logaddexp(0.0, eta)
        lp = (Y * log_p + (1.0 - Y) * log_q).sum() \
             - 0.5 * (c * c).sum() / sigma_c ** 2 \
             - 0.5 * (l * l).sum() / sigma_l ** 2 \
             - n_case * np.log(sigma_c) - n_rater * np.log(sigma_l)

        if verbose and (it % 5 == 0 or it == n_outer - 1):
            print(f"    it={it:3d}  lp={lp:.2f}  sigma_c={sigma_c:.3f}  "
                  f"sigma_l={sigma_l:.3f}  mu={mu:.3f}")

        if abs(lp - prev_lp) < tol * (abs(prev_lp) + 1.0):
            if verbose:
                print(f"    converged at it={it}")
            break
        prev_lp = lp

    # Laplace-approximation posterior SDs (diagonal)
    eta = mu + c[case_ids] + l[rater_ids]
    p = expit(eta)
    w = p * (1.0 - p)
    hess_c_diag = np.bincount(case_ids, weights=w, minlength=n_case) \
                  + 1.0 / sigma_c ** 2
    hess_l_diag = np.bincount(rater_ids, weights=w, minlength=n_rater) \
                  + 1.0 / sigma_l ** 2
    sd_c = 1.0 / np.sqrt(hess_c_diag)
    sd_l = 1.0 / np.sqrt(hess_l_diag)

    return dict(
        mu=float(mu), c=c, l=l, sd_c=sd_c, sd_l=sd_l,
        sigma_c=float(sigma_c), sigma_l=float(sigma_l),
        n_case_obs=n_case_obs, n_rater_obs=n_rater_obs,
        log_posterior=float(lp), iterations=int(it + 1),
    )


def fit_one(name):
    print(f"\n=== {name} ===")
    t0 = time.time()
    long = pd.read_csv(DATA_DIR / f"{name}.csv")
    meta = json.load(open(DATA_DIR / f"{name}_meta.json"))
    rater_names = meta["rater_index_to_name"]

    r = fit_main_effects(long)
    t1 = time.time()
    print(f"  fit time: {t1 - t0:.1f}s")

    fit_dir = DATA_DIR / f"{name}_fit"
    fit_dir.mkdir(exist_ok=True)

    # per-case
    case_pos_rate = long.groupby("case_id")["Y"].mean().reindex(
        range(len(r["c"])), fill_value=np.nan).to_numpy()
    c_df = pd.DataFrame({
        "case_id": np.arange(len(r["c"])),
        "c_mean": r["c"], "c_sd": r["sd_c"],
        "n_annotations": r["n_case_obs"].astype(int),
        "pos_rate": case_pos_rate,
    })
    c_df.to_csv(fit_dir / "c.csv", index=False)

    # per-rater
    l_df = pd.DataFrame({
        "rater_id": np.arange(len(r["l"])),
        "rater_name": rater_names,
        "l_mean": r["l"], "l_sd": r["sd_l"],
        "n_annotations": r["n_rater_obs"].astype(int),
    })
    l_df.to_csv(fit_dir / "l.csv", index=False)

    # summary
    summary = dict(
        name=name, mu=r["mu"],
        sigma_c=r["sigma_c"], sigma_l=r["sigma_l"],
        log_posterior=r["log_posterior"],
        iterations=r["iterations"],
        fit_seconds=round(t1 - t0, 2),
        n_obs=int(len(long)),
        n_cases=int(len(r["c"])),
        n_raters=int(len(r["l"])),
    )
    with open(fit_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    summaries = []
    for name in DATASETS:
        summaries.append(fit_one(name))
    print("\n=== All fits ===")
    for s in summaries:
        print(f"  {s['name']:20s}  mu={s['mu']:+.2f}  "
              f"sigma_c={s['sigma_c']:.2f}  sigma_l={s['sigma_l']:.2f}  "
              f"time={s['fit_seconds']:.1f}s")


if __name__ == "__main__":
    main()
