"""Does exploiting the cross-task skill correlation let the multi-task test
stop earlier?  —  experiment on real data.

Setup
-----
K = 6 IIIC tasks (seizure, lpd, gpd, lrda, grda, other) — exactly what the
cortex_web deployable test certifies. The fitted population prior Σ has a
cross-task ℓ–ℓ correlation r_ℓ ≈ 0.37 (real raters are empirically ~0.78
correlated). Same validated engine (deployment/simulate_test.py machinery),
two priors, paired by candidate:
  * CORRELATED  : the fitted Σ (cross-task ℓ coupling intact).
  * INDEPENDENT : Σ block-diagonalised to each task's own 2×2 (t,ℓ) block
                  (zero cross-task) — i.e. the K tasks tested in isolation.

Two stopping targets (both: the ideal-test gray-zone Bayesian SPRT, mapped
σ→ℓ; higher ℓ = better):
  (1) PER-TASK  — every task must resolve:
        PASS_k iff P(ℓ_k ≥ ℓ*_k+δ)≥STOP_P ; FAIL_k iff P(ℓ_k ≤ ℓ*_k−δ)≥STOP_P
      stop when all K resolved.
  (2) AGGREGATE — one overall competence verdict on the mean margin
        m = (1/K) Σ_k (ℓ_k − ℓ*_k):
        PASS iff P(m ≥ +δ)≥STOP_P ; FAIL iff P(m ≤ −δ)≥STOP_P.
      m is linear in θ, so its posterior is exact: mean aᵀμ, var aᵀΣ_post a.
      Every question on ANY task informs m — this is where a shared-factor
      correlation pays off.

True candidates ("real data"):
  * REAL  : the 812 clinicians fitted across all 6 IIIC tasks
            (data/labels/fits_hier_block/<task>/raters.csv) → real θ each.
  * POP   : θ ~ N(0, Σ_correlated), large-N smooth reference.
  * CONC  : concordant sweep — all tasks at a common ℓ (mechanism check).
Responses are generated from the candidate's true θ via the engine's λ-lapse
probit. Both priors face the same candidate and the same RNG.

Outputs land in out/ (CSVs + summary.json); plots are made by make_plots.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "engine", ROOT, ROOT / "deployment"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import simulate_test as ST  # noqa: E402

IIIC = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
FITS = ROOT / "data" / "labels" / "fits_hier_block"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# stopping-rule config
STOP_P = 0.95
DELTA = 0.10
N_MIN_PER_TASK = 3
N_MAX_PER_TASK = 80
N_MAX_TOTAL = 6 * N_MAX_PER_TASK
SEEDS = (0,)                 # one response-noise draw per candidate (N is large)
POOL_PER_TASK = 3000         # dense random per-task item pool (from the ~40k raw)
POOL_SEED = 12345
POP_N = 800
RNG_POP = 20260529
LAM = ST.LAPSE_RATE
ONE_M_2L = ST._ONE_MINUS_2LAMBDA


# ───────────────────────── prior + bank ─────────────────────────

def load_iiic():
    Sigma7, bank7, ell7 = ST.load_deployment()
    tasks7 = ST.deployment_task_names()
    idx, keep = [], []
    for t in IIIC:
        k = tasks7.index(t)
        keep.append(k)
        idx += [2 * k, 2 * k + 1]
    Sigma6 = Sigma7[np.ix_(idx, idx)]
    ell6 = np.array([ell7[k] for k in keep])
    # dense random per-task pool from the raw corpus (retains the informative
    # near-boundary items a sparse linspace pool would miss).
    rng = np.random.default_rng(POOL_SEED)
    pools = []
    for k in keep:
        s = bank7[k]["s_mean"].to_numpy()
        if len(s) > POOL_PER_TASK:
            s = rng.choice(s, size=POOL_PER_TASK, replace=False)
        pools.append(np.ascontiguousarray(s, dtype=float))
    return Sigma6, ell6, pools


def block_diagonal(Sigma, K):
    B = np.zeros_like(Sigma)
    for k in range(K):
        a, b = 2 * k, 2 * k + 2
        B[a:b, a:b] = Sigma[a:b, a:b]
    return B


# ───────────────────────── fast engine inner loop ─────────────────────────

def _lapse(eta_c):
    Phi = norm.cdf(eta_c)
    p = LAM + ONE_M_2L * Phi
    pp = ONE_M_2L * norm.pdf(eta_c)
    return p, pp


def _select(mu, Sig, cand_tasks, pools, used):
    """Numpy port of ST.select_next_case scoring (no pandas)."""
    best = None
    for k in cand_tasks:
        free = ~used[k]
        if not free.any():
            continue
        s_arr = pools[k][free]
        jt, jl = 2 * k, 2 * k + 1
        t = mu[jt]
        el = np.exp(mu[jl])
        eta = el * (s_arr + t)
        Stt, Stl, Sll = Sig[jt, jt], Sig[jt, jl], Sig[jl, jl]
        v = (el ** 2) * (Stt + 2 * (s_arr + t) * Stl + (s_arr + t) ** 2 * Sll)
        eta_c = np.clip(eta, -6.0, 6.0)
        p, pp = _lapse(eta_c)
        w = pp ** 2 / (p * (1.0 - p))
        score = np.log1p(np.maximum(w * v, 0.0))
        i = int(np.argmax(score))
        if best is None or score[i] > best[0]:
            loc = np.flatnonzero(free)[i]
            best = (score[i], k, loc, float(s_arr[i]))
    return best


def _update(mu, Sig, k, s, Y):
    """One Newton step (mirrors ST.update_state exactly)."""
    jt, jl = 2 * k, 2 * k + 1
    el = np.exp(mu[jl])
    eta = el * (s + mu[jt])
    eta_c = np.clip(eta, -6.0, 6.0)
    p, pp = _lapse(eta_c)
    w = float(pp ** 2 / (p * (1.0 - p)))
    pp_c = max(float(pp), 1e-12)
    z = eta_c + (Y - p) / pp_c
    D = mu.shape[0]
    x = np.zeros(D)
    x[jt] = el
    x[jl] = el * (s + mu[jt])
    Sinv = np.linalg.inv(Sig)
    Sinv2 = Sinv + w * np.outer(x, x)
    Sig2 = np.linalg.inv(Sinv2)
    mu2 = Sig2 @ (Sinv @ mu + w * x * z)
    K = D // 2
    for kk in range(K):
        mu2[2 * kk + 1] = np.clip(mu2[2 * kk + 1], -3, 3)
        mu2[2 * kk] = np.clip(mu2[2 * kk], -3, 3)
    return mu2, Sig2


def _p_pass_marg(mu, Sig, ell_thr):
    """P(ℓ_k > ell_thr_k) per task under the Gaussian marginal."""
    K = len(ell_thr)
    out = np.empty(K)
    for k in range(K):
        m = mu[2 * k + 1]
        sd = max(np.sqrt(Sig[2 * k + 1, 2 * k + 1]), 1e-6)
        out[k] = 1.0 - norm.cdf(ell_thr[k], loc=m, scale=sd)
    return out


def run_pertask(theta, Sig0, ell, pools, rng):
    """Per-task all-resolve SPRT. Returns (n_per_task, decisions)."""
    K = len(ell)
    mu = np.zeros(2 * K)
    Sig = Sig0.copy()
    npt = np.zeros(K, dtype=int)
    dec = ["pending"] * K
    used = [np.zeros(len(pools[k]), dtype=bool) for k in range(K)]
    for _ in range(N_MAX_TOTAL):
        pend = [k for k in range(K) if dec[k] == "pending"]
        if not pend:
            break
        under = [k for k in pend if npt[k] < N_MIN_PER_TASK]
        cand = under if under else pend
        sel = _select(mu, Sig, cand, pools, used)
        if sel is None:
            break
        _, k, loc, s = sel
        used[k][loc] = True
        Y = _draw(theta, k, s, rng)
        mu, Sig = _update(mu, Sig, k, s, Y)
        npt[k] += 1
        pp = _p_pass_marg(mu, Sig, ell + DELTA)
        pf = 1.0 - _p_pass_marg(mu, Sig, ell - DELTA)
        for kk in pend:
            if npt[kk] < N_MIN_PER_TASK:
                continue
            if pp[kk] >= STOP_P:
                dec[kk] = "pass"
            elif pf[kk] >= STOP_P:
                dec[kk] = "fail"
            elif npt[kk] >= N_MAX_PER_TASK:
                dec[kk] = "refer"
    for kk in range(K):
        if dec[kk] == "pending":
            dec[kk] = "refer"
    return npt, dec


def run_aggregate(theta, Sig0, ell, pools, rng):
    """Aggregate competence SPRT on m = mean_k(ℓ_k − ℓ*_k). Returns
    (total_q, decision, n_per_task)."""
    K = len(ell)
    a = np.zeros(2 * K)
    a[1::2] = 1.0 / K                 # selector on the ℓ-slots
    c = float(np.mean(ell))           # m = aᵀθ − c
    mu = np.zeros(2 * K)
    Sig = Sig0.copy()
    npt = np.zeros(K, dtype=int)
    used = [np.zeros(len(pools[k]), dtype=bool) for k in range(K)]
    dec = "refer"
    total = 0
    for n in range(1, N_MAX_TOTAL + 1):
        cand = [k for k in range(K) if not used[k].all()]
        if not cand:
            break
        # minimal coverage: ensure each task seen N_MIN_PER_TASK before free pick
        under = [k for k in cand if npt[k] < N_MIN_PER_TASK]
        sel = _select(mu, Sig, under if under else cand, pools, used)
        if sel is None:
            break
        _, k, loc, s = sel
        used[k][loc] = True
        Y = _draw(theta, k, s, rng)
        mu, Sig = _update(mu, Sig, k, s, Y)
        npt[k] += 1
        total = n
        m_mean = float(a @ mu) - c
        m_sd = float(np.sqrt(max(a @ Sig @ a, 1e-12)))
        p_pass = 1.0 - norm.cdf(+DELTA, loc=m_mean, scale=m_sd)
        p_fail = norm.cdf(-DELTA, loc=m_mean, scale=m_sd)
        if n >= K * N_MIN_PER_TASK:
            if p_pass >= STOP_P:
                dec = "pass"
                break
            if p_fail >= STOP_P:
                dec = "fail"
                break
    return total, dec, npt


def _draw(theta, k, s, rng):
    eta = np.exp(theta[2 * k + 1]) * (s + theta[2 * k])
    p = LAM + ONE_M_2L * float(norm.cdf(eta))
    return int(rng.random() < p)


# ───────────────────────── ground truth ─────────────────────────

def truth_pertask(true_ell, ell):
    out = []
    for lk, lc in zip(true_ell, ell):
        out.append("pass" if lk >= lc + DELTA else
                   "fail" if lk <= lc - DELTA else "gray")
    return out


def truth_aggregate(true_ell, ell):
    m = float(np.mean(true_ell - ell))
    return "pass" if m >= DELTA else "fail" if m <= -DELTA else "gray", m


# ───────────────────────── candidate sets ─────────────────────────

def load_real():
    tabs = {t: pd.read_csv(FITS / t / "raters.csv").set_index("canonical_name")
            for t in IIIC}
    common = sorted(set.intersection(*[set(df.index) for df in tabs.values()]))
    Theta = np.zeros((len(common), 2 * len(IIIC)))
    expert = []
    for i, nm in enumerate(common):
        for k, t in enumerate(IIIC):
            Theta[i, 2 * k] = float(tabs[t].loc[nm, "t_mean"])
            Theta[i, 2 * k + 1] = float(tabs[t].loc[nm, "ell_mean"])
        expert.append(str(tabs["seizure"].loc[nm, "expertise_level"]))
    return common, Theta, expert


# ───────────────────────── driver ─────────────────────────

def run_block(label, Theta, ids, Sig_c, Sig_i, ell, pools, extra=None):
    rows = []
    K = len(ell)
    for ci, theta in enumerate(Theta):
        tl = theta[1::2].copy()
        gt_pt = truth_pertask(tl, ell)
        gt_agg, m_true = truth_aggregate(tl, ell)
        # concordance = spread of the candidate's per-task margins (low spread
        # = uniformly skilled = the regime where the shared factor helps most).
        concord_sd = float(np.std(tl - ell))
        for cond, Sig in (("correlated", Sig_c), ("independent", Sig_i)):
            for sd in SEEDS:
                rng = np.random.default_rng(abs(hash((ids[ci], cond, sd))) % (2**32))
                npt, dec = run_pertask(theta, Sig, ell, pools, rng)
                rng2 = np.random.default_rng(abs(hash((ids[ci], cond, sd, "agg"))) % (2**32))
                tot_a, dec_a, npt_a = run_aggregate(theta, Sig, ell, pools, rng2)
                ncorr = sum(1 for k in range(K)
                            if gt_pt[k] in ("pass", "fail") and dec[k] == gt_pt[k])
                nwrong = sum(1 for k in range(K)
                             if gt_pt[k] in ("pass", "fail")
                             and dec[k] in ("pass", "fail") and dec[k] != gt_pt[k])
                # aggregate calibration: correct / wrong / over-confident-on-gray
                agg_corr = int(gt_agg in ("pass", "fail") and dec_a == gt_agg)
                agg_wrong = int(gt_agg in ("pass", "fail")
                                and dec_a in ("pass", "fail") and dec_a != gt_agg)
                agg_over = int(gt_agg == "gray" and dec_a in ("pass", "fail"))
                row = dict(block=label, cand=ids[ci], cond=cond, seed=sd,
                           pt_total=int(npt.sum()),
                           pt_resolved=sum(d in ("pass", "fail") for d in dec),
                           pt_correct=ncorr, pt_wrong=nwrong,
                           agg_total=int(tot_a), agg_decision=dec_a,
                           agg_truth=gt_agg, agg_correct=agg_corr,
                           agg_wrong=agg_wrong, agg_over=agg_over,
                           m_true=m_true, concord_sd=concord_sd)
                if extra:
                    row.update(extra(ci))
                rows.append(row)
    return pd.DataFrame(rows)


def paired(df, total_col):
    piv = df.groupby(["cand", "cond"])[total_col].mean().unstack("cond").dropna()
    sav = piv["independent"] - piv["correlated"]
    return dict(median_corr=float(piv["correlated"].median()),
                median_indep=float(piv["independent"].median()),
                median_saving=float(sav.median()),
                mean_saving=float(sav.mean()),
                pct_saving=float(100 * sav.median() / max(piv["independent"].median(), 1e-9)),
                frac_faster=float((sav > 0).mean()))


def acc(df, c_col, w_col):
    c, w = df[c_col].sum(), df[w_col].sum()
    return float(c / max(c + w, 1))


def main():
    Sigma6, ell6, pools = load_iiic()
    K = len(ell6)
    Sind = block_diagonal(Sigma6, K)
    lidx = [1, 3, 5, 7, 9, 11]
    rL = Sigma6[np.ix_(lidx, lidx)]
    rmean = float(rL[~np.eye(K, dtype=bool)].mean())
    print(f"K={K} IIIC; prior r_ℓ(cross-task)≈{rmean:.3f}; "
          f"ℓ*={dict(zip(IIIC, np.round(ell6,2)))}")
    print(f"rule: STOP_P={STOP_P}, δ={DELTA}, N_min/task={N_MIN_PER_TASK}, "
          f"N_max/task={N_MAX_PER_TASK}, pool/task={POOL_PER_TASK}, seeds={SEEDS}\n")

    # CONC: concordant mechanism check — all tasks share a common true ℓ
    conc_levels = np.round(np.linspace(-1.0, 2.0, 13), 3)
    Theta_c = np.zeros((len(conc_levels), 2 * K))
    for i, L in enumerate(conc_levels):
        Theta_c[i, 1::2] = L
    ids_c = [f"L={L:+.2f}" for L in conc_levels]
    df_conc = run_block("conc", Theta_c, ids_c, Sigma6, Sind, ell6, pools,
                        extra=lambda i: dict(level=conc_levels[i]))
    df_conc.to_csv(OUT / "results_conc.csv", index=False)
    print("CONC done")

    # REAL raters
    names, Theta_r, expert = load_real()
    print(f"REAL raters across all 6 IIIC tasks: {len(names)}")
    df_real = run_block("real", Theta_r, names, Sigma6, Sind, ell6, pools,
                        extra=lambda i: dict(expertise=expert[i]))
    df_real.to_csv(OUT / "results_real.csv", index=False)
    print("REAL done")

    # POP draws
    rng = np.random.default_rng(RNG_POP)
    Theta_p = rng.multivariate_normal(np.zeros(2 * K), Sigma6, size=POP_N)
    ids_p = [f"pop{i}" for i in range(POP_N)]
    df_pop = run_block("pop", Theta_p, ids_p, Sigma6, Sind, ell6, pools)
    df_pop.to_csv(OUT / "results_pop.csv", index=False)
    print("POP done")

    summary = {
        "config": dict(stop_p=STOP_P, delta=DELTA, n_min_per_task=N_MIN_PER_TASK,
                       n_max_per_task=N_MAX_PER_TASK, pool=POOL_PER_TASK,
                       r_ell_prior=rmean, seeds=list(SEEDS)),
        "real": {
            "per_task": paired(df_real, "pt_total"),
            "aggregate": paired(df_real, "agg_total"),
            "per_task_acc": {"correlated": acc(df_real[df_real.cond == "correlated"], "pt_correct", "pt_wrong"),
                             "independent": acc(df_real[df_real.cond == "independent"], "pt_correct", "pt_wrong")},
            "agg_acc": {"correlated": acc(df_real[df_real.cond == "correlated"], "agg_correct", "agg_wrong"),
                        "independent": acc(df_real[df_real.cond == "independent"], "agg_correct", "agg_wrong")},
        },
        "pop": {
            "per_task": paired(df_pop, "pt_total"),
            "aggregate": paired(df_pop, "agg_total"),
            "agg_acc": {"correlated": acc(df_pop[df_pop.cond == "correlated"], "agg_correct", "agg_wrong"),
                        "independent": acc(df_pop[df_pop.cond == "independent"], "agg_correct", "agg_wrong")},
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
