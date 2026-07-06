"""A2 — Bayesian refit of the headline frame + SBC of the fitting pipeline.

(a) POSTERIOR: affine-invariant ensemble MCMC (Goodman–Weare stretch move,
    numpy-only) over the t1_loo M2 globals (μ_ℓ, σ_ℓ, μ_θ, σ_θ, λ_fa, λ_miss)
    with the same GH-marginal likelihood used for the point fits. Weakly
    informative priors: μ ~ N(0,1²), σ ~ HalfN(1), λ ~ Beta(1.2, 30).
    Walkers parallelized over the worker pool.

(b) SBC: draw J parameter sets from the prior, simulate t1-shaped datasets
    (U users × R reads each, real signal values resampled), refit by the SAME
    NM pipeline, and check rank-uniformity of the truth within each
    posterior approximation (here: NM point ± Laplace SE ranks are too
    coarse, so SBC uses the MCMC machinery on a reduced grid — J×short
    chains, parallel over the pool). Validates that the fitter recovers
    parameters it claims to recover.

Run:  python3 -m studies.study_extset_bayes [--smoke]
Out:  figures/data_extset_bayes.npz
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.stats import norm

from studies.study_extset_link import Frame, _unpack, link_prob
from training.extset_adapter import case_split, load_task1

N_WORKERS = 40
SEED = 20260612
N_WALKERS = 24
N_STEPS = 1500           # post-burnin kept steps per walker (full)
N_BURN = 500
SBC_J = 96               # SBC replicates (full)
SBC_USERS, SBC_READS = 120, 150

_FRAME = None            # fork-inherited dataset


def logprior(x):
    mu_l, sd_l, mu_t, sd_t, a_fa, a_miss = x
    if not (0.02 < sd_l < 3.0 and 0.02 < sd_t < 3.0):
        return -np.inf
    lp = -0.5 * (mu_l ** 2 + mu_t ** 2)               # N(0,1) on means
    lp += -0.5 * ((sd_l / 1.0) ** 2 + (sd_t / 1.0) ** 2)   # HalfN(1) on sds
    for a in (a_fa, a_miss):                          # Beta(1.2,30) on λ/0.49
        lam = 0.49 / (1 + np.exp(-np.clip(a, -40, 40)))
        u = np.clip(lam / 0.49, 1e-12, 1 - 1e-12)
        lp += 0.2 * np.log(u) + 29.0 * np.log1p(-u)
        lp += np.log(u * (1 - u))                     # logit Jacobian
    return lp


def loglik(x):
    p = _unpack("M2", x)
    return -_FRAME.nll("M2", p)


def logpost(x):
    lp = logprior(x)
    return lp + loglik(x) if np.isfinite(lp) else -np.inf


_EVAL_POOL = None        # optional pool for parallel proposal evaluation


def _map_logpost(props):
    if _EVAL_POOL is not None:
        return np.array(_EVAL_POOL.map(logpost, props))
    return np.array([logpost(p) for p in props])


def stretch_move(coords, logps, rng, a=2.0):
    """One Goodman–Weare ensemble sweep (two half-updates)."""
    n, dim = coords.shape
    half = n // 2
    for s0, s1 in ((slice(0, half), slice(half, n)),
                   (slice(half, n), slice(0, half))):
        sub, other = coords[s0], coords[s1]
        z = ((a - 1.0) * rng.random(len(sub)) + 1.0) ** 2 / a
        partners = other[rng.integers(0, len(other), len(sub))]
        prop = partners + z[:, None] * (sub - partners)
        logp_prop = _map_logpost(prop)
        accept = (np.log(rng.random(len(sub)))
                  < (dim - 1) * np.log(z) + logp_prop - logps[s0])
        coords[s0][accept] = prop[accept]
        logps[s0][accept] = logp_prop[accept]
    return coords, logps


def run_mcmc(x0, n_steps, n_burn, seed, log_every=None):
    rng = np.random.default_rng(seed)
    coords = x0 + 0.05 * rng.standard_normal((N_WALKERS, len(x0)))
    logps = np.array([logpost(c) for c in coords])
    chain = []
    for step in range(n_burn + n_steps):
        coords, logps = stretch_move(coords, logps, rng)
        if step >= n_burn:
            chain.append(coords.copy())
        if log_every and step % log_every == 0:
            print(f"    step {step}/{n_burn+n_steps} "
                  f"logp[max]={logps.max():.0f}", flush=True)
    return np.array(chain)            # (n_steps, walkers, dim)


def simulate_dataset(x, s_pool, sd_pool, rng):
    """t1-shaped synthetic data from the M2 model at parameters x."""
    p = _unpack("M2", x)
    user = np.repeat(np.arange(SBC_USERS), SBC_READS)
    idx = rng.integers(0, len(s_pool), len(user))
    s, sd = s_pool[idx], sd_pool[idx]
    ell = p["mu_l"] + p["sd_l"] * rng.standard_normal(SBC_USERS)
    th = p["mu_t"] + p["sd_t"] * rng.standard_normal(SBC_USERS)
    pr = link_prob("M2", p, s, sd, ell[user], th[user])
    y = (rng.random(len(user)) < pr).astype(int)
    seg = idx                                     # signal id = case id
    return user, s, sd, y, seg


def _sbc_one(j):
    """One SBC replicate: draw truth from prior, simulate, short-chain
    posterior, return rank of truth per parameter."""
    global _FRAME
    rng = np.random.default_rng(SEED + 1000 * j)
    # prior draw
    while True:
        x_true = np.array([rng.normal(0, 1), abs(rng.normal(0, 1)) + 0.02,
                           rng.normal(0, 1), abs(rng.normal(0, 1)) + 0.02,
                           0.0, 0.0])
        lam = rng.beta(1.2, 30.0, size=2) * 0.49
        x_true[4:] = np.log(lam / 0.49 / (1 - lam / 0.49))
        if np.isfinite(logprior(x_true)):
            break
    user, s, sd, y, seg = simulate_dataset(x_true, _S_POOL, _SD_POOL, rng)
    fit = case_split(seg, seg % 2, seed=SEED + j)   # dummy balanced strata
    _FRAME = Frame(user, s, sd, y, fit)
    # initialize at the NM point fit — SBC then validates the PIPELINE
    # (NM + ensemble posterior) exactly as it is used on real data
    x_nm, _ = _FRAME.fit_model("M2")
    chain = run_mcmc(x_nm, n_steps=300, n_burn=200, seed=SEED + j)
    flat = chain.reshape(-1, 6)
    thin = flat[:: max(1, len(flat) // 255)][:255]
    return (thin < x_true[None, :]).sum(axis=0), len(thin)


def main(smoke=False):
    global _FRAME, _S_POOL, _SD_POOL
    t1 = load_task1()
    fit = case_split(t1["seg"], t1["gold"])
    user, s, sd, y = t1["user"], t1["s_loo"], t1["s_sd"], t1["y"]
    _S_POOL, _SD_POOL = np.unique(t1["s_loo"]), None
    # pool of (s, sd) pairs at the seg level for simulation
    _, first = np.unique(t1["seg"], return_index=True)
    _S_POOL, _SD_POOL = s[first], sd[first]
    if smoke:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(np.unique(user), 60, replace=False))
        m = np.isin(user, list(keep))
        user, s, sd, y, fit = (a[m] for a in (user, s, sd, y, fit))
    _FRAME = Frame(user, s, sd, y, fit)

    d = np.load("figures/data_extset_link.npz", allow_pickle=True)
    x0 = d["t1_loo_M2_x"]
    n_steps, n_burn = (60, 40) if smoke else (N_STEPS, N_BURN)
    print(f"(a) posterior: {N_WALKERS} walkers × {n_burn}+{n_steps} steps",
          flush=True)
    t0 = time.time()
    global _EVAL_POOL
    if not smoke:
        _EVAL_POOL = mp.Pool(N_WALKERS // 2)     # fork inherits _FRAME
    chain = run_mcmc(x0, n_steps, n_burn, SEED, log_every=100)
    if _EVAL_POOL is not None:
        _EVAL_POOL.close()
        _EVAL_POOL = None
    flat = chain.reshape(-1, 6)
    lam = 0.49 / (1 + np.exp(-np.clip(flat[:, 4:], -40, 40)))
    names = ["mu_l", "sd_l", "mu_t", "sd_t"]
    print(f"  ({time.time()-t0:.0f}s)  posterior 95% intervals:", flush=True)
    for i, nm in enumerate(names):
        lo, md, hi = np.percentile(flat[:, i], [2.5, 50, 97.5])
        print(f"    {nm:>7}: {md:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    for i, nm in enumerate(["lam_fa", "lam_miss"]):
        lo, md, hi = np.percentile(lam[:, i], [2.5, 50, 97.5])
        print(f"    {nm:>7}: {md:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    pr_engine = float((lam.max(axis=1) < 0.025).mean())
    print(f"  P(both λ < engine's 0.025 | data) = {pr_engine:.3f}",
          flush=True)

    print("(b) SBC of the fitter", flush=True)
    J = 8 if smoke else SBC_J
    with mp.Pool(min(N_WORKERS, J)) as pool:
        out = pool.map(_sbc_one, range(J))
    ranks = np.array([r for r, _ in out])
    L = out[0][1]
    # uniformity: chi-square on rank-histogram deciles per parameter
    from scipy.stats import chisquare
    pvals = []
    for i in range(6):
        h, _ = np.histogram(ranks[:, i], bins=10, range=(0, L))
        pvals.append(chisquare(h).pvalue)
    print("  SBC rank-uniformity p-values:",
          [f"{p:.2f}" for p in pvals], flush=True)

    if not smoke:
        np.savez("figures/data_extset_bayes.npz", chain=flat, lam=lam,
                 pr_engine=pr_engine, sbc_ranks=ranks, sbc_L=L,
                 sbc_pvals=np.array(pvals))
        print("saved figures/data_extset_bayes.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
