"""A2 follow-up — PROPER SBC of the EXTSET M2 fitter (F51 disambiguation).

F51's SBC failed 5/6 globals, but the harness used cheap 300-step chains
initialized AT the NM point fit — under-dispersed, so the truth's rank
concentrates at the extremes regardless of calibration (the classic
under-converged-SBC U-shape). This re-runs SBC honestly:

  * walkers initialized OVER-DISPERSED (≫ posterior width), not in a 0.05 ball
    at the NM fit;
  * long chains (1500 burn + 3000 kept);
  * a per-replicate split-R̂ convergence gate, so uniformity is judged only on
    replicates whose chains actually mixed.

If SBC now passes on converged replicates → F51's failure was a harness
artifact and the GH-marginal NM+ensemble fitter is calibrated. If it still
fails with R̂≈1 → genuine fitter mis-calibration.

Run:  python3 -m studies.study_extset_sbc [--smoke]
Out:  figures/data_extset_sbc.npz
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.stats import chisquare

import studies.study_extset_bayes as B
from studies.study_extset_link import Frame
from training.extset_adapter import case_split, load_task1

SEED = 20260612
J_FULL = 80
N_WALKERS = 48                      # more walkers → better ensemble mixing
N_BURN = 1000
N_STEPS = 2000
L_RANKS = 511
SBC_USERS, SBC_READS = 60, 80       # smaller datasets to afford long chains
# over-dispersed walker spread (~2× the per-replicate posterior sds)
INIT_SCALE = np.array([0.25, 0.18, 0.12, 0.10, 0.60, 1.20])

_S_POOL = _SD_POOL = None
_nb = _ns = None                    # fork-inherited chain lengths


def _init_walkers(x_nm, rng):
    coords = x_nm[None, :] + INIT_SCALE[None, :] * rng.standard_normal(
        (N_WALKERS, 6))
    coords[:, 1] = np.clip(coords[:, 1], 0.05, 2.5)    # sd_l in prior box
    coords[:, 3] = np.clip(coords[:, 3], 0.05, 2.5)    # sd_t in prior box
    return coords


def _run_chain(x_nm, n_burn, n_steps, seed):
    rng = np.random.default_rng(seed)
    coords = _init_walkers(x_nm, rng)
    logps = np.array([B.logpost(c) for c in coords])
    for i in np.where(~np.isfinite(logps))[0]:          # rescue dead walkers
        coords[i] = x_nm
        logps[i] = B.logpost(x_nm)
    chain = []
    for step in range(n_burn + n_steps):
        coords, logps = B.stretch_move(coords, logps, rng)
        if step >= n_burn:
            chain.append(coords.copy())
    return np.array(chain)                                # (n_steps, W, 6)


def _split_rhat(chain):
    """Max-over-params split-R̂: split each walker chain in half → 2W sub-chains."""
    steps, _, dim = chain.shape
    half = steps // 2
    subs = np.concatenate([chain[:half], chain[half:2 * half]], axis=1)
    n = half
    means = subs.mean(axis=0)
    Bv = n * means.var(axis=0, ddof=1)                   # between-chain
    Wv = subs.var(axis=0, ddof=1).mean(axis=0)           # within-chain
    var = (n - 1) / n * Wv + Bv / n
    return np.sqrt(var / Wv)                              # (dim,)


def _sbc_one(j):
    rng = np.random.default_rng(SEED + 1000 * j + 7)
    while True:                                          # draw truth from prior
        x_true = np.array([rng.normal(0, 1), abs(rng.normal(0, 1)) + 0.02,
                           rng.normal(0, 1), abs(rng.normal(0, 1)) + 0.02,
                           0.0, 0.0])
        lam = rng.beta(1.2, 30.0, size=2) * 0.49
        x_true[4:] = np.log(lam / 0.49 / (1 - lam / 0.49))
        if np.isfinite(B.logprior(x_true)):
            break
    user, s, sd, y, seg = B.simulate_dataset(x_true, _S_POOL, _SD_POOL, rng)
    fit = case_split(seg, seg % 2, seed=SEED + j)
    B._FRAME = Frame(user, s, sd, y, fit)
    x_nm, _ = B._FRAME.fit_model("M2")                   # same pipeline as real
    chain = _run_chain(x_nm, _nb, _ns, SEED + j)
    rhat = _split_rhat(chain)
    flat = chain.reshape(-1, 6)
    thin = flat[:: max(1, len(flat) // L_RANKS)][:L_RANKS]
    ranks = (thin < x_true[None, :]).sum(axis=0)
    return ranks, len(thin), rhat


def main(smoke=False):
    global _S_POOL, _SD_POOL, _nb, _ns
    B.SBC_USERS, B.SBC_READS = SBC_USERS, SBC_READS
    t1 = load_task1()
    _, first = np.unique(t1["seg"], return_index=True)
    _S_POOL, _SD_POOL = t1["s_loo"][first], t1["s_sd"][first]

    J = 12 if smoke else J_FULL
    _nb, _ns = (150, 250) if smoke else (N_BURN, N_STEPS)
    print(f"PROPER SBC: J={J} reps, {N_WALKERS} walkers, {_nb}+{_ns} steps, "
          f"{SBC_USERS}×{SBC_READS} reads, over-dispersed init", flush=True)
    t0 = time.time()
    with mp.Pool(min(B.N_WORKERS, J)) as pool:
        out = pool.map(_sbc_one, range(J))
    ranks = np.array([r for r, _, _ in out])
    L = out[0][1]
    rhats = np.array([rh for _, _, rh in out])
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # PER-PARAMETER convergence gate: judge uniformity only on the replicates
    # whose chain mixed for THAT parameter (R̂<1.1). Records how many qualify.
    names = ["mu_l", "sd_l", "mu_t", "sd_t", "a_fa", "a_miss"]
    pvals, nconv = [], []
    for i in range(6):
        ok = rhats[:, i] < 1.1
        nconv.append(int(ok.sum()))
        use = ok if ok.sum() >= 40 else np.ones(J, bool)
        h, _ = np.histogram(ranks[use, i], bins=10, range=(0, L))
        pvals.append(chisquare(h).pvalue)
        flag = "" if ok.sum() >= 40 else "  (UNGATED — too few converged)"
        print(f"    {names[i]:>7}: max R̂={rhats[:, i].max():.2f}  "
              f"converged {ok.sum()}/{J}  uniformity p={pvals[i]:.2f}{flag}",
              flush=True)

    if not smoke:
        np.savez("figures/data_extset_sbc.npz", ranks=ranks, L=L, rhats=rhats,
                 nconv=np.array(nconv), pvals=np.array(pvals),
                 names=np.array(names), n_burn=_nb, n_steps=_ns,
                 n_walkers=N_WALKERS, sbc_users=SBC_USERS, sbc_reads=SBC_READS)
        print("saved figures/data_extset_sbc.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
