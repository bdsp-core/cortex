"""F4.1 diagnostic: per-step fidelity of coarse-to-fine item selection,
decoupled from the chaotic session cascade.

Runs a *full-grid* session (the ground-truth trajectory).  At every
post-update state we recompute, on the SAME state, the item chosen by:
  - full grid (exhaustive argmin, ground truth)
  - single-bracket coarse-to-fine (current shipped design)
  - top-M-bracket coarse-to-fine (refine around the M lowest coarse cells)
and record the per-step (k, s) disagreement rate and the relative
expected-loss gap of the disagreeing picks.

This isolates the *approximation* error from the butterfly-effect
trajectory divergence (a single different pick reshapes the posterior
and decorrelates everything downstream — so per-trajectory n_q equality
is unachievable for any approximate selector and is the WRONG criterion;
the right question is the per-step disagreement rate and its EV cost).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core_mcmc as cm  # noqa: E402
from tests.conftest import true_params_array  # noqa: E402


def c2f_topM(loss_of, n, n_coarse, M):
    """Coarse-to-fine with refinement around the M lowest coarse cells."""
    full = np.arange(n)
    if n_coarse is None or n <= int(n_coarse):
        return int(full[int(np.argmin(loss_of(full)))])
    nc = int(n_coarse)
    coarse = np.unique(np.round(np.linspace(0, n - 1, nc)).astype(int))
    Lc = loss_of(coarse)
    h = int(np.ceil(n / nc))
    order = np.argsort(Lc)[:max(1, M)]
    wins = [coarse]
    for j in order:
        c = int(coarse[j])
        wins.append(np.arange(max(0, c - h), min(n - 1, c + h) + 1))
    cand = np.unique(np.concatenate(wins))
    return int(cand[int(np.argmin(loss_of(cand)))])


def choose_full(state, banks):
    best_loss, bk, bs = np.inf, 0, 0.0
    for k in range(len(banks)):
        L = cm._expected_loss_vec(state, k, banks[k])
        i = int(np.argmin(L))
        if L[i] < best_loss:
            best_loss, bk, bs = L[i], k, float(banks[k][i])
    return bk, bs, best_loss


def choose_c2f(state, banks, n_coarse, M):
    best_loss, bk, bs = np.inf, 0, 0.0
    for k in range(len(banks)):
        sigs = banks[k]
        idx = c2f_topM(lambda ii: cm._expected_loss_vec(state, k, sigs[ii]),
                       len(sigs), n_coarse, M)
        L = float(cm._expected_loss_vec(state, k, sigs[idx:idx + 1])[0])
        if L < best_loss:
            best_loss, bk, bs = L, k, float(sigs[idx])
    return bk, bs, best_loss


def main():
    K = 6
    obj = cm.load_fitted_Sigma(os.path.join(
        os.path.dirname(cm.__file__), "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    banks = [np.linspace(-2.0, 2.0, 250) for _ in range(K)]

    designs = [("single M=1 nc=40", 40, 1),
               ("topM M=3  nc=40", 40, 3),
               ("topM M=3  nc=24", 24, 3),
               ("topM M=5  nc=32", 32, 5)]
    agg = {d[0]: {"steps": 0, "disagree": 0, "max_gap": 0.0,
                  "sum_gap": 0.0, "n_eval": 0} for d in designs}

    for seed in range(6):
        rng = np.random.default_rng(seed)
        l = np.random.default_rng(100 + seed).normal(0.4, 0.3, K)
        t = np.random.default_rng(200 + seed).normal(0.0, 0.5, K)
        tp = true_params_array(t, l)
        state = cm.make_state_hier(500, K, 0.378, rng,
                                   Sigma_l=Corr_l, Sigma_t=Corr_l)
        for q in range(350):
            bk, bs, full_loss = choose_full(state, banks)
            for name, nc, M in designs:
                ck, cs, c_loss = choose_c2f(state, banks, nc, M)
                a = agg[name]
                a["steps"] += 1
                if (ck, cs) != (bk, bs):
                    a["disagree"] += 1
                    gap = (c_loss - full_loss) / abs(full_loss)
                    a["max_gap"] = max(a["max_gap"], gap)
                    a["sum_gap"] += gap
                # candidate-count proxy for speedup
                ncoarse = min(len(banks[0]), nc)
                a["n_eval"] += ncoarse + M * (2 * int(np.ceil(250 / nc)) + 1)
            # advance the ground-truth trajectory
            tk = tp[bk * 2]; tl = tp[bk * 2 + 1]
            y = cm.simulate_response(bs, tk, tl, rng)
            cm.update(state, bk, bs, y)
            if cm.ess(state["w"]) < 0.5 * 500:
                cm.resample_and_rejuvenate(state, rng, 15, 0.5)

    print(f"{'design':<18} {'disagree%':>10} {'mean_gap':>10} "
          f"{'max_gap':>10} {'~speedup':>10}")
    for name, *_ in designs:
        a = agg[name]
        dr = 100.0 * a["disagree"] / a["steps"]
        mg = (a["sum_gap"] / a["disagree"]) if a["disagree"] else 0.0
        sp = (250.0 * K * a["steps"]) / a["n_eval"]
        print(f"{name:<18} {dr:>9.2f}% {mg:>10.2e} "
              f"{a['max_gap']:>10.2e} {sp:>9.1f}x")


if __name__ == "__main__":
    main()
