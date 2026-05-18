"""F4.1 diagnostic: isolated choose_item wall-clock speedup, full grid vs
top-3 coarse-to-fine, at several bank sizes.  Sets an evidence-based
acceptance threshold (the end-to-end factor is Amdahl-diluted by the rest
of the per-question pipeline, so the *algorithmic* claim must be measured
on choose_item itself)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core_mcmc as cm  # noqa: E402
from tests.conftest import true_params_array  # noqa: E402


def main():
    K = 6
    obj = cm.load_fitted_Sigma(os.path.join(
        os.path.dirname(cm.__file__), "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)

    for nbank in (250, 600, 1705):
        banks = [np.linspace(-2.0, 2.0, nbank) for _ in range(K)]
        # Build a battery of realistic mid-session states.
        states = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            l = np.random.default_rng(100 + seed).normal(0.4, 0.3, K)
            t = np.random.default_rng(200 + seed).normal(0.0, 0.5, K)
            tp = true_params_array(t, l)
            st = cm.make_state_hier(500, K, 0.378, rng,
                                    Sigma_l=Corr_l, Sigma_t=Corr_l)
            for q in range(40):
                kk, ss = cm.choose_item(st, banks)
                y = cm.simulate_response(ss, tp[kk * 2], tp[kk * 2 + 1], rng)
                cm.update(st, kk, ss, y)
            states.append(st)

        reps = 8
        t0 = time.time()
        for _ in range(reps):
            for st in states:
                cm.choose_item(st, banks, n_subsample=None)
        t_full = time.time() - t0
        t0 = time.time()
        for _ in range(reps):
            for st in states:
                cm.choose_item(st, banks, n_subsample=40)
        t_sub = time.time() - t0
        print(f"bank={nbank:>5}  full={t_full:7.3f}s  c2f={t_sub:7.3f}s  "
              f"choose_item speedup={t_full / t_sub:5.2f}x")


if __name__ == "__main__":
    main()
