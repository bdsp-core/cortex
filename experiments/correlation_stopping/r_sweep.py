"""How strong would the cross-task correlation have to be for 'exploit the
correlation to stop earlier' to actually pay off?

For each r in a sweep, build a clean exchangeable prior Σ(r) (unit variances;
ℓ-block correlation r; t-block independent), draw a population whose TRUE skills
have exactly that correlation, and compare the aggregate SPRT under the
correlated prior Σ(r) vs the block-diagonal (isolation) prior. Reports median
questions-to-aggregate-decision and accuracy vs r.

The fitted deployment prior is r≈0.37 — this locates it on the curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "engine", ROOT, ROOT / "deployment"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "exp", Path(__file__).resolve().parent / "run_experiment.py")
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

OUT = exp.OUT
R_VALUES = [0.0, 0.2, 0.37, 0.55, 0.75, 0.9]
N_CAND = 250
SWEEP_SEED = 4242


def sigma_of_r(ell_block_r, K):
    """Clean exchangeable prior: each task's (t,ℓ) is unit-variance; t's are
    cross-independent; ℓ's have pairwise correlation `ell_block_r`."""
    D = 2 * K
    S = np.eye(D)
    for a in range(K):
        for b in range(K):
            if a != b:
                S[2 * a + 1, 2 * b + 1] = ell_block_r
    return S


def main():
    _, ell6, pools = exp.load_iiic()
    K = len(ell6)
    rng = np.random.default_rng(SWEEP_SEED)
    rows = []
    for r in R_VALUES:
        Sig_c = sigma_of_r(r, K)
        Sig_i = exp.block_diagonal(Sig_c, K)
        # true population correlated at exactly r (matches the engine prior)
        Theta = rng.multivariate_normal(np.zeros(2 * K), Sig_c, size=N_CAND)
        for cond, Sig in (("correlated", Sig_c), ("independent", Sig_i)):
            tot, correct, wrong, n = [], 0, 0, 0
            for ci, theta in enumerate(Theta):
                tl = theta[1::2]
                gt, _ = exp.truth_aggregate(tl, ell6)
                rr = np.random.default_rng(abs(hash((r, cond, ci))) % (2**32))
                q, dec, _ = exp.run_aggregate(theta, Sig, ell6, pools, rr)
                tot.append(q)
                if gt in ("pass", "fail"):
                    n += 1
                    if dec == gt:
                        correct += 1
                    elif dec in ("pass", "fail"):
                        wrong += 1
            rows.append(dict(r=r, cond=cond, median_q=float(np.median(tot)),
                             mean_q=float(np.mean(tot)),
                             acc=float(correct / max(correct + wrong, 1))))
        print(f"r={r}: done")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_rsweep.csv", index=False)
    print(df.to_string(index=False))

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = df.pivot(index="r", columns="cond", values="median_q")
    acc = df.pivot(index="r", columns="cond", values="acc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(piv.index, piv["independent"], "-o", color="#d95f0e", label="independent")
    ax.plot(piv.index, piv["correlated"], "-o", color="#2b8cbe", label="correlated")
    ax.axvline(0.37, color="0.5", ls="--", lw=1)
    ax.text(0.37, ax.get_ylim()[1] * 0.95, " fitted r_ℓ=0.37", fontsize=8, va="top")
    ax.set_xlabel("cross-task ℓ correlation r (prior = truth)")
    ax.set_ylabel("median questions to aggregate decision")
    ax.set_title("Speed: correlation only pays off when r is large")
    ax.legend()
    ax = axes[1]
    ax.plot(acc.index, acc["independent"], "-o", color="#d95f0e", label="independent")
    ax.plot(acc.index, acc["correlated"], "-o", color="#2b8cbe", label="correlated")
    ax.axvline(0.37, color="0.5", ls="--", lw=1)
    ax.set_xlabel("cross-task ℓ correlation r")
    ax.set_ylabel("aggregate decision accuracy")
    ax.set_title("Accuracy (well-specified: prior = truth)")
    ax.legend()
    fig.suptitle("Correlation-strength sweep — how strong must r be to stop "
                 "earlier on the aggregate?", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig6_r_sweep.png", bbox_inches="tight")
    print("wrote", OUT / "fig6_r_sweep.png")


if __name__ == "__main__":
    main()
