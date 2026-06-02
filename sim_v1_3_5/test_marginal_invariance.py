"""T1 — dynamic proof that AD6Policy's stop statistic is a PURE function of the
per-domain marginals (no cross-domain / joint structure).

Method: build a particle state {w, L:(N,K)}; compute AD6's pi/mcse/R/verdicts.
Then independently ROW-PERMUTE every column j>=1 of L. This preserves each
marginal column L[:,k] and the weights w EXACTLY, while destroying ALL joint
(cross-column) structure. If AD6 reads only marginals, pi/mcse/R/verdicts must
be BITWISE identical. Any difference would prove the formula uses joint info.
"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from cortex_policy import AD6Policy  # noqa: E402

K, N = 7, 600
rng = np.random.default_rng(20260601)
# A CORRELATED skill cloud (mean 1.0 so some domains clearly PASS), so joint
# structure is real and the verdict path is exercised.
A = rng.normal(size=(K, K))
Sig = A @ A.T + np.eye(K)
L = rng.multivariate_normal(np.full(K, 1.0), Sig, size=N)    # (N,K), correlated
# UNIFORM weights: a column permutation then preserves each domain's WEIGHTED
# marginal EXACTLY (multiset-only) while scrambling cross-domain joint structure.
# (With non-uniform weights, permuting a column would break its own w<->ell
# pairing — that is a flaw, not a property of AD6.)
w = np.full(N, 1.0 / N)

ell_star = np.full(K, 0.2)
var_prior = np.diag(Sig).copy()
n_per_task = rng.integers(20, 40, size=K).tolist()           # all past N_MIN

def run(L_in):
    # R_star=0 so the info gate opens and the PASS/FAIL verdict path is
    # actually exercised (not left at PENDING) — proves verdicts are joint-
    # invariant too, not just the pi/mcse/R statistics.
    p = AD6Policy(list(ell_star), list(var_prior), n_min=20, R_star=0.0,
                  alpha=0.25, Z=2.0)
    p.reset(K)
    dec = p({"w": w.copy(), "l": L_in.copy()}, {}, list(n_per_task), K)
    d = dec.diagnostics
    return (np.array(d["pi"]), np.array(d["mcse"]), np.array(d["R"]),
            list(d["verdicts"]))

pi0, mcse0, R0, v0 = run(L)

# permute every column j>=1 independently -> marginals identical, joint destroyed
Lp = L.copy()
for j in range(1, K):
    Lp[:, j] = L[rng.permutation(N), j]
# sanity: each marginal column is a permutation of the original (same multiset)
assert all(np.array_equal(np.sort(L[:, k]), np.sort(Lp[:, k])) for k in range(K))
# and the JOINT correlation is genuinely changed (so the test has teeth)
corr_before = np.corrcoef(L.T)[0, 1]
corr_after = np.corrcoef(Lp.T)[0, 1]

pi1, mcse1, R1, v1 = run(Lp)

dpi = float(np.max(np.abs(pi0 - pi1)))
dmcse = float(np.max(np.abs(mcse0 - mcse1)))
dR = float(np.max(np.abs(R0 - R1)))
verd_eq = (v0 == v1)

print("=== T1: AD6 marginal-only invariance proof ===")
print(f"joint corr(col0,col1) before={corr_before:+.3f}  after-permute={corr_after:+.3f}"
      f"  (joint structure genuinely changed)")
print(f"max|Δpi|   = {dpi:.2e}")
print(f"max|Δmcse| = {dmcse:.2e}")
print(f"max|ΔR|    = {dR:.2e}")
print(f"verdicts identical = {verd_eq}  ({v0})")
# tolerance 1e-12: a permuted weighted sum reorders, so pi carries ~1e-16
# float-summation noise; anything above 1e-12 would be real joint leakage.
ok = (dpi < 1e-12 and dmcse < 1e-12 and dR < 1e-12 and verd_eq)
print(f"\nRESULT: {'PASS — stop statistic is a pure function of per-domain marginals (joint structure scrambled, output unchanged to float precision)' if ok else 'FAIL — joint structure leaked into the stop statistic'}")
sys.exit(0 if ok else 1)
