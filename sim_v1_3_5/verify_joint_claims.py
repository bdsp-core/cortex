"""Independent verification of the workflow's three load-bearing claims for the
joint-verdict-vector stopping rule NO-GO. Self-contained (no engine import)."""
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn
from scipy.optimize import brentq

# ===== CLAIM 1: live K=7 ell-block correlation value =====
S = np.load("Sigma_l_fitted_k7.npy", allow_pickle=True).item()
C = np.asarray(S["Corr_l"], float)
K = C.shape[0]
off = C[~np.eye(K, dtype=bool)]
print("=== CLAIM 1: live K=7 Corr_l (ell-block) ===")
print(f"shape={C.shape}  off-diag mean={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}")
print(f"spike(row0) vs IIIC: {np.round(C[0,1:],3)}")
print("AGENT CLAIMED: mean 0.135, min 0.010, max 0.291, spike~independent")
S6 = np.load("Sigma_l_fitted.npy", allow_pickle=True).item()
C6 = np.asarray(S6["Corr_l"], float); o6 = C6[~np.eye(C6.shape[0], dtype=bool)]
print(f"legacy K={C6.shape[0]} file off-diag mean={o6.mean():.4f} (agent says this is the stale 0.25)")


def orthant_p(m, r, K=7):
    cov = (1 - r) * np.eye(K) + r * np.ones((K, K))
    # P(all X_k > -m) = P(all X_k <= m) by symmetry of mean-0 Gaussian
    return mvn(mean=np.zeros(K), cov=cov, allow_singular=True).cdf(np.full(K, m))


def margin_for_95(r, K=7):
    return brentq(lambda m: orthant_p(m, r, K) - 0.95, 0.5, 6.0)


print("\n=== CLAIM 2: orthant margin & question ratio (questions ~ margin^2) ===")
m_indep = norm.ppf(0.95 ** (1 / 7))
print(f"brute (product) per-domain margin for joint 0.95 = {m_indep:.4f}  (agent: 2.4421)")
for r in (0.0, 0.135, 0.37, 0.78):
    m = margin_for_95(r)
    ratio = (m_indep / m) ** 2
    print(f"  r={r:.3f}: hier margin={m:.4f}  question ratio brute/hier={ratio:.3f}x")
print("AGENT CLAIMED: r=0.135->1.010x, r=0.37->1.047x, r=0.78->1.271x")


def post_corr(prior_r, info, K=7):
    Sig0 = (1 - prior_r) * np.eye(K) + prior_r * np.ones((K, K))
    Prec = np.linalg.inv(Sig0) + info * np.eye(K)
    Sp = np.linalg.inv(Prec)
    d = np.sqrt(np.diag(Sp))
    Cp = Sp / np.outer(d, d)
    return Cp[~np.eye(K, dtype=bool)].mean()


print("\n=== CLAIM 3: posterior corr decay as per-domain info accrues ===")
for prior_r in (0.135, 0.37, 0.78):
    row = [f"info={I}:{post_corr(prior_r, I):.3f}" for I in (0, 1, 3, 5, 6, 10)]
    print(f"  prior r={prior_r}:  " + "  ".join(row))
print("AGENT CLAIMED: at info~5-6, prior 0.78 -> post ~0.10; prior 0.135 -> ~0.02")
