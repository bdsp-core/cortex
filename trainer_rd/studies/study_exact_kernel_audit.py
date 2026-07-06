"""Math-audit reproduction study (2026-07-01, docs/AUDIT_POMDP_MATH.md MA-1).

The shipped soft-rule propagate marginalizes the stimulus realization s_real
INDEPENDENTLY in the likelihood and the transition, but the learner's response
y and its state update share the SAME s_real draw. The exact one-step kernel
conditions the transition on the observed y:

    mean:      E[p(s_real) | y, θ]        (vs shipped E[p(s_real) | θ])
    variance:  q_t² + α_t²·Var[p(s_real) | y, θ]   (vs shipped q_t²)

and analogously E[w|y], Var[w|y] for the skill channel. Moments use the
21-node Gauss–Hermite rule `training_filter._GHE_*` (the conditional
integrand is harder than F30's unconditional bump: 11 nodes leave up to
2.2e-2 moment error at s_sd/σ ∈ [1,2]; 21 nodes ≤ 2e-3 — see the M15 node
justification in training_filter.py), with node posterior
∝ GH_weight × P(y | s_node, θ). At s_sd=0 every correction vanishes and the
kernel is bit-identical to shipped. `TaskFilter(exact_kernel=True)` is the
production implementation and reproduces the cond+var arm exactly.

Design: OPEN-LOOP paired comparison — the item stream is oracle-placed at the
learner's TRUE state, so all filter arms see the identical (s, y, y*, s_sd)
sequence and differences are pure filter quality. Arms: shipped, smear_w
(F30 hardening), cond (exact mean), cond+var (exact mean + variance).

Headline (30 seeds × 300 trials, N=1000, s_sd=0.85, eval-seeded prior;
21-node rule — the original 11-node audit run differed only in MC noise):

    arm       RMSE(t)  RMSE(ell)  cov_t@90  cov_l@90  width_t
    shipped    0.2074    0.0887     0.631     0.771     0.386
    smear_w    0.2073    0.0815     0.630     0.893     0.386
    cond       0.2035    0.0815     0.755     0.909     0.486
    cond+var   0.1989    0.0815     0.906     0.913     0.673  ← nominal,
                                                          better RMSE
The width_t column shows the mechanism: shipped intervals are ~40% too
narrow for the error they carry (overconfidence, not noise).

Run:  python3 -m studies.study_exact_kernel_audit [--smoke]
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import norm

from training.bridge_conventions import LAPSE_RATE, SKILL_MODE_MULTIPLIER
from training.learner_sim import Learner, LearnerParams
from training.training_filter import TaskFilter, _GHE_X, _GHE_WN

M = SKILL_MODE_MULTIPLIER


class CondTaskFilter(TaskFilter):
    """Soft-rule propagate with the exact conditional kernel MEAN:
    δ = E[p(s_real)|y,θ] − y*, skill weight E[w(s_real)|y,θ]."""

    _with_var = False

    def propagate(self, s, y_star, feedback=True, *, y=None, s_sd=0.0):
        if self.p.rule != "soft" or y is None or not s_sd:
            return super().propagate(s, y_star, feedback, y=y, s_sd=s_sd)
        s = float(s)
        sigma = np.exp(-self.ell)
        t = -self.theta
        nodes = s + np.sqrt(2.0) * float(s_sd) * _GHE_X            # (11,)
        zz = (nodes[:, None] - t[None, :]) / sigma[None, :]       # (11,N)
        p_node = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(zz)
        like = p_node if y == 1 else (1.0 - p_node)
        qw = _GHE_WN[:, None] * like                # node posterior given y
        Z = qw.sum(axis=0)
        Z = np.where(Z > 0, Z, 1.0)
        Ep = (qw * p_node).sum(axis=0) / Z
        d = np.abs(nodes[:, None] - t[None, :]) / sigma[None, :]
        w_node = np.exp(-((d - M) ** 2) / (2.0 * self.p.rho ** 2))
        Ew = (qw * w_node).sum(axis=0) / Z
        if self._with_var:
            Vp = np.maximum((qw * p_node ** 2).sum(axis=0) / Z - Ep ** 2, 0.0)
            Vw = np.maximum((qw * w_node ** 2).sum(axis=0) / Z - Ew ** 2, 0.0)
        else:
            Vp = Vw = 0.0
        delta = Ep - y_star
        sd_t = np.sqrt(self.p.q_t ** 2 + (self.p.alpha_t ** 2) * Vp)
        t_new = (t + self.learn * self.p.alpha_t * delta
                 + sd_t * self.rng.standard_normal(self.N))
        f = 1.0 if feedback else 0.0
        log_sig = np.log(sigma)
        gap = log_sig - np.log(self.p.sigma_inf)
        g_sigma = -f * self.learn * Ew * gap
        sd_s = np.sqrt(self.p.q_sigma ** 2
                       + (self.p.alpha_sigma * gap) ** 2 * Vw)
        log_sig_new = (log_sig + self.p.alpha_sigma * g_sigma
                       + sd_s * self.rng.standard_normal(self.N))
        self.theta = -t_new
        self.ell = -log_sig_new


class CondVarTaskFilter(CondTaskFilter):
    """Exact conditional kernel mean AND variance."""
    _with_var = True


def _wq(vals, w, q):
    o = np.argsort(vals)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(q, cw, vals[o])


ARMS = {
    # explicit exact_kernel=False: these arms are the PRE-M17 kernel record
    # (M17/D26 flipped the TaskFilter default to exact)
    "shipped": lambda th, el, tp, sd: TaskFilter(th, el, tp, seed=sd,
                                                 exact_kernel=False),
    "smear_w": lambda th, el, tp, sd: TaskFilter(th, el, tp, seed=sd,
                                                 smear_w=True,
                                                 exact_kernel=False),
    "cond": lambda th, el, tp, sd: CondTaskFilter(th, el, tp, seed=sd),
    "cond+var": lambda th, el, tp, sd: CondVarTaskFilter(th, el, tp, seed=sd),
}


def run_seed(seed, *, n_trials=300, N=1000, ssd=0.85):
    """One paired run: identical learner + stream for every filter arm."""
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.48,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    lnr = Learner([1.5], [0.8], tp, seed=seed)
    stream_rng = np.random.default_rng(10_000 + seed)
    prior_rng = np.random.default_rng(20_000 + seed)
    # eval-seeded prior: centered on truth, SD 0.25 (D1-like handoff)
    th0 = -0.8 + 0.25 * prior_rng.standard_normal(N)
    el0 = -np.log(1.5) + 0.25 * prior_rng.standard_normal(N)
    filters = {a: mk(th0, el0, tp, seed + 1) for a, mk in ARMS.items()}
    err = {a: {"t": [], "l": [], "ct": [], "cl": [], "wt": []} for a in ARMS}
    side = 1
    for _ in range(n_trials):
        s = side * M * lnr.sigma[0]           # oracle placement → common stream
        side *= -1
        y_star = int(s > 0)
        s_real = s + ssd * stream_rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        true_t = float(lnr.t[0])
        true_l = float(-np.log(lnr.sigma[0]))
        for a, f in filters.items():
            f.step(s, y, y_star, s_sd=ssd, feedback=True)
            mt, ml = f.mean()
            err[a]["t"].append((-mt) - true_t)
            err[a]["l"].append(ml - true_l)
            lo, hi = _wq(-f.theta, f.w, 0.05), _wq(-f.theta, f.w, 0.95)
            err[a]["ct"].append(lo <= true_t <= hi)
            err[a]["wt"].append(hi - lo)
            err[a]["cl"].append(_wq(f.ell, f.w, 0.05) <= true_l
                                <= _wq(f.ell, f.w, 0.95))
    return {a: (float(np.sqrt(np.mean(np.square(e["t"])))),
                float(np.sqrt(np.mean(np.square(e["l"])))),
                float(np.mean(e["ct"])), float(np.mean(e["cl"])),
                float(np.mean(e["wt"])))
            for a, e in err.items()}


def main(n_seeds=30, **kw):
    agg = {a: [] for a in ARMS}
    for sd in range(n_seeds):
        for a, v in run_seed(sd, **kw).items():
            agg[a].append(v)
    print(f"{'arm':>9} {'RMSE(t)':>8} {'RMSE(ell)':>9} {'cov_t@90':>9} "
          f"{'cov_l@90':>9} {'width_t':>8}")
    for a in ARMS:
        v = np.array(agg[a])
        print(f"{a:>9} {v[:, 0].mean():8.4f} {v[:, 1].mean():9.4f} "
              f"{v[:, 2].mean():9.3f} {v[:, 3].mean():9.3f} "
              f"{v[:, 4].mean():8.3f}")
    base = np.array(agg["shipped"])
    n = np.sqrt(n_seeds)
    for a in ("smear_w", "cond", "cond+var"):
        v = np.array(agg[a])
        print(f"paired {a}−shipped: ΔRMSE(t) {(v[:,0]-base[:,0]).mean():+.4f}"
              f"±{(v[:,0]-base[:,0]).std()/n:.4f}  "
              f"ΔRMSE(ℓ) {(v[:,1]-base[:,1]).mean():+.4f}"
              f"±{(v[:,1]-base[:,1]).std()/n:.4f}  "
              f"Δcov_t {(v[:,2]-base[:,2]).mean():+.3f}"
              f"±{(v[:,2]-base[:,2]).std()/n:.3f}")
    return agg


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    main(n_seeds=(5 if smoke else 30),
         n_trials=(150 if smoke else 300), N=(500 if smoke else 1000))
