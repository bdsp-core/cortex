"""W1 — real observation-model fit: the M0–M4 link ladder (SAP §2).

Marginal per-domain frame. Hierarchical: per-user (ℓ_u, θ_u) ~ N(μ, diag σ²),
marginalized by 2-D Gauss–Hermite quadrature (81 nodes); global parameters fit
by maximum marginal likelihood on the CASE-split fit half. Model selection by
fit-half AIC; ONLY the selected model's held-out ΔELPD vs M0 is reported
(P1/P1b) — no selection on the held-out half.

Ladder (each adds ONE deviation from the engine's locked likelihood):
  M0  λ=0.025 fixed (the shipped model verbatim)
  M1  free symmetric λ
  M2  asymmetric (λ_fa, λ_miss)
  M3  M2 with slope-matched Student-t link (ν free)
  M4  M2 with scaled probit smearing Φ(a(s+θ)/√(1+γ²a²s_sd²)) — nests M2 (γ=0)

Frames: task1 (binary, K=1 — the engine frame exactly) and task2 marginal
one-vs-rest per domain (K=6). Signals: LOO primary, shipped sensitivity.

Run:  python3 -m studies.study_extset_link [--smoke]
Out:  figures/data_extset_link.npz
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist

from training.extset_adapter import load_task1, load_task2, case_split

GH_N = 9                       # nodes per latent dim (81 joint)
SD_BOX = (0.05, 3.0)           # σ_ℓ, σ_θ bounds
LAM_HI = 0.49                  # lapse box (λ_fa + λ_miss < 0.98)
N_BOOT = 2000
SEED = 20260612

MODELS = ("M0", "M1", "M2", "M3", "M4")
N_PARAMS = {"M0": 4, "M1": 5, "M2": 6, "M3": 7, "M4": 7}
X0_EXTRA = {"M0": [], "M1": [-3.0], "M2": [-3.0, -3.0],
            "M3": [-3.0, -3.0, np.log(6.0)], "M4": [-3.0, -3.0, -1.0]}


def _gh_grid():
    """2-D Gauss–Hermite nodes/weights for a standard-normal pair."""
    x, w = np.polynomial.hermite_e.hermegauss(GH_N)
    L, T = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w).ravel()
    return L.ravel(), T.ravel(), W / W.sum()


_GH_L, _GH_T, _GH_W = _gh_grid()


def _sig(x):
    return LAM_HI / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _unpack(model, x):
    p = {"mu_l": x[0], "sd_l": x[1], "mu_t": x[2], "sd_t": x[3]}
    if model == "M0":
        p["lam_fa"] = p["lam_miss"] = 0.025
    elif model == "M1":
        p["lam_fa"] = p["lam_miss"] = _sig(x[4])
    else:
        p["lam_fa"], p["lam_miss"] = _sig(x[4]), _sig(x[5])
    if model == "M3":
        p["nu"] = 1.5 + np.exp(x[6])
    if model == "M4":
        p["gamma"] = np.exp(x[6])
    return p


def link_prob(model, p, s, s_sd, ell, theta):
    """P(y=1 | s) — reads on axis 0, GH nodes on axis 1."""
    a = np.exp(ell)
    z = a * (s + theta)
    if model == "M4":
        z = z / np.sqrt(1.0 + (p["gamma"] * a * s_sd) ** 2)
    if model == "M3":
        k = norm.pdf(0) / tdist.pdf(0, p["nu"])      # slope-match at 0
        base = tdist.cdf(k * z, p["nu"])
    else:
        base = norm.cdf(z)
    return p["lam_fa"] + (1.0 - p["lam_fa"] - p["lam_miss"]) * base


class Frame:
    """One (frame, signal-variant) dataset, vectorized over users via
    reduceat on user-sorted read arrays."""

    def __init__(self, user, s, s_sd, y, fit_mask):
        order = np.argsort(user, kind="stable")
        user, self.fitm = user[order], fit_mask[order]
        self.s, self.sd, self.y = s[order], s_sd[order], y[order].astype(float)
        self.uids = np.unique(user)
        uidx = np.searchsorted(self.uids, user)
        # per-half user-sorted views + reduceat starts (users may be absent
        # from a half; counts give the alignment)
        self.half = {}
        for name, m in (("fit", self.fitm), ("held", ~self.fitm)):
            cnt = np.bincount(uidx[m], minlength=len(self.uids))
            self.half[name] = dict(
                s=self.s[m], sd=self.sd[m], y=self.y[m],
                starts=np.r_[0, np.cumsum(cnt)[:-1]], cnt=cnt)

    def _user_ll(self, model, p, h):
        """(n_users, 81) summed read log-liks per user for one half."""
        ell = p["mu_l"] + p["sd_l"] * _GH_L
        theta = p["mu_t"] + p["sd_t"] * _GH_T
        pr = link_prob(model, p, h["s"][:, None], h["sd"][:, None],
                       ell[None, :], theta[None, :])
        pr = np.clip(pr, 1e-9, 1 - 1e-9)
        ll = (np.log(pr) * h["y"][:, None]
              + np.log1p(-pr) * (1.0 - h["y"][:, None]))
        out = np.add.reduceat(ll, h["starts"], axis=0)
        out[h["cnt"] == 0] = 0.0        # reduceat artifact on empty users
        return out

    def nll(self, model, p):
        lw = np.log(_GH_W)[None, :] + self._user_ll(model, p, self.half["fit"])
        mx = lw.max(axis=1, keepdims=True)
        return -(mx[:, 0] + np.log(np.exp(lw - mx).sum(axis=1))).sum()

    def fit_model(self, model, x0=None):
        if x0 is None:
            x0 = np.r_[[0.0, 0.8, 0.0, 0.8], X0_EXTRA[model]]
        def obj(x):
            if not (SD_BOX[0] < x[1] < SD_BOX[1]
                    and SD_BOX[0] < x[3] < SD_BOX[1]):
                return 1e12
            return self.nll(model, _unpack(model, x))
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 3000, "fatol": 1e-2, "xatol": 1e-4})
        return r.x, r.fun

    def heldout_elpd(self, model, x):
        """Per-user held-out summed log-prob under the fit-half GH posterior.
        Returns (lp_user, n_user)."""
        p = _unpack(model, x)
        lw = np.log(_GH_W)[None, :] + self._user_ll(model, p, self.half["fit"])
        post = np.exp(lw - lw.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)         # (n_users, 81)

        h = self.half["held"]
        ell = p["mu_l"] + p["sd_l"] * _GH_L
        theta = p["mu_t"] + p["sd_t"] * _GH_T
        pr = link_prob(model, p, h["s"][:, None], h["sd"][:, None],
                       ell[None, :], theta[None, :])
        pr = np.clip(pr, 1e-9, 1 - 1e-9)
        post_per_read = np.repeat(post, h["cnt"], axis=0)
        mix = (pr * post_per_read).sum(axis=1)
        lp = np.log(mix) * h["y"] + np.log1p(-mix) * (1.0 - h["y"])
        lp_user = np.add.reduceat(lp, h["starts"])
        lp_user[h["cnt"] == 0] = 0.0
        return lp_user, h["cnt"]


def boot_delta(lpA, lpB, n, seed=SEED):
    """User-cluster bootstrap CI for the per-read ELPD difference A−B."""
    rng = np.random.default_rng(seed)
    keep = n > 0
    lpA, lpB, n = lpA[keep], lpB[keep], n[keep]
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(n), len(n))
        diffs.append((lpA[idx] - lpB[idx]).sum() / n[idx].sum())
    return ((lpA - lpB).sum() / n.sum(),) + tuple(
        np.percentile(diffs, [2.5, 97.5]))


def run_frame(name, user, s, s_sd, y, seg, gold_for_split, smoke):
    fit_mask = case_split(seg, gold_for_split)
    if smoke:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(np.unique(user),
                              min(60, len(np.unique(user))), replace=False))
        sel = np.isin(user, list(keep))
        user, s, s_sd, y, fit_mask = (a[sel] for a in
                                      (user, s, s_sd, y, fit_mask))
    fr = Frame(user, s, s_sd, y, fit_mask)
    res, x_warm = {}, None
    for model in MODELS:
        t0 = time.time()
        # two starts: cold default + previous member's optimum extended (the
        # ladder nests at the start point, so fit-NLL is ~monotone down it)
        starts = [None]
        if x_warm is not None:
            warm = (np.r_[x_warm[:min(len(x_warm), N_PARAMS[model])],
                          X0_EXTRA[model][len(x_warm) - 4:]]
                    if N_PARAMS[model] >= len(x_warm)
                    else x_warm[:N_PARAMS[model]])
            starts = [warm] if smoke else [None, warm]
        x, nll = min((fr.fit_model(model, x0) for x0 in starts),
                     key=lambda r: r[1])
        if model in ("M1", "M2"):
            x_warm = x
        res[model] = {"x": x, "nll": nll, "aic": 2 * nll + 2 * N_PARAMS[model]}
        print(f"  [{name}] {model}: NLL {nll:.0f} AIC {res[model]['aic']:.0f}"
              f"  params {np.round(x, 3)}  ({time.time()-t0:.0f}s)",
              flush=True)
    # selection on FIT half only; held-out evaluated for selected + M0
    best = min((m for m in MODELS if m != "M0"),
               key=lambda m: res[m]["aic"])
    lp0, n0 = fr.heldout_elpd("M0", res["M0"]["x"])
    lpb, _ = fr.heldout_elpd(best, res[best]["x"])
    pt, lo, hi = boot_delta(lpb, lp0, n0)
    print(f"  [{name}] selected {best} (fit-AIC); held-out ΔELPD/read vs M0 "
          f"= {pt:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
    return res, (best, pt, lo, hi)


def _frames(smoke):
    """(key, args) for every (frame, signal-variant) job."""
    t1 = load_task1()
    for sigvar in ("loo",) if smoke else ("loo", "shipped"):
        s = t1["s_loo"] if sigvar == "loo" else t1["s_shipped"]
        yield (f"t1_{sigvar}", (t1["user"], s, t1["s_sd"], t1["y"],
                                t1["seg"], t1["gold"]))
    t2 = load_task2()
    hit = t2["gold"] >= 0
    for k in (0, 3) if smoke else range(6):
        for sigvar in ("loo",) if smoke else ("loo", "shipped"):
            s = t2["s_loo"] if sigvar == "loo" else t2["s_shipped"]
            yield (f"t2_d{k+2}_{sigvar}",
                   (t2["user"][hit], s[hit, k], t2["s_sd"][hit, k],
                    (t2["label"] == k).astype(int)[hit], t2["seg"][hit],
                    t2["gold"][hit]))


def _job(item):
    key, args, smoke = item
    return key, run_frame(key, *args, smoke)


def main(smoke=False):
    import multiprocessing as mp
    jobs = [(k, a, smoke) for k, a in _frames(smoke)]
    fits, summary = {}, {}
    if smoke:
        results = [_job(j) for j in jobs]
    else:
        with mp.Pool(min(len(jobs), max(os.cpu_count() - 4, 1))) as pool:
            results = pool.map(_job, jobs)
    for key, (res, summ) in results:
        fits[key], summary[key] = res, summ

    if not smoke:
        np.savez("figures/data_extset_link.npz",
                 summary=np.array([(k,) + tuple(v) for k, v in
                                   summary.items()], dtype=object),
                 **{f"{k}_{m}_x": r["x"] for k, res in fits.items()
                    for m, r in res.items()},
                 **{f"{k}_{m}_nll": r["nll"] for k, res in fits.items()
                    for m, r in res.items()})
        print("saved figures/data_extset_link.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
