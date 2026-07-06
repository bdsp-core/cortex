"""A1/M5 — joint forced-choice (softmax) fit of the task2 reads (SAP §2.2 M5).

The six marginal one-vs-rest fits needed large class-specific miss-lapses
(0.06–0.34, F43). M5 tests the structural explanation: a single softmax over
all six domain signals with ONE per-user skill scalar —

    P(label=j | reads i, user u) = (1−λ)·softmax_j(e^{ℓ_u}·β·s_ij + c_j) + λ/6
    ℓ_u ~ N(0, σ_ℓ²)   (mean absorbed by β; 9-node GH marginalization)

Comparison (same case split, same held-out reads, label log-loss):
  COMPOSED — the 6 fitted marginal models (selected per frame, LOO signals)
  composed via P(j) ∝ p_j·Π_{k≠j}(1−p_k) with per-frame user posteriors.

Run:  python3 -m studies.study_extset_m5 [--smoke]
Out:  figures/data_extset_m5.npz
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.optimize import minimize

from studies.study_extset_link import (Frame, _unpack, boot_delta, link_prob,
                                       _GH_L, _GH_T, _GH_W)
from training.extset_adapter import case_split, load_task2

GH_N = 9
SEED = 20260612
FIT_NPZ = "figures/data_extset_link.npz"


def _gh1():
    x, w = np.polynomial.hermite_e.hermegauss(GH_N)
    return x, w / w.sum()


_G1X, _G1W = _gh1()
# 2-D grid for M5b (skill × user-lapse)
_G2L, _G2T = (g.ravel() for g in np.meshgrid(_G1X, _G1X, indexing="ij"))
_G2W = np.outer(_G1W, _G1W).ravel()


class M5:
    """Vectorized marginal-likelihood machinery for the softmax model.
    two_dim=True → M5b (per-user lapse as a second latent dimension)."""

    def __init__(self, user, S, label, fit_mask, two_dim=False):
        self.two_dim = two_dim
        self.W = _G2W if two_dim else _G1W
        order = np.argsort(user, kind="stable")
        self.S, self.label = S[order], label[order]
        self.fitm = fit_mask[order]
        user = user[order]
        self.uids = np.unique(user)
        uidx = np.searchsorted(self.uids, user)
        self.half = {}
        for name, m in (("fit", self.fitm), ("held", ~self.fitm)):
            cnt = np.bincount(uidx[m], minlength=len(self.uids))
            self.half[name] = dict(S=self.S[m], lab=self.label[m],
                                   starts=np.r_[0, np.cumsum(cnt)[:-1]],
                                   cnt=cnt)

    @staticmethod
    def _read_ll(p, S, lab):
        """(n_reads, n_nodes) log P(label) per latent node.

        M5  (1-D user): nodes = skill only, scalar λ.
        M5b (2-D user): nodes = skill × user-lapse (logit λ_u ~ N(m, s²));
        chosen when p carries 'm_lam'/'s_lam'."""
        beta = p["beta"]
        c = p["c"][None, :, None]                       # (1, 6, 1)
        if "m_lam" in p:                                # M5b: 2-D GH
            a = np.exp(p["sd_l"] * _G2L)[None, None, :]
            lam = 0.49 / (1 + np.exp(-(p["m_lam"] + p["s_lam"] * _G2T))
                          )[None, :]
        else:                                           # M5: 1-D GH
            a = np.exp(p["sd_l"] * _G1X)[None, None, :]
            lam = p["lam"]
        z = a * beta * S[:, :, None] + c                # (n, 6, nodes)
        z -= z.max(axis=1, keepdims=True)
        soft = np.exp(z)
        soft /= soft.sum(axis=1, keepdims=True)
        pj = (1 - lam) * soft[np.arange(len(lab)), lab, :] + lam / 6.0
        return np.log(np.clip(pj, 1e-12, None))         # (n, nodes)

    def _user_ll(self, p, h):
        ll = self._read_ll(p, h["S"], h["lab"])
        out = np.add.reduceat(ll, h["starts"], axis=0)
        out[h["cnt"] == 0] = 0.0
        return out

    def nll(self, p):
        lw = np.log(self.W)[None, :] + self._user_ll(p, self.half["fit"])
        mx = lw.max(axis=1, keepdims=True)
        return -(mx[:, 0] + np.log(np.exp(lw - mx).sum(axis=1))).sum()

    def fit(self):
        if self.two_dim:
            def unpack(x):
                return dict(beta=np.exp(x[0]), sd_l=abs(x[1]) + 1e-3,
                            m_lam=x[2], s_lam=abs(x[3]) + 1e-3,
                            c=np.r_[0.0, x[4:9]])
            x0 = np.r_[0.0, 0.6, -3.0, 1.0, np.zeros(5)]
        else:
            def unpack(x):
                return dict(beta=np.exp(x[0]), sd_l=abs(x[1]) + 1e-3,
                            lam=0.49 / (1 + np.exp(-np.clip(x[2], -40, 40))),
                            c=np.r_[0.0, x[3:8]])
            x0 = np.r_[0.0, 0.6, -3.0, np.zeros(5)]
        r = minimize(lambda x: self.nll(unpack(x)), x0,
                     method="Nelder-Mead",
                     options={"maxiter": 8000, "fatol": 1e-2})
        return unpack(r.x), r.fun, r.x

    def heldout(self, p):
        lw = np.log(self.W)[None, :] + self._user_ll(p, self.half["fit"])
        post = np.exp(lw - lw.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        h = self.half["held"]
        ll = np.exp(self._read_ll(p, h["S"], h["lab"]))
        mix = (ll * np.repeat(post, h["cnt"], axis=0)).sum(axis=1)
        lp = np.log(np.clip(mix, 1e-12, None))
        lp_user = np.add.reduceat(lp, h["starts"])
        lp_user[h["cnt"] == 0] = 0.0
        return lp_user, h["cnt"]


def composed_marginal_heldout(user, S, S_sd, label, fit_mask):
    """Held-out label log-prob from the 6 fitted marginal models composed via
    one-vs-rest independence, P(j) ∝ p_j·Π_{k≠j}(1−p_k) — the structure M5
    relaxes. All Frames sort reads by the same stable user key, so held-half
    rows align across frames (and with M5) by construction.

    Returns (lp_user, n_user) grouped exactly like M5.heldout."""
    d = np.load(FIT_NPZ, allow_pickle=True)
    sel = {r[0]: r[1] for r in d["summary"]}
    P = None
    for k in range(6):
        key, best = f"t2_d{k+2}_loo", sel[f"t2_d{k+2}_loo"]
        p = _unpack(best, d[f"{key}_{best}_x"])
        fr = Frame(user, S[:, k], S_sd[:, k], (label == k).astype(int),
                   fit_mask)
        lw = np.log(_GH_W)[None, :] + fr._user_ll(best, p, fr.half["fit"])
        post = np.exp(lw - lw.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        h = fr.half["held"]
        ell = p["mu_l"] + p["sd_l"] * _GH_L
        theta = p["mu_t"] + p["sd_t"] * _GH_T
        pr = np.clip(link_prob(best, p, h["s"][:, None], h["sd"][:, None],
                               ell[None, :], theta[None, :]), 1e-9, 1 - 1e-9)
        mix = (pr * np.repeat(post, h["cnt"], axis=0)).sum(axis=1)
        if P is None:
            P = np.empty((len(mix), 6))
            held_lab, starts, cnt = h["y"], h["starts"], h["cnt"]
        P[:, k] = mix
    # compose: P(label=j) ∝ p_j · Π_{k≠j}(1−p_k)
    log1m = np.log1p(-P)
    logj = np.log(P) + (log1m.sum(axis=1, keepdims=True) - log1m)
    logj -= np.log(np.exp(logj - logj.max(axis=1, keepdims=True))
                   .sum(axis=1, keepdims=True)) + logj.max(axis=1,
                                                           keepdims=True)
    order = np.argsort(user, kind="stable")
    lab_held = label[order][~fit_mask[order]]
    lp = logj[np.arange(len(lab_held)), lab_held]
    lp_user = np.add.reduceat(lp, starts)
    lp_user[cnt == 0] = 0.0
    return lp_user, cnt


def main(smoke=False):
    t2 = load_task2()
    hit = np.ones(len(t2["label"]), bool)        # gold-free likelihood
    seg = t2["seg"]
    fit_mask = case_split(seg, t2["gold"])       # stratified by gold/-1
    user, S, S_sd, label = (t2["user"], t2["s_loo"], t2["s_sd"], t2["label"])
    if smoke:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(np.unique(user), 80, replace=False))
        m = np.isin(user, list(keep))
        user, S, S_sd, label, fit_mask = (a[m] for a in
                                          (user, S, S_sd, label, fit_mask))

    results = {}
    for name, two in (("M5", False), ("M5b", True)):
        m = M5(user, S, label, fit_mask, two_dim=two)
        t0 = time.time()
        p, nll, x = m.fit()
        lam_txt = (f"λ_u~logitN({p['m_lam']:.2f},{p['s_lam']:.2f})"
                   if two else f"λ={p['lam']:.4f}")
        print(f"{name} fit: NLL {nll:.0f}  β={p['beta']:.3f} "
              f"σ_ℓ={p['sd_l']:.3f} {lam_txt}  c={np.round(p['c'], 2)}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        lp, n = m.heldout(p)
        print(f"{name} held-out label elpd/read: {lp.sum()/n.sum():.4f}",
              flush=True)
        results[name] = (p, nll, x, lp, n)

    lpc, nc = composed_marginal_heldout(user, S, S_sd, label, fit_mask)
    print(f"composed-marginal held-out elpd/read: {lpc.sum()/nc.sum():.4f}",
          flush=True)
    deltas = {}
    for name, (p, nll, x, lp, n) in results.items():
        deltas[name] = boot_delta(lp, lpc, n)
        pt, lo, hi = deltas[name]
        print(f"{name} − composed ΔELPD/read = {pt:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]", flush=True)
    pt, lo, hi = boot_delta(results["M5b"][3], results["M5"][3],
                            results["M5"][4])
    print(f"M5b − M5 ΔELPD/read = {pt:+.4f} [{lo:+.4f},{hi:+.4f}]",
          flush=True)

    if not smoke:
        np.savez("figures/data_extset_m5.npz",
                 **{f"{k}_x": v[2] for k, v in results.items()},
                 **{f"{k}_nll": v[1] for k, v in results.items()},
                 **{f"{k}_lp": v[3] for k, v in results.items()},
                 n_user=results["M5"][4], lp_composed=lpc,
                 **{f"delta_{k}": np.array(v) for k, v in deltas.items()})
        print("saved figures/data_extset_m5.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
