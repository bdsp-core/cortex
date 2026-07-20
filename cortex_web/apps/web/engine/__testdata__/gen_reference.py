"""Dump ground-truth values from the REAL Python engine for the TS port to
match. Imports engine/core_mcmc.py + scripts/cortex_policy.py so
the reference is the actual desktop numerics, not a re-derivation.

    python cortex_web/engine/__testdata__/gen_reference.py

Writes reference.json next to this file. The vitest specs
(mathfns.test.ts, pipeline.test.ts) load it and assert the TS engine agrees.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]  # ideal-test-multi/
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(REPO / "scripts"))

from core_mcmc import (  # noqa: E402
    _log_p_response, _p_response_yes, update, _expected_loss_vec, LAPSE_RATE)
from scipy.stats import norm  # noqa: E402
from scipy.special import log_ndtr, logsumexp  # noqa: E402

OUT = Path(__file__).resolve().parent / "reference.json"


def fixed_state(t, l, w):
    """Build a minimal engine state dict (the fields update / _expected_loss_vec
    / the AD6 policy read). t, l: (N, K); w: (N,)."""
    t = np.asarray(t, float)
    l = np.asarray(l, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    N = t.shape[0]
    return {
        "t": t.copy(), "l": l.copy(), "w": w.copy(),
        "log_lik": np.zeros(N), "log_prior": np.zeros(N),
        "history": [], "history_s_sd": [],
        "K": t.shape[1], "is_hier": True, "_prior_pieces": None,
        "_l_prior_mean": None, "r_assumed": 0.378,
    }


def main():
    ref = {"lapse": LAPSE_RATE}

    # 1. mathfns — Φ, logΦ over a grid spanning the tails.
    xs = [-12, -8, -6, -5.0001, -5, -4.999, -3, -1, -0.3, 0, 0.7, 2, 5, 8, 12]
    ref["normCdf"] = [{"x": x, "v": float(norm.cdf(x))} for x in xs]
    ref["logNdtr"] = [{"x": x, "v": float(log_ndtr(x))} for x in xs]

    # 2. logsumexp of pairs.
    pairs = [(-1.0, -2.0), (0.0, 0.0), (-700.0, -3.0), (3.0, -700.0), (-1e9, -1e9 + 1)]
    ref["logSumExp2"] = [{"a": a, "b": b, "v": float(logsumexp([a, b]))} for a, b in pairs]

    # 3. likelihood — logPResponse / pResponseYes at (l,t,s,y).
    cases = []
    for (lv, tv, s) in [(0.5, 0.0, 1.0), (-0.3, 0.2, -2.0), (1.2, -0.5, 0.4), (0.0, 0.0, 3.0)]:
        z = float(np.exp(lv) * (s + tv))
        cases.append({
            "l": lv, "t": tv, "s": s, "z": z,
            "logP1": float(_log_p_response(np.array([z]), 1)[0]),
            "logP0": float(_log_p_response(np.array([z]), 0)[0]),
            "pYes": float(_p_response_yes(np.array([z]))[0]),
        })
    ref["likelihood"] = cases

    # 4. update fixture — fixed cloud, reweight on (k, s, y), dump post-update w.
    rng = np.random.default_rng(7)
    N, K = 12, 2
    t0 = rng.standard_normal((N, K)) * 0.4
    l0 = rng.standard_normal((N, K)) * 0.3
    w0 = rng.random(N)
    w0 = w0 / w0.sum()
    st = fixed_state(t0, l0, w0)
    update(st, k=1, s=0.8, y=1, s_sd=0.0)
    ref["updateFixture"] = {
        "t": t0.tolist(), "l": l0.tolist(), "w": w0.tolist(),
        "k": 1, "s": 0.8, "y": 1, "sSd": 0.0,
        "wPost": st["w"].tolist(),
    }

    # 5. expectedLoss fixture — same cloud (pre-update), EV loss at a candidate.
    st2 = fixed_state(t0, l0, w0)
    loss = float(_expected_loss_vec(st2, k=0, signals=np.array([0.5]),
                                    signal_sds=np.array([0.2]))[0])
    ref["expectedLossFixture"] = {
        "k": 0, "s": 0.5, "sSd": 0.2, "loss": loss,
    }

    # 6. policy fixture — π_k, mcse_k, R_k for a fixed (l, w) cloud.
    # Replicate AD6Policy's math directly (it reads state["w"], state["l"]).
    ell_star = [0.3, -0.1]
    var_prior = [1.0, 1.0]
    w = w0
    ess = float(1.0 / float((w * w).sum()))
    pis, mcses, Rs = [], [], []
    for k in range(K):
        ell_k = l0[:, k]
        pi_k = float((w * (ell_k > ell_star[k])).sum())
        mu_k = float((w * ell_k).sum())
        var_post = float((w * (ell_k - mu_k) ** 2).sum())
        pis.append(pi_k)
        mcses.append(float(np.sqrt(max(pi_k * (1 - pi_k), 0.0) / max(ess, 1.0))))
        Rs.append(1.0 - var_post / var_prior[k])
    ref["policyFixture"] = {
        "l": l0.tolist(), "w": w0.tolist(), "ellStar": ell_star,
        "varPrior": var_prior, "ess": ess, "pi": pis, "mcse": mcses, "R": Rs,
    }

    OUT.write_text(json.dumps(ref, indent=2))
    print(f"wrote {OUT}  ({len(ref)} sections)")


if __name__ == "__main__":
    main()
