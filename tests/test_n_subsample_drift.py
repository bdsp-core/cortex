"""Scope-guard for the engine's coarse-to-fine item selection (``n_subsample``),
established for the v2 paper_sims Tier-0 CPU speedup (2026-06-04).

FINDING (verified here): ``core_mcmc.choose_item(n_subsample=40)`` is
BIT-IDENTICAL to full-grid selection on the PLAIN expected-loss surface
(m_bracket=3 → 0.00% disagreement, matching core_mcmc.py:529-536), BUT NOT on
the s_sd-PROPAGATED surface — there the per-candidate s_sd attenuation makes the
loss-vs-bank-index surface non-smooth, so m_bracket=3 misses ~13% of picks and
bit-identity is only recovered at m_bracket≈16 (which, with windows covering most
of the bank, yields no useful speedup).

CONSEQUENCE (pinned by this test): the fast path is safe ONLY for sims that do
NOT propagate s_sd — i.e. SIM2 (Mode-A efficiency, bank_sds=None) and the SBC
CONTROL/PLUGIN arms. It is UNSAFE for the live AD6 / UNCERT path (CortexSession
always passes bank_sds), which therefore stays full-grid; the GPU surrogate is
the speed lever there. Do NOT enable n_subsample on any bank_sds path expecting
bit-identity.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _p in (_REPO, os.path.join(_REPO, "engine"),
           os.path.join(_REPO, "engine", "variants")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import core_mcmc as cm  # noqa: E402

K = 7
N = 300
NSUB = 40
BANK = [np.linspace(-2.5, 2.5, 250) for _ in range(K)]


def _disagreements(with_sds: bool, m_bracket: int) -> tuple:
    """Per-step (k, s) disagreement between full-grid and coarse-to-fine
    (n_subsample=40, given m_bracket) over a real evolving K=7 cloud."""
    rng_sds = np.random.default_rng(7)
    sds = ([np.abs(rng_sds.normal(0.3, 0.15, len(BANK[k]))) for k in range(K)]
           if with_sds else None)
    steps = disagree = 0
    for seed in range(2):
        rng = np.random.default_rng(seed)
        l = np.random.default_rng(100 + seed).normal(0.4, 0.3, K)
        t = np.random.default_rng(200 + seed).normal(0.0, 0.5, K)
        tp = np.empty(2 * K)
        tp[0::2], tp[1::2] = t, l
        state = cm.make_state_hier(N, K, 0.378, rng)
        for _q in range(120):
            # full-grid argmin (ground truth)
            bl, bk, bs = np.inf, 0, 0.0
            for k in range(K):
                sd_k = sds[k] if sds is not None else None
                L = cm._expected_loss_vec(state, k, BANK[k], signal_sds=sd_k)
                i = int(np.argmin(L))
                if L[i] < bl:
                    bl, bk, bs = L[i], k, float(BANK[k][i])
            # coarse-to-fine with the given bracket
            cl, ck, cs = np.inf, 0, 0.0
            for k in range(K):
                sd_k = sds[k] if sds is not None else None
                idx = cm._coarse_to_fine_argmin(
                    lambda ii: cm._expected_loss_vec(
                        state, k, BANK[k][ii],
                        signal_sds=(sd_k[ii] if sd_k is not None else None)),
                    250, NSUB, m_bracket=m_bracket)
                Lv = float(cm._expected_loss_vec(
                    state, k, BANK[k][idx:idx + 1],
                    signal_sds=(sd_k[idx:idx + 1] if sd_k is not None else None))[0])
                if Lv < cl:
                    cl, ck, cs = Lv, k, float(BANK[k][idx])
            steps += 1
            if (bk, bs) != (ck, cs):
                disagree += 1
            y = cm.simulate_response(bs, tp[bk * 2], tp[bk * 2 + 1], rng)
            cm.update(state, bk, bs, y)
            if cm.ess(state["w"]) < 0.5 * N:
                cm.resample_and_rejuvenate(state, rng, 15, 0.5)
    return steps, disagree


def test_plain_surface_bit_identical_at_default_bracket():
    """Tier-0 usable path: plain EV surface, m_bracket=3 → 0.00% disagreement."""
    steps, disagree = _disagreements(with_sds=False, m_bracket=3)
    assert disagree == 0, f"{disagree}/{steps} plain-surface disagreements"


def test_propagated_surface_not_bit_identical_at_default_bracket():
    """Negative result (pinned): on the s_sd surface m_bracket=3 is NOT
    bit-identical — n_subsample must not be used on a bank_sds path expecting
    bit-identity (the live AD6/UNCERT path stays full-grid)."""
    steps, disagree = _disagreements(with_sds=True, m_bracket=3)
    assert disagree > 0, (
        "UNEXPECTED: s_sd surface became bit-identical at m_bracket=3 — "
        "re-evaluate whether n_subsample is now safe on the propagated path")


def test_propagated_surface_recovers_at_high_bracket():
    """Bit-identity on the s_sd surface is only recovered at m_bracket=16
    (documented as speedup-negative — windows cover most of the bank)."""
    steps, disagree = _disagreements(with_sds=True, m_bracket=16)
    assert disagree == 0, f"{disagree}/{steps} s_sd disagreements at m_bracket=16"
