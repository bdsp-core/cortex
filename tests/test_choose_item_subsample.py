"""F4.1 — choose_item coarse-to-fine optimization (OC-campaign enabler).

Pure coarse subsampling is near-optimal *per question* but its small
errors compound over a session (empirically +43% n_q at n_coarse=40 on a
250-item bank), which would bias the OC/simulation n_q estimates.  The
shipped optimization is instead a deterministic **coarse-to-fine argmin**:
a coarse evenly-spaced grid brackets the minimum, then a local
full-resolution refine recovers the exact argmin on the smooth, unimodal
expected-loss-vs-difficulty surface.

Pins:
  - _coarse_to_fine_argmin: returns the exact full-grid argmin when
    n_coarse is None or ≥ n; recovers the exact argmin on a smooth
    unimodal surface; deterministic (no RNG); brute_k mirror identical.
  - Subsampled choose_item picks the *same* item as the full grid (exact
    recovery — not merely "near-optimal").
  - End-to-end: n_q-to-δ with n_subsample≈40 is EXACTLY identical
    per-seed (no bias) to the full-grid run (bitwise-identical
    trajectory — top-3 bracketing has 0.00% per-step disagreement).
  - Isolated choose_item speedup ≥3× on a representative (≥600-item)
    bank (the algorithmic claim; the end-to-end per-question gain is
    Amdahl-diluted by the rest of the SMC pipeline and is bank-size
    dependent — ≈1.9× at 250 items, ≈7.5× at the ~1705-item
    production bank, scripts/diag_choose_item_speed.py).
  - Default (n_subsample=None) is byte-identical to pre-F4.1 behaviour.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import core_mcmc
import core_mcmc_brute_k
from tests.conftest import true_params_array


# ── _coarse_to_fine_argmin contract ───────────────────────────────────

def test_coarse_to_fine_contract():
    c2f = core_mcmc._coarse_to_fine_argmin

    # Smooth unimodal surface with the minimum off the coarse grid.
    n = 250
    true_i = 173
    loss = (np.arange(n) - true_i) ** 2.0
    loss_of = lambda ii: loss[ii]

    # None or n_coarse ≥ n → exact full-grid argmin.
    assert c2f(loss_of, n, None) == true_i
    assert c2f(loss_of, n, n + 10) == true_i
    # Top-M coarse-to-fine recovers the EXACT argmin even though true_i
    # is off the coarse grid (a refine window brackets and resolves it).
    assert c2f(loss_of, n, 40) == true_i
    assert c2f(loss_of, n, 24) == true_i
    # Deterministic.
    assert c2f(loss_of, n, 40) == c2f(loss_of, n, 40)
    # brute_k mirror is identical.
    assert core_mcmc_brute_k._coarse_to_fine_argmin(loss_of, n, 40) == true_i
    # Asymmetric surface, minimum near an endpoint (window clamps).
    loss2 = np.abs(np.arange(n) - 4.0)
    assert c2f(lambda ii: loss2[ii], n, 32) == 4

    # Bimodal surface constructed RELATIVE to the actual coarse nodes so
    # the single-bracket miss is guaranteed:
    #   • a shallow decoy whose minimum sits exactly ON a coarse node
    #     (so it is the global coarse argmin),
    #   • a narrow deep TRUE basin centred a few indices off a *different*
    #     coarse node, with that node on its shoulder (2nd-lowest coarse
    #     value) and the basin floor strictly between coarse nodes.
    # h = ceil(n/nc): single-bracket refines only the decoy window →
    # never reaches the true basin; top-M (default 3) also refines the
    # shoulder node, whose ±h window covers the true minimum.
    nc = 24
    coarse = np.unique(np.round(np.linspace(0, n - 1, nc)).astype(int))
    h = int(np.ceil(n / nc))
    decoy_i = int(coarse[5])
    shoulder = int(coarse[15])
    true2_i = shoulder + (h // 2 + 2)   # off-grid, well inside ±h of shoulder
    assert true2_i - decoy_i > 3 * h    # decoy window cannot reach it
    xx = np.arange(n)
    # Steep/narrow decoy: only its exact node is low among coarse nodes,
    # so the next-lowest coarse values are the TRUE basin's shoulders
    # (→ they land in the top-3 and get refined by default M=3).
    decoy = 5.00 * np.abs(xx - decoy_i) + 1.00                 # steep decoy, min 1.00
    spike = 1.50 * np.abs(xx - true2_i) + 0.02                 # narrow deep true, min 0.02
    bim = np.minimum(decoy, spike)
    assert int(coarse[int(np.argmin(bim[coarse]))]) == decoy_i  # decoy wins coarsely
    assert c2f(lambda ii: bim[ii], n, nc, m_bracket=1) != true2_i  # single MISSES
    assert c2f(lambda ii: bim[ii], n, nc) == true2_i               # default M=3 recovers
    assert core_mcmc_brute_k._coarse_to_fine_argmin(
        lambda ii: bim[ii], n, nc) == true2_i


# ── exact recovery of the full-grid item ──────────────────────────────

def _make_state(K=6, N=400, seed=0):
    rng = np.random.default_rng(seed)
    obj = core_mcmc.load_fitted_Sigma(
        __import__("os").path.join(
            __import__("os").path.dirname(core_mcmc.__file__),
            "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    st = core_mcmc.make_state_hier(N, K, r_assumed=0.378, rng=rng,
                                   Sigma_l=Corr_l, Sigma_t=Corr_l)
    for (k, s, y) in [(0, 1.0, 1), (3, -0.5, 0), (1, 0.7, 1), (4, -1.2, 0)]:
        core_mcmc.update(st, k, s, y)
    return st


def test_subsampled_choice_matches_full_grid():
    """coarse-to-fine picks the *same* (k, s) as the exhaustive full-grid
    argmin across a battery of states — exact recovery, because the EV
    surface over the difficulty-stratified bank is smooth and unimodal."""
    banks = [np.linspace(-2.0, 2.0, 250) for _ in range(6)]
    for seed in range(8):
        st = _make_state(seed=seed)
        # Exhaustive full-grid optimum (the pre-F4.1 logic).
        best_loss, best_k, best_s = np.inf, 0, 0.0
        for k in range(6):
            L = core_mcmc._expected_loss_vec(st, k, banks[k])
            i = int(np.argmin(L))
            if L[i] < best_loss:
                best_loss, best_k, best_s = L[i], k, float(banks[k][i])
        k_s, s_s = core_mcmc.choose_item(st, banks, n_subsample=40)
        assert (k_s, s_s) == (best_k, best_s), (
            f"seed={seed}: coarse-to-fine ({k_s},{s_s}) != "
            f"full-grid ({best_k},{best_s})")


def test_subsample_none_is_unchanged():
    """n_subsample=None must reproduce the exact pre-F4.1 selection."""
    banks = [np.linspace(-2.0, 2.0, 137 + k) for k in range(6)]
    st = _make_state(seed=3)
    best_loss, best_k, best_s = np.inf, 0, 0.0
    for k in range(6):
        L = core_mcmc._expected_loss_vec(st, k, banks[k])
        i = int(np.argmin(L))
        if L[i] < best_loss:
            best_loss, best_k, best_s = L[i], k, float(banks[k][i])
    k2, s2 = core_mcmc.choose_item(st, banks, n_subsample=None)
    assert (k2, s2) == (best_k, best_s)


# ── determinism ───────────────────────────────────────────────────────

def test_subsampled_session_is_deterministic():
    K = 3
    tp = true_params_array(*(np.random.default_rng(1).normal(0.4, 0.3, K),
                             np.random.default_rng(2).normal(0.0, 0.5, K)))
    kw = dict(method="hier", true_params=tp, K=K, r_assumed=0.378,
              max_q=120, delta_auroc=0.05, N=300, seed=7,
              ess_threshold_frac=0.5, n_subsample=40)
    a = core_mcmc.run_session_mcmc_auroc(**kw)
    b = core_mcmc.run_session_mcmc_auroc(**kw)
    np.testing.assert_array_equal(a["final_lo"], b["final_lo"])
    np.testing.assert_array_equal(a["final_hi"], b["final_hi"])
    assert a["n_questions"] == b["n_questions"]


# ── end-to-end equivalence (no n_q bias) + speedup ────────────────────

@pytest.mark.slow
def test_subsample_endtoend_exact_equivalence():
    """Full-grid vs coarse-to-fine n_subsample=40: n_q-to-δ is EXACTLY
    identical per seed.  Top-3 bracketing has 0.00% per-step disagreement
    with the exhaustive argmin, so the whole adaptive trajectory is
    bitwise identical — there is no chaotic cascade and therefore zero
    n_q / AUROC bias in the OC/simulation campaign."""
    K = 6
    obj = core_mcmc.load_fitted_Sigma(
        __import__("os").path.join(
            __import__("os").path.dirname(core_mcmc.__file__),
            "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    banks = [np.linspace(-2.0, 2.0, 250) for _ in range(K)]

    def run(n_sub, seed):
        l = np.random.default_rng(100 + seed).normal(0.4, 0.3, K)
        t = np.random.default_rng(200 + seed).normal(0.0, 0.5, K)
        tp = true_params_array(t, l)
        return core_mcmc.run_session_mcmc_auroc(
            method="hier", true_params=tp, K=K, r_assumed=0.378,
            Sigma_l=Corr_l, Sigma_t=Corr_l, bank_signals=banks,
            max_q=600, delta_auroc=0.05, N=500, seed=seed,
            ess_threshold_frac=0.5, n_subsample=n_sub)

    nq_full, nq_sub = [], []
    for seed in range(4):
        nq_full.append(run(None, seed)["n_questions"])
        nq_sub.append(run(40, seed)["n_questions"])
    assert nq_sub == nq_full, (
        f"n_q must be identical per seed (exact-argmin recovery): "
        f"full={nq_full} sub={nq_sub}")


@pytest.mark.slow
def test_choose_item_speedup_on_representative_bank():
    """The *algorithmic* F4.1 claim: isolated choose_item is ≥3× faster
    with n_subsample=40 (top-3 coarse-to-fine) than the full grid on a
    representative ≥600-item bank.  (Smaller banks see less gain — numpy
    fixed costs dominate — and the per-question end-to-end gain is
    further Amdahl-diluted by the rest of the SMC pipeline; the headline
    speedup is a property of choose_item itself and grows with bank
    size: ≈4× at 600, ≈7.5× at the ~1705-item production bank.)"""
    K = 6
    nbank = 600
    obj = core_mcmc.load_fitted_Sigma(
        __import__("os").path.join(
            __import__("os").path.dirname(core_mcmc.__file__),
            "Sigma_l_fitted.npy"))
    Corr_l = np.asarray(obj["Corr_l"], dtype=float)
    banks = [np.linspace(-2.0, 2.0, nbank) for _ in range(K)]

    # Battery of realistic mid-session states.
    states = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        l = np.random.default_rng(100 + seed).normal(0.4, 0.3, K)
        t = np.random.default_rng(200 + seed).normal(0.0, 0.5, K)
        tp = true_params_array(t, l)
        st = core_mcmc.make_state_hier(500, K, 0.378, rng,
                                       Sigma_l=Corr_l, Sigma_t=Corr_l)
        for _ in range(40):
            kk, ss = core_mcmc.choose_item(st, banks)
            y = core_mcmc.simulate_response(ss, tp[kk * 2],
                                            tp[kk * 2 + 1], rng)
            core_mcmc.update(st, kk, ss, y)
        states.append(st)

    reps = 3
    t0 = time.time()
    for _ in range(reps):
        for st in states:
            core_mcmc.choose_item(st, banks, n_subsample=None)
    t_full = time.time() - t0
    t0 = time.time()
    for _ in range(reps):
        for st in states:
            core_mcmc.choose_item(st, banks, n_subsample=40)
    t_sub = time.time() - t0
    speedup = t_full / t_sub
    assert speedup >= 3.0, (
        f"choose_item speedup {speedup:.2f}× below 3× target on a "
        f"{nbank}-item bank (t_full={t_full:.2f}s t_sub={t_sub:.2f}s)")
