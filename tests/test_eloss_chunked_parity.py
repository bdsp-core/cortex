"""Parity guard for the web-deployment chunked selection kernel
(`_expected_loss_vec_chunked`, added for N=1200). It must be BIT-IDENTICAL to
the stock `_expected_loss_vec` (chunking only changes memory access, not the
per-candidate math) so the web instrument == the desktop instrument, and
`choose_item(chunk=...)` must pick the same item as the default.
"""
import numpy as np
import pytest

from core_mcmc import (_expected_loss_vec, _expected_loss_vec_chunked,
                       make_state_hier, choose_item)

K = 7


@pytest.mark.parametrize("N", [200, 1200])
@pytest.mark.parametrize("n_sig", [10, 64, 65, 533])
@pytest.mark.parametrize("with_sds,include_theta",
                         [(False, True), (True, True), (False, False)])
def test_chunked_bit_identical_to_stock(N, n_sig, with_sds, include_theta):
    rng = np.random.default_rng(N * 31 + n_sig)
    state = make_state_hier(N, K, 0.3, rng)
    signals = rng.normal(size=n_sig)
    sds = np.abs(rng.normal(size=n_sig)) + 0.05 if with_sds else None
    for k in range(K):
        stock = _expected_loss_vec(state, k, signals, signal_sds=sds,
                                   include_theta=include_theta)
        for chunk in (32, 64, 128):
            ch = _expected_loss_vec_chunked(
                state, k, signals, signal_sds=sds,
                include_theta=include_theta, chunk=chunk)
            assert np.array_equal(stock, ch), \
                f"chunked != stock (k={k}, chunk={chunk}, N={N}, n_sig={n_sig})"


def test_choose_item_chunk_same_pick():
    rng = np.random.default_rng(7)
    N = 600
    state = make_state_hier(N, K, 0.3, rng)
    bank = [rng.normal(size=300) for _ in range(K)]
    sds = [np.abs(rng.normal(size=300)) + 0.05 for _ in range(K)]
    segids = [np.arange(300) + 1000 * kk for kk in range(K)]
    default = choose_item(state, bank, bank_sds=sds, return_sd=True,
                          bank_segids=segids)
    chunked = choose_item(state, bank, bank_sds=sds, return_sd=True,
                          bank_segids=segids, chunk=64)
    assert default == chunked


def test_default_path_unchanged():
    """chunk=None (default) must be the exact stock call — frozen-engine safe."""
    rng = np.random.default_rng(3)
    state = make_state_hier(400, K, 0.3, rng)
    sig = rng.normal(size=120)
    assert np.array_equal(
        _expected_loss_vec(state, 2, sig),
        _expected_loss_vec_chunked(state, 2, sig, chunk=10_000))  # chunk>=n → one call
