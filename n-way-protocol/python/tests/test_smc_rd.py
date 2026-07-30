from __future__ import annotations

import numpy as np
import pytest

from nway_protocol.reference import resample_indices
from nway_protocol.smc_rd import make_rqmc_cloud


@pytest.mark.parametrize("scheme", ["multinomial", "stratified", "residual", "rqmc_sorted"])
def test_resampling_schemes_are_finite_and_in_range(scheme):
    weights = np.asarray([0.05, 0.15, 0.30, 0.50])
    particles = np.arange(12, dtype=float).reshape(4, 3)
    ancestors = resample_indices(
        weights, np.random.default_rng(5), scheme, particles,
    )
    assert ancestors.shape == (4,)
    assert np.issubdtype(ancestors.dtype, np.integer)
    assert np.all((0 <= ancestors) & (ancestors < 4))


@pytest.mark.parametrize("scheme", ["stratified", "residual", "rqmc_sorted"])
def test_research_resampling_frequencies_are_close_to_target(scheme):
    weights = np.asarray([0.08, 0.17, 0.30, 0.45])
    particles = np.asarray([
        [-1.0, -0.5], [-0.2, 0.1], [0.3, 0.4], [1.0, 0.8],
    ])
    rng = np.random.default_rng(91)
    counts = np.zeros(4)
    replicates = 3000
    for _ in range(replicates):
        counts += np.bincount(
            resample_indices(weights, rng, scheme, particles), minlength=4,
        )
    frequencies = counts / counts.sum()
    np.testing.assert_allclose(frequencies, weights, atol=0.015)


def test_scrambled_sobol_prior_cloud_has_valid_shape_and_weights():
    cloud = make_rqmc_cloud(256, np.eye(7), np.eye(7), seed=17)
    assert cloud.t.shape == (256, 7)
    assert cloud.l.shape == (256, 7)
    assert np.all(np.isfinite(cloud.t)) and np.all(np.isfinite(cloud.l))
    assert np.isclose(cloud.w.sum(), 1)
    assert np.max(np.abs(cloud.t.mean(axis=0))) < 0.08
