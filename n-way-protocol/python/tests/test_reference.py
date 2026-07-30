from __future__ import annotations

import numpy as np

from nway_protocol.qualification import QualificationConfig, run_replicate, summarize
from nway_protocol.reference import (
    Observation,
    binary_p_yes,
    expected_loss,
    f1_probabilities,
    f1_probabilities_artifact,
    make_cloud,
    posterior_moments,
    signal_z,
    update,
)

GROUP = (1, 2, 3, 4, 5, 6)


def test_f1_preserves_binary_marginal_and_normalizes():
    t = np.array([[0.1, -0.2, 0.3, -0.1, 0.4, -0.5, 0.2]])
    l = np.array([[0.2, 0.1, -0.3, 0.4, 0, -0.1, 0.5]])
    s = np.linspace(-1, 1, 7)
    sd = np.linspace(0.02, 0.08, 7)
    probabilities = f1_probabilities(t, l, s, sd, 3, GROUP, 0.9912, 0)
    z = signal_z(l[:, 3], t[:, 3], s[3], sd[3])
    assert probabilities[0, GROUP.index(3)] == binary_p_yes(z)[0]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14)


def test_wrong_pick_identity_changes_posterior():
    rng = np.random.default_rng(8)
    base = make_cloud(64, np.eye(7), np.eye(7), rng)
    s = np.linspace(-1.2, 1.1, 7)
    sd = np.full(7, 0.05)
    import copy
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    update(a, Observation("categorical_f1", 1, 0, 2, group=GROUP), s, sd, 0.9912, 0)
    update(b, Observation("categorical_f1", 1, 0, 6, group=GROUP), s, sd, 0.9912, 0)
    assert not np.allclose(a.w, b.w)


def test_artifact_mixture_preserves_focal_marginal_and_updates_finitely():
    rng = np.random.default_rng(81)
    cloud = make_cloud(64, np.eye(7), np.eye(7), rng)
    s = np.linspace(-1.2, 1.1, 7)
    sd = np.full(7, 0.05)
    draws = ((0.9, 0.0, 0.5), (1.1, 0.02, 0.5))
    mixture = f1_probabilities_artifact(
        cloud.t, cloud.l, s, sd, 1, GROUP, 0.9912, 0, draws,
    )
    plugin = f1_probabilities(cloud.t, cloud.l, s, sd, 1, GROUP, 0.9912, 0)
    manual_mixture = sum(
        weight * f1_probabilities(cloud.t, cloud.l, s, sd, 1, GROUP, beta, lapse)
        for beta, lapse, weight in draws
    )
    np.testing.assert_allclose(mixture, manual_mixture, atol=1e-14)
    np.testing.assert_allclose(
        mixture[:, GROUP.index(1)], plugin[:, GROUP.index(1)], atol=1e-14,
    )
    np.testing.assert_allclose(mixture.sum(axis=1), 1, atol=1e-14)
    update(
        cloud, Observation("categorical_f1", 1, 0, 2, group=GROUP),
        s, sd, 0.9912, 0, draws,
    )
    assert np.all(np.isfinite(cloud.w)) and np.isclose(cloud.w.sum(), 1)


def test_categorical_expected_loss_is_finite():
    cloud = make_cloud(96, np.eye(7), np.eye(7), np.random.default_rng(9))
    s = np.linspace(-0.8, 1.4, 7)
    sd = np.full(7, 0.08)
    loss = expected_loss(cloud, 2, s, sd, GROUP, 0.9912, 0)
    moments = posterior_moments(cloud)
    baseline = np.square(moments["t_sd"]).sum() + np.square(moments["l_sd"]).sum()
    assert 0 <= loss <= baseline


def test_adaptive_qualification_smoke_reports_both_parameters():
    config = QualificationConfig(
        replicates=1, particles=48, own_cap=1,
        bank_segments=12, selector_per_domain=2, mh_steps=1,
    )
    rows = run_replicate(62_100_999, config)
    result = summarize(rows, config)
    assert set(result["arms"]) == {"binary", "categorical_f1"}
    assert result["arms"]["categorical_f1"]["questions"] == 6
    assert np.isfinite(result["ratios"]["skill_width_nway_over_binary"])
    assert np.isfinite(result["ratios"]["bias_width_nway_over_binary"])
    assert result["arms"]["categorical_f1"]["skill_sbc"]["n"] == 6
    assert len(result["arms"]["categorical_f1"]["bias_sbc"]["decile_histogram"]) == 10
    assert result["arms"]["categorical_f1"]["skill_coverage_uncertainty"]["trials"] == 6
    assert "skill_coverage_nway_minus_binary" in result["paired_comparisons"]
