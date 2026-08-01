import numpy as np

from nway_protocol.artifact_engine_frame import (
    fit_reader_model,
    population_draws,
    random_effects_log_beta,
)
from nway_protocol.engine_frame_refit import ReadRecord


def _read(asked: int, pick: int, s_mean: tuple, s_sd: tuple) -> ReadRecord:
    return ReadRecord(
        reader_id="reader", source_id="case", asked_k=asked, pick_k=pick,
        s_mean=s_mean, s_sd=s_sd,
    )


def test_random_effects_recovers_homogeneous_population() -> None:
    fits = [
        {"beta": 1.2, "standard_error": 0.012, "n_wrong_picks": 100}
        for _ in range(8)
    ]
    effects = random_effects_log_beta(fits)
    assert abs(np.exp(effects["mu_log_beta"]) - 1.2) < 1e-9
    assert effects["tau_log_beta"] == 0.0
    assert effects["n_readers_used"] == 8


def test_random_effects_excludes_bound_hitting_fits() -> None:
    fits = [
        {"beta": 1.0, "standard_error": 0.05, "n_wrong_picks": 100},
        {"beta": 1.1, "standard_error": 0.05, "n_wrong_picks": 100},
        {"beta": 99.99, "standard_error": 0.05, "n_wrong_picks": 100},
        {"beta": 1.0, "standard_error": float("inf"), "n_wrong_picks": 100},
    ]
    assert random_effects_log_beta(fits)["n_readers_used"] == 2


def test_population_draws_are_sorted_quantiles_with_equal_weight() -> None:
    effects = {"mu_log_beta": 0.0, "tau_log_beta": 0.3, "se_mu_log_beta": 0.0}
    draws = population_draws(effects, 0.02, 9)
    betas = [draw["beta"] for draw in draws]
    assert betas == sorted(betas)
    assert abs(betas[4] - 1.0) < 1e-12  # median draw sits at exp(mu)
    assert all(abs(draw["weight"] - 1 / 9) < 1e-12 for draw in draws)
    assert all(draw["distractor_lapse"] == 0.02 for draw in draws)


def test_reader_model_separates_bias_from_sensitivity() -> None:
    rng = np.random.default_rng(7)
    s_grid = rng.normal(0.8, 0.6, size=400)
    reads = []
    for value in s_grid:
        s_mean = (0.0, float(value), 0.0, 0.0, 0.0, 0.0, 0.0)
        s_sd = (1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0)
        # A biased-but-sensitive reader: correct whenever shifted evidence
        # clears zero, wrong otherwise (deterministically sharp).
        correct = value - 0.9 > 0.0
        reads.append(_read(1, 1 if correct else 2, s_mean, s_sd))
    model = fit_reader_model(reads)
    assert model["t"][0] < -0.4  # bias absorbed by t, not by collapsing l
    assert model["l"] > 0.5  # sharp reader keeps high sensitivity
