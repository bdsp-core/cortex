from dataclasses import replace

from nway_protocol.qualification import QualificationConfig, _run_arm

from draw_latent_rd.harness import _run_categorical_draw_latent

TINY = QualificationConfig(
    particles=64, own_cap=2, bank_segments=24, selector_per_domain=3,
    mh_steps=2, formal_sbc=True, truth_t_mean=0.0, truth_l_mean=0.0,
    truth_sd=1.0, artifact_draws=((0.9912, 0.15, 1.0),),
)


def test_single_atom_reduces_to_shipping_mixture_arm() -> None:
    """With one atom the draw latent is degenerate: the draw-latent engine
    must reproduce the shipping single-draw mixture arm byte-for-byte."""
    for seed in (9_100_001, 9_100_002, 9_100_003):
        shipping = _run_arm(seed, "categorical_f1", TINY)
        draw_latent, diagnostics = _run_categorical_draw_latent(seed, TINY)
        assert diagnostics["atoms_surviving"] == 1
        assert shipping.__dict__ == draw_latent.__dict__


def test_two_identical_atoms_match_single_atom_statistics() -> None:
    """Splitting the same (beta, lapse) across two atoms changes lineage
    bookkeeping but not the model; posterior summaries must agree."""
    seed = 9_100_004
    single, _ = _run_categorical_draw_latent(seed, TINY)
    double, _ = _run_categorical_draw_latent(seed, replace(
        TINY, artifact_draws=((0.9912, 0.15, 0.5), (0.9912, 0.15, 0.5)),
    ))
    assert abs(single.skill_rmse - double.skill_rmse) < 0.15
    assert abs(single.skill_coverage - double.skill_coverage) <= 1 / 6 + 1e-9
