"""The learning protocol engine, packaged for sharing.

Three layers (architecture: docs/learning_engine_derivation.md and the
engine paper draft in the parent repo):

  M1 population learner  multimodel.run_hier_multi — one hierarchical
                         fit per repository yields the artifact
                         (rates, floors + dispersion, gate, lapse,
                         state priors).
  M2 session engine      msengine.MSSessionEngine / MSBelief — particle
                         belief over per-learner state, exam-margin
                         placement, double-eta readiness, logged-only
                         plateau signals.
  ports                  msengine.belief_seed_from_testing_result and
                         msengine.prior_blocks_from_population — the
                         exact handoff with the adaptive testing
                         package (docs/handoff_contract.md).

Run `python demo_end_to_end.py` at the package root for a fully
synthetic fit -> artifact -> closed-loop-training demonstration.
"""
from .learnmodel import (gate_centers, gate_railed, learn_weight,
                         learn_weight_coef, wilson_simplex,
                         zstar_from_coef)
from .mixedengine import (MixedBelief, MixedSessionEngine,
                          artifact_from_posterior,
                          floor_joint_from_draws)
from .registry import (Registry, auroc_of_ell, ell_of_auroc,
                       ell_reduced, p_assert)
from .msengine import (MSBelief, MSSessionEngine,
                       belief_seed_from_testing_result,
                       prior_blocks_from_population)

# Calibration-stage surface (M1: NumPyro/JAX) resolves LAZILY so the
# session-time import path stays numpy/scipy-only — host runtimes ban JAX
# at module boundaries (single-thread BLAS determinism contracts), and the
# engine's own architecture keeps fitting calibration-side (frozen
# artifacts). `from learning_engine import run_hier_multi` still works; it
# just imports jax at that moment, never before.
_CALIBRATION_EXPORTS = ("hier_multi_model", "make_multi_cohort",
                        "multi_lls_one", "multi_loglik_one",
                        "pooled_loglik", "run_hier_multi")


def __getattr__(name):
    if name in _CALIBRATION_EXPORTS:
        from . import multimodel
        return getattr(multimodel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MSBelief", "MSSessionEngine",
    "belief_seed_from_testing_result", "prior_blocks_from_population",
    "run_hier_multi", "hier_multi_model", "make_multi_cohort",
    "multi_lls_one", "multi_loglik_one", "pooled_loglik",
    "gate_centers", "gate_railed", "learn_weight", "learn_weight_coef",
    "wilson_simplex", "zstar_from_coef",
    "Registry", "auroc_of_ell", "ell_of_auroc", "ell_reduced", "p_assert",
    "MixedBelief", "MixedSessionEngine", "artifact_from_posterior",
    "floor_joint_from_draws",
]
