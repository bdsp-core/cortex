"""Phase 2: Mode-B IUT-rule toggle (FIX-T1.9 dispute resolution).

Unified-merge decision (2026-05-18): default to the audit-endorsed
joint-posterior stopping rule; Berger min-of-marginals preserved verbatim
and selectable. Paper-1 is unaffected (Mode-A has no IUT); Mode-B is
deprecated for Paper-1 (Paper-2 scope).
"""
import inspect

import numpy as np
import pytest

from engine_mode_b import run_session_mcmc_certification


def test_default_iut_rule_is_joint():
    sig = inspect.signature(run_session_mcmc_certification)
    assert sig.parameters["iut_rule"].default == "joint"


def test_invalid_iut_rule_raises():
    with pytest.raises(ValueError, match="iut_rule must be"):
        run_session_mcmc_certification(
            method="hier", true_params=[0.0, 1.0, 0.0, 1.0], K=2,
            r_assumed=0.3, l_star=np.zeros(2), max_q=1, N=16, seed=0,
            iut_rule="bogus")


@pytest.mark.parametrize("rule", ["joint", "berger_marginals"])
def test_both_rules_run_and_pass_a_clear_rater(rule):
    out = run_session_mcmc_certification(
        method="hier", true_params=[0.0, 1.2, 0.0, 1.2], K=2,
        r_assumed=0.3, l_star=np.zeros(2), max_q=120, N=300, seed=1,
        iut_rule=rule)
    assert set(out["decisions"]) <= {-1, 0, 1}
    assert out["n_questions"] >= 1


@pytest.mark.slow
def test_joint_is_no_less_conservative_than_berger():
    """Joint-posterior P(all pass) <= min marginal, so the joint rule
    cannot fire EARLIER than Berger min-of-marginals on the same stream."""
    kw = dict(method="hier", true_params=[0.0, 1.0, 0.0, 1.0], K=2,
              r_assumed=0.3, l_star=np.zeros(2), max_q=200, N=400)
    for seed in (0, 1, 2):
        j = run_session_mcmc_certification(seed=seed, iut_rule="joint", **kw)
        b = run_session_mcmc_certification(
            seed=seed, iut_rule="berger_marginals", **kw)
        assert j["n_questions"] >= b["n_questions"], (
            f"seed {seed}: joint fired earlier ({j['n_questions']}) than "
            f"berger ({b['n_questions']}) — joint must be >= conservative")


@pytest.mark.slow
def test_brute_parity_threads_iut_rule():
    """The brute comparator accepts the same toggle (parity)."""
    for rule in ("joint", "berger_marginals"):
        out = run_session_mcmc_certification(
            method="brute", true_params=[0.0, 1.2, 0.0, 1.2], K=2,
            r_assumed=0.3, l_star=np.zeros(2), max_q=120, N=300, seed=3,
            iut_rule=rule)
        assert out["n_questions"] >= 1
    with pytest.raises(ValueError, match="iut_rule must be"):
        run_session_mcmc_certification(
            method="brute", true_params=[0.0, 1.0, 0.0, 1.0], K=2,
            r_assumed=0.3, l_star=np.zeros(2), max_q=1, N=16, seed=0,
            iut_rule="nope")
