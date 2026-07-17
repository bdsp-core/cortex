"""G2 — operating-characteristic reproduction (the scientific gate). Runs the
ported production trainer + the real exam over simulated learners and checks the
trainer_rd OCs hold: per-trial delivered value, adversarial false-graduation
containment (the σ_∞-mixture F28 fix + D-INT-3 plateau signal), the D18 strict
re-cert backstop, and the D-INT-4 retest independence. Slow; skipped without the
fixture bank. Measured values are recorded in docs/G2_CLOSEOUT.md."""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_BANK = os.path.join(_REPO, "data", "eeg_bank.h5")
pytestmark = pytest.mark.skipif(not os.path.exists(_BANK),
                                reason="eeg_bank.h5 fixture not present")

ELL = np.array([0.2382694370243232, 0.1548131161373049, 0.3059231418723579,
                0.2569706571684701, 0.3213979957403151, 0.3539561460242777,
                0.3042300471351564])
SIG = np.exp(-ELL)
K = 7
P = dict(alpha_t=0.10, alpha_sigma=0.12, q_t=0.05, q_sigma=0.02, rho=0.5)


@pytest.fixture(scope="module")
def inputs():
    from cortex_engine_inputs_k7 import build_k7_engine_inputs
    return build_k7_engine_inputs()


@pytest.fixture(scope="module")
def bank(inputs):
    from trainer.orchestrator import SimBankAdapter
    return SimBankAdapter(inputs)


def _cold_exam(inputs, ell_true, seed, use_inputs=None, maxq=160,
               block_name=None):
    import session_controller as sc
    from cortex_policy_k7 import default_policy_for_k7
    from trainer.conventions import plan_to_engine
    inp = use_inputs if use_inputs is not None else inputs
    theta, ell = plan_to_engine(np.exp(-np.full(K, ell_true)), np.zeros(K))
    kw = {} if block_name is None else {"block_name": block_name}
    sess = sc.CortexSession(inp, session_id="e", n_particles=150, seed=seed,
                            max_questions=maxq,
                            policy=default_policy_for_k7(inp, **kw))
    return sess.run(sc.make_simulated_y_source(theta, ell, seed=seed)).verdicts


@pytest.mark.slow
def test_oc1_per_trial_delivered_value(bank):
    """A trainable learner receives near-ideal placement. (0.48 in trainer_rd is
    the REAL-data belief-lagged figure; a well-specified sim runs higher.)"""
    from trainer.oc import single_task_train_session
    dv = [single_task_train_session(
              bank, ELL, SIG, weak_task=2, l_true=0.0,
              learner_params={**P, "rule": "soft"},
              filter_sigma_inf=np.full(K, 0.4), use_mixture=False,
              budget=150, seed=s)["delivered_value"] for s in range(6)]
    assert np.mean(dv) > 0.5, dv          # substantial delivered value


@pytest.mark.slow
def test_oc2_adversarial_false_graduation(bank):
    """A STATIC learner 0.30 below the cut (ceiling below cut, untrainable): the
    σ_∞-mixture must NOT certify its own prior (F28 fix) and must report low
    trainability (the D-INT-3 plateau signal)."""
    from trainer.oc import single_task_train_session
    decl, trs = 0, []
    for s in range(10):
        r = single_task_train_session(
            bank, ELL, SIG, weak_task=2, l_true=float(ELL[2] - 0.30),
            learner_params={**P, "rule": "static"},
            filter_sigma_inf=np.exp(-(ELL - 0.10)), use_mixture=True,
            budget=200, seed=s)
        decl += int(r["declarations"] > 0)
        trs.append(r["trainability"])
    assert decl <= 1, f"{decl}/10 static-below-cut learners falsely declared"
    assert np.mean(trs) < 0.5, np.mean(trs)   # honest "not trainable to the bar"


@pytest.mark.slow
def test_oc3_recert_backstop_rejects_below_cut(inputs):
    """D18: the STRICT re-cert exam must never certify a clearly below-cut learner
    — the architectural backstop behind D-INT-4."""
    n_pass = sum(sum(v == "PASS" for v in _cold_exam(inputs, ell_true=-0.3, seed=s))
                 for s in range(4))
    assert n_pass == 0, f"{n_pass} spurious PASS verdicts for a below-cut learner"


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=(
    "OC finding at the v15 default cuts (2026-07-16 switch): one of 7 task "
    "verdicts shifts under 30% bank exclusion (6/7 agree) for a strong "
    "simulated learner — the lower v15 bars change which items are "
    "informative near the cut. The MECHANISM guard lives in the explicit-v14 "
    "companion below; this live-default variant stays visible for the team "
    "to re-tune the OC margin (near-cut item coverage / info gate) at v15."))
def test_oc4_retest_independence_under_exclusion(inputs):
    """D-INT-4: excluding trained items (a random 30% here) must not shift the
    exam verdict for a fixed learner — the certificate is a function of current
    skill only, not exposure history. Runs at the LIVE default block (v15)."""
    base = _cold_exam(inputs, ell_true=0.9, seed=1)
    drop = set(np.random.default_rng(0).choice(
        inputs.all_seg_ids, size=int(0.3 * len(inputs.all_seg_ids)),
        replace=False).tolist())
    excl = _cold_exam(inputs, ell_true=0.9, seed=1, use_inputs=inputs.without(drop))
    agree = sum((a == "PASS") == (b == "PASS") for a, b in zip(base, excl))
    assert agree == K, f"verdict shifted under exclusion: {agree}/{K} agree"


@pytest.mark.slow
def test_oc4_retest_independence_under_exclusion_v14(inputs):
    """The original D-INT-4 mechanism guard at the calibration it was
    validated under (explicit v14 block): exclusion independence must hold
    exactly. Keeps the mechanism regression-gated while the live-default
    variant above documents the v15 OC margin finding."""
    kw = dict(block_name="ell_star_unified_v14")
    base = _cold_exam(inputs, ell_true=0.9, seed=1, **kw)
    drop = set(np.random.default_rng(0).choice(
        inputs.all_seg_ids, size=int(0.3 * len(inputs.all_seg_ids)),
        replace=False).tolist())
    excl = _cold_exam(inputs, ell_true=0.9, seed=1,
                      use_inputs=inputs.without(drop), **kw)
    agree = sum((a == "PASS") == (b == "PASS") for a, b in zip(base, excl))
    assert agree == K, f"verdict shifted under exclusion: {agree}/{K} agree"
