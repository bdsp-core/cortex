"""The full decision surface of every rollout/exposure gate.

These five resolvers decide, per sitting, which stopping policy a participant
gets, where its compute runs, whether training is offered, and which
allocation mode the trainer uses. They are server-owned and fail-closed by
design, and they look similar enough to invite unification — so this table
pins what each one actually returns across modes, allowlist states, and
participant shapes BEFORE any shared helper is introduced.

Two families exist and they are NOT interchangeable:

  * email-allowlist (termination_policy_for, compute_mode_for): an
    unrecognised mode falls through to the SAFE value explicitly, and the
    participant is looked up from the db by the resolver itself.
  * tri-key (training_enabled, enabled, alloc_for): the caller supplies the
    participant row, matching is against code / public_id / email, and an
    unrecognised mode lands in the allowlist branch rather than being
    special-cased — which reaches the same safe default only because an empty
    allowlist returns it.

The families also differ on whitespace: the tri-key engine gates strip their
candidates, training_enabled does not.
"""
from __future__ import annotations

import pytest

from . import dashboard_logic, engine_trainer
from .compute_rollout import DUAL_BRANCH_AUTO, SERIAL, compute_mode_for
from .policy_rollout import (
    AD6_POLICY,
    PRECISION_C1,
    PRECISION_POLICY,
    precision_recalibration_for,
    termination_policy_for,
)


class _Db:
    """Minimal participant lookup for the email-allowlist resolvers."""

    def __init__(self, participant):
        self._participant = participant

    def get_participant(self, _code):
        return self._participant


ALLOW = frozenset({"u-code", "123456789", "pilot@example.test"})
PARTICIPANT = {"public_id": "123456789", "email": "pilot@example.test"}


# ── training_enabled: off | all | cohort(tri-key) ──────────────────────────

@pytest.mark.parametrize("cfg,code,participant,expected", [
    ({"training_mode": "off"}, "u-code", PARTICIPANT, False),
    ({"training_mode": "all"}, "nobody", None, True),
    ({}, "nobody", None, True),                       # default posture is "all"
    ({"training_mode": "cohort"}, "u-code", None, False),          # no allowlist
    ({"training_mode": "cohort", "training_allowlist": ALLOW},
     "u-code", None, True),                                        # by code
    ({"training_mode": "cohort", "training_allowlist": ALLOW},
     "U-CODE", None, True),                                        # case-folded
    ({"training_mode": "cohort", "training_allowlist": ALLOW},
     "other", PARTICIPANT, True),                                  # by public_id/email
    ({"training_mode": "cohort", "training_allowlist": ALLOW},
     "other", {"public_id": "999", "email": "no@example.test"}, False),
    # An unrecognised mode is NOT special-cased; it takes the allowlist branch.
    ({"training_mode": "bogus", "training_allowlist": ALLOW},
     "u-code", None, True),
    ({"training_mode": "bogus"}, "u-code", None, False),
])
def test_training_enabled_matrix(cfg, code, participant, expected):
    assert dashboard_logic.training_enabled(cfg, code, participant) is expected


# ── engine_trainer.enabled: off | all | cohort(tri-key, stripped) ──────────

@pytest.mark.parametrize("cfg,code,participant,expected", [
    ({"trainer_engine": "off"}, "u-code", PARTICIPANT, False),
    ({"trainer_engine": "all"}, "nobody", None, True),
    ({}, "nobody", None, True),                       # default posture is "all"
    ({"trainer_engine": "cohort"}, "u-code", None, False),
    ({"trainer_engine": "cohort", "trainer_engine_allowlist": ALLOW},
     "u-code", None, True),
    ({"trainer_engine": "cohort", "trainer_engine_allowlist": ALLOW},
     "  U-Code  ", None, True),                       # stripped, unlike above
    ({"trainer_engine": "cohort", "trainer_engine_allowlist": ALLOW},
     "other", PARTICIPANT, True),
    ({"trainer_engine": "cohort", "trainer_engine_allowlist": ALLOW},
     "other", {"public_id": "999", "email": "no@example.test"}, False),
    ({"trainer_engine": "bogus", "trainer_engine_allowlist": ALLOW},
     "u-code", None, True),
])
def test_engine_enabled_matrix(cfg, code, participant, expected):
    assert engine_trainer.enabled(cfg, code, participant) is expected


# ── alloc_for: thompson | greedy(default) + tri-key opt-in ─────────────────

@pytest.mark.parametrize("cfg,code,participant,expected", [
    ({"trainer_alloc": "thompson"}, "nobody", None, "thompson"),
    ({}, "nobody", None, "greedy"),                   # default posture is greedy
    ({"trainer_alloc": "greedy"}, "u-code", None, "greedy"),
    ({"trainer_alloc": "greedy", "trainer_alloc_allowlist": ALLOW},
     "u-code", None, "thompson"),
    ({"trainer_alloc": "greedy", "trainer_alloc_allowlist": ALLOW},
     "other", PARTICIPANT, "thompson"),
    ({"trainer_alloc": "greedy", "trainer_alloc_allowlist": ALLOW},
     "other", {"public_id": "999", "email": "no@example.test"}, "greedy"),
])
def test_alloc_for_matrix(cfg, code, participant, expected):
    assert engine_trainer.alloc_for(cfg, code, participant) == expected


# ── termination_policy_for: all | email_allowlist | fail closed ────────────

@pytest.mark.parametrize("cfg,participant,expected", [
    ({"precision_policy_rollout": "all"}, None, PRECISION_POLICY),
    ({"precision_policy_rollout": "off"}, PARTICIPANT, AD6_POLICY),
    ({}, PARTICIPANT, AD6_POLICY),                    # unset fails closed
    ({"precision_policy_rollout": "bogus"}, PARTICIPANT, AD6_POLICY),
    ({"precision_policy_rollout": "email_allowlist"}, PARTICIPANT, AD6_POLICY),
    ({"precision_policy_rollout": "email_allowlist",
      "precision_policy_emails": frozenset({"pilot@example.test"})},
     PARTICIPANT, PRECISION_POLICY),
    ({"precision_policy_rollout": "email_allowlist",
      "precision_policy_emails": frozenset({"pilot@example.test"})},
     {"email": "  Pilot@Example.Test  "}, PRECISION_POLICY),   # normalised
    ({"precision_policy_rollout": "email_allowlist",
      "precision_policy_emails": frozenset({"pilot@example.test"})},
     None, AD6_POLICY),                               # no participant row
])
def test_termination_policy_matrix(cfg, participant, expected):
    assert termination_policy_for(_Db(participant), cfg, "reader") == expected


# ── precision_recalibration_for: gated on Precision, then all|allowlist ────

@pytest.mark.parametrize("cfg,participant,policy,expected", [
    # AD6 sittings never carry a recalibration, whatever the rollout says.
    ({"precision_c1_rollout": "all"}, PARTICIPANT, AD6_POLICY, None),
    ({"precision_c1_rollout": "all"}, None, PRECISION_POLICY, PRECISION_C1),
    ({"precision_c1_rollout": "off"}, PARTICIPANT, PRECISION_POLICY, None),
    ({}, PARTICIPANT, PRECISION_POLICY, None),        # unset fails closed
    ({"precision_c1_rollout": "bogus"}, PARTICIPANT, PRECISION_POLICY, None),
    ({"precision_c1_rollout": "email_allowlist"},
     PARTICIPANT, PRECISION_POLICY, None),            # no allowlist
    ({"precision_c1_rollout": "email_allowlist",
      "precision_c1_emails": frozenset({"pilot@example.test"})},
     PARTICIPANT, PRECISION_POLICY, PRECISION_C1),
    ({"precision_c1_rollout": "email_allowlist",
      "precision_c1_emails": frozenset({"pilot@example.test"})},
     {"email": "  Pilot@Example.Test  "}, PRECISION_POLICY, PRECISION_C1),
    ({"precision_c1_rollout": "email_allowlist",
      "precision_c1_emails": frozenset({"pilot@example.test"})},
     None, PRECISION_POLICY, None),                   # no participant row
])
def test_precision_recalibration_matrix(cfg, participant, policy, expected):
    assert precision_recalibration_for(
        _Db(participant), cfg, "reader", policy) == expected


# ── compute_mode_for: gated on Precision, then all | allowlist ─────────────

@pytest.mark.parametrize("cfg,participant,policy,expected", [
    # AD6 never accelerates, whatever the compute rollout says.
    ({"precision_compute_rollout": "all"}, PARTICIPANT, AD6_POLICY, SERIAL),
    ({"precision_compute_rollout": "all"}, None, PRECISION_POLICY, DUAL_BRANCH_AUTO),
    ({}, PARTICIPANT, PRECISION_POLICY, SERIAL),      # unset fails closed
    ({"precision_compute_rollout": "bogus"}, PARTICIPANT, PRECISION_POLICY, SERIAL),
    ({"precision_compute_rollout": "email_allowlist"},
     PARTICIPANT, PRECISION_POLICY, SERIAL),
    ({"precision_compute_rollout": "email_allowlist",
      "precision_compute_emails": frozenset({"pilot@example.test"})},
     PARTICIPANT, PRECISION_POLICY, DUAL_BRANCH_AUTO),
    ({"precision_compute_rollout": "email_allowlist",
      "precision_compute_emails": frozenset({"pilot@example.test"})},
     None, PRECISION_POLICY, SERIAL),
])
def test_compute_mode_matrix(cfg, participant, policy, expected):
    assert compute_mode_for(_Db(participant), cfg, "reader", policy) == expected
