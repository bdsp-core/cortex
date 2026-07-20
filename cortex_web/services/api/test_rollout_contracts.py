from api.compute_rollout import DUAL_BRANCH_AUTO, SERIAL, compute_mode_for
from api.policy_rollout import AD6_POLICY, PRECISION_POLICY, termination_policy_for


class ParticipantStore:
    def __init__(self, email: str | None):
        self.email = email

    def get_participant(self, _code: str):
        return {"email": self.email} if self.email is not None else None


def test_termination_rollout_is_server_owned_and_fail_closed():
    db = ParticipantStore(" Pilot@Example.Test ")
    cfg = {
        "precision_policy_rollout": "email_allowlist",
        "precision_policy_emails": frozenset({"pilot@example.test"}),
    }
    assert termination_policy_for(db, cfg, "reader") == PRECISION_POLICY
    assert termination_policy_for(db, {**cfg, "precision_policy_emails": frozenset()}, "reader") == AD6_POLICY
    assert termination_policy_for(db, {"precision_policy_rollout": "unexpected"}, "reader") == AD6_POLICY
    assert termination_policy_for(db, {"precision_policy_rollout": "all"}, "reader") == PRECISION_POLICY


def test_compute_rollout_cannot_accelerate_ad6_and_fails_closed():
    db = ParticipantStore("pilot@example.test")
    allowlisted = {
        "precision_compute_rollout": "email_allowlist",
        "precision_compute_emails": frozenset({"pilot@example.test"}),
    }
    assert compute_mode_for(db, allowlisted, "reader", PRECISION_POLICY) == DUAL_BRANCH_AUTO
    assert compute_mode_for(db, allowlisted, "reader", AD6_POLICY) == SERIAL
    assert compute_mode_for(
        db,
        {"precision_compute_rollout": "unexpected"},
        "reader",
        PRECISION_POLICY,
    ) == SERIAL
