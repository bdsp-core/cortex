from cortex_termination_policy import StopDecision, TerminationPolicy


def test_contract_defaults_are_backward_compatible() -> None:
    decision = StopDecision(False, "continue")
    assert decision.verdicts is None
    assert decision.domain_statuses is None
    assert TerminationPolicy().finalize_verdicts() is None
