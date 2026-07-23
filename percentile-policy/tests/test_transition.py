from __future__ import annotations

import unittest

from percentile_policy.model import DOMAINS
from percentile_policy.transition import evaluate_transition


def ready_metrics() -> dict:
    return {
        "domains": {
            domain: {
                "effectiveFirstAttemptN": 600,
                "maxCentralIntervalHalfwidthPp": 4.0,
                "maxAnchorDriftPp": 2.0,
            }
            for domain in DOMAINS
        },
        "stableWindows": 2,
        "requiredStrataCoverage": 0.9,
        "largestSiteShare": 0.2,
        "firstAttemptOnly": True,
        "preTrainingOnly": True,
        "instrumentCompatibilityValidated": True,
        "posteriorCalibrationPassed": True,
        "difReviewPassed": True,
        "dataQualityPassed": True,
        "privacyAndConsentApproved": True,
        "independentScientificReviewApproved": True,
        "governanceCutoverApproved": True,
    }


class TransitionTests(unittest.TestCase):
    def test_ready_still_never_auto_switches(self):
        result = evaluate_transition(ready_metrics())
        self.assertTrue(result["eligibleForGovernedCutover"])
        self.assertFalse(result["automaticCutoverAllowed"])

    def test_one_domain_failure_blocks_cutover(self):
        metrics = ready_metrics()
        metrics["domains"]["lpd"]["effectiveFirstAttemptN"] = 499
        result = evaluate_transition(metrics)
        self.assertFalse(result["eligibleForGovernedCutover"])
        failed = [g["name"] for g in result["gates"] if not g["passed"]]
        self.assertEqual(failed, ["lpd.effective_n"])

    def test_governance_is_a_required_gate(self):
        metrics = ready_metrics()
        metrics["governanceCutoverApproved"] = False
        result = evaluate_transition(metrics)
        self.assertFalse(result["eligibleForGovernedCutover"])


if __name__ == "__main__":
    unittest.main()
