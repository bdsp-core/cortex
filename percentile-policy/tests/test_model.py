from __future__ import annotations

from copy import deepcopy
import unittest

from percentile_policy.model import (
    DOMAINS,
    artifact_content_hash,
    cdf_from_quantile_curve,
    score_candidate,
    verify_artifact,
)


def artifact() -> dict:
    domains = {
        domain: {
            "bootstrapQuantiles": [
                [-1.0, 0.0, 1.0],
                [-1.1, 0.0, 1.1],
            ],
            "centralQuantiles": [-1.0, 0.0, 1.0],
            "sensitivityQuantiles": {
                "strict": [-0.8, 0.0, 0.8],
            },
        }
        for domain in DOMAINS
    }
    value = {
        "schemaVersion": "cortex-percentile-norm/v1",
        "normId": "test",
        "domainOrder": list(DOMAINS),
        "referenceLabel": "test reference",
        "domains": domains,
        "requiredWarnings": ["test warning"],
    }
    value["artifactSha256"] = artifact_content_hash(value)
    return value


class CdfTests(unittest.TestCase):
    def test_curve_inversion(self):
        curve = [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.assertEqual(cdf_from_quantile_curve(-3.0, curve), 0.0)
        self.assertAlmostEqual(cdf_from_quantile_curve(0.0, curve), 0.5)
        self.assertEqual(cdf_from_quantile_curve(3.0, curve), 1.0)
        points = [cdf_from_quantile_curve(x / 10, curve) for x in range(-30, 31)]
        self.assertEqual(points, sorted(points))

    def test_artifact_hash_detects_change(self):
        value = artifact()
        self.assertEqual(verify_artifact(value), [])
        changed = deepcopy(value)
        changed["domains"]["spike"]["bootstrapQuantiles"][0][0] = -9.0
        self.assertIn(
            "artifactSha256 does not match canonical content",
            verify_artifact(changed),
        )

    def test_artifact_rejects_nonmonotone_sensitivity_curve(self):
        value = artifact()
        value["domains"]["spike"]["sensitivityQuantiles"]["strict"] = [
            -1.0, 1.0, 0.0,
        ]
        value["artifactSha256"] = artifact_content_hash(value)
        self.assertIn(
            "spike: non-monotone quantile in sensitivityQuantiles.strict",
            verify_artifact(value),
        )

    def test_score_midpoint(self):
        result = score_candidate(
            artifact(),
            {"domains": {"spike": {"ellSamples": [0.0], "weights": [1.0]}}},
        )
        score = result["domains"]["spike"]
        self.assertAlmostEqual(score["estimate"], 50.0)
        self.assertEqual(score["display"]["estimate"], 50)
        self.assertEqual(score["candidateMethod"], "posterior_particles")
        self.assertEqual(
            result["policyStatus"], "provisional_historical_calibration_cohort"
        )

    def test_candidate_uncertainty_is_propagated(self):
        result = score_candidate(
            artifact(),
            {
                "domains": {
                    "spike": {
                        "ellSamples": [-1.0, 1.0],
                        "weights": [0.5, 0.5],
                    }
                }
            },
        )
        score = result["domains"]["spike"]
        self.assertLess(score["lower"], 50)
        self.assertGreater(score["upper"], 50)
        self.assertAlmostEqual(score["estimate"], 50.0)

    def test_mean_sd_path_is_explicitly_labeled(self):
        result = score_candidate(
            artifact(),
            {"domains": {"spike": {"ellMean": 0.0, "ellSd": 0.2}}},
        )
        self.assertEqual(
            result["domains"]["spike"]["candidateMethod"],
            "normal_approximation_from_mean_sd",
        )


if __name__ == "__main__":
    unittest.main()
