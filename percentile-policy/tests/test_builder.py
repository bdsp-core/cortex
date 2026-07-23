from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from percentile_policy.builder import QualityPolicy, build_artifact
from percentile_policy.model import DOMAINS, verify_artifact


class BuilderTests(unittest.TestCase):
    def test_build_is_pinned_filtered_and_deidentified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fits.csv"
            fields = [
                "domain", "rater_id", "rater_name", "sigma", "theta",
                "se_sigma", "se_theta", "n_trials", "converged",
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for domain in DOMAINS:
                    for idx, sigma in enumerate((0.5, 0.8, 1.2, 1.5)):
                        writer.writerow(
                            {
                                "domain": domain,
                                "rater_id": idx,
                                "rater_name": f"private-{domain}-{idx}",
                                "sigma": sigma,
                                "theta": 0,
                                "se_sigma": 0.05,
                                "se_theta": 0.05,
                                "n_trials": 50,
                                "converged": "True",
                            }
                        )
                    writer.writerow(
                        {
                            "domain": domain,
                            "rater_id": 999,
                            "rater_name": "must-not-appear",
                            "sigma": 0.02,
                            "theta": 0,
                            "se_sigma": 0.01,
                            "se_theta": 0.01,
                            "n_trials": 50,
                            "converged": "True",
                        }
                    )
            source_hash = sha256(source.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "generated": "test",
                        "true_producer": "test",
                        "k7_artifacts_sha256": {
                            "sdt_fits_k7.csv": source_hash,
                        },
                    }
                )
            )
            artifact, audit = build_artifact(
                source_csv=source,
                manifest_path=manifest,
                bootstrap_replicates=100,
                quantile_points=101,
                seed=7,
                quality_policy=QualityPolicy(),
                created_utc="2026-01-01T00:00:00Z",
            )
            rebuilt, _ = build_artifact(
                source_csv=source,
                manifest_path=manifest,
                bootstrap_replicates=100,
                quantile_points=101,
                seed=7,
                quality_policy=QualityPolicy(),
                created_utc="2026-01-01T00:00:00Z",
            )
            self.assertEqual(verify_artifact(artifact), [])
            self.assertEqual(
                rebuilt["artifactSha256"], artifact["artifactSha256"]
            )
            self.assertEqual(artifact["domains"]["spike"]["eligibleFitRows"], 4)
            self.assertEqual(
                audit["domains"]["spike"]["exclusionReasons"]["optimizer_boundary"],
                1,
            )
            serialized = json.dumps(artifact)
            self.assertNotIn("private-", serialized)
            self.assertNotIn("must-not-appear", serialized)

    def test_manifest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fits.csv"
            source.write_text("domain\n")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {"k7_artifacts_sha256": {"sdt_fits_k7.csv": "wrong"}}
                )
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_artifact(
                    source_csv=source,
                    manifest_path=manifest,
                    bootstrap_replicates=100,
                    quantile_points=101,
                )


if __name__ == "__main__":
    unittest.main()
