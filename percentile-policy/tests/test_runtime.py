from __future__ import annotations

import struct
import unittest

from percentile_policy.model import DOMAINS, artifact_content_hash
from percentile_policy.runtime import (
    build_runtime,
    metadata_content_hash,
    verify_runtime_metadata,
)


def artifact() -> dict:
    domains = {
        domain: {
            "eligibleFitRows": 25,
            "bootstrapQuantiles": [[-1.0, 0.0, 1.0], [-1.1, 0.0, 1.1]],
            "centralQuantiles": [-1.0, 0.0, 1.0],
            "sensitivityQuantiles": {"strict": [-0.8, 0.0, 0.8]},
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


class RuntimeTests(unittest.TestCase):
    def test_layout_is_contiguous_self_hashed_and_little_endian(self):
        metadata, data = build_runtime(
            artifact(), metadata_url="/x.json", data_url="/x.bin"
        )
        self.assertEqual(verify_runtime_metadata(metadata), [])
        self.assertEqual(metadata["metadataSha256"], metadata_content_hash(metadata))
        self.assertEqual(len(data), metadata["floatCount"] * 4)
        self.assertEqual(struct.unpack_from("<fff", data), (-1.0, 0.0, 1.0))
        self.assertEqual(
            metadata["domains"]["spike"]["bootstrap"],
            {"offset": 3, "count": 6, "rows": 2, "cols": 3},
        )

    def test_metadata_hash_detects_change(self):
        metadata, _data = build_runtime(
            artifact(), metadata_url="/x.json", data_url="/x.bin"
        )
        metadata["referenceLabel"] = "changed"
        self.assertIn(
            "metadataSha256 does not match canonical content",
            verify_runtime_metadata(metadata),
        )


if __name__ == "__main__":
    unittest.main()
