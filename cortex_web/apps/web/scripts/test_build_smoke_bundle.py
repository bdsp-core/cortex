import json
from pathlib import Path

from build_smoke_bundle import build_bundle


def test_synthetic_smoke_bundle_has_both_rendering_contracts(tmp_path: Path):
    manifest_path = build_bundle(tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text())

    assert manifest["version"] == "_smoke"
    assert manifest["nSegments"] == 20
    spike = [segment for segment in manifest["segments"] if segment["testClass"] == "spike"]
    iiic = [segment for segment in manifest["segments"] if segment["testClass"] == "iiic"]
    assert len(spike) == 6
    assert len(iiic) == 14
    assert all(segment["spec"] == "" for segment in spike)
    assert all(segment["specShape"] == [64, 96] for segment in iiic)

    for segment in manifest["segments"]:
        eeg = manifest_path.parent / segment["eeg"]
        assert eeg.stat().st_size == segment["nCh"] * segment["nSamp"] * 2
        if segment["spec"]:
            spec = manifest_path.parent / segment["spec"]
            assert spec.stat().st_size == 64 * 96
