"""Phase 0 structural self-tests.

Asserts every invariant the Phase-0 gate promises: directory skeleton,
package importability, console-entry-point resolvability (loud stub, not
ImportError), pyproject/requirements consistency, tamper-evident source
provenance, and the target Python version. Fast (no slow/nightly markers).
"""

import hashlib
import importlib
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "engine", "engine/variants", "deployment", "pipeline", "calibration",
    "bridge", "tests", "tests/mode_b", "experiments", "slides", "docs",
    "archive", "data", "data/labels", "data/deployment_prior",
    "data/engine_inputs",
]
PACKAGES = ["engine", "engine.variants", "deployment", "pipeline",
            "calibration", "bridge"]
ENTRY_POINTS = {
    "ilae-paper": ("bridge.run_multi_auroc_bridge", "main", "Phase 2"),
    "ilae-calibrate": ("calibration.run_youden_calibration", "main", "Phase 3"),
    "ilae-deploy": ("deployment.run_deployment_sim", "main", "Phase 4"),
}


def test_python_is_311():
    assert sys.version_info[:2] == (3, 11), (
        f"target env is 3.11.x; got {sys.version.split()[0]}"
    )
    assert (ROOT / ".python-version").read_text().strip() == "3.11.9"


@pytest.mark.parametrize("rel", EXPECTED_DIRS)
def test_directory_structure(rel):
    assert (ROOT / rel).is_dir(), f"missing skeleton dir: {rel}"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_packages_importable(pkg):
    mod = importlib.import_module(pkg)
    assert mod is not None


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_entry_point_resolves_to_loud_stub(name):
    mod_name, fn_name, phase = ENTRY_POINTS[name]
    mod = importlib.import_module(mod_name)  # must not ImportError
    fn = getattr(mod, fn_name)
    with pytest.raises(NotImplementedError) as exc:
        fn()
    msg = str(exc.value)
    assert name in msg and phase in msg, (
        f"{name} stub message must name the script and implementing phase; "
        f"got: {msg!r}"
    )


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def test_pyproject_declares_three_scripts_with_expected_targets():
    pp = _pyproject()
    scripts = pp["project"]["scripts"]
    assert set(scripts) == set(ENTRY_POINTS)
    for name, (mod_name, fn_name, _phase) in ENTRY_POINTS.items():
        assert scripts[name] == f"{mod_name}:{fn_name}"


def test_pyproject_package_list_matches_skeleton():
    pp = _pyproject()
    assert set(pp["tool"]["setuptools"]["packages"]) == set(PACKAGES)
    assert pp["project"]["requires-python"] == "==3.11.*"


def _norm_req(spec: str) -> str:
    """Normalize a requirement to its `name==version` token: drop inline
    `#` comments and surrounding whitespace."""
    return spec.split("#", 1)[0].strip()


def test_requirements_and_pyproject_pins_are_consistent():
    """Every requirements.txt pin must appear identically in pyproject
    (runtime deps + optional 'test'), so there is one effective pin set."""
    req_lines = [
        _norm_req(ln) for ln in (ROOT / "requirements.txt").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    req = {ln.split("==")[0].lower(): ln for ln in req_lines if "==" in ln}
    pp = _pyproject()
    pp_pins = list(pp["project"]["dependencies"]) + list(
        pp["project"]["optional-dependencies"]["test"]
    )
    pp_map = {_norm_req(p).split("==")[0].lower(): _norm_req(p)
              for p in pp_pins if "==" in p}
    missing = sorted(set(req) - set(pp_map))
    assert not missing, f"requirements pins absent from pyproject: {missing}"
    mismatched = {k: (req[k], pp_map[k]) for k in req if req[k] != pp_map[k]}
    assert not mismatched, f"version mismatch req vs pyproject: {mismatched}"


def test_source_manifests_present_and_tamper_evident():
    """The recorded manifest-of-manifests MD5s in MERGE_SOURCE_MANIFEST.md
    must match the actual checksum files (provenance record is self-consistent)."""
    md = (ROOT / "docs" / "MERGE_SOURCE_MANIFEST.md").read_text()
    rows = dict(re.findall(r"`(\w[\w.]+\.md5)`\s*\|\s*`([0-9a-f]{32})`", md))
    assert set(rows) == {
        "source_methodology.md5", "source_pi_deployment.md5",
        "reference_consulted.md5",
    }, f"unexpected manifest set in provenance doc: {sorted(rows)}"
    for fname, recorded in rows.items():
        p = ROOT / "docs" / "_manifests" / fname
        assert p.is_file() and p.stat().st_size > 0, f"empty/missing {fname}"
        actual = hashlib.md5(p.read_bytes()).hexdigest()
        assert actual == recorded, (
            f"{fname}: recorded {recorded} != actual {actual} "
            f"(source repo changed since merge snapshot, or doc is stale)"
        )


def test_gitignore_covers_new_artifacts():
    gi = (ROOT / ".gitignore").read_text()
    for needed in (".venv/", "__pycache__/", "*.egg-info/", ".pytest_cache/"):
        assert needed in gi, f".gitignore missing {needed}"
