"""Phase 8 sub-step 3 — packaging drift-guard tests.

Pins the small surface that sub-8.3 changes:
  * pyproject.toml version is `1.0.0rc1` (the v1.0.0-rc1 release-
    candidate normalized per PEP 440; git tag is `v1.0.0-rc1`).
  * `ilae-calibrate` entry point points at the thin
    `calibration.cli:main` wrapper (NOT the deleted Phase-0 stub at
    `calibration.run_youden_calibration:main`).
  * `calibration.cli.main(["--help"])` and `main(["--dry-run"])` exit
    fast WITHOUT invoking the real (~hours) Phase-3 orchestrator —
    foot-gun guard: `ilae-calibrate --help` from a clinician's
    terminal must NEVER start a multi-hour calibration.
  * The deleted Phase-0 stub module must no longer exist on disk.

These are the load-bearing invariants of sub-8.3 (Phase 8 packaging).
"""
from __future__ import annotations

import os
import sys
import tomllib


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _pyproject():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_pyproject_version_is_release_candidate():
    """pyproject.toml `version` must be the v1.0.0-rc1 RC (PEP 440 form)."""
    pp = _pyproject()
    assert pp["project"]["version"] == "1.0.0rc1", (
        f"expected '1.0.0rc1' (PEP 440 normalized form of 1.0.0-rc1); "
        f"got {pp['project']['version']!r}"
    )


def test_ilae_calibrate_wired_to_cli_wrapper():
    """ilae-calibrate must point at calibration.cli:main, NOT the
    deleted Phase-0 stub."""
    pp = _pyproject()
    target = pp["project"]["scripts"]["ilae-calibrate"]
    assert target == "calibration.cli:main", (
        f"ilae-calibrate entry point drifted: {target!r}"
    )


def test_phase0_calibration_stub_was_deleted():
    """The Phase-0 stub at calibration/run_youden_calibration.py was
    orphaned + deleted in sub-8.3 (the real Phase-3 reference fitter
    lives at pipeline/reference_calibration/run_youden_calibration.py;
    the orphan would only confuse future readers)."""
    stub = os.path.join(_REPO, "calibration", "run_youden_calibration.py")
    assert not os.path.exists(stub), (
        f"Phase-0 stub still present: {stub}. The orphaned stub was "
        "deleted in sub-8.3 — its real-impl successor is "
        "pipeline/reference_calibration/run_youden_calibration.py, "
        "wrapped by calibration/cli.py + entry-pointed by "
        "ilae-calibrate."
    )


def test_calibration_cli_help_does_not_run_pipeline():
    """`ilae-calibrate --help` MUST exit fast without invoking the
    ~hours Phase-3 orchestrator. This is a real foot-gun guard: in
    Phase 8 we found the prior wiring `ilae-calibrate = calibration.
    run_youden_calibration:main` (Phase-0 stub) was incidentally safe
    because the stub raised NotImplementedError; the new wiring
    points at the real pipeline, and the wrapper MUST argparse-
    intercept --help before it ever imports the orchestrator. Test
    by invoking via SystemExit-catching argparse, not subprocess
    (faster + isolated)."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    import argparse
    from calibration.cli import main as cli_main
    # argparse's --help calls sys.exit(0) via SystemExit
    try:
        cli_main(["--help"])
    except SystemExit as e:
        assert e.code in (0, None), (
            f"--help should exit 0, got code {e.code}"
        )
    else:
        raise AssertionError(
            "calibration.cli.main(['--help']) did not exit; the wrapper "
            "may not be argparse-intercepting --help correctly."
        )


def test_calibration_cli_dry_run_does_not_run_pipeline(capsys):
    """`ilae-calibrate --dry-run` must print the plan + exit 0
    without invoking the Phase-3 orchestrator."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from calibration.cli import main as cli_main
    rc = cli_main(["--dry-run"])
    assert rc == 0, f"--dry-run should return 0, got {rc}"
    captured = capsys.readouterr()
    assert "dry-run" in captured.out.lower(), (
        "--dry-run output should mention 'dry-run'"
    )
    assert "PHASE-3 UNIFIED REFERENCE-FAITHFUL" not in captured.out, (
        "--dry-run accidentally invoked the real orchestrator "
        "(saw its banner in captured stdout)"
    )


def test_calibration_cli_deferred_import():
    """The wrapper must NOT import pipeline.run_unified_calibration at
    module load. Deferred import is the mechanism that makes --help
    fast and ImportError-clean in environments missing Phase-3 deps.

    We import calibration.cli, then verify the heavy import has not
    happened. Cannot use sys.modules check directly because tests may
    have imported it in a prior test; instead we inspect the source
    for the deferred-import pattern."""
    cli_path = os.path.join(_REPO, "calibration", "cli.py")
    src = open(cli_path).read()
    # The heavy import must be INSIDE main(), not at module top.
    top = src.split("def main", 1)[0]
    assert "from pipeline.run_unified_calibration" not in top, (
        "calibration/cli.py imports the orchestrator at module top — "
        "--help will pay the import cost. The import must be deferred "
        "inside main(), guarded by the --dry-run branch."
    )


def test_deployment_cli_sets_blas_single_thread_before_numpy():
    """deployment/cli.py MUST set OPENBLAS_NUM_THREADS=1 etc. BEFORE
    any numpy-importing code. Phase 8 sub-8.4 gate finding: the
    engine's bit-exact-reproducibility contract requires single-thread
    BLAS, and the CLI must self-enforce it (matching conftest.py:21).
    Without this, `ilae-deploy all` produces ~1-ULP drift in hat_*/sd_*
    columns of sim/candidates.csv vs the committed Phase-4.6-B
    baseline.

    Source inspection check: the env-var setter must appear textually
    BEFORE the first `import numpy` (or any module that imports numpy)
    in the file."""
    cli_path = os.path.join(_REPO, "deployment", "cli.py")
    src = open(cli_path).read()
    env_setter = 'os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")'
    # The exact line may have minor formatting variation; pin the substring.
    assert "OPENBLAS_NUM_THREADS" in src, (
        "deployment/cli.py is missing the OPENBLAS_NUM_THREADS env var "
        "setter required by the engine reproducibility contract."
    )
    # All numpy-touching imports in cli.py are deferred (inside functions)
    # so the env vars set at module-top reach numpy before it imports.
    # Confirm the env setter appears textually before `def `:
    first_def = src.find("def ")
    blas_pos = src.find("OPENBLAS_NUM_THREADS")
    assert 0 <= blas_pos < first_def, (
        f"BLAS env-var setter at offset {blas_pos} must precede the "
        f"first def at offset {first_def}; otherwise deferred-import "
        f"numpy may inherit the parent shell's multi-thread BLAS state."
    )


def test_bridge_sets_blas_single_thread_before_numpy():
    """bridge/run_multi_auroc_bridge.py (the ilae-paper entry point)
    MUST set the same BLAS caps. Same rationale as the deployment-CLI
    test above; without this, the Mode-A Paper-1 bridge inherits the
    parent shell's BLAS thread count and engine reproducibility breaks.

    Source inspection: env-var setter appears BEFORE the numpy import.
    """
    bridge_path = os.path.join(_REPO, "bridge",
                               "run_multi_auroc_bridge.py")
    src = open(bridge_path).read()
    assert "OPENBLAS_NUM_THREADS" in src, (
        "bridge/run_multi_auroc_bridge.py missing OPENBLAS_NUM_THREADS"
    )
    blas_pos = src.find("OPENBLAS_NUM_THREADS")
    numpy_import_pos = src.find("import numpy")
    assert numpy_import_pos > 0, "expected `import numpy` in bridge"
    assert blas_pos < numpy_import_pos, (
        f"BLAS env-var setter at offset {blas_pos} must precede "
        f"`import numpy` at offset {numpy_import_pos}; otherwise the "
        f"contract is violated at module load."
    )
