import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "deploy" / "scripts" / "release_switch.sh"


def _release(root: Path, name: str) -> Path:
    release = root / "releases" / name
    (release / "services" / "api").mkdir(parents=True)
    (release / "apps" / "web" / "dist").mkdir(parents=True)
    (release / "services" / "api" / "app.py").write_text("# test\n")
    (release / "apps" / "web" / "dist" / "index.html").write_text("test\n")
    (release / "RELEASE").write_text(f"{name}\n")
    return release


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CORTEX_APP_ROOT": str(root),
            "CORTEX_RELEASE_TEST_MODE": "1",
        },
    )


def test_first_activation_preserves_legacy_and_rollback_is_atomic(tmp_path):
    live = tmp_path / "cortex_web"
    (live / "services" / "api").mkdir(parents=True)
    (live / "apps" / "web" / "dist").mkdir(parents=True)
    (live / "services" / "api" / "app.py").write_text("# legacy\n")
    (live / "apps" / "web" / "dist" / "index.html").write_text("legacy\n")
    (live / "RELEASE").write_text("legacy\n")
    candidate = _release(tmp_path, "candidate")

    activated = _run(tmp_path, "activate", str(candidate))
    assert activated.returncode == 0, activated.stderr
    assert live.is_symlink()
    assert live.resolve() == candidate
    previous = Path((tmp_path / "release-state" / "previous").read_text().strip())
    assert previous.parent == tmp_path / "releases"
    assert (previous / "RELEASE").read_text() == "legacy\n"

    rolled_back = _run(tmp_path, "rollback")
    assert rolled_back.returncode == 0, rolled_back.stderr
    assert live.resolve() == previous
    assert Path((tmp_path / "release-state" / "previous").read_text().strip()) == candidate


def test_activation_rejects_a_tree_outside_release_root(tmp_path):
    (tmp_path / "releases").mkdir()
    live = tmp_path / "cortex_web"
    live.mkdir()
    outside = tmp_path / "outside"
    (outside / "services" / "api").mkdir(parents=True)
    (outside / "apps" / "web" / "dist").mkdir(parents=True)
    (outside / "services" / "api" / "app.py").write_text("# test\n")
    (outside / "apps" / "web" / "dist" / "index.html").write_text("test\n")
    (outside / "RELEASE").write_text("outside\n")

    rejected = _run(tmp_path, "activate", str(outside))
    assert rejected.returncode == 2
    assert "outside" in rejected.stderr
    assert not live.is_symlink()
