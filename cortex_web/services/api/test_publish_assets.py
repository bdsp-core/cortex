import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "deploy" / "scripts" / "publish_assets.sh"


def _asset(tree: Path, name: str, content: str) -> None:
    assets = tree / "apps" / "web" / "dist" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / name).write_text(content)


def _run(root: Path, candidate: Path | None = None) -> subprocess.CompletedProcess[str]:
    args = [str(SCRIPT)]
    if candidate is not None:
        args.append(str(candidate))
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CORTEX_APP_ROOT": str(root)},
    )


def test_publish_backfills_retained_releases_and_candidate(tmp_path):
    old = tmp_path / "releases" / "old"
    candidate = tmp_path / "releases" / "candidate"
    _asset(old, "worker-old.js", "old worker")
    _asset(old, "worker-old.js.br", "old compressed worker")
    _asset(old, "worker-old.js.map", "private source map")
    _asset(old, "worker-old.js.map.br", "private compressed source map")
    _asset(candidate, "worker-new.js", "new worker")

    result = _run(tmp_path, candidate)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "assets" / "worker-old.js").read_text() == "old worker"
    assert (tmp_path / "assets" / "worker-old.js.br").read_text() == "old compressed worker"
    assert (tmp_path / "assets" / "worker-new.js").read_text() == "new worker"
    assert not (tmp_path / "assets" / "worker-old.js.map").exists()
    assert not (tmp_path / "assets" / "worker-old.js.map.br").exists()


def test_publish_is_idempotent_but_rejects_hash_collisions(tmp_path):
    release = tmp_path / "releases" / "release"
    _asset(release, "worker-same-hash.js", "first bytes")

    assert _run(tmp_path, release).returncode == 0
    assert _run(tmp_path, release).returncode == 0

    (release / "apps" / "web" / "dist" / "assets" / "worker-same-hash.js").write_text(
        "different bytes"
    )
    collision = _run(tmp_path, release)
    assert collision.returncode == 2
    assert "content-hash collision" in collision.stderr
