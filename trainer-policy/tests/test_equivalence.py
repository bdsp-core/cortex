"""Drift and behavior gates for the root trainer-policy representation."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from trainer_policy import MixedBelief as MirrorBelief
from trainer_policy import Registry as MirrorRegistry
from trainer_policy.api import TrainerManager, TrainerPolicy, TrainerSession
from trainer_policy.deployed import EngineManager, EngineSession
from trainer_policy.policy import LETrainerPolicy as MirrorPolicy


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
MIRROR = PACKAGE_ROOT / "trainer_policy"
PRODUCTION = REPO_ROOT / "cortex_web" / "learning-engine-cleaned"

CORE_FILES = {
    "__init__.py": "learning_engine/__init__.py",
    "learnmodel.py": "learning_engine/learnmodel.py",
    "mixedengine.py": "learning_engine/mixedengine.py",
    "msengine.py": "learning_engine/msengine.py",
    "multimodel.py": "learning_engine/multimodel.py",
    "registry.py": "learning_engine/registry.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unified_public_names_preserve_deployed_class_identity() -> None:
    assert TrainerPolicy is MirrorPolicy
    assert TrainerSession is EngineSession
    assert TrainerManager is EngineManager


def test_core_algorithm_is_byte_exact_with_production() -> None:
    for mirror_name, production_name in CORE_FILES.items():
        assert (MIRROR / mirror_name).read_bytes() == \
            (PRODUCTION / production_name).read_bytes(), mirror_name


def test_frozen_artifact_content_and_source_hash_match_production() -> None:
    mirror = MIRROR / "artifacts" / "nway_dynamics_v1_1.json"
    production = PRODUCTION / "artifacts" / "nway_dynamics_v1_1.json"
    assert json.loads(mirror.read_text()) == json.loads(production.read_text())
    assert _sha(production) == \
        "e44011563bac376bd94c7678c07677b50a1f973cbef71a16b871df84a0ca1765"


def test_policy_diff_is_only_the_package_local_import() -> None:
    mirror = (MIRROR / "policy.py").read_text()
    production = (PRODUCTION / "adapter" / "le_adapter.py").read_text()
    normalized = mirror.replace(
        "from . import MixedBelief, Registry",
        "from learning_engine import MixedBelief, Registry",
    )
    assert normalized == production


def test_deployed_service_diff_is_only_local_import_and_artifact_path() -> None:
    mirror = (MIRROR / "deployed.py").read_text()
    production = (
        REPO_ROOT / "cortex_web" / "services" / "api" / "engine_trainer.py"
    ).read_text()
    local_path = '''# Self-contained reviewer mirror: the deployed artifact is package data.
ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "nway_dynamics_v1_1.json"
)'''
    production_path = '''# The vendored package lives INSIDE cortex_web/ so the standard deploy
# (which rsyncs cortex_web/ only) ships it to the box; the repo-root
# location is the pre-integration fallback for older checkouts.
_CW = Path(__file__).resolve().parents[2]          # cortex_web/
_PKG = _CW / "learning-engine-cleaned"
if not _PKG.exists():
    _PKG = _CW.parent / "learning-engine-cleaned"
ARTIFACT_PATH = _PKG / "artifacts" / "nway_dynamics_v1_1.json"'''
    local_engine = '''def _engine():
    """Lazy, cached import of the numeric stack and local policy package."""
    global _ENG
    if _ENG is None:
        import numpy as np
        from . import MixedBelief, Registry
        from .policy import LETrainerPolicy
        _ENG = (np, MixedBelief, Registry, LETrainerPolicy)
    return _ENG'''
    production_engine = '''def _engine():
    """Lazy, cached import of the numeric stack + vendored package."""
    global _ENG
    if _ENG is None:
        import sys
        if str(_PKG) not in sys.path:
            sys.path.insert(0, str(_PKG))
        import numpy as np
        from learning_engine import MixedBelief, Registry
        from adapter.le_adapter import LETrainerPolicy
        _ENG = (np, MixedBelief, Registry, LETrainerPolicy)
    return _ENG'''
    assert local_path in mirror and local_engine in mirror
    normalized = mirror.replace(local_path, production_path)
    normalized = normalized.replace(local_engine, production_engine)
    assert normalized == production


def test_runtime_modules_do_not_import_cortex_web_or_legacy_package_names() -> None:
    forbidden = ("cortex_web", "learning_engine", "adapter")
    for path in MIRROR.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in names for prefix in forbidden
            ), f"host import in {path.name}: {names}"


def test_session_import_surface_is_jax_free() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    code = (
        "import sys\n"
        "from trainer_policy import MixedBelief, Registry\n"
        "from trainer_policy.policy import LETrainerPolicy\n"
        "import trainer_policy.deployed\n"
        "assert 'jax' not in sys.modules\n"
        "assert 'numpyro' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _small_artifact() -> dict:
    codes = ["d0", "d1"]
    return {
        "codes": codes,
        "alpha_t": [0.05, 0.04],
        "alpha_s": [0.02, 0.015],
        "lam": 0.02,
        "q_t": 0.0,
        "q_s": 0.006,
        "state_prior": {
            "mu_t0": [0.0, 0.1],
            "tau_t0": [0.3, 0.35],
            "mu_u0": [0.8, 0.7],
            "tau_u0": [0.3, 0.3],
            "gamma": [0.4, 0.45],
        },
        "floor_prior": ([-1.2, -1.1], [0.25, 0.3]),
    }


def _candidate_map() -> dict[str, dict]:
    out = {}
    for domain_index, code in enumerate(("d0", "d1")):
        signals = np.concatenate((
            np.linspace(0.15, 2.2, 40),
            -np.linspace(0.15, 2.2, 40),
        ))
        out[code] = {
            "seg_id": np.arange(80, dtype=np.int64) + 1000 * domain_index,
            "s": signals,
            "s_sd": np.zeros(80),
            "y_star": (signals > 0).astype(np.int64),
        }
    return out


def test_reusable_policy_matches_production_choice_and_belief() -> None:
    production_engine = importlib.import_module("learning_engine")
    production_adapter = importlib.import_module("adapter.le_adapter")
    artifact = _small_artifact()
    candidates = _candidate_map()

    prod_registry = production_engine.Registry([(c, "binary") for c in artifact["codes"]])
    mirror_registry = MirrorRegistry([(c, "binary") for c in artifact["codes"]])
    prod_belief = production_engine.MixedBelief(
        artifact, prod_registry, N=160, rng=np.random.default_rng(71)
    )
    mirror_belief = MirrorBelief(
        artifact, mirror_registry, N=160, rng=np.random.default_rng(71)
    )
    cuts = {c: 99.0 for c in artifact["codes"]}
    prod_policy = production_adapter.LETrainerPolicy(
        prod_belief,
        prod_registry,
        cuts,
        candidates.get,
        alpha=0.0,
        Z=2.0,
        sd_floor=0.23,
        alloc="thompson",
        share_cap=0.65,
        rng=np.random.default_rng(72),
    )
    mirror_policy = MirrorPolicy(
        mirror_belief,
        mirror_registry,
        cuts,
        candidates.get,
        alpha=0.0,
        Z=2.0,
        sd_floor=0.23,
        alloc="thompson",
        share_cap=0.65,
        rng=np.random.default_rng(72),
    )

    for question in range(18):
        prod_choice = prod_policy.step()
        mirror_choice = mirror_policy.step()
        assert mirror_choice == prod_choice
        response = int(prod_choice["y_star"] if question % 4 else 1 - prod_choice["y_star"])
        prod_policy.record(prod_choice, response)
        mirror_policy.record(mirror_choice, response)
        for field in ("t", "u", "u_inf", "w", "a_t", "a_s"):
            assert np.array_equal(
                getattr(mirror_belief, field), getattr(prod_belief, field)
            ), (question, field)


class _FakeDB:
    def __init__(self):
        self.retention = {}

    def get_retention(self, _code):
        return list(self.retention.values())

    def latest_result_for_code(self, _code):
        return None

    def session_trials(self, _session_id):
        return []

    def _fetchall(self, _query, _params):
        return []

    def upsert_retention(self, _code, domain, next_due_utc, interval_s):
        self.retention[domain] = {
            "domain": domain,
            "next_due_utc": next_due_utc,
            "interval_s": interval_s,
        }


class _FakeBank:
    def __init__(self):
        codes = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
        words = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]
        self.engine = {
            "taskCodes": codes,
            "taskPatternWords": words,
            "ellStar": [0.0] * len(codes),
        }
        segments = []
        seg_id = 1
        for index in range(16):
            s_mean = [(-1.8 + 0.24 * index)] + [0.0] * 6
            segments.append({
                "segId": seg_id,
                "patternClass": "spike",
                "applicableTaskIdx": [0],
                "sMean": s_mean,
                "sSd": [0.0] * 7,
            })
            seg_id += 1
        for gold in range(1, 7):
            for index in range(8):
                row = [-2.0] + [(-1.1 + 0.27 * index) for _ in range(6)]
                row[gold] += 1.2
                segments.append({
                    "segId": seg_id,
                    "patternClass": words[gold],
                    "applicableTaskIdx": [1, 2, 3, 4, 5, 6],
                    "sMean": row,
                    "sSd": [0.0] * 7,
                })
                seg_id += 1
        self.segments = segments
        self._by_id = {row["segId"]: row for row in segments}


def test_deployed_session_policy_matches_production() -> None:
    production = importlib.import_module("cortex_web.services.api.engine_trainer")
    mirror = importlib.import_module("trainer_policy.deployed")
    bank = _FakeBank()
    seg_ids = [row["segId"] for row in bank.segments]
    prod_session = production.EngineSession(
        _FakeDB(), bank, "parity-sitting", "participant", seg_ids,
        alloc="thompson",
    )
    mirror_session = mirror.EngineSession(
        _FakeDB(), bank, "parity-sitting", "participant", seg_ids,
        alloc="thompson",
    )

    assert mirror_session.attain0 == prod_session.attain0
    assert mirror_session.seed_unique == prod_session.seed_unique
    assert mirror_session.snapshot() == prod_session.snapshot()
    for _ in range(12):
        prod_item = prod_session.next_item()
        mirror_item = mirror_session.next_item()
        assert mirror_item == prod_item
        if prod_item is None:
            break
        prod_session.record(prod_item["segId"], prod_item["yStar"])
        mirror_session.record(mirror_item["segId"], mirror_item["yStar"])
        assert mirror_session.snapshot() == prod_session.snapshot()
