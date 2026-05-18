"""Minimal pytest-substitute runner for environments without pytest installed.

Discovers `test_*` functions in all `tests/test_*.py` modules, applies the
fixture machinery for the small set of fixtures we use (`tmp_results_dir`,
`synthetic_true_params`, `cert_config`), and runs each test in isolation.

Honors:
  - module-level `pytestmark = pytest.mark.slow`        → SKIP unless --slow
  - per-test marker via attribute (set by `pytest.mark.slow`) → same

Usage (from project root):
  python tests/_run_all.py            # fast tests only
  python tests/_run_all.py --slow     # include slow tests
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import os
import sys
import time
import traceback
import types
import warnings


# ---------------------------------------------------------------------------
# Fake `pytest` shim so test modules that `import pytest` keep working.
# Provides only the attributes we actually use:
#   - pytest.mark.slow             (no-op marker that we recognise)
#   - pytest.fixture               (decorator; we handle our few fixtures by name)
#   - pytest.importorskip(name)    (raise SkipTest if missing)
#   - pytest.skip(msg)             (raise SkipTest)
# ---------------------------------------------------------------------------

class SkipTest(Exception):
    pass


class _Marker:
    def __init__(self, name):
        self.name = name

    def __call__(self, fn):
        existing = getattr(fn, "_pytest_marks", set())
        existing.add(self.name)
        fn._pytest_marks = existing
        return fn


class _MarkNamespace:
    def __getattr__(self, name):
        return _Marker(name)


def _fixture(*args, **kwargs):
    """Decorator: just tag the function as a fixture so we can find it later."""
    def decorator(fn):
        fn._is_fixture = True
        return fn
    if args and callable(args[0]):
        # used as bare @pytest.fixture
        return decorator(args[0])
    return decorator


def _importorskip(name):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise SkipTest(f"missing dependency: {name} ({e})")


def _skip(msg=""):
    raise SkipTest(msg)


_pytest_shim = types.ModuleType("pytest")
_pytest_shim.mark = _MarkNamespace()
_pytest_shim.fixture = _fixture
_pytest_shim.importorskip = _importorskip
_pytest_shim.skip = _skip
_pytest_shim.SkipTest = SkipTest
sys.modules["pytest"] = _pytest_shim


# ---------------------------------------------------------------------------
# Setup project path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixture resolution: hard-coded to the small set we use in conftest.py
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402  (after sys.modules['pytest'] set)


def _fx_tmp_results_dir():
    import tempfile
    return tempfile.mkdtemp(prefix="ilae_test_")


def _fx_synthetic_true_params():
    def _make(K=3, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "l_true": rng.normal(0.4, 0.3, size=K),
            "t_true": rng.normal(0.0, 0.5, size=K),
        }
    return _make


def _fx_cert_config():
    try:
        import yaml
    except ImportError:
        raise SkipTest("PyYAML missing")
    cfg_path = os.path.join(_PROJECT_ROOT, "cert_config.yaml")
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)


_FIXTURES = {
    "tmp_results_dir": _fx_tmp_results_dir,
    "synthetic_true_params": _fx_synthetic_true_params,
    "cert_config": _fx_cert_config,
}


def _resolve_args(fn):
    sig = inspect.signature(fn)
    args = []
    for name in sig.parameters:
        if name not in _FIXTURES:
            raise RuntimeError(f"unknown fixture {name!r} requested by {fn.__name__}")
        args.append(_FIXTURES[name]())
    return args


# ---------------------------------------------------------------------------
# Discovery & execution
# ---------------------------------------------------------------------------

def _discover_tests(tests_dir):
    out = []
    for name in sorted(os.listdir(tests_dir)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        path = os.path.join(tests_dir, name)
        mod_name = name[:-3]  # strip .py
        out.append((mod_name, path))
    return out


def _module_marks(mod):
    pm = getattr(mod, "pytestmark", None)
    if pm is None:
        return set()
    if isinstance(pm, _Marker):
        return {pm.name}
    if isinstance(pm, (list, tuple)):
        return {m.name for m in pm if isinstance(m, _Marker)}
    return set()


def main():
    include_slow = "--slow" in sys.argv
    only_module = None
    for arg in sys.argv[1:]:
        if arg.startswith("--module="):
            only_module = arg.split("=", 1)[1]

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    discovered = _discover_tests(tests_dir)

    n_total = n_passed = n_failed = n_skipped = 0
    failures = []
    t_start = time.time()

    for mod_name, path in discovered:
        if only_module and mod_name != only_module:
            continue
        try:
            mod = _load_module(path, mod_name)
        except SkipTest as e:
            print(f"SKIP module {mod_name}: {e}")
            continue
        except Exception:
            print(f"\n!! FAILED to import {mod_name}:")
            traceback.print_exc()
            n_failed += 1
            failures.append((mod_name, "import", traceback.format_exc()))
            continue

        mod_marks = _module_marks(mod)

        for tname, tfn in inspect.getmembers(mod, inspect.isfunction):
            if not tname.startswith("test_"):
                continue
            if tfn.__module__ != mod_name:
                continue
            marks = mod_marks | getattr(tfn, "_pytest_marks", set())
            if "slow" in marks and not include_slow:
                n_skipped += 1
                print(f"  SKIP (slow) {mod_name}::{tname}")
                continue
            n_total += 1
            t0 = time.time()
            try:
                args = _resolve_args(tfn)
                tfn(*args)
            except SkipTest as e:
                n_skipped += 1
                print(f"  SKIP {mod_name}::{tname}: {e}")
                continue
            except AssertionError:
                dt = time.time() - t0
                tb = traceback.format_exc()
                print(f"  FAIL {mod_name}::{tname}  ({dt:.2f}s)")
                print(tb)
                failures.append((mod_name, tname, tb))
                n_failed += 1
                continue
            except Exception:
                dt = time.time() - t0
                tb = traceback.format_exc()
                print(f"  ERROR {mod_name}::{tname}  ({dt:.2f}s)")
                print(tb)
                failures.append((mod_name, tname, tb))
                n_failed += 1
                continue
            dt = time.time() - t0
            print(f"  PASS {mod_name}::{tname}  ({dt:.2f}s)")
            n_passed += 1

    t_total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Total: {n_total} run, {n_passed} passed, {n_failed} failed, "
          f"{n_skipped} skipped  ({t_total:.1f}s)")
    if failures:
        print("\nFAILED:")
        for mod_name, tname, _ in failures:
            print(f"  - {mod_name}::{tname}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
