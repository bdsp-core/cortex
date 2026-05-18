"""Process-pool parallelization determinism + correctness tests.

Pins:
  - configure_blas_single_thread sets all BLAS env caps to "1".
  - resolve_max_workers clamps to [1, cpu-2] and honours requests.
  - parallel_map with 1 worker (serial fallback) == multi-worker, bitwise,
    for a deterministic seeded workload (this is the reproducibility
    guarantee that lets us parallelize the paper-grade campaign).
  - parallel_map preserves task order when ordered=True.
  - Worker exceptions are captured as error sentinels, not silently dropped.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _parallel import (  # noqa: E402
    configure_blas_single_thread,
    resolve_max_workers,
    parallel_map,
    count_errors,
    _THREAD_ENV_VARS,
)


# ── BLAS env configuration ────────────────────────────────────────────

def test_configure_blas_single_thread_sets_all_caps():
    configure_blas_single_thread()
    for var in _THREAD_ENV_VARS:
        assert os.environ[var] == "1", f"{var} not set to 1"


# ── worker-count resolution ───────────────────────────────────────────

def test_resolve_max_workers_clamps_and_defaults():
    import multiprocessing as mp
    cap = max(1, mp.cpu_count() - 2)
    # None → min(14, cap)
    assert resolve_max_workers(None) == min(14, cap)
    # Over-request clamps to cap
    assert resolve_max_workers(9999) == cap
    # Under-request honoured
    assert resolve_max_workers(2) == 2
    # Zero / negative floored to 1
    assert resolve_max_workers(0) == 1
    assert resolve_max_workers(-5) == 1


# ── deterministic workload: parallel == serial, bitwise ───────────────
# Top-level worker (picklable under spawn).

def _seeded_mcmc_like_worker(task):
    """A deterministic numeric workload that mimics a seeded MCMC rep:
    draw from a seeded RNG, do some linear algebra, return summary floats.
    Bitwise-reproducible iff BLAS is single-threaded and the seed is fixed.
    """
    rng = np.random.default_rng(task["seed"])
    A = rng.standard_normal((64, 64))
    # Symmetric PSD matrix; eigvalsh reduction order depends on BLAS threads.
    M = A @ A.T
    w = np.linalg.eigvalsh(M)
    x = rng.standard_normal(64)
    sol = np.linalg.solve(M + np.eye(64), x)
    return {
        "idx": task["idx"],
        "trace": float(np.trace(M)),
        "eig_sum": float(w.sum()),
        "eig_max": float(w.max()),
        "sol_norm": float(np.linalg.norm(sol)),
    }


def _failing_worker(task):
    if task["idx"] == 2:
        raise ValueError("intentional failure on idx=2")
    return {"idx": task["idx"], "ok": True}


def test_parallel_equals_serial_bitwise():
    tasks = [{"idx": i, "seed": 1000 + i} for i in range(12)]
    serial = parallel_map(_seeded_mcmc_like_worker, tasks,
                          max_workers=1, desc="serial", progress_every=100)
    parallel = parallel_map(_seeded_mcmc_like_worker, tasks,
                            max_workers=6, desc="parallel", progress_every=100)
    assert len(serial) == len(parallel) == 12
    for s, p in zip(serial, parallel):
        assert s["idx"] == p["idx"]
        # Bitwise equality (float repr) — single-thread BLAS guarantees it.
        assert s["trace"] == p["trace"], (s, p)
        assert s["eig_sum"] == p["eig_sum"], (s, p)
        assert s["eig_max"] == p["eig_max"], (s, p)
        assert s["sol_norm"] == p["sol_norm"], (s, p)


def test_parallel_map_preserves_order():
    tasks = [{"idx": i, "seed": i} for i in range(20)]
    res = parallel_map(_seeded_mcmc_like_worker, tasks,
                        max_workers=6, desc="order", progress_every=100)
    assert [r["idx"] for r in res] == list(range(20)), \
        "ordered=True must return results in task order"


def test_parallel_map_captures_worker_errors():
    tasks = [{"idx": i} for i in range(5)]
    res = parallel_map(_failing_worker, tasks, max_workers=4,
                        desc="errs", progress_every=100)
    assert len(res) == 5
    assert count_errors(res) == 1
    err = next(r for r in res if isinstance(r, dict) and "__error__" in r)
    assert err["__task_index__"] == 2
    assert "intentional failure" in err["__error__"]
    # Non-failing tasks still returned correctly.
    ok = [r for r in res if isinstance(r, dict) and r.get("ok")]
    assert len(ok) == 4
