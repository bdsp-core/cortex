"""Shared process-pool parallelization utility for compute-heavy scripts.

Designed for the Apple-Silicon M3 Max (12 performance + 4 efficiency cores,
128 GB RAM).  The MCMC repetitions are embarrassingly parallel — each rep
is independently seeded and shares no mutable state — so a
ProcessPoolExecutor over reps gives near-linear speedup until core
saturation.

CRITICAL — BLAS oversubscription
--------------------------------
NumPy on Apple Silicon links Accelerate/vecLib (and conda builds may link
OpenBLAS / MKL).  If each of 14 worker processes also spawns 16 BLAS
threads, the machine is 14×16 = 224-way oversubscribed and runs SLOWER
than serial.  We force every BLAS backend to a single thread per process
via environment variables.  These MUST be set *before* NumPy is imported.

Because macOS uses the "spawn" start method, each worker re-imports the
target module from scratch; the parent's `os.environ` is inherited by the
spawned child, so setting the vars once in the parent before pool creation
is sufficient.  We additionally pass an `initializer` as defence in depth.

REPRODUCIBILITY
---------------
Single-thread BLAS also makes results bitwise-deterministic: multi-threaded
BLAS reductions sum in nondeterministic order.  With single-thread BLAS and
per-rep seeding (`seed_base + rep_id`), parallel output is identical to
serial output.  `tests/test_parallel_determinism.py` pins this.
"""
from __future__ import annotations

import os

# ── BLAS thread caps — set BEFORE any numpy import anywhere in the process ──
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",   # Apple Accelerate / vecLib (Apple Silicon)
    "NUMEXPR_NUM_THREADS",
)


def configure_blas_single_thread() -> None:
    """Force every BLAS backend to one thread per process.

    Call this at the very top of any script (before `import numpy`) AND
    pass it as the ProcessPoolExecutor `initializer`.
    """
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"


# Apply immediately on import so that `from _parallel import ...` at the top
# of a script (before numpy) configures the parent process too.
configure_blas_single_thread()


import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any, Callable, Iterable, List, Optional, Sequence


DEFAULT_MAX_WORKERS = 14   # leave 2 logical cores for OS + main process


def resolve_max_workers(requested: Optional[int] = None) -> int:
    """Clamp the requested worker count to a safe range for this machine.

    Defaults to min(DEFAULT_MAX_WORKERS, cpu_count - 2); never returns < 1.
    """
    import multiprocessing as mp

    n_cpu = mp.cpu_count()
    cap = max(1, n_cpu - 2)
    if requested is None:
        return min(DEFAULT_MAX_WORKERS, cap)
    return max(1, min(int(requested), cap))


def parallel_map(
    worker_fn: Callable[[Any], Any],
    tasks: Sequence[Any],
    *,
    max_workers: Optional[int] = None,
    desc: str = "tasks",
    ordered: bool = True,
    progress_every: int = 10,
    serial_fallback_on_one_worker: bool = True,
) -> List[Any]:
    """Run `worker_fn` over `tasks` in a spawn-based process pool.

    Parameters
    ----------
    worker_fn : callable
        A *top-level* (module-level, picklable) function taking one task
        descriptor and returning a picklable result.  Closures and lambdas
        will fail under the spawn start method.
    tasks : sequence
        Self-contained task descriptors.  Each must carry everything the
        worker needs (including its own RNG seed) — workers share no state.
    max_workers : int, optional
        Worker process count.  Defaults to `resolve_max_workers()`.
    desc : str
        Label used in progress lines.
    ordered : bool
        If True (default), results are returned in the same order as
        `tasks` (essential for reproducibility / joining back to inputs).
        If False, returned in completion order (faster first-result).
    progress_every : int
        Emit a progress line every N completed tasks.
    serial_fallback_on_one_worker : bool
        If the resolved worker count is 1, run serially in-process (avoids
        pool overhead and simplifies debugging / determinism tests).

    Returns
    -------
    list
        Results.  If `ordered`, `result[i]` corresponds to `tasks[i]`.
    """
    n = len(tasks)
    workers = resolve_max_workers(max_workers)
    t0 = time.time()

    if workers == 1 and serial_fallback_on_one_worker:
        print(f"[parallel] running {n} {desc} serially (1 worker)", flush=True)
        out = []
        for i, task in enumerate(tasks):
            out.append(worker_fn(task))
            if (i + 1) % progress_every == 0 or (i + 1) == n:
                _emit_progress(desc, i + 1, n, t0)
        return out

    print(f"[parallel] running {n} {desc} on {workers} workers "
          f"(spawn; single-thread BLAS)", flush=True)

    ctx = get_context("spawn")
    results: List[Any] = [None] * n
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=configure_blas_single_thread,
    ) as ex:
        future_to_idx = {
            ex.submit(worker_fn, task): i for i, task in enumerate(tasks)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:  # noqa: BLE001 — surface worker tracebacks
                tb = traceback.format_exc()
                print(f"[parallel] task {idx} ({desc}) FAILED: {e}\n{tb}",
                      file=sys.stderr, flush=True)
                results[idx] = {"__error__": str(e), "__traceback__": tb,
                                "__task_index__": idx}
            completed += 1
            if completed % progress_every == 0 or completed == n:
                _emit_progress(desc, completed, n, t0)

    if not ordered:
        # Strip Nones (shouldn't be any) and return completion-order copy.
        return [r for r in results if r is not None]
    return results


def _emit_progress(desc: str, done: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta_min = (total - done) / rate / 60 if rate > 0 else float("nan")
    print(f"[parallel] {desc}: {done}/{total} "
          f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta_min:.1f} min",
          flush=True)


def count_errors(results: Iterable[Any]) -> int:
    """Number of result entries that are error sentinels from parallel_map."""
    return sum(1 for r in results
               if isinstance(r, dict) and "__error__" in r)
