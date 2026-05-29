"""Phase 9 Layer-1 — variant comparison orchestrator.

Spawns `fit_joint_k7.py` as subprocess workers, one per GPU, iterating
(task × variant) pairs:

  IIIC variant comparison (default):  6 IIIC tasks × 3 variants = 18 fits
  Full K=7:                           7 tasks × 1 winning variant = 7 fits

Each worker is a separate Python process so JAX's per-process device
allocation is clean (one GPU per process; CUDA_VISIBLE_DEVICES + the
XLA_PYTHON_CLIENT_MEM_FRACTION set inside fit_joint_k7).

Concurrency = number of GPUs (default 2). At Eli's "use all GPUs +
90 % CPUs" directive, this saturates both A4500s while leaving CPU
headroom for I/O + the SVI gauge invariance check.

Two modes:
  --mode variant_comparison  (default): 6 IIIC × 3 variants on SVI;
                                        then runs variant_selection_k7
                                        to emit variant_selection_k7.json
  --mode winner_full         : 7 tasks × 1 variant (the WINNER from
                                        variant_selection_k7.json) on SVI
                                        AND NUTS-validate

  --mode smoke               : 1 task × 3 variants in --smoke mode
                                        (used to verify the orchestrator
                                        end-to-end without compute spend)

Outputs flow through `calibration/joint/{task}_k7_{variant}_*.npz/.json`
and a per-run orchestrator log at `calibration/joint/orchestrator_log.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "calibration" / "joint"

IIIC_TASKS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
SPIKE_TASK = "spike"
ALL_TASKS = [SPIKE_TASK] + IIIC_TASKS
VARIANTS = ["A", "B", "C"]


@dataclass
class FitSpec:
    task: str
    variant: str
    method: str  # "svi" or "nuts"
    extra_args: tuple = ()


def _format_spec(spec: FitSpec) -> str:
    return f"{spec.task} V_{spec.variant} {spec.method}"


def _run_one_fit(spec: FitSpec, gpu: int, smoke: bool,
                 svi_steps: int, warmup: int, samples: int, chains: int,
                 subsample: int | None, seed: int = 0) -> dict:
    """Invoke fit_joint_k7.py as a subprocess pinned to one GPU.
    Returns a dict with task, variant, method, gpu, wall_s, returncode, stdout_tail."""
    cmd = [
        sys.executable, "-m", "pipeline.joint_calibration.fit_joint_k7",
        "--task", spec.task,
        "--variant", spec.variant,
        "--method", spec.method,
        "--gpu", str(gpu),
        "--seed", str(seed),
    ]
    if smoke:
        cmd.append("--smoke")
    else:
        cmd.extend(["--svi-steps", str(svi_steps)])
        if spec.method == "nuts":
            cmd.extend(["--warmup", str(warmup), "--samples", str(samples),
                        "--chains", str(chains)])
        if subsample is not None:
            cmd.extend(["--subsample", str(subsample)])
    cmd.extend(spec.extra_args)

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(REPO), capture_output=True, text=True,
        env={**os.environ},  # inherit; child sets its own CUDA_VISIBLE_DEVICES
    )
    wall_s = time.time() - t0
    return {
        "task": spec.task,
        "variant": spec.variant,
        "method": spec.method,
        "gpu": gpu,
        "wall_s": round(wall_s, 1),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "").splitlines()[-5:],
        "stderr_tail": (proc.stderr or "").splitlines()[-5:],
        "cmd": cmd,
    }


def _run_parallel(specs: list[FitSpec], n_gpus: int, smoke: bool,
                  svi_steps: int, warmup: int, samples: int, chains: int,
                  subsample: int | None, seed: int = 0) -> list[dict]:
    """Run a list of (task, variant, method) specs in parallel across
    n_gpus subprocess workers. Each worker pulls the next spec from the
    shared queue; GPU assignment rotates worker-id → gpu_id."""
    results: list[dict] = []

    # ProcessPoolExecutor with n_gpus workers; each worker runs one fit
    # at a time, on its assigned GPU. We assign GPU = worker_idx % n_gpus.
    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        futures: list[Future] = []
        for i, spec in enumerate(specs):
            gpu = i % n_gpus
            futures.append(pool.submit(
                _run_one_fit, spec, gpu, smoke,
                svi_steps, warmup, samples, chains, subsample, seed))
        for fut in futures:
            r = fut.result()
            results.append(r)
            status = "ok" if r["returncode"] == 0 else "FAIL"
            print(f"  [{status}] {r['task']:>5} V_{r['variant']} "
                  f"{r['method']:>4} gpu={r['gpu']} wall={r['wall_s']}s",
                  flush=True)
            if r["returncode"] != 0:
                for line in r["stderr_tail"]:
                    print(f"      stderr: {line}", flush=True)
    return results


def mode_variant_comparison(n_gpus: int, smoke: bool,
                            svi_steps: int, subsample: int | None,
                            seed: int) -> dict:
    """Run 6 IIIC × 3 variants on SVI; then invoke variant_selection_k7."""
    specs = [FitSpec(task=t, variant=v, method="svi")
             for t in IIIC_TASKS for v in VARIANTS]
    print(f"=== Mode: variant_comparison (6 IIIC × 3 variants = {len(specs)} fits) ===",
          flush=True)
    print(f"  n_gpus={n_gpus}  smoke={smoke}  svi_steps={svi_steps}  "
          f"subsample={subsample}", flush=True)
    t0 = time.time()
    results = _run_parallel(specs, n_gpus, smoke, svi_steps,
                            0, 0, 0, subsample, seed)
    wall_s = time.time() - t0
    print(f"=== variant_comparison fits complete in {wall_s:.1f}s ===",
          flush=True)

    n_failed = sum(1 for r in results if r["returncode"] != 0)
    if n_failed:
        print(f"!!! {n_failed} of {len(results)} fits failed !!!", flush=True)
        return {"results": results, "wall_s": wall_s,
                "n_failed": n_failed, "selection": None}

    # Run gate harness
    print("=== Running variant_selection_k7 gates ===", flush=True)
    from . import variant_selection_k7
    report = variant_selection_k7.select_variants()
    print(f"GLOBAL WINNER: {report['global_winner']}", flush=True)
    print(f"REASON: {report.get('global_winner_reason')}", flush=True)
    return {"results": results, "wall_s": wall_s, "n_failed": 0,
            "selection": report}


def mode_winner_full(winner: str, n_gpus: int, smoke: bool, svi_steps: int,
                     warmup: int, samples: int, chains: int,
                     subsample: int | None, seed: int) -> dict:
    """Run 7 tasks × 1 variant (the winner) on SVI; then NUTS-validate."""
    if winner not in VARIANTS:
        raise ValueError(f"winner {winner!r} must be one of {VARIANTS}")
    svi_specs = [FitSpec(task=t, variant=winner, method="svi")
                 for t in ALL_TASKS]
    nuts_specs = [FitSpec(task=t, variant=winner, method="nuts")
                  for t in ALL_TASKS]
    print(f"=== Mode: winner_full (V_{winner}; 7 SVI + 7 NUTS = {len(svi_specs)+len(nuts_specs)} fits) ===",
          flush=True)
    t0 = time.time()
    svi_results = _run_parallel(svi_specs, n_gpus, smoke, svi_steps,
                                0, 0, 0, subsample, seed)
    nuts_results = _run_parallel(nuts_specs, n_gpus, smoke, svi_steps,
                                 warmup, samples, chains, subsample, seed)
    wall_s = time.time() - t0
    return {"svi_results": svi_results, "nuts_results": nuts_results,
            "wall_s": wall_s, "winner": winner}


def mode_smoke(n_gpus: int) -> dict:
    """1 task × 3 variants in --smoke mode (verifies orchestrator)."""
    specs = [FitSpec(task="iic", variant=v, method="svi") for v in VARIANTS]
    print(f"=== Mode: smoke (iic × 3 variants, --smoke) ===", flush=True)
    t0 = time.time()
    results = _run_parallel(specs, n_gpus, smoke=True,
                            svi_steps=4000, warmup=0, samples=0, chains=0,
                            subsample=None, seed=0)
    wall_s = time.time() - t0
    return {"results": results, "wall_s": wall_s}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["variant_comparison", "winner_full", "smoke"])
    ap.add_argument("--n-gpus", type=int, default=2,
                    help="number of GPUs to use concurrently (default 2)")
    ap.add_argument("--smoke", action="store_true",
                    help="passthrough --smoke to fit_joint_k7 (small subsample)")
    ap.add_argument("--svi-steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--subsample", type=int, default=None,
                    help="cap N per fit (None = full corpus per task)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--winner", default=None,
                    help="winner variant for --mode winner_full (or read "
                         "from variant_selection_k7.json if absent)")
    args = ap.parse_args(argv)

    if args.mode == "variant_comparison":
        out = mode_variant_comparison(
            n_gpus=args.n_gpus, smoke=args.smoke,
            svi_steps=args.svi_steps, subsample=args.subsample,
            seed=args.seed)
    elif args.mode == "winner_full":
        winner = args.winner
        if winner is None:
            sel_path = JOINT / "variant_selection_k7.json"
            if not sel_path.exists():
                raise FileNotFoundError(
                    f"--winner not given and {sel_path} absent. Run "
                    "--mode variant_comparison first.")
            sel = json.loads(sel_path.read_text())
            winner = sel.get("global_winner")
            if winner is None:
                raise RuntimeError(
                    "variant_selection_k7.json has no global_winner; "
                    "human review required.")
            print(f"[winner_full] reading winner={winner!r} from {sel_path}",
                  flush=True)
        out = mode_winner_full(
            winner=winner, n_gpus=args.n_gpus, smoke=args.smoke,
            svi_steps=args.svi_steps, warmup=args.warmup,
            samples=args.samples, chains=args.chains,
            subsample=args.subsample, seed=args.seed)
    elif args.mode == "smoke":
        out = mode_smoke(n_gpus=args.n_gpus)
    else:
        raise ValueError(args.mode)

    log_path = JOINT / "orchestrator_log.json"
    log_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"orchestrator log → {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
