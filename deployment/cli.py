"""`ilae-deploy` — the clinical-deployment pipeline CLI.

Phase 4.7: a thin orchestrator wrapping the (already gated, ported)
deployment stages — `freeze → simulate → plot` — exactly as the plan
specifies. Each stage is a single-purpose module; this layer only
sequences them.

  ilae-deploy            # default: all (freeze → simulate → plot)
  ilae-deploy all        # same
  ilae-deploy freeze     # re-freeze data/deployment_prior from the
                          #   K=7 fits (data/labels/{fits,fits_hier_block})
  ilae-deploy simulate   # run the tiered synthetic panel sim (seed=0)
  ilae-deploy plot       # render the K-agnostic deployment figures

`plot` is BEST-EFFORT inside the pipeline (decision 2026-05-19):
figures are derived viz, NOT gating artifacts, and matplotlib/style
issues must never break the freeze→simulate pipeline that produces the
shipped artifacts. In the `all` pipeline a plot failure logs a warning
and the pipeline still succeeds. The standalone `plot` subcommand
surfaces the error (exit 1) so an explicit run is not silent.
"""
from __future__ import annotations

import argparse
import sys
import traceback


def _freeze() -> int:
    from deployment.freeze_deployment_prior import main as freeze_main
    print("[ilae-deploy] freeze: re-freezing data/deployment_prior …",
          flush=True)
    freeze_main()
    return 0


def _simulate(seed: int = 0, out_dir=None) -> int:
    from deployment.run_deployment_sim import main as sim_main
    print(f"[ilae-deploy] simulate: tiered synthetic panel (seed={seed})…",
          flush=True)
    sim_main(seed=seed, out_dir=out_dir)
    return 0


def _plot(best_effort: bool) -> int:
    """Render the deployment figures. best_effort=True (pipeline use):
    a failure warns + returns 0 so freeze/simulate are not undone by a
    viz problem. best_effort=False (explicit `plot`): error → exit 1."""
    try:
        from deployment.plot_deploy import main as plot_main
        print("[ilae-deploy] plot: rendering deployment figures …",
              flush=True)
        plot_main()
        return 0
    except Exception as exc:                      # noqa: BLE001
        msg = (f"[ilae-deploy] plot FAILED ({type(exc).__name__}: {exc})")
        if best_effort:
            print(msg + " — BEST-EFFORT: figures are derived viz, not "
                  "gating artifacts; the freeze→simulate pipeline "
                  "SUCCEEDED. Continuing.", file=sys.stderr, flush=True)
            traceback.print_exc()
            return 0
        print(msg, file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ilae-deploy",
        description="Clinical-deployment pipeline: freeze → simulate "
                    "→ plot (K=7, v13 ℓ*, decoupled RNG).")
    p.add_argument("stage", nargs="?", default="all",
                   choices=["all", "freeze", "simulate", "plot"],
                   help="pipeline stage (default: all)")
    p.add_argument("--seed", type=int, default=0,
                   help="simulate seed (default 0)")
    p.add_argument("--out-dir", default=None,
                   help="simulate output dir (default: the canonical "
                        "data/deployment_prior/sim)")
    a = p.parse_args(argv)

    if a.stage == "freeze":
        return _freeze()
    if a.stage == "simulate":
        return _simulate(a.seed, a.out_dir)
    if a.stage == "plot":
        return _plot(best_effort=False)           # explicit → fatal
    # all: freeze → simulate → plot(best-effort)
    rc = _freeze()
    if rc:
        return rc
    rc = _simulate(a.seed, a.out_dir)
    if rc:
        return rc
    return _plot(best_effort=True)                 # pipeline → non-fatal


if __name__ == "__main__":
    raise SystemExit(main())
