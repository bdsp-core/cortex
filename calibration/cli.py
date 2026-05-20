"""`ilae-calibrate` — the unified calibration-recompute CLI.

Phase 8 sub-8.3 (2026-05-20): thin argparse wrapper around the
Phase-3 orchestrator at `pipeline.run_unified_calibration:main()`.
Mirrors the `deployment/cli.py` pattern from Phase 4.7.

Why a wrapper (and not `ilae-calibrate = pipeline.run_unified_
calibration:main` directly): the Phase-3 orchestrator takes no
argv and runs the full ~hours-long recompute the moment its `main()`
is invoked, including on `--help`. Wiring it as the entry point
directly would mean `ilae-calibrate --help` silently starts a
multi-hour calibration on a clinician's machine. This wrapper
adds a `--help`-safe argparse layer + a `--dry-run` flag for
exploration, and only invokes the real orchestrator on the explicit
`all` (default) stage.

  ilae-calibrate --help        # fast (this docstring)
  ilae-calibrate --dry-run     # print what would run; exit 0
  ilae-calibrate all           # default: full Phase-3 recompute
  ilae-calibrate               # same as `all`

The Phase-3 orchestrator is fixed-step: it runs steps a→i in order
(build inputs → fit main effects → fit SDT → assemble sdt_fits.csv
→ reconcile matrix → run Youden → spike σ\* → D7 independent panel
→ emit cert_config v13). There is no useful per-step CLI; this
wrapper does not expose one.
"""
from __future__ import annotations

import argparse
import sys


def _dry_run() -> int:
    print("=== ilae-calibrate --dry-run ===")
    print("Would invoke: pipeline.run_unified_calibration:main()")
    print()
    print("The Phase-3 orchestrator runs fixed steps a–i in order:")
    print("  a. build reference-schema inputs (7 tasks)")
    print("  b. fit Rasch main effects per domain")
    print("  c. fit per-rater probit-lapse SDT")
    print("  d. assemble sdt_fits.csv")
    print("  e. reconcile cross_domain_rater_matrix")
    print("  f. run CV-top-14 two-stage Youden")
    print("  g. spike σ* calibration")
    print("  h. D7 independent-panel sanity")
    print("  i. emit calibration/cert_config.yaml v13")
    print()
    print("Expected wall: ~hours on a single machine. Re-run only when")
    print("the data/labels/ corpus or the reference-fitter code changes.")
    print()
    print("Outputs (canonical):")
    print("  data/engine_inputs/sdt_fits.csv")
    print("  calibration/youden_ell_star.json")
    print("  calibration/cert_config.yaml")
    print()
    print("To run for real:  ilae-calibrate           # default = all")
    print("                  ilae-calibrate all")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ilae-calibrate",
        description="Unified non-circular CV Youden calibration (Phase 3 "
                    "orchestrator). Recomputes the v13 cert_config + "
                    "sdt_fits + youden_ell_star on data/labels/. Expected "
                    "wall: ~hours.")
    p.add_argument("stage", nargs="?", default="all",
                   choices=["all"],
                   help="pipeline stage (default: all; the Phase-3 "
                        "orchestrator is fixed-step and does not expose "
                        "per-step invocation)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the pipeline plan and exit without running")
    a = p.parse_args(argv)

    if a.dry_run:
        return _dry_run()

    # Defer the heavy import until we actually intend to run — keeps
    # `--help` fast and ImportError-free in environments missing the
    # Phase-3 pipeline-level deps.
    from pipeline.run_unified_calibration import main as _run
    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
