"""Entry point: clinical deployment runtime (PI Laplace/EKF simulate_test).

Phase 0 STUB. Real implementation lands in Phase 4 (port PI
scripts/run_deployment_sim.py + simulate_test.py, ROOT-parameterized,
TASKS extended 6 -> 7 per D5/R6). See UNIFIED_REPO_MERGE_PLAN.md Phase 4.
"""

_PHASE = "Phase 4 (deployment integration, K=7)"


def main(seed: int = 0) -> int:  # noqa: ARG001 - signature matches PI run_deployment_sim.main
    raise NotImplementedError(
        f"ilae-deploy is a Phase 0 skeleton stub; implemented in {_PHASE}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
