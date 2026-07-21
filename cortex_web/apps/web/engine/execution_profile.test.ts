import { describe, expect, it } from "vitest";

import { selectExecutionProfile } from "./execution_profile";

describe("selectExecutionProfile", () => {
  it("keeps the rollout disabled unless dual-branch execution is requested", () => {
    expect(selectExecutionProfile({
      requested: "serial",
      policy: "precision_v1",
      hardwareConcurrency: 48,
      workerAvailable: true,
    })).toEqual({
      mode: "serial",
      computeWorkers: 1,
      reason: "rollout_disabled",
    });
  });

  it("never parallelizes the AD6 rollback path", () => {
    expect(selectExecutionProfile({
      requested: "dual_branch_auto",
      policy: "ad6",
      hardwareConcurrency: 48,
      workerAvailable: true,
    })).toEqual({
      mode: "serial",
      computeWorkers: 1,
      reason: "policy_not_parallelized",
    });
  });

  it("falls back safely when workers or reliable concurrency are unavailable", () => {
    expect(selectExecutionProfile({
      requested: "dual_branch_auto",
      policy: "precision_v1",
      hardwareConcurrency: 8,
      workerAvailable: false,
    }).reason).toBe("worker_unavailable");

    for (const hardwareConcurrency of [undefined, 0, Number.NaN]) {
      expect(selectExecutionProfile({
        requested: "dual_branch_auto",
        policy: "precision_v1",
        hardwareConcurrency,
        workerAvailable: true,
      }).reason).toBe("concurrency_unknown");
    }

    for (const hardwareConcurrency of [1, 2, 3]) {
      expect(selectExecutionProfile({
        requested: "dual_branch_auto",
        policy: "precision_v1",
        hardwareConcurrency,
        workerAvailable: true,
      })).toEqual({
        mode: "serial",
        computeWorkers: 1,
        reason: "low_concurrency",
      });
    }
  });

  it("scales ranked outcome workers while reserving device headroom", () => {
    const cases = [
      { hardwareConcurrency: 4, computeWorkers: 2 },
      { hardwareConcurrency: 5, computeWorkers: 3 },
      { hardwareConcurrency: 7, computeWorkers: 3 },
      { hardwareConcurrency: 8, computeWorkers: 5 },
      { hardwareConcurrency: 11, computeWorkers: 5 },
      { hardwareConcurrency: 12, computeWorkers: 6 },
      { hardwareConcurrency: 48, computeWorkers: 6 },
      { hardwareConcurrency: 256, computeWorkers: 6 },
    ];
    for (const { hardwareConcurrency, computeWorkers } of cases) {
      expect(selectExecutionProfile({
        requested: "dual_branch_auto",
        policy: "precision_v1",
        hardwareConcurrency,
        workerAvailable: true,
      })).toEqual({
        mode: "adaptive_pool",
        computeWorkers,
        reason: "adaptive_pool_eligible",
      });
    }
  });
});
