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

  it("uses exactly two workers on eligible devices, regardless of extra cores", () => {
    for (const hardwareConcurrency of [4, 8, 48, 256]) {
      expect(selectExecutionProfile({
        requested: "dual_branch_auto",
        policy: "precision_v1",
        hardwareConcurrency,
        workerAvailable: true,
      })).toEqual({
        mode: "dual_branch",
        computeWorkers: 2,
        reason: "dual_branch_eligible",
      });
    }
  });
});
