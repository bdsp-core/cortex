import type {
  ComputeExecutionMode, RequestedComputeMode, TerminationPolicyName,
} from "./types";

export interface ExecutionProfile {
  mode: Exclude<ComputeExecutionMode, "serial_fallback">;
  computeWorkers: 1 | 2;
  reason:
    | "rollout_disabled"
    | "policy_not_parallelized"
    | "worker_unavailable"
    | "concurrency_unknown"
    | "low_concurrency"
    | "dual_branch_eligible";
}

/**
 * Conservative device adaptation. hardwareConcurrency is only a browser hint;
 * the binary response model has exactly two useful branches, so more than two
 * compute workers can only add contention.
 */
export function selectExecutionProfile(args: {
  requested: RequestedComputeMode;
  policy: TerminationPolicyName;
  hardwareConcurrency?: number;
  workerAvailable: boolean;
}): ExecutionProfile {
  if (args.requested !== "dual_branch_auto") {
    return { mode: "serial", computeWorkers: 1, reason: "rollout_disabled" };
  }
  if (args.policy !== "precision_v1") {
    return { mode: "serial", computeWorkers: 1, reason: "policy_not_parallelized" };
  }
  if (!args.workerAvailable) {
    return { mode: "serial", computeWorkers: 1, reason: "worker_unavailable" };
  }
  const cores = args.hardwareConcurrency;
  if (!Number.isFinite(cores) || !cores || cores < 1) {
    return { mode: "serial", computeWorkers: 1, reason: "concurrency_unknown" };
  }
  if (cores < 4) {
    return { mode: "serial", computeWorkers: 1, reason: "low_concurrency" };
  }
  return { mode: "dual_branch", computeWorkers: 2, reason: "dual_branch_eligible" };
}
