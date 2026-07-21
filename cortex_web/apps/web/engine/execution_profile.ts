import type {
  ComputeExecutionMode, RequestedComputeMode, TerminationPolicyName,
} from "./types";

export interface ExecutionProfile {
  mode: Exclude<ComputeExecutionMode, "serial_fallback">;
  computeWorkers: number;
  reason:
    | "rollout_disabled"
    | "policy_not_parallelized"
    | "worker_unavailable"
    | "concurrency_unknown"
    | "low_concurrency"
    | "adaptive_pool_eligible";
}

/**
 * Conservative startup bounds. The coordinator is one compute worker and the
 * remainder are immutable ranked-outcome helpers. At least one reported core
 * is reserved, and two are reserved on devices reporting eight or more.
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
  const computeWorkers = cores >= 12 ? 6
    : cores >= 8 ? 5
      : cores >= 5 ? 3 : 2;
  return { mode: "adaptive_pool", computeWorkers, reason: "adaptive_pool_eligible" };
}
