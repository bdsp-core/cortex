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
    | "calibration_selected_serial"
    | "adaptive_pool_eligible";
}

export interface WorkerCalibrationSample {
  workers: number;
  durationMs: number;
  heartbeatMaxDelayMs: number;
}

export interface RuntimeLoadSample {
  sampleCount: number;
  meanDelayMs: number;
  maxDelayMs: number;
}

/** Centralized, timing-only runtime guard. Eight 250 ms main-realm samples
 * represent a two-second window; either a severe frame/task stall or sustained
 * mean delay requests one conservative downward pool step. */
export const RUNTIME_LOAD_PROFILE = {
  windowSamples: 8,
  meanDelayLimitMs: 20,
  maxDelayLimitMs: 50,
} as const;

export function runtimeLoadExceedsLimit(sample: RuntimeLoadSample): boolean {
  return Number.isFinite(sample.sampleCount) && sample.sampleCount >= 1
    && Number.isFinite(sample.meanDelayMs) && sample.meanDelayMs >= 0
    && Number.isFinite(sample.maxDelayMs) && sample.maxDelayMs >= 0
    && (sample.meanDelayMs > RUNTIME_LOAD_PROFILE.meanDelayLimitMs
      || sample.maxDelayMs > RUNTIME_LOAD_PROFILE.maxDelayLimitMs);
}

/** Downward-only staircase for pools that may also have calibrated to 3 or 5. */
export function nextLowerWorkerCount(current: number): number {
  if (!Number.isInteger(current) || current <= 1) return 1;
  if (current > 4) return 4;
  if (current > 2) return 2;
  return 1;
}

/** Choose the smallest pool within 5% of the fastest responsive sample. This
 * preserves CPU/memory headroom when an additional worker has only a marginal
 * throughput benefit. Timing is device configuration only and never enters
 * statistical policy state. */
export function selectCalibratedWorkerCount(
  samples: readonly WorkerCalibrationSample[], heartbeatLimitMs = 50,
): number {
  const valid = samples.filter((sample) => Number.isInteger(sample.workers)
    && sample.workers >= 1
    && Number.isFinite(sample.durationMs) && sample.durationMs > 0
    && Number.isFinite(sample.heartbeatMaxDelayMs)
    && sample.heartbeatMaxDelayMs <= heartbeatLimitMs);
  if (valid.length === 0) return 1;
  const fastest = Math.min(...valid.map((sample) => sample.durationMs));
  return Math.min(...valid
    .filter((sample) => sample.durationMs <= fastest * 1.05)
    .map((sample) => sample.workers));
}

/**
 * Conservative startup bounds for the persistent selector/MH shard pool.
 * The authoritative coordinator is separate. At least one reported core is
 * reserved, and two are reserved on devices reporting eight or more.
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
