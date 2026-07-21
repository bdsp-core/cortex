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

export type RuntimeLoadTrigger = "sustained_mean" | "repeated_max";

export interface RuntimeLoadReductionRequest extends RuntimeLoadSample {
  trigger: RuntimeLoadTrigger;
  evidenceWindowCount: number;
}

export interface RuntimeLoadGuardState {
  consecutiveMaxDelayWindows: number;
  cooldownWindowsRemaining: number;
}

export interface RuntimeLoadGuardDecision {
  state: RuntimeLoadGuardState;
  reduction: RuntimeLoadReductionRequest | null;
}

/** Centralized, timing-only runtime guard. Eight 250 ms main-realm samples
 * represent a two-second window. Sustained mean delay requests a conservative
 * downward step immediately. A max-only spike must recur in consecutive
 * windows so an isolated render/GC task cannot ratchet the pool downward.
 * After a request, a 40-second cooldown prevents cascading reductions. */
export const RUNTIME_LOAD_PROFILE = {
  windowSamples: 8,
  meanDelayLimitMs: 20,
  maxDelayLimitMs: 50,
  consecutiveMaxDelayWindows: 2,
  cooldownWindowsAfterReduction: 20,
} as const;

function validRuntimeLoadWindow(sample: RuntimeLoadSample): boolean {
  return Number.isFinite(sample.sampleCount)
    && sample.sampleCount >= RUNTIME_LOAD_PROFILE.windowSamples
    && Number.isFinite(sample.meanDelayMs) && sample.meanDelayMs >= 0
    && Number.isFinite(sample.maxDelayMs) && sample.maxDelayMs >= 0;
}

export function runtimeLoadExceedsLimit(sample: RuntimeLoadSample): boolean {
  return validRuntimeLoadWindow(sample)
    && (sample.meanDelayMs > RUNTIME_LOAD_PROFILE.meanDelayLimitMs
      || sample.maxDelayMs > RUNTIME_LOAD_PROFILE.maxDelayLimitMs);
}

export function initialRuntimeLoadGuardState(): RuntimeLoadGuardState {
  return { consecutiveMaxDelayWindows: 0, cooldownWindowsRemaining: 0 };
}

/** Evaluate every completed heartbeat window, including clean windows. The
 * returned state is device-timing state only and never enters inference. */
export function observeRuntimeLoadWindow(
  state: RuntimeLoadGuardState, sample: RuntimeLoadSample,
): RuntimeLoadGuardDecision {
  if (!validRuntimeLoadWindow(sample)) {
    return {
      state: { ...state, consecutiveMaxDelayWindows: 0 },
      reduction: null,
    };
  }

  if (state.cooldownWindowsRemaining > 0) {
    return {
      state: {
        consecutiveMaxDelayWindows: 0,
        cooldownWindowsRemaining: state.cooldownWindowsRemaining - 1,
      },
      reduction: null,
    };
  }

  if (sample.meanDelayMs > RUNTIME_LOAD_PROFILE.meanDelayLimitMs) {
    return {
      state: {
        consecutiveMaxDelayWindows: 0,
        cooldownWindowsRemaining: RUNTIME_LOAD_PROFILE.cooldownWindowsAfterReduction,
      },
      reduction: {
        ...sample,
        trigger: "sustained_mean",
        evidenceWindowCount: 1,
      },
    };
  }

  if (sample.maxDelayMs > RUNTIME_LOAD_PROFILE.maxDelayLimitMs) {
    const consecutiveMaxDelayWindows = state.consecutiveMaxDelayWindows + 1;
    if (consecutiveMaxDelayWindows
        >= RUNTIME_LOAD_PROFILE.consecutiveMaxDelayWindows) {
      return {
        state: {
          consecutiveMaxDelayWindows: 0,
          cooldownWindowsRemaining: RUNTIME_LOAD_PROFILE.cooldownWindowsAfterReduction,
        },
        reduction: {
          ...sample,
          trigger: "repeated_max",
          evidenceWindowCount: consecutiveMaxDelayWindows,
        },
      };
    }
    return {
      state: { ...state, consecutiveMaxDelayWindows },
      reduction: null,
    };
  }

  return {
    state: { ...state, consecutiveMaxDelayWindows: 0 },
    reduction: null,
  };
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
