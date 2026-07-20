import type {
  EnginePerformanceEvent, EngineStepTiming, ExecutionProfileEvent,
} from "../engine/types";

export interface TimingDistribution {
  count: number;
  p50Ms: number | null;
  p95Ms: number | null;
  maxMs: number | null;
}

export interface EnginePerformanceSummaryV1 {
  schemaVersion: 1;
  profile: Omit<ExecutionProfileEvent, "kind"> | null;
  replayedTrials: number;
  answerToItem: TimingDistribution;
  engineTotal: TimingDistribution;
  selection: TimingDistribution;
  rejuvenation: TimingDistribution;
  requiredBranchReadyCount: number;
  serialFallbackCount: number;
  slowestSteps: Array<{
    trialIndex: number;
    totalMs: number;
    selectionMs: number;
    rejuvenationMs: number;
    executionMode: EngineStepTiming["executionMode"];
  }>;
}

function rounded(value: number): number {
  return Math.round(value * 100) / 100;
}

function distribution(values: number[]): TimingDistribution {
  if (values.length === 0) {
    return { count: 0, p50Ms: null, p95Ms: null, maxMs: null };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.ceil(q * sorted.length) - 1];
  return {
    count: sorted.length,
    p50Ms: rounded(at(0.5)),
    p95Ms: rounded(at(0.95)),
    maxMs: rounded(sorted[sorted.length - 1]),
  };
}

/** Bounded, internal-only aggregate; raw timing never enters engine state. */
export class EnginePerformanceCollector {
  private profile: ExecutionProfileEvent | null = null;
  private answerToItem: number[] = [];
  private steps: EngineStepTiming[] = [];

  record(event: EnginePerformanceEvent): void {
    if (event.kind === "execution_profile") this.profile = event;
    else if (event.kind === "answer_to_item") this.answerToItem.push(event.durationMs);
    else this.steps.push(event);
  }

  summary(replayedTrials = 0): EnginePerformanceSummaryV1 {
    const slowestSteps = [...this.steps]
      .sort((a, b) => b.totalMs - a.totalMs)
      .slice(0, 10)
      .map((step) => ({
        trialIndex: step.trialIndex,
        totalMs: rounded(step.totalMs),
        selectionMs: rounded(step.selectionMs),
        rejuvenationMs: rounded(step.rejuvenationMs),
        executionMode: step.executionMode,
      }));
    return {
      schemaVersion: 1,
      profile: this.profile ? {
        requested: this.profile.requested,
        executionMode: this.profile.executionMode,
        reason: this.profile.reason,
        hardwareConcurrency: this.profile.hardwareConcurrency,
      } : null,
      replayedTrials,
      answerToItem: distribution(this.answerToItem),
      engineTotal: distribution(this.steps.map((step) => step.totalMs)),
      selection: distribution(this.steps.map((step) => step.selectionMs)),
      rejuvenation: distribution(this.steps.map((step) => step.rejuvenationMs)),
      requiredBranchReadyCount: this.steps.filter(
        (step) => step.requiredBranchReadyAtAnswer,
      ).length,
      serialFallbackCount: this.steps.filter(
        (step) => step.executionMode === "serial_fallback",
      ).length,
      slowestSteps,
    };
  }
}
