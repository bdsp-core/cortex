import type {
  EnginePerformanceEvent, EngineStepTiming, ExecutionProfileEvent,
  RuntimePoolAdjustmentEvent,
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

export interface ScalarDistribution {
  count: number;
  p50: number | null;
  p95: number | null;
  max: number | null;
}

export interface EnginePerformanceSummaryV2
  extends Omit<EnginePerformanceSummaryV1, "schemaVersion"> {
  schemaVersion: 2;
  phaseV2: {
    candidateBankPreparation: TimingDistribution;
    selector: {
      candidatePreparation: TimingDistribution;
      posteriorMoments: TimingDistribution;
      coarseMinSdScan: TimingDistribution;
      entropyScan: TimingDistribution;
      fisherScan: TimingDistribution;
      exactRefinement: TimingDistribution;
      candidateCount: ScalarDistribution;
      shortlistCount: ScalarDistribution;
    };
    particle: {
      categoricalUpdate: TimingDistribution;
      ess: TimingDistribution;
      resampling: TimingDistribution;
      mhProposalGeneration: TimingDistribution;
      mhPrior: TimingDistribution;
      mhHistoryLikelihood: TimingDistribution;
      mhAcceptance: TimingDistribution;
      mhCopying: TimingDistribution;
    };
    outcomes: {
      observedRankCounts: number[];
      observedProbability: ScalarDistribution;
      answerDispatchDelay: TimingDistribution;
      cachedProbabilityMass: ScalarDistribution;
    };
    branches: {
      queuedCount: number;
      startedCount: number;
      readyAtAnswerCount: number;
      adoptedCount: number;
      cancelledCount: number;
      discardedCount: number;
      cancellationPhaseCounts: Record<string, number>;
      adoptedWork: TimingDistribution;
      discardedWork: TimingDistribution;
    };
    eventLoopHeartbeat: {
      sampleCount: number;
      meanDelayMs: number | null;
      maxDelayMs: number | null;
    };
    runtimePool: {
      adjustmentCount: number;
      finalWorkerCount: number | null;
      adjustments: Array<Omit<RuntimePoolAdjustmentEvent, "kind">>;
    };
  };
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

function scalarDistribution(values: number[]): ScalarDistribution {
  const timing = distribution(values);
  return {
    count: timing.count,
    p50: timing.p50Ms,
    p95: timing.p95Ms,
    max: timing.maxMs,
  };
}

/** Bounded, internal-only aggregate; raw timing never enters engine state. */
export class EnginePerformanceCollector {
  private profile: ExecutionProfileEvent | null = null;
  private answerToItem: number[] = [];
  private steps: EngineStepTiming[] = [];
  private heartbeat = { sampleCount: 0, weightedDelay: 0, maxDelayMs: 0 };
  private runtimePoolAdjustments: RuntimePoolAdjustmentEvent[] = [];

  record(event: EnginePerformanceEvent): void {
    if (event.kind === "execution_profile") this.profile = event;
    else if (event.kind === "answer_to_item") this.answerToItem.push(event.durationMs);
    else if (event.kind === "engine_step") this.steps.push(event);
    else if (event.kind === "event_loop_heartbeat") {
      this.heartbeat.sampleCount += event.sampleCount;
      this.heartbeat.weightedDelay += event.meanDelayMs * event.sampleCount;
      this.heartbeat.maxDelayMs = Math.max(
        this.heartbeat.maxDelayMs, event.maxDelayMs,
      );
    } else this.runtimePoolAdjustments.push(event);
  }

  summary(replayedTrials = 0): EnginePerformanceSummaryV2 {
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
    const phases = this.steps.flatMap((step) => step.phaseV2 ? [step.phaseV2] : []);
    const branches = phases.flatMap((phase) => phase.branches);
    const observedRanks = new Array(6).fill(0);
    for (const phase of phases) {
      if (phase.observedOutcomeRank !== null && phase.observedOutcomeRank >= 1) {
        while (observedRanks.length < phase.observedOutcomeRank) observedRanks.push(0);
        observedRanks[phase.observedOutcomeRank - 1] += 1;
      }
    }
    return {
      schemaVersion: 2,
      profile: this.profile ? {
        requested: this.profile.requested,
        executionMode: this.profile.executionMode,
        reason: this.profile.reason,
        hardwareConcurrency: this.profile.hardwareConcurrency,
        selectedWorkerCount: this.profile.selectedWorkerCount,
        calibrationResult: this.profile.calibrationResult,
        estimatedWorkerMemoryBytes: this.profile.estimatedWorkerMemoryBytes,
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
      phaseV2: {
        candidateBankPreparation: distribution(
          this.steps.map((step) => step.bankPreparationMs),
        ),
        selector: {
          candidatePreparation: distribution(
            phases.map((phase) => phase.selection.candidatePreparationMs),
          ),
          posteriorMoments: distribution(
            phases.map((phase) => phase.selection.posteriorMomentsMs),
          ),
          coarseMinSdScan: distribution(
            phases.map((phase) => phase.selection.coarseMinSdScanMs),
          ),
          entropyScan: distribution(
            phases.map((phase) => phase.selection.entropyScanMs),
          ),
          fisherScan: distribution(
            phases.map((phase) => phase.selection.fisherScanMs),
          ),
          exactRefinement: distribution(
            phases.map((phase) => phase.selection.exactRefinementMs),
          ),
          candidateCount: scalarDistribution(
            phases.map((phase) => phase.selection.candidateCount),
          ),
          shortlistCount: scalarDistribution(
            phases.map((phase) => phase.selection.shortlistCount),
          ),
        },
        particle: {
          categoricalUpdate: distribution(
            phases.map((phase) => phase.particle.categoricalUpdateMs),
          ),
          ess: distribution(phases.map((phase) => phase.particle.essMs)),
          resampling: distribution(
            phases.map((phase) => phase.particle.resamplingMs),
          ),
          mhProposalGeneration: distribution(
            phases.map((phase) => phase.particle.mhProposalGenerationMs),
          ),
          mhPrior: distribution(phases.map((phase) => phase.particle.mhPriorMs)),
          mhHistoryLikelihood: distribution(
            phases.map((phase) => phase.particle.mhHistoryLikelihoodMs),
          ),
          mhAcceptance: distribution(
            phases.map((phase) => phase.particle.mhAcceptanceMs),
          ),
          mhCopying: distribution(
            phases.map((phase) => phase.particle.mhCopyingMs),
          ),
        },
        outcomes: {
          observedRankCounts: observedRanks,
          observedProbability: scalarDistribution(phases.flatMap((phase) =>
            phase.observedOutcomeProbability === null
              ? [] : [phase.observedOutcomeProbability])),
          answerDispatchDelay: distribution(
            phases.map((phase) => phase.answerDispatchDelayMs),
          ),
          cachedProbabilityMass: scalarDistribution(
            phases.map((phase) => phase.cachedProbabilityMass),
          ),
        },
        branches: {
          queuedCount: branches.filter((branch) => branch.queued).length,
          startedCount: branches.filter((branch) => branch.started).length,
          readyAtAnswerCount: branches.filter((branch) => branch.readyAtAnswer).length,
          adoptedCount: branches.filter((branch) => branch.adopted).length,
          cancelledCount: branches.filter((branch) => branch.cancelled).length,
          discardedCount: branches.filter((branch) => branch.discarded).length,
          cancellationPhaseCounts: branches.reduce<Record<string, number>>(
            (counts, branch) => {
              if (branch.cancelled) {
                const phase = branch.cancellationPhase ?? "between_phases";
                counts[phase] = (counts[phase] ?? 0) + 1;
              }
              return counts;
            },
            {},
          ),
          adoptedWork: distribution(branches.flatMap((branch) =>
            branch.adopted && branch.durationMs !== null ? [branch.durationMs] : [])),
          discardedWork: distribution(branches.flatMap((branch) =>
            branch.discarded && branch.durationMs !== null ? [branch.durationMs] : [])),
        },
        eventLoopHeartbeat: {
          sampleCount: this.heartbeat.sampleCount,
          meanDelayMs: this.heartbeat.sampleCount > 0
            ? rounded(this.heartbeat.weightedDelay / this.heartbeat.sampleCount) : null,
          maxDelayMs: this.heartbeat.sampleCount > 0
            ? rounded(this.heartbeat.maxDelayMs) : null,
        },
        runtimePool: {
          adjustmentCount: this.runtimePoolAdjustments.length,
          finalWorkerCount: this.runtimePoolAdjustments.at(-1)?.selectedWorkerCount
            ?? this.profile?.selectedWorkerCount ?? null,
          adjustments: this.runtimePoolAdjustments.map(({ kind: _kind, ...event }) => event),
        },
      },
    };
  }
}
