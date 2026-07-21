import { describe, expect, it } from "vitest";

import { EnginePerformanceCollector } from "./performanceSummary";

describe("internal engine performance summary", () => {
  it("is bounded and derives stable nearest-rank timing summaries", () => {
    const collector = new EnginePerformanceCollector();
    collector.record({
      kind: "execution_profile", requested: "dual_branch_auto",
      executionMode: "adaptive_pool", reason: "adaptive_pool_eligible",
      hardwareConcurrency: 8,
    });
    for (let i = 1; i <= 20; i += 1) {
      collector.record({ kind: "answer_to_item", trialIndex: i, durationMs: i });
      collector.record({
        kind: "engine_step", trialIndex: i, y: 1, pick: 1, rejuvenated: i === 20,
        bankPreparationMs: 0, updateMs: 1, rejuvenationMs: i === 20 ? 5 : 0,
        bookkeepingMs: 1, policyMs: 1, diagnosticsMs: 1,
        selectionMs: i, totalMs: i * 2, executionMode: "dual_branch",
        speculative: true, requiredBranchReadyAtAnswer: i <= 15,
        phaseV2: {
          selection: {
            candidatePreparationMs: 0, posteriorMomentsMs: 0,
            coarseMinSdScanMs: 0, entropyScanMs: 0, fisherScanMs: 0,
            exactRefinementMs: 0, candidateCount: 0, shortlistCount: 0,
          },
          particle: {
            categoricalUpdateMs: 0, essMs: 0, resamplingMs: 0,
            mhProposalGenerationMs: 0, mhPriorMs: 0,
            mhHistoryLikelihoodMs: 0, mhAcceptanceMs: 0, mhCopyingMs: 0,
          },
          observedOutcomeRank: 1,
          observedOutcomeProbability: 0.5,
          answerDispatchDelayMs: i,
          cachedProbabilityMass: 0.5,
          branches: [],
        },
      });
    }
    collector.record({
      kind: "runtime_pool_adjustment", trialIndex: 10,
      previousWorkerCount: 6, selectedWorkerCount: 4,
      reason: "main_realm_load", sampleCount: 8,
      meanDelayMs: 25, maxDelayMs: 75,
    });
    const result = collector.summary(3);
    expect(result.schemaVersion).toBe(2);
    expect(result.answerToItem).toEqual({ count: 20, p50Ms: 10, p95Ms: 19, maxMs: 20 });
    expect(result.engineTotal).toEqual({ count: 20, p50Ms: 20, p95Ms: 38, maxMs: 40 });
    expect(result.requiredBranchReadyCount).toBe(15);
    expect(result.replayedTrials).toBe(3);
    expect(result.slowestSteps).toHaveLength(10);
    expect(result.slowestSteps[0].trialIndex).toBe(20);
    expect(result.phaseV2.selector.fisherScan.count).toBe(20);
    expect(result.phaseV2.outcomes.answerDispatchDelay.count).toBe(20);
    expect(result.phaseV2.outcomes.answerDispatchDelay.p95Ms).toBe(19);
    expect(result.phaseV2.eventLoopHeartbeat.sampleCount).toBe(0);
    expect(result.phaseV2.runtimePool.adjustmentCount).toBe(1);
    expect(result.phaseV2.runtimePool.finalWorkerCount).toBe(4);
    expect(result.phaseV2.runtimePool.adjustments[0].trialIndex).toBe(10);
  });
});
