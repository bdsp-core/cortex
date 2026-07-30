import { describe, expect, it } from "vitest";
import { expectedLoss as productionExpectedLoss } from "../../cortex_web/apps/web/engine/choose_item";
import {
  auditSelectorRegret, chooseCandidate, expectedPosteriorLoss, predictedOutcomeDistribution,
  shortlistCandidates,
} from "../src/selector";
import type { Candidate, ProtocolSegment } from "../src/types";
import {
  artifact, binaryProfile, manualState, nwayProfile, randomState, segments,
} from "./fixtures";

function candidatesFor(bank: readonly ProtocolSegment[], taskIndices = [1, 2, 3, 4, 5, 6]): Candidate[] {
  const result: Candidate[] = [];
  for (const segment of bank) {
    for (const askedK of taskIndices) {
      result.push({
        askedK,
        segmentIndex: segment.segmentIndex,
        segId: segment.segId,
        focalSignal: segment.sMean[askedK],
        focalSignalSd: segment.sSd[askedK],
      });
    }
  }
  return result;
}

describe("categorical total-variance selector", () => {
  it("matches production expected loss for a binary candidate", () => {
    const state = manualState();
    const segment = segments(1)[0];
    const candidate: Candidate = {
      askedK: 0, segmentIndex: 0, segId: segment.segId,
      focalSignal: segment.sMean[0], focalSignalSd: segment.sSd[0],
    };
    const protocol = expectedPosteriorLoss(
      state, binaryProfile, undefined, candidate, segment,
    );
    const production = productionExpectedLoss(
      state as never, 0, segment.sMean[0], segment.sSd[0],
    );
    expect(protocol).toBeCloseTo(production, 12);
  });

  it("marginalizes over every registered IIIC outcome", () => {
    const state = randomState(64);
    const segment = segments(1)[0];
    const distribution = predictedOutcomeDistribution(
      state, nwayProfile, artifact, 3, segment,
    );
    expect(distribution.map((entry) => entry.outcome)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(distribution.reduce((sum, entry) => sum + entry.probability, 0)).toBeCloseTo(1, 13);
    const candidate = candidatesFor([segment], [3])[0];
    const loss = expectedPosteriorLoss(state, nwayProfile, artifact, candidate, segment);
    expect(Number.isFinite(loss)).toBe(true);
    expect(loss).toBeGreaterThanOrEqual(0);
  });

  it("builds a deterministic bounded shortlist and retains forced content", () => {
    const state = randomState(48);
    const bank = segments(120);
    const candidates = candidatesFor(bank);
    const forced = candidates[candidates.length - 1];
    const options = {
      fullScanLimit: 20,
      coarsePerTask: 8,
      entropyPerTask: 3,
      forcedSegmentIds: new Set([forced.segId]),
    };
    const first = shortlistCandidates(state, nwayProfile, artifact, candidates, bank, options);
    const second = shortlistCandidates(state, nwayProfile, artifact, candidates, bank, options);
    expect(first.map((entry) => `${entry.askedK}/${entry.segId}`)).toEqual(
      second.map((entry) => `${entry.askedK}/${entry.segId}`),
    );
    expect(first.length).toBeLessThan(candidates.length);
    expect(first.some((entry) => entry === forced)).toBe(true);
  });

  it("chooses a finite-loss eligible candidate", () => {
    const state = randomState(64);
    const bank = segments(8);
    const candidates = candidatesFor(bank);
    const chosen = chooseCandidate(state, nwayProfile, artifact, candidates, bank);
    expect(candidates.some((candidate) => (
      candidate.segId === chosen.segId && candidate.askedK === chosen.askedK
    ))).toBe(true);
    expect(Number.isFinite(chosen.loss)).toBe(true);
  });

  it("reports shortlist regret against an exhaustive categorical scan", () => {
    const state = randomState(48);
    const bank = segments(20);
    const candidates = candidatesFor(bank);
    const audit = auditSelectorRegret(
      state, nwayProfile, artifact, candidates, bank,
      { fullScanLimit: 10, coarsePerTask: 4, entropyPerTask: 2 },
    );
    expect(audit.shortlistSize).toBeLessThan(audit.candidateCount);
    expect(audit.absoluteRegret).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(audit.relativeRegret)).toBe(true);
  });
});

