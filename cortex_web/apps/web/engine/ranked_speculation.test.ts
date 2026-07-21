import { describe, expect, it } from "vitest";

import {
  RankedSpeculationRejuvenationDeferredError,
  isRankedSpeculationRejuvenationDeferred,
  RANKED_SPECULATION_PROFILE,
  shouldExpandRankedSpeculation,
} from "./ranked_speculation";

describe("bounded ranked speculation profile", () => {
  it("allows only material rank-two work with measured device headroom", () => {
    expect(shouldExpandRankedSpeculation({
      rank: 2, probability: 0.2, workerCount: 4,
      separateBranchExecutorActive: false,
    })).toBe(true);
    for (const input of [
      { rank: 3, probability: 0.2, workerCount: 6, separateBranchExecutorActive: false },
      { rank: 2, probability: 0.09, workerCount: 6, separateBranchExecutorActive: false },
      { rank: 2, probability: 0.2, workerCount: 2, separateBranchExecutorActive: false },
      { rank: 2, probability: 0.2, workerCount: 6, separateBranchExecutorActive: true },
    ]) {
      expect(shouldExpandRankedSpeculation(input)).toBe(false);
    }
  });

  it("keeps speculative MH disabled at the central adjustment point", () => {
    expect(RANKED_SPECULATION_PROFILE.allowMhRejuvenation).toBe(false);
    expect(isRankedSpeculationRejuvenationDeferred(
      new RankedSpeculationRejuvenationDeferredError(),
    )).toBe(true);
    expect(isRankedSpeculationRejuvenationDeferred(new Error("other"))).toBe(false);
  });
});
