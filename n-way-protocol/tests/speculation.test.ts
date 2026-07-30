import { describe, expect, it } from "vitest";
import {
  executeRankedBranches, planRankedSpeculation, rankOutcomes,
} from "../src/speculation";

const distribution = [
  { outcome: 1, probability: 0.12 },
  { outcome: 2, probability: 0.46 },
  { outcome: 3, probability: 0.08 },
  { outcome: 4, probability: 0.17 },
  { outcome: 5, probability: 0.10 },
  { outcome: 6, probability: 0.07 },
];

describe("bounded ranked speculation", () => {
  it("uses the coordinator and at most one helper", () => {
    const plan = planRankedSpeculation(distribution, true);
    expect(plan.coordinator.outcome).toBe(2);
    expect(plan.helper?.outcome).toBe(4);
    expect(plan.deferred).toHaveLength(4);
    expect(plan.cachedProbabilityMass).toBeCloseTo(0.63, 14);
    expect(rankOutcomes(distribution).map((entry) => entry.rank)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("adopts only the actual cached response branch", async () => {
    const authoritative = { value: 10 };
    const result = await executeRankedBranches({
      authoritative,
      distribution,
      actualOutcome: Promise.resolve(4),
      cloneCore: (core) => ({ ...core }),
      advance: async (core, outcome) => ({ value: core.value + outcome }),
      helperAvailable: true,
    });
    expect(result).toMatchObject({ outcome: 4, rank: 2, cacheHit: true, result: { value: 14 } });
    expect(authoritative.value).toBe(10);
  });

  it("computes an uncached actual outcome from an untouched clone", async () => {
    const authoritative = { values: [10] };
    const result = await executeRankedBranches({
      authoritative,
      distribution,
      actualOutcome: Promise.resolve(6),
      cloneCore: (core) => ({ values: core.values.slice() }),
      advance: async (core, outcome) => {
        core.values.push(outcome);
        return core.values;
      },
      helperAvailable: true,
    });
    expect(result.cacheHit).toBe(false);
    expect(result.rank).toBe(6);
    expect(result.result).toEqual([10, 6]);
    expect(authoritative.values).toEqual([10]);
  });
});

