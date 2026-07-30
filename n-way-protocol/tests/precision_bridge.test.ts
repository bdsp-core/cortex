import { describe, expect, it } from "vitest";
import { PrecisionPolicy } from "../../cortex_web/apps/web/engine/precision_policy";
import { evaluateFrozenPrecision, precisionBankArrays } from "../src/precision_bridge";
import type { Candidate } from "../src/types";
import { K, manualState } from "./fixtures";

function candidates(): Candidate[] {
  const values: Candidate[] = [];
  for (let askedK = 0; askedK < K; askedK += 1) {
    for (let i = 0; i < 25; i += 1) {
      values.push({
        askedK,
        segmentIndex: askedK * 25 + i,
        segId: 10_000 + askedK * 25 + i,
        focalSignal: -1.2 + i * 0.1,
        focalSignalSd: 0.05,
      });
    }
  }
  return values;
}

describe("unchanged PrecisionPolicy bridge", () => {
  it("presents only focal/direct evidence to the frozen policy", () => {
    const all = candidates();
    const remaining = new Set(all.map((candidate) => candidate.segId));
    const bank = precisionBankArrays(K, all, remaining);
    expect(bank.sMean.every((row) => row.length === 25)).toBe(true);

    const identity = Array.from({ length: K }, (_, i) => (
      Array.from({ length: K }, (_, j) => i === j ? 1 : 0)
    ));
    const policy = PrecisionPolicy.fromInputs({
      taskCodes: ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
      taskLabels: ["Spike", "SZ", "LPD", "GPD", "LRDA", "GRDA", "IIC"],
      taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
      corrL: identity,
      corrT: identity,
      ellStar: new Array(K).fill(0),
      precisionBandEdges: Array.from({ length: K }, () => [-0.4, 0.4]),
      segments: [],
    });
    policy.recordAdministered(2, -0.7);
    const result = evaluateFrozenPrecision({
      policy,
      state: manualState(64),
      nPerTask: [0, 0, 1, 0, 0, 0, 0],
      remainingCandidates: all,
      remainingSegmentIds: remaining,
    });
    expect(result.policyName).toBe("precision_v1");
    expect(result.stop).toBe(false);
    const snapshot = policy.snapshot();
    expect(snapshot.name).toBe("precision_v1");
    if (snapshot.name === "precision_v1") {
      expect(snapshot.bandAdministered.flat().reduce((a, b) => a + b, 0)).toBe(1);
    }
  });
});
