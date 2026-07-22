// Tests for the posterior resolution-confidence readout. It is the
// min over tasks of max(π_k, 1−π_k) — symmetric in π (never reveals
// pass vs fail), and bound by the least-resolved task.

import { describe, expect, it } from "vitest";
import { resolutionConfidence } from "./progress";
import { TrialDiag } from "../engine";

function diagWithPi(pi: number[]): TrialDiag {
  // resolutionConfidence only reads .pi; cast a partial.
  return { pi } as unknown as TrialDiag;
}

describe("resolutionConfidence", () => {
  it("is 0.5 when a task is maximally undecided", () => {
    expect(resolutionConfidence(diagWithPi([0.99, 0.5, 0.01]))).toBeCloseTo(0.5, 12);
  });

  it("is symmetric: π and 1−π give the same confidence", () => {
    expect(resolutionConfidence(diagWithPi([0.9]))).toBeCloseTo(
      resolutionConfidence(diagWithPi([0.1]))!, 12);
  });

  it("is bound by the least-resolved (most-uncertain) task", () => {
    // tasks at 0.95/0.97 confidence, one at 0.7 → min is 0.7
    expect(resolutionConfidence(diagWithPi([0.95, 0.03, 0.7]))).toBeCloseTo(0.7, 12);
  });

  it("approaches 1 when every task is decisive", () => {
    expect(resolutionConfidence(diagWithPi([0.999, 0.001, 0.998]))).toBeGreaterThan(0.99);
  });

  it("is absent for cut-independent Precision diagnostics", () => {
    expect(resolutionConfidence(diagWithPi([]))).toBeNull();
  });
});
