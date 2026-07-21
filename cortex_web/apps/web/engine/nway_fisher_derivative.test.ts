import { describe, expect, it } from "vitest";

import {
  fillResponseProbabilities, fillScreeningProbabilitiesAndJacobians,
  IIIC_TASK_INDICES, makeResponseProbabilityWorkspace, makeScreeningJacobianWorkspace,
} from "./nway_likelihood";
import type { ComputeSegmentMeta } from "./types";

// Frozen before qualification. The existing CDF uses a bounded rational erfc
// approximation, while the analytical derivative is for the intended normal
// CDF, so central differences agree to this explicit numerical envelope.
const DERIVATIVE_ABSOLUTE_TOLERANCE = 2e-6;
const DERIVATIVE_RELATIVE_TOLERANCE = 2e-4;
const FINITE_DIFFERENCE = 1e-4;

const K = 7;

function assertDerivative(actual: number, reference: number): void {
  const absolute = Math.abs(actual - reference);
  const relative = absolute / Math.max(Math.abs(reference), 1e-12);
  expect(
    absolute <= DERIVATIVE_ABSOLUTE_TOLERANCE
      || relative <= DERIVATIVE_RELATIVE_TOLERANCE,
    `analytical=${actual} central=${reference} abs=${absolute} rel=${relative}`,
  ).toBe(true);
}

function segment(caseIndex: number): ComputeSegmentMeta {
  return {
    segId: 9000 + caseIndex,
    applicableTaskIdx: IIIC_TASK_INDICES.slice(),
    sMean: Array.from({ length: K }, (_, k) =>
      Math.sin((caseIndex + 1) * (k + 0.37)) * (caseIndex === 3 ? 4.5 : 1.8)),
    sSd: Array.from({ length: K }, (_, k) =>
      caseIndex === 2 ? (k + 1) * 0.21 : 0.02 + ((caseIndex + k) % 5) * 0.09),
  };
}

describe("analytical categorical Fisher Jacobian", () => {
  it("matches frozen screening probabilities and central differences", () => {
    for (let caseIndex = 0; caseIndex < 4; caseIndex++) {
      const candidate = segment(caseIndex);
      const t = Float64Array.from({ length: K }, (_, k) =>
        Math.cos((caseIndex + 0.71) * (k + 1.13)) * (caseIndex === 3 ? 3.8 : 1.4));
      const l = Float64Array.from({ length: K }, (_, k) =>
        Math.sin((caseIndex + 1.29) * (k + 0.83)) * (caseIndex === 3 ? 2.7 : 1.1));
      for (const kind of ["spike", "iiic"] as const) {
        const asked = kind === "spike" ? 0 : IIIC_TASK_INDICES[caseIndex + 1];
        const outcomeCount = kind === "spike" ? 2 : IIIC_TASK_INDICES.length;
        const analytical = new Float64Array(outcomeCount);
        const frozen = new Float64Array(outcomeCount);
        const plus2 = new Float64Array(outcomeCount);
        const plus = new Float64Array(outcomeCount);
        const minus = new Float64Array(outcomeCount);
        const minus2 = new Float64Array(outcomeCount);
        const analyticalWorkspace = makeScreeningJacobianWorkspace(K);
        const frozenWorkspace = makeResponseProbabilityWorkspace(K);
        fillScreeningProbabilitiesAndJacobians(
          kind, asked, candidate, t, l, 0, K, analytical, analyticalWorkspace,
        );
        fillResponseProbabilities(
          kind, asked, candidate, t, l, 0, K, frozen, frozenWorkspace, true,
        );
        expect(Array.from(analytical)).toEqual(Array.from(frozen));

        const relevant = kind === "spike" ? [asked] : IIIC_TASK_INDICES;
        for (const parameter of ["bias", "skill"] as const) {
          const values = parameter === "bias" ? t : l;
          const jacobian = parameter === "bias"
            ? analyticalWorkspace.biasJacobian : analyticalWorkspace.skillJacobian;
          for (const k of relevant) {
            const original = values[k];
            values[k] = original + 2 * FINITE_DIFFERENCE;
            fillResponseProbabilities(
              kind, asked, candidate, t, l, 0, K, plus2, frozenWorkspace, true,
            );
            values[k] = original + FINITE_DIFFERENCE;
            fillResponseProbabilities(
              kind, asked, candidate, t, l, 0, K, plus, frozenWorkspace, true,
            );
            values[k] = original - FINITE_DIFFERENCE;
            fillResponseProbabilities(
              kind, asked, candidate, t, l, 0, K, minus, frozenWorkspace, true,
            );
            values[k] = original - 2 * FINITE_DIFFERENCE;
            fillResponseProbabilities(
              kind, asked, candidate, t, l, 0, K, minus2, frozenWorkspace, true,
            );
            values[k] = original;
            for (let outcome = 0; outcome < outcomeCount; outcome++) {
              assertDerivative(
                jacobian[k * outcomeCount + outcome],
                (-plus2[outcome] + 8 * plus[outcome] - 8 * minus[outcome]
                  + minus2[outcome]) / (12 * FINITE_DIFFERENCE),
              );
            }
          }
        }
      }
    }
  });
});
