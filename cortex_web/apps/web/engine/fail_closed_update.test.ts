import { describe, expect, it } from "vitest";

import { PosteriorUpdateError, update } from "./particles";
import type { ParticleState, PriorPieces } from "./types";

function fixture(): ParticleState {
  const pieces: PriorPieces = { K: 1, sigmaInv: [[1]], logDet: 0, L: [[1]] };
  return {
    N: 2,
    K: 1,
    t: Float64Array.from([0, 0]),
    l: Float64Array.from([0, Number.NaN]),
    w: Float64Array.from([0.25, 0.75]),
    logPrior: Float64Array.from([0, 0]),
    logLik: Float64Array.from([-1, -2]),
    history: [{ kind: "binary", k: 0, s: 0.1, y: 1, sSd: 0.2, rawPick: 0 }],
    prior: { tPieces: pieces, lPieces: pieces },
  };
}

describe("posterior update is transactional and fail-closed", () => {
  it("throws without changing weights, likelihoods, or history", () => {
    const state = fixture();
    const beforeWeights = state.w.slice();
    const beforeLogLik = state.logLik.slice();
    const beforeHistory = state.history.map((row) => ({ ...row }));

    expect(() => update(state, 0, 1, 1, 0.1)).toThrow(PosteriorUpdateError);
    expect(state.w).toEqual(beforeWeights);
    expect(state.logLik).toEqual(beforeLogLik);
    expect(state.history).toEqual(beforeHistory);
  });
});
