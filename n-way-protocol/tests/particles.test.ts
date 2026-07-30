import { describe, expect, it } from "vitest";
import { precomputePriorPair } from "../../cortex_web/apps/web/engine/prior";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { makeObservation } from "../src/likelihood";
import {
  cloneProtocolState, logLikelihoodHistory, makeProtocolState,
  PosteriorUpdateError, resampleAndRejuvenateProtocol, updateProtocol,
} from "../src/particles";
import {
  artifact, binaryProfile, K, manualState, nwayProfile, segments,
} from "./fixtures";

function expectArraysEqual(a: Float64Array, b: Float64Array, tolerance = 0): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i += 1) {
    if (Number.isNaN(a[i]) || Number.isNaN(b[i])) {
      expect(Number.isNaN(a[i]) && Number.isNaN(b[i])).toBe(true);
      continue;
    }
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tolerance);
  }
}

describe("protocol particle update and history", () => {
  it("keeps the binary path byte-identical for the same observation", () => {
    const source = manualState();
    const a = cloneProtocolState(source);
    const b = cloneProtocolState(source);
    const bank = segments(1);
    updateProtocol(a, nwayProfile, artifact, bank, makeObservation(nwayProfile, 0, 0, 0));
    updateProtocol(b, binaryProfile, undefined, bank, makeObservation(binaryProfile, 0, 0, 0));
    expectArraysEqual(a.w, b.w);
    expectArraysEqual(a.logLik, b.logLik);
  });

  it("retains the wrong class rather than collapsing it to y=0", () => {
    const bank = segments(1);
    const pickTwo = manualState();
    const pickSix = manualState();
    updateProtocol(
      pickTwo, nwayProfile, artifact, bank, makeObservation(nwayProfile, 1, 0, 2),
    );
    updateProtocol(
      pickSix, nwayProfile, artifact, bank, makeObservation(nwayProfile, 1, 0, 6),
    );
    expect(pickTwo.history[0].rawPick).toBe(2);
    expect(pickSix.history[0].rawPick).toBe(6);
    expect(Array.from(pickTwo.w)).not.toEqual(Array.from(pickSix.w));
  });

  it("is fail-closed and leaves state untouched on invalid input", () => {
    const state = manualState();
    state.w[2] = Number.NaN;
    const before = cloneProtocolState(state);
    expect(() => updateProtocol(
      state, nwayProfile, artifact, segments(1), makeObservation(nwayProfile, 1, 0, 3),
    )).toThrow(PosteriorUpdateError);
    expectArraysEqual(state.w, before.w);
    expectArraysEqual(state.logLik, before.logLik);
    expect(state.history).toEqual([]);
  });

  it("replays mixed history exactly through rejuvenation", () => {
    const identity = Array.from({ length: K }, (_, i) => (
      Array.from({ length: K }, (_, j) => i === j ? 1 : 0)
    ));
    const prior = precomputePriorPair(identity);
    const rng = new Rng(1999);
    const state = makeProtocolState(96, K, prior, rng);
    const bank = segments(2);
    updateProtocol(state, nwayProfile, artifact, bank, makeObservation(nwayProfile, 0, 0, K));
    updateProtocol(state, nwayProfile, artifact, bank, makeObservation(nwayProfile, 2, 1, 5));
    resampleAndRejuvenateProtocol(
      state, nwayProfile, artifact, bank, rng, 2, 2.38 / Math.sqrt(2 * K), 2,
    );
    const replayed = logLikelihoodHistory(
      state, nwayProfile, artifact, bank, state.t, state.l,
    );
    expectArraysEqual(replayed, state.logLik, 1e-12);
    expect(state.lastRejuvenation?.qIndex).toBe(2);
  });
});
