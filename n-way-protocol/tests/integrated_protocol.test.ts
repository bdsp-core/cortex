import { describe, expect, it } from "vitest";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { chooseProtocolCandidate } from "../src/integrated_protocol";
import { makeObservation, responseProbabilities } from "../src/likelihood";
import {
  cloneProtocolState, logLikelihoodHistory, resampleAndRejuvenateProtocol,
  updateProtocol,
} from "../src/particles";
import { assertReplayCompatible, profileStamp, validateProfile } from "../src/profile";
import { chooseResearchCandidate } from "../src/research_selector";
import { chooseCandidate } from "../src/selector";
import type { Candidate, ProtocolSegment } from "../src/types";
import {
  artifact, ensembleArtifact, integratedNwayProfile, K, nwayProfile,
  randomState, segments,
} from "./fixtures";

function candidatesFor(bank: readonly ProtocolSegment[]): Candidate[] {
  return bank.flatMap((segment) => [1, 2, 3, 4, 5, 6].map((askedK) => ({
    askedK,
    segmentIndex: segment.segmentIndex,
    segId: segment.segId,
    focalSignal: segment.sMean[askedK],
    focalSignalSd: segment.sSd[askedK],
  })));
}

describe("integrated n-way R&D profile", () => {
  it("uses the exact weighted artifact mixture and preserves the focal marginal", () => {
    const state = randomState(8);
    const segment = segments(1)[0];
    const actual = responseProbabilities(
      integratedNwayProfile, ensembleArtifact, 3, segment,
      state.t, state.l, 0, K,
    );
    const components = ensembleArtifact.draws.map((draw) => responseProbabilities(
      nwayProfile,
      { ...artifact, beta: draw.beta, distractorLapse: draw.distractorLapse },
      3, segment, state.t, state.l, 0, K,
    ));
    for (let index = 0; index < actual.length; index += 1) {
      const expected = ensembleArtifact.draws.reduce(
        (sum, draw, drawIndex) => sum + draw.weight * components[drawIndex][index].probability,
        0,
      );
      expect(actual[index].probability).toBeCloseTo(expected, 14);
    }
    expect(actual.find(({ outcome }) => outcome === 3)!.probability)
      .toBe(components[0].find(({ outcome }) => outcome === 3)!.probability);
  });

  it("updates and replays an ensemble-created history through MH", () => {
    const bank = segments(2);
    const state = randomState(96);
    const rng = new Rng(9421);
    updateProtocol(
      state, integratedNwayProfile, ensembleArtifact, bank,
      makeObservation(integratedNwayProfile, 2, 0, 5),
    );
    updateProtocol(
      state, integratedNwayProfile, ensembleArtifact, bank,
      makeObservation(integratedNwayProfile, 4, 1, 4),
    );
    resampleAndRejuvenateProtocol(
      state, integratedNwayProfile, ensembleArtifact, bank,
      rng, 2, 2.38 / Math.sqrt(2 * K), 2,
    );
    const replayed = logLikelihoodHistory(
      state, integratedNwayProfile, ensembleArtifact, bank, state.t, state.l,
    );
    for (let n = 0; n < state.N; n += 1) {
      expect(replayed[n]).toBeCloseTo(state.logLik[n], 12);
    }
  });

  it("dispatches frozen and integrated profiles to their declared selectors", () => {
    const bank = segments(24);
    const candidates = candidatesFor(bank);
    const options = {
      fullScanLimit: 10, coarsePerTask: 2, entropyPerTask: 1, fisherPerTask: 2,
    };
    const frozenState = randomState(64);
    const integratedState = cloneProtocolState(frozenState);
    expect(chooseProtocolCandidate(
      frozenState, nwayProfile, artifact, candidates, bank, options,
    )).toEqual(chooseCandidate(
      frozenState, nwayProfile, artifact, candidates, bank, options,
    ));
    expect(chooseProtocolCandidate(
      integratedState, integratedNwayProfile, ensembleArtifact,
      candidates, bank, options,
    )).toEqual(chooseResearchCandidate(
      integratedState, integratedNwayProfile, ensembleArtifact,
      candidates, bank, { ...options, objective: "total_variance" },
    ));
  });

  it("validates and stamps integrated sessions separately from frozen sessions", () => {
    expect(() => validateProfile(
      integratedNwayProfile, K, ensembleArtifact, true,
    )).not.toThrow();
    expect(() => assertReplayCompatible(
      profileStamp(nwayProfile), profileStamp(integratedNwayProfile),
    )).toThrow(/resume_incompatible/);
  });
});
