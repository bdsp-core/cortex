import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { makeObservation, responseProbabilities } from "../src/likelihood";
import { updateProtocol } from "../src/particles";
import { expectedPosteriorLoss } from "../src/selector";
import type { Candidate, ProtocolParticleState } from "../src/types";
import {
  artifact, ensembleArtifact, integratedNwayProfile, nwayProfile,
} from "./fixtures";

const reference = JSON.parse(readFileSync(fileURLToPath(new URL(
  "./fixtures/python_reference.json", import.meta.url,
)), "utf8"));
const integratedReference = JSON.parse(readFileSync(fileURLToPath(new URL(
  "./fixtures/python_integrated_reference.json", import.meta.url,
)), "utf8"));
const TOLERANCE = 5e-7;

function flatten(rows: number[][]): Float64Array {
  return Float64Array.from(rows.flat());
}

function expectClose(actual: ArrayLike<number>, expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i += 1) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(TOLERANCE);
  }
}

function state(): ProtocolParticleState {
  return {
    N: reference.t.length,
    K: reference.K,
    t: flatten(reference.t),
    l: flatten(reference.l),
    w: Float64Array.from(reference.initialWeights),
    logPrior: new Float64Array(reference.t.length),
    logLik: new Float64Array(reference.t.length),
    history: [],
    prior: {
      tPieces: { K: reference.K, sigmaInv: [], logDet: 0, L: [] },
      lPieces: { K: reference.K, sigmaInv: [], logDet: 0, L: [] },
    },
  };
}

describe("independent Python/TypeScript F1 parity", () => {
  it("matches every response probability for every fixed particle", () => {
    const current = state();
    for (let n = 0; n < current.N; n += 1) {
      const probabilities = responseProbabilities(
        nwayProfile, artifact, reference.askedK, reference.segment,
        current.t, current.l, n, current.K,
      ).map((entry) => entry.probability);
      expectClose(probabilities, reference.responseProbabilities[n]);
    }
  });

  it("matches wrong-pick weights, log likelihoods, and six-outcome loss", () => {
    const current = state();
    updateProtocol(
      current, nwayProfile, artifact, [reference.segment],
      makeObservation(nwayProfile, reference.askedK, 0, reference.wrongPick),
    );
    expectClose(current.w, reference.updatedWeights);
    expectClose(current.logLik, reference.updatedLogLik);
    const candidate: Candidate = {
      askedK: reference.askedK,
      segmentIndex: 0,
      segId: reference.segment.segId,
      focalSignal: reference.segment.sMean[reference.askedK],
      focalSignalSd: reference.segment.sSd[reference.askedK],
    };
    const untouched = state();
    const loss = expectedPosteriorLoss(
      untouched, nwayProfile, artifact, candidate, reference.segment,
    );
    expect(Math.abs(loss - reference.expectedLoss)).toBeLessThanOrEqual(TOLERANCE);
  });

  it("matches the integrated nine-draw ensemble probabilities, update, and loss", () => {
    expect(ensembleArtifact.artifactId).toBe(integratedReference.artifactId);
    expect(ensembleArtifact.sha256).toBe(integratedReference.artifactSha256);
    const current = state();
    for (let n = 0; n < current.N; n += 1) {
      const probabilities = responseProbabilities(
        integratedNwayProfile, ensembleArtifact, reference.askedK,
        reference.segment, current.t, current.l, n, current.K,
      ).map((entry) => entry.probability);
      expectClose(probabilities, integratedReference.responseProbabilities[n]);
    }
    updateProtocol(
      current, integratedNwayProfile, ensembleArtifact, [reference.segment],
      makeObservation(
        integratedNwayProfile, reference.askedK, 0, reference.wrongPick,
      ),
    );
    expectClose(current.w, integratedReference.updatedWeights);
    expectClose(current.logLik, integratedReference.updatedLogLik);
    const untouched = state();
    const loss = expectedPosteriorLoss(
      untouched,
      integratedNwayProfile,
      ensembleArtifact,
      {
        askedK: reference.askedK,
        segmentIndex: 0,
        segId: reference.segment.segId,
        focalSignal: reference.segment.sMean[reference.askedK],
        focalSignalSd: reference.segment.sSd[reference.askedK],
      },
      reference.segment,
    );
    expect(Math.abs(loss - integratedReference.expectedLoss)).toBeLessThanOrEqual(TOLERANCE);
  });
});
