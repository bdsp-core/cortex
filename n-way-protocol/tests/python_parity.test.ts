import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { normalizedArtifactDraws } from "../src/artifact";
import { logProbability, makeObservation, responseProbabilities } from "../src/likelihood";
import { atomPosterior, updateProtocol } from "../src/particles";
import { expectedPosteriorLoss } from "../src/selector";
import type {
  Candidate, ConditionalF1ArtifactEnsemble, EngineProfile, ProtocolParticleState,
  ProtocolSegment,
} from "../src/types";
import {
  artifact, ensembleArtifact, integratedNwayProfile, nwayProfile,
} from "./fixtures";

const reference = JSON.parse(readFileSync(fileURLToPath(new URL(
  "./fixtures/python_reference.json", import.meta.url,
)), "utf8"));
const integratedReference = JSON.parse(readFileSync(fileURLToPath(new URL(
  "./fixtures/python_integrated_reference.json", import.meta.url,
)), "utf8"));
const drawLatentReference = JSON.parse(readFileSync(fileURLToPath(new URL(
  "./fixtures/python_draw_latent_reference.json", import.meta.url,
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

const drawLatentArtifact: ConditionalF1ArtifactEnsemble = {
  schemaVersion: 1,
  artifactId: "draw-latent-parity-fixture",
  model: "iiic_conditional_f1_v1_artifact_ensemble",
  qualification: "exploratory_unqualified",
  draws: drawLatentReference.atoms.map(
    ([beta, distractorLapse, weight]: [number, number, number]) => (
      { beta, distractorLapse, weight }
    ),
  ),
  binaryLapse: 0.025,
  sha256: "d".repeat(64),
  provenance: {
    source: "scripts/make_draw_latent_fixture.py (draw_latent_rd oracle)",
    fitSplit: "deterministic fixed cloud; not promotion eligible",
    dr07: "not_qualified",
  },
};

const drawLatentProfile: EngineProfile = {
  ...nwayProfile,
  responseArtifactId: drawLatentArtifact.artifactId,
  responseArtifactSha256: drawLatentArtifact.sha256,
  responseAggregation: "draw_latent",
};

function drawLatentState(): ProtocolParticleState {
  return {
    ...state(),
    N: drawLatentReference.t.length,
    t: flatten(drawLatentReference.t),
    l: flatten(drawLatentReference.l),
    w: Float64Array.from(drawLatentReference.initialWeights),
    logPrior: new Float64Array(drawLatentReference.t.length),
    logLik: new Float64Array(drawLatentReference.t.length),
    atomIndex: Int32Array.from(drawLatentReference.atomAssignment),
  };
}

describe("independent Python/TypeScript draw-latent parity", () => {
  it("matches the oracle's per-atom trajectory and atom posterior", () => {
    const current = drawLatentState();
    const bank: ProtocolSegment[] = drawLatentReference.segments;
    const draws = normalizedArtifactDraws(drawLatentArtifact);
    for (const step of drawLatentReference.steps) {
      const observation = makeObservation(
        drawLatentProfile, step.observation.askedK,
        step.observation.segmentIndex, step.observation.rawPick,
      );
      expect(observation.kind).toBe(step.observation.kind);
      const segment = bank[observation.segmentIndex];
      const stepLogLik = Array.from({ length: current.N }, (_, n) => (
        logProbability(
          drawLatentProfile, drawLatentArtifact, observation, segment,
          current.t, current.l, n, current.K,
          draws[current.atomIndex![n]],
        )
      ));
      expectClose(stepLogLik, step.stepLogLik);
      updateProtocol(current, drawLatentProfile, drawLatentArtifact, bank, observation);
      expectClose(current.w, step.weights);
      expectClose(current.logLik, step.logLik);
    }
    expectClose(
      atomPosterior(current, drawLatentArtifact),
      drawLatentReference.atomPosterior,
    );
  });

  it("resolves atoms away from the fixed-weight mixture on the same trajectory", () => {
    // Sanity guard for the fixture itself: with three distinct atoms the
    // draw-latent posterior must differ from the mixture posterior, so the
    // parity above cannot silently pass through the mixture code path.
    const mixture = drawLatentState();
    delete mixture.atomIndex;
    const mixtureProfile: EngineProfile = {
      ...drawLatentProfile, responseAggregation: "mixture",
    };
    for (const step of drawLatentReference.steps) {
      updateProtocol(
        mixture, mixtureProfile, drawLatentArtifact,
        drawLatentReference.segments,
        makeObservation(
          mixtureProfile, step.observation.askedK,
          step.observation.segmentIndex, step.observation.rawPick,
        ),
      );
    }
    const final = drawLatentReference.steps[drawLatentReference.steps.length - 1];
    const drift = Math.max(...Array.from(
      mixture.w, (weight, n) => Math.abs(weight - final.weights[n]),
    ));
    expect(drift).toBeGreaterThan(1e-3);
  });
});
