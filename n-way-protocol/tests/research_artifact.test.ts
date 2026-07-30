import { describe, expect, it } from "vitest";
import { makeObservation, responseProbabilities } from "../src/likelihood";
import {
  logProbabilityWithArtifactEnsemble, responseProbabilitiesWithArtifactEnsemble,
  type ResearchArtifactEnsemble,
} from "../src/research_artifact";
import { artifact, K, nwayProfile, segments } from "./fixtures";

const ensemble: ResearchArtifactEnsemble = {
  baseArtifact: artifact,
  draws: [
    { beta: 0.82, distractorLapse: 0.03, weight: 1 },
    { beta: 1.19, distractorLapse: 0.00, weight: 3 },
  ],
};

describe("research artifact ensemble", () => {
  it("equals the manually weighted categorical probability mixture", () => {
    const segment = segments(1)[0];
    const t = Float64Array.from([0.1, -0.2, 0.3, -0.1, 0.4, -0.5, 0.2]);
    const l = Float64Array.from([0.2, 0.1, -0.3, 0.4, 0, -0.1, 0.5]);
    const actual = responseProbabilitiesWithArtifactEnsemble(
      nwayProfile, ensemble, 3, segment, t, l, 0, K,
    );
    const components = ensemble.draws.map((draw) => responseProbabilities(
      nwayProfile, { ...artifact, ...draw }, 3, segment, t, l, 0, K,
    ));
    for (let index = 0; index < actual.length; index += 1) {
      const expected = 0.25 * components[0][index].probability
        + 0.75 * components[1][index].probability;
      expect(actual[index].probability).toBeCloseTo(expected, 14);
    }
    expect(actual.reduce((sum, entry) => sum + entry.probability, 0)).toBeCloseTo(1, 14);
  });

  it("preserves the focal marginal and mixes likelihoods on probability scale", () => {
    const segment = segments(1)[0];
    const t = Float64Array.from([0.1, -0.2, 0.3, -0.1, 0.4, -0.5, 0.2]);
    const l = Float64Array.from([0.2, 0.1, -0.3, 0.4, 0, -0.1, 0.5]);
    const probabilities = responseProbabilitiesWithArtifactEnsemble(
      nwayProfile, ensemble, 3, segment, t, l, 0, K,
    );
    const scalar = responseProbabilities(
      nwayProfile, artifact, 3, segment, t, l, 0, K,
    );
    expect(probabilities.find(({ outcome }) => outcome === 3)!.probability)
      .toBe(scalar.find(({ outcome }) => outcome === 3)!.probability);
    const observation = makeObservation(nwayProfile, 3, 0, 5);
    const probability = probabilities.find(({ outcome }) => outcome === 5)!.probability;
    expect(Math.exp(logProbabilityWithArtifactEnsemble(
      nwayProfile, ensemble, observation, segment, t, l, 0, K,
    ))).toBeCloseTo(probability, 14);
  });

  it("rejects invalid draws", () => {
    expect(() => responseProbabilitiesWithArtifactEnsemble(
      nwayProfile, { ...ensemble, draws: [] }, 3, segments(1)[0],
      new Float64Array(K), new Float64Array(K), 0, K,
    )).toThrow(/must contain a draw/);
    expect(() => responseProbabilitiesWithArtifactEnsemble(
      nwayProfile,
      { ...ensemble, draws: [{ beta: 1, distractorLapse: 1.1 }] },
      3, segments(1)[0], new Float64Array(K), new Float64Array(K), 0, K,
    )).toThrow(/lapse/);
  });
});
