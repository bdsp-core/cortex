import { describe, expect, it } from "vitest";
import {
  binaryPYes, logPBinary, logProbability, makeObservation, responseProbabilities,
  signalZ,
} from "../src/likelihood";
import { artifact, K, nwayProfile, segments } from "./fixtures";

describe("conditional F1 response likelihood", () => {
  it("preserves the focal binary marginal exactly", () => {
    const segment = segments(1)[0];
    const t = Float64Array.from([0.1, -0.2, 0.3, -0.1, 0.4, -0.5, 0.2]);
    const l = Float64Array.from([0.2, 0.1, -0.3, 0.4, 0, -0.1, 0.5]);
    const askedK = 3;
    const probabilities = responseProbabilities(
      nwayProfile, artifact, askedK, segment, t, l, 0, K,
    );
    const own = probabilities.find((entry) => entry.outcome === askedK)!.probability;
    const z = signalZ(l[askedK], t[askedK], segment.sMean[askedK], segment.sSd[askedK]);
    expect(own).toBe(binaryPYes(z));
    expect(Math.log(own)).toBeCloseTo(logPBinary(z, 1), 13);
    expect(probabilities.reduce((sum, entry) => sum + entry.probability, 0)).toBeCloseTo(1, 14);
  });

  it("uses the identity of a wrong pick and remains finite in extreme tails", () => {
    const segment = segments(1)[0];
    const t = Float64Array.from([0, -20, 20, -10, 10, -5, 5]);
    const l = Float64Array.from([0, 5, 5, 4, 4, 3, 3]);
    const values = [2, 3, 4, 5, 6].map((pick) => logProbability(
      nwayProfile, artifact, makeObservation(nwayProfile, 1, 0, pick),
      segment, t, l, 0, K,
    ));
    expect(values.every(Number.isFinite)).toBe(true);
    expect(new Set(values.map((value) => value.toFixed(8))).size).toBeGreaterThan(1);
  });

  it("reduces the distractor layer to uniform when its lapse is one", () => {
    const uniformArtifact = { ...artifact, distractorLapse: 1 };
    const segment = segments(1)[0];
    const t = new Float64Array(K);
    const l = new Float64Array(K);
    const probabilities = responseProbabilities(
      nwayProfile, uniformArtifact, 1, segment, t, l, 0, K,
    );
    const wrong = probabilities.filter((entry) => entry.outcome !== 1).map((entry) => entry.probability);
    for (const probability of wrong) expect(probability).toBeCloseTo(wrong[0], 14);
  });

  it("rejects a categorical pick outside the registered IIIC group", () => {
    expect(() => makeObservation(nwayProfile, 2, 0, 0)).toThrow(/outside response group/);
  });
});

