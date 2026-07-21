import { describe, expect, it } from "vitest";

import { logPResponse, signalZ } from "./likelihood";
import { logSumExp, logSumExp2 } from "./mathfns";
import {
  IIIC_TASK_INDICES, logObservationProbability, makeObservationLikelihoodWorkspace,
} from "./nway_likelihood";
import { NWAY_ARTIFACT, type ArtifactDraw } from "./nway_profile";
import type { CategoricalParticleObservation } from "./types";

function baselineDistractor(
  observation: CategoricalParticleObservation,
  t: Float64Array, l: Float64Array, offset: number, draw: ArtifactDraw,
): number {
  const distractors = IIIC_TASK_INDICES.filter((k) => k !== observation.askedK);
  const logits = distractors.map((k) => draw.beta * signalZ(
    l[offset + k], t[offset + k], observation.sMean[k], observation.sSd[k],
  ));
  const pickIndex = distractors.indexOf(observation.pickK);
  if (pickIndex < 0) throw new Error("invalid test distractor");
  const logSoftmax = logits[pickIndex] - logSumExp(logits);
  const uniform = draw.distractorLapse === 0
    ? -Infinity
    : Math.log(draw.distractorLapse) - Math.log(distractors.length);
  const directed = draw.distractorLapse === 1
    ? -Infinity
    : Math.log1p(-draw.distractorLapse) + logSoftmax;
  return logSumExp2(uniform, directed);
}

function baselineProbability(
  observation: CategoricalParticleObservation,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
): number {
  const offset = particleIndex * K;
  const ownZ = signalZ(
    l[offset + observation.askedK], t[offset + observation.askedK],
    observation.sMean[observation.askedK], observation.sSd[observation.askedK],
  );
  if (observation.pickK === observation.askedK) return logPResponse(ownZ, 1);
  const mixture = NWAY_ARTIFACT.draws.map((draw) => (
    Math.log(draw.weight) + baselineDistractor(observation, t, l, offset, draw)
  ));
  return logPResponse(ownZ, 0) + logSumExp(mixture);
}

describe("allocation-free categorical likelihood", () => {
  it("is bit-exact with the frozen allocation-based implementation", () => {
    const K = 7;
    const t = Float64Array.from({ length: 6 * K }, (_, index) =>
      Math.sin(index * 1.731) * (index % 5 === 0 ? 4.5 : 1.2));
    const l = Float64Array.from({ length: 6 * K }, (_, index) =>
      Math.cos(index * 0.917) * (index % 7 === 0 ? 5.1 : 1.4));
    const sMean = [-0.7, -1.9, -0.4, 0.1, 0.8, 1.7, 2.6];
    const sSd = [0.03, 0.04, 0.08, 0.12, 0.2, 0.35, 0.6];
    const workspace = makeObservationLikelihoodWorkspace();
    for (let particle = 0; particle < 6; particle++) {
      for (const askedK of IIIC_TASK_INDICES) {
        for (const pickK of IIIC_TASK_INDICES) {
          const observation: CategoricalParticleObservation = {
            kind: "categorical_f1", askedK, pickK, sMean, sSd,
          };
          expect(logObservationProbability(
            observation, t, l, particle, K, workspace,
          )).toBe(baselineProbability(observation, t, l, particle, K));
        }
      }
    }
  });
});
