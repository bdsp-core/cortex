import { describe, expect, it } from "vitest";

import { logPResponse, pResponseYes, signalZ } from "./likelihood";
import { logSumExp, logSumExp2 } from "./mathfns";
import {
  fillResponseProbabilities, IIIC_TASK_INDICES, logObservationProbability,
  makeObservationLikelihoodWorkspace, makeResponseProbabilityWorkspace,
} from "./nway_likelihood";
import { NWAY_ARTIFACT, type ArtifactDraw } from "./nway_profile";
import type { CategoricalParticleObservation, ComputeSegmentMeta } from "./types";

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

function baselineResponseProbabilities(
  askedK: number, segment: ComputeSegmentMeta,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
): Float64Array {
  const offset = particleIndex * K;
  const z = new Float64Array(K);
  const wrong = new Float64Array(K);
  for (const k of IIIC_TASK_INDICES) {
    z[k] = signalZ(
      l[offset + k], t[offset + k], segment.sMean[k], segment.sSd[k],
    );
  }
  const own = pResponseYes(z[askedK]);
  for (const draw of NWAY_ARTIFACT.draws) {
    let maximum = -Infinity;
    for (const k of IIIC_TASK_INDICES) {
      if (k !== askedK) maximum = Math.max(maximum, draw.beta * z[k]);
    }
    let denominator = 0;
    for (const k of IIIC_TASK_INDICES) {
      if (k !== askedK) denominator += Math.exp(draw.beta * z[k] - maximum);
    }
    for (const k of IIIC_TASK_INDICES) {
      if (k === askedK) continue;
      const directed = Math.exp(draw.beta * z[k] - maximum) / denominator;
      wrong[k] += draw.weight * (
        draw.distractorLapse / 5 + (1 - draw.distractorLapse) * directed
      );
    }
  }
  const output = new Float64Array(IIIC_TASK_INDICES.length);
  for (const k of IIIC_TASK_INDICES) {
    output[k - 1] = k === askedK ? own : (1 - own) * wrong[k];
  }
  return output;
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

  it("keeps exact response probabilities with cached particle skill scales", () => {
    const K = 7;
    const particles = 5;
    const t = Float64Array.from({ length: particles * K }, (_, index) =>
      Math.sin(index * 0.731) * 2.4);
    const l = Float64Array.from({ length: particles * K }, (_, index) =>
      Math.cos(index * 1.117) * 1.9);
    const skillScale = Float64Array.from(l, Math.exp);
    const segment: ComputeSegmentMeta = {
      segId: 8123,
      applicableTaskIdx: IIIC_TASK_INDICES.slice(),
      sMean: [-0.7, -1.9, -0.4, 0.1, 0.8, 1.7, 2.6],
      sSd: [0.03, 0.04, 0.08, 0.12, 0.2, 0.35, 0.6],
    };
    const reference = new Float64Array(IIIC_TASK_INDICES.length);
    const cached = new Float64Array(IIIC_TASK_INDICES.length);
    const referenceWorkspace = makeResponseProbabilityWorkspace(K);
    const cachedWorkspace = makeResponseProbabilityWorkspace(K);
    for (let particle = 0; particle < particles; particle++) {
      for (const askedK of IIIC_TASK_INDICES) {
        fillResponseProbabilities(
          "iiic", askedK, segment, t, l, particle, K,
          reference, referenceWorkspace, false,
        );
        fillResponseProbabilities(
          "iiic", askedK, segment, t, l, particle, K,
          cached, cachedWorkspace, false, skillScale,
        );
        expect(Array.from(cached)).toEqual(Array.from(
          baselineResponseProbabilities(askedK, segment, t, l, particle, K),
        ));
        expect(Array.from(cached)).toEqual(Array.from(reference));
      }
    }
  });
});
