import { logSumExp2 } from "./mathfns";
import { logPResponse, pResponseYes, signalZ } from "./likelihood";
import { NWAY_ARTIFACT, type ArtifactDraw } from "./nway_profile";
import type {
  BinaryParticleObservation, CategoricalParticleObservation,
  ComputeSegmentMeta, ParticleObservation,
} from "./types";

export const IIIC_TASK_INDICES = Object.freeze([1, 2, 3, 4, 5, 6]);

function zFor(
  taskK: number, sMean: readonly number[], sSd: readonly number[],
  t: Float64Array, l: Float64Array, particleOffset: number,
): number {
  return signalZ(
    l[particleOffset + taskK], t[particleOffset + taskK],
    sMean[taskK], sSd[taskK],
  );
}

function logDistractorProbability(
  askedK: number, pickK: number,
  sMean: ArrayLike<number>, sSd: ArrayLike<number>, signalOffset: number,
  t: Float64Array, l: Float64Array, particleOffset: number,
  draw: ArtifactDraw,
  workspace: ObservationLikelihoodWorkspace,
): number {
  if (pickK === askedK
      || pickK < IIIC_TASK_INDICES[0]
      || pickK > IIIC_TASK_INDICES[IIIC_TASK_INDICES.length - 1]) {
    throw new Error("categorical pick is not a valid IIIC distractor");
  }
  let maximum = -Infinity;
  let pickedLogit = -Infinity;
  let distractorIndex = 0;
  for (const k of IIIC_TASK_INDICES) {
    if (k === askedK) continue;
    const signalK = signalOffset + k;
    const logit = draw.beta * signalZ(
      l[particleOffset + k], t[particleOffset + k], sMean[signalK], sSd[signalK],
    );
    workspace.distractorLogits[distractorIndex++] = logit;
    if (logit > maximum) maximum = logit;
    if (k === pickK) pickedLogit = logit;
  }
  let denominator = 0;
  for (let i = 0; i < workspace.distractorLogits.length; i++) {
    denominator += Math.exp(workspace.distractorLogits[i] - maximum);
  }
  const logSoftmax = pickedLogit - (maximum + Math.log(denominator));
  const uniform = draw.distractorLapse === 0
    ? -Infinity
    : Math.log(draw.distractorLapse) - Math.log(IIIC_TASK_INDICES.length - 1);
  const directed = draw.distractorLapse === 1
    ? -Infinity
    : Math.log1p(-draw.distractorLapse) + logSoftmax;
  return logSumExp2(uniform, directed);
}

export interface ObservationLikelihoodWorkspace {
  mixture: Float64Array;
  distractorLogits: Float64Array;
}

export function makeObservationLikelihoodWorkspace(): ObservationLikelihoodWorkspace {
  return {
    mixture: new Float64Array(NWAY_ARTIFACT.draws.length),
    distractorLogits: new Float64Array(IIIC_TASK_INDICES.length - 1),
  };
}

function logSumExpMixture(values: Float64Array): number {
  let maximum = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > maximum) maximum = values[i];
  }
  if (maximum === -Infinity) return -Infinity;
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += Math.exp(values[i] - maximum);
  return maximum + Math.log(sum);
}

export function logCategoricalObservationProbability(
  askedK: number, pickK: number,
  sMean: ArrayLike<number>, sSd: ArrayLike<number>, signalOffset: number,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
  workspace = makeObservationLikelihoodWorkspace(),
): number {
  const particleOffset = particleIndex * K;
  const ownSignalK = signalOffset + askedK;
  const ownZ = signalZ(
    l[particleOffset + askedK], t[particleOffset + askedK],
    sMean[ownSignalK], sSd[ownSignalK],
  );
  if (pickK === askedK) return logPResponse(ownZ, 1);
  for (let i = 0; i < NWAY_ARTIFACT.draws.length; i++) {
    const draw = NWAY_ARTIFACT.draws[i];
    workspace.mixture[i] = Math.log(draw.weight) + logDistractorProbability(
      askedK, pickK, sMean, sSd, signalOffset,
      t, l, particleOffset, draw, workspace,
    );
  }
  return logPResponse(ownZ, 0) + logSumExpMixture(workspace.mixture);
}

export function logObservationProbability(
  observation: ParticleObservation,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
  workspace = makeObservationLikelihoodWorkspace(),
): number {
  const offset = particleIndex * K;
  if (observation.kind === "binary") {
    return logPResponse(
      signalZ(l[offset + observation.k], t[offset + observation.k],
        observation.s, observation.sSd),
      observation.y,
    );
  }
  return logCategoricalObservationProbability(
    observation.askedK, observation.pickK,
    observation.sMean, observation.sSd, 0,
    t, l, particleIndex, K, workspace,
  );
}

export function makeResponseObservation(
  askedK: number, segment: ComputeSegmentMeta, rawPick: number,
  taskClass: "iiic" | "spike",
): ParticleObservation {
  if (taskClass === "spike") {
    if (rawPick !== askedK && rawPick !== segment.sMean.length) {
      throw new Error(`spike response must be task ${askedK} or no sentinel`);
    }
    return {
      kind: "binary", k: askedK, s: segment.sMean[askedK],
      sSd: segment.sSd[askedK], y: rawPick === askedK ? 1 : 0, rawPick,
    } satisfies BinaryParticleObservation;
  }
  if (!IIIC_TASK_INDICES.includes(rawPick)) {
    throw new Error(`IIIC response ${rawPick} is outside the six-way registry`);
  }
  return {
    kind: "categorical_f1", askedK, pickK: rawPick,
    sMean: segment.sMean.slice(), sSd: segment.sSd.slice(),
  } satisfies CategoricalParticleObservation;
}

export interface ResponseProbabilityWorkspace {
  z: Float64Array;
  wrong: Float64Array;
}

export function makeResponseProbabilityWorkspace(K: number): ResponseProbabilityWorkspace {
  return { z: new Float64Array(K), wrong: new Float64Array(K) };
}

/** Allocation-free selector hot path; outcome order is [1..6] for IIIC and
 * [askedK,K-sentinel] for spike. `screening=true` uses only the frozen
 * moment-matched draw and is forbidden for the final objective/update. */
export function fillResponseProbabilities(
  taskClass: "iiic" | "spike", askedK: number, segment: ComputeSegmentMeta,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
  output: Float64Array, workspace: ResponseProbabilityWorkspace,
  screening = false,
): void {
  const offset = particleIndex * K;
  if (taskClass === "spike") {
    if (output.length !== 2) throw new Error("spike response output must have length two");
    const yes = pResponseYes(signalZ(
      l[offset + askedK], t[offset + askedK],
      segment.sMean[askedK], segment.sSd[askedK],
    ));
    output[0] = yes;
    output[1] = 1 - yes;
    return;
  }
  if (output.length !== IIIC_TASK_INDICES.length) {
    throw new Error("IIIC response output must have length six");
  }
  output.fill(0);
  workspace.wrong.fill(0);
  for (const k of IIIC_TASK_INDICES) {
    workspace.z[k] = zFor(k, segment.sMean, segment.sSd, t, l, offset);
  }
  const own = pResponseYes(workspace.z[askedK]);
  const draws = screening ? SCREEN_DRAW : NWAY_ARTIFACT.draws;
  for (const draw of draws) {
    let maximum = -Infinity;
    for (const k of IIIC_TASK_INDICES) {
      if (k !== askedK) maximum = Math.max(maximum, draw.beta * workspace.z[k]);
    }
    let denominator = 0;
    for (const k of IIIC_TASK_INDICES) {
      if (k !== askedK) denominator += Math.exp(draw.beta * workspace.z[k] - maximum);
    }
    for (const k of IIIC_TASK_INDICES) {
      if (k === askedK) continue;
      const directed = Math.exp(draw.beta * workspace.z[k] - maximum) / denominator;
      workspace.wrong[k] += draw.weight * (
        draw.distractorLapse / 5 + (1 - draw.distractorLapse) * directed
      );
    }
  }
  for (const k of IIIC_TASK_INDICES) {
    output[k - 1] = k === askedK ? own : (1 - own) * workspace.wrong[k];
  }
}

export function responseProbabilities(
  taskClass: "iiic" | "spike", askedK: number, segment: ComputeSegmentMeta,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
): { outcome: number; probability: number }[] {
  const output = new Float64Array(taskClass === "spike" ? 2 : 6);
  fillResponseProbabilities(
    taskClass, askedK, segment, t, l, particleIndex, K,
    output, makeResponseProbabilityWorkspace(K), false,
  );
  return Array.from(output, (probability, index) => ({
    outcome: taskClass === "spike" ? (index === 0 ? askedK : K) : index + 1,
    probability,
  }));
}

const SCREEN_DRAW: readonly ArtifactDraw[] = Object.freeze([{
  beta: NWAY_ARTIFACT.draws.reduce((sum, draw) => sum + draw.weight * draw.beta, 0),
  distractorLapse: NWAY_ARTIFACT.draws.reduce(
    (sum, draw) => sum + draw.weight * draw.distractorLapse, 0,
  ),
  weight: 1,
}]);
