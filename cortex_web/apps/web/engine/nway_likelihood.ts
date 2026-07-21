import { logSumExp2 } from "./mathfns";
import { LAPSE_RATE, logPResponse, pResponseYes, signalZ } from "./likelihood";
import { NWAY_ARTIFACT, type ArtifactDraw } from "./nway_profile";
import type {
  BinaryParticleObservation, CategoricalParticleObservation,
  ComputeSegmentMeta, ParticleObservation,
} from "./types";

export const IIIC_TASK_INDICES = Object.freeze([1, 2, 3, 4, 5, 6]);

function zFor(
  taskK: number, sMean: readonly number[], sSd: readonly number[],
  t: Float64Array, l: Float64Array, particleOffset: number,
  skillScale?: Float64Array,
): number {
  if (skillScale) {
    const scale = skillScale[particleOffset + taskK];
    let z = scale * (sMean[taskK] + t[particleOffset + taskK]);
    if (sSd[taskK] !== 0) {
      z /= Math.sqrt(1 + (scale * sSd[taskK]) ** 2);
    }
    return z;
  }
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

export interface ScreeningJacobianWorkspace extends ResponseProbabilityWorkspace {
  biasJacobian: Float64Array;
  skillJacobian: Float64Array;
  signalBiasDerivative: Float64Array;
  signalSkillDerivative: Float64Array;
}

/** Reusable selector workspace. Matrix rows follow configured task indices and
 * columns follow the response vector, so adding non-categorical domains only
 * changes K and does not require a new hot-loop implementation. */
export function makeScreeningJacobianWorkspace(
  K: number, maximumOutcomeCount = IIIC_TASK_INDICES.length,
): ScreeningJacobianWorkspace {
  return {
    ...makeResponseProbabilityWorkspace(K),
    biasJacobian: new Float64Array(K * maximumOutcomeCount),
    skillJacobian: new Float64Array(K * maximumOutcomeCount),
    signalBiasDerivative: new Float64Array(K),
    signalSkillDerivative: new Float64Array(K),
  };
}

/** Allocation-free selector hot path; outcome order is [1..6] for IIIC and
 * [askedK,K-sentinel] for spike. `screening=true` uses only the frozen
 * moment-matched draw and is forbidden for the final objective/update. */
export function fillResponseProbabilities(
  taskClass: "iiic" | "spike", askedK: number, segment: ComputeSegmentMeta,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
  output: Float64Array, workspace: ResponseProbabilityWorkspace,
  screening = false,
  skillScale?: Float64Array,
): void {
  const offset = particleIndex * K;
  if (taskClass === "spike") {
    if (output.length !== 2) throw new Error("spike response output must have length two");
    const yes = pResponseYes(zFor(
      askedK, segment.sMean, segment.sSd, t, l, offset, skillScale,
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
    workspace.z[k] = zFor(
      k, segment.sMean, segment.sSd, t, l, offset, skillScale,
    );
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

const NORMAL_PDF_SCALE = 1 / Math.sqrt(2 * Math.PI);

function fillSignalAndDerivatives(
  k: number, segment: ComputeSegmentMeta,
  t: Float64Array, l: Float64Array, offset: number,
  workspace: ScreeningJacobianWorkspace,
): void {
  const skillScale = Math.exp(l[offset + k]);
  const signalSd = segment.sSd[k];
  const attenuationSquared = (skillScale * signalSd) ** 2;
  const attenuation = Math.sqrt(1 + attenuationSquared);
  const z = skillScale * (segment.sMean[k] + t[offset + k]) / attenuation;
  workspace.z[k] = z;
  workspace.signalBiasDerivative[k] = skillScale / attenuation;
  workspace.signalSkillDerivative[k] = z / (1 + attenuationSquared);
}

function responseDerivative(z: number, signalDerivative: number): number {
  return (1 - 2 * LAPSE_RATE) * NORMAL_PDF_SCALE
    * Math.exp(-0.5 * z * z) * signalDerivative;
}

/** Fill the frozen Fisher-screen response vector and its analytical Jacobians.
 * Jacobian entry `(k * outcomeCount + r)` is d P(outcome r) / d t_k or d l_k.
 * The response probabilities themselves deliberately use the existing frozen
 * implementation so only the derivative calculation changes. */
export function fillScreeningProbabilitiesAndJacobians(
  taskClass: "iiic" | "spike", askedK: number, segment: ComputeSegmentMeta,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
  output: Float64Array, workspace: ScreeningJacobianWorkspace,
): void {
  const outcomeCount = taskClass === "spike" ? 2 : IIIC_TASK_INDICES.length;
  if (output.length !== outcomeCount) {
    throw new Error(`screening response output must have length ${outcomeCount}`);
  }
  const matrixLength = K * outcomeCount;
  if (workspace.biasJacobian.length < matrixLength
      || workspace.skillJacobian.length < matrixLength) {
    throw new Error("screening Jacobian workspace is too small");
  }
  fillResponseProbabilities(
    taskClass, askedK, segment, t, l, particleIndex, K,
    output, workspace, true,
  );
  workspace.biasJacobian.fill(0, 0, matrixLength);
  workspace.skillJacobian.fill(0, 0, matrixLength);
  const offset = particleIndex * K;
  if (taskClass === "spike") {
    fillSignalAndDerivatives(askedK, segment, t, l, offset, workspace);
    const bias = responseDerivative(
      workspace.z[askedK], workspace.signalBiasDerivative[askedK],
    );
    const skill = responseDerivative(
      workspace.z[askedK], workspace.signalSkillDerivative[askedK],
    );
    const row = askedK * outcomeCount;
    workspace.biasJacobian[row] = bias;
    workspace.biasJacobian[row + 1] = -bias;
    workspace.skillJacobian[row] = skill;
    workspace.skillJacobian[row + 1] = -skill;
    return;
  }

  for (const k of IIIC_TASK_INDICES) {
    fillSignalAndDerivatives(k, segment, t, l, offset, workspace);
  }
  const ownProbability = output[askedK - 1];
  const draw = SCREEN_DRAW[0];
  let maximum = -Infinity;
  for (const k of IIIC_TASK_INDICES) {
    if (k !== askedK) maximum = Math.max(maximum, draw.beta * workspace.z[k]);
  }
  let denominator = 0;
  for (const k of IIIC_TASK_INDICES) {
    if (k === askedK) continue;
    const softmax = Math.exp(draw.beta * workspace.z[k] - maximum);
    workspace.wrong[k] = softmax;
    denominator += softmax;
  }
  for (const k of IIIC_TASK_INDICES) {
    if (k !== askedK) workspace.wrong[k] /= denominator;
  }

  const ownBiasDerivative = responseDerivative(
    workspace.z[askedK], workspace.signalBiasDerivative[askedK],
  );
  const ownSkillDerivative = responseDerivative(
    workspace.z[askedK], workspace.signalSkillDerivative[askedK],
  );
  const askedRow = askedK * outcomeCount;
  workspace.biasJacobian[askedRow + askedK - 1] = ownBiasDerivative;
  workspace.skillJacobian[askedRow + askedK - 1] = ownSkillDerivative;
  for (const k of IIIC_TASK_INDICES) {
    if (k === askedK) continue;
    const outcome = k - 1;
    const wrongGivenIncorrect = output[outcome] / (1 - ownProbability);
    workspace.biasJacobian[askedRow + outcome] =
      -ownBiasDerivative * wrongGivenIncorrect;
    workspace.skillJacobian[askedRow + outcome] =
      -ownSkillDerivative * wrongGivenIncorrect;
  }

  const directedScale = (1 - ownProbability) * (1 - draw.distractorLapse) * draw.beta;
  for (const parameterK of IIIC_TASK_INDICES) {
    if (parameterK === askedK) continue;
    const parameterSoftmax = workspace.wrong[parameterK];
    const biasScale = directedScale * workspace.signalBiasDerivative[parameterK]
      * parameterSoftmax;
    const skillScale = directedScale * workspace.signalSkillDerivative[parameterK]
      * parameterSoftmax;
    const row = parameterK * outcomeCount;
    for (const outcomeK of IIIC_TASK_INDICES) {
      if (outcomeK === askedK) continue;
      const contrast = outcomeK === parameterK
        ? 1 - parameterSoftmax : -workspace.wrong[outcomeK];
      const outcome = outcomeK - 1;
      workspace.biasJacobian[row + outcome] = biasScale * contrast;
      workspace.skillJacobian[row + outcome] = skillScale * contrast;
    }
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
