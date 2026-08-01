import { logSumExp2 } from "./mathfns";
import {
  LAPSE_RATE, logPResponse, pResponseYes, signalZ, signalZFromScale,
} from "./likelihood";
import {
  NWAY_ARTIFACT, NWAY_DRAW_LATENT_ARTIFACT, NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT,
  nwayResponseAggregationOf,
  type ArtifactDraw,
} from "./nway_profile";
import type {
  BinaryParticleObservation, CategoricalParticleObservation,
  ComputeEngineInputs, ComputeSegmentMeta, NWayPreparedAtom,
  NWayResponseAggregation, NWayResponseRuntime, ParticleObservation,
} from "./types";

export const IIIC_TASK_INDICES = Object.freeze([1, 2, 3, 4, 5, 6]);

/** Observation-invariant log-domain terms for one artifact draw. Shared by the
 * frozen history table and profile-gated runtimes so both aggregation modes
 * run the identical arithmetic. */
export function prepareResponseDraw(draw: ArtifactDraw): NWayPreparedAtom {
  return {
    ...draw,
    logWeight: Math.log(draw.weight),
    uniformLogProbability: draw.distractorLapse === 0
      ? -Infinity
      : Math.log(draw.distractorLapse) - Math.log(IIIC_TASK_INDICES.length - 1),
    directedLogProbability: draw.distractorLapse === 1
      ? -Infinity
      : Math.log1p(-draw.distractorLapse),
  };
}

// These terms are invariant across every particle, observation, and MH step.
// Keep them on the log-domain history path only: the selector mixes response
// probabilities and intentionally retains its existing arithmetic.
const HISTORY_DRAWS: readonly NWayPreparedAtom[] = Object.freeze(
  NWAY_ARTIFACT.draws.map(prepareResponseDraw),
);

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
  draw: NWayPreparedAtom,
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
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
    if (k === askedK) continue;
    const logit = draw.beta * workspace.distractorSignals[distractorIndex];
    workspace.distractorLogits[distractorIndex++] = logit;
    if (logit > maximum) maximum = logit;
    if (k === pickK) pickedLogit = logit;
  }
  let denominator = 0;
  for (let i = 0; i < workspace.distractorLogits.length; i++) {
    denominator += Math.exp(workspace.distractorLogits[i] - maximum);
  }
  const logSoftmax = pickedLogit - (maximum + Math.log(denominator));
  // Seven of the nine frozen draws have exactly zero distractor lapse. For
  // those draws logSumExp(-Infinity, logSoftmax) is exactly logSoftmax, so the
  // generic mixture and its transcendental calls are pure repeated work.
  if (draw.distractorLapse === 0) return logSoftmax;
  return logSumExp2(
    draw.uniformLogProbability,
    draw.directedLogProbability + logSoftmax,
  );
}

export interface ObservationLikelihoodWorkspace {
  mixture: Float64Array;
  distractorSignals: Float64Array;
  distractorLogits: Float64Array;
}

export function makeObservationLikelihoodWorkspace(
  mixtureSize = NWAY_ARTIFACT.draws.length,
): ObservationLikelihoodWorkspace {
  return {
    mixture: new Float64Array(mixtureSize),
    distractorSignals: new Float64Array(IIIC_TASK_INDICES.length - 1),
    distractorLogits: new Float64Array(IIIC_TASK_INDICES.length - 1),
  };
}

function historySignalZ(
  taskK: number,
  sMean: ArrayLike<number>, sSd: ArrayLike<number>, signalOffset: number,
  t: Float64Array, l: Float64Array, particleOffset: number,
  skillScale?: Float64Array,
): number {
  const signalK = signalOffset + taskK;
  return skillScale
    ? signalZFromScale(
        skillScale[particleOffset + taskK], t[particleOffset + taskK],
        sMean[signalK], sSd[signalK],
      )
    : signalZ(
        l[particleOffset + taskK], t[particleOffset + taskK],
        sMean[signalK], sSd[signalK],
      );
}

function logSumExpMixture(values: Float64Array, count = values.length): number {
  let maximum = -Infinity;
  for (let i = 0; i < count; i++) {
    if (values[i] > maximum) maximum = values[i];
  }
  if (maximum === -Infinity) return -Infinity;
  let sum = 0;
  for (let i = 0; i < count; i++) sum += Math.exp(values[i] - maximum);
  return maximum + Math.log(sum);
}

export function logCategoricalObservationProbability(
  askedK: number, pickK: number,
  sMean: ArrayLike<number>, sSd: ArrayLike<number>, signalOffset: number,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
  workspace = makeObservationLikelihoodWorkspace(),
  skillScale?: Float64Array,
  runtime?: NWayResponseRuntime,
  atomDraw?: NWayPreparedAtom,
): number {
  const particleOffset = particleIndex * K;
  const ownZ = historySignalZ(
    askedK, sMean, sSd, signalOffset, t, l, particleOffset, skillScale,
  );
  if (pickK === askedK) return logPResponse(ownZ, 1);
  let distractorIndex = 0;
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
    if (k === askedK) continue;
    workspace.distractorSignals[distractorIndex++] = historySignalZ(
      k, sMean, sSd, signalOffset, t, l, particleOffset, skillScale,
    );
  }
  if (atomDraw) {
    // Draw-latent aggregation: this particle scores the pick under its own
    // atom's (beta, distractorLapse), never the fixed-weight mixture average.
    return logPResponse(ownZ, 0) + logDistractorProbability(
      askedK, pickK, atomDraw, workspace,
    );
  }
  const draws = runtime ? runtime.atoms : HISTORY_DRAWS;
  if (workspace.mixture.length < draws.length) {
    throw new Error("observation likelihood workspace is too small for the mixture");
  }
  for (let i = 0; i < draws.length; i++) {
    const draw = draws[i];
    workspace.mixture[i] = draw.logWeight + logDistractorProbability(
      askedK, pickK, draw, workspace,
    );
  }
  return logPResponse(ownZ, 0) + logSumExpMixture(workspace.mixture, draws.length);
}

export function logObservationProbability(
  observation: ParticleObservation,
  t: Float64Array, l: Float64Array, particleIndex: number, K: number,
  workspace = makeObservationLikelihoodWorkspace(),
  runtime?: NWayResponseRuntime,
  atomDraw?: NWayPreparedAtom,
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
    t, l, particleIndex, K, workspace, undefined, runtime, atomDraw,
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
  distractorExponentials: Float64Array;
}

export function makeResponseProbabilityWorkspace(K: number): ResponseProbabilityWorkspace {
  return {
    z: new Float64Array(K),
    wrong: new Float64Array(K),
    distractorExponentials: new Float64Array(K),
  };
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
  runtime?: NWayResponseRuntime,
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
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
    workspace.z[k] = zFor(
      k, segment.sMean, segment.sSd, t, l, offset, skillScale,
    );
  }
  const own = pResponseYes(workspace.z[askedK]);
  const draws = screening
    ? (runtime ? runtime.screen : SCREEN_DRAW)
    : (runtime ? runtime.atoms : NWAY_ARTIFACT.draws);
  for (let drawIndex = 0; drawIndex < draws.length; drawIndex++) {
    const draw = draws[drawIndex];
    const beta = draw.beta;
    const weight = draw.weight;
    const distractorLapse = draw.distractorLapse;
    const uniformShare = distractorLapse / 5;
    const directedShare = 1 - distractorLapse;
    let maximum = -Infinity;
    for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
      const k = IIIC_TASK_INDICES[taskIndex];
      if (k !== askedK) maximum = Math.max(maximum, beta * workspace.z[k]);
    }
    let denominator = 0;
    for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
      const k = IIIC_TASK_INDICES[taskIndex];
      if (k === askedK) continue;
      const exponential = Math.exp(beta * workspace.z[k] - maximum);
      workspace.distractorExponentials[k] = exponential;
      denominator += exponential;
    }
    for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
      const k = IIIC_TASK_INDICES[taskIndex];
      if (k === askedK) continue;
      const directed = workspace.distractorExponentials[k] / denominator;
      workspace.wrong[k] += weight * (uniformShare + directedShare * directed);
    }
  }
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
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
  runtime?: NWayResponseRuntime,
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
    output, workspace, true, undefined, runtime,
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

  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
    fillSignalAndDerivatives(k, segment, t, l, offset, workspace);
  }
  const ownProbability = output[askedK - 1];
  const draw = (runtime ? runtime.screen : SCREEN_DRAW)[0];
  let maximum = -Infinity;
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
    if (k !== askedK) maximum = Math.max(maximum, draw.beta * workspace.z[k]);
  }
  let denominator = 0;
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
    if (k === askedK) continue;
    const softmax = Math.exp(draw.beta * workspace.z[k] - maximum);
    workspace.wrong[k] = softmax;
    denominator += softmax;
  }
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
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
  for (let taskIndex = 0; taskIndex < IIIC_TASK_INDICES.length; taskIndex++) {
    const k = IIIC_TASK_INDICES[taskIndex];
    if (k === askedK) continue;
    const outcome = k - 1;
    const wrongGivenIncorrect = output[outcome] / (1 - ownProbability);
    workspace.biasJacobian[askedRow + outcome] =
      -ownBiasDerivative * wrongGivenIncorrect;
    workspace.skillJacobian[askedRow + outcome] =
      -ownSkillDerivative * wrongGivenIncorrect;
  }

  const directedScale = (1 - ownProbability) * (1 - draw.distractorLapse) * draw.beta;
  for (let parameterIndex = 0; parameterIndex < IIIC_TASK_INDICES.length;
    parameterIndex++) {
    const parameterK = IIIC_TASK_INDICES[parameterIndex];
    if (parameterK === askedK) continue;
    const parameterSoftmax = workspace.wrong[parameterK];
    const biasScale = directedScale * workspace.signalBiasDerivative[parameterK]
      * parameterSoftmax;
    const skillScale = directedScale * workspace.signalSkillDerivative[parameterK]
      * parameterSoftmax;
    const row = parameterK * outcomeCount;
    for (let outcomeIndex = 0; outcomeIndex < IIIC_TASK_INDICES.length;
      outcomeIndex++) {
      const outcomeK = IIIC_TASK_INDICES[outcomeIndex];
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
  runtime?: NWayResponseRuntime,
): { outcome: number; probability: number }[] {
  const output = new Float64Array(taskClass === "spike" ? 2 : 6);
  fillResponseProbabilities(
    taskClass, askedK, segment, t, l, particleIndex, K,
    output, makeResponseProbabilityWorkspace(K), false, undefined, runtime,
  );
  return Array.from(output, (probability, index) => ({
    outcome: taskClass === "spike" ? (index === 0 ? askedK : K) : index + 1,
    probability,
  }));
}

/** Cheap deterministic single-draw surrogate used only by Fisher screening. */
export function momentMatchedScreenDraw(
  draws: readonly ArtifactDraw[],
): ArtifactDraw {
  return {
    beta: draws.reduce((sum, draw) => sum + draw.weight * draw.beta, 0),
    distractorLapse: draws.reduce(
      (sum, draw) => sum + draw.weight * draw.distractorLapse, 0,
    ),
    weight: 1,
  };
}

const SCREEN_DRAW: readonly ArtifactDraw[] = Object.freeze([
  momentMatchedScreenDraw(NWAY_ARTIFACT.draws),
]);

/**
 * Build a validated response-model runtime for a profile-gated artifact.
 * Mirrors the isolated port's normalizedArtifactDraws fail-closed checks; the
 * prepared log terms use the exact frozen history-draw arithmetic.
 */
export function buildNWayResponseRuntime(
  aggregation: NWayResponseAggregation,
  artifactId: string,
  draws: readonly ArtifactDraw[],
): NWayResponseRuntime {
  if (draws.length === 0) throw new Error("response runtime requires at least one draw");
  let weightSum = 0;
  for (const draw of draws) {
    if (!Number.isFinite(draw.beta) || draw.beta <= 0) {
      throw new Error("artifact beta must be finite and positive");
    }
    if (!Number.isFinite(draw.distractorLapse)
        || draw.distractorLapse < 0 || draw.distractorLapse > 1) {
      throw new Error("artifact distractor lapse must be in [0, 1]");
    }
    if (!Number.isFinite(draw.weight) || draw.weight <= 0) {
      throw new Error("artifact draw weights must be finite and positive");
    }
    weightSum += draw.weight;
  }
  const normalized = draws.map((draw) => ({ ...draw, weight: draw.weight / weightSum }));
  return {
    aggregation,
    artifactId,
    atoms: Object.freeze(normalized.map(prepareResponseDraw)),
    screen: Object.freeze([momentMatchedScreenDraw(normalized)]),
  };
}

// Built lazily once each; both constants are frozen for the process lifetime.
let drawLatentRuntime: NWayResponseRuntime | null = null;
let qualifiedDrawLatentRuntime: NWayResponseRuntime | null = null;

/**
 * Resolve the profile-stamped response runtime for a session's inputs.
 * Mixture stamps (or absent stamps) resolve to undefined — the frozen
 * NWAY_ARTIFACT path, byte-identical to the shipped engine. A draw-latent
 * stamp resolves to the runtime of the artifact it names — the qualified
 * nesting34 table or the research atoms17 table — fail-closed against any
 * stamp/constant divergence: coordinator, selector workers, and branch
 * workers all derive the same runtime from the same server-authoritative
 * stamp, so no serialized state ever carries the table itself.
 */
export function nwayResponseRuntimeFor(
  inputs: ComputeEngineInputs,
): NWayResponseRuntime | undefined {
  if (nwayResponseAggregationOf(inputs) !== "draw_latent") return undefined;
  const stamp = inputs.nwayProfile;
  if (stamp?.responseArtifactId === NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.artifactId
      && stamp.responseArtifactSha256 === NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.sha256) {
    qualifiedDrawLatentRuntime ??= buildNWayResponseRuntime(
      "draw_latent",
      NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.artifactId,
      NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.draws,
    );
    return qualifiedDrawLatentRuntime;
  }
  if (stamp?.responseArtifactId !== NWAY_DRAW_LATENT_ARTIFACT.artifactId
      || stamp.responseArtifactSha256 !== NWAY_DRAW_LATENT_ARTIFACT.sha256) {
    throw new Error("draw-latent stamp does not name the registered artifact");
  }
  drawLatentRuntime ??= buildNWayResponseRuntime(
    "draw_latent",
    NWAY_DRAW_LATENT_ARTIFACT.artifactId,
    NWAY_DRAW_LATENT_ARTIFACT.draws,
  );
  return drawLatentRuntime;
}
