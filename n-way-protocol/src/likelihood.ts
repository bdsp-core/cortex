import { logNdtr, logSumExp, logSumExp2, normCdf } from "./math";
import { normalizedArtifactDraws } from "./artifact";
import { groupForTask } from "./profile";
import type {
  ConditionalF1ArtifactDraw, ConditionalF1ResponseArtifact, EngineProfile,
  Observation, ProtocolSegment, ResponseGroup,
} from "./types";

export const BINARY_LAPSE = 0.025;
const LOG_LAPSE = Math.log(BINARY_LAPSE);
const LOG_ONE_MINUS_TWO_LAPSE = Math.log1p(-2 * BINARY_LAPSE);

export function signalZ(l: number, t: number, s: number, sSd = 0): number {
  const sensitivity = Math.exp(l);
  let z = sensitivity * (s + t);
  if (sSd !== 0) z /= Math.sqrt(1 + (sensitivity * sSd) ** 2);
  return z;
}

export function binaryPYes(z: number): number {
  return BINARY_LAPSE + (1 - 2 * BINARY_LAPSE) * normCdf(z);
}

export function logPBinary(z: number, y: 0 | 1): number {
  const cdf = LOG_ONE_MINUS_TWO_LAPSE + logNdtr(y === 1 ? z : -z);
  return logSumExp2(cdf, LOG_LAPSE);
}

export function makeObservation(
  profile: EngineProfile,
  askedK: number,
  segmentIndex: number,
  rawPick: number,
): Observation {
  const group = groupForTask(profile, askedK);
  if (group.link === "binary") {
    return {
      kind: "binary", askedK, segmentIndex, rawPick,
      y: rawPick === askedK ? 1 : 0,
    };
  }
  if (!group.taskIndices.includes(rawPick)) {
    throw new Error(`categorical pick ${rawPick} is outside response group ${group.id}`);
  }
  return {
    kind: "categorical_f1", groupId: group.id, askedK, segmentIndex,
    rawPick, pickK: rawPick,
  };
}

function zFor(
  taskK: number,
  segment: ProtocolSegment,
  t: Float64Array,
  l: Float64Array,
  particleOffset: number,
): number {
  return signalZ(
    l[particleOffset + taskK], t[particleOffset + taskK],
    segment.sMean[taskK], segment.sSd[taskK],
  );
}

function logDistractorProbability(
  group: ResponseGroup,
  askedK: number,
  pickK: number,
  segment: ProtocolSegment,
  t: Float64Array,
  l: Float64Array,
  particleOffset: number,
  draw: ConditionalF1ArtifactDraw,
): number {
  const distractors = group.taskIndices.filter((taskK) => taskK !== askedK);
  const logits = distractors.map(
    (taskK) => draw.beta * zFor(taskK, segment, t, l, particleOffset),
  );
  const pickIndex = distractors.indexOf(pickK);
  if (pickIndex < 0) throw new Error("pick is not a valid distractor");
  const logSoftmax = logits[pickIndex] - logSumExp(logits);
  const uniform = draw.distractorLapse === 0
    ? -Infinity
    : Math.log(draw.distractorLapse) - Math.log(distractors.length);
  const directed = draw.distractorLapse === 1
    ? -Infinity
    : Math.log1p(-draw.distractorLapse) + logSoftmax;
  return logSumExp2(uniform, directed);
}

export function logProbability(
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  observation: Observation,
  segment: ProtocolSegment,
  t: Float64Array,
  l: Float64Array,
  particleIndex: number,
  taskCount: number,
  atomDraw?: ConditionalF1ArtifactDraw,
): number {
  if (segment.segmentIndex !== observation.segmentIndex) {
    throw new Error("observation/segment index mismatch");
  }
  const offset = particleIndex * taskCount;
  const ownZ = zFor(observation.askedK, segment, t, l, offset);
  if (observation.kind === "binary") return logPBinary(ownZ, observation.y);
  if (!artifact) throw new Error("categorical observation requires an F1 artifact");
  const group = groupForTask(profile, observation.askedK);
  if (group.id !== observation.groupId || group.link !== "categorical_f1") {
    throw new Error("observation response group mismatch");
  }
  if (observation.pickK === observation.askedK) return logPBinary(ownZ, 1);
  if (atomDraw) {
    // Draw-latent aggregation: this particle scores the pick under its own
    // atom's (beta, distractorLapse), not the fixed-weight mixture average.
    return logPBinary(ownZ, 0) + logDistractorProbability(
      group, observation.askedK, observation.pickK,
      segment, t, l, offset, atomDraw,
    );
  }
  if (artifact.model === "iiic_conditional_f1_v1") {
    return logPBinary(ownZ, 0) + logDistractorProbability(
      group, observation.askedK, observation.pickK,
      segment, t, l, offset,
      { beta: artifact.beta, distractorLapse: artifact.distractorLapse, weight: 1 },
    );
  }
  const drawLogProbabilities = normalizedArtifactDraws(artifact).map((draw) => (
    Math.log(draw.weight) + logDistractorProbability(
      group, observation.askedK, observation.pickK,
      segment, t, l, offset, draw,
    )
  ));
  return logPBinary(ownZ, 0) + logSumExp(drawLogProbabilities);
}

export function responseProbabilities(
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  askedK: number,
  segment: ProtocolSegment,
  t: Float64Array,
  l: Float64Array,
  particleIndex: number,
  taskCount: number,
): { outcome: number; probability: number }[] {
  const group = groupForTask(profile, askedK);
  if (group.link === "binary") {
    const offset = particleIndex * taskCount;
    const yes = binaryPYes(zFor(askedK, segment, t, l, offset));
    return [{ outcome: askedK, probability: yes }, { outcome: taskCount, probability: 1 - yes }];
  }
  if (!artifact) throw new Error("categorical response group requires an F1 artifact");
  // The selector calls this N × candidates times. Compute the distractor
  // softmax once per particle/candidate rather than rebuilding it separately
  // for each of the six possible outcomes.
  const offset = particleIndex * taskCount;
  const own = binaryPYes(zFor(askedK, segment, t, l, offset));
  const distractors = group.taskIndices.filter((taskK) => taskK !== askedK);
  const distractorZ = distractors.map(
    (taskK) => zFor(taskK, segment, t, l, offset),
  );
  const wrongProbabilities = new Float64Array(distractors.length);
  for (const draw of normalizedArtifactDraws(artifact)) {
    const logits = distractorZ.map((z) => draw.beta * z);
    const maximum = Math.max(...logits);
    const exponentials = logits.map((logit) => Math.exp(logit - maximum));
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);
    for (let index = 0; index < distractors.length; index += 1) {
      const directed = exponentials[index] / denominator;
      const q = draw.distractorLapse / distractors.length
        + (1 - draw.distractorLapse) * directed;
      wrongProbabilities[index] += draw.weight * q;
    }
  }
  return group.taskIndices.map((outcome) => {
    if (outcome === askedK) return { outcome, probability: own };
    const distractorIndex = distractors.indexOf(outcome);
    return { outcome, probability: (1 - own) * wrongProbabilities[distractorIndex] };
  });
}
