import { logSumExp } from "./math";
import { logProbability, responseProbabilities } from "./likelihood";
import type {
  ConditionalF1Artifact, EngineProfile, Observation, ProtocolSegment,
} from "./types";

/**
 * A research-only posterior/bootstrap draw for the conditional-F1 artifact.
 * The production-shaped scalar artifact remains unchanged.
 */
export interface ResearchArtifactDraw {
  beta: number;
  distractorLapse: number;
  weight?: number;
}

export interface ResearchArtifactEnsemble {
  baseArtifact: ConditionalF1Artifact;
  draws: readonly ResearchArtifactDraw[];
}

function normalizedDraws(
  ensemble: ResearchArtifactEnsemble,
): { artifact: ConditionalF1Artifact; weight: number }[] {
  if (ensemble.draws.length === 0) throw new Error("artifact ensemble must contain a draw");
  let weightSum = 0;
  for (const draw of ensemble.draws) {
    const weight = draw.weight ?? 1;
    if (!Number.isFinite(draw.beta) || draw.beta <= 0) {
      throw new Error("artifact ensemble beta must be finite and positive");
    }
    if (!Number.isFinite(draw.distractorLapse)
      || draw.distractorLapse < 0 || draw.distractorLapse > 1) {
      throw new Error("artifact ensemble distractor lapse must be in [0, 1]");
    }
    if (!Number.isFinite(weight) || weight <= 0) {
      throw new Error("artifact ensemble weights must be finite and positive");
    }
    weightSum += weight;
  }
  return ensemble.draws.map((draw) => ({
    artifact: {
      ...ensemble.baseArtifact,
      beta: draw.beta,
      distractorLapse: draw.distractorLapse,
    },
    weight: (draw.weight ?? 1) / weightSum,
  }));
}

export function responseProbabilitiesWithArtifactEnsemble(
  profile: EngineProfile,
  ensemble: ResearchArtifactEnsemble,
  askedK: number,
  segment: ProtocolSegment,
  t: Float64Array,
  l: Float64Array,
  particleIndex: number,
  taskCount: number,
): { outcome: number; probability: number }[] {
  const draws = normalizedDraws(ensemble);
  const result = responseProbabilities(
    profile, draws[0].artifact, askedK, segment, t, l, particleIndex, taskCount,
  ).map(({ outcome }) => ({ outcome, probability: 0 }));
  for (const draw of draws) {
    const probabilities = responseProbabilities(
      profile, draw.artifact, askedK, segment, t, l, particleIndex, taskCount,
    );
    for (let index = 0; index < result.length; index += 1) {
      if (result[index].outcome !== probabilities[index].outcome) {
        throw new Error("artifact ensemble outcome order mismatch");
      }
      result[index].probability += draw.weight * probabilities[index].probability;
    }
  }
  return result;
}

export function logProbabilityWithArtifactEnsemble(
  profile: EngineProfile,
  ensemble: ResearchArtifactEnsemble,
  observation: Observation,
  segment: ProtocolSegment,
  t: Float64Array,
  l: Float64Array,
  particleIndex: number,
  taskCount: number,
): number {
  const draws = normalizedDraws(ensemble);
  return logSumExp(draws.map((draw) => (
    Math.log(draw.weight) + logProbability(
      profile, draw.artifact, observation, segment,
      t, l, particleIndex, taskCount,
    )
  )));
}
