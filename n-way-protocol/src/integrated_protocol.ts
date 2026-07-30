import { chooseResearchCandidate, type ResearchSelectorOptions } from "./research_selector";
import { chooseCandidate, type ShortlistOptions } from "./selector";
import type {
  Candidate, ChosenCandidate, ConditionalF1ResponseArtifact, EngineProfile,
  ProtocolParticleState, ProtocolSegment,
} from "./types";

/**
 * Immutable-profile selector dispatch. The integrated selector only augments
 * the shortlist; exact total-posterior-variance loss still makes the choice.
 */
export function chooseProtocolCandidate(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidates: readonly Candidate[],
  segments: readonly ProtocolSegment[],
  options: ShortlistOptions & Pick<ResearchSelectorOptions, "fisherPerTask" | "finiteDifference"> = {},
): ChosenCandidate {
  if (profile.selectorVersion === "categorical_fisher_totalvar_v1") {
    return chooseResearchCandidate(
      state, profile, artifact, candidates, segments,
      { ...options, objective: "total_variance" },
    );
  }
  return chooseCandidate(state, profile, artifact, candidates, segments, options);
}
