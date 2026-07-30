/**
 * Selector challengers. The immutable integrated-profile dispatcher opts into
 * the proven Fisher/total-variance path; frozen categorical_totalvar_v1
 * sessions remain on the original selector.
 */
import { momentMatchedArtifact } from "./artifact";
import { responseProbabilities } from "./likelihood";
import { posteriorMoments } from "./particles";
import {
  chooseCandidate, expectedPosteriorLoss, shortlistCandidates,
  type ShortlistOptions,
} from "./selector";
import type {
  Candidate, ChosenCandidate, ConditionalF1ResponseArtifact, EngineProfile,
  ProtocolParticleState, ProtocolSegment,
} from "./types";

export type ResearchObjective = "total_variance" | "bias_weighted" | "mutual_information";

export interface ResearchSelectorOptions extends ShortlistOptions {
  fisherPerTask?: number;
  finiteDifference?: number;
  objective?: ResearchObjective;
  skillWeight?: number;
  biasWeight?: number;
}

export interface ResearchSelectorAudit {
  candidateCount: number;
  baselineShortlistSize: number;
  challengerShortlistSize: number;
  baseline: ChosenCandidate;
  challenger: ChosenCandidate;
  exact: ChosenCandidate;
  baselineRegret: number;
  challengerRegret: number;
}

type PosteriorMoments = ReturnType<typeof posteriorMoments>;

function probabilityVector(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  askedK: number,
  segment: ProtocolSegment,
  t: Float64Array,
  l: Float64Array,
): number[] {
  return responseProbabilities(
    profile, artifact, askedK, segment, t, l, 0, state.K,
  ).map((entry) => entry.probability);
}

/**
 * Diagonal Laplace/Fisher approximation used only to screen candidates.  The
 * exact six-outcome objective still makes the final choice.
 */
function fisherUtilityWithMoments(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidate: Candidate,
  segment: ProtocolSegment,
  options: ResearchSelectorOptions,
  moments: PosteriorMoments,
): number {
  const meanT = Float64Array.from(moments.tMean);
  const meanL = Float64Array.from(moments.lMean);
  const screenArtifact = momentMatchedArtifact(artifact);
  const step = options.finiteDifference ?? 1e-3;
  const base = probabilityVector(
    state, profile, screenArtifact, candidate.askedK, segment, meanT, meanL,
  );
  let utility = 0;
  for (const parameter of ["bias", "skill"] as const) {
    const values = parameter === "bias" ? meanT : meanL;
    const variances = parameter === "bias" ? moments.tSd : moments.lSd;
    const weight = parameter === "bias"
      ? (options.biasWeight ?? 1)
      : (options.skillWeight ?? 1);
    for (let k = 0; k < state.K; k += 1) {
      const original = values[k];
      values[k] = original + step;
      const plus = probabilityVector(
        state, profile, screenArtifact, candidate.askedK, segment, meanT, meanL,
      );
      values[k] = original - step;
      const minus = probabilityVector(
        state, profile, screenArtifact, candidate.askedK, segment, meanT, meanL,
      );
      values[k] = original;
      let information = 0;
      for (let r = 0; r < base.length; r += 1) {
        const derivative = (plus[r] - minus[r]) / (2 * step);
        information += derivative * derivative / Math.max(base[r], 1e-12);
      }
      const variance = variances[k] ** 2;
      utility += weight * variance * variance * information / (1 + variance * information);
    }
  }
  return utility;
}

export function approximateFisherUtility(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidate: Candidate,
  segment: ProtocolSegment,
  options: ResearchSelectorOptions = {},
): number {
  return fisherUtilityWithMoments(
    state, profile, artifact, candidate, segment, options,
    posteriorMoments(state),
  );
}

export function fisherAugmentedShortlist(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidates: readonly Candidate[],
  segments: readonly ProtocolSegment[],
  options: ResearchSelectorOptions = {},
): Candidate[] {
  const baseline = shortlistCandidates(
    state, profile, artifact, candidates, segments, options,
  );
  if (candidates.length <= (options.fullScanLimit ?? 512)) return baseline;
  const selected = new Set(baseline);
  const fisherPerTask = options.fisherPerTask ?? 8;
  const moments = posteriorMoments(state);
  for (let askedK = 0; askedK < state.K; askedK += 1) {
    const scored = candidates
      .filter((candidate) => candidate.askedK === askedK)
      .map((candidate) => ({
        candidate,
        utility: fisherUtilityWithMoments(
          state, profile, artifact, candidate,
          segments[candidate.segmentIndex], options, moments,
        ),
      }))
      .sort((a, b) => (b.utility - a.utility) || (a.candidate.segId - b.candidate.segId));
    for (const entry of scored.slice(0, fisherPerTask)) selected.add(entry.candidate);
  }
  return candidates.filter((candidate) => selected.has(candidate));
}

function weightedExpectedLoss(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidate: Candidate,
  segment: ProtocolSegment,
  skillWeight: number,
  biasWeight: number,
): number {
  const moments = posteriorMoments(state);
  let baseline = 0;
  for (let k = 0; k < state.K; k += 1) {
    baseline += biasWeight * moments.tSd[k] ** 2 + skillWeight * moments.lSd[k] ** 2;
  }
  const template = responseProbabilities(
    profile, artifact, candidate.askedK, segment, state.t, state.l, 0, state.K,
  );
  const outcomeIndex = new Map(template.map((entry, index) => [entry.outcome, index]));
  const mass = new Float64Array(template.length);
  const conditionalT = Array.from(
    { length: template.length }, () => new Float64Array(state.K),
  );
  const conditionalL = Array.from(
    { length: template.length }, () => new Float64Array(state.K),
  );
  let weightSum = 0;
  for (const weight of state.w) weightSum += weight;
  for (let n = 0; n < state.N; n += 1) {
    const normalizedWeight = state.w[n] / weightSum;
    for (const response of responseProbabilities(
      profile, artifact, candidate.askedK, segment,
      state.t, state.l, n, state.K,
    )) {
      const r = outcomeIndex.get(response.outcome)!;
      const joint = normalizedWeight * response.probability;
      mass[r] += joint;
      for (let k = 0; k < state.K; k += 1) {
        const offset = n * state.K + k;
        conditionalT[r][k] += joint * state.t[offset];
        conditionalL[r][k] += joint * state.l[offset];
      }
    }
  }
  let between = 0;
  for (let r = 0; r < mass.length; r += 1) {
    if (mass[r] <= 1e-15) continue;
    for (let k = 0; k < state.K; k += 1) {
      const tDelta = conditionalT[r][k] / mass[r] - moments.tMean[k];
      const lDelta = conditionalL[r][k] / mass[r] - moments.lMean[k];
      between += mass[r] * (
        biasWeight * tDelta * tDelta + skillWeight * lDelta * lDelta
      );
    }
  }
  return baseline - between;
}

/** Mutual information between the observed response and particle identity. */
export function responseMutualInformation(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidate: Candidate,
  segment: ProtocolSegment,
): number {
  const template = responseProbabilities(
    profile, artifact, candidate.askedK, segment, state.t, state.l, 0, state.K,
  );
  const outcomeIndex = new Map(template.map((entry, index) => [entry.outcome, index]));
  const marginal = new Float64Array(template.length);
  const rows: number[][] = [];
  let weightSum = 0;
  for (const weight of state.w) weightSum += weight;
  for (let n = 0; n < state.N; n += 1) {
    const probabilities = responseProbabilities(
      profile, artifact, candidate.askedK, segment,
      state.t, state.l, n, state.K,
    );
    const row = new Array<number>(template.length).fill(0);
    for (const response of probabilities) {
      const r = outcomeIndex.get(response.outcome)!;
      row[r] = response.probability;
      marginal[r] += state.w[n] / weightSum * response.probability;
    }
    rows.push(row);
  }
  let information = 0;
  for (let n = 0; n < state.N; n += 1) {
    const weight = state.w[n] / weightSum;
    for (let r = 0; r < marginal.length; r += 1) {
      const probability = rows[n][r];
      if (probability > 0 && marginal[r] > 0) {
        information += weight * probability * Math.log(probability / marginal[r]);
      }
    }
  }
  return information;
}

function researchLoss(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidate: Candidate,
  segment: ProtocolSegment,
  options: ResearchSelectorOptions,
): number {
  switch (options.objective ?? "total_variance") {
    case "total_variance":
      return expectedPosteriorLoss(state, profile, artifact, candidate, segment);
    case "bias_weighted":
      return weightedExpectedLoss(
        state, profile, artifact, candidate, segment,
        options.skillWeight ?? 1, options.biasWeight ?? 1.5,
      );
    case "mutual_information":
      return -responseMutualInformation(state, profile, artifact, candidate, segment);
  }
}

export function chooseResearchCandidate(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidates: readonly Candidate[],
  segments: readonly ProtocolSegment[],
  options: ResearchSelectorOptions = {},
): ChosenCandidate {
  const shortlist = fisherAugmentedShortlist(
    state, profile, artifact, candidates, segments, options,
  );
  let best: ChosenCandidate | undefined;
  for (const candidate of shortlist) {
    const loss = researchLoss(
      state, profile, artifact, candidate,
      segments[candidate.segmentIndex], options,
    );
    if (!best || loss < best.loss) best = { ...candidate, loss };
  }
  if (!best) throw new Error("candidate bank is empty");
  return best;
}

export function auditResearchShortlist(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidates: readonly Candidate[],
  segments: readonly ProtocolSegment[],
  options: ResearchSelectorOptions = {},
): ResearchSelectorAudit {
  const baselineShortlist = shortlistCandidates(
    state, profile, artifact, candidates, segments, options,
  );
  const challengerShortlist = fisherAugmentedShortlist(
    state, profile, artifact, candidates, segments, options,
  );
  const forceExact = { ...options, fullScanLimit: Number.POSITIVE_INFINITY };
  const baseline = chooseCandidate(
    state, profile, artifact, baselineShortlist, segments, forceExact,
  );
  const challenger = chooseCandidate(
    state, profile, artifact, challengerShortlist, segments, forceExact,
  );
  const exact = chooseCandidate(state, profile, artifact, candidates, segments, forceExact);
  return {
    candidateCount: candidates.length,
    baselineShortlistSize: baselineShortlist.length,
    challengerShortlistSize: challengerShortlist.length,
    baseline,
    challenger,
    exact,
    baselineRegret: Math.max(0, baseline.loss - exact.loss),
    challengerRegret: Math.max(0, challenger.loss - exact.loss),
  };
}
