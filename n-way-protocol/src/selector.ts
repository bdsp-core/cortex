import { responseProbabilities } from "./likelihood";
import { posteriorMoments } from "./particles";
import type {
  Candidate, ChosenCandidate, ConditionalF1ResponseArtifact, EngineProfile,
  ProtocolParticleState, ProtocolSegment,
} from "./types";

interface LossWorkspace {
  baselineVariance: number;
  meanT: Float64Array;
  meanL: Float64Array;
}

export interface ShortlistOptions {
  fullScanLimit?: number;
  coarsePerTask?: number;
  entropyPerTask?: number;
  forcedSegmentIds?: ReadonlySet<number>;
}

export interface SelectorRegretAudit {
  candidateCount: number;
  shortlistSize: number;
  sameChoice: boolean;
  approximate: ChosenCandidate;
  exact: ChosenCandidate;
  absoluteRegret: number;
  relativeRegret: number;
}

function workspace(state: ProtocolParticleState): LossWorkspace {
  const moments = posteriorMoments(state);
  let baselineVariance = 0;
  for (let k = 0; k < state.K; k += 1) {
    baselineVariance += moments.tSd[k] ** 2 + moments.lSd[k] ** 2;
  }
  return {
    baselineVariance,
    meanT: Float64Array.from(moments.tMean),
    meanL: Float64Array.from(moments.lMean),
  };
}

export function predictedOutcomeDistribution(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  askedK: number,
  segment: ProtocolSegment,
): { outcome: number; probability: number }[] {
  const outcomes = responseProbabilities(
    profile, artifact, askedK, segment,
    state.t, state.l, 0, state.K,
  ).map(({ outcome }) => ({ outcome, probability: 0 }));
  const index = new Map(outcomes.map((entry, i) => [entry.outcome, i]));
  let weightSum = 0;
  for (let n = 0; n < state.N; n += 1) {
    weightSum += state.w[n];
    for (const entry of responseProbabilities(
      profile, artifact, askedK, segment,
      state.t, state.l, n, state.K,
    )) {
      outcomes[index.get(entry.outcome)!].probability += state.w[n] * entry.probability;
    }
  }
  for (const entry of outcomes) entry.probability /= weightSum;
  return outcomes;
}

export function expectedPosteriorLoss(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidate: Candidate,
  segment: ProtocolSegment,
  cachedWorkspace = workspace(state),
): number {
  if (candidate.segmentIndex !== segment.segmentIndex) {
    throw new Error("candidate/segment index mismatch");
  }
  const outcomeTemplate = responseProbabilities(
    profile, artifact, candidate.askedK, segment,
    state.t, state.l, 0, state.K,
  );
  const outcomeIndex = new Map(outcomeTemplate.map((entry, i) => [entry.outcome, i]));
  const responseMass = new Float64Array(outcomeTemplate.length);
  const conditionalT = Array.from(
    { length: outcomeTemplate.length }, () => new Float64Array(state.K),
  );
  const conditionalL = Array.from(
    { length: outcomeTemplate.length }, () => new Float64Array(state.K),
  );
  let weightSum = 0;
  for (let n = 0; n < state.N; n += 1) weightSum += state.w[n];
  for (let n = 0; n < state.N; n += 1) {
    const normalizedWeight = state.w[n] / weightSum;
    for (const response of responseProbabilities(
      profile, artifact, candidate.askedK, segment,
      state.t, state.l, n, state.K,
    )) {
      const r = outcomeIndex.get(response.outcome)!;
      const joint = normalizedWeight * response.probability;
      responseMass[r] += joint;
      for (let k = 0; k < state.K; k += 1) {
        const offset = n * state.K + k;
        conditionalT[r][k] += joint * state.t[offset];
        conditionalL[r][k] += joint * state.l[offset];
      }
    }
  }
  let betweenResponseVariance = 0;
  for (let r = 0; r < responseMass.length; r += 1) {
    if (responseMass[r] <= 1e-15) continue;
    for (let k = 0; k < state.K; k += 1) {
      const responseT = conditionalT[r][k] / responseMass[r];
      const responseL = conditionalL[r][k] / responseMass[r];
      betweenResponseVariance += responseMass[r] * (
        (responseT - cachedWorkspace.meanT[k]) ** 2
        + (responseL - cachedWorkspace.meanL[k]) ** 2
      );
    }
  }
  return cachedWorkspace.baselineVariance - betweenResponseVariance;
}

function entropy(probabilities: readonly number[]): number {
  let value = 0;
  for (const probability of probabilities) {
    if (probability > 0) value -= probability * Math.log(probability);
  }
  return value;
}

function evenlySpaced(length: number, count: number): number[] {
  if (length <= count) return Array.from({ length }, (_, index) => index);
  const result: number[] = [];
  for (let i = 0; i < count; i += 1) {
    result.push(Math.round(i * (length - 1) / (count - 1)));
  }
  return result;
}

export function shortlistCandidates(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidates: readonly Candidate[],
  segments: readonly ProtocolSegment[],
  options: ShortlistOptions = {},
): Candidate[] {
  const fullScanLimit = options.fullScanLimit ?? 512;
  if (candidates.length <= fullScanLimit) return candidates.slice();
  const coarsePerTask = options.coarsePerTask ?? 32;
  const entropyPerTask = options.entropyPerTask ?? 12;
  const selected = new Set<Candidate>();
  const moments = posteriorMoments(state);
  const meanT = Float64Array.from(moments.tMean);
  const meanL = Float64Array.from(moments.lMean);

  for (let askedK = 0; askedK < state.K; askedK += 1) {
    const domain = candidates
      .filter((candidate) => candidate.askedK === askedK)
      .sort((a, b) => (a.focalSignal - b.focalSignal) || (a.segId - b.segId));
    for (const index of evenlySpaced(domain.length, coarsePerTask)) selected.add(domain[index]);

    const binWidth = Math.max(1, Math.ceil(domain.length / coarsePerTask));
    for (let start = 0; start < domain.length; start += binWidth) {
      const bin = domain.slice(start, start + binWidth);
      if (bin.length) {
        selected.add(bin.reduce((best, value) => (
          value.focalSignalSd < best.focalSignalSd ? value : best
        )));
      }
    }

    const byEntropy = domain.map((candidate) => {
      const segment = segments[candidate.segmentIndex];
      const probabilities = responseProbabilities(
        profile, artifact, askedK, segment, meanT, meanL, 0, state.K,
      ).map((entry) => entry.probability);
      return { candidate, entropy: entropy(probabilities) };
    }).sort((a, b) => (b.entropy - a.entropy) || (a.candidate.segId - b.candidate.segId));
    for (const entry of byEntropy.slice(0, entropyPerTask)) selected.add(entry.candidate);
  }
  for (const candidate of candidates) {
    if (options.forcedSegmentIds?.has(candidate.segId)) selected.add(candidate);
  }
  return candidates.filter((candidate) => selected.has(candidate));
}

export function chooseCandidate(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidates: readonly Candidate[],
  segments: readonly ProtocolSegment[],
  options: ShortlistOptions = {},
): ChosenCandidate {
  const shortlist = shortlistCandidates(state, profile, artifact, candidates, segments, options);
  const cachedWorkspace = workspace(state);
  let best: ChosenCandidate | undefined;
  for (const candidate of shortlist) {
    const loss = expectedPosteriorLoss(
      state, profile, artifact, candidate,
      segments[candidate.segmentIndex], cachedWorkspace,
    );
    if (!best || loss < best.loss) best = { ...candidate, loss };
  }
  if (!best) throw new Error("candidate bank is empty");
  return best;
}

export function auditSelectorRegret(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  candidates: readonly Candidate[],
  segments: readonly ProtocolSegment[],
  options: ShortlistOptions = {},
): SelectorRegretAudit {
  const shortlist = shortlistCandidates(state, profile, artifact, candidates, segments, options);
  const approximate = chooseCandidate(state, profile, artifact, shortlist, segments, {
    ...options, fullScanLimit: Number.POSITIVE_INFINITY,
  });
  const exact = chooseCandidate(state, profile, artifact, candidates, segments, {
    ...options, fullScanLimit: Number.POSITIVE_INFINITY,
  });
  const absoluteRegret = Math.max(0, approximate.loss - exact.loss);
  return {
    candidateCount: candidates.length,
    shortlistSize: shortlist.length,
    sameChoice: approximate.segId === exact.segId && approximate.askedK === exact.askedK,
    approximate,
    exact,
    absoluteRegret,
    relativeRegret: absoluteRegret / Math.max(Math.abs(exact.loss), Number.EPSILON),
  };
}
