import { cholesky, covRows, symSqrtClipped, type Mat } from "../../cortex_web/apps/web/engine/linalg";
import { logPriorOne, samplePrior } from "../../cortex_web/apps/web/engine/prior";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { logProbability } from "./likelihood";
import type {
  ConditionalF1ResponseArtifact, EngineProfile, Observation, PriorPair,
  ProtocolParticleState, ProtocolSegment,
} from "./types";

export class PosteriorUpdateError extends Error {}

export function makeProtocolState(
  particleCount: number,
  taskCount: number,
  prior: PriorPair,
  rng: Rng,
): ProtocolParticleState {
  const t = new Float64Array(particleCount * taskCount);
  const l = new Float64Array(particleCount * taskCount);
  const w = new Float64Array(particleCount).fill(1 / particleCount);
  const logPrior = new Float64Array(particleCount);
  const logLik = new Float64Array(particleCount);
  samplePrior(particleCount, prior.tPieces, prior.lPieces, rng, t, l, logPrior);
  return {
    N: particleCount, K: taskCount, t, l, w, logPrior, logLik,
    history: [], prior,
  };
}

export function cloneProtocolState(state: ProtocolParticleState): ProtocolParticleState {
  return {
    N: state.N,
    K: state.K,
    t: state.t.slice(),
    l: state.l.slice(),
    w: state.w.slice(),
    logPrior: state.logPrior.slice(),
    logLik: state.logLik.slice(),
    history: state.history.map((observation) => ({ ...observation })),
    prior: state.prior,
    ...(state.lastRejuvenation
      ? { lastRejuvenation: { ...state.lastRejuvenation } }
      : {}),
  };
}

function segmentFor(
  segments: readonly ProtocolSegment[],
  observation: Observation,
): ProtocolSegment {
  const segment = segments[observation.segmentIndex];
  if (!segment || segment.segmentIndex !== observation.segmentIndex) {
    throw new Error(`segment index ${observation.segmentIndex} is unavailable`);
  }
  return segment;
}

function validatePreUpdate(state: ProtocolParticleState): void {
  let weightSum = 0;
  for (let n = 0; n < state.N; n += 1) {
    if (!Number.isFinite(state.w[n]) || state.w[n] < 0) {
      throw new PosteriorUpdateError("pre-update particle weights are invalid");
    }
    if (!Number.isFinite(state.logLik[n])) {
      throw new PosteriorUpdateError("pre-update log likelihood is invalid");
    }
    weightSum += state.w[n];
  }
  if (!(weightSum > 0) || !Number.isFinite(weightSum)) {
    throw new PosteriorUpdateError("pre-update particle weights have zero mass");
  }
}

/**
 * Transactional posterior update. Binary observations retain the production
 * multiply/normalize operation order. Categorical observations normalize in
 * log space so a sharply resolved distractor softmax cannot underflow the
 * complete cloud to zero mass.
 */
export function updateProtocol(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  segments: readonly ProtocolSegment[],
  observation: Observation,
): void {
  validatePreUpdate(state);
  const segment = segmentFor(segments, observation);
  const nextLogLik = new Float64Array(state.N);
  const likelihood = new Float64Array(state.N);
  let maximumLogWeight = -Infinity;
  for (let n = 0; n < state.N; n += 1) {
    const lp = logProbability(
      profile, artifact, observation, segment,
      state.t, state.l, n, state.K,
    );
    const cumulative = state.logLik[n] + lp;
    if (!Number.isFinite(lp) || !Number.isFinite(cumulative)) {
      throw new PosteriorUpdateError("posterior update produced non-finite likelihood");
    }
    likelihood[n] = lp;
    nextLogLik[n] = cumulative;
    if (state.w[n] > 0) maximumLogWeight = Math.max(maximumLogWeight, Math.log(state.w[n]) + lp);
  }

  const nextW = new Float64Array(state.N);
  let sum = 0;
  if (observation.kind === "binary") {
    for (let n = 0; n < state.N; n += 1) {
      const value = state.w[n] * Math.exp(likelihood[n]);
      if (!Number.isFinite(value)) throw new PosteriorUpdateError("binary update overflowed");
      nextW[n] = value;
      sum += value;
    }
  } else {
    if (!Number.isFinite(maximumLogWeight)) {
      throw new PosteriorUpdateError("categorical update has zero mass");
    }
    for (let n = 0; n < state.N; n += 1) {
      const value = state.w[n] === 0
        ? 0
        : Math.exp(Math.log(state.w[n]) + likelihood[n] - maximumLogWeight);
      nextW[n] = value;
      sum += value;
    }
  }
  if (!(sum > 0) || !Number.isFinite(sum)) {
    throw new PosteriorUpdateError("posterior update has zero or non-finite mass");
  }
  for (let n = 0; n < state.N; n += 1) nextW[n] /= sum;

  state.w = nextW;
  state.logLik = nextLogLik;
  state.history.push({ ...observation });
}

export function ess(weights: Float64Array): number {
  let squared = 0;
  for (const weight of weights) squared += weight * weight;
  return 1 / squared;
}

export function logLikelihoodHistory(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  segments: readonly ProtocolSegment[],
  proposedT: Float64Array,
  proposedL: Float64Array,
  output = new Float64Array(state.N),
): Float64Array {
  output.fill(0);
  for (const observation of state.history) {
    const segment = segmentFor(segments, observation);
    for (let n = 0; n < state.N; n += 1) {
      output[n] += logProbability(
        profile, artifact, observation, segment,
        proposedT, proposedL, n, state.K,
      );
    }
  }
  return output;
}

export function resampleAndRejuvenateProtocol(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
  segments: readonly ProtocolSegment[],
  rng: Rng,
  nMhSteps: number,
  proposalScale: number,
  qIndex = -1,
): number {
  const { N, K } = state;
  const ancestors = rng.resampleIndices(state.w, N);
  const distinctAncestors = new Set(ancestors).size;
  const t = new Float64Array(N * K);
  const l = new Float64Array(N * K);
  const logPrior = new Float64Array(N);
  const logLik = new Float64Array(N);
  for (let n = 0; n < N; n += 1) {
    const source = ancestors[n];
    t.set(state.t.subarray(source * K, source * K + K), n * K);
    l.set(state.l.subarray(source * K, source * K + K), n * K);
    logPrior[n] = state.logPrior[source];
    logLik[n] = state.logLik[source];
  }
  state.t = t;
  state.l = l;
  state.logPrior = logPrior;
  state.logLik = logLik;
  state.w.fill(1 / N);

  const dimensions = 2 * K;
  const theta = new Float64Array(N * dimensions);
  const proposedT = new Float64Array(N * K);
  const proposedL = new Float64Array(N * K);
  const proposedPrior = new Float64Array(N);
  const proposedLik = new Float64Array(N);
  const gaussian = new Float64Array(dimensions);
  let acceptanceTotal = 0;

  for (let step = 0; step < nMhSteps; step += 1) {
    for (let n = 0; n < N; n += 1) {
      for (let k = 0; k < K; k += 1) {
        theta[n * dimensions + k] = state.t[n * K + k];
        theta[n * dimensions + K + k] = state.l[n * K + k];
      }
    }
    const covariance: Mat = covRows(theta, N, dimensions);
    for (let d = 0; d < dimensions; d += 1) covariance[d][d] += 1e-6;
    let factor: Mat;
    try {
      factor = cholesky(covariance);
    } catch {
      factor = symSqrtClipped(covariance, 1e-6);
    }
    for (let n = 0; n < N; n += 1) {
      rng.fillGaussian(gaussian);
      for (let d = 0; d < dimensions; d += 1) {
        let delta = 0;
        for (let j = 0; j < dimensions; j += 1) delta += gaussian[j] * factor[d][j];
        const value = theta[n * dimensions + d] + proposalScale * delta;
        if (d < K) proposedT[n * K + d] = value;
        else proposedL[n * K + d - K] = value;
      }
    }
    for (let n = 0; n < N; n += 1) {
      proposedPrior[n] = logPriorOne(
        proposedT, proposedL, n, state.prior.tPieces, state.prior.lPieces,
      );
    }
    logLikelihoodHistory(
      state, profile, artifact, segments, proposedT, proposedL, proposedLik,
    );
    let accepted = 0;
    for (let n = 0; n < N; n += 1) {
      const logAlpha = proposedPrior[n] + proposedLik[n]
        - state.logPrior[n] - state.logLik[n];
      if (Math.log(rng.random()) < logAlpha) {
        state.t.set(proposedT.subarray(n * K, n * K + K), n * K);
        state.l.set(proposedL.subarray(n * K, n * K + K), n * K);
        state.logPrior[n] = proposedPrior[n];
        state.logLik[n] = proposedLik[n];
        accepted += 1;
      }
    }
    acceptanceTotal += accepted / N;
  }
  const acceptanceRate = acceptanceTotal / Math.max(nMhSteps, 1);
  state.lastRejuvenation = {
    qIndex,
    acceptanceRate,
    distinctAncestors,
    distinctAncestorFraction: distinctAncestors / N,
  };
  return acceptanceRate;
}

export function posteriorMoments(state: ProtocolParticleState): {
  tMean: number[];
  lMean: number[];
  tSd: number[];
  lSd: number[];
} {
  const tMean = new Array<number>(state.K).fill(0);
  const lMean = new Array<number>(state.K).fill(0);
  const tSecond = new Array<number>(state.K).fill(0);
  const lSecond = new Array<number>(state.K).fill(0);
  for (let n = 0; n < state.N; n += 1) {
    for (let k = 0; k < state.K; k += 1) {
      const offset = n * state.K + k;
      tMean[k] += state.w[n] * state.t[offset];
      lMean[k] += state.w[n] * state.l[offset];
      tSecond[k] += state.w[n] * state.t[offset] ** 2;
      lSecond[k] += state.w[n] * state.l[offset] ** 2;
    }
  }
  return {
    tMean,
    lMean,
    tSd: tMean.map((mean, k) => Math.sqrt(Math.max(0, tSecond[k] - mean ** 2))),
    lSd: lMean.map((mean, k) => Math.sqrt(Math.max(0, lSecond[k] - mean ** 2))),
  };
}
