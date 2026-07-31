import { cholesky, covRows, symSqrtClipped, type Mat } from "../../cortex_web/apps/web/engine/linalg";
import { logPriorOne, samplePrior } from "../../cortex_web/apps/web/engine/prior";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { normalizedArtifactDraws } from "./artifact";
import { logProbability } from "./likelihood";
import type {
  ConditionalF1ArtifactDraw, ConditionalF1ResponseArtifact, EngineProfile,
  Observation, PriorPair, ProtocolParticleState, ProtocolSegment,
  ResponseAggregation,
} from "./types";

export class PosteriorUpdateError extends Error {}

export function makeProtocolState(
  particleCount: number,
  taskCount: number,
  prior: PriorPair,
  rng: Rng,
  options?: {
    responseAggregation?: ResponseAggregation;
    artifact?: ConditionalF1ResponseArtifact;
  },
): ProtocolParticleState {
  const t = new Float64Array(particleCount * taskCount);
  const l = new Float64Array(particleCount * taskCount);
  const w = new Float64Array(particleCount).fill(1 / particleCount);
  const logPrior = new Float64Array(particleCount);
  const logLik = new Float64Array(particleCount);
  samplePrior(particleCount, prior.tPieces, prior.lPieces, rng, t, l, logPrior);
  const state: ProtocolParticleState = {
    N: particleCount, K: taskCount, t, l, w, logPrior, logLik,
    history: [], prior,
  };
  if ((options?.responseAggregation ?? "mixture") === "draw_latent") {
    if (!options?.artifact) {
      throw new Error("draw-latent cloud creation requires a response artifact");
    }
    state.atomIndex = sampleAtomLineage(particleCount, options.artifact, rng);
  }
  return state;
}

/**
 * Atom init for draw-latent aggregation: one categorical draw per particle
 * from the normalized artifact weights, taken at cloud creation (after the
 * prior sample, mirroring make_draw_cloud in draw_latent_rd/engine.py).
 */
export function sampleAtomLineage(
  particleCount: number,
  artifact: ConditionalF1ResponseArtifact,
  rng: Rng,
): Int32Array {
  const draws = normalizedArtifactDraws(artifact);
  const atomIndex = new Int32Array(particleCount);
  for (let n = 0; n < particleCount; n += 1) {
    const u = rng.random();
    let cumulative = 0;
    let chosen = draws.length - 1;
    for (let d = 0; d < draws.length; d += 1) {
      cumulative += draws[d].weight;
      if (u < cumulative) {
        chosen = d;
        break;
      }
    }
    atomIndex[n] = chosen;
  }
  return atomIndex;
}

/**
 * Resolve the per-particle atom table for draw-latent aggregation, or
 * undefined for the shipping mixture path. Validates that the state carries
 * a complete, in-range atom lineage before any likelihood is evaluated.
 */
function drawLatentAtoms(
  state: ProtocolParticleState,
  profile: EngineProfile,
  artifact: ConditionalF1ResponseArtifact | undefined,
): ConditionalF1ArtifactDraw[] | undefined {
  if ((profile.responseAggregation ?? "mixture") !== "draw_latent") return undefined;
  if (!artifact) throw new Error("draw-latent aggregation requires a response artifact");
  if (!state.atomIndex || state.atomIndex.length !== state.N) {
    throw new Error("draw-latent state is missing per-particle atom lineage");
  }
  const draws = normalizedArtifactDraws(artifact);
  for (let n = 0; n < state.N; n += 1) {
    const atom = state.atomIndex[n];
    if (atom < 0 || atom >= draws.length) {
      throw new Error("atom lineage index is outside the artifact draws");
    }
  }
  return draws;
}

/** Posterior mass per artifact atom — the session's inferred draw distribution. */
export function atomPosterior(
  state: ProtocolParticleState,
  artifact: ConditionalF1ResponseArtifact,
): number[] {
  const draws = normalizedArtifactDraws(artifact);
  if (!state.atomIndex || state.atomIndex.length !== state.N) {
    throw new Error("draw-latent state is missing per-particle atom lineage");
  }
  const mass = new Array<number>(draws.length).fill(0);
  for (let n = 0; n < state.N; n += 1) {
    const atom = state.atomIndex[n];
    if (atom < 0 || atom >= draws.length) {
      throw new Error("atom lineage index is outside the artifact draws");
    }
    mass[atom] += state.w[n];
  }
  return mass;
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
    ...(state.atomIndex ? { atomIndex: state.atomIndex.slice() } : {}),
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
  const atoms = drawLatentAtoms(state, profile, artifact);
  const nextLogLik = new Float64Array(state.N);
  const likelihood = new Float64Array(state.N);
  let maximumLogWeight = -Infinity;
  for (let n = 0; n < state.N; n += 1) {
    const lp = logProbability(
      profile, artifact, observation, segment,
      state.t, state.l, n, state.K,
      atoms && atoms[state.atomIndex![n]],
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
  const atoms = drawLatentAtoms(state, profile, artifact);
  for (const observation of state.history) {
    const segment = segmentFor(segments, observation);
    for (let n = 0; n < state.N; n += 1) {
      output[n] += logProbability(
        profile, artifact, observation, segment,
        proposedT, proposedL, n, state.K,
        atoms && atoms[state.atomIndex![n]],
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
  if (state.atomIndex) {
    // Atom indices ride ancestor selection as lineage; MH below proposes
    // (t, l) only, so the lineage is fixed for the rest of this sweep.
    const atomIndex = new Int32Array(N);
    for (let n = 0; n < N; n += 1) atomIndex[n] = state.atomIndex[ancestors[n]];
    state.atomIndex = atomIndex;
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
