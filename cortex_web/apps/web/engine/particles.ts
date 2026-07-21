// Particle cloud: init, reweight (update), ESS, resample + MH rejuvenation.
// Port of make_state_hier / update / ess / resample_and_rejuvenate /
// mh_rejuvenate from engine/core_mcmc.py.

import type {
  PackedParticleHistory, ParticleObservation, ParticlePhaseTimingV2, ParticleState, PriorPair,
} from "./types";
import { logPResponse, signalZ } from "./likelihood";
import {
  logCategoricalObservationProbability, logObservationProbability,
  makeObservationLikelihoodWorkspace,
} from "./nway_likelihood";
import { logPriorOne, samplePrior } from "./prior";
import { covRows, cholesky, symSqrtClipped, Mat } from "./linalg";
import { Rng } from "./rng";
import type { NWaySelectionExecutor } from "./nway_selector_executor";
import {
  isSpeculationCancelled, speculationCancellationCheckpoint,
  throwIfSpeculationCancelled,
} from "./speculation_cancellation";

export function makeState(
  N: number,
  K: number,
  prior: PriorPair,
  rng: Rng,
): ParticleState {
  const t = new Float64Array(N * K);
  const l = new Float64Array(N * K);
  const w = new Float64Array(N).fill(1 / N);
  const logPrior = new Float64Array(N);
  const logLik = new Float64Array(N);
  samplePrior(N, prior.tPieces, prior.lPieces, rng, t, l, logPrior);
  return {
    N, K, t, l, w, logPrior, logLik, history: [],
    packedHistory: makePackedHistory(K), prior,
  };
}

function makePackedHistory(K: number, capacity = 16): PackedParticleHistory {
  return {
    K,
    length: 0,
    capacity,
    kind: new Uint8Array(capacity),
    taskK: new Int8Array(capacity),
    pick: new Int8Array(capacity),
    binaryS: new Float64Array(capacity),
    binarySd: new Float64Array(capacity),
    signalMean: new Float64Array(capacity * K),
    signalSd: new Float64Array(capacity * K),
  };
}

function clonePackedHistory(history: PackedParticleHistory): PackedParticleHistory {
  return {
    K: history.K,
    length: history.length,
    capacity: history.capacity,
    kind: history.kind.slice(),
    taskK: history.taskK.slice(),
    pick: history.pick.slice(),
    binaryS: history.binaryS.slice(),
    binarySd: history.binarySd.slice(),
    signalMean: history.signalMean.slice(),
    signalSd: history.signalSd.slice(),
  };
}

function growPackedHistory(history: PackedParticleHistory): void {
  const capacity = history.capacity * 2;
  const grow = <T extends Uint8Array | Int8Array | Float64Array>(
    source: T, next: T,
  ): T => {
    next.set(source);
    return next;
  };
  history.kind = grow(history.kind, new Uint8Array(capacity));
  history.taskK = grow(history.taskK, new Int8Array(capacity));
  history.pick = grow(history.pick, new Int8Array(capacity));
  history.binaryS = grow(history.binaryS, new Float64Array(capacity));
  history.binarySd = grow(history.binarySd, new Float64Array(capacity));
  history.signalMean = grow(history.signalMean, new Float64Array(capacity * history.K));
  history.signalSd = grow(history.signalSd, new Float64Array(capacity * history.K));
  history.capacity = capacity;
}

function appendPackedObservation(
  history: PackedParticleHistory, observation: ParticleObservation,
): void {
  if (history.length === history.capacity) growPackedHistory(history);
  const index = history.length;
  if (observation.kind === "binary") {
    history.kind[index] = 0;
    history.taskK[index] = observation.k;
    history.pick[index] = observation.y;
    history.binaryS[index] = observation.s;
    history.binarySd[index] = observation.sSd;
  } else {
    history.kind[index] = 1;
    history.taskK[index] = observation.askedK;
    history.pick[index] = observation.pickK;
    const offset = index * history.K;
    for (let k = 0; k < history.K; k++) {
      history.signalMean[offset + k] = observation.sMean[k];
      history.signalSd[offset + k] = observation.sSd[k];
    }
  }
  history.length += 1;
}

export function packParticleHistory(
  observations: readonly ParticleObservation[], K: number,
): PackedParticleHistory {
  let capacity = 16;
  while (capacity < observations.length) capacity *= 2;
  const packed = makePackedHistory(K, capacity);
  for (const observation of observations) appendPackedObservation(packed, observation);
  return packed;
}

function ensurePackedHistory(st: ParticleState): PackedParticleHistory {
  if (!st.packedHistory || st.packedHistory.K !== st.K
      || st.packedHistory.length !== st.history.length) {
    st.packedHistory = packParticleHistory(st.history, st.K);
  }
  return st.packedHistory;
}

// Exact deep copy of a particle cloud — for speculative branch isolation
// (engine/advance.ts). Typed arrays are sliced; history is shallow-copied (its
// {k,s,y,sSd} elements are never mutated, only appended); the immutable prior
// pieces are shared.
export function cloneState(st: ParticleState): ParticleState {
  return {
    N: st.N,
    K: st.K,
    t: st.t.slice(),
    l: st.l.slice(),
    w: st.w.slice(),
    logPrior: st.logPrior.slice(),
    logLik: st.logLik.slice(),
    history: st.history.map((observation) => observation.kind === "categorical_f1"
      ? { ...observation, sMean: observation.sMean.slice(), sSd: observation.sSd.slice() }
      : { ...observation }),
    ...(st.packedHistory
      ? { packedHistory: clonePackedHistory(st.packedHistory) }
      : {}),
    prior: st.prior,
    ...(st.lastRejuvenation ? { lastRejuvenation: { ...st.lastRejuvenation } } : {}),
  };
}

export class PosteriorUpdateError extends Error {}

// Reweight by the likelihood of (k, s, y). All derived fields are validated
// before commit: a zero/non-finite update throws with the cloud byte-identical
// to its pre-answer state. The old uniform-reset behavior was not fail-closed.
export function update(
  st: ParticleState,
  k: number,
  s: number,
  y: 0 | 1,
  sSd = 0,
): void {
  const { N, K, t, l, w, logLik } = st;
  if (y !== 0 && y !== 1) throw new Error(`binary response y must be 0 or 1, got ${y}`);
  if (!Number.isFinite(s) || !Number.isFinite(sSd) || sSd < 0) {
    throw new Error(`invalid signal parameters s=${s}, sSd=${sSd}`);
  }
  let oldSum = 0;
  for (let n = 0; n < N; n++) {
    if (!Number.isFinite(w[n]) || w[n] < 0) {
      throw new PosteriorUpdateError("pre-update particle weights are invalid");
    }
    if (!Number.isFinite(logLik[n])) {
      throw new PosteriorUpdateError("pre-update log likelihood is invalid");
    }
    oldSum += w[n];
  }
  if (!Number.isFinite(oldSum) || oldSum <= 0) {
    throw new PosteriorUpdateError("pre-update particle weights are invalid");
  }

  const nextW = new Float64Array(N);
  const nextLogLik = new Float64Array(N);
  let sumW = 0;
  for (let n = 0; n < N; n++) {
    const off = n * K + k;
    const z = signalZ(l[off], t[off], s, sSd);
    const lp = logPResponse(z, y);
    const ll = logLik[n] + lp;
    const nw = w[n] * Math.exp(lp);
    if (!Number.isFinite(lp) || !Number.isFinite(ll) || !Number.isFinite(nw)) {
      throw new PosteriorUpdateError("posterior update has zero mass or non-finite values");
    }
    nextLogLik[n] = ll;
    nextW[n] = nw;
    sumW += nw;
  }
  if (!Number.isFinite(sumW) || sumW <= 0) {
    throw new PosteriorUpdateError("posterior update has zero mass or non-finite values");
  }
  for (let n = 0; n < N; n++) nextW[n] /= sumW;
  st.w = nextW;
  st.logLik = nextLogLik;
  const observation = { kind: "binary", k, s, y, sSd, rawPick: y === 1 ? k : st.K } as const;
  const packedHistory = ensurePackedHistory(st);
  st.history.push(observation);
  appendPackedObservation(packedHistory, observation);
}

// Production response update. IIIC observations retain the raw category and
// normalize in log space; spike observations preserve the original binary
// multiply/normalize order. All arrays commit transactionally only after every
// particle and the normalizer validate.
export function updateObservation(
  st: ParticleState, observation: ParticleObservation,
  timing?: ParticlePhaseTimingV2,
): void {
  if (observation.kind === "binary") {
    update(st, observation.k, observation.s, observation.y, observation.sSd);
    st.history[st.history.length - 1] = { ...observation };
    return;
  }
  const categoricalStartedAt = performance.now();
  const { N, K, w, logLik } = st;
  if (observation.sMean.length !== K || observation.sSd.length !== K
      || observation.sMean.some((value) => !Number.isFinite(value))
      || observation.sSd.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("invalid categorical signal vectors");
  }
  let maximumLogWeight = -Infinity;
  const likelihood = new Float64Array(N);
  const nextLogLik = new Float64Array(N);
  const likelihoodWorkspace = makeObservationLikelihoodWorkspace();
  for (let n = 0; n < N; n++) {
    if (!Number.isFinite(w[n]) || w[n] < 0 || !Number.isFinite(logLik[n])) {
      throw new PosteriorUpdateError("pre-update particle state is invalid");
    }
    const lp = logObservationProbability(
      observation, st.t, st.l, n, K, likelihoodWorkspace,
    );
    const cumulative = logLik[n] + lp;
    if (!Number.isFinite(lp) || !Number.isFinite(cumulative)) {
      throw new PosteriorUpdateError("categorical posterior update is non-finite");
    }
    likelihood[n] = lp;
    nextLogLik[n] = cumulative;
    if (w[n] > 0) maximumLogWeight = Math.max(
      maximumLogWeight, Math.log(w[n]) + lp,
    );
  }
  if (!Number.isFinite(maximumLogWeight)) {
    throw new PosteriorUpdateError("categorical posterior update has zero mass");
  }
  const nextW = new Float64Array(N);
  let sumW = 0;
  for (let n = 0; n < N; n++) {
    const value = w[n] === 0
      ? 0
      : Math.exp(Math.log(w[n]) + likelihood[n] - maximumLogWeight);
    nextW[n] = value;
    sumW += value;
  }
  if (!Number.isFinite(sumW) || sumW <= 0) {
    throw new PosteriorUpdateError("categorical posterior update has invalid mass");
  }
  for (let n = 0; n < N; n++) nextW[n] /= sumW;
  st.w = nextW;
  st.logLik = nextLogLik;
  const storedObservation = {
    ...observation,
    sMean: observation.sMean.slice(),
    sSd: observation.sSd.slice(),
  };
  const packedHistory = ensurePackedHistory(st);
  st.history.push(storedObservation);
  appendPackedObservation(packedHistory, storedObservation);
  if (timing) timing.categoricalUpdateMs += performance.now() - categoricalStartedAt;
}

export function ess(w: Float64Array): number {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i] * w[i];
  return 1 / s;
}

// cumulative log-likelihood of the full history for a proposed (t',l') cloud.
// Vectorized over particles; mirrors _log_lik_history.
export function logLikPackedHistory(
  history: PackedParticleHistory,
  N: number,
  K: number,
  tNew: Float64Array,
  lNew: Float64Array,
  out: Float64Array,
): void {
  if (tNew.length !== N * K || lNew.length !== N * K || out.length !== N) {
    throw new Error("packed history likelihood shard dimensions are invalid");
  }
  out.fill(0);
  const likelihoodWorkspace = makeObservationLikelihoodWorkspace();
  for (let historyIndex = 0; historyIndex < history.length; historyIndex++) {
    const taskK = history.taskK[historyIndex];
    if (history.kind[historyIndex] === 0) {
      const s = history.binaryS[historyIndex];
      const sSd = history.binarySd[historyIndex];
      const y = history.pick[historyIndex] as 0 | 1;
      for (let n = 0; n < N; n++) {
        const offset = n * K + taskK;
        out[n] += logPResponse(signalZ(
          lNew[offset], tNew[offset], s, sSd,
        ), y);
      }
    } else {
      const signalOffset = historyIndex * K;
      const pickK = history.pick[historyIndex];
      for (let n = 0; n < N; n++) {
        out[n] += logCategoricalObservationProbability(
          taskK, pickK,
          history.signalMean, history.signalSd, signalOffset,
          tNew, lNew, n, K, likelihoodWorkspace,
        );
      }
    }
  }
}

function logLikHistory(
  st: ParticleState,
  tNew: Float64Array,
  lNew: Float64Array,
  out: Float64Array,
): void {
  logLikPackedHistory(ensurePackedHistory(st), st.N, st.K, tNew, lNew, out);
}

// Multinomial resample, then n MH-rejuvenation steps. proposalScale =
// 2.38/√(2K). Returns mean acceptance rate (diagnostic).
export function resampleAndRejuvenate(
  st: ParticleState,
  rng: Rng,
  nMhSteps: number,
  proposalScale: number,
  qIndex = -1,
  timing?: ParticlePhaseTimingV2,
): number {
  const { N, K } = st;
  const resamplingStartedAt = performance.now();
  // --- multinomial resample ---
  const idx = rng.resampleIndices(st.w, N);
  const distinctAncestors = new Set(idx).size;
  const t2 = new Float64Array(N * K);
  const l2 = new Float64Array(N * K);
  const lp2 = new Float64Array(N);
  const ll2 = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    const src = idx[n];
    t2.set(st.t.subarray(src * K, src * K + K), n * K);
    l2.set(st.l.subarray(src * K, src * K + K), n * K);
    lp2[n] = st.logPrior[src];
    ll2[n] = st.logLik[src];
  }
  st.t = t2;
  st.l = l2;
  st.logPrior = lp2;
  st.logLik = ll2;
  st.w.fill(1 / N);
  if (timing) timing.resamplingMs += performance.now() - resamplingStartedAt;

  // --- MH rejuvenation ---
  const D = 2 * K;
  const theta = new Float64Array(N * D); // [t | l] per particle
  const accepts: number[] = [];
  const tNew = new Float64Array(N * K);
  const lNew = new Float64Array(N * K);
  const lpNew = new Float64Array(N);
  const llNew = new Float64Array(N);
  const eps = new Float64Array(D);
  const accepted = new Uint8Array(N);

  for (let step = 0; step < nMhSteps; step++) {
    const proposalStartedAt = performance.now();
    // pack theta = [t, l]
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < K; i++) {
        theta[n * D + i] = st.t[n * K + i];
        theta[n * D + K + i] = st.l[n * K + i];
      }
    }
    // proposal scale matrix from cloud covariance (+1e-6 I); chol or eigh
    const cov: Mat = covRows(theta, N, D);
    for (let i = 0; i < D; i++) cov[i][i] += 1e-6;
    let F: Mat;
    try {
      F = cholesky(cov); // lower-triangular
    } catch {
      F = symSqrtClipped(cov, 1e-6);
    }
    // propose theta' = theta + scale·(ε · Fᵀ)
    for (let n = 0; n < N; n++) {
      rng.fillGaussian(eps);
      for (let i = 0; i < D; i++) {
        let acc = 0;
        for (let j = 0; j < D; j++) acc += eps[j] * F[i][j];
        const v = theta[n * D + i] + proposalScale * acc;
        if (i < K) tNew[n * K + i] = v;
        else lNew[n * K + (i - K)] = v;
      }
    }
    if (timing) {
      timing.mhProposalGenerationMs += performance.now() - proposalStartedAt;
    }
    // log posterior at proposal
    const priorStartedAt = performance.now();
    for (let n = 0; n < N; n++)
      lpNew[n] = logPriorOne(tNew, lNew, n, st.prior.tPieces, st.prior.lPieces);
    if (timing) timing.mhPriorMs += performance.now() - priorStartedAt;
    const historyStartedAt = performance.now();
    logLikHistory(st, tNew, lNew, llNew);
    if (timing) timing.mhHistoryLikelihoodMs += performance.now() - historyStartedAt;
    // accept
    const acceptanceStartedAt = performance.now();
    let nAcc = 0;
    for (let n = 0; n < N; n++) {
      const logAlpha = lpNew[n] + llNew[n] - (st.logPrior[n] + st.logLik[n]);
      if (Math.log(rng.random()) < logAlpha) {
        accepted[n] = 1;
        nAcc++;
      } else accepted[n] = 0;
    }
    if (timing) timing.mhAcceptanceMs += performance.now() - acceptanceStartedAt;
    const copyingStartedAt = performance.now();
    for (let n = 0; n < N; n++) {
      if (!accepted[n]) continue;
      for (let i = 0; i < K; i++) {
        st.t[n * K + i] = tNew[n * K + i];
        st.l[n * K + i] = lNew[n * K + i];
      }
      st.logPrior[n] = lpNew[n];
      st.logLik[n] = llNew[n];
    }
    if (timing) timing.mhCopyingMs += performance.now() - copyingStartedAt;
    accepts.push(nAcc / N);
  }
  const acceptanceRate = accepts.reduce((a, b) => a + b, 0) / (accepts.length || 1);
  st.lastRejuvenation = {
    qIndex,
    acceptanceRate,
    distinctAncestors,
    distinctAncestorFraction: distinctAncestors / N,
  };
  return acceptanceRate;
}

/** Exact asynchronous counterpart used by the adaptive pool. RNG consumption,
 * proposal construction, priors, acceptance order, and copying are identical
 * to resampleAndRejuvenate. Only per-particle full-history likelihoods are
 * delegated; each worker retains original history order. */
export async function resampleAndRejuvenateWithExecutor(
  st: ParticleState,
  rng: Rng,
  nMhSteps: number,
  proposalScale: number,
  executor: NWaySelectionExecutor,
  qIndex = -1,
  timing?: ParticlePhaseTimingV2,
  cancellationSignal?: AbortSignal,
): Promise<number> {
  await speculationCancellationCheckpoint(cancellationSignal);
  const { N, K } = st;
  const resamplingStartedAt = performance.now();
  const idx = rng.resampleIndices(st.w, N);
  const distinctAncestors = new Set(idx).size;
  const t2 = new Float64Array(N * K);
  const l2 = new Float64Array(N * K);
  const lp2 = new Float64Array(N);
  const ll2 = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    const src = idx[n];
    t2.set(st.t.subarray(src * K, src * K + K), n * K);
    l2.set(st.l.subarray(src * K, src * K + K), n * K);
    lp2[n] = st.logPrior[src];
    ll2[n] = st.logLik[src];
  }
  st.t = t2;
  st.l = l2;
  st.logPrior = lp2;
  st.logLik = ll2;
  st.w.fill(1 / N);
  if (timing) timing.resamplingMs += performance.now() - resamplingStartedAt;

  const D = 2 * K;
  const theta = new Float64Array(N * D);
  const accepts: number[] = [];
  const tNew = new Float64Array(N * K);
  const lNew = new Float64Array(N * K);
  const lpNew = new Float64Array(N);
  const llNew = new Float64Array(N);
  const eps = new Float64Array(D);
  const accepted = new Uint8Array(N);
  const history = ensurePackedHistory(st);
  let parallelAvailable = true;

  for (let step = 0; step < nMhSteps; step++) {
    const proposalStartedAt = performance.now();
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < K; i++) {
        theta[n * D + i] = st.t[n * K + i];
        theta[n * D + K + i] = st.l[n * K + i];
      }
    }
    const cov: Mat = covRows(theta, N, D);
    for (let i = 0; i < D; i++) cov[i][i] += 1e-6;
    let F: Mat;
    try {
      F = cholesky(cov);
    } catch {
      F = symSqrtClipped(cov, 1e-6);
    }
    for (let n = 0; n < N; n++) {
      rng.fillGaussian(eps);
      for (let i = 0; i < D; i++) {
        let acc = 0;
        for (let j = 0; j < D; j++) acc += eps[j] * F[i][j];
        const value = theta[n * D + i] + proposalScale * acc;
        if (i < K) tNew[n * K + i] = value;
        else lNew[n * K + (i - K)] = value;
      }
    }
    if (timing) {
      timing.mhProposalGenerationMs += performance.now() - proposalStartedAt;
    }
    const priorStartedAt = performance.now();
    for (let n = 0; n < N; n++) {
      lpNew[n] = logPriorOne(tNew, lNew, n, st.prior.tPieces, st.prior.lPieces);
    }
    if (timing) timing.mhPriorMs += performance.now() - priorStartedAt;
    const historyStartedAt = performance.now();
    if (parallelAvailable) {
      try {
        llNew.set(await executor.historyLikelihood(history, N, K, tNew, lNew));
      } catch (error) {
        if (isSpeculationCancelled(error)) throw error;
        executor.dispose();
        parallelAvailable = false;
        logLikPackedHistory(history, N, K, tNew, lNew, llNew);
      }
    } else {
      logLikPackedHistory(history, N, K, tNew, lNew, llNew);
    }
    if (timing) timing.mhHistoryLikelihoodMs += performance.now() - historyStartedAt;
    await speculationCancellationCheckpoint(cancellationSignal);
    const acceptanceStartedAt = performance.now();
    let nAcc = 0;
    for (let n = 0; n < N; n++) {
      const logAlpha = lpNew[n] + llNew[n] - (st.logPrior[n] + st.logLik[n]);
      if (Math.log(rng.random()) < logAlpha) {
        accepted[n] = 1;
        nAcc++;
      } else accepted[n] = 0;
    }
    if (timing) timing.mhAcceptanceMs += performance.now() - acceptanceStartedAt;
    const copyingStartedAt = performance.now();
    for (let n = 0; n < N; n++) {
      if (!accepted[n]) continue;
      for (let i = 0; i < K; i++) {
        st.t[n * K + i] = tNew[n * K + i];
        st.l[n * K + i] = lNew[n * K + i];
      }
      st.logPrior[n] = lpNew[n];
      st.logLik[n] = llNew[n];
    }
    if (timing) timing.mhCopyingMs += performance.now() - copyingStartedAt;
    accepts.push(nAcc / N);
    throwIfSpeculationCancelled(cancellationSignal);
  }
  const acceptanceRate = accepts.reduce((a, b) => a + b, 0) / (accepts.length || 1);
  st.lastRejuvenation = {
    qIndex,
    acceptanceRate,
    distinctAncestors,
    distinctAncestorFraction: distinctAncestors / N,
  };
  return acceptanceRate;
}

// weighted per-task posterior means + standard deviations (telemetry).
// Single pass: accumulates first and second moments simultaneously.
export function posteriorMeans(st: Pick<ParticleState, "N" | "K" | "t" | "l" | "w">): {
  tMean: number[]; lMean: number[]; tSd: number[]; lSd: number[];
} {
  const { N, K, t, l, w } = st;
  const tMean = new Array(K).fill(0);
  const lMean = new Array(K).fill(0);
  const tM2 = new Array(K).fill(0);
  const lM2 = new Array(K).fill(0);
  for (let n = 0; n < N; n++) {
    const wn = w[n];
    for (let k = 0; k < K; k++) {
      const tk = t[n * K + k];
      const lk = l[n * K + k];
      tMean[k] += wn * tk;
      lMean[k] += wn * lk;
      tM2[k] += wn * tk * tk;
      lM2[k] += wn * lk * lk;
    }
  }
  const tSd = new Array(K).fill(0);
  const lSd = new Array(K).fill(0);
  for (let k = 0; k < K; k++) {
    tSd[k] = Math.sqrt(Math.max(0, tM2[k] - tMean[k] * tMean[k]));
    lSd[k] = Math.sqrt(Math.max(0, lM2[k] - lMean[k] * lMean[k]));
  }
  return { tMean, lMean, tSd, lSd };
}
