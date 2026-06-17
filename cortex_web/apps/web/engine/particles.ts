// Particle cloud: init, reweight (update), ESS, resample + MH rejuvenation.
// Port of make_state_hier / update / ess / resample_and_rejuvenate /
// mh_rejuvenate from engine/core_mcmc.py.

import { ParticleState, PriorPair } from "./types";
import { logPResponse, signalZ } from "./likelihood";
import { logPriorOne, samplePrior } from "./prior";
import { covRows, cholesky, symSqrtClipped, Mat } from "./linalg";
import { Rng } from "./rng";

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
  return { N, K, t, l, w, logPrior, logLik, history: [], prior };
}

// Reweight by the likelihood of (k, s, y); update history + logLik in place.
export function update(
  st: ParticleState,
  k: number,
  s: number,
  y: 0 | 1,
  sSd = 0,
): void {
  const { N, K, t, l, w, logLik } = st;
  let sumW = 0;
  for (let n = 0; n < N; n++) {
    const off = n * K + k;
    const z = signalZ(l[off], t[off], s, sSd);
    const lp = logPResponse(z, y);
    logLik[n] += lp;
    const nw = w[n] * Math.exp(lp);
    w[n] = nw;
    sumW += nw;
  }
  if (sumW <= 0) {
    w.fill(1 / N);
  } else {
    for (let n = 0; n < N; n++) w[n] /= sumW;
  }
  st.history.push({ k, s, y, sSd });
}

export function ess(w: Float64Array): number {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i] * w[i];
  return 1 / s;
}

// cumulative log-likelihood of the full history for a proposed (t',l') cloud.
// Vectorized over particles; mirrors _log_lik_history.
function logLikHistory(
  st: ParticleState,
  tNew: Float64Array,
  lNew: Float64Array,
  out: Float64Array,
): void {
  const { N, K, history } = st;
  out.fill(0);
  for (const { k, s, y, sSd } of history) {
    for (let n = 0; n < N; n++) {
      const off = n * K + k;
      const z = signalZ(lNew[off], tNew[off], s, sSd);
      out[n] += logPResponse(z, y);
    }
  }
}

// Multinomial resample, then n MH-rejuvenation steps. proposalScale =
// 2.38/√(2K). Returns mean acceptance rate (diagnostic).
export function resampleAndRejuvenate(
  st: ParticleState,
  rng: Rng,
  nMhSteps: number,
  proposalScale: number,
): number {
  const { N, K } = st;
  // --- multinomial resample ---
  const idx = rng.resampleIndices(st.w, N);
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

  // --- MH rejuvenation ---
  const D = 2 * K;
  const theta = new Float64Array(N * D); // [t | l] per particle
  const accepts: number[] = [];
  const tNew = new Float64Array(N * K);
  const lNew = new Float64Array(N * K);
  const lpNew = new Float64Array(N);
  const llNew = new Float64Array(N);
  const eps = new Float64Array(D);

  for (let step = 0; step < nMhSteps; step++) {
    // pack theta = [t, l]
    for (let n = 0; n < N; n++) {
      for (let i = 0; i < K; i++) {
        theta[n * D + i] = st.t[n * K + i];
        theta[n * D + K + i] = st.l[n * K + i];
      }
    }
    // proposal scale matrix from cloud covariance (+1e-6 I); chol or eigh
    let cov: Mat = covRows(theta, N, D);
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
    // log posterior at proposal
    for (let n = 0; n < N; n++)
      lpNew[n] = logPriorOne(tNew, lNew, n, st.prior.tPieces, st.prior.lPieces);
    logLikHistory(st, tNew, lNew, llNew);
    // accept
    let nAcc = 0;
    for (let n = 0; n < N; n++) {
      const logAlpha = lpNew[n] + llNew[n] - (st.logPrior[n] + st.logLik[n]);
      if (Math.log(rng.random()) < logAlpha) {
        for (let i = 0; i < K; i++) {
          st.t[n * K + i] = tNew[n * K + i];
          st.l[n * K + i] = lNew[n * K + i];
        }
        st.logPrior[n] = lpNew[n];
        st.logLik[n] = llNew[n];
        nAcc++;
      }
    }
    accepts.push(nAcc / N);
  }
  return accepts.reduce((a, b) => a + b, 0) / (accepts.length || 1);
}

// weighted per-task posterior means + standard deviations (telemetry).
// Single pass: accumulates first and second moments simultaneously.
export function posteriorMeans(st: ParticleState): {
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
