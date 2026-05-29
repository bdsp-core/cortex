// Hierarchical zero-mean MVN prior — port of the engine's
// _precompute_prior_pieces / sample_prior_hier_K / log_prior_hier.
//
// The desktop uses the SAME fitted Corr_l (K×K, unit diagonal) for BOTH the
// t-block and the l-block (make_state_hier(Sigma_l=Corr_l, Sigma_t=Corr_l)),
// so a single PriorPieces serves both. Zero mean (no boundary-prior shift in
// the Mode-A live test).

import { addJitter, cholesky, invSPD, logDetSPD, Mat } from "./linalg";
import { PriorPieces } from "./types";
import { Rng } from "./rng";

export function precomputePrior(corrL: Mat): PriorPieces {
  const K = corrL.length;
  const S = addJitter(corrL, 1e-9);
  return {
    K,
    sigmaInv: invSPD(S),
    logDet: logDetSPD(S),
    L: cholesky(S),
  };
}

// log N(x | 0, Σ) up to the constant that cancels in the MH ratio, for one
// block x (length K). Matches _log_mvn_zero_mean: -0.5 xᵀΣ⁻¹x - 0.5 log|Σ|.
function logMvnZeroMean(
  x: Float64Array,
  off: number,
  K: number,
  sigmaInv: Mat,
  logDet: number,
): number {
  let qf = 0;
  for (let i = 0; i < K; i++) {
    const xi = x[off + i];
    let row = 0;
    for (let j = 0; j < K; j++) row += sigmaInv[i][j] * x[off + j];
    qf += xi * row;
  }
  return -0.5 * qf - 0.5 * logDet;
}

// log p(t_n, l_n) for particle n (both blocks share the prior).
export function logPriorOne(
  t: Float64Array,
  l: Float64Array,
  n: number,
  p: PriorPieces,
): number {
  const off = n * p.K;
  return (
    logMvnZeroMean(t, off, p.K, p.sigmaInv, p.logDet) +
    logMvnZeroMean(l, off, p.K, p.sigmaInv, p.logDet)
  );
}

// Sample N particles from the prior. Fills t,l (row-major N*K) and returns
// the per-particle log-prior. x = ε · Lᵀ with ε ~ N(0,I).
export function samplePrior(
  N: number,
  p: PriorPieces,
  rng: Rng,
  t: Float64Array,
  l: Float64Array,
  logPrior: Float64Array,
): void {
  const K = p.K;
  const eps = new Float64Array(K);
  for (let n = 0; n < N; n++) {
    const off = n * K;
    // t-block
    rng.fillGaussian(eps);
    for (let i = 0; i < K; i++) {
      let acc = 0;
      // (ε · Lᵀ)_i = Σ_j ε_j L_ij   (L lower-triangular)
      for (let j = 0; j <= i; j++) acc += eps[j] * p.L[i][j];
      t[off + i] = acc;
    }
    // l-block
    rng.fillGaussian(eps);
    for (let i = 0; i < K; i++) {
      let acc = 0;
      for (let j = 0; j <= i; j++) acc += eps[j] * p.L[i][j];
      l[off + i] = acc;
    }
    logPrior[n] = logPriorOne(t, l, n, p);
  }
}
