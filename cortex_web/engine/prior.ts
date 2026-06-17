// Hierarchical zero-mean MVN prior — port of the engine's
// _precompute_prior_pieces / sample_prior_hier_K / log_prior_hier.
//
// The frozen-pilot instrument uses the SAME fitted Corr_l (K×K, unit diagonal)
// for BOTH the t-block and the l-block (make_state_hier(Sigma_l=Corr_l,
// Sigma_t=Corr_l)). v15 staging (OPT-IN) supplies a SEPARATE Corr_t for the
// t-block only; the l-block ALWAYS keeps Corr_l. We carry a {tPieces, lPieces}
// pair: on the pilot path both are precomputePrior(corrL) and every float op is
// in the same order as the single-PriorPieces era (bit-identical). Zero mean
// (no boundary-prior shift in the Mode-A live test).

import { addJitter, cholesky, invSPD, logDetSPD, Mat } from "./linalg";
import { PriorPieces, PriorPair } from "./types";
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

// Build the t-/l-block prior pair. l-block ALWAYS uses corrL; t-block uses
// corrT when supplied (v15 OPT-IN), else corrL (frozen-pilot — both blocks
// precomputePrior(corrL), bit-identical to the single-PriorPieces era).
export function precomputePriorPair(corrL: Mat, corrT?: Mat): PriorPair {
  return {
    tPieces: precomputePrior(corrT ?? corrL),
    lPieces: precomputePrior(corrL),
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

// log p(t_n, l_n) for particle n. t-block uses tPieces, l-block uses lPieces.
// On the pilot path tPieces === lPieces structurally (both precomputePrior(
// corrL)) so this is identical to the single-PriorPieces version.
export function logPriorOne(
  t: Float64Array,
  l: Float64Array,
  n: number,
  tPieces: PriorPieces,
  lPieces: PriorPieces,
): number {
  const off = n * tPieces.K;
  return (
    logMvnZeroMean(t, off, tPieces.K, tPieces.sigmaInv, tPieces.logDet) +
    logMvnZeroMean(l, off, lPieces.K, lPieces.sigmaInv, lPieces.logDet)
  );
}

// Sample N particles from the prior. Fills t,l (row-major N*K) and returns
// the per-particle log-prior. x = ε · Lᵀ with ε ~ N(0,I). The t-block uses
// tPieces.L, the l-block uses lPieces.L; the RNG draw order (t-block ε then
// l-block ε per particle) is unchanged from the single-PriorPieces era, so on
// the pilot path (tPieces.L === lPieces.L numerically) the output is
// bit-identical.
export function samplePrior(
  N: number,
  tPieces: PriorPieces,
  lPieces: PriorPieces,
  rng: Rng,
  t: Float64Array,
  l: Float64Array,
  logPrior: Float64Array,
): void {
  const K = tPieces.K;
  const eps = new Float64Array(K);
  for (let n = 0; n < N; n++) {
    const off = n * K;
    // t-block
    rng.fillGaussian(eps);
    for (let i = 0; i < K; i++) {
      let acc = 0;
      // (ε · Lᵀ)_i = Σ_j ε_j L_ij   (L lower-triangular)
      for (let j = 0; j <= i; j++) acc += eps[j] * tPieces.L[i][j];
      t[off + i] = acc;
    }
    // l-block
    rng.fillGaussian(eps);
    for (let i = 0; i < K; i++) {
      let acc = 0;
      for (let j = 0; j <= i; j++) acc += eps[j] * lPieces.L[i][j];
      l[off + i] = acc;
    }
    logPrior[n] = logPriorOne(t, l, n, tPieces, lPieces);
  }
}
