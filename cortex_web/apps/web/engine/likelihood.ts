// Lapse-probit response model — port of engine/core_mcmc.py
// _log_p_response / _p_response_yes.
//
//   P(y=1 | z) = λ + (1 − 2λ)·Φ(z)        (spike-paper Eq. 2, symmetric lapse)
//   P(y=0 | z) = λ + (1 − 2λ)·Φ(−z)
//   z = exp(l_k)·(s + t_k),  optionally /√(1+(e^l·s_sd)²)
//
// λ = LAPSE_RATE = 0.025, fixed everywhere (the repo's hard invariant).

import { normCdf, logNdtr, logSumExp2 } from "./mathfns";

export const LAPSE_RATE = 0.025;
export const Z_BUFFER = 2.0;

const LOG_LAPSE = Math.log(LAPSE_RATE);
const LOG_ONE_MINUS_TWO_LAPSE = Math.log1p(-2.0 * LAPSE_RATE);

// log P(y | z; λ). Scalar form (the engine vectorizes over particles by
// calling this per-particle; see particles.ts for the hot loop).
export function logPResponse(z: number, y: 0 | 1): number {
  const a = LOG_ONE_MINUS_TWO_LAPSE + (y === 1 ? logNdtr(z) : logNdtr(-z));
  return logSumExp2(a, LOG_LAPSE);
}

// P(y=1 | z; λ) — used for item-selection moments.
export function pResponseYes(z: number): number {
  return LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * normCdf(z);
}

// z = exp(l)·(s + t), with optional s_sd attenuation (Phase 3.5).
export function signalZ(l: number, t: number, s: number, sSd = 0): number {
  const el = Math.exp(l);
  let z = el * (s + t);
  if (sSd !== 0) z = z / Math.sqrt(1.0 + (el * sSd) ** 2);
  return z;
}
