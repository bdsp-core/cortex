// One-step expected-reward scorer (G3 TS port of trainer/reward._expected_reward).
// The policy's bias-correction term wraps this. Deterministic; tolerance-parity
// gated. `learn` is always 1 here (the p_static mixture is an opt-in Python
// ablation, default off, and is not ported).
import { normCdf } from '../engine/mathfns';
import { LAPSE_RATE, SKILL_MODE_MULTIPLIER } from './conventions';
import type { FilterParams } from './filter';

export interface RewardWeights {
  betaT: number;
  betaSigma: number;
  betaR: number;
}

export interface CloudView {
  theta: number[];
  ell: number[];
  w: number[];
}

// Q(s): weighted expected one-step reward over the particle cloud, using the
// filter's OWN assumed dynamics (`params`). Soft-rule / y-independent form.
// s, yStar: (n,) candidate arrays → (n,) Q. sSd: scalar or (n,).
export function expectedReward(
  cloud: CloudView, params: FilterParams, s: number[], yStar: number[],
  weights: RewardWeights,
  opts: { retBonus?: number[]; sSd?: number | number[]; barEll?: number | null } = {},
): number[] {
  const n = s.length;
  const N = cloud.theta.length;
  const sSdArr = Array.isArray(opts.sSd)
    ? opts.sSd
    : new Array(n).fill(opts.sSd ?? 0);
  const Q = new Array(n).fill(0);
  const logSigInf = Math.log(params.sigmaInf);
  for (let c = 0; c < n; c++) {
    const sc = s[c];
    const yc = yStar[c];
    const ssd = sSdArr[c];
    const rb = opts.retBonus ? opts.retBonus[c] : 0;
    let q = 0;
    for (let i = 0; i < N; i++) {
      const sigma = Math.exp(-cloud.ell[i]);
      const t = -cloud.theta[i];
      const el = Math.exp(cloud.ell[i]);
      let z = el * (sc + cloud.theta[i]);
      z = z / Math.sqrt(1 + (el * ssd) ** 2);
      const pYes = LAPSE_RATE + (1 - 2 * LAPSE_RATE) * normCdf(z);
      const delta = pYes - yc;
      const tNew = t + params.alphaT * delta;
      const d = Math.abs(sc - t) / sigma;
      const v = params.rho ** 2 + (ssd / sigma) ** 2;
      const wWeight = (params.rho / Math.sqrt(v))
        * Math.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2) / (2 * v));
      const logSigNew = Math.log(sigma)
        - params.alphaSigma * wWeight * (Math.log(sigma) - logSigInf);
      let sigGain = sigma - Math.exp(logSigNew);
      if (opts.barEll != null && !(cloud.ell[i] < opts.barEll)) sigGain = 0;
      const R = weights.betaT * (Math.abs(t) - Math.abs(tNew))
        + weights.betaSigma * sigGain + weights.betaR * rb;
      q += R * cloud.w[i];
    }
    Q[c] = q;
  }
  return Q;
}
