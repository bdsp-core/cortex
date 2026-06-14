// AUROC machinery for the results screen — port of engine/auroc.py.
//
// Under the joint generative model Y ~ Bernoulli(Φ(e^ℓ (s + t))) and the
// single-task reference distribution s ~ N(±1, 1), the AUROC for one task is
// a deterministic function of the rater's log-skill ℓ:
//
//     AUROC(ℓ) = Φ( √2 / √(e^(−2ℓ) + 1) )
//
// Bias t doesn't enter — AUROC is a discrimination measure independent of
// the operating point. So with a posterior on ℓ we get a posterior on AUROC
// by passing each particle (or quantile) through this function.
//
// For the results screen we summarize the posterior with (a) a point
// estimate at the posterior mean of ℓ and (b) an approximate 95% CrI by
// transforming ℓ ± 1.96·SD(ℓ). The pass cut on AUROC is just AUROC(ℓ*),
// where ℓ* is the per-task Youden cut score from `cert_config`.

import { normCdf } from "./mathfns";

/** AUROC as a function of log-skill ℓ. */
export function aurocFromL(l: number): number {
  const u = Math.SQRT2 / Math.sqrt(Math.exp(-2 * l) + 1);
  return normCdf(u);
}

/** Posterior summary of AUROC for one task. `lMean` and `lSd` are the
 *  posterior mean and standard deviation of ℓ. Returns the point estimate
 *  (transform of the mean) and a 95% CrI (transform of ℓ ± 1.96·SD). The
 *  bounds are clamped to [0, 1] so a tiny SD around ℓ≈0 doesn't print 100.1%.
 */
export function aurocSummary(lMean: number, lSd: number): {
  point: number;
  lo95: number;
  hi95: number;
  halfwidth: number;
} {
  const z = 1.959963984540054; // 0.975 quantile of N(0,1)
  const point = aurocFromL(lMean);
  const lo95 = Math.max(0, Math.min(1, aurocFromL(lMean - z * lSd)));
  const hi95 = Math.max(0, Math.min(1, aurocFromL(lMean + z * lSd)));
  return { point, lo95, hi95, halfwidth: (hi95 - lo95) / 2 };
}
