// Port of engine/auroc.py — per-task AUROC as a function of the posterior
// log-skill l. Equal-variance binormal SDT with MU_S = SIGMA_S = 1:
//
//   AUROC(l) = Φ( √2 · MU_S / √(exp(-l)² + SIGMA_S²) )
//
// The results screen shows the posterior-MEAN AUROC per task plus a (1-α)
// credible halfwidth, both read off the final particle cloud.

import { normCdf } from "./mathfns";

const MU_S = 1.0;
const SIGMA_S = 1.0;

export function aurocFromL(l: number): number {
  const sigma = Math.exp(-l);
  const u = (Math.SQRT2 * MU_S) / Math.sqrt(sigma * sigma + SIGMA_S * SIGMA_S);
  return normCdf(u);
}

// Weighted quantile of `vals` under normalized `w` (both length n), `vals`
// need not be sorted. Linear in n after an argsort.
function weightedQuantile(vals: number[], w: Float64Array, q: number): number {
  const idx = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
  let cum = 0;
  for (const i of idx) {
    cum += w[i];
    if (cum >= q) return vals[i];
  }
  return vals[idx[idx.length - 1]];
}

// Posterior-mean AUROC + (1-alpha) credible halfwidth per task k, from the
// final particle cloud (l is row-major N*K; w is the normalized weights).
export function aurocSummary(
  l: Float64Array,
  w: Float64Array,
  N: number,
  K: number,
  alpha = 0.05,
): { mean: number[]; hw: number[] } {
  const mean: number[] = new Array(K).fill(0);
  const hw: number[] = new Array(K).fill(0);
  for (let k = 0; k < K; k++) {
    const vals = new Array(N);
    let m = 0;
    for (let n = 0; n < N; n++) {
      const a = aurocFromL(l[n * K + k]);
      vals[n] = a;
      m += w[n] * a;
    }
    mean[k] = m;
    const lo = weightedQuantile(vals, w, alpha / 2);
    const hi = weightedQuantile(vals, w, 1 - alpha / 2);
    hw[k] = (hi - lo) / 2;
  }
  return { mean, hw };
}
