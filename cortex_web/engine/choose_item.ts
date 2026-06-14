// A-optimal item selection — port of _expected_loss_vec /
// _coarse_to_fine_argmin / choose_item from engine/core_mcmc.py.
//
// For a candidate (task k, signal s), the expected post-answer TOTAL
// posterior variance over all 2K coordinates is computed by marginalizing
// over the predicted answer y. choose_item picks the (k, s, segId) that
// minimizes it across the remaining bank.

import { ParticleState } from "./types";
import { pResponseYes, signalZ } from "./likelihood";

// Expected total posterior variance for ONE candidate (task k, signal s,sSd).
// Returns the scalar EV loss. (The Python version vectorizes over a signal
// grid; here each bank candidate is one scalar call, which is plenty fast at
// N=600 and matches the per-segment bank structure.)
export function expectedLoss(
  st: ParticleState,
  k: number,
  s: number,
  sSd: number,
): number {
  const { N, K, t, l, w } = st;
  // predicted P(yes) per particle and the two reweightings
  const p = new Float64Array(N);
  let pYes = 0;
  let sumY1 = 0;
  let sumY0 = 0;
  for (let n = 0; n < N; n++) {
    const off = n * K + k;
    const z = signalZ(l[off], t[off], s, sSd);
    let pn = pResponseYes(z);
    if (pn < 1e-9) pn = 1e-9;
    else if (pn > 1 - 1e-9) pn = 1 - 1e-9;
    p[n] = pn;
    const wy1 = pn * w[n];
    const wy0 = (1 - pn) * w[n];
    pYes += wy1;
    sumY1 += wy1;
    sumY0 += wy0;
  }
  // total variance over all 2K coords under each hypothetical answer
  let totalY1 = 0;
  let totalY0 = 0;
  for (let kk = 0; kk < K; kk++) {
    // t-block coord kk
    totalY1 += weightedVar(t, kk, K, p, w, sumY1, true);
    totalY0 += weightedVar(t, kk, K, p, w, sumY0, false);
    // l-block coord kk
    totalY1 += weightedVar(l, kk, K, p, w, sumY1, true);
    totalY0 += weightedVar(l, kk, K, p, w, sumY0, false);
  }
  return pYes * totalY1 + (1 - pYes) * totalY0;
}

// weighted variance of coordinate `kk` under the y=1 (useP=true) or y=0
// reweighting w_y ∝ p·w (or (1-p)·w), normalized by `norm`.
function weightedVar(
  arr: Float64Array,
  kk: number,
  K: number,
  p: Float64Array,
  w: Float64Array,
  norm: number,
  useP: boolean,
): number {
  const N = w.length;
  let mu = 0;
  for (let n = 0; n < N; n++) {
    const wy = (useP ? p[n] : 1 - p[n]) * w[n];
    mu += wy * arr[n * K + kk];
  }
  mu /= norm;
  let v = 0;
  for (let n = 0; n < N; n++) {
    const wy = (useP ? p[n] : 1 - p[n]) * w[n];
    const d = arr[n * K + kk] - mu;
    v += wy * d * d;
  }
  return v / norm;
}

export interface BankArrays {
  // per task k: parallel arrays over the remaining candidate segments
  sMean: number[][]; // [K][nRemaining]
  sSd: number[][];
  segId: number[][];
}

export interface Chosen {
  k: number;
  s: number;
  sSd: number;
  segId: number;
  loss: number;
}

// Global argmin over the ACTIVE tasks × their remaining candidates. `active`
// restricts which task indices are considered (spike-first sectioning passes
// [0]; otherwise all unresolved tasks with a non-empty bank). Defaults to all.
export function chooseItem(st: ParticleState, bank: BankArrays, active?: number[]): Chosen {
  const tasks = active ?? Array.from({ length: st.K }, (_, k) => k);
  let best: Chosen = { k: 0, s: 0, sSd: 0, segId: -1, loss: Infinity };
  for (const k of tasks) {
    const sigs = bank.sMean[k];
    const sds = bank.sSd[k];
    const ids = bank.segId[k];
    for (let i = 0; i < sigs.length; i++) {
      const loss = expectedLoss(st, k, sigs[i], sds[i]);
      if (loss < best.loss) {
        best = { k, s: sigs[i], sSd: sds[i], segId: ids[i], loss };
      }
    }
  }
  return best;
}

// Trial-0 variation: collect all (loss, k, i), take the top-N lowest, pick one
// uniformly (per-examinee opening question). Mirrors CortexSession._pick_top_n.
export function chooseFirstItem(
  st: ParticleState,
  bank: BankArrays,
  topN: number,
  rng: { int: (n: number) => number },
  active?: number[],
): Chosen {
  const tasks = active ?? Array.from({ length: st.K }, (_, k) => k);
  const scored: Chosen[] = [];
  for (const k of tasks) {
    const sigs = bank.sMean[k];
    for (let i = 0; i < sigs.length; i++) {
      scored.push({
        k,
        s: sigs[i],
        sSd: bank.sSd[k][i],
        segId: bank.segId[k][i],
        loss: expectedLoss(st, k, sigs[i], bank.sSd[k][i]),
      });
    }
  }
  scored.sort((a, b) => a.loss - b.loss);
  const nTop = Math.min(topN, scored.length);
  return scored[rng.int(nTop)];
}
