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

export interface SelectionOptions {
  nSubsample?: number;
  uncertaintyAware?: boolean;
}

function roundToEven(x: number): number {
  const lo = Math.floor(x), frac = x - lo;
  if (frac < 0.5) return lo;
  if (frac > 0.5) return lo + 1;
  return lo % 2 === 0 ? lo : lo + 1;
}

function uniqueSorted(values: number[]): number[] {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function stableLossOrder(indices: number[], losses: number[]): number[] {
  return indices.map((value, position) => ({ value, position, loss: losses[position] }))
    .sort((a, b) => (a.loss - b.loss) || (a.position - b.position))
    .map((x) => x.value);
}

function uncertaintyAwareTopIndices(
  st: ParticleState, k: number, sigs: number[], sds: number[],
  nTop: number, nCoarse: number,
): number[] {
  const n = sigs.length;
  nTop = Math.min(Math.max(1, nTop), n);
  const full = Array.from({ length: n }, (_, i) => i);
  const losses = (indices: number[]) => indices.map(
    (i) => expectedLoss(st, k, sigs[i], sds[i]),
  );
  if (n <= nCoarse) return stableLossOrder(full, losses(full)).slice(0, nTop);

  // np.array_split(arange(n), nCoarse): the first n%nCoarse bins receive one
  // extra element. Each bin contributes its centre and first minimum-sd item.
  const q = Math.floor(n / nCoarse), r = n % nCoarse;
  const representatives: number[] = [];
  let start = 0;
  for (let b = 0; b < nCoarse; b++) {
    const size = q + (b < r ? 1 : 0);
    if (!size) continue;
    representatives.push(start + Math.floor(size / 2));
    let minI = start;
    for (let i = start + 1; i < start + size; i++) {
      if (sds[i] < sds[minI]) minI = i;
    }
    representatives.push(minI);
    start += size;
  }
  const reps = uniqueSorted(representatives);
  const best = stableLossOrder(reps, losses(reps)).slice(0, Math.max(3, nTop));
  const h = Math.ceil(n / nCoarse);
  const candidates = reps.slice();
  for (const center of best) {
    for (let i = Math.max(0, center - h); i <= Math.min(n - 1, center + h); i++) {
      candidates.push(i);
    }
  }
  const cand = uniqueSorted(candidates);
  return stableLossOrder(cand, losses(cand)).slice(0, nTop);
}

function coarseToFineTopIndices(
  st: ParticleState, k: number, sigs: number[], sds: number[],
  nTop: number, nCoarse: number,
): number[] {
  const n = sigs.length;
  nTop = Math.min(Math.max(1, nTop), n);
  const full = Array.from({ length: n }, (_, i) => i);
  const losses = (indices: number[]) => indices.map(
    (i) => expectedLoss(st, k, sigs[i], sds[i]),
  );
  if (n <= nCoarse) return stableLossOrder(full, losses(full)).slice(0, nTop);
  const coarse = uniqueSorted(Array.from({ length: nCoarse }, (_, i) =>
    roundToEven(i * (n - 1) / (nCoarse - 1))));
  const brackets = stableLossOrder(coarse, losses(coarse)).slice(0, Math.max(nTop, 3));
  const h = Math.ceil(n / nCoarse);
  const candidates = coarse.slice();
  for (const center of brackets) {
    for (let i = Math.max(0, center - h); i <= Math.min(n - 1, center + h); i++) {
      candidates.push(i);
    }
  }
  const cand = uniqueSorted(candidates);
  return stableLossOrder(cand, losses(cand)).slice(0, nTop);
}

function candidateIndices(
  st: ParticleState, k: number, sigs: number[], sds: number[],
  nTop: number, options: SelectionOptions,
): number[] {
  if (!sigs.length) return [];
  const nCoarse = options.nSubsample;
  if (!nCoarse || sigs.length <= nCoarse) {
    const idx = Array.from({ length: sigs.length }, (_, i) => i);
    const losses = idx.map((i) => expectedLoss(st, k, sigs[i], sds[i]));
    return stableLossOrder(idx, losses).slice(0, nTop);
  }
  return options.uncertaintyAware
    ? uncertaintyAwareTopIndices(st, k, sigs, sds, nTop, nCoarse)
    : coarseToFineTopIndices(st, k, sigs, sds, nTop, nCoarse);
}

// Precision's coarse-to-fine scan assumes each domain is stable-mergesorted by
// signal. Index is the explicit stability tie-break, matching Python.
export function sortBankBySignalStable(bank: BankArrays): BankArrays {
  const out: BankArrays = { sMean: [], sSd: [], segId: [] };
  for (let k = 0; k < bank.sMean.length; k++) {
    const order = bank.sMean[k].map((value, index) => ({ value, index }))
      .sort((a, b) => (a.value - b.value) || (a.index - b.index))
      .map((x) => x.index);
    out.sMean.push(order.map((i) => bank.sMean[k][i]));
    out.sSd.push(order.map((i) => bank.sSd[k][i]));
    out.segId.push(order.map((i) => bank.segId[k][i]));
  }
  return out;
}

// Global argmin over all tasks × remaining candidates. `excludedTasks` (used
// by the session-level variety cap) skips whole task indices; if nothing
// remains, returns segId === -1 so the caller can retry without exclusions.
export function chooseItem(
  st: ParticleState,
  bank: BankArrays,
  excludedTasks?: ReadonlySet<number>,
  options: SelectionOptions = {},
): Chosen {
  let best: Chosen = { k: 0, s: 0, sSd: 0, segId: -1, loss: Infinity };
  for (let k = 0; k < st.K; k++) {
    if (excludedTasks?.has(k)) continue;
    const sigs = bank.sMean[k];
    const sds = bank.sSd[k];
    const ids = bank.segId[k];
    const indices = candidateIndices(st, k, sigs, sds, 1, options);
    for (const i of indices) {
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
// `excludedTasks` is honored the same way as in chooseItem so the phase-aware
// session can scope trial 0 to (e.g.) the spike block only.
export function chooseFirstItem(
  st: ParticleState,
  bank: BankArrays,
  topN: number,
  rng: { int: (n: number) => number },
  excludedTasks?: ReadonlySet<number>,
  options: SelectionOptions = {},
): Chosen {
  const scored: Chosen[] = [];
  for (let k = 0; k < st.K; k++) {
    if (excludedTasks?.has(k)) continue;
    const sigs = bank.sMean[k];
    const indices = candidateIndices(st, k, sigs, bank.sSd[k], topN, options);
    for (const i of indices) {
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
  return nTop ? scored[rng.int(nTop)] : { k: 0, s: 0, sSd: 0, segId: -1, loss: Infinity };
}
