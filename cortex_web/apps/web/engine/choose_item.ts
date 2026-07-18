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
// Returns the scalar EV loss. The Python version vectorizes over a signal
// grid; the browser evaluates served-bank candidates through a reused scalar
// workspace so it retains the per-segment selector and stable tie semantics.
export function expectedLoss(
  st: ParticleState,
  k: number,
  s: number,
  sSd: number,
): number {
  return expectedLossWithWorkspace(st, k, s, sSd, makeLossWorkspace(st));
}

interface ExpectedLossWorkspace {
  meanT1: Float64Array;
  meanT0: Float64Array;
  meanL1: Float64Array;
  meanL0: Float64Array;
  baselineTotalVariance: number;
}

function makeLossWorkspace(st: ParticleState): ExpectedLossWorkspace {
  const vector = () => new Float64Array(st.K);
  return {
    meanT1: vector(), meanT0: vector(), meanL1: vector(), meanL0: vector(),
    baselineTotalVariance: totalPosteriorVariance(st),
  };
}

function totalPosteriorVariance(st: ParticleState): number {
  const meanT = new Float64Array(st.K);
  const meanL = new Float64Array(st.K);
  let weightSum = 0;
  for (let n = 0; n < st.N; n++) {
    const weight = st.w[n];
    weightSum += weight;
    const base = n * st.K;
    for (let k = 0; k < st.K; k++) {
      meanT[k] += weight * st.t[base + k];
      meanL[k] += weight * st.l[base + k];
    }
  }
  for (let k = 0; k < st.K; k++) {
    meanT[k] /= weightSum;
    meanL[k] /= weightSum;
  }
  let total = 0;
  for (let n = 0; n < st.N; n++) {
    const weight = st.w[n] / weightSum;
    const base = n * st.K;
    for (let k = 0; k < st.K; k++) {
      const dt = st.t[base + k] - meanT[k];
      const dl = st.l[base + k] - meanL[k];
      total += weight * dt * dt;
      total += weight * dl * dl;
    }
  }
  return total;
}

function expectedLossWithWorkspace(
  st: ParticleState,
  k: number,
  s: number,
  sSd: number,
  workspace: ExpectedLossWorkspace,
): number {
  const { N, K, t, l, w } = st;
  // By the law of total variance,
  //   E_y[Var(X|y)] = Var(X) - Var_y(E[X|y]).
  // The current total variance is candidate-independent and cached once per
  // domain scan. Each candidate therefore needs only its two conditional
  // means, not a second full particle pass for 28 conditional variances.
  const { meanT1, meanT0, meanL1, meanL0 } = workspace;
  meanT1.fill(0);
  meanT0.fill(0);
  meanL1.fill(0);
  meanL0.fill(0);
  let pYes = 0;
  let sumY1 = 0;
  let sumY0 = 0;
  for (let n = 0; n < N; n++) {
    const off = n * K + k;
    const z = signalZ(l[off], t[off], s, sSd);
    let pn = pResponseYes(z);
    if (pn < 1e-9) pn = 1e-9;
    else if (pn > 1 - 1e-9) pn = 1 - 1e-9;
    const wy1 = pn * w[n];
    const wy0 = (1 - pn) * w[n];
    pYes += wy1;
    sumY1 += wy1;
    sumY0 += wy0;
    const base = n * K;
    for (let kk = 0; kk < K; kk++) {
      const tv = t[base + kk];
      const lv = l[base + kk];
      meanT1[kk] += wy1 * tv;
      meanT0[kk] += wy0 * tv;
      meanL1[kk] += wy1 * lv;
      meanL0[kk] += wy0 * lv;
    }
  }
  for (let kk = 0; kk < K; kk++) {
    meanT1[kk] /= sumY1;
    meanT0[kk] /= sumY0;
    meanL1[kk] /= sumY1;
    meanL0[kk] /= sumY0;
  }

  let betweenAnswerVariance = 0;
  for (let kk = 0; kk < K; kk++) {
    const dt = meanT1[kk] - meanT0[kk];
    const dl = meanL1[kk] - meanL0[kk];
    betweenAnswerVariance += dt * dt + dl * dl;
  }
  const predictedYes = pYes / (sumY1 + sumY0);
  return workspace.baselineTotalVariance
    - predictedYes * (1 - predictedYes) * betweenAnswerVariance;
}

function expectedLossEvaluator(
  st: ParticleState, k: number,
): (s: number, sSd: number) => number {
  const workspace = makeLossWorkspace(st);
  return (s, sSd) => expectedLossWithWorkspace(st, k, s, sSd, workspace);
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

// Posterior-predictive probability for the binary response used by the engine
// (y=1 iff the participant selects the asked task). Session speculation uses
// this only to decide which immutable branch to compute first; it never enters
// stopping, selection, or the adopted posterior calculation.
export function predictedYesProbability(
  st: ParticleState,
  k: number,
  s: number,
  sSd: number,
): number {
  let weighted = 0;
  let totalWeight = 0;
  for (let n = 0; n < st.N; n++) {
    const off = n * st.K + k;
    const pn = pResponseYes(signalZ(st.l[off], st.t[off], s, sSd));
    weighted += st.w[n] * pn;
    totalWeight += st.w[n];
  }
  return totalWeight > 0 ? weighted / totalWeight : 0.5;
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
  // Representatives are included again in the refinement set. Cache their
  // exact losses so the expensive N×2K expectation is not evaluated twice.
  const lossCache = new Map<number, number>();
  const evaluate = expectedLossEvaluator(st, k);
  const losses = (indices: number[]) => indices.map((i) => {
    let loss = lossCache.get(i);
    if (loss === undefined) {
      loss = evaluate(sigs[i], sds[i]);
      lossCache.set(i, loss);
    }
    return loss;
  });
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
  const lossCache = new Map<number, number>();
  const evaluate = expectedLossEvaluator(st, k);
  const losses = (indices: number[]) => indices.map((i) => {
    let loss = lossCache.get(i);
    if (loss === undefined) {
      loss = evaluate(sigs[i], sds[i]);
      lossCache.set(i, loss);
    }
    return loss;
  });
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
    const evaluate = expectedLossEvaluator(st, k);
    const losses = idx.map((i) => evaluate(sigs[i], sds[i]));
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
