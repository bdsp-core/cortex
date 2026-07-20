import type { BankArrays, Chosen } from "./choose_item";
import {
  fillResponseProbabilities, makeResponseProbabilityWorkspace,
} from "./nway_likelihood";
import { posteriorMeans } from "./particles";
import type { ComputeEngineInputs, ComputeSegmentMeta, ParticleState } from "./types";

interface Candidate {
  k: number;
  segment: ComputeSegmentMeta;
}

interface LossWorkspace {
  baselineVariance: number;
  meanT: Float64Array;
  meanL: Float64Array;
}

const FULL_SCAN_LIMIT = 512;
const COARSE_PER_TASK = 32;
const ENTROPY_PER_TASK = 12;
const FISHER_PER_TASK = 8;
const FINITE_DIFFERENCE = 1e-3;

function taskClass(inputs: ComputeEngineInputs, k: number): "iiic" | "spike" {
  const value = inputs.taskClasses?.[k];
  if (value !== "iiic" && value !== "spike") {
    throw new Error(`task ${k} is absent from the n-way response registry`);
  }
  return value;
}

function candidates(
  bank: BankArrays, excludedTasks?: ReadonlySet<number>,
): Candidate[] {
  if (!bank.segment) throw new Error("n-way selector requires full segment signals");
  const result: Candidate[] = [];
  for (let k = 0; k < bank.segId.length; k++) {
    if (excludedTasks?.has(k)) continue;
    for (let index = 0; index < bank.segId[k].length; index++) {
      const segment = bank.segment[k][index];
      if (!segment || segment.segId !== bank.segId[k][index]) {
        throw new Error("n-way candidate metadata is misaligned");
      }
      result.push({ k, segment });
    }
  }
  return result;
}

function workspace(st: ParticleState): LossWorkspace {
  const moments = posteriorMeans(st);
  let baselineVariance = 0;
  for (let k = 0; k < st.K; k++) {
    baselineVariance += moments.tSd[k] ** 2 + moments.lSd[k] ** 2;
  }
  return {
    baselineVariance,
    meanT: Float64Array.from(moments.tMean),
    meanL: Float64Array.from(moments.lMean),
  };
}

export function expectedNWayLoss(
  st: ParticleState, inputs: ComputeEngineInputs,
  candidate: Candidate, cached = workspace(st),
): number {
  const kind = taskClass(inputs, candidate.k);
  const outcomeCount = kind === "spike" ? 2 : 6;
  const probabilities = new Float64Array(outcomeCount);
  const probabilityWorkspace = makeResponseProbabilityWorkspace(st.K);
  const mass = new Float64Array(outcomeCount);
  const conditionalT = Array.from({ length: outcomeCount }, () => new Float64Array(st.K));
  const conditionalL = Array.from({ length: outcomeCount }, () => new Float64Array(st.K));
  let weightSum = 0;
  for (const weight of st.w) weightSum += weight;
  for (let n = 0; n < st.N; n++) {
    const weight = st.w[n] / weightSum;
    fillResponseProbabilities(
      kind, candidate.k, candidate.segment, st.t, st.l, n, st.K,
      probabilities, probabilityWorkspace,
    );
    for (let r = 0; r < outcomeCount; r++) {
      const joint = weight * probabilities[r];
      mass[r] += joint;
      const offset = n * st.K;
      for (let k = 0; k < st.K; k++) {
        conditionalT[r][k] += joint * st.t[offset + k];
        conditionalL[r][k] += joint * st.l[offset + k];
      }
    }
  }
  let between = 0;
  for (let r = 0; r < mass.length; r++) {
    if (mass[r] <= 1e-15) continue;
    for (let k = 0; k < st.K; k++) {
      const dt = conditionalT[r][k] / mass[r] - cached.meanT[k];
      const dl = conditionalL[r][k] / mass[r] - cached.meanL[k];
      between += mass[r] * (dt * dt + dl * dl);
    }
  }
  return cached.baselineVariance - between;
}

function entropy(probabilities: Iterable<number>): number {
  let value = 0;
  for (const probability of probabilities) {
    if (probability > 0) value -= probability * Math.log(probability);
  }
  return value;
}

function evenlySpaced(length: number, count: number): number[] {
  if (length <= count) return Array.from({ length }, (_, index) => index);
  return Array.from({ length: count }, (_, index) =>
    Math.round(index * (length - 1) / (count - 1)));
}

function highestScoring(
  domain: readonly Candidate[], count: number,
  score: (candidate: Candidate) => number,
): Candidate[] {
  const best: { candidate: Candidate; value: number }[] = [];
  for (const candidate of domain) {
    const entry = { candidate, value: score(candidate) };
    let position = best.length;
    while (position > 0) {
      const previous = best[position - 1];
      const better = entry.value > previous.value
        || (entry.value === previous.value
          && entry.candidate.segment.segId < previous.candidate.segment.segId);
      if (!better) break;
      position--;
    }
    if (position < count) {
      best.splice(position, 0, entry);
      if (best.length > count) best.pop();
    }
  }
  return best.map((entry) => entry.candidate);
}

function fisherUtility(
  st: ParticleState, inputs: ComputeEngineInputs, candidate: Candidate,
  moments: ReturnType<typeof posteriorMeans>, workspace: {
    meanT: Float64Array;
    meanL: Float64Array;
    probability: ReturnType<typeof makeResponseProbabilityWorkspace>;
    base2: Float64Array;
    plus2: Float64Array;
    minus2: Float64Array;
    base6: Float64Array;
    plus6: Float64Array;
    minus6: Float64Array;
  },
): number {
  const kind = taskClass(inputs, candidate.k);
  const { meanT, meanL } = workspace;
  const vector = (output: Float64Array) => fillResponseProbabilities(
    kind, candidate.k, candidate.segment, meanT, meanL, 0, st.K,
    output, workspace.probability, true,
  );
  const base = kind === "spike" ? workspace.base2 : workspace.base6;
  const plus = kind === "spike" ? workspace.plus2 : workspace.plus6;
  const minus = kind === "spike" ? workspace.minus2 : workspace.minus6;
  vector(base);
  let utility = 0;
  for (const parameter of ["bias", "skill"] as const) {
    const values = parameter === "bias" ? meanT : meanL;
    const deviations = parameter === "bias" ? moments.tSd : moments.lSd;
    const relevant = kind === "spike" ? [candidate.k] : [1, 2, 3, 4, 5, 6];
    for (const k of relevant) {
      const original = values[k];
      values[k] = original + FINITE_DIFFERENCE;
      vector(plus);
      values[k] = original - FINITE_DIFFERENCE;
      vector(minus);
      values[k] = original;
      let information = 0;
      for (let r = 0; r < base.length; r++) {
        const derivative = (plus[r] - minus[r]) / (2 * FINITE_DIFFERENCE);
        information += derivative * derivative / Math.max(base[r], 1e-12);
      }
      const variance = deviations[k] ** 2;
      utility += variance * variance * information / (1 + variance * information);
    }
  }
  return utility;
}

function shortlist(
  st: ParticleState, inputs: ComputeEngineInputs, all: Candidate[],
): Candidate[] {
  if (all.length <= FULL_SCAN_LIMIT) return all.slice();
  const selected = new Set<Candidate>();
  const moments = posteriorMeans(st);
  const meanT = Float64Array.from(moments.tMean);
  const meanL = Float64Array.from(moments.lMean);
  const probabilityWorkspace = makeResponseProbabilityWorkspace(st.K);
  const probabilities = new Float64Array(6);
  const binaryProbabilities = new Float64Array(2);
  const fisherWorkspace = {
    meanT: Float64Array.from(moments.tMean),
    meanL: Float64Array.from(moments.lMean),
    probability: makeResponseProbabilityWorkspace(st.K),
    base2: new Float64Array(2), plus2: new Float64Array(2), minus2: new Float64Array(2),
    base6: new Float64Array(6), plus6: new Float64Array(6), minus6: new Float64Array(6),
  };
  const domains = Array.from({ length: st.K }, () => [] as Candidate[]);
  for (const candidate of all) domains[candidate.k].push(candidate);
  for (let askedK = 0; askedK < st.K; askedK++) {
    const domain = domains[askedK];
    for (const index of evenlySpaced(domain.length, COARSE_PER_TASK)) {
      selected.add(domain[index]);
    }
    const binWidth = Math.max(1, Math.ceil(domain.length / COARSE_PER_TASK));
    for (let start = 0; start < domain.length; start += binWidth) {
      const end = Math.min(domain.length, start + binWidth);
      let best = domain[start];
      for (let index = start + 1; index < end; index++) {
        if (domain[index].segment.sSd[askedK] < best.segment.sSd[askedK]) {
          best = domain[index];
        }
      }
      if (best) selected.add(best);
    }
    const kind = taskClass(inputs, askedK);
    const byEntropy = highestScoring(domain, ENTROPY_PER_TASK, (candidate) => {
        const output = kind === "spike" ? binaryProbabilities : probabilities;
        fillResponseProbabilities(
          kind, askedK, candidate.segment, meanT, meanL, 0, st.K,
          output, probabilityWorkspace,
        );
        return entropy(output);
      });
    for (const candidate of byEntropy) selected.add(candidate);

    const byFisher = highestScoring(domain, FISHER_PER_TASK, (candidate) =>
      fisherUtility(st, inputs, candidate, moments, fisherWorkspace));
    for (const candidate of byFisher) selected.add(candidate);
  }
  return all.filter((candidate) => selected.has(candidate));
}

function score(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  excludedTasks?: ReadonlySet<number>,
): Chosen[] {
  const cached = workspace(st);
  return shortlist(st, inputs, candidates(bank, excludedTasks)).map((candidate) => ({
    k: candidate.k,
    s: candidate.segment.sMean[candidate.k],
    sSd: candidate.segment.sSd[candidate.k],
    segId: candidate.segment.segId,
    segment: candidate.segment,
    loss: expectedNWayLoss(st, inputs, candidate, cached),
  })).sort((a, b) => a.loss - b.loss);
}

const NO_ITEM: Chosen = { k: 0, s: 0, sSd: 0, segId: -1, loss: Infinity };

export function chooseNWayItem(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  excludedTasks?: ReadonlySet<number>,
): Chosen {
  return score(st, inputs, bank, excludedTasks)[0] ?? NO_ITEM;
}

export function chooseFirstNWayItem(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  topN: number, rng: { int: (n: number) => number },
  excludedTasks?: ReadonlySet<number>,
): Chosen {
  const scored = score(st, inputs, bank, excludedTasks);
  const count = Math.min(topN, scored.length);
  return count ? scored[rng.int(count)] : NO_ITEM;
}

export function predictedOutcomeDistribution(
  st: ParticleState, inputs: ComputeEngineInputs, chosen: Chosen,
): { outcome: number; probability: number }[] {
  if (!chosen.segment) throw new Error("n-way chosen item lacks segment signals");
  const kind = taskClass(inputs, chosen.k);
  const probabilities = new Float64Array(kind === "spike" ? 2 : 6);
  const probabilityWorkspace = makeResponseProbabilityWorkspace(st.K);
  const template = Array.from(probabilities, (_probability, index) => ({
    outcome: kind === "spike" ? (index === 0 ? chosen.k : st.K) : index + 1,
    probability: 0,
  }));
  let weightSum = 0;
  for (let n = 0; n < st.N; n++) {
    weightSum += st.w[n];
    fillResponseProbabilities(
      kind, chosen.k, chosen.segment, st.t, st.l, n, st.K,
      probabilities, probabilityWorkspace,
    );
    for (let r = 0; r < probabilities.length; r++) {
      template[r].probability += st.w[n] * probabilities[r];
    }
  }
  for (const entry of template) entry.probability /= weightSum;
  return template;
}

export function rankOutcomes(
  distribution: readonly { outcome: number; probability: number }[],
): { outcome: number; probability: number; rank: number }[] {
  const total = distribution.reduce((sum, entry) => sum + entry.probability, 0);
  if (!(total > 0) || !Number.isFinite(total)) throw new Error("invalid outcome distribution");
  return distribution.map((entry, index) => ({ ...entry, index }))
    .sort((a, b) => (b.probability - a.probability) || (a.index - b.index))
    .map((entry, rank) => ({
      outcome: entry.outcome, probability: entry.probability / total, rank: rank + 1,
    }));
}
