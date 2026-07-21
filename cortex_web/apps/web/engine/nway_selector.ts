import type { BankArrays, Chosen } from "./choose_item";
import {
  fillResponseProbabilities, fillScreeningProbabilitiesAndJacobians,
  IIIC_TASK_INDICES, makeResponseProbabilityWorkspace, makeScreeningJacobianWorkspace,
} from "./nway_likelihood";
import { posteriorMeans } from "./particles";
import type {
  ComputeEngineInputs, ComputeSegmentMeta, ParticleState, SelectionPhaseTimingV2,
} from "./types";
import type { NWaySelectionExecutor } from "./nway_selector_executor";

export interface NWayCandidate {
  k: number;
  segment: ComputeSegmentMeta;
}

export type NWaySelectionState = Pick<ParticleState, "N" | "K" | "t" | "l" | "w">;

export interface NWayLossWorkspace {
  baselineVariance: number;
  meanT: Float64Array;
  meanL: Float64Array;
  skillScale: Float64Array;
}

export interface NWayScreeningMoments {
  K: number;
  tMean: number[];
  lMean: number[];
  tSd: number[];
  lSd: number[];
}

export interface NWayDomainScreenResult {
  taskK: number;
  entropySegIds: number[];
  fisherSegIds: number[];
  entropyMs: number;
  fisherMs: number;
}

const FULL_SCAN_LIMIT = 512;
const COARSE_PER_TASK = 32;
const ENTROPY_PER_TASK = 12;
const FISHER_PER_TASK = 8;

function taskClass(inputs: ComputeEngineInputs, k: number): "iiic" | "spike" {
  const value = inputs.taskClasses?.[k];
  if (value !== "iiic" && value !== "spike") {
    throw new Error(`task ${k} is absent from the n-way response registry`);
  }
  return value;
}

function candidates(
  bank: BankArrays, excludedTasks?: ReadonlySet<number>,
): NWayCandidate[] {
  if (!bank.segment) throw new Error("n-way selector requires full segment signals");
  const result: NWayCandidate[] = [];
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

export function makeNWayLossWorkspace(st: NWaySelectionState): NWayLossWorkspace {
  const moments = posteriorMeans(st);
  let baselineVariance = 0;
  for (let k = 0; k < st.K; k++) {
    baselineVariance += moments.tSd[k] ** 2 + moments.lSd[k] ** 2;
  }
  const skillScale = new Float64Array(st.l.length);
  for (let index = 0; index < st.l.length; index++) {
    skillScale[index] = Math.exp(st.l[index]);
  }
  return {
    baselineVariance,
    meanT: Float64Array.from(moments.tMean),
    meanL: Float64Array.from(moments.lMean),
    skillScale,
  };
}

export function expectedNWayLoss(
  st: NWaySelectionState, inputs: ComputeEngineInputs,
  candidate: NWayCandidate, cached = makeNWayLossWorkspace(st),
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
      probabilities, probabilityWorkspace, false, cached.skillScale,
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
  domain: readonly NWayCandidate[], count: number,
  score: (candidate: NWayCandidate) => number,
): NWayCandidate[] {
  const best: { candidate: NWayCandidate; value: number }[] = [];
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
  st: Pick<NWaySelectionState, "K">,
  inputs: ComputeEngineInputs, candidate: NWayCandidate,
  moments: ReturnType<typeof posteriorMeans>, workspace: {
    meanT: Float64Array;
    meanL: Float64Array;
    jacobian: ReturnType<typeof makeScreeningJacobianWorkspace>;
    base2: Float64Array;
    base6: Float64Array;
  },
): number {
  const kind = taskClass(inputs, candidate.k);
  const { meanT, meanL } = workspace;
  const base = kind === "spike" ? workspace.base2 : workspace.base6;
  fillScreeningProbabilitiesAndJacobians(
    kind, candidate.k, candidate.segment, meanT, meanL, 0, st.K,
    base, workspace.jacobian,
  );
  let utility = 0;
  for (const parameter of ["bias", "skill"] as const) {
    const deviations = parameter === "bias" ? moments.tSd : moments.lSd;
    const jacobian = parameter === "bias"
      ? workspace.jacobian.biasJacobian : workspace.jacobian.skillJacobian;
    const relevant = kind === "spike" ? [candidate.k] : IIIC_TASK_INDICES;
    for (const k of relevant) {
      let information = 0;
      const row = k * base.length;
      for (let r = 0; r < base.length; r++) {
        const derivative = jacobian[row + r];
        information += derivative * derivative / Math.max(base[r], 1e-12);
      }
      const variance = deviations[k] ** 2;
      utility += variance * variance * information / (1 + variance * information);
    }
  }
  return utility;
}

export function screenNWayDomain(
  inputs: ComputeEngineInputs, taskK: number,
  domain: readonly NWayCandidate[], moments: NWayScreeningMoments,
): NWayDomainScreenResult {
  const kind = taskClass(inputs, taskK);
  const meanT = Float64Array.from(moments.tMean);
  const meanL = Float64Array.from(moments.lMean);
  const probabilities = new Float64Array(kind === "spike" ? 2 : 6);
  const probabilityWorkspace = makeResponseProbabilityWorkspace(moments.K);
  const entropyStartedAt = performance.now();
  const byEntropy = highestScoring(domain, ENTROPY_PER_TASK, (candidate) => {
    fillResponseProbabilities(
      kind, taskK, candidate.segment, meanT, meanL, 0, moments.K,
      probabilities, probabilityWorkspace,
    );
    return entropy(probabilities);
  });
  const entropyMs = performance.now() - entropyStartedAt;
  const fisherWorkspace = {
    meanT: Float64Array.from(moments.tMean),
    meanL: Float64Array.from(moments.lMean),
    jacobian: makeScreeningJacobianWorkspace(moments.K),
    base2: new Float64Array(2),
    base6: new Float64Array(6),
  };
  const fisherStartedAt = performance.now();
  const byFisher = highestScoring(domain, FISHER_PER_TASK, (candidate) =>
    fisherUtility({ K: moments.K }, inputs, candidate, moments, fisherWorkspace));
  return {
    taskK,
    entropySegIds: byEntropy.map((candidate) => candidate.segment.segId),
    fisherSegIds: byFisher.map((candidate) => candidate.segment.segId),
    entropyMs,
    fisherMs: performance.now() - fisherStartedAt,
  };
}

function shortlist(
  st: NWaySelectionState, inputs: ComputeEngineInputs, all: NWayCandidate[],
  timing?: SelectionPhaseTimingV2,
): NWayCandidate[] {
  if (all.length <= FULL_SCAN_LIMIT) {
    if (timing) timing.shortlistCount += all.length;
    return all.slice();
  }
  const selected = new Set<NWayCandidate>();
  const momentsStartedAt = performance.now();
  const moments = posteriorMeans(st);
  const meanT = Float64Array.from(moments.tMean);
  const meanL = Float64Array.from(moments.lMean);
  const probabilityWorkspace = makeResponseProbabilityWorkspace(st.K);
  const probabilities = new Float64Array(6);
  const binaryProbabilities = new Float64Array(2);
  const fisherWorkspace = {
    meanT: Float64Array.from(moments.tMean),
    meanL: Float64Array.from(moments.lMean),
    jacobian: makeScreeningJacobianWorkspace(st.K),
    base2: new Float64Array(2),
    base6: new Float64Array(6),
  };
  if (timing) timing.posteriorMomentsMs += performance.now() - momentsStartedAt;
  const domains = Array.from({ length: st.K }, () => [] as NWayCandidate[]);
  for (const candidate of all) domains[candidate.k].push(candidate);
  for (let askedK = 0; askedK < st.K; askedK++) {
    const domain = domains[askedK];
    const coarseStartedAt = performance.now();
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
    if (timing) timing.coarseMinSdScanMs += performance.now() - coarseStartedAt;
    const kind = taskClass(inputs, askedK);
    const entropyStartedAt = performance.now();
    const byEntropy = highestScoring(domain, ENTROPY_PER_TASK, (candidate) => {
        const output = kind === "spike" ? binaryProbabilities : probabilities;
        fillResponseProbabilities(
          kind, askedK, candidate.segment, meanT, meanL, 0, st.K,
          output, probabilityWorkspace,
        );
        return entropy(output);
      });
    for (const candidate of byEntropy) selected.add(candidate);
    if (timing) timing.entropyScanMs += performance.now() - entropyStartedAt;

    const fisherStartedAt = performance.now();
    const byFisher = highestScoring(domain, FISHER_PER_TASK, (candidate) =>
      fisherUtility(st, inputs, candidate, moments, fisherWorkspace));
    for (const candidate of byFisher) selected.add(candidate);
    if (timing) timing.fisherScanMs += performance.now() - fisherStartedAt;
  }
  const result = all.filter((candidate) => selected.has(candidate));
  if (timing) timing.shortlistCount += result.length;
  return result;
}

function score(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  excludedTasks?: ReadonlySet<number>,
  timing?: SelectionPhaseTimingV2,
): Chosen[] {
  const momentsStartedAt = performance.now();
  const cached = makeNWayLossWorkspace(st);
  if (timing) timing.posteriorMomentsMs += performance.now() - momentsStartedAt;
  const candidatesStartedAt = performance.now();
  const all = candidates(bank, excludedTasks);
  if (timing) {
    timing.candidatePreparationMs += performance.now() - candidatesStartedAt;
    timing.candidateCount += all.length;
  }
  const shortlisted = shortlist(st, inputs, all, timing);
  const refinementStartedAt = performance.now();
  const scored = shortlisted.map((candidate) => ({
    k: candidate.k,
    s: candidate.segment.sMean[candidate.k],
    sSd: candidate.segment.sSd[candidate.k],
    segId: candidate.segment.segId,
    segment: candidate.segment,
    loss: expectedNWayLoss(st, inputs, candidate, cached),
  })).sort((a, b) => a.loss - b.loss);
  if (timing) timing.exactRefinementMs += performance.now() - refinementStartedAt;
  return scored;
}

export function prepareNWayShortlist(
  st: NWaySelectionState, inputs: ComputeEngineInputs, bank: BankArrays,
  excludedTasks?: ReadonlySet<number>, timing?: SelectionPhaseTimingV2,
): NWayCandidate[] {
  const candidatesStartedAt = performance.now();
  const all = candidates(bank, excludedTasks);
  if (timing) {
    timing.candidatePreparationMs += performance.now() - candidatesStartedAt;
    timing.candidateCount += all.length;
  }
  return shortlist(st, inputs, all, timing);
}

export async function prepareNWayShortlistWithExecutor(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  executor: NWaySelectionExecutor,
  excludedTasks?: ReadonlySet<number>, timing?: SelectionPhaseTimingV2,
): Promise<NWayCandidate[]> {
  void inputs; // executor workers own the immutable configured task registry
  const candidatesStartedAt = performance.now();
  const all = candidates(bank, excludedTasks);
  if (timing) {
    timing.candidatePreparationMs += performance.now() - candidatesStartedAt;
    timing.candidateCount += all.length;
  }
  if (all.length <= FULL_SCAN_LIMIT) {
    if (timing) timing.shortlistCount += all.length;
    return all.slice();
  }
  const momentsStartedAt = performance.now();
  const rawMoments = posteriorMeans(st);
  const moments: NWayScreeningMoments = { K: st.K, ...rawMoments };
  if (timing) timing.posteriorMomentsMs += performance.now() - momentsStartedAt;
  const domains = Array.from({ length: st.K }, () => [] as NWayCandidate[]);
  for (const candidate of all) domains[candidate.k].push(candidate);
  const selected = new Set<NWayCandidate>();
  const activeDomains: { taskK: number; segIds: number[] }[] = [];
  for (let taskK = 0; taskK < domains.length; taskK++) {
    const domain = domains[taskK];
    if (domain.length === 0) continue;
    const coarseStartedAt = performance.now();
    for (const index of evenlySpaced(domain.length, COARSE_PER_TASK)) {
      selected.add(domain[index]);
    }
    const binWidth = Math.max(1, Math.ceil(domain.length / COARSE_PER_TASK));
    for (let start = 0; start < domain.length; start += binWidth) {
      const end = Math.min(domain.length, start + binWidth);
      let best = domain[start];
      for (let index = start + 1; index < end; index++) {
        if (domain[index].segment.sSd[taskK] < best.segment.sSd[taskK]) {
          best = domain[index];
        }
      }
      if (best) selected.add(best);
    }
    if (timing) timing.coarseMinSdScanMs += performance.now() - coarseStartedAt;
    activeDomains.push({
      taskK,
      segIds: domain.map((candidate) => candidate.segment.segId),
    });
  }
  const screens = await executor.screen(moments, activeDomains);
  for (const screen of screens) {
    const domain = domains[screen.taskK];
    const byId = new Map(domain.map((candidate) => [candidate.segment.segId, candidate]));
    for (const segId of screen.entropySegIds) {
      const candidate = byId.get(segId);
      if (!candidate) throw new Error("entropy screen returned an ineligible segment");
      selected.add(candidate);
    }
    for (const segId of screen.fisherSegIds) {
      const candidate = byId.get(segId);
      if (!candidate) throw new Error("Fisher screen returned an ineligible segment");
      selected.add(candidate);
    }
  }
  if (timing && screens.length > 0) {
    timing.entropyScanMs += Math.max(...screens.map((screen) => screen.entropyMs));
    timing.fisherScanMs += Math.max(...screens.map((screen) => screen.fisherMs));
  }
  const result = all.filter((candidate) => selected.has(candidate));
  if (timing) timing.shortlistCount += result.length;
  return result;
}

export function scoreNWayCandidateLosses(
  st: NWaySelectionState, inputs: ComputeEngineInputs,
  candidateList: readonly NWayCandidate[],
): Float64Array {
  const cached = makeNWayLossWorkspace(st);
  const losses = new Float64Array(candidateList.length);
  for (let index = 0; index < candidateList.length; index++) {
    losses[index] = expectedNWayLoss(st, inputs, candidateList[index], cached);
  }
  return losses;
}

export function chosenFromNWayLosses(
  candidateList: readonly NWayCandidate[], losses: ArrayLike<number>,
): Chosen[] {
  if (losses.length !== candidateList.length) {
    throw new Error("n-way candidate loss vector is misaligned");
  }
  return candidateList.map((candidate, index) => ({
    k: candidate.k,
    s: candidate.segment.sMean[candidate.k],
    sSd: candidate.segment.sSd[candidate.k],
    segId: candidate.segment.segId,
    segment: candidate.segment,
    loss: losses[index],
  })).sort((a, b) => a.loss - b.loss);
}

const NO_ITEM: Chosen = { k: 0, s: 0, sSd: 0, segId: -1, loss: Infinity };

export function chooseNWayItem(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  excludedTasks?: ReadonlySet<number>,
  timing?: SelectionPhaseTimingV2,
): Chosen {
  return score(st, inputs, bank, excludedTasks, timing)[0] ?? NO_ITEM;
}

export function chooseFirstNWayItem(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  topN: number, rng: { int: (n: number) => number },
  excludedTasks?: ReadonlySet<number>,
  timing?: SelectionPhaseTimingV2,
): Chosen {
  const scored = score(st, inputs, bank, excludedTasks, timing);
  const count = Math.min(topN, scored.length);
  return count ? scored[rng.int(count)] : NO_ITEM;
}

export async function chooseNWayItemWithExecutor(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  executor: NWaySelectionExecutor,
  excludedTasks?: ReadonlySet<number>, timing?: SelectionPhaseTimingV2,
): Promise<Chosen> {
  const shortlisted = await prepareNWayShortlistWithExecutor(
    st, inputs, bank, executor, excludedTasks, timing,
  );
  const refinementStartedAt = performance.now();
  const losses = await executor.score(st, shortlisted);
  const scored = chosenFromNWayLosses(shortlisted, losses);
  if (timing) timing.exactRefinementMs += performance.now() - refinementStartedAt;
  return scored[0] ?? NO_ITEM;
}

export async function chooseFirstNWayItemWithExecutor(
  st: ParticleState, inputs: ComputeEngineInputs, bank: BankArrays,
  topN: number, rng: { int: (n: number) => number },
  executor: NWaySelectionExecutor,
  excludedTasks?: ReadonlySet<number>, timing?: SelectionPhaseTimingV2,
): Promise<Chosen> {
  const shortlisted = await prepareNWayShortlistWithExecutor(
    st, inputs, bank, executor, excludedTasks, timing,
  );
  const refinementStartedAt = performance.now();
  const losses = await executor.score(st, shortlisted);
  const scored = chosenFromNWayLosses(shortlisted, losses);
  if (timing) timing.exactRefinementMs += performance.now() - refinementStartedAt;
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
