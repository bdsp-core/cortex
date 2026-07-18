// Frozen cut-independent PrecisionPolicy (`frontier_p90guard_m3`).
//
// This is a direct TypeScript port of precision-policy/src/precision_policy:
// reported intervals are equal-tailed weighted 95% quantiles, while stopping
// uses the distinct point-centred radius max(mean-low, high-mean) plus its
// quantile MCSE guard. Certification cuts are unavailable to evaluate(); they
// are applied only by finalizeResult() after a domain is DETERMINED.

import type { BankArrays } from "./choose_item";
import type { EngineInputs, ParticleState, RejuvenationTelemetry } from "./types";
import { ess } from "./particles";
import type {
  EngineTerminationPolicy, FinalPolicyResult, PolicyResult,
} from "./policy";

export const PRECISION_STATUS = {
  ACTIVE: "ACTIVE",
  ESTIMATE_COMPLETE: "ESTIMATE_COMPLETE",
  UNDETERMINABLE_CAP: "UNDETERMINABLE_CAP",
  UNDETERMINABLE_BANK: "UNDETERMINABLE_BANK",
  DETERMINED: "DETERMINED",
} as const;

export const CUT_CLASSIFICATION = {
  ABOVE_CUT: "ABOVE_CUT",
  BELOW_CUT: "BELOW_CUT",
  INDETERMINATE_AT_CUT: "INDETERMINATE_AT_CUT",
} as const;

export const PRECISION_TASK_CODES = [
  "spike", "sz", "lpd", "gpd", "lrda", "grda", "iic",
] as const;
export const PRECISION_CONTRACTION_BY_DOMAIN = [
  1.50, 1.30, 1.35, 1.30, 1.30, 1.30, 1.25,
] as const;
export const PRECISION_BAND_MIN = 3;
export const PRECISION_N_MIN = 20;
export const PRECISION_PER_DOMAIN_CAP = 60;
export const PRECISION_PERSISTENCE = 2;
export const PRECISION_ESS_FLOOR_FRACTION = 0.5;
export const PRECISION_RADIUS_MCSE_Z = 1.645;
export const PRECISION_RADIUS_MCSE_INFLATION = 1.5962415320776275;
export const PRECISION_SURROGATE_ACCEPTANCE_FLOOR = 0.20;
export const PRECISION_SURROGATE_ANCESTRY_FLOOR = 0.35;

type Interval = [number, number];

export interface PrecisionBankTelemetry extends Record<string, unknown> {
  remainingBankCounts: number[];
  bandAdministered: number[][];
  bandRemaining: number[][];
  bandDeficits: number[][];
  lastRejuvenation?: RejuvenationTelemetry;
}

export interface PrecisionDiagnostics extends Record<string, unknown> {
  confidence: number;
  skillIntervals: Interval[];
  biasIntervals: Interval[];
  skillPosteriorMean: number[];
  skillIntervalHalfwidth: number[];
  skillPointCenteredRadius: number[];
  skillPointCenteredRadiusMcse: number[];
  skillTolerance: number[];
  contractionFractionByDomain: number[];
  bandMin: number;
  bandEdges: number[][];
  precisionStatistic: "point_centered_radius";
  precisionStatisticValue: number[];
  precisionStatisticMcse: number[];
  precisionStatisticMet: boolean[];
  precisionNow: boolean[];
  precisionPersistent: boolean[];
  streakCounts: number[];
  evidenceFloorMet: boolean[];
  contentFloorMet: boolean[];
  contentDeficits: number[][];
  statuses: string[];
  terminalReasons: (string | null)[];
  nPerTask: number[];
  reliabilityMode: "quantile_mcse";
  ess: number;
  essPass: boolean;
  rejuvenationAcceptance: number | null;
  acceptancePass: boolean;
  distinctAncestorFraction: number | null;
  ancestryPass: boolean;
  precisionStatisticMcseZ: number;
  precisionStatisticMcseInflation: number;
  precisionStatisticMcseMultiplier: number;
  guardPass: boolean[];
  guardedPrecisionStatistic: number[];
}

function assertFiniteVector(values: ArrayLike<number>, name: string): void {
  if (!values.length) throw new Error(`${name} must be non-empty`);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new Error(`${name} contains non-finite values`);
  }
}

// NumPy's default quantile method (`linear`): index=(n-1)q with interpolation.
function linearQuantile(values: number[], q: number): number {
  if (!values.length) throw new Error("cannot take a quantile of an empty vector");
  const ordered = values.slice().sort((a, b) => a - b);
  const h = (ordered.length - 1) * q;
  const lo = Math.floor(h), hi = Math.ceil(h);
  return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]);
}

function deriveBandEdges(inputs: EngineInputs): number[][] {
  const K = inputs.taskCodes.length;
  return Array.from({ length: K }, (_, k) => {
    const values: number[] = [];
    for (const seg of inputs.segments) {
      const applicable = seg.applicableTaskIdx;
      if (applicable && !applicable.includes(k)) continue;
      const value = seg.sMean[k];
      if (Number.isFinite(value)) values.push(value);
    }
    if (values.length < 3) throw new Error(`domain ${k} cannot define signal terciles`);
    const q1 = linearQuantile(values, 1 / 3);
    const q2 = linearQuantile(values, 2 / 3);
    if (!(q1 < q2)) throw new Error(`domain ${k} has degenerate signal terciles`);
    return [q1, q2];
  });
}

// First weighted-CDF crossing with stable value/index ordering, matching
// np.argsort(kind="mergesort") + np.searchsorted(side="left").
export function weightedQuantile(
  values: Float64Array, weights: Float64Array, q: number,
): number {
  return weightedQuantileInOrder(values, weights, q, stableValueOrder(values));
}

function stableValueOrder(values: Float64Array): number[] {
  const order = Array.from({ length: values.length }, (_, i) => i);
  order.sort((a, b) => (values[a] - values[b]) || (a - b));
  return order;
}

function weightedQuantileInOrder(
  values: Float64Array, weights: Float64Array, q: number, order: number[],
): number {
  let cdf = 0;
  for (const i of order) {
    cdf += weights[i];
    if (cdf >= q) return values[i];
  }
  return values[order[order.length - 1]];
}

function quantileDensity(
  values: Float64Array, weights: Float64Array, q: number, essValue: number,
  order: number[],
): number {
  let bandwidth = Math.min(0.02, q / 2, (1 - q) / 2);
  bandwidth = Math.max(bandwidth, Math.min(0.01, 1 / Math.sqrt(Math.max(essValue, 1))));
  const q0 = Math.max(0, q - bandwidth);
  const q1 = Math.min(1, q + bandwidth);
  const x0 = weightedQuantileInOrder(values, weights, q0, order);
  const x1 = weightedQuantileInOrder(values, weights, q1, order);
  const span = x1 - x0;
  return Number.isFinite(span) && span > 0 ? (q1 - q0) / span : Number.NaN;
}

function pointCenteredRadiusMcse(
  values: Float64Array,
  weights: Float64Array,
  qLow: number,
  qHigh: number,
  essValue: number,
  low: number,
  high: number,
  mean: number,
  order: number[],
): number {
  const fLow = quantileDensity(values, weights, qLow, essValue, order);
  const fHigh = quantileDensity(values, weights, qHigh, essValue, order);
  if (!Number.isFinite(fLow) || !Number.isFinite(fHigh) || fLow <= 0 || fHigh <= 0) {
    return Number.NaN;
  }
  const nEff = Math.max(essValue, 1);
  const standardError = (which: "left" | "right"): number => {
    const influence = new Float64Array(values.length);
    let weightedMean = 0;
    for (let i = 0; i < values.length; i++) {
      const ifMean = values[i] - mean;
      const ifLow = (qLow - (values[i] <= low ? 1 : 0)) / fLow;
      const ifHigh = (qHigh - (values[i] <= high ? 1 : 0)) / fHigh;
      const x = which === "left" ? ifMean - ifLow : ifHigh - ifMean;
      influence[i] = x;
      weightedMean += weights[i] * x;
    }
    let variance = 0;
    for (let i = 0; i < influence.length; i++) {
      const d = influence[i] - weightedMean;
      variance += weights[i] * d * d;
    }
    return Math.sqrt(Math.max(variance / nEff, 0));
  };
  return Math.max(standardError("left"), standardError("right"));
}

function classifyIntervalAgainstCut(interval: Interval, cut: number): string {
  if (interval[0] > cut) return CUT_CLASSIFICATION.ABOVE_CUT;
  if (interval[1] < cut) return CUT_CLASSIFICATION.BELOW_CUT;
  return CUT_CLASSIFICATION.INDETERMINATE_AT_CUT;
}

function cloneDiag(d: PrecisionDiagnostics | null): PrecisionDiagnostics | null {
  if (!d) return null;
  return {
    ...d,
    skillIntervals: d.skillIntervals.map((x) => [...x] as Interval),
    biasIntervals: d.biasIntervals.map((x) => [...x] as Interval),
    skillPosteriorMean: d.skillPosteriorMean.slice(),
    skillIntervalHalfwidth: d.skillIntervalHalfwidth.slice(),
    skillPointCenteredRadius: d.skillPointCenteredRadius.slice(),
    skillPointCenteredRadiusMcse: d.skillPointCenteredRadiusMcse.slice(),
    skillTolerance: d.skillTolerance.slice(),
    contractionFractionByDomain: d.contractionFractionByDomain.slice(),
    bandEdges: d.bandEdges.map((x) => x.slice()),
    precisionStatisticValue: d.precisionStatisticValue.slice(),
    precisionStatisticMcse: d.precisionStatisticMcse.slice(),
    precisionStatisticMet: d.precisionStatisticMet.slice(),
    precisionNow: d.precisionNow.slice(),
    precisionPersistent: d.precisionPersistent.slice(),
    streakCounts: d.streakCounts.slice(),
    evidenceFloorMet: d.evidenceFloorMet.slice(),
    contentFloorMet: d.contentFloorMet.slice(),
    contentDeficits: d.contentDeficits.map((x) => x.slice()),
    statuses: d.statuses.slice(),
    terminalReasons: d.terminalReasons.slice(),
    nPerTask: d.nPerTask.slice(),
    guardPass: d.guardPass.slice(),
    guardedPrecisionStatistic: d.guardedPrecisionStatistic.slice(),
  };
}

export class PrecisionPolicy implements EngineTerminationPolicy {
  readonly name = "precision_v1" as const;
  readonly activeLabel = PRECISION_STATUS.ACTIVE;
  readonly nMin = PRECISION_N_MIN;
  readonly perDomainCap = PRECISION_PER_DOMAIN_CAP;
  readonly persistence = PRECISION_PERSISTENCE;
  readonly bandMin = PRECISION_BAND_MIN;
  readonly bandEdges: number[][];
  readonly skillTolerance: number[];

  private statuses: string[];
  private streaks: number[];
  private terminalReasons: (string | null)[];
  private bandAdministered: number[][];
  private lastDiag: PrecisionDiagnostics | null = null;

  private constructor(
    private readonly varPrior: number[],
    bandEdges: number[][],
  ) {
    assertFiniteVector(varPrior, "varPrior");
    if (varPrior.some((x) => x <= 0)) throw new Error("varPrior must be positive");
    this.bandEdges = bandEdges.map((x) => x.slice());
    this.skillTolerance = varPrior.map(
      (v, k) => PRECISION_CONTRACTION_BY_DOMAIN[k] * Math.sqrt(v),
    );
    this.statuses = new Array(varPrior.length).fill(PRECISION_STATUS.ACTIVE);
    this.streaks = new Array(varPrior.length).fill(0);
    this.terminalReasons = new Array(varPrior.length).fill(null);
    this.bandAdministered = Array.from({ length: varPrior.length }, () => [0, 0, 0]);
  }

  static fromInputs(inputs: EngineInputs): PrecisionPolicy {
    if (inputs.taskCodes.join(",") !== PRECISION_TASK_CODES.join(",")) {
      throw new Error(
        `precision_v1 requires task order ${PRECISION_TASK_CODES.join(",")}`,
      );
    }
    if (!inputs.corrT) throw new Error("precision_v1 requires Corr_t");
    const varPrior = inputs.corrL.map((row, k) => row[k]);
    const edges = inputs.precisionBandEdges ?? deriveBandEdges(inputs);
    if (edges.length !== varPrior.length || edges.some(
      (x) => x.length !== 2 || !Number.isFinite(x[0]) || !Number.isFinite(x[1]) || x[0] >= x[1],
    )) throw new Error("precision_v1 requires two increasing band edges per domain");
    return new PrecisionPolicy(varPrior, edges);
  }

  reset(K: number): void {
    if (K !== this.varPrior.length) {
      throw new Error(`K=${K} != configured domains=${this.varPrior.length}`);
    }
    this.statuses.fill(PRECISION_STATUS.ACTIVE);
    this.streaks.fill(0);
    this.terminalReasons.fill(null);
    this.bandAdministered = Array.from({ length: K }, () => [0, 0, 0]);
    this.lastDiag = null;
  }

  clone(): PrecisionPolicy {
    const p = new PrecisionPolicy(this.varPrior, this.bandEdges);
    p.statuses = this.statuses.slice();
    p.streaks = this.streaks.slice();
    p.terminalReasons = this.terminalReasons.slice();
    p.bandAdministered = this.bandAdministered.map((x) => x.slice());
    p.lastDiag = cloneDiag(this.lastDiag);
    return p;
  }

  get domainStatuses(): string[] { return this.statuses.slice(); }

  bandIndex(k: number, signal: number): number {
    const [q1, q2] = this.bandEdges[k];
    return signal < q1 ? 0 : signal < q2 ? 1 : 2;
  }

  recordAdministered(k: number, signal: number): void {
    this.bandAdministered[k][this.bandIndex(k, signal)] += 1;
  }

  private contentDeficits(): number[][] {
    return this.bandAdministered.map(
      (row) => row.map((x) => Math.max(0, this.bandMin - x)),
    );
  }

  bankTelemetry(bank: BankArrays, lastRejuvenation?: RejuvenationTelemetry): PrecisionBankTelemetry {
    const bandRemaining = bank.sMean.map((signals, k) => {
      const counts = [0, 0, 0];
      for (const s of signals) counts[this.bandIndex(k, s)] += 1;
      return counts;
    });
    return {
      remainingBankCounts: bank.sMean.map((x) => x.length),
      bandAdministered: this.bandAdministered.map((x) => x.slice()),
      bandRemaining,
      bandDeficits: this.contentDeficits(),
      ...(lastRejuvenation ? { lastRejuvenation } : {}),
    };
  }

  private applyBankTermination(
    telemetry: PrecisionBankTelemetry,
    nPerTask: number[],
    deficits: number[][],
  ): void {
    for (let k = 0; k < this.statuses.length; k++) {
      const status = this.statuses[k];
      if (status.startsWith("UNDETERMINABLE_") || status !== PRECISION_STATUS.ACTIVE) continue;
      const remaining = telemetry.remainingBankCounts[k];
      let impossible = nPerTask[k] + remaining < this.nMin || remaining <= 0;
      if (!impossible) {
        impossible = telemetry.bandRemaining[k].some(
          (supply, band) => supply < deficits[k][band],
        );
      }
      if (impossible) {
        this.statuses[k] = PRECISION_STATUS.UNDETERMINABLE_BANK;
        this.terminalReasons[k] = PRECISION_STATUS.UNDETERMINABLE_BANK;
      }
    }
  }

  observeBankFeasibility(telemetry: PrecisionBankTelemetry, nPerTask: number[]): PolicyResult {
    this.applyBankTermination(telemetry, nPerTask, telemetry.bandDeficits);
    return this.currentResult(essFromDiag(this.lastDiag));
  }

  // Trial-0 feasibility + per-turn content reservation/floor-progress filter.
  prepareCandidates(bank: BankArrays, nPerTask: number[]): BankArrays {
    const telemetry = this.bankTelemetry(bank);
    this.observeBankFeasibility(telemetry, nPerTask);
    const deficits = telemetry.bandDeficits;
    const supplies = telemetry.bandRemaining;

    const reservedFor = new Map<number, Set<number>>();
    for (let k = 0; k < this.statuses.length; k++) {
      if (this.statuses[k].startsWith("UNDETERMINABLE_")) continue;
      for (let band = 0; band < 3; band++) {
        if (deficits[k][band] <= 0 || supplies[k][band] !== deficits[k][band]) continue;
        for (let i = 0; i < bank.segId[k].length; i++) {
          if (this.bandIndex(k, bank.sMean[k][i]) !== band) continue;
          const id = bank.segId[k][i];
          const owners = reservedFor.get(id) ?? new Set<number>();
          owners.add(k);
          reservedFor.set(id, owners);
        }
      }
    }

    const floorOnly = new Set<number>();
    if (this.lastDiag) {
      for (let k = 0; k < this.statuses.length; k++) {
        if (this.statuses[k] === PRECISION_STATUS.ACTIVE
          && this.lastDiag.precisionPersistent[k]
          && this.lastDiag.evidenceFloorMet[k]
          && !this.lastDiag.contentFloorMet[k]) floorOnly.add(k);
      }
    }
    // Frozen floor_progress_deadline=true.
    for (let k = 0; k < this.statuses.length; k++) {
      if (this.statuses[k] !== PRECISION_STATUS.ACTIVE || !deficits[k].some(Boolean)) continue;
      const toEvidence = Math.max(0, this.nMin - nPerTask[k]);
      if (deficits[k].reduce((a, b) => a + b, 0) >= toEvidence) floorOnly.add(k);
    }

    const out: BankArrays = {
      sMean: Array.from({ length: this.statuses.length }, () => []),
      sSd: Array.from({ length: this.statuses.length }, () => []),
      segId: Array.from({ length: this.statuses.length }, () => []),
    };
    for (let k = 0; k < this.statuses.length; k++) {
      for (let i = 0; i < bank.segId[k].length; i++) {
        const id = bank.segId[k][i];
        const owners = reservedFor.get(id);
        if (owners && !owners.has(k)) continue;
        const band = this.bandIndex(k, bank.sMean[k][i]);
        if (floorOnly.has(k) && deficits[k][band] <= 0) continue;
        out.sMean[k].push(bank.sMean[k][i]);
        out.sSd[k].push(bank.sSd[k][i]);
        out.segId[k].push(id);
      }
    }
    return out;
  }

  evaluate(
    st: ParticleState,
    nPerTask: number[],
    rawTelemetry: Record<string, unknown> = {},
  ): PolicyResult {
    const K = st.K;
    const wSum = st.w.reduce((a, b) => a + b, 0);
    if (!Number.isFinite(wSum) || wSum <= 0) throw new Error("invalid particle weights");
    const weights = new Float64Array(st.N);
    for (let i = 0; i < st.N; i++) {
      const wi = st.w[i];
      if (!Number.isFinite(wi) || wi < 0) throw new Error("invalid particle weights");
      weights[i] = wi / wSum;
    }
    const essValue = ess(weights);
    const qLow = 0.025, qHigh = 0.975;
    const intervals: Interval[] = [];
    const biasIntervals: Interval[] = [];
    const means: number[] = [];
    const halfwidths: number[] = [];
    const radii: number[] = [];
    const radiusMcses: number[] = [];

    for (let k = 0; k < K; k++) {
      const skill = new Float64Array(st.N);
      const bias = new Float64Array(st.N);
      let mean = 0;
      for (let n = 0; n < st.N; n++) {
        skill[n] = st.l[n * K + k];
        bias[n] = st.t[n * K + k];
        mean += weights[n] * skill[n];
      }
      // The same ordered skill cloud serves the interval and all four density
      // quantiles used by the MCSE calculation. Re-sorting it for each q was
      // the dominant avoidable cost in the browser's per-answer path.
      const skillOrder = stableValueOrder(skill);
      const biasOrder = stableValueOrder(bias);
      const low = weightedQuantileInOrder(skill, weights, qLow, skillOrder);
      const high = weightedQuantileInOrder(skill, weights, qHigh, skillOrder);
      const radius = Math.max(mean - low, high - mean);
      intervals.push([low, high]);
      biasIntervals.push([
        weightedQuantileInOrder(bias, weights, qLow, biasOrder),
        weightedQuantileInOrder(bias, weights, qHigh, biasOrder),
      ]);
      means.push(mean);
      halfwidths.push((high - low) / 2);
      radii.push(radius);
      radiusMcses.push(
        pointCenteredRadiusMcse(
          skill, weights, qLow, qHigh, essValue, low, high, mean, skillOrder,
        ),
      );
    }

    const telemetry = rawTelemetry as PrecisionBankTelemetry;
    const deficits = telemetry.bandDeficits ?? this.contentDeficits();
    const contentMet = deficits.map((row) => row.every((x) => x === 0));
    const evidenceMet = nPerTask.map((n) => n >= this.nMin);
    const essPass = essValue >= PRECISION_ESS_FLOOR_FRACTION * st.N;
    const event = (telemetry.lastRejuvenation ?? st.lastRejuvenation) as
      RejuvenationTelemetry | undefined;
    const acceptance = event?.acceptanceRate ?? null;
    const ancestry = event?.distinctAncestorFraction ?? null;
    const acceptancePass = !event || acceptance! >= PRECISION_SURROGATE_ACCEPTANCE_FLOOR;
    const ancestryPass = !event || ancestry! >= PRECISION_SURROGATE_ANCESTRY_FLOOR;
    const multiplier = PRECISION_RADIUS_MCSE_Z * PRECISION_RADIUS_MCSE_INFLATION;
    const guarded = radii.map((r, k) => r + multiplier * radiusMcses[k]);
    const guardPass = guarded.map(
      (x, k) => essPass && Number.isFinite(x) && x <= this.skillTolerance[k],
    );
    const statisticMet = radii.map((r, k) => r <= this.skillTolerance[k]);
    const precisionNow = statisticMet.map((x, k) => x && guardPass[k]);

    for (let k = 0; k < K; k++) {
      if (this.statuses[k].startsWith("UNDETERMINABLE_")) continue;
      this.streaks[k] = precisionNow[k] ? this.streaks[k] + 1 : 0;
      const complete = this.streaks[k] >= this.persistence && evidenceMet[k] && contentMet[k];
      this.statuses[k] = complete
        ? PRECISION_STATUS.ESTIMATE_COMPLETE
        : PRECISION_STATUS.ACTIVE;
      // The 60th answer is evaluated before CAP terminalization.
      if (this.statuses[k] === PRECISION_STATUS.ACTIVE && nPerTask[k] >= this.perDomainCap) {
        this.statuses[k] = PRECISION_STATUS.UNDETERMINABLE_CAP;
        this.terminalReasons[k] = PRECISION_STATUS.UNDETERMINABLE_CAP;
      }
    }
    // BANK terminalization runs after post-update CAP terminalization.
    if (telemetry.remainingBankCounts) this.applyBankTermination(telemetry, nPerTask, deficits);

    const persistent = this.streaks.map((x) => x >= this.persistence);
    this.lastDiag = {
      confidence: 0.95,
      skillIntervals: intervals,
      biasIntervals,
      skillPosteriorMean: means,
      skillIntervalHalfwidth: halfwidths,
      skillPointCenteredRadius: radii,
      skillPointCenteredRadiusMcse: radiusMcses,
      skillTolerance: this.skillTolerance.slice(),
      contractionFractionByDomain: Array.from(PRECISION_CONTRACTION_BY_DOMAIN),
      bandMin: this.bandMin,
      bandEdges: this.bandEdges.map((x) => x.slice()),
      precisionStatistic: "point_centered_radius",
      precisionStatisticValue: radii.slice(),
      precisionStatisticMcse: radiusMcses.slice(),
      precisionStatisticMet: statisticMet,
      precisionNow,
      precisionPersistent: persistent,
      streakCounts: this.streaks.slice(),
      evidenceFloorMet: evidenceMet,
      contentFloorMet: contentMet,
      contentDeficits: deficits.map((x) => x.slice()),
      statuses: this.statuses.slice(),
      terminalReasons: this.terminalReasons.slice(),
      nPerTask: nPerTask.slice(),
      reliabilityMode: "quantile_mcse",
      ess: essValue,
      essPass,
      rejuvenationAcceptance: acceptance,
      acceptancePass,
      distinctAncestorFraction: ancestry,
      ancestryPass,
      precisionStatisticMcseZ: PRECISION_RADIUS_MCSE_Z,
      precisionStatisticMcseInflation: PRECISION_RADIUS_MCSE_INFLATION,
      precisionStatisticMcseMultiplier: multiplier,
      guardPass,
      guardedPrecisionStatistic: guarded,
    };
    return this.currentResult(essValue);
  }

  private currentResult(essValue: number): PolicyResult {
    const stop = this.statuses.every((s) => s !== PRECISION_STATUS.ACTIVE);
    return {
      policyName: this.name,
      stop,
      stopReason: stop ? "all_estimated_or_undeterminable" : "continue",
      selectionStates: this.statuses.slice(),
      verdicts: [],
      pi: [], mcse: [], R: [], ess: essValue,
      domainStatuses: this.statuses.slice(),
      determinations: this.statuses.map((s) =>
        s === PRECISION_STATUS.ESTIMATE_COMPLETE ? PRECISION_STATUS.DETERMINED : s),
      terminalReasons: this.terminalReasons.slice(),
      streakCounts: this.streaks.slice(),
      ...(this.lastDiag ? { diagnostics: this.lastDiag } : {}),
    };
  }

  finalizeResult(ellStar: number[]): FinalPolicyResult {
    if (ellStar.length !== this.statuses.length) throw new Error("ellStar/domain mismatch");
    const intervals = this.lastDiag?.skillIntervals;
    const verdicts = this.statuses.map((status, k) => {
      if (status === PRECISION_STATUS.ESTIMATE_COMPLETE) {
        if (!intervals) throw new Error("determined domain has no reported interval");
        // DETERMINED-only downstream reporting boundary.
        return classifyIntervalAgainstCut(intervals[k], ellStar[k]);
      }
      return status;
    });
    return {
      verdicts,
      domainStatuses: this.statuses.slice(),
      determinations: this.statuses.map((s) =>
        s === PRECISION_STATUS.ESTIMATE_COMPLETE ? PRECISION_STATUS.DETERMINED : s),
      terminalReasons: this.terminalReasons.slice(),
      skillIntervals: intervals?.map((x) => [...x] as Interval),
      biasIntervals: this.lastDiag?.biasIntervals.map((x) => [...x] as Interval),
    };
  }
}

function essFromDiag(d: PrecisionDiagnostics | null): number {
  return d?.ess ?? 0;
}
