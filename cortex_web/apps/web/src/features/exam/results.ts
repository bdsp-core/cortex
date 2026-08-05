// Post-sitting result derivation: the statistics App.tsx used to compute
// inline between its worker callbacks and its network calls.
//
// These are pure functions of the finished session, so they belong beside the
// other exam logic rather than inside a component's onDone handler — and,
// unlike code living in a callback, they can be tested directly.
//
// Reporting only. Every number here is read out of the engine's finished
// state (posterior means, the final particle cloud, per-task AUROC); nothing
// re-derives or adjusts a verdict.

import type {
  ComputeEngineInputs, EngineInputs, TrialDiag,
} from "../../../engine";
import { empiricalPoint, onCurvePoint } from "../../roc";
import type { PercentileDomainScore } from "../../percentile/types";

export interface RocDatum {
  auroc: number;
  hw: number;
  opFar: number | null;
  opHr: number | null;
}

// The trajectory cloud the engine returns: l is [T,N,K] row-major, w is [T,N].
export interface TrajectoryCloud {
  shape: [number, number, number];
  l: Float32Array | Float64Array | number[];
  w: Float32Array | Float64Array | number[];
}

/**
 * Per-task posterior SD of ℓ from the FINAL particle-cloud step — the ± band
 * on the ℓ evolution chart. Weighted std over the N particles at t = T-1.
 * A task whose weights sum to zero has no usable spread and yields null
 * rather than a fabricated 0.
 */
export function cloudSdPerTask(traj: TrajectoryCloud,
                               nTasks: number): (number | null)[] {
  const [tCount, particles, tasks] = traj.shape as [number, number, number];
  const { l, w } = traj;
  const last = tCount - 1;
  return Array.from({ length: nTasks }, (_unused, k) => {
    let wsum = 0, mean = 0;
    for (let i = 0; i < particles; i++) {
      const weight = w[last * particles + i];
      wsum += weight;
      mean += weight * l[(last * particles + i) * tasks + k];
    }
    if (wsum <= 0) return null;
    mean /= wsum;
    let varAcc = 0;
    for (let i = 0; i < particles; i++) {
      const weight = w[last * particles + i];
      const dl = l[(last * particles + i) * tasks + k] - mean;
      varAcc += weight * dl * dl;
    }
    return Math.sqrt(varAcc / wsum);
  });
}

/**
 * Per-task ROC: posterior-mean AUROC plus the examinee's empirical operating
 * point projected onto the binormal curve. Spike truth = sign(s_mean); IIIC
 * truth = the segment's pattern class.
 */
export function perTaskRoc(
  finalAuroc: number[] | undefined,
  finalAurocHw: number[] | undefined,
  trials: TrialDiag[],
  inputs: Pick<EngineInputs, "taskPatternWords" | "taskClasses">
         | Pick<ComputeEngineInputs, "taskPatternWords" | "taskClasses">,
  patternClassOf: (segId: number) => string | undefined,
): RocDatum[] {
  const words = inputs.taskPatternWords;
  return (finalAuroc ?? []).map((auroc, k) => {
    const spike = inputs.taskClasses?.[k] === "spike";
    const empirical = empiricalPoint(trials, k, words[k], patternClassOf, spike);
    const onCurve = empirical ? onCurvePoint(auroc, empirical[0]) : null;
    return {
      auroc,
      hw: finalAurocHw?.[k] ?? 0,
      opFar: onCurve?.[0] ?? null,
      opHr: onCurve?.[1] ?? null,
    };
  });
}

export interface PerTaskRecord {
  taskK: number;
  code: string;
  label: string;
  ell: number | null;
  theta: number | null;
  sd: number | null;
  ellStar: number | null;
  auroc: number | null;
  aurocHw: number | null;
  verdict: string;
  determination: unknown;
  skillInterval: unknown;
  biasInterval: unknown;
  biasFlag: unknown;
  biasFlagWithheldReason: unknown;
  percentile: PercentileDomainScore | null;
}

/**
 * The per-task certification record persisted with a result, so the dashboard
 * reads genuine numbers rather than sample data. ℓ/θ are the final posterior
 * means from the last engine diagnostic, σ is the cloud SD, ℓ* is the bundle's
 * cut-score, AUROC is the per-task posterior mean. The backend turns these
 * into the first "eval" trajectory point per domain.
 */
export function buildPerTaskRecords(args: {
  inputs: Pick<EngineInputs, "taskCodes" | "taskLabels" | "ellStar">
        | Pick<ComputeEngineInputs, "taskCodes" | "taskLabels" | "ellStar">;
  lastDiag: { lMean?: number[]; tMean?: number[] } | null;
  sdPerTask: (number | null)[];
  roc: RocDatum[];
  verdicts?: string[];
  determinations?: unknown[];
  skillIntervals?: unknown[];
  biasIntervals?: unknown[];
  biasFlags?: unknown[];
  biasFlagWithheldReasons?: unknown[];
  percentiles?: Record<string, PercentileDomainScore>;
}): PerTaskRecord[] {
  const { inputs, lastDiag: d, sdPerTask, roc } = args;
  return (inputs.taskCodes ?? []).map((code, k) => ({
    taskK: k,
    code,
    label: inputs.taskLabels?.[k] ?? code,
    ell: d?.lMean?.[k] ?? null,
    theta: d?.tMean?.[k] ?? null,
    sd: sdPerTask[k] ?? null,
    ellStar: inputs.ellStar?.[k] ?? null,
    auroc: roc[k]?.auroc ?? null,
    aurocHw: roc[k]?.hw ?? null,
    verdict: args.verdicts?.[k] ?? "PENDING",
    determination: args.determinations?.[k] ?? null,
    skillInterval: args.skillIntervals?.[k] ?? null,
    biasInterval: args.biasIntervals?.[k] ?? null,
    biasFlag: args.biasFlags?.[k] ?? null,
    biasFlagWithheldReason: args.biasFlagWithheldReasons?.[k] ?? null,
    percentile: args.percentiles?.[code] ?? null,
  }));
}
