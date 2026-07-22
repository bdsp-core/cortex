import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type {
  EngineInputs, ParticleState, PriorPair, PriorPieces,
} from "./types";
import {
  BIAS_FLAG,
  biasFlagFor,
  CUT_CLASSIFICATION,
  PRECISION_BIAS_FLAG_TAU,
  PRECISION_STATUS,
  PrecisionPolicy,
} from "./precision_policy";

const reference = JSON.parse(readFileSync(fileURLToPath(
  new URL("./__testdata__/precision_reference.json", import.meta.url),
), "utf8"));

const TASKS = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"];

function identity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => Number(i === j)));
}

function inputs(): EngineInputs {
  return {
    taskCodes: TASKS,
    taskLabels: TASKS,
    taskPatternWords: TASKS,
    corrL: identity(7),
    corrT: identity(7),
    nParticles: 1200,
    perDomainCap: 60,
    terminationPolicy: "precision_v1",
    precisionBandEdges: reference.band_edges,
    ellStar: new Array(7).fill(-1),
    segments: [],
  };
}

function flat(rows: number[][]): Float64Array {
  const n = rows.length, kCount = rows[0].length;
  const out = new Float64Array(n * kCount);
  for (let nIdx = 0; nIdx < n; nIdx++) {
    for (let k = 0; k < kCount; k++) out[nIdx * kCount + k] = rows[nIdx][k];
  }
  return out;
}

function state(which: "narrow" | "broad"): ParticleState {
  const cloud = reference.clouds[which];
  const n = cloud.w.length, kCount = 7;
  const pieces: PriorPieces = { K: kCount, sigmaInv: [], logDet: 0, L: [] };
  const prior: PriorPair = { tPieces: pieces, lPieces: pieces };
  return {
    N: n,
    K: kCount,
    t: flat(cloud.t),
    l: flat(cloud.l),
    w: Float64Array.from(cloud.w),
    logPrior: new Float64Array(n),
    logLik: new Float64Array(n),
    history: [],
    prior,
  };
}

function telemetry() {
  return {
    remainingBankCounts: reference.telemetry.remaining_bank_counts,
    bandAdministered: reference.telemetry.band_administered,
    bandRemaining: reference.telemetry.band_remaining,
    bandDeficits: Array.from({ length: 7 }, () => [0, 0, 0]),
  };
}

function expectVectorClose(actual: number[], expected: number[], tolerance = 1e-12): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThan(tolerance);
  }
}

describe("PrecisionPolicy fixed-cloud Python parity", () => {
  it("matches intervals, point-centred radii, guards, statuses, and reversion", () => {
    const policy = PrecisionPolicy.fromInputs(inputs());
    for (const [index, cloud] of (["narrow", "narrow", "broad"] as const).entries()) {
      const result = policy.evaluate(state(cloud), reference.n_per_task, telemetry());
      const expected = reference.expected[index];
      const actual = result.diagnostics! as any;
      const wanted = expected.diagnostics;

      expect(result.stop).toBe(expected.stop);
      expect(result.stopReason).toBe(expected.stop_reason);
      expect(result.domainStatuses).toEqual(expected.domain_statuses);
      expect(result.streakCounts).toEqual(expected.streak_counts);
      expect(actual.precisionStatistic).toBe("point_centered_radius");
      expect(actual.reliabilityMode).toBe("quantile_mcse");
      expect(actual.skillIntervals).toHaveLength(7);
      for (let k = 0; k < 7; k++) {
        expectVectorClose(actual.skillIntervals[k], wanted.skill_intervals[k]);
        expectVectorClose(actual.biasIntervals[k], wanted.bias_intervals[k]);
      }
      expectVectorClose(actual.skillPointCenteredRadius, wanted.skill_point_centered_radius);
      expectVectorClose(
        actual.skillPointCenteredRadiusMcse,
        wanted.skill_point_centered_radius_mcse,
      );
      expectVectorClose(actual.guardedPrecisionStatistic, wanted.guarded_precision_statistic);
      expect(actual.guardPass).toEqual(wanted.guard_pass);
      expect(actual.precisionNow).toEqual(wanted.precision_now);
    }
    // Precision completion is fresh-derived: the broad third cloud reopens
    // every formerly complete domain rather than preserving an AD6-style lock.
    expect(policy.domainStatuses).toEqual(new Array(7).fill(PRECISION_STATUS.ACTIVE));
  });
});

describe("PrecisionPolicy frozen boundary semantics", () => {
  it("applies certification cuts only to estimate-complete domains", () => {
    const policy = PrecisionPolicy.fromInputs(inputs());
    policy.evaluate(state("narrow"), reference.n_per_task, telemetry());
    policy.evaluate(state("narrow"), reference.n_per_task, telemetry());
    const determined = policy.finalizeResult(new Array(7).fill(-1));
    expect(determined.determinations).toEqual(new Array(7).fill(PRECISION_STATUS.DETERMINED));
    expect(determined.verdicts).toEqual(new Array(7).fill(CUT_CLASSIFICATION.ABOVE_CUT));

    policy.evaluate(state("broad"), reference.n_per_task, telemetry());
    const reopened = policy.finalizeResult(new Array(7).fill(999));
    expect(reopened.verdicts).toEqual(new Array(7).fill(PRECISION_STATUS.ACTIVE));
  });

  it("evaluates the 60th response before CAP terminalization", () => {
    const completes = PrecisionPolicy.fromInputs(inputs());
    completes.evaluate(state("narrow"), new Array(7).fill(59), telemetry());
    completes.evaluate(state("narrow"), new Array(7).fill(60), telemetry());
    expect(completes.domainStatuses).toEqual(
      new Array(7).fill(PRECISION_STATUS.ESTIMATE_COMPLETE),
    );

    const caps = PrecisionPolicy.fromInputs(inputs());
    caps.evaluate(state("broad"), new Array(7).fill(59), telemetry());
    caps.evaluate(state("broad"), new Array(7).fill(60), telemetry());
    expect(caps.domainStatuses).toEqual(
      new Array(7).fill(PRECISION_STATUS.UNDETERMINABLE_CAP),
    );
  });

  it("can terminalize an infeasible bank before the first question", () => {
    const policy = PrecisionPolicy.fromInputs(inputs());
    const decision = policy.observeBankFeasibility({
      remainingBankCounts: new Array(7).fill(8),
      bandAdministered: Array.from({ length: 7 }, () => [0, 0, 0]),
      bandRemaining: Array.from({ length: 7 }, () => [8, 0, 0]),
      bandDeficits: Array.from({ length: 7 }, () => [3, 3, 3]),
    }, new Array(7).fill(0));
    expect(decision.stop).toBe(true);
    expect(decision.domainStatuses).toEqual(
      new Array(7).fill(PRECISION_STATUS.UNDETERMINABLE_BANK),
    );
  });
});

describe("report-only bias flags", () => {
  it("flags only when the whole 95% interval clears ±tau", () => {
    const tau = PRECISION_BIAS_FLAG_TAU;
    expect(biasFlagFor([tau + 0.1, tau + 1])).toBe(BIAS_FLAG.EXTREME_OVERCALLER);
    expect(biasFlagFor([-tau - 1, -tau - 0.1])).toBe(BIAS_FLAG.EXTREME_UNDERCALLER);
    // The no-data prior interval (±1.96 in prior-SD units) must never flag.
    expect(biasFlagFor([-1.96, 1.96])).toBeNull();
    // Confidently non-zero but not extreme: interval straddles tau.
    expect(biasFlagFor([tau - 0.5, tau + 2])).toBeNull();
    expect(biasFlagFor([-tau - 2, -tau + 0.5])).toBeNull();
  });

  it("emits per-domain flags aligned with the reported bias intervals", () => {
    const policy = PrecisionPolicy.fromInputs(inputs());
    policy.evaluate(state("narrow"), reference.n_per_task, telemetry());
    const final = policy.finalizeResult(new Array(7).fill(-1));
    expect(final.biasFlags).toHaveLength(7);
    final.biasFlags!.forEach((flag, k) => {
      expect(flag).toBe(biasFlagFor(final.biasIntervals![k]));
    });
  });

  it("never alters stopping statuses or cut verdicts", () => {
    const flagged = PrecisionPolicy.fromInputs(inputs());
    flagged.evaluate(state("narrow"), reference.n_per_task, telemetry());
    flagged.evaluate(state("narrow"), reference.n_per_task, telemetry());
    const final = flagged.finalizeResult(new Array(7).fill(-1));
    // Identical to the frozen-boundary expectations regardless of flag values.
    expect(final.determinations).toEqual(new Array(7).fill(PRECISION_STATUS.DETERMINED));
    expect(final.verdicts).toEqual(new Array(7).fill(CUT_CLASSIFICATION.ABOVE_CUT));
  });
});
