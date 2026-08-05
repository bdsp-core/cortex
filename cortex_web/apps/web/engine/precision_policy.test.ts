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
  gradedBiasFlagFor,
  PRECISION_BIAS_FLAG_P_CONFIRMED,
  PRECISION_BIAS_FLAG_P_STAR,
  PRECISION_BIAS_FLAG_P_WATCH,
  PRECISION_BIAS_FLAG_TAU,
  PRECISION_STATUS,
  PrecisionPolicy,
} from "./precision_policy";
import { NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT } from "./nway_profile";

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
      // Graded-flag reference (additive fixture keys): tail masses and the
      // graded/gated rule both reproduce the Python-side computation.
      expectVectorClose(actual.biasTailOver, expected.bias_tail_over);
      expectVectorClose(actual.biasTailUnder, expected.bias_tail_under);
      for (let k = 0; k < 7; k++) {
        const graded = gradedBiasFlagFor(
          actual.biasTailOver[k], actual.biasTailUnder[k],
          actual.ess, actual.essPass, actual.statuses[k],
          actual.evidenceFloorMet[k], actual.contentFloorMet[k],
        );
        expect(graded.flag).toBe(expected.graded_flags[k].flag);
        expect(graded.withheldReason).toBe(expected.graded_flags[k].withheld_reason);
      }
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

  it("keeps the shipped 20/2 floors on an unstamped sitting (drift guard)", () => {
    const policy = PrecisionPolicy.fromInputs(inputs());
    expect(policy.nMin).toBe(20);
    expect(policy.persistence).toBe(2);
    // Below the evidence floor the shipped policy never completes, however
    // many consecutive evaluations pass the precision criterion.
    policy.evaluate(state("narrow"), new Array(7).fill(10), telemetry());
    policy.evaluate(state("narrow"), new Array(7).fill(10), telemetry());
    policy.evaluate(state("narrow"), new Array(7).fill(10), telemetry());
    expect(policy.domainStatuses).toEqual(
      new Array(7).fill(PRECISION_STATUS.ACTIVE),
    );
  });

  it("applies the c1 recalibration only under the server stamp", () => {
    const policy = PrecisionPolicy.fromInputs(
      { ...inputs(), precisionRecalibration: "c1" });
    expect(policy.nMin).toBe(0);
    expect(policy.persistence).toBe(3);
    // Two passing evaluations are no longer enough (persistence 3)...
    policy.evaluate(state("narrow"), new Array(7).fill(10), telemetry());
    policy.evaluate(state("narrow"), new Array(7).fill(10), telemetry());
    expect(policy.domainStatuses).toEqual(
      new Array(7).fill(PRECISION_STATUS.ACTIVE),
    );
    // ...the third completes, below the retired 20-question floor.
    policy.evaluate(state("narrow"), new Array(7).fill(10), telemetry());
    expect(policy.domainStatuses).toEqual(
      new Array(7).fill(PRECISION_STATUS.ESTIMATE_COMPLETE),
    );
    // Speculative-branch clones carry the recalibration with them.
    expect(policy.clone().nMin).toBe(0);
    expect(policy.clone().persistence).toBe(3);
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

describe("graded bias flags", () => {
  // Wide-open gates: qualifying domain, effectively guard-free ESS.
  const fire = (tailOver: number, tailUnder = 0, essValue = 1e12) =>
    gradedBiasFlagFor(tailOver, tailUnder, essValue, true, PRECISION_STATUS.ACTIVE, true, true);
  const tierRank = (flag: string | null): number =>
    flag === null ? 0
      : flag.startsWith("WATCH_") ? 1
        : flag.startsWith("EXTREME_CONFIRMED_") ? 3 : 2;

  it("keeps the confirmed tier equivalent to interval-clears on every fixture step (N5 identity)", () => {
    for (const expected of reference.expected) {
      for (let k = 0; k < 7; k++) {
        const interval = expected.diagnostics.bias_intervals[k];
        expect(expected.bias_tail_over[k] >= PRECISION_BIAS_FLAG_P_CONFIRMED)
          .toBe(interval[0] > PRECISION_BIAS_FLAG_TAU);
        expect(expected.bias_tail_under[k] >= PRECISION_BIAS_FLAG_P_CONFIRMED)
          .toBe(interval[1] < -PRECISION_BIAS_FLAG_TAU);
      }
    }
  });

  it("orders the tiers as nested CONFIRMED ⊂ EXTREME ⊂ WATCH, monotone in tail mass", () => {
    expect(PRECISION_BIAS_FLAG_P_WATCH).toBeLessThan(PRECISION_BIAS_FLAG_P_STAR);
    expect(PRECISION_BIAS_FLAG_P_STAR).toBeLessThan(PRECISION_BIAS_FLAG_P_CONFIRMED);
    expect(fire(0.51).flag).toBe(BIAS_FLAG.WATCH_OVERCALLER);
    expect(fire(0.6).flag).toBe(BIAS_FLAG.EXTREME_OVERCALLER);
    expect(fire(0.99).flag).toBe(BIAS_FLAG.EXTREME_CONFIRMED_OVERCALLER);
    expect(fire(0, 0.51).flag).toBe(BIAS_FLAG.WATCH_UNDERCALLER);
    expect(fire(0, 0.6).flag).toBe(BIAS_FLAG.EXTREME_UNDERCALLER);
    expect(fire(0, 0.99).flag).toBe(BIAS_FLAG.EXTREME_CONFIRMED_UNDERCALLER);
    let last = 0;
    for (const mass of [0, 0.2, 0.4, 0.49, 0.51, 0.56, 0.6, 0.8, 0.95, 0.99, 1.0]) {
      const rank = tierRank(fire(mass).flag);
      expect(rank).toBeGreaterThanOrEqual(last);
      last = rank;
    }
  });

  it("withholds the flag, with a reason, when the domain would not qualify a skill statement", () => {
    for (const status of [PRECISION_STATUS.UNDETERMINABLE_CAP, PRECISION_STATUS.UNDETERMINABLE_BANK]) {
      const graded = gradedBiasFlagFor(1.0, 0, 1e12, true, status, true, true);
      expect(graded).toEqual({ flag: null, withheldReason: "undeterminable_domain" });
    }
    const floors: [boolean, boolean, boolean][] = [
      [false, true, true], [true, false, true], [true, true, false],
    ];
    for (const [evidence, content, essPass] of floors) {
      const graded = gradedBiasFlagFor(
        1.0, 0, 1e12, essPass, PRECISION_STATUS.ACTIVE, evidence, content);
      expect(graded).toEqual({ flag: null, withheldReason: "insufficient_evidence" });
    }
    // A qualifying terminal domain still reports; the historical firing set
    // lands on the confirmed tier.
    expect(gradedBiasFlagFor(1.0, 0, 1e12, true, PRECISION_STATUS.ESTIMATE_COMPLETE, true, true))
      .toEqual({ flag: BIAS_FLAG.EXTREME_CONFIRMED_OVERCALLER, withheldReason: null });
  });

  it("demotes or silences a raw tail mass whose MCSE-guarded value is sub-threshold", () => {
    // Raw 0.6 clears p* = 0.575, but the guard prices the Monte-Carlo error:
    // ESS 25 → guarded ≈ 0.439 (no flag); ESS 200 → ≈ 0.543 (watch only);
    // ESS 1e6 → extreme, matching the unguarded classification.
    expect(fire(0.6, 0, 25)).toEqual({ flag: null, withheldReason: null });
    expect(fire(0.6, 0, 200).flag).toBe(BIAS_FLAG.WATCH_OVERCALLER);
    expect(fire(0.6, 0, 1e6).flag).toBe(BIAS_FLAG.EXTREME_OVERCALLER);
  });

  it("round-trips tails and graded flags through snapshot/clone/restore", () => {
    const policy = PrecisionPolicy.fromInputs(inputs());
    policy.evaluate(state("broad"), reference.n_per_task, telemetry());
    const final = policy.finalizeResult(new Array(7).fill(-1));
    expect(final.biasFlags).toHaveLength(7);
    expect(final.biasFlagWithheldReasons).toHaveLength(7);

    const restored = PrecisionPolicy.fromInputs(inputs());
    restored.restore(policy.snapshot());
    expect(restored.finalizeResult(new Array(7).fill(-1))).toEqual(final);
    expect(policy.clone().finalizeResult(new Array(7).fill(-1))).toEqual(final);
  });

  it("restores legacy snapshots without tail fields to the historical reporting", () => {
    const policy = PrecisionPolicy.fromInputs(inputs());
    policy.evaluate(state("broad"), reference.n_per_task, telemetry());
    const snapshot = policy.snapshot();
    delete (snapshot.lastDiag as Record<string, unknown>).biasTailOver;
    delete (snapshot.lastDiag as Record<string, unknown>).biasTailUnder;

    const restored = PrecisionPolicy.fromInputs(inputs());
    restored.restore(snapshot);
    const final = restored.finalizeResult(new Array(7).fill(-1));
    expect(final.biasFlags).toEqual(final.biasIntervals!.map(biasFlagFor));
    expect(final.biasFlagWithheldReasons).toBeUndefined();
  });

  it("pins the flag calibration to its qualification artifact (drift guard)", () => {
    // Graded-threshold provenance: bias-reporting-policy/g5_operating_point.json
    // (sha256 b5b8ed5f…91bc, 2026-08-05), earned on the served draw-latent
    // response profile below. Sign-conditional shrinkage measured on that
    // campaign (G1_DIAGNOSIS.md DL addendum, Q100): E[t̂−t | t>1] = −0.524
    // (n=133), E[t̂−t | t<−1] = +0.349 (n=101); mixture −0.224/+0.209 —
    // over-callers are structurally the harder call. A response-profile or
    // artifact change invalidates the calibration: re-run the G5 harness and
    // re-pin these values together.
    expect(NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.artifactId)
      .toBe("iiic-f1-engine-frame-nesting34-20260730");
    expect(PRECISION_BIAS_FLAG_P_STAR).toBe(0.575);
    expect(PRECISION_BIAS_FLAG_P_WATCH).toBe(0.5);
    expect(PRECISION_BIAS_FLAG_P_CONFIRMED).toBe(0.975);
    expect(PRECISION_BIAS_FLAG_TAU).toBe(1.0);
  });
});
