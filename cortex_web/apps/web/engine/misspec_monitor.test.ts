import { describe, expect, it } from "vitest";

import {
  advanceCore, cloneCore, type AdvanceParams, type SessionCore,
} from "./advance";
import { restoreCore, snapshotCore } from "./core_snapshot";
import {
  DISTRACTOR_MONITOR_THRESHOLD, binaryReduction, makeDistractorMonitor,
  monitorIncrement, observeDistractorMonitor, rebuildBinaryReducedHistory,
} from "./misspec_monitor";
import { NWAY_ARTIFACT, expectedNWayProfile } from "./nway_profile";
import { makeState, updateObservation } from "./particles";
import { precomputePriorPair } from "./prior";
import { PRECISION_STATUS, PrecisionPolicy } from "./precision_policy";
import { Rng } from "./rng";
import type {
  CategoricalParticleObservation, ComputeEngineInputs,
} from "./types";

const CODES = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"];
const K = CODES.length;

function nwayInputs(): ComputeEngineInputs {
  return {
    taskCodes: CODES,
    taskLabels: CODES,
    taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
    taskClasses: ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
    corrL: Array.from({ length: K }, (_, i) =>
      Array.from({ length: K }, (_, j) => i === j ? 1 : 0.08)),
    corrT: Array.from({ length: K }, (_, i) =>
      Array.from({ length: K }, (_, j) => i === j ? 1 : -0.04)),
    ellStar: new Array(K).fill(0.5),
    nParticles: 1200,
    perDomainCap: 60,
    terminationPolicy: "precision_v1",
    precisionBandEdges: Array.from({ length: K }, () => [-0.4, 0.4]),
    nwayProfile: expectedNWayProfile("a".repeat(64)),
    segments: Array.from({ length: 72 }, (_, id) => {
      const sMean = new Array(K).fill(0);
      const sSd = new Array(K).fill(0.1);
      for (let k = 1; k < K; k++) sMean[k] = ((id + k) % 5 - 2) * 0.9;
      return { segId: id, applicableTaskIdx: [1, 2, 3, 4, 5, 6], sMean, sSd };
    }),
  };
}

function makeCore(config: ComputeEngineInputs, seed = 771): SessionCore {
  const prior = precomputePriorPair(config.corrL, config.corrT);
  const rng = new Rng(seed);
  const policy = PrecisionPolicy.fromInputs(config);
  policy.reset(K);
  return {
    state: makeState(160, K, prior, rng),
    rng,
    policy,
    remaining: new Set(config.segments.map((s) => s.segId)),
    nPerTask: new Array(K).fill(0),
    cappedTasks: new Set(),
    lastOutcomes: new Array(K).fill(PRECISION_STATUS.ACTIVE),
    lastTaskK: -1,
    streakCount: 0,
  };
}

const PARAMS: AdvanceParams = {
  K,
  nParticles: 160,
  perDomainCap: 60,
  nMhSteps: 2,
  essThresholdFrac: 0.5,
  proposalScale: 2.38 / Math.sqrt(2 * K),
  firstItemTopN: 10,
  maxConsecutiveSameDomain: 5,
  k7Spike: true,
  spikeIdx: 0,
  nSubsample: 16,
  uncertaintyAwareSubsample: true,
};

/** The distractor the deployed mixture considers least likely for a segment. */
function worstPick(askedK: number, sMean: number[]): number {
  let pick = -1;
  let lowest = Infinity;
  for (let k = 1; k < K; k++) {
    if (k === askedK) continue;
    const evidence = monitorIncrement(askedK, k, sMean);
    if (evidence < lowest) { lowest = evidence; pick = k; }
  }
  return pick;
}

describe("distractor-misspecification monitor", () => {
  it("matches the Python reference increments bit-for-bit", () => {
    // Generated from nway_protocol.misspec_monitor.increment with the
    // deployed floor-0.15 ensemble draws.
    const cases: [number, number, number[], number][] = [
      [3, 5, [-0.7, -0.3, 0.15, 0.55, -0.1, 0.8, -0.45], 0.60959874633024658],
      [1, 2, [0.0, 2.1, -0.6, 0.3, -1.4, 0.9, 0.2], -0.61166415928844153],
      [6, 1, [0.0, -0.2, 1.3, -0.8, 0.4, -1.1, 0.6], -0.42336655658396727],
    ];
    for (const [askedK, pickK, sMean, expected] of cases) {
      expect(Math.abs(monitorIncrement(askedK, pickK, sMean) - expected))
        .toBeLessThanOrEqual(1e-12);
    }
  });

  it("consumes wrong picks only and never falls below the lapse-floor bound", () => {
    expect(() => monitorIncrement(2, 2, new Array(K).fill(0))).toThrow(/wrong picks/);
    const spread = [0, 2.4, -2.1, 1.8, -1.5, 2.0, -2.2];
    let worst = Infinity;
    for (let asked = 1; asked < K; asked++) {
      for (let pick = 1; pick < K; pick++) {
        if (pick === asked) continue;
        worst = Math.min(worst, monitorIncrement(asked, pick, spread));
      }
    }
    expect(worst).toBeGreaterThanOrEqual(Math.log(NWAY_ARTIFACT.robustnessFloor));
  });

  it("runs a one-way CUSUM that never banks positive credit", () => {
    const monitor = makeDistractorMonitor();
    observeDistractorMonitor(monitor, 3.0);
    expect(monitor.statistic).toBe(0);
    observeDistractorMonitor(monitor, -DISTRACTOR_MONITOR_THRESHOLD);
    expect(monitor.tripped).toBe(false);
    observeDistractorMonitor(monitor, -0.1);
    expect(monitor.tripped).toBe(true);
    expect(monitor.trippedAt).toBe(3);
    observeDistractorMonitor(monitor, 99);
    expect(monitor.trippedAt).toBe(3);
    expect(monitor.wrongPicks).toBe(3);
  });

  it("reduces a categorical observation to its exact binary margin", () => {
    const observation: CategoricalParticleObservation = {
      kind: "categorical_f1", askedK: 2, pickK: 5,
      sMean: [0, 0.1, 0.7, -0.2, 0.3, -0.4, 0.5],
      sSd: [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
    };
    expect(binaryReduction(observation)).toEqual({
      kind: "binary", k: 2, s: 0.7, sSd: 0.12, y: 0, rawPick: 5,
    });
    expect(binaryReduction({ ...observation, pickK: 2 }).y).toBe(1);
  });

  it("rebuilds history binary-reduced with exact importance reweighting", () => {
    const config = nwayInputs();
    const prior = precomputePriorPair(config.corrL, config.corrT);
    const state = makeState(64, K, prior, new Rng(15));
    const segment = config.segments[3];
    updateObservation(state, {
      kind: "categorical_f1", askedK: 1, pickK: 4,
      sMean: segment.sMean.slice(), sSd: segment.sSd.slice(),
    });
    updateObservation(state, {
      kind: "binary", k: 0, s: 0.4, sSd: 0.1, y: 1, rawPick: 0,
    });
    const before = {
      w: state.w.slice(), logLik: state.logLik.slice(),
      t: state.t.slice(), l: state.l.slice(),
    };
    rebuildBinaryReducedHistory(state);
    expect(state.history.every((entry) => entry.kind === "binary")).toBe(true);
    expect(state.history[0]).toMatchObject({ k: 1, y: 0, rawPick: 4 });
    expect(state.t).toEqual(before.t);
    expect(state.l).toEqual(before.l);
    const total = Array.from(state.w).reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 12);
    // w' must be proportional to w x exp(reducedLogLik - previousLogLik).
    const ratios = Array.from(state.w).map((value, n) =>
      value / (before.w[n] * Math.exp(state.logLik[n] - before.logLik[n])));
    for (const ratio of ratios) {
      expect(Math.abs(ratio / ratios[0] - 1)).toBeLessThanOrEqual(1e-12);
    }
  });

  it("trips inside advanceCore, fails closed, and keeps the session running", () => {
    const config = nwayInputs();
    const core = makeCore(config);
    let result;
    let trials = 0;
    while (trials < 40) {
      const segment = config.segments
        .find((candidate) => core.remaining.has(candidate.segId))!;
      const askedK = 1 + (trials % 6);
      const chosen = {
        k: askedK, s: segment.sMean[askedK], sSd: segment.sSd[askedK],
        segId: segment.segId, loss: 0, segment,
      };
      result = advanceCore(
        core, config, chosen, worstPick(askedK, segment.sMean), PARAMS,
        trials, undefined, true,
      );
      trials += 1;
      if (result.diag.distractorMonitor?.tripped) break;
    }
    expect(result!.diag.distractorMonitor?.tripped).toBe(true);
    expect(core.distractorMonitor?.tripped).toBe(true);
    // Fail-closed: the full history is binary-reduced at the trip.
    expect(core.state.history.every((entry) => entry.kind === "binary")).toBe(true);
    const trippedAtTrial = trials;

    // Later categorical responses are reduced before touching the posterior,
    // and the session keeps producing finite normalized state.
    const segment = config.segments
      .find((candidate) => core.remaining.has(candidate.segId))!;
    const chosen = {
      k: 2, s: segment.sMean[2], sSd: segment.sSd[2],
      segId: segment.segId, loss: 0, segment,
    };
    advanceCore(core, config, chosen, 2, PARAMS, trippedAtTrial, undefined, true);
    const appended = core.state.history[core.state.history.length - 1];
    expect(appended).toMatchObject({ kind: "binary", k: 2, y: 1, rawPick: 2 });
    expect(core.distractorMonitor?.trippedAt).toBeLessThanOrEqual(trippedAtTrial);
    const total = Array.from(core.state.w).reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 12);
    expect(Array.from(core.state.logLik).every(Number.isFinite)).toBe(true);
  });

  it("round-trips the monitor through snapshots and stays deterministic", () => {
    const config = nwayInputs();
    const core = makeCore(config, 992);
    const segment = config.segments[0];
    const chosen = {
      k: 1, s: segment.sMean[1], sSd: segment.sSd[1],
      segId: segment.segId, loss: 0, segment,
    };
    advanceCore(
      core, config, chosen, worstPick(1, segment.sMean), PARAMS, 0, undefined, true,
    );
    expect(core.distractorMonitor).toBeDefined();

    const prior = precomputePriorPair(config.corrL, config.corrT);
    const restored = restoreCore(snapshotCore(core), config, prior);
    expect(restored.distractorMonitor).toEqual(core.distractorMonitor);

    const next = config.segments[1];
    const nextChosen = {
      k: 2, s: next.sMean[2], sSd: next.sSd[2],
      segId: next.segId, loss: 0, segment: next,
    };
    const reference = advanceCore(
      cloneCore(core), config, nextChosen, worstPick(2, next.sMean),
      PARAMS, 1, undefined, true,
    );
    const actual = advanceCore(
      restored, config, nextChosen, worstPick(2, next.sMean),
      PARAMS, 1, undefined, true,
    );
    expect(actual.diag.distractorMonitor).toEqual(reference.diag.distractorMonitor);
    expect(Array.from(actual.core.state.w)).toEqual(Array.from(reference.core.state.w));
  });
});
