import { describe, expect, it } from "vitest";

import {
  advanceCore, cloneCore, type AdvanceParams, type SessionCore,
} from "./advance";
import { restoreCore, snapshotCore, snapshotTransferables } from "./core_snapshot";
import { makeState } from "./particles";
import { precomputePriorPair } from "./prior";
import { PRECISION_STATUS, PrecisionPolicy } from "./precision_policy";
import { Rng } from "./rng";
import type { ComputeEngineInputs } from "./types";

const CODES = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"];

function inputs(): ComputeEngineInputs {
  const K = CODES.length;
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
    segments: Array.from({ length: K * 12 }, (_, id) => {
      const owner = Math.floor(id / 12);
      const applicable = owner === 0 ? [0] : [1, 2, 3, 4, 5, 6];
      const signal = -1.1 + 0.2 * (id % 12);
      const sMean = new Array(K).fill(0);
      const sSd = new Array(K).fill(0);
      for (const k of applicable) { sMean[k] = signal + 0.01 * k; sSd[k] = 0.25; }
      return { segId: id, applicableTaskIdx: applicable, sMean, sSd };
    }),
  };
}

function withoutTiming<T extends { timing: unknown }>(value: T): Omit<T, "timing"> {
  const { timing: _timing, ...deterministic } = value;
  return deterministic;
}

describe("session-core worker snapshots", () => {
  it("round-trips RNG state including the cached Gaussian", () => {
    const rng = new Rng(9182);
    rng.gaussian(); // leaves the paired Gaussian cached
    const restored = Rng.fromSnapshot(rng.snapshot());
    expect(Array.from({ length: 20 }, () => restored.random()))
      .toEqual(Array.from({ length: 20 }, () => rng.random()));
  });

  it("produces the exact same rejuvenated Precision step after restore", () => {
    const config = inputs();
    const prior = precomputePriorPair(config.corrL, config.corrT);
    const rng = new Rng(441);
    const policy = PrecisionPolicy.fromInputs(config);
    policy.reset(CODES.length);
    const core: SessionCore = {
      state: makeState(80, CODES.length, prior, rng),
      rng,
      policy,
      remaining: new Set(config.segments.map((s) => s.segId)),
      nPerTask: new Array(CODES.length).fill(0),
      cappedTasks: new Set(),
      lastOutcomes: new Array(CODES.length).fill(PRECISION_STATUS.ACTIVE),
      lastTaskK: -1,
      streakCount: 0,
    };
    const params: AdvanceParams = {
      K: CODES.length,
      nParticles: 80,
      perDomainCap: 60,
      nMhSteps: 2,
      essThresholdFrac: 1.1, // force the RNG-consuming rejuvenation path
      proposalScale: 2.38 / Math.sqrt(2 * CODES.length),
      firstItemTopN: 10,
      maxConsecutiveSameDomain: 5,
      k7Spike: true,
      spikeIdx: 0,
      nSubsample: 16,
      uncertaintyAwareSubsample: true,
    };
    const first = config.segments[0];
    const chosen = {
      k: 0, s: first.sMean[0], sSd: first.sSd[0], segId: first.segId, loss: 0,
    };

    const snapshot = snapshotCore(core);
    expect(snapshotTransferables(snapshot)).toHaveLength(7);
    const restored = restoreCore(snapshot, config, prior);
    const reference = advanceCore(cloneCore(core), config, chosen, 1, params, 0);
    const actual = advanceCore(restored, config, chosen, 1, params, 0);

    expect(withoutTiming(actual)).toEqual(withoutTiming(reference));
    expect(actual.core.rng.snapshot()).toEqual(reference.core.rng.snapshot());
    expect(actual.core.policy.snapshot()).toEqual(reference.core.policy.snapshot());
  });
});
