import { describe, expect, it } from "vitest";

import type { BankArrays } from "./choose_item";
import { makeResponseObservation, responseProbabilities } from "./nway_likelihood";
import { NWAY_ARTIFACT, expectedNWayProfile, validateNWayInputs } from "./nway_profile";
import { chooseNWayItem, expectedNWayLoss, rankOutcomes } from "./nway_selector";
import {
  cloneState, logLikPackedHistory, resampleAndRejuvenate,
  resampleAndRejuvenateWithExecutor, updateObservation,
} from "./particles";
import type { NWaySelectionExecutor } from "./nway_selector_executor";
import { Rng } from "./rng";
import { WebCortexSession } from "./session";
import type {
  ComputeEngineInputs, ComputeSegmentMeta, ParticleState, PriorPieces,
} from "./types";

const K = 7;

function state(): ParticleState {
  const pieces: PriorPieces = {
    K, sigmaInv: Array.from({ length: K }, (_, i) =>
      Array.from({ length: K }, (_x, j) => i === j ? 1 : 0)),
    logDet: 0,
    L: Array.from({ length: K }, (_, i) =>
      Array.from({ length: K }, (_x, j) => i === j ? 1 : 0)),
  };
  return {
    N: 4, K,
    t: Float64Array.from([
      0, -0.8, 0.9, -0.3, 0.2, -0.1, 0.4,
      0, 0.7, -0.6, 0.5, -0.2, 0.1, -0.4,
      0, -0.2, 0.1, 0.8, -0.7, 0.3, -0.1,
      0, 0.4, -0.3, -0.5, 0.6, -0.8, 0.7,
    ]),
    l: Float64Array.from([
      0, 0.4, -0.2, 0.1, -0.1, 0.2, -0.3,
      0, -0.3, 0.5, -0.2, 0.3, -0.1, 0.2,
      0, 0.1, -0.4, 0.6, -0.2, 0.4, -0.1,
      0, -0.2, 0.3, -0.1, 0.5, -0.3, 0.4,
    ]),
    w: Float64Array.from([0.1, 0.2, 0.3, 0.4]),
    logPrior: new Float64Array(4),
    logLik: new Float64Array(4),
    history: [], prior: { tPieces: pieces, lPieces: pieces },
  };
}

function segment(segId = 10, shift = 0): ComputeSegmentMeta {
  return {
    segId, applicableTaskIdx: [1, 2, 3, 4, 5, 6],
    sMean: [0, -0.7 + shift, 0.9 + shift, -0.2, 0.4, -0.5, 0.1],
    sSd: [0.1, 0.12, 0.18, 0.15, 0.11, 0.17, 0.13],
  };
}

function inputs(segments = [segment()]): ComputeEngineInputs {
  const identity = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_x, j) => i === j ? 1 : 0));
  return {
    taskCodes: ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
    taskLabels: ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
    taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
    taskClasses: ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
    corrL: identity, corrT: identity, ellStar: new Array(K).fill(0),
    nParticles: 1200, perDomainCap: 60, terminationPolicy: "precision_v1",
    precisionBandEdges: Array.from({ length: K }, () => [-0.5, 0.5]),
    nwayProfile: expectedNWayProfile("a".repeat(64)), segments,
  };
}

describe("production native n-way integration", () => {
  it("freezes and validates the exact integrated artifact/profile stamp", () => {
    expect(NWAY_ARTIFACT.draws).toHaveLength(9);
    expect(NWAY_ARTIFACT.sha256).toBe(
      "0654fc210e67ece152dcaef6b40641fc9c94cb259d100ed3829d90224bc5ef8b",
    );
    expect(validateNWayInputs(inputs()).selectorVersion)
      .toBe("categorical_fisher_totalvar_v1");
    const altered = inputs();
    altered.nwayProfile = { ...altered.nwayProfile!, responseArtifactSha256: "b".repeat(64) };
    expect(() => validateNWayInputs(altered)).toThrow(/responseArtifactSha256/);
  });

  it("produces a normalized six-outcome IIIC distribution per particle", () => {
    const st = state();
    for (let n = 0; n < st.N; n++) {
      const probabilities = responseProbabilities(
        "iiic", 1, segment(), st.t, st.l, n, K,
      );
      expect(probabilities.map((entry) => entry.outcome)).toEqual([1, 2, 3, 4, 5, 6]);
      expect(probabilities.reduce((sum, entry) => sum + entry.probability, 0))
        .toBeCloseTo(1, 13);
      expect(probabilities.every((entry) => entry.probability > 0)).toBe(true);
    }
  });

  it("matches the independent Python nine-draw reference", () => {
    const referenceSegment: ComputeSegmentMeta = {
      segId: 9001, applicableTaskIdx: [1, 2, 3, 4, 5, 6],
      sMean: [-0.7, -0.3, 0.15, 0.55, -0.1, 0.8, -0.45],
      sSd: [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    };
    const st = state();
    st.N = 3;
    st.t = Float64Array.from([
      0.1, -0.2, 0.3, -0.1, 0.4, -0.5, 0.2,
      -0.3, 0.4, -0.1, 0.2, -0.2, 0.3, -0.4,
      0.5, -0.4, 0.2, -0.3, 0.1, 0, 0.35,
    ]);
    st.l = Float64Array.from([
      0.2, 0.1, -0.3, 0.4, 0, -0.1, 0.5,
      -0.2, 0.3, 0.2, -0.4, 0.1, 0.4, -0.3,
      0.4, -0.2, 0.1, 0.3, -0.1, 0.2, 0,
    ]);
    st.w = Float64Array.from([0.2, 0.3, 0.5]);
    st.logPrior = new Float64Array(3);
    st.logLik = new Float64Array(3);
    const expected = [
      [0.02842462526964153, 0.06987883358206612, 0.7357329395211821,
        0.06750360211709902, 0.06556667172352094, 0.03289332778649028],
      [0.04173144519701685, 0.03873151599576798, 0.6826694882233317,
        0.02610982947406671, 0.1913818089834033, 0.01937591212641342],
      [0.03164000823759353, 0.0835950632990778, 0.625116461074542,
        0.05644863148961285, 0.15217266168866855, 0.05102717421050528],
    ];
    for (let n = 0; n < st.N; n++) {
      const actual = responseProbabilities(
        "iiic", 3, referenceSegment, st.t, st.l, n, K,
      ).map((entry) => entry.probability);
      actual.forEach((value, index) =>
        expect(Math.abs(value - expected[n][index])).toBeLessThanOrEqual(5e-7));
    }
    expect(Math.abs(expectedNWayLoss(
      st, inputs([referenceSegment]), { k: 3, segment: referenceSegment },
    ) - 0.9024135503263186)).toBeLessThanOrEqual(5e-7);
    updateObservation(st, makeResponseObservation(3, referenceSegment, 5, "iiic"));
    const expectedWeights = [0.08944108851356372, 0.3916028570738769, 0.5189560544125595];
    Array.from(st.w).forEach((value, index) =>
      expect(Math.abs(value - expectedWeights[index])).toBeLessThanOrEqual(5e-7));
  });

  it("retains which wrong IIIC category was selected instead of collapsing to no", () => {
    const a = state();
    const b = cloneState(a);
    updateObservation(a, makeResponseObservation(1, segment(), 2, "iiic"));
    updateObservation(b, makeResponseObservation(1, segment(), 5, "iiic"));
    expect(a.history[0]).toMatchObject({ kind: "categorical_f1", askedK: 1, pickK: 2 });
    expect(b.history[0]).toMatchObject({ kind: "categorical_f1", askedK: 1, pickK: 5 });
    expect(Array.from(a.w)).not.toEqual(Array.from(b.w));
    expect(Array.from(a.w).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 14);
    expect(Array.from(b.w).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 14);
  });

  it("replays categorical history during MH rejuvenation", () => {
    const st = state();
    updateObservation(st, makeResponseObservation(2, segment(), 6, "iiic"));
    st.w = Float64Array.from([0.97, 0.01, 0.01, 0.01]);
    resampleAndRejuvenate(st, new Rng(9901), 2, 0.1, 4);
    expect(st.history).toHaveLength(1);
    expect(st.history[0]).toMatchObject({ kind: "categorical_f1", askedK: 2, pickK: 6 });
    expect(st.packedHistory?.length).toBe(1);
    expect(Array.from(st.packedHistory?.signalMean.slice(0, K) ?? []))
      .toEqual(segment().sMean);
    const cloned = cloneState(st);
    expect(cloned.packedHistory).toEqual(st.packedHistory);
    expect(cloned.packedHistory?.signalMean).not.toBe(st.packedHistory?.signalMean);
    expect(Array.from(st.logLik).every(Number.isFinite)).toBe(true);
    expect(Array.from(st.w).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 14);
    expect(st.lastRejuvenation?.qIndex).toBe(4);
  });

  it("keeps exact particle and RNG state when history likelihood is asynchronous", async () => {
    const serial = state();
    updateObservation(serial, makeResponseObservation(2, segment(), 6, "iiic"));
    serial.w = Float64Array.from([0.97, 0.01, 0.01, 0.01]);
    const parallel = cloneState(serial);
    const serialRng = new Rng(9901);
    const parallelRng = new Rng(9901);
    const executor: NWaySelectionExecutor = {
      workerCount: 2,
      ready: () => Promise.resolve(),
      score: () => Promise.reject(new Error("not used")),
      screen: () => Promise.reject(new Error("not used")),
      historyLikelihood: (history, N, particleK, t, l) => {
        const result = new Float64Array(N);
        logLikPackedHistory(history, N, particleK, t, l, result);
        return Promise.resolve(result);
      },
      dispose: () => undefined,
    };
    resampleAndRejuvenate(serial, serialRng, 2, 0.1, 4);
    await resampleAndRejuvenateWithExecutor(
      parallel, parallelRng, 2, 0.1, executor, 4,
    );
    expect(parallel.t).toEqual(serial.t);
    expect(parallel.l).toEqual(serial.l);
    expect(parallel.w).toEqual(serial.w);
    expect(parallel.logPrior).toEqual(serial.logPrior);
    expect(parallel.logLik).toEqual(serial.logLik);
    expect(parallel.lastRejuvenation).toEqual(serial.lastRejuvenation);
    expect(parallelRng.snapshot()).toEqual(serialRng.snapshot());
  });

  it("fails closed to exact local MH history evaluation after a helper error", async () => {
    const baseline = state();
    updateObservation(baseline, makeResponseObservation(2, segment(), 6, "iiic"));
    baseline.w = Float64Array.from([0.97, 0.01, 0.01, 0.01]);
    const serial = cloneState(baseline);
    const failed = cloneState(baseline);
    const serialRng = new Rng(9901);
    const failedRng = new Rng(9901);
    let disposed = 0;
    const executor: NWaySelectionExecutor = {
      workerCount: 2,
      ready: () => Promise.resolve(),
      score: () => Promise.reject(new Error("not used")),
      screen: () => Promise.reject(new Error("not used")),
      historyLikelihood: () => Promise.reject(new Error("injected helper failure")),
      dispose: () => { disposed += 1; },
    };
    resampleAndRejuvenate(serial, serialRng, 2, 0.1, 4);
    await resampleAndRejuvenateWithExecutor(
      failed, failedRng, 2, 0.1, executor, 4,
    );
    expect(disposed).toBe(1);
    expect(failed.t).toEqual(serial.t);
    expect(failed.l).toEqual(serial.l);
    expect(failed.w).toEqual(serial.w);
    expect(failed.logPrior).toEqual(serial.logPrior);
    expect(failed.logLik).toEqual(serial.logLik);
    expect(failed.lastRejuvenation).toEqual(serial.lastRejuvenation);
    expect(failedRng.snapshot()).toEqual(serialRng.snapshot());
  });

  it("keeps spike as the only binary response group", () => {
    const spike = { ...segment(), applicableTaskIdx: [0] };
    const yes = makeResponseObservation(0, spike, 0, "spike");
    const no = makeResponseObservation(0, spike, K, "spike");
    expect(yes).toMatchObject({ kind: "binary", y: 1, rawPick: 0 });
    expect(no).toMatchObject({ kind: "binary", y: 0, rawPick: K });
    expect(() => makeResponseObservation(1, segment(), 0, "iiic")).toThrow(/six-way/);
  });

  it("uses the exact multi-outcome total-variance objective after shortlisting", () => {
    const segments = [segment(10, -0.3), segment(11, 0), segment(12, 0.4)];
    const bank: BankArrays = {
      sMean: Array.from({ length: K }, () => []),
      sSd: Array.from({ length: K }, () => []),
      segId: Array.from({ length: K }, () => []),
      segment: Array.from({ length: K }, () => []),
    };
    for (const candidate of segments) {
      bank.sMean[1].push(candidate.sMean[1]);
      bank.sSd[1].push(candidate.sSd[1]);
      bank.segId[1].push(candidate.segId);
      bank.segment![1].push(candidate);
    }
    const chosen = chooseNWayItem(
      state(), inputs(segments), bank,
      new Set([0, 2, 3, 4, 5, 6]),
    );
    expect(segments.map((entry) => entry.segId)).toContain(chosen.segId);
    expect(Number.isFinite(chosen.loss)).toBe(true);
    expect(chosen.segment?.segId).toBe(chosen.segId);
  });

  it("ranks six-way speculation deterministically and bounds it to raw outcomes", () => {
    const ranked = rankOutcomes([
      { outcome: 1, probability: 0.1 }, { outcome: 2, probability: 0.4 },
      { outcome: 3, probability: 0.2 }, { outcome: 4, probability: 0.1 },
      { outcome: 5, probability: 0.1 }, { outcome: 6, probability: 0.1 },
    ]);
    expect(ranked[0]).toMatchObject({ outcome: 2, rank: 1 });
    expect(ranked[1]).toMatchObject({ outcome: 3, rank: 2 });
    expect(ranked.map((entry) => entry.outcome).sort()).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("carries a wrong raw category through a real precision session step", async () => {
    const bank = Array.from({ length: 21 }, (_, index) => {
      const band = index % 3;
      const value = [-1, 0, 1][band] + index * 1e-3;
      return {
        ...segment(100 + index),
        sMean: [0, value, value + 0.1, value - 0.1, value + 0.2, value - 0.2, value + 0.05],
      };
    });
    const config = inputs(bank);
    let askedK = -1;
    const session = new WebCortexSession(config, "nway-e2e", 7821, {
      onItem: (item) => {
        askedK = item.taskK;
        const wrongPick = askedK === 1 ? 2 : 1;
        queueMicrotask(() => session.submitAnswer(wrongPick));
      },
      onTrial: () => session.abort(),
    });
    const result = await session.run();
    expect(result.nQuestions).toBe(1);
    expect(askedK).toBeGreaterThanOrEqual(1);
    expect(result.trials[0]).toMatchObject({
      taskK: askedK, responseKind: "categorical_f1", y: 0,
    });
    expect(result.trials[0].pick).not.toBe(askedK);
    expect(result.nwayProfile).toEqual(config.nwayProfile);
  }, 30_000);
});
