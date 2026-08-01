// Draw-latent response aggregation (construction B) — engine-core tests.
//
// Port of n-way-protocol tests/draw_latent.test.ts onto the production
// particle engine: each particle scores categorical picks under ITS OWN
// artifact atom, atom indices ride resampling as lineage, MH rejuvenation
// moves (t, l) only, and a single-atom ensemble reduces bit-for-bit to the
// mixture arithmetic. The byte-identical-off guarantee for the shipped
// mixture path is pinned by the untouched existing fixtures
// (nway_integration.test.ts Python references, v15_drift snapshot,
// pipeline.test.ts golden suite) — no state without a responseRuntime ever
// reaches a single new arithmetic branch.

import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  advanceCore, cloneCore, type AdvanceParams, type SessionCore,
} from "./advance";
import type { BranchExecutor } from "./branch_executor";
import type { AdvanceResult } from "./advance";
import type { Chosen } from "./choose_item";
import {
  restoreCore, snapshotCore, snapshotTransferables,
} from "./core_snapshot";
import {
  buildNWayResponseRuntime, makeResponseObservation, nwayResponseRuntimeFor,
} from "./nway_likelihood";
import { PRECISION_STATUS, PrecisionPolicy } from "./precision_policy";
import { WebCortexSession } from "./session";
import {
  NWAY_ARTIFACT, NWAY_DRAW_LATENT_ARTIFACT, NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT,
  expectedDrawLatentNWayProfile, expectedNWayProfile,
  expectedQualifiedDrawLatentNWayProfile, validateNWayInputs,
} from "./nway_profile";
import type { NWaySelectionExecutor } from "./nway_selector_executor";
import {
  PosteriorUpdateError, cloneState, logLikPackedHistory, makeState,
  resampleAndRejuvenate, resampleAndRejuvenateWithExecutor, updateObservation,
} from "./particles";
import { precomputePriorPair } from "./prior";
import { Rng } from "./rng";
import type {
  ComputeEngineInputs, ComputeSegmentMeta, NWayResponseRuntime, ParticleState,
  PriorPieces,
} from "./types";

const K = 7;

function drawLatentInputs(
  segments: ComputeSegmentMeta[] = [segment(9101, 0)],
): ComputeEngineInputs {
  return {
    taskCodes: ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
    taskLabels: ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
    taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
    taskClasses: ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
    corrL: identity(), corrT: identity(), ellStar: new Array(K).fill(0),
    nParticles: 1200, perDomainCap: 60, terminationPolicy: "precision_v1",
    precisionBandEdges: Array.from({ length: K }, () => [-0.5, 0.5]),
    nwayProfile: expectedDrawLatentNWayProfile("a".repeat(64)), segments,
  };
}

const singleAtomRuntime = (aggregation: "mixture" | "draw_latent") =>
  buildNWayResponseRuntime(aggregation, "draw-latent-single-atom", [
    { beta: 0.9912, distractorLapse: 0.15, weight: 1 },
  ]);

const threeAtomRuntime = buildNWayResponseRuntime(
  "draw_latent", "draw-latent-three-atoms", [
    { beta: 0.85, distractorLapse: 0.15, weight: 0.5 },
    { beta: 1.05, distractorLapse: 0, weight: 0.3 },
    { beta: 1.35, distractorLapse: 0.05, weight: 0.2 },
  ],
);

function identity(): number[][] {
  return Array.from({ length: K }, (_unused, i) =>
    Array.from({ length: K }, (_x, j) => Number(i === j)));
}

function segment(segId: number, shift = 0): ComputeSegmentMeta {
  return {
    segId,
    applicableTaskIdx: [0, 1, 2, 3, 4, 5, 6],
    sMean: [-0.4 + shift, -0.7 + shift, 0.9 + shift, -0.2, 0.4, -0.5, 0.1],
    sSd: [0.1, 0.12, 0.18, 0.15, 0.11, 0.17, 0.13],
  };
}

const SEGMENTS = [segment(9101, 0), segment(9102, 0.35)];

function priorPair() {
  return precomputePriorPair(identity(), identity());
}

/** Small literal cloud (nway_integration.test.ts shape family). */
function manualState(runtime?: NWayResponseRuntime): ParticleState {
  const pieces: PriorPieces = {
    K, sigmaInv: identity(), logDet: 0, L: identity(),
  };
  const N = 6;
  const t = new Float64Array(N * K);
  const l = new Float64Array(N * K);
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < K; k++) {
      t[n * K + k] = (n - 2.5) * 0.18 + (k - 3) * 0.03;
      l[n * K + k] = (n - 2.5) * 0.22 - (k - 3) * 0.025;
    }
  }
  return {
    N, K, t, l,
    w: Float64Array.from([0.1, 0.15, 0.2, 0.25, 0.18, 0.12]),
    logPrior: new Float64Array(N),
    logLik: new Float64Array(N),
    history: [],
    prior: { tPieces: pieces, lPieces: pieces },
    ...(runtime ? { responseRuntime: runtime } : {}),
  };
}

/** Fixed trajectory: [askedK, segmentIndex, rawPick] with a binary spike
 * trial plus wrong and right categorical picks (isolated-port TRAJECTORY). */
const TRAJECTORY: readonly [number, number, number][] = [
  [3, 1, 5], [1, 1, 1], [5, 0, 2], [0, 0, 0], [2, 1, 6], [4, 0, 4],
];

function runTrajectory(state: ParticleState): void {
  const rng = new Rng(90_210);
  for (const [askedK, segmentIndex, rawPick] of TRAJECTORY) {
    updateObservation(state, makeResponseObservation(
      askedK, SEGMENTS[segmentIndex], rawPick, askedK === 0 ? "spike" : "iiic",
    ));
    resampleAndRejuvenate(state, rng, 2, 2.38 / Math.sqrt(2 * K));
  }
}

describe("draw-latent response aggregation (construction B)", () => {
  it("reduces bit-for-bit to the mixture path with a single atom", () => {
    const mixtureState = makeState(48, K, priorPair(), new Rng(4242), singleAtomRuntime("mixture"));
    const drawState = cloneState(mixtureState);
    drawState.responseRuntime = singleAtomRuntime("draw_latent");
    drawState.atomIndex = new Int32Array(48);
    runTrajectory(mixtureState);
    runTrajectory(drawState);
    expect(Array.from(drawState.t)).toEqual(Array.from(mixtureState.t));
    expect(Array.from(drawState.l)).toEqual(Array.from(mixtureState.l));
    expect(Array.from(drawState.w)).toEqual(Array.from(mixtureState.w));
    expect(Array.from(drawState.logPrior)).toEqual(Array.from(mixtureState.logPrior));
    expect(Array.from(drawState.logLik)).toEqual(Array.from(mixtureState.logLik));
    expect(drawState.lastRejuvenation).toEqual(mixtureState.lastRejuvenation);
    expect(drawState.history).toEqual(mixtureState.history);
    expect(Array.from(drawState.atomIndex!)).toEqual(new Array(48).fill(0));
  });

  it("rides atom indices through ancestor selection as lineage", () => {
    const state = manualState(threeAtomRuntime);
    state.atomIndex = Int32Array.from([0, 1, 2, 0, 1, 2]);
    state.w = Float64Array.from([0, 0, 0, 0, 1, 0]);
    resampleAndRejuvenate(state, new Rng(11), 0, 0.4);
    expect(Array.from(state.atomIndex)).toEqual([1, 1, 1, 1, 1, 1]);
    expect(state.lastRejuvenation?.distinctAncestors).toBe(1);
  });

  it("initializes atom lineage from the artifact weights at cloud creation", () => {
    const drawState = makeState(256, K, priorPair(), new Rng(7), threeAtomRuntime);
    const mixtureState = makeState(256, K, priorPair(), new Rng(7));
    expect(Array.from(drawState.t)).toEqual(Array.from(mixtureState.t));
    expect(Array.from(drawState.l)).toEqual(Array.from(mixtureState.l));
    expect(mixtureState.atomIndex).toBeUndefined();
    expect(drawState.atomIndex).toHaveLength(256);
    const counts = [0, 0, 0];
    for (const atom of drawState.atomIndex!) {
      expect(atom).toBeGreaterThanOrEqual(0);
      expect(atom).toBeLessThan(3);
      counts[atom] += 1;
    }
    for (const count of counts) expect(count).toBeGreaterThan(0);
  });

  it("fails closed on inconsistent aggregation state before evidence lands", () => {
    const missingLineage = manualState(threeAtomRuntime);
    const observation = makeResponseObservation(2, SEGMENTS[0], 5, "iiic");
    expect(() => updateObservation(missingLineage, observation))
      .toThrow(/missing per-particle atom lineage/);

    const outOfRange = manualState(threeAtomRuntime);
    outOfRange.atomIndex = Int32Array.from([0, 0, 0, 0, 0, 9]);
    expect(() => updateObservation(outOfRange, observation))
      .toThrow(/outside the artifact atoms/);

    const mixtureWithLineage = manualState();
    mixtureWithLineage.atomIndex = new Int32Array(6);
    expect(() => updateObservation(mixtureWithLineage, observation))
      .toThrow(PosteriorUpdateError);
    expect(() => updateObservation(mixtureWithLineage, observation))
      .toThrow(/must not carry atom lineage/);

    // Fail-closed means the cloud is untouched by the refused update.
    const before = manualState(threeAtomRuntime);
    expect(Array.from(missingLineage.w)).toEqual(Array.from(before.w));
    expect(missingLineage.history).toEqual([]);
  });

  it("keeps exact state when draw-latent MH history replay is asynchronous", async () => {
    const seeded = makeState(24, K, priorPair(), new Rng(13), threeAtomRuntime);
    updateObservation(seeded, makeResponseObservation(3, SEGMENTS[0], 5, "iiic"));
    updateObservation(seeded, makeResponseObservation(5, SEGMENTS[1], 2, "iiic"));
    const serial = cloneState(seeded);
    const parallel = cloneState(seeded);
    const serialRng = new Rng(9901);
    const parallelRng = new Rng(9901);
    const executor: NWaySelectionExecutor = {
      workerCount: 2,
      ready: () => Promise.resolve(),
      score: () => Promise.reject(new Error("not used")),
      screen: () => Promise.reject(new Error("not used")),
      historyLikelihood: (history, N, particleK, t, l, atomIndex) => {
        const result = new Float64Array(N);
        logLikPackedHistory(
          history, N, particleK, t, l, result,
          undefined, threeAtomRuntime, atomIndex,
        );
        return Promise.resolve(result);
      },
      dispose: () => undefined,
    };
    resampleAndRejuvenate(serial, serialRng, 2, 0.1, 4);
    await resampleAndRejuvenateWithExecutor(parallel, parallelRng, 2, 0.1, executor, 4);
    expect(parallel.t).toEqual(serial.t);
    expect(parallel.l).toEqual(serial.l);
    expect(parallel.w).toEqual(serial.w);
    expect(parallel.logPrior).toEqual(serial.logPrior);
    expect(parallel.logLik).toEqual(serial.logLik);
    expect(Array.from(parallel.atomIndex!)).toEqual(Array.from(serial.atomIndex!));
    expect(parallel.lastRejuvenation).toEqual(serial.lastRejuvenation);
    expect(parallelRng.snapshot()).toEqual(serialRng.snapshot());
  });

  it("freezes and validates the research atoms17 draw-latent profile stamp", () => {
    // Canonical-draws hash discipline (artifact_floor.py / the constants
    // generator): SHA-256 of the compact sorted-key JSON of the draw table.
    const canonical = JSON.stringify(
      NWAY_DRAW_LATENT_ARTIFACT.draws.map((draw) => ({
        beta: draw.beta,
        distractorLapse: draw.distractorLapse,
        weight: draw.weight,
      })),
    );
    expect(createHash("sha256").update(canonical).digest("hex"))
      .toBe(NWAY_DRAW_LATENT_ARTIFACT.sha256);
    expect(NWAY_DRAW_LATENT_ARTIFACT.draws).toHaveLength(17);
    expect(NWAY_DRAW_LATENT_ARTIFACT.robustnessFloor).toBe(0.15);
    for (const draw of NWAY_DRAW_LATENT_ARTIFACT.draws) {
      expect(draw.distractorLapse)
        .toBeGreaterThanOrEqual(NWAY_DRAW_LATENT_ARTIFACT.robustnessFloor);
    }
    // Research-only: never confusable with the qualified production artifact.
    expect(NWAY_DRAW_LATENT_ARTIFACT.approval).toBe("research_only_not_promoted");
    expect(NWAY_DRAW_LATENT_ARTIFACT.artifactId).not.toBe(NWAY_ARTIFACT.artifactId);

    const config = drawLatentInputs();
    expect(validateNWayInputs(config).responseAggregation).toBe("draw_latent");
    const runtime = nwayResponseRuntimeFor(config)!;
    expect(runtime.aggregation).toBe("draw_latent");
    expect(runtime.atoms).toHaveLength(17);
    expect(nwayResponseRuntimeFor(config)).toBe(runtime); // cached singleton

    // Tampered stamps fail closed on the exact offending key.
    const wrongArtifact = drawLatentInputs();
    wrongArtifact.nwayProfile = {
      ...wrongArtifact.nwayProfile!, responseArtifactSha256: "b".repeat(64),
    };
    expect(() => validateNWayInputs(wrongArtifact)).toThrow(/responseArtifactSha256/);
    expect(() => nwayResponseRuntimeFor(wrongArtifact))
      .toThrow(/does not name the registered artifact/);
    const wrongAggregation = drawLatentInputs();
    wrongAggregation.nwayProfile = {
      ...wrongAggregation.nwayProfile!,
      responseAggregation: "blend" as unknown as "mixture",
    };
    expect(() => validateNWayInputs(wrongAggregation)).toThrow(/responseAggregation/);
    // A mixture stamp claiming the draw-latent profile id is refused too.
    const mixtureClaim = drawLatentInputs();
    mixtureClaim.nwayProfile = {
      ...expectedNWayProfile("a".repeat(64)),
      engineProfileId: "precision_nway_f1_engine_frame_atoms17_draw_latent_rd_v1",
    };
    expect(() => validateNWayInputs(mixtureClaim)).toThrow(/engineProfileId/);
    // And the mixture stamp resolves to NO runtime — the frozen path.
    const mixture = drawLatentInputs();
    mixture.nwayProfile = expectedNWayProfile("a".repeat(64));
    expect(nwayResponseRuntimeFor(mixture)).toBeUndefined();
  });

  it("registers the QUALIFIED nesting34 artifact and resolves its runtime", () => {
    // The canonical-draws discipline is Python's (artifact_floor.py):
    // json.dumps renders integral floats as "1.0" where JSON.stringify says
    // "1" — the nesting atom's lambda_d = 1 hits exactly that difference, so
    // the re-derivation formats numbers the Python way.
    const pyNumber = (value: number) =>
      Number.isInteger(value) ? `${value}.0` : `${value}`;
    const canonical = `[${NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.draws.map((draw) =>
      `{"beta":${pyNumber(draw.beta)},"distractorLapse":${
        pyNumber(draw.distractorLapse)},"weight":${pyNumber(draw.weight)}}`,
    ).join(",")}]`;
    expect(createHash("sha256").update(canonical).digest("hex"))
      .toBe(NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.sha256);
    expect(NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.draws).toHaveLength(34);
    expect(NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.robustnessFloor).toBe(0.15);
    for (const draw of NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.draws) {
      expect(draw.distractorLapse)
        .toBeGreaterThanOrEqual(NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.robustnessFloor);
    }
    // The lambda_d = 1 binary-nesting atom closes the grid: uniform-collapse
    // sessions concentrate on it and the engine gracefully becomes binary.
    const nesting = NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.draws.at(-1)!;
    expect(nesting.distractorLapse).toBe(1);
    expect(NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.artifactId)
      .not.toBe(NWAY_DRAW_LATENT_ARTIFACT.artifactId);

    const config = drawLatentInputs();
    config.nwayProfile = expectedQualifiedDrawLatentNWayProfile("a".repeat(64));
    expect(validateNWayInputs(config).engineProfileId)
      .toBe("precision_nway_f1_nesting34_draw_latent_v1");
    const runtime = nwayResponseRuntimeFor(config)!;
    expect(runtime.aggregation).toBe("draw_latent");
    expect(runtime.atoms).toHaveLength(34);
    expect(runtime.artifactId).toBe(NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT.artifactId);
    expect(nwayResponseRuntimeFor(config)).toBe(runtime); // cached singleton
    // The research and qualified stamps resolve DIFFERENT runtimes.
    expect(nwayResponseRuntimeFor(drawLatentInputs())).not.toBe(runtime);

    // A tampered qualified stamp fails closed on the exact offending key.
    const wrongSha = drawLatentInputs();
    wrongSha.nwayProfile = {
      ...expectedQualifiedDrawLatentNWayProfile("a".repeat(64)),
      responseArtifactSha256: "c".repeat(64),
    };
    expect(() => validateNWayInputs(wrongSha)).toThrow(/responseArtifactSha256/);
    expect(() => nwayResponseRuntimeFor(wrongSha))
      .toThrow(/does not name the registered artifact/);

    // The retired legacy unfloored stamp is refused on the profile id: it
    // matches neither the mixture nor either draw-latent expectation.
    const legacy = drawLatentInputs();
    legacy.nwayProfile = {
      ...expectedNWayProfile("a".repeat(64)),
      engineProfileId: "precision_nway_f1_ensemble9_fisher_v1",
      responseArtifactId: "iiic-f1-crossfit-ensemble9-rd-20260720",
    };
    expect(() => validateNWayInputs(legacy)).toThrow(/engineProfileId/);
  });

  it("refuses history replay whose lineage contradicts the aggregation", () => {
    const state = manualState(threeAtomRuntime);
    state.atomIndex = Int32Array.from([0, 1, 2, 0, 1, 2]);
    updateObservation(state, makeResponseObservation(2, SEGMENTS[0], 5, "iiic"));
    const output = new Float64Array(state.N);
    expect(() => logLikPackedHistory(
      state.packedHistory!, state.N, K, state.t, state.l, output,
      undefined, threeAtomRuntime, undefined,
    )).toThrow(/missing per-particle atom lineage/);
    expect(() => logLikPackedHistory(
      state.packedHistory!, state.N, K, state.t, state.l, output,
      undefined, undefined, state.atomIndex,
    )).toThrow(/atom lineage requires draw-latent aggregation/);
  });
});

function iiicSegment(segId: number, shift: number): ComputeSegmentMeta {
  return {
    segId,
    applicableTaskIdx: [1, 2, 3, 4, 5, 6],
    sMean: [0, shift, shift + 0.1, shift - 0.1, shift + 0.2, shift - 0.2, shift + 0.05],
    sSd: [0.1, 0.12, 0.18, 0.15, 0.11, 0.17, 0.13],
  };
}

function iiicBank(count: number): ComputeSegmentMeta[] {
  return Array.from({ length: count }, (_unused, index) =>
    iiicSegment(100 + index, [-1, 0, 1][index % 3] + index * 1e-3));
}

function withoutTiming<T extends { timing: unknown }>(value: T): Omit<T, "timing"> {
  const { timing: _timing, ...deterministic } = value;
  return deterministic;
}

function drawLatentCore(config: ComputeEngineInputs, nParticles: number): SessionCore {
  const prior = precomputePriorPair(config.corrL, config.corrT);
  const rng = new Rng(441);
  const policy = PrecisionPolicy.fromInputs(config);
  policy.reset(K);
  return {
    state: makeState(nParticles, K, prior, rng, nwayResponseRuntimeFor(config)),
    rng,
    policy,
    remaining: new Set(config.segments.map((entry) => entry.segId)),
    nPerTask: new Array(K).fill(0),
    cappedTasks: new Set<number>(),
    lastOutcomes: new Array(K).fill(PRECISION_STATUS.ACTIVE),
    lastTaskK: -1,
    streakCount: 0,
  };
}

const CORE_PARAMS: AdvanceParams = {
  K,
  nParticles: 80,
  perDomainCap: 60,
  nMhSteps: 2,
  essThresholdFrac: 1.1, // force the lineage-riding rejuvenation path
  proposalScale: 2.38 / Math.sqrt(2 * K),
  firstItemTopN: 10,
  maxConsecutiveSameDomain: 5,
  k7Spike: true,
  spikeIdx: 0,
  nSubsample: 16,
  uncertaintyAwareSubsample: true,
};

describe("draw-latent state-carrying surfaces", () => {
  it("carries atom lineage through core snapshot, restore, and adoption", () => {
    const config = drawLatentInputs(iiicBank(24));
    const core = drawLatentCore(config, 80);
    expect(core.state.atomIndex).toHaveLength(80);
    const prior = core.state.prior;
    const first = config.segments[0];
    const chosen: Chosen = {
      k: 1, s: first.sMean[1], sSd: first.sSd[1], segId: first.segId, loss: 0,
    };

    const snapshot = snapshotCore(core);
    // Lineage is one more owned buffer on the worker-transfer manifest.
    expect(snapshotTransferables(snapshot)).toHaveLength(8);
    expect(Array.from(snapshot.state.atomIndex!))
      .toEqual(Array.from(core.state.atomIndex!));
    const restored = restoreCore(snapshot, config, prior);
    expect(restored.state.responseRuntime).toBe(nwayResponseRuntimeFor(config));
    expect(Array.from(restored.state.atomIndex!))
      .toEqual(Array.from(core.state.atomIndex!));

    // A wrong categorical pick on the restored branch is bit-identical to the
    // same step on a direct clone — including the resampled lineage.
    const reference = advanceCore(cloneCore(core), config, chosen, 5, CORE_PARAMS, 0);
    const adopted = advanceCore(restored, config, chosen, 5, CORE_PARAMS, 0);
    expect(withoutTiming(adopted)).toEqual(withoutTiming(reference));
    expect(Array.from(adopted.core.state.atomIndex!))
      .toEqual(Array.from(reference.core.state.atomIndex!));
    expect(adopted.core.rng.snapshot()).toEqual(reference.core.rng.snapshot());
    // The authoritative core and its lineage stay untouched by the branch.
    expect(Array.from(core.state.atomIndex!))
      .toEqual(Array.from(snapshot.state.atomIndex!));
    expect(core.state.history).toHaveLength(0);
  });

  it("fails closed when a snapshot loses lineage or crosses aggregation modes", () => {
    const config = drawLatentInputs(iiicBank(12));
    const core = drawLatentCore(config, 24);
    const prior = core.state.prior;
    const snapshot = snapshotCore(core);

    const { atomIndex: _dropped, ...withoutLineage } = snapshot.state;
    expect(() => restoreCore(
      { ...snapshot, state: withoutLineage }, config, prior,
    )).toThrow(/missing per-particle atom lineage/);

    const outOfRange = snapshotCore(core);
    outOfRange.state.atomIndex![0] = 99;
    expect(() => restoreCore(outOfRange, config, prior))
      .toThrow(/outside the artifact atoms/);

    // A mixture session must refuse a snapshot that carries lineage...
    const mixtureConfig = drawLatentInputs(iiicBank(12));
    mixtureConfig.nwayProfile = expectedNWayProfile("a".repeat(64));
    expect(() => restoreCore(snapshotCore(core), mixtureConfig, prior))
      .toThrow(/must not carry atom lineage/);

    // ...and a draw-latent session must refuse a mixture snapshot.
    const mixtureCore = drawLatentCore(mixtureConfig, 24);
    expect(mixtureCore.state.atomIndex).toBeUndefined();
    expect(() => restoreCore(snapshotCore(mixtureCore), config, prior))
      .toThrow(/missing per-particle atom lineage/);
  });

  it("is bit-identical serial vs speculative with snapshot branch adoption", async () => {
    /** branch_executor.test.ts pattern: structured-clone semantics without
     * browser globals; lineage must survive the worker snapshot protocol. */
    class SnapshotBranchExecutor implements BranchExecutor {
      readonly capacity = 1;
      private readonly prior;
      constructor(private readonly config: ComputeEngineInputs) {
        this.prior = precomputePriorPair(config.corrL, config.corrT);
      }
      advance(
        core: SessionCore, chosen: Chosen, params: AdvanceParams,
        trialIndex: number, pick: number,
      ): Promise<AdvanceResult> {
        const snapshot = snapshotCore(core);
        return Promise.resolve().then(() => {
          const restored = restoreCore(snapshot, this.config, this.prior);
          const result = advanceCore(
            restored, this.config, chosen, pick, params, trialIndex,
          );
          result.timing.executionMode = "dual_branch";
          result.timing.speculative = true;
          return result;
        });
      }
      dispose(): void {}
    }

    async function run(speculative: boolean) {
      const config = drawLatentInputs(iiicBank(21));
      let trials = 0;
      const session = new WebCortexSession(config, "draw-latent-e2e", 7821, {
        onItem: (item) => {
          const wrongPick = item.taskK === 1 ? 2 : 1;
          queueMicrotask(() => session.submitAnswer(wrongPick));
        },
        onTrial: () => { trials += 1; if (trials >= 2) session.abort(); },
      }, speculative ? {
        speculative: true,
        branchExecutor: new SnapshotBranchExecutor(config),
      } : {});
      return session.run();
    }

    const serial = await run(false);
    const speculated = await run(true);
    expect(speculated.trials.length).toBeGreaterThanOrEqual(2);
    expect(speculated.servedSegIds).toEqual(serial.servedSegIds);
    expect(speculated.trials).toEqual(serial.trials);
    expect(speculated.traj).toEqual(serial.traj);
    expect(serial.trials[0].responseKind).toBe("categorical_f1");
    expect(serial.nwayProfile?.responseAggregation).toBe("draw_latent");
  }, 120_000);
});
