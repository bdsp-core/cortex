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
  buildNWayResponseRuntime, makeResponseObservation, nwayResponseRuntimeFor,
} from "./nway_likelihood";
import {
  NWAY_ARTIFACT, NWAY_DRAW_LATENT_ARTIFACT, expectedDrawLatentNWayProfile,
  expectedNWayProfile, validateNWayInputs,
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
