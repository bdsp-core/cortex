import { describe, expect, it } from "vitest";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import { makeObservation } from "../src/likelihood";
import {
  atomPosterior, cloneProtocolState, makeProtocolState,
  resampleAndRejuvenateProtocol, updateProtocol,
} from "../src/particles";
import { profileStamp, validateProfile } from "../src/profile";
import { restoreProtocolState, snapshotProtocolState } from "../src/snapshot";
import { executeRankedBranches } from "../src/speculation";
import type {
  ConditionalF1ArtifactEnsemble, EngineProfile, ProtocolParticleState,
} from "../src/types";
import {
  binaryProfile, K, manualState, nwayProfile, randomState, segments,
} from "./fixtures";

function ensemble(
  draws: ConditionalF1ArtifactEnsemble["draws"],
  artifactId: string,
): ConditionalF1ArtifactEnsemble {
  return {
    schemaVersion: 1,
    artifactId,
    model: "iiic_conditional_f1_v1_artifact_ensemble",
    qualification: "exploratory_unqualified",
    draws,
    binaryLapse: 0.025,
    sha256: "c".repeat(64),
    provenance: {
      source: "draw-latent engine test fixture",
      fitSplit: "synthetic; not promotion eligible",
      dr07: "not_qualified",
    },
  };
}

const singleAtomArtifact = ensemble(
  [{ beta: 0.9912, distractorLapse: 0.15, weight: 1 }],
  "draw-latent-single-atom",
);
const threeAtomArtifact = ensemble(
  [
    { beta: 0.85, distractorLapse: 0.15, weight: 0.5 },
    { beta: 1.05, distractorLapse: 0, weight: 0.3 },
    { beta: 1.35, distractorLapse: 0.05, weight: 0.2 },
  ],
  "draw-latent-three-atoms",
);

function profileFor(
  artifact: ConditionalF1ArtifactEnsemble,
  responseAggregation?: "mixture" | "draw_latent",
): EngineProfile {
  return {
    ...nwayProfile,
    responseArtifactId: artifact.artifactId,
    responseArtifactSha256: artifact.sha256,
    ...(responseAggregation ? { responseAggregation } : {}),
  };
}

const mixtureProfile = profileFor(singleAtomArtifact);
const drawLatentProfile = profileFor(singleAtomArtifact, "draw_latent");
const drawLatentThreeProfile = profileFor(threeAtomArtifact, "draw_latent");

/** Fixed trajectory: [askedK, segmentIndex, rawPick] with binary + wrong/right picks. */
const TRAJECTORY: readonly [number, number, number][] = [
  [3, 1, 5], [1, 1, 1], [5, 3, 2], [0, 0, 0], [2, 1, 6], [4, 3, 4],
];

function runTrajectory(
  profile: EngineProfile,
  artifact: ConditionalF1ArtifactEnsemble,
  state: ProtocolParticleState,
): void {
  const bank = segments(4);
  const rng = new Rng(90_210);
  for (const [askedK, segmentIndex, rawPick] of TRAJECTORY) {
    updateProtocol(
      state, profile, artifact, bank,
      makeObservation(profile, askedK, segmentIndex, rawPick),
    );
    resampleAndRejuvenateProtocol(
      state, profile, artifact, bank, rng, 2, 2.38 / Math.sqrt(2 * K),
    );
  }
}

describe("draw-latent response aggregation (construction B)", () => {
  it("reduces bit-for-bit to the mixture path with a single atom", () => {
    const mixtureState = randomState(48);
    const drawState = cloneProtocolState(mixtureState);
    drawState.atomIndex = new Int32Array(48);
    runTrajectory(mixtureProfile, singleAtomArtifact, mixtureState);
    runTrajectory(drawLatentProfile, singleAtomArtifact, drawState);
    expect(Array.from(drawState.t)).toEqual(Array.from(mixtureState.t));
    expect(Array.from(drawState.l)).toEqual(Array.from(mixtureState.l));
    expect(Array.from(drawState.w)).toEqual(Array.from(mixtureState.w));
    expect(Array.from(drawState.logPrior)).toEqual(Array.from(mixtureState.logPrior));
    expect(Array.from(drawState.logLik)).toEqual(Array.from(mixtureState.logLik));
    expect(drawState.lastRejuvenation).toEqual(mixtureState.lastRejuvenation);
    expect(drawState.history).toEqual(mixtureState.history);
    const [mass] = atomPosterior(drawState, singleAtomArtifact);
    expect(mass).toBeCloseTo(1, 12);
  });

  it("rides atom indices through ancestor selection as lineage", () => {
    const state = manualState(6);
    state.atomIndex = Int32Array.from([0, 1, 2, 0, 1, 2]);
    state.w = Float64Array.from([0, 0, 0, 0, 1, 0]);
    resampleAndRejuvenateProtocol(
      state, drawLatentThreeProfile, threeAtomArtifact, segments(1),
      new Rng(11), 0, 0.4,
    );
    expect(Array.from(state.atomIndex)).toEqual([1, 1, 1, 1, 1, 1]);
    expect(state.lastRejuvenation?.distinctAncestors).toBe(1);
    const mass = atomPosterior(state, threeAtomArtifact);
    expect(mass[0]).toBe(0);
    expect(mass[1]).toBeCloseTo(1, 12);
    expect(mass[2]).toBe(0);
  });

  it("initializes atom lineage from the artifact weights at cloud creation", () => {
    const prior = manualState().prior;
    const drawState = makeProtocolState(256, K, prior, new Rng(7), {
      responseAggregation: "draw_latent", artifact: threeAtomArtifact,
    });
    const mixtureState = makeProtocolState(256, K, prior, new Rng(7));
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
    expect(() => makeProtocolState(8, K, prior, new Rng(7), {
      responseAggregation: "draw_latent",
    })).toThrow(/requires a response artifact/);
  });

  it("fails closed when the draw-latent state has no atom lineage", () => {
    const state = randomState(8);
    expect(() => updateProtocol(
      state, drawLatentProfile, singleAtomArtifact, segments(1),
      makeObservation(drawLatentProfile, 2, 0, 5),
    )).toThrow(/atom lineage/);
    state.atomIndex = Int32Array.from([0, 0, 0, 0, 0, 0, 0, 9]);
    expect(() => updateProtocol(
      state, drawLatentProfile, singleAtomArtifact, segments(1),
      makeObservation(drawLatentProfile, 2, 0, 5),
    )).toThrow(/outside the artifact draws/);
  });

  it("carries atom lineage through snapshot, restore, and resumed updates", () => {
    const state = makeProtocolState(24, K, manualState().prior, new Rng(13), {
      responseAggregation: "draw_latent", artifact: threeAtomArtifact,
    });
    const bank = segments(2);
    updateProtocol(
      state, drawLatentThreeProfile, threeAtomArtifact, bank,
      makeObservation(drawLatentThreeProfile, 3, 0, 5),
    );
    const stamp = profileStamp(drawLatentThreeProfile);
    expect(stamp.responseAggregation).toBe("draw_latent");
    const snapshot = snapshotProtocolState(state, stamp);
    expect(Array.from(snapshot.atomIndex!)).toEqual(Array.from(state.atomIndex!));
    const restored = restoreProtocolState(snapshot, stamp, state.prior);
    expect(Array.from(restored.atomIndex!)).toEqual(Array.from(state.atomIndex!));
    restored.atomIndex![0] = 2;
    expect(snapshot.atomIndex![0]).toBe(state.atomIndex![0]);
    restored.atomIndex![0] = state.atomIndex![0];

    // A resumed session must continue the exact per-atom likelihood stream.
    const continued = cloneProtocolState(state);
    const observation = makeObservation(drawLatentThreeProfile, 5, 1, 2);
    updateProtocol(continued, drawLatentThreeProfile, threeAtomArtifact, bank, observation);
    updateProtocol(restored, drawLatentThreeProfile, threeAtomArtifact, bank, observation);
    expect(Array.from(restored.w)).toEqual(Array.from(continued.w));
    expect(Array.from(restored.logLik)).toEqual(Array.from(continued.logLik));
  });

  it("fails closed when a draw-latent snapshot loses lineage or mode", () => {
    const state = makeProtocolState(12, K, manualState().prior, new Rng(29), {
      responseAggregation: "draw_latent", artifact: threeAtomArtifact,
    });
    const stamp = profileStamp(drawLatentThreeProfile);
    const snapshot = snapshotProtocolState(state, stamp);
    const { atomIndex: _dropped, ...withoutLineage } = snapshot;
    expect(() => restoreProtocolState(
      withoutLineage, stamp, state.prior,
    )).toThrow(/missing per-particle atom lineage/);
    const mixtureStamp = profileStamp(profileFor(threeAtomArtifact));
    expect(() => restoreProtocolState(
      snapshot, mixtureStamp, state.prior,
    )).toThrow(/resume_incompatible:responseAggregation/);
    const mixtureSnapshot = snapshotProtocolState(manualState(12), mixtureStamp);
    expect(() => restoreProtocolState(
      { ...mixtureSnapshot, atomIndex: state.atomIndex!.slice() },
      mixtureStamp, state.prior,
    )).toThrow(/mixture snapshot must not carry atom lineage/);
  });

  it("preserves atom lineage across speculation clones and ranked adoption", async () => {
    const authoritative = makeProtocolState(16, K, manualState().prior, new Rng(41), {
      responseAggregation: "draw_latent", artifact: threeAtomArtifact,
    });
    const lineage = Array.from(authoritative.atomIndex!);
    const bank = segments(1);
    const distribution = [1, 2, 3, 4, 5, 6].map((outcome) => ({
      outcome, probability: outcome === 4 ? 0.4 : 0.12,
    }));
    const adopted = await executeRankedBranches({
      authoritative,
      distribution,
      actualOutcome: Promise.resolve(6),
      cloneCore: cloneProtocolState,
      advance: async (core, outcome) => {
        updateProtocol(
          core, drawLatentThreeProfile, threeAtomArtifact, bank,
          makeObservation(drawLatentThreeProfile, 3, 0, outcome),
        );
        return core;
      },
      helperAvailable: true,
    });
    expect(adopted.cacheHit).toBe(false);
    expect(Array.from(adopted.result.atomIndex!)).toEqual(lineage);
    expect(adopted.result.history).toHaveLength(1);
    // The authoritative core and its lineage stay untouched by branches.
    expect(authoritative.history).toHaveLength(0);
    expect(Array.from(authoritative.atomIndex!)).toEqual(lineage);

    const cached = await executeRankedBranches({
      authoritative,
      distribution,
      actualOutcome: Promise.resolve(4),
      cloneCore: cloneProtocolState,
      advance: async (core, outcome) => {
        updateProtocol(
          core, drawLatentThreeProfile, threeAtomArtifact, bank,
          makeObservation(drawLatentThreeProfile, 3, 0, outcome),
        );
        return core;
      },
      helperAvailable: true,
    });
    expect(cached.cacheHit).toBe(true);
    expect(Array.from(cached.result.atomIndex!)).toEqual(lineage);
  });

  it("accepts a categorical draw-latent profile and rejects a binary one", () => {
    expect(() => validateProfile(
      drawLatentThreeProfile, K, threeAtomArtifact, true,
    )).not.toThrow();
    expect(() => validateProfile(
      { ...binaryProfile, responseAggregation: "draw_latent" }, K,
    )).toThrow(/binary profile contains categorical configuration/);
  });
});
