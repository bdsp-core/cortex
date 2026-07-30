import { describe, expect, it } from "vitest";
import { makeObservation } from "../src/likelihood";
import { updateProtocol } from "../src/particles";
import { profileStamp } from "../src/profile";
import { restoreProtocolState, snapshotProtocolState } from "../src/snapshot";
import { artifact, manualState, nwayProfile, segments } from "./fixtures";

describe("versioned n-way state snapshots", () => {
  it("round-trips categorical history and typed arrays exactly", () => {
    const state = manualState();
    const bank = segments(1);
    updateProtocol(
      state, nwayProfile, artifact, bank,
      makeObservation(nwayProfile, 2, 0, 5),
    );
    const stamp = profileStamp(nwayProfile);
    const snapshot = snapshotProtocolState(state, stamp);
    const restored = restoreProtocolState(snapshot, stamp, state.prior);
    expect(Array.from(restored.w)).toEqual(Array.from(state.w));
    expect(Array.from(restored.logLik)).toEqual(Array.from(state.logLik));
    expect(restored.history).toEqual(state.history);
    restored.w[0] = 99;
    expect(snapshot.w[0]).not.toBe(99);
  });

  it("rejects restoration under a drifted response artifact", () => {
    const state = manualState();
    const stamp = profileStamp(nwayProfile);
    const snapshot = snapshotProtocolState(state, stamp);
    expect(() => restoreProtocolState(snapshot, {
      ...stamp, responseArtifactSha256: "f".repeat(64),
    }, state.prior)).toThrow(/resume_incompatible/);
  });
});

