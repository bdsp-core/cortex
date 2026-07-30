import { describe, expect, it } from "vitest";
import {
  assertReplayCompatible, profileStamp, validateArtifact, validateProfile,
} from "../src/profile";
import { artifact, binaryProfile, K, nwayProfile } from "./fixtures";

describe("immutable engine profile", () => {
  it("rejects exploratory artifacts on the promotable path", () => {
    expect(() => validateArtifact(artifact)).toThrow(/unqualified/);
    expect(() => validateArtifact(artifact, true)).not.toThrow();
    expect(() => validateProfile(nwayProfile, K, artifact, true)).not.toThrow();
  });

  it("validates the binary rollback profile without an artifact", () => {
    expect(() => validateProfile(binaryProfile, K)).not.toThrow();
  });

  it("detects profile drift during resume", () => {
    const stored = profileStamp(nwayProfile);
    expect(() => assertReplayCompatible(stored, { ...stored })).not.toThrow();
    expect(() => assertReplayCompatible(stored, {
      ...stored, engineAlgorithmVersion: "nway_protocol_0.2.0",
    })).toThrow("resume_incompatible:engineAlgorithmVersion");
  });

  it("rejects response registries with overlapping tasks", () => {
    const invalid = {
      ...nwayProfile,
      responseGroups: [
        ...nwayProfile.responseGroups,
        { id: "overlap", link: "binary" as const, taskIndices: [1] },
      ],
    };
    expect(() => validateProfile(invalid, K, artifact, true)).toThrow(/multiple/);
  });
});

