import { normalizedArtifactDraws } from "./artifact";
import type {
  ConditionalF1ResponseArtifact, EngineProfile, ProfileStamp, ResponseGroup,
} from "./types";

const SHA256 = /^[a-f0-9]{64}$/;

function assertSha256(value: string, label: string): void {
  if (!SHA256.test(value)) throw new Error(`${label} must be a lowercase SHA-256 digest`);
}

export function validateArtifact(
  artifact: ConditionalF1ResponseArtifact,
  allowUnqualified = false,
): void {
  if (artifact.schemaVersion !== 1
      || (artifact.model !== "iiic_conditional_f1_v1"
        && artifact.model !== "iiic_conditional_f1_v1_artifact_ensemble")) {
    throw new Error("unsupported F1 artifact schema or model");
  }
  normalizedArtifactDraws(artifact);
  if (artifact.binaryLapse !== 0.025) throw new Error("binary lapse invariant violated");
  assertSha256(artifact.sha256, "artifact.sha256");
  if (!allowUnqualified && artifact.qualification !== "qualified") {
    throw new Error("unqualified response artifact is forbidden for promotable sessions");
  }
  if (!allowUnqualified && artifact.provenance.dr07 !== "qualified") {
    throw new Error("DR07-qualified artifact is required for promotable sessions");
  }
}

function validateGroups(groups: readonly ResponseGroup[], taskCount: number): void {
  const seen = new Set<number>();
  for (const group of groups) {
    if (!group.id || group.taskIndices.length === 0) throw new Error("empty response group");
    if (group.link === "binary" && group.taskIndices.length !== 1) {
      throw new Error(`binary group ${group.id} must contain exactly one task`);
    }
    if (group.link === "categorical_f1" && group.taskIndices.length < 3) {
      throw new Error(`categorical group ${group.id} must contain at least three tasks`);
    }
    for (const taskK of group.taskIndices) {
      if (!Number.isInteger(taskK) || taskK < 0 || taskK >= taskCount) {
        throw new Error(`invalid task index ${taskK} in group ${group.id}`);
      }
      if (seen.has(taskK)) throw new Error(`task ${taskK} occurs in multiple response groups`);
      seen.add(taskK);
    }
  }
  if (seen.size !== taskCount) throw new Error("response groups must cover every task exactly once");
}

export function validateProfile(
  profile: EngineProfile,
  taskCount: number,
  artifact?: ConditionalF1ResponseArtifact,
  allowUnqualified = false,
): void {
  if (profile.schemaVersion !== 1 || profile.terminationPolicy !== "precision_v1") {
    throw new Error("unsupported engine profile");
  }
  assertSha256(profile.candidateBankSha256, "candidateBankSha256");
  validateGroups(profile.responseGroups, taskCount);
  const categorical = profile.responseGroups.some((group) => group.link === "categorical_f1");
  if (profile.responseModel === "binary_ovr_v1") {
    if (categorical || profile.responseArtifactId !== null
        || profile.responseArtifactSha256 !== null
        || profile.selectorVersion !== "binary_totalvar_v1"
        || (profile.responseAggregation ?? "mixture") !== "mixture") {
      throw new Error("binary profile contains categorical configuration");
    }
    return;
  }
  if (!categorical || !artifact) throw new Error("F1 profile requires a categorical group and artifact");
  validateArtifact(artifact, allowUnqualified);
  if (profile.responseArtifactId !== artifact.artifactId
      || profile.responseArtifactSha256 !== artifact.sha256) {
    throw new Error("profile response artifact stamp mismatch");
  }
  if (profile.selectorVersion !== "categorical_totalvar_v1"
      && profile.selectorVersion !== "categorical_fisher_totalvar_v1") {
    throw new Error("F1 profile requires a categorical selector");
  }
}

export function groupForTask(profile: EngineProfile, taskK: number): ResponseGroup {
  const group = profile.responseGroups.find((candidate) => candidate.taskIndices.includes(taskK));
  if (!group) throw new Error(`task ${taskK} is absent from the response registry`);
  return group;
}

export function profileStamp(profile: EngineProfile): ProfileStamp {
  return {
    engineProfileId: profile.engineProfileId,
    responseModel: profile.responseModel,
    responseArtifactId: profile.responseArtifactId,
    responseArtifactSha256: profile.responseArtifactSha256,
    engineAlgorithmVersion: profile.engineAlgorithmVersion,
    selectorVersion: profile.selectorVersion,
    candidateBankSha256: profile.candidateBankSha256,
    responseAggregation: profile.responseAggregation ?? "mixture",
  };
}

export function assertReplayCompatible(stored: ProfileStamp, offered: ProfileStamp): void {
  for (const key of Object.keys(stored) as (keyof ProfileStamp)[]) {
    if (stored[key] !== offered[key]) throw new Error(`resume_incompatible:${key}`);
  }
}
