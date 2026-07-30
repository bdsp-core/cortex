import type {
  ComputeEngineInputs, NWayProfileStamp,
} from "./types";
import {
  NWAY_RESPONSE_MODEL, NWAY_SELECTOR_VERSION,
} from "./types";

export interface ArtifactDraw {
  beta: number;
  distractorLapse: number;
  weight: number;
}

// Frozen integrated standard selected in the isolated n-way protocol, with
// the per-draw distractor lapse floored at the owner-approved robustness
// floor: the fit pins the lapse on the zero boundary, and the floor is the
// engine-side margin against the N3b distractor-misspecification collapse.
// The nine draws are mixed at the response-probability level; the weighted
// mean is used only by the cheap Fisher shortlist screen.
export const NWAY_ARTIFACT = Object.freeze({
  artifactId: "iiic-f1-crossfit-ensemble9-rd-20260720-floor015",
  sha256: "2e35c400739d74778c33c6a9b2f2b8590a9c6f96f3b83841fc3fc6091db8457c",
  robustnessFloor: 0.15,
  draws: Object.freeze([
    { beta: 0.8722308125877214, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 0.95165593450824, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 0.9765851716341936, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 0.9945610869560302, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 1.0173203953763694, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 1.0418629877991512, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 1.0637853303885672, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 1.0915479400184864, distractorLapse: 0.15, weight: 1 / 9 },
    { beta: 1.1945942284733333, distractorLapse: 0.15, weight: 1 / 9 },
  ] satisfies readonly ArtifactDraw[]),
  // Honest provenance: the owner approved the unfloored simulation-proven
  // integrated profile as the production standard on 2026-07-20, and the
  // 0.15 lapse floor on 2026-07-30 (evidence: n-way-protocol
  // reports/PHASE1_REPORT.md). DR07 was not separately qualified; do not
  // relabel that research result.
  approval: "owner_approved_floor015_standard_2026-07-30",
  sourceQualification: "dr07_not_qualified",
} as const);

export const NWAY_PARTICLE_PROFILE = "production_1200p_ess050_30mh_rd";
export const NWAY_ENGINE_ALGORITHM = "nway_protocol_0.2.0-rd";

export function expectedNWayProfile(candidateBankSha256: string): NWayProfileStamp {
  return {
    engineProfileId: "precision_nway_f1_ensemble9_floor015_fisher_v1",
    responseModel: NWAY_RESPONSE_MODEL,
    responseArtifactId: NWAY_ARTIFACT.artifactId,
    responseArtifactSha256: NWAY_ARTIFACT.sha256,
    selectorVersion: NWAY_SELECTOR_VERSION,
    particleProfileVersion: NWAY_PARTICLE_PROFILE,
    engineAlgorithmVersion: NWAY_ENGINE_ALGORITHM,
    candidateBankSha256,
  };
}

const SHA256 = /^[a-f0-9]{64}$/;

export function validateNWayInputs(inputs: ComputeEngineInputs): NWayProfileStamp {
  const stamp = inputs.nwayProfile;
  if (!stamp) throw new Error("production session is missing its n-way profile stamp");
  const expected = expectedNWayProfile(stamp.candidateBankSha256);
  for (const key of Object.keys(expected) as (keyof NWayProfileStamp)[]) {
    if (stamp[key] !== expected[key]) throw new Error(`n-way profile mismatch: ${key}`);
  }
  if (!SHA256.test(stamp.candidateBankSha256)) {
    throw new Error("n-way candidate bank stamp must be a lowercase SHA-256 digest");
  }
  const expectedClasses = ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"];
  if (inputs.taskCodes.join(",") !== "spike,sz,lpd,gpd,lrda,grda,iic"
      || inputs.taskClasses?.join(",") !== expectedClasses.join(",")) {
    throw new Error("n-way standard requires the frozen K=7 spike + IIIC task registry");
  }
  if (inputs.nParticles !== 1200 || inputs.terminationPolicy !== "precision_v1") {
    throw new Error("n-way standard requires precision_v1 with 1200 particles");
  }
  return stamp;
}

export function isNWaySession(inputs: ComputeEngineInputs): boolean {
  return inputs.nwayProfile !== undefined;
}
