import type {
  ComputeEngineInputs, NWayProfileStamp, NWayResponseAggregation,
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

// Research-only draw-latent artifact (construction B): 17 population atoms
// from the 2026-07-30 engine-frame hierarchical refit, floored at the same
// owner-approved 0.15 robustness floor. NOT qualified for serving: the fitter
// payload is research_only_not_promoted / promotionForbidden, and the server
// stamps this profile only through the explicit non-production research
// escape (services/api/nway_profile.py). Regenerate with
// n-way-protocol/scripts/make_engine_profile_constants.py --emit-atoms17
// (the generator is validated bit-for-bit against the floor015 table above
// before it is trusted; sha256 is the canonical compact sorted-key JSON of
// the draws array, the artifact_floor.py discipline).
export const NWAY_DRAW_LATENT_ARTIFACT = Object.freeze({
  artifactId: "iiic-f1-engine-frame-atoms17-rd-20260730",
  sha256: "b25bd1c5e680b77df394eeccab0961bd1f67c1593ff3d4a144d0b92eb8f90fb8",
  robustnessFloor: 0.15,
  draws: Object.freeze([
    { beta: 0.6030846479252363, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.692381482844384, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.74831249929785, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.7934937607093849, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.8335652252676693, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.8709885194308897, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.9071741680803833, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.9431120331186459, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 0.9796349974777218, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.017572350455256, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.05788365900443, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.1018339586269643, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.1513013009569921, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.209442059664358, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.2824651855790978, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.3860635387599891, distractorLapse: 0.15, weight: 0.058823529411764705 },
    { beta: 1.5912935797399814, distractorLapse: 0.15, weight: 0.058823529411764705 },
  ] satisfies readonly ArtifactDraw[]),
  approval: "research_only_not_promoted",
  sourceQualification: "dr07_gate_passed_owner_ratification_pending",
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

/** The research draw-latent stamp — stampable only through the server's
 * explicit non-production escape; never the default for any sitting. */
export function expectedDrawLatentNWayProfile(
  candidateBankSha256: string,
): NWayProfileStamp {
  return {
    engineProfileId: "precision_nway_f1_engine_frame_atoms17_draw_latent_rd_v1",
    responseModel: NWAY_RESPONSE_MODEL,
    responseArtifactId: NWAY_DRAW_LATENT_ARTIFACT.artifactId,
    responseArtifactSha256: NWAY_DRAW_LATENT_ARTIFACT.sha256,
    selectorVersion: NWAY_SELECTOR_VERSION,
    particleProfileVersion: NWAY_PARTICLE_PROFILE,
    engineAlgorithmVersion: NWAY_ENGINE_ALGORITHM,
    candidateBankSha256,
    responseAggregation: "draw_latent",
  };
}

const SHA256 = /^[a-f0-9]{64}$/;

/** Server-authoritative aggregation; absent means the shipping mixture. */
export function nwayResponseAggregationOf(
  inputs: ComputeEngineInputs,
): NWayResponseAggregation {
  const aggregation = inputs.nwayProfile?.responseAggregation ?? "mixture";
  if (aggregation !== "mixture" && aggregation !== "draw_latent") {
    throw new Error("n-way profile mismatch: responseAggregation");
  }
  return aggregation;
}

export function validateNWayInputs(inputs: ComputeEngineInputs): NWayProfileStamp {
  const stamp = inputs.nwayProfile;
  if (!stamp) throw new Error("production session is missing its n-way profile stamp");
  const expected = nwayResponseAggregationOf(inputs) === "draw_latent"
    ? expectedDrawLatentNWayProfile(stamp.candidateBankSha256)
    : expectedNWayProfile(stamp.candidateBankSha256);
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
