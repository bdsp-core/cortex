import { precomputePriorPair } from "../../cortex_web/apps/web/engine/prior";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import type {
  ConditionalF1Artifact, ConditionalF1ArtifactEnsemble, EngineProfile,
  ProtocolParticleState, ProtocolSegment,
} from "../src/types";

export const K = 7;

export const artifact: ConditionalF1Artifact = {
  schemaVersion: 1,
  artifactId: "iiic-f1-stage2-exploratory",
  model: "iiic_conditional_f1_v1",
  qualification: "exploratory_unqualified",
  beta: 0.9912,
  distractorLapse: 0,
  binaryLapse: 0.025,
  sha256: "a".repeat(64),
  provenance: {
    source: "nway_link_rd stage2 exploratory fit",
    fitSplit: "exploratory; not promotion eligible",
    dr07: "not_qualified",
  },
};

export const nwayProfile: EngineProfile = {
  schemaVersion: 1,
  engineProfileId: "precision_nway_f1_protocol_v1",
  terminationPolicy: "precision_v1",
  responseModel: "iiic_conditional_f1_v1",
  responseArtifactId: artifact.artifactId,
  responseArtifactSha256: artifact.sha256,
  selectorVersion: "categorical_totalvar_v1",
  particleProfileVersion: "protocol_test",
  engineAlgorithmVersion: "nway_protocol_0.1.0",
  candidateBankSha256: "b".repeat(64),
  responseGroups: [
    { id: "spike", link: "binary", taskIndices: [0] },
    { id: "iiic", link: "categorical_f1", taskIndices: [1, 2, 3, 4, 5, 6] },
  ],
};

export const ensembleArtifact: ConditionalF1ArtifactEnsemble = {
  schemaVersion: 1,
  artifactId: "iiic-f1-crossfit-ensemble9-rd-20260720",
  model: "iiic_conditional_f1_v1_artifact_ensemble",
  qualification: "exploratory_unqualified",
  draws: [
    { beta: 0.8722308125877214, distractorLapse: 0, weight: 1 / 9 },
    { beta: 0.95165593450824, distractorLapse: 0, weight: 1 / 9 },
    { beta: 0.9765851716341936, distractorLapse: 0, weight: 1 / 9 },
    { beta: 0.9945610869560302, distractorLapse: 0, weight: 1 / 9 },
    { beta: 1.0173203953763694, distractorLapse: 0, weight: 1 / 9 },
    { beta: 1.0418629877991512, distractorLapse: 0.013049422609172988, weight: 1 / 9 },
    { beta: 1.0637853303885672, distractorLapse: 0, weight: 1 / 9 },
    { beta: 1.0915479400184864, distractorLapse: 0, weight: 1 / 9 },
    { beta: 1.1945942284733333, distractorLapse: 0.018631450805286682, weight: 1 / 9 },
  ],
  binaryLapse: 0.025,
  sha256: "0654fc210e67ece152dcaef6b40641fc9c94cb259d100ed3829d90224bc5ef8b",
  provenance: {
    source: "crossfit artifact test fixture",
    fitSplit: "nine deterministic beta-order quantiles",
    dr07: "not_qualified",
  },
};

export const integratedNwayProfile: EngineProfile = {
  ...nwayProfile,
  engineProfileId: "precision_nway_f1_integrated_test",
  responseArtifactId: ensembleArtifact.artifactId,
  responseArtifactSha256: ensembleArtifact.sha256,
  selectorVersion: "categorical_fisher_totalvar_v1",
  engineAlgorithmVersion: "nway_protocol_0.2.0-rd",
};

export const binaryProfile: EngineProfile = {
  ...nwayProfile,
  engineProfileId: "precision_binary_protocol_v1",
  responseModel: "binary_ovr_v1",
  responseArtifactId: null,
  responseArtifactSha256: null,
  selectorVersion: "binary_totalvar_v1",
  responseGroups: Array.from({ length: K }, (_, taskK) => ({
    id: `binary-${taskK}`,
    link: "binary" as const,
    taskIndices: [taskK],
  })),
};

export function segments(count = 4): ProtocolSegment[] {
  return Array.from({ length: count }, (_, segmentIndex) => ({
    segmentIndex,
    segId: 1000 + segmentIndex,
    sMean: Array.from({ length: K }, (_, k) => -1.2 + 0.35 * k + 0.1 * segmentIndex),
    sSd: Array.from({ length: K }, (_, k) => 0.03 + 0.01 * ((k + segmentIndex) % 3)),
    applicableTaskIdx: segmentIndex % 2 === 0 ? [0] : [1, 2, 3, 4, 5, 6],
  }));
}

export function manualState(particleCount = 5): ProtocolParticleState {
  const identity = Array.from({ length: K }, (_, i) => (
    Array.from({ length: K }, (_, j) => i === j ? 1 : 0)
  ));
  const prior = precomputePriorPair(identity);
  const t = new Float64Array(particleCount * K);
  const l = new Float64Array(particleCount * K);
  for (let n = 0; n < particleCount; n += 1) {
    for (let k = 0; k < K; k += 1) {
      t[n * K + k] = (n - 2) * 0.18 + (k - 3) * 0.03;
      l[n * K + k] = (n - 2) * 0.22 - (k - 3) * 0.025;
    }
  }
  return {
    N: particleCount,
    K,
    t,
    l,
    w: new Float64Array(particleCount).fill(1 / particleCount),
    logPrior: new Float64Array(particleCount),
    logLik: new Float64Array(particleCount),
    history: [],
    prior,
  };
}

export function randomState(particleCount = 128): ProtocolParticleState {
  const state = manualState(particleCount);
  const rng = new Rng(8137);
  for (let i = 0; i < state.t.length; i += 1) {
    state.t[i] = rng.gaussian();
    state.l[i] = rng.gaussian();
  }
  return state;
}
