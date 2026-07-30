export type ResponseModelName = "binary_ovr_v1" | "iiic_conditional_f1_v1";
export type ResponseLink = "binary" | "categorical_f1";
export type ArtifactQualification = "exploratory_unqualified" | "qualified";

export interface ResponseGroup {
  id: string;
  link: ResponseLink;
  taskIndices: number[];
}

export interface ConditionalF1Artifact {
  schemaVersion: 1;
  artifactId: string;
  model: "iiic_conditional_f1_v1";
  qualification: ArtifactQualification;
  beta: number;
  distractorLapse: number;
  binaryLapse: 0.025;
  sha256: string;
  provenance: {
    source: string;
    fitSplit: string;
    dr07: "not_qualified" | "qualified";
  };
}

export interface ConditionalF1ArtifactDraw {
  beta: number;
  distractorLapse: number;
  weight: number;
}

export interface ConditionalF1ArtifactEnsemble {
  schemaVersion: 1;
  artifactId: string;
  model: "iiic_conditional_f1_v1_artifact_ensemble";
  qualification: ArtifactQualification;
  draws: readonly ConditionalF1ArtifactDraw[];
  binaryLapse: 0.025;
  sha256: string;
  provenance: {
    source: string;
    fitSplit: string;
    dr07: "not_qualified" | "qualified";
  };
}

export type ConditionalF1ResponseArtifact =
  | ConditionalF1Artifact
  | ConditionalF1ArtifactEnsemble;

export interface EngineProfile {
  schemaVersion: 1;
  engineProfileId: string;
  terminationPolicy: "precision_v1";
  responseModel: ResponseModelName;
  responseArtifactId: string | null;
  responseArtifactSha256: string | null;
  selectorVersion:
    | "binary_totalvar_v1"
    | "categorical_totalvar_v1"
    | "categorical_fisher_totalvar_v1";
  particleProfileVersion: string;
  engineAlgorithmVersion: string;
  candidateBankSha256: string;
  responseGroups: ResponseGroup[];
}

export interface ProtocolSegment {
  segmentIndex: number;
  segId: number;
  sMean: readonly number[];
  sSd: readonly number[];
  applicableTaskIdx: readonly number[];
}

interface ObservationBase {
  askedK: number;
  segmentIndex: number;
  rawPick: number;
}

export interface BinaryObservation extends ObservationBase {
  kind: "binary";
  y: 0 | 1;
}

export interface CategoricalF1Observation extends ObservationBase {
  kind: "categorical_f1";
  groupId: string;
  pickK: number;
}

export type Observation = BinaryObservation | CategoricalF1Observation;

export interface PriorPieces {
  K: number;
  sigmaInv: number[][];
  logDet: number;
  L: number[][];
}

export interface PriorPair {
  tPieces: PriorPieces;
  lPieces: PriorPieces;
}

export interface RejuvenationTelemetry {
  qIndex: number;
  acceptanceRate: number;
  distinctAncestors: number;
  distinctAncestorFraction: number;
}

export interface ProtocolParticleState {
  N: number;
  K: number;
  t: Float64Array;
  l: Float64Array;
  w: Float64Array;
  logPrior: Float64Array;
  logLik: Float64Array;
  history: Observation[];
  prior: PriorPair;
  lastRejuvenation?: RejuvenationTelemetry;
}

export interface Candidate {
  askedK: number;
  segmentIndex: number;
  segId: number;
  focalSignal: number;
  focalSignalSd: number;
}

export interface ChosenCandidate extends Candidate {
  loss: number;
}

export interface ProfileStamp {
  engineProfileId: string;
  responseModel: ResponseModelName;
  responseArtifactId: string | null;
  responseArtifactSha256: string | null;
  engineAlgorithmVersion: string;
  selectorVersion: EngineProfile["selectorVersion"];
  candidateBankSha256: string;
}
