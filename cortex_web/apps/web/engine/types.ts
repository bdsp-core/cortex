// Shared engine types.

export interface SegmentMeta {
  segId: number;
  patternClass: string; // "spike" | "seizure" | "lpd" | "gpd" | "lrda" | "grda" | "other"
  testClass?: "iiic" | "spike"; // K=7 manifests carry this; pre-K=7 bundles omit it
  // Indices of tasks this segment can serve as a candidate for. IIIC segments
  // → [1..6]; spike segments → [0]. Pre-K=7 bundles omit this — the engine
  // falls back to "applies to all K tasks" for backward compat.
  applicableTaskIdx?: number[];
  sMean: number[]; // length K, per-task signal mean (sentinel 0 at inapplicable idx)
  sSd: number[]; // length K, per-task signal posterior SD (sentinel 0 at inapplicable idx)
  fsHz: number;
  nCh: number;
  nSamp: number;
  channelNames: string[];
  eeg: string; // relative path / key of the int16 EEG blob
  spec: string; // relative path / key of the spectrogram blob
}

export interface EngineInputs {
  taskCodes: string[]; // K=7: ["spike","sz","lpd","gpd","lrda","grda","iic"]
  taskLabels: string[]; // K=7: ["Spike","Seizure","LPD","GPD","LRDA","GRDA","Other"]
  taskPatternWords: string[]; // K=7: ["spike","seizure","lpd","gpd","lrda","grda","other"]
  // Per-task UI routing class. "spike" → SpikeViewer (binary mark-or-not);
  // "iiic" → the 6-button IIIC viewer. Optional for pre-K=7 bundle compat.
  taskClasses?: ("iiic" | "spike")[];
  // cert_config block the bundle was built against ("ell_star_unified_v14"
  // default; "v13" for the old K=6 bundles, "v15" for the post-pilot freeze).
  certBlock?: string;
  corrL: number[][]; // (K×K) fitted prior correlation (l-block; default both blocks)
  // OPT-IN (v15 staging). When present, the t-block uses this separate K×K
  // correlation instead of corrL; the l-block ALWAYS keeps corrL. Absent on the
  // frozen-pilot manifest, where t- and l-blocks share corrL (bit-identical).
  corrT?: number[][]; // (K×K) fitted t-block prior correlation (v15)
  // OPT-IN (v15 staging). Particle count; defaults to N_PARTICLES (600) when
  // absent. v15 manifests carry 1200.
  nParticles?: number;
  // OPT-IN (v1.6 adaptive termination). Per-domain question budget: a task
  // still PENDING after this many of its own questions is REFERred. Absent →
  // engine default PER_DOMAIN_CAP. Emitted by the v15+ bundle build.
  perDomainCap?: number;
  // Server-authoritative stopping-policy stamp. It is written when a sitting
  // is created and returned unchanged on resume; clients never choose it.
  // Absent on legacy/local fixtures => AD6 (the public rollback/default path).
  terminationPolicy?: TerminationPolicyName;
  // Frozen full-bank signal terciles used by precision_v1's content floor.
  // The API derives these from the complete served manifest before drawing the
  // per-session subset, so a participant's random draw cannot move the floor.
  precisionBandEdges?: number[][];
  ellStar: number[]; // (K) Youden cut-scores from the cert_config block
  segments: SegmentMeta[];
}

// Particle cloud. t,l are row-major Float64Array(N*K).
export interface ParticleState {
  N: number;
  K: number;
  t: Float64Array;
  l: Float64Array;
  w: Float64Array; // (N) normalized weights
  logPrior: Float64Array; // (N)
  logLik: Float64Array; // (N)
  history: { k: number; s: number; y: 0 | 1; sSd: number }[];
  // Separate prior pieces for the t- and l-blocks. On the frozen-pilot path
  // both are precomputePrior(corrL) and the computation is bit-identical to the
  // single-PriorPieces era; on the v15 path tPieces uses corrT.
  prior: PriorPair;
  // Most recent resample/rejuvenation event. PrecisionPolicy reports these
  // diagnostics (acceptance 0.20 / ancestry 0.35); quantile_mcse reliability
  // remains the load-bearing guard in the frozen profile.
  lastRejuvenation?: RejuvenationTelemetry;
}

export type TerminationPolicyName = "ad6" | "precision_v1";

export interface RejuvenationTelemetry {
  qIndex: number;
  acceptanceRate: number;
  distinctAncestors: number;
  distinctAncestorFraction: number;
}

export interface PriorPieces {
  K: number;
  sigmaInv: number[][]; // K×K
  logDet: number; // log|Σ|
  L: number[][]; // K×K Cholesky factor (lower)
}

// t-/l-block prior pieces. Pilot path: tPieces === precomputePrior(corrL) and
// lPieces === precomputePrior(corrL). v15 path: tPieces uses corrT.
export interface PriorPair {
  tPieces: PriorPieces;
  lPieces: PriorPieces;
}

export interface TrialDiag {
  trialIndex: number;
  taskK: number;
  segId: number;
  s: number;
  sSd: number;
  y: 0 | 1;
  ess: number;
  rejuv: boolean;
  pi: number[]; // per-task pass-mass
  mcse: number[];
  R: number[]; // per-task info gate
  verdicts: string[];
  nPerTask: number[];
  tMean: number[];
  lMean: number[];
  aurocHw: number[]; // per-task AUROC credible halfwidth (for the collapse video)
  terminationPolicy?: TerminationPolicyName;
  domainStatuses?: string[];
  determinations?: string[];
  terminalReasons?: (string | null)[];
  precisionStreakCounts?: number[];
  skillIntervals?: [number, number][];
  biasIntervals?: [number, number][];
  skillPointCenteredRadius?: number[];
  skillPointCenteredRadiusMcse?: number[];
  guardedPrecisionStatistic?: number[];
  skillTolerance?: number[];
  lastRejuvenation?: RejuvenationTelemetry;
}
