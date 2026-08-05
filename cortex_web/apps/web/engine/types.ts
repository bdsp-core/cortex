// Shared engine types.

/** The only per-question fields consumed by adaptive engine mathematics. */
export interface ComputeSegmentMeta {
  segId: number;
  // Indices of tasks this segment can serve as a candidate for. IIIC segments
  // → [1..6]; spike segments → [0]. Pre-K=7 bundles omit this and the
  // engine falls back to all K tasks.
  applicableTaskIdx?: number[];
  sMean: number[];
  sSd: number[];
}

/** Full participant-facing metadata retained by Bundle for rendering. */
export interface SegmentMeta extends ComputeSegmentMeta {
  patternClass: string; // "spike" | "seizure" | "lpd" | "gpd" | "lrda" | "grda" | "other"
  testClass?: "iiic" | "spike"; // K=7 manifests carry this; pre-K=7 bundles omit it
  fsHz: number;
  nCh: number;
  nSamp: number;
  channelNames: string[];
  eeg: string; // relative path / key of the int16 EEG blob
  spec: string; // relative path / key of the spectrogram blob
}

export interface EngineInputs<S extends ComputeSegmentMeta = SegmentMeta> {
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
  // Server-authoritative precision_v1 stopping recalibration stamp ("c1" =
  // evidence floor n_min 20->0 + declaration persistence 2->3, qualified by
  // the 2026-08 nmin-stopping-study). Written at sitting creation, persisted,
  // and returned unchanged on resume/replay; clients never choose it. Absent
  // => the shipped 20/2 configuration, byte-identical to pre-c1 behavior.
  precisionRecalibration?: "c1";
  // Server-authoritative graded bias-flag reporting stamp
  // (CORTEX_BIAS_FLAG_TIERS). "all" => the session summary carries the graded
  // WATCH/EXTREME/EXTREME_CONFIRMED flags plus withheld reasons; absent or any
  // other value => the historical interval-clears flags, byte-identical to
  // pre-tier behavior (fail-closed). Report-only: never read by stopping,
  // selection, or verdicts.
  biasFlagTiers?: "legacy" | "all";
  // Server-authoritative response/selector profile. New production sittings
  // carry this exact stamp; it is persisted with the sitting and checked on
  // resume/result ingest so an engine upgrade cannot silently reinterpret an
  // answer stream.
  nwayProfile?: NWayProfileStamp;
  // Frozen full-bank signal terciles used by precision_v1's content floor.
  // The API derives these from the complete served manifest before drawing the
  // per-session subset, so a participant's random draw cannot move the floor.
  precisionBandEdges?: number[][];
  ellStar: number[]; // (K) Youden cut-scores from the cert_config block
  // Computation-only metadata. The UI's Bundle retains the corresponding full
  // SegmentMeta records (EEG/spec paths, rendering shape, labels) by segId.
  segments: S[];
}

export type ComputeEngineInputs = EngineInputs<ComputeSegmentMeta>;

export const NWAY_RESPONSE_MODEL = "iiic_conditional_f1_v1" as const;
export const NWAY_SELECTOR_VERSION = "categorical_fisher_totalvar_v1" as const;

/**
 * How the response-artifact ensemble enters the categorical likelihood.
 *
 * - "mixture": every particle scores the pick under the fixed-weight average
 *   across draws (the shipping behavior; absent means "mixture" so every
 *   existing stamp is untouched).
 * - "draw_latent": each particle carries an artifact-atom index and scores the
 *   pick under its own atom's (beta, distractorLapse); atom indices ride
 *   resampling as lineage and never move during MH rejuvenation
 *   (construction B, ported from n-way-protocol draw_latent_rd/engine.py).
 */
export type NWayResponseAggregation = "mixture" | "draw_latent";

/** One artifact atom with its observation-invariant log-domain terms
 * precomputed (identical arithmetic to the frozen history-draw preparation). */
export interface NWayPreparedAtom {
  beta: number;
  distractorLapse: number;
  weight: number;
  logWeight: number;
  uniformLogProbability: number;
  directedLogProbability: number;
}

/**
 * Profile-resolved response-model runtime. Absent on a ParticleState means the
 * frozen NWAY_ARTIFACT mixture path — byte-identical to the shipped engine.
 * Present, it names the artifact whose atoms the session scores under:
 * aggregation "draw_latent" scores each particle under its own atom (the state
 * must carry atomIndex lineage); aggregation "mixture" mixes the same table
 * with fixed weights (the degenerate-reduction test harness path).
 */
export interface NWayResponseRuntime {
  aggregation: NWayResponseAggregation;
  artifactId: string;
  atoms: readonly NWayPreparedAtom[];
  /** Single moment-matched draw used only by the cheap Fisher shortlist screen. */
  screen: readonly { beta: number; distractorLapse: number; weight: number }[];
}

export interface NWayProfileStamp {
  engineProfileId: string;
  responseModel: typeof NWAY_RESPONSE_MODEL;
  responseArtifactId: string;
  responseArtifactSha256: string;
  selectorVersion: typeof NWAY_SELECTOR_VERSION;
  particleProfileVersion: string;
  engineAlgorithmVersion: string;
  candidateBankSha256: string;
  /** Absent means "mixture", so every existing stamp is untouched. */
  responseAggregation?: NWayResponseAggregation;
}

export interface BinaryParticleObservation {
  kind: "binary";
  k: number;
  s: number;
  sSd: number;
  y: 0 | 1;
  rawPick: number;
}

export interface CategoricalParticleObservation {
  kind: "categorical_f1";
  askedK: number;
  pickK: number;
  sMean: number[];
  sSd: number[];
}

export type ParticleObservation =
  | BinaryParticleObservation
  | CategoricalParticleObservation;

/** Derived structure-of-arrays cache for allocation-free history replay.
 * `history` remains authoritative and snapshot-safe; this cache is rebuilt
 * after restore and never changes observation order or arithmetic. */
export interface PackedParticleHistory {
  K: number;
  length: number;
  capacity: number;
  kind: Uint8Array;
  taskK: Int8Array;
  pick: Int8Array;
  binaryS: Float64Array;
  binarySd: Float64Array;
  signalMean: Float64Array;
  signalSd: Float64Array;
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
  history: ParticleObservation[];
  packedHistory?: PackedParticleHistory;
  // Separate prior pieces for the t- and l-blocks. On the frozen-pilot path
  // both are precomputePrior(corrL) and the computation is bit-identical to the
  // single-PriorPieces era; on the v15 path tPieces uses corrT.
  prior: PriorPair;
  // Most recent resample/rejuvenation event. PrecisionPolicy reports these
  // diagnostics (acceptance 0.20 / ancestry 0.35); quantile_mcse reliability
  // remains the load-bearing guard in the frozen profile.
  lastRejuvenation?: RejuvenationTelemetry;
  // Draw-latent aggregation (construction B): per-particle artifact-atom
  // lineage; entry n indexes responseRuntime.atoms for particle n. Present
  // exactly when responseRuntime.aggregation is "draw_latent"; absent on every
  // mixture-path session so the shipped state shape is untouched.
  atomIndex?: Int32Array;
  // Profile-resolved response-model runtime, shared immutably across clones
  // like `prior`. Absent → the frozen NWAY_ARTIFACT mixture path.
  responseRuntime?: NWayResponseRuntime;
}

export type TerminationPolicyName = "ad6" | "precision_v1";

export interface RejuvenationTelemetry {
  qIndex: number;
  acceptanceRate: number;
  distinctAncestors: number;
  distinctAncestorFraction: number;
}

/** Schema-v2 wall-clock attribution. These values are observational only and
 * must never be read by inference, selection, stopping, or RNG code. */
export interface SelectionPhaseTimingV2 {
  candidatePreparationMs: number;
  posteriorMomentsMs: number;
  coarseMinSdScanMs: number;
  entropyScanMs: number;
  fisherScanMs: number;
  exactRefinementMs: number;
  candidateCount: number;
  shortlistCount: number;
}

export interface ParticlePhaseTimingV2 {
  categoricalUpdateMs: number;
  essMs: number;
  resamplingMs: number;
  mhProposalGenerationMs: number;
  mhPriorMs: number;
  mhHistoryLikelihoodMs: number;
  mhAcceptanceMs: number;
  mhCopyingMs: number;
}

export interface BranchLifecycleV2 {
  outcome: number;
  rank: number;
  probability: number;
  role: "coordinator" | "coordinator_expansion" | "helper" | "serial_required";
  queued: boolean;
  started: boolean;
  readyAtAnswer: boolean;
  adopted: boolean;
  cancelled: boolean;
  cancellationPhase?: import("./speculation_cancellation").SpeculationCancellationPhase;
  /** The exact update crossed the frozen ESS threshold, so optional ranked
   * work stopped before MH and the observed branch retained all 30 steps. */
  deferredForRejuvenation?: boolean;
  discarded: boolean;
  durationMs: number | null;
}

export interface EngineStepPhaseTimingV2 {
  selection: SelectionPhaseTimingV2;
  particle: ParticlePhaseTimingV2;
  observedOutcomeRank: number | null;
  observedOutcomeProbability: number | null;
  /** Time from the participant's main-realm submission until the coordinator
   * could service the answer message. Telemetry only. */
  answerDispatchDelayMs: number;
  cachedProbabilityMass: number;
  branches: BranchLifecycleV2[];
}

/** Non-deterministic wall-clock attribution, never part of policy state. */
export interface EngineStepTiming {
  kind: "engine_step";
  trialIndex: number;
  y: 0 | 1;
  pick: number;
  rejuvenated: boolean;
  bankPreparationMs: number;
  updateMs: number;
  rejuvenationMs: number;
  bookkeepingMs: number;
  policyMs: number;
  diagnosticsMs: number;
  selectionMs: number;
  totalMs: number;
  executionMode: ComputeExecutionMode;
  speculative: boolean;
  requiredBranchReadyAtAnswer: boolean;
  phaseV2?: EngineStepPhaseTimingV2;
}

export interface AnswerToItemTiming {
  kind: "answer_to_item";
  trialIndex: number;
  durationMs: number;
}

export interface AnswerToMediaReadyTiming {
  kind: "answer_to_media_ready";
  trialIndex: number;
  durationMs: number;
}

export interface ExecutionProfileEvent {
  kind: "execution_profile";
  requested: RequestedComputeMode;
  executionMode: Exclude<ComputeExecutionMode, "serial_fallback">;
  reason: string;
  hardwareConcurrency: number | null;
  selectedWorkerCount?: number;
  calibrationResult?: string;
  estimatedWorkerMemoryBytes?: number;
}

export interface EventLoopHeartbeatEvent {
  kind: "event_loop_heartbeat";
  sampleCount: number;
  meanDelayMs: number;
  maxDelayMs: number;
}

/** Historical schema-v2 event retained for reading results from releases that
 * resized the pool at runtime. Current sessions never emit this event. */
export interface RuntimePoolAdjustmentEvent {
  kind: "runtime_pool_adjustment";
  trialIndex: number;
  previousWorkerCount: number;
  selectedWorkerCount: number;
  reason: "main_realm_load";
  trigger: "sustained_mean" | "repeated_max";
  evidenceWindowCount: number;
  sampleCount: number;
  meanDelayMs: number;
  maxDelayMs: number;
}

export type EnginePerformanceEvent =
  | EngineStepTiming
  | AnswerToItemTiming
  | AnswerToMediaReadyTiming
  | ExecutionProfileEvent
  | EventLoopHeartbeatEvent
  | RuntimePoolAdjustmentEvent;

export type RequestedComputeMode = "serial" | "dual_branch_auto";
export type ComputeExecutionMode =
  | "serial" | "dual_branch" | "adaptive_pool" | "serial_fallback";

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
  // Raw task-axis response. For IIIC this is the retained six-way category;
  // for spike it is task 0 (yes) or the K sentinel (no).
  pick: number;
  responseKind: "binary" | "categorical_f1";
  matchedAskedTask: boolean;
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
  // Distractor-misspecification CUSUM after this trial (n-way sessions only).
  distractorMonitor?: {
    statistic: number;
    tripped: boolean;
    wrongPicks: number;
    trippedAt: number | null;
  };
}
