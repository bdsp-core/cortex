// Fully local session runtime for the draw-latent n-way engine.
//
// One LocalSession is one adaptive sitting driven end-to-end through the
// isolated TS engine, read-only:
//   - cloud creation      src/particles.ts makeProtocolState (draw_latent)
//   - item selection      src/integrated_protocol.ts chooseProtocolCandidate
//   - posterior update    src/session.ts advanceProtocol (update + ESS-gated
//                         resample/rejuvenate + direct-evidence ledger)
//   - stopping            the UNCHANGED production PrecisionPolicy, evaluated
//                         in-process through src/precision_bridge.ts exactly
//                         as the qualification sidecar does (per-domain cap
//                         60, nMin 20, band edges = full-served-bank terciles
//                         mirroring qualification.py _tercile_band_edges).
//
// Simulated readers answer with the conditional-F1 truth model, mirroring
// draw_latent_rd/harness.py: pick ~ f1_probabilities(truth_t, truth_l, s,
// s_sd, asked_k, IIIC_GROUP, beta, lapse), sampled from a dedicated response
// RNG stream.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import type { PolicyResult } from "../../cortex_web/apps/web/engine/policy";
import {
  PRECISION_STATUS, PRECISION_TASK_CODES, PrecisionPolicy, weightedQuantile,
} from "../../cortex_web/apps/web/engine/precision_policy";
import { precomputePriorPair } from "../../cortex_web/apps/web/engine/prior";
import { Rng } from "../../cortex_web/apps/web/engine/rng";
import type { ComputeEngineInputs } from "../../cortex_web/apps/web/engine/types";
import { chooseProtocolCandidate } from "../src/integrated_protocol";
import { responseProbabilities } from "../src/likelihood";
import { atomPosterior, makeProtocolState, posteriorMoments } from "../src/particles";
import { evaluateFrozenPrecision } from "../src/precision_bridge";
import { validateProfile } from "../src/profile";
import { advanceProtocol, makeLedger, type DirectEvidenceLedger } from "../src/session";
import type {
  Candidate, ChosenCandidate, ConditionalF1Artifact, ConditionalF1ArtifactEnsemble,
  EngineProfile, ProtocolParticleState, ProtocolSegment,
} from "../src/types";

export const K = 7;
export const IIIC_GROUP = [1, 2, 3, 4, 5, 6] as const;
export const TASK_LABELS = [
  "Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "IIC/Other",
] as const;

export const PER_DOMAIN_CAP = 60;
export const N_MIN = 20;
export const ESS_THRESHOLD_FRACTION = 0.5;
export const PROPOSAL_SCALE = 2.38 / Math.sqrt(2 * K);

// N(0,1) terciles: structural placeholder for a domain with no servable
// signals (the spike axis on the IIIC-only bank). Mirrors qualification.py
// _PLACEHOLDER_EDGES; such a domain has zero candidates and terminalizes as
// UNDETERMINABLE_BANK on the first evaluation.
export const PLACEHOLDER_EDGES: readonly [number, number] =
  [-0.43072729929545744, 0.43072729929545744];

function sha256Hex(bytes: Buffer): string {
  return createHash("sha256").update(bytes).digest("hex");
}

// NumPy default linear quantile: index (n-1)q with interpolation, matching
// session_bank.py:_linear_quantile and the policy's own deriveBandEdges.
function linearQuantile(values: readonly number[], q: number): number {
  const ordered = values.slice().sort((a, b) => a - b);
  const h = (ordered.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]);
}

export interface ServedBankRow {
  segId: number;
  sMean: number[];
  sSd: number[];
}

export interface ServedBank {
  path: string;
  sha256: string;
  rows: ServedBankRow[];
  /** Full-bank signal terciles per domain (the production content-floor edges). */
  bandEdges: [number, number][];
}

/** Load .artifacts/categorical_bank_axes.csv (stage_real_bank.py schema). */
export function loadServedBank(path: string): ServedBank {
  const bytes = readFileSync(path);
  const lines = bytes.toString("utf8").split("\n").filter((line) => line.trim().length > 0);
  const header = "segment_index,seg_id,"
    + "s_mean_0,s_mean_1,s_mean_2,s_mean_3,s_mean_4,s_mean_5,s_mean_6,"
    + "s_sd_0,s_sd_1,s_sd_2,s_sd_3,s_sd_4,s_sd_5,s_sd_6";
  if (lines[0].trim() !== header) {
    throw new Error(`unexpected served-bank header in ${path}`);
  }
  const rows: ServedBankRow[] = [];
  for (const line of lines.slice(1)) {
    const parts = line.split(",");
    if (parts.length !== 16) throw new Error(`malformed served-bank row: ${line}`);
    const numbers = parts.map(Number);
    rows.push({
      segId: numbers[1],
      sMean: numbers.slice(2, 9),
      sSd: numbers.slice(9, 16),
    });
  }
  const bandEdges: [number, number][] = [];
  for (let k = 0; k < K; k += 1) {
    const finite = rows.map((row) => row.sMean[k]).filter(Number.isFinite);
    if (finite.length < 3) {
      bandEdges.push([...PLACEHOLDER_EDGES]);
      continue;
    }
    const q1 = linearQuantile(finite, 1 / 3);
    const q2 = linearQuantile(finite, 2 / 3);
    bandEdges.push(q1 < q2 ? [q1, q2] : [...PLACEHOLDER_EDGES]);
  }
  return { path, sha256: sha256Hex(bytes), rows, bandEdges };
}

interface Atoms17Payload {
  bootstrap: { draws: { beta: number; distractor_lapse: number; weight: number }[] };
  model: string;
  variant?: string;
  status?: string;
  provenance?: { derivation?: string; dr07?: string };
}

/**
 * Load the 17-atom engine-frame artifact (Python emitter schema) as the TS
 * ensemble artifact. The file is research-only (promotionForbidden), so the
 * TS qualification field stays "exploratory_unqualified" and the profile is
 * validated with allowUnqualified=true — no promotion surface is touched.
 */
export function loadAtoms17Artifact(path: string): ConditionalF1ArtifactEnsemble {
  const bytes = readFileSync(path);
  const payload = JSON.parse(bytes.toString("utf8")) as Atoms17Payload;
  if (payload.model !== "iiic_conditional_f1_v1_artifact_ensemble") {
    throw new Error(`unexpected artifact model ${payload.model} in ${path}`);
  }
  const draws = payload.bootstrap.draws.map((draw) => ({
    beta: draw.beta,
    distractorLapse: draw.distractor_lapse,
    weight: draw.weight,
  }));
  return {
    schemaVersion: 1,
    artifactId: `iiic-f1-${(payload.variant ?? "engine-frame-atoms17").replace(/_/g, "-")}`,
    model: "iiic_conditional_f1_v1_artifact_ensemble",
    qualification: "exploratory_unqualified",
    draws,
    binaryLapse: 0.025,
    sha256: sha256Hex(bytes),
    provenance: {
      source: payload.provenance?.derivation ?? "engine-frame atoms17 artifact",
      fitSplit: payload.status ?? "research_only_not_promoted",
      dr07: payload.provenance?.dr07 === "qualified" ? "qualified" : "not_qualified",
    },
  };
}

export function buildLocalProfile(
  artifact: ConditionalF1ArtifactEnsemble,
  candidateBankSha256: string,
): EngineProfile {
  const profile: EngineProfile = {
    schemaVersion: 1,
    engineProfileId: "precision_nway_f1_draw_latent_atoms17_local_v1",
    terminationPolicy: "precision_v1",
    responseModel: "iiic_conditional_f1_v1",
    responseArtifactId: artifact.artifactId,
    responseArtifactSha256: artifact.sha256,
    selectorVersion: "categorical_fisher_totalvar_v1",
    particleProfileVersion: "local_1200p_ess050_30mh",
    engineAlgorithmVersion: "nway_protocol_0.2.0-rd",
    candidateBankSha256,
    responseGroups: [
      { id: "spike", link: "binary", taskIndices: [0] },
      { id: "iiic", link: "categorical_f1", taskIndices: [1, 2, 3, 4, 5, 6] },
    ],
    responseAggregation: "draw_latent",
  };
  validateProfile(profile, K, artifact, true);
  return profile;
}

export interface ReaderConfig {
  preset: string | null;
  beta: number;
  lapse: number;
  skillShift: number;
}

export const READER_PRESETS: Record<string, Omit<ReaderConfig, "preset">> = {
  sharp_expert: { beta: 1.6, lapse: 0, skillShift: 0 },
  typical: { beta: 1.0, lapse: 0, skillShift: 0 },
  diffuse: { beta: 0.7, lapse: 0, skillShift: 0 },
  guesser: { beta: 1.0, lapse: 1.0, skillShift: 0 },
};

export interface SelectorQuotas {
  fullScanLimit: number;
  coarsePerTask: number;
  entropyPerTask: number;
  fisherPerTask: number;
}

// Reduced shortlist quotas keep per-answer latency interactive; the exact
// total-posterior-variance loss that makes the final choice is unchanged.
export const DEFAULT_SELECTOR_QUOTAS: SelectorQuotas = {
  fullScanLimit: 512,
  coarsePerTask: 6,
  entropyPerTask: 3,
  fisherPerTask: 3,
};

export interface SessionOptions {
  mode: "manual" | "simulated";
  reader?: Partial<ReaderConfig> & { preset?: string };
  seed?: number;
  bankSegments?: number;
  particles?: number;
  mhSteps?: number;
  truthSd?: number;
  selector?: Partial<SelectorQuotas>;
}

export interface SessionConfig {
  mode: "manual" | "simulated";
  seed: number;
  bankSegments: number;
  particles: number;
  mhSteps: number;
  truthSd: number;
  selector: SelectorQuotas;
  perDomainCap: number;
  nMin: number;
}

function resolveReader(
  options: SessionOptions["reader"],
): ReaderConfig {
  const preset = options?.preset ?? null;
  if (preset !== null && !(preset in READER_PRESETS)) {
    throw new HttpError(400, `unknown reader preset ${preset}`);
  }
  const base = preset !== null
    ? READER_PRESETS[preset]
    : { beta: 1.0, lapse: 0, skillShift: 0 };
  const reader: ReaderConfig = {
    preset,
    beta: options?.beta ?? base.beta,
    lapse: options?.lapse ?? base.lapse,
    skillShift: options?.skillShift ?? base.skillShift,
  };
  if (!Number.isFinite(reader.beta) || reader.beta <= 0) {
    throw new HttpError(400, "reader beta must be finite and positive");
  }
  if (!Number.isFinite(reader.lapse) || reader.lapse < 0 || reader.lapse > 1) {
    throw new HttpError(400, "reader lapse must be in [0, 1]");
  }
  if (!Number.isFinite(reader.skillShift)) {
    throw new HttpError(400, "reader skill shift must be finite");
  }
  return reader;
}

export class HttpError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

interface TruthState {
  reader: ReaderConfig;
  t: Float64Array;
  l: Float64Array;
  artifact: ConditionalF1Artifact;
}

function truthArtifact(reader: ReaderConfig): ConditionalF1Artifact {
  return {
    schemaVersion: 1,
    artifactId: "local-simulated-truth",
    model: "iiic_conditional_f1_v1",
    qualification: "exploratory_unqualified",
    beta: reader.beta,
    distractorLapse: reader.lapse,
    binaryLapse: 0.025,
    sha256: "0".repeat(64),
    provenance: {
      source: "local simulated reader (harness.py truth model)",
      fitSplit: "synthetic; response generation only",
      dr07: "not_qualified",
    },
  };
}

export interface StepRecord {
  questionNumber: number;
  askedK: number;
  segId: number;
  rawPick: number;
  matchedAskedTask: boolean;
  ess: number;
  rejuvenated: boolean;
  stepMs: number;
}

/** Browser-measured answer timing (manual answers only; simulated = nulls). */
export interface ClientTiming {
  reactionMs: number | null;
  shownClientUtc: string | null;
  answeredClientUtc: string | null;
}

/**
 * The draw-latent measurement payload persisted INSIDE `trials.diag` (that is
 * what the production TEXT column carries): post-update engine state after
 * this trial's answer was absorbed.
 */
export interface TrialDiag {
  /** Post-update posterior mass over the 17 artifact atoms. */
  atom_posterior: number[];
  atom_map_beta: number;
  atom_mean_beta: number;
  /** Per-domain (K=7) posterior mean + equal-tailed 95% CI. */
  skill: { mean: number; ci95: [number, number] }[];
  bias: { mean: number; ci95: [number, number] }[];
  ess: number;
  /** ESS-gated resample/rejuvenate fired on this update. */
  resampled: boolean;
  /** Per-domain (K=7) Precision selection states after this answer. */
  precision_statuses: string[];
  policy_stop: boolean;
}

/** One trial in production `trials` semantics, handed to the persistence hooks. */
export interface TrialCapture {
  /** 0-based, matching the production (session_id, trial_index) key. */
  trialIndex: number;
  /** REAL bank segment id. */
  segId: number;
  /** Asked class (= the presented item's class: the local-gold convention). */
  taskK: number;
  pick: number;
  /** 1 iff pick === taskK (local-gold convention, see README). */
  isCorrect: 0 | 1;
  reactionMs: number | null;
  shownClientUtc: string | null;
  answeredClientUtc: string | null;
  diag: TrialDiag;
}

export interface PersistenceHooks {
  onCreate(session: LocalSession): void;
  onTrial(session: LocalSession, capture: TrialCapture): void;
  onStop(session: LocalSession): void;
}

export interface DomainStatusView {
  k: number;
  code: string;
  label: string;
  status: string;
  terminalReason: string | null;
  n: number;
  streak: number;
  skill: { mean: number; ci95: [number, number] };
  bias: { mean: number; ci95: [number, number] };
  radius: number | null;
  guardedRadius: number | null;
  tolerance: number | null;
}

export interface SessionStatusView {
  sessionId: string;
  mode: "manual" | "simulated";
  config: SessionConfig;
  reader: ReaderConfig | null;
  stopped: boolean;
  stopReason: string | null;
  questionsAsked: number;
  remainingSegments: number;
  currentItem: {
    questionNumber: number;
    askedK: number;
    askedClass: string;
    segId: number;
    focalSignal: number;
    segmentSignals: { k: number; label: string; sMean: number; sSd: number }[];
  } | null;
  nPerTask: number[];
  domains: DomainStatusView[];
  atomPosterior: {
    betas: number[];
    lapses: number[];
    mass: number[];
    mapBeta: number;
    meanBeta: number;
  };
  policy: {
    stop: boolean;
    stopReason: string;
    selectionStates: string[];
    ess: number | null;
  } | null;
  history: StepRecord[];
  timing: { totalMs: number; lastStepMs: number | null };
}

let sessionCounter = 0;

export class LocalSession {
  readonly id: string;
  readonly config: SessionConfig;
  private readonly profile: EngineProfile;
  private readonly artifact: ConditionalF1ArtifactEnsemble;
  private readonly segments: ProtocolSegment[];
  private readonly candidates: Candidate[];
  private readonly state: ProtocolParticleState;
  private readonly ledger: DirectEvidenceLedger;
  private readonly policy: PrecisionPolicy;
  private readonly mhRng: Rng;
  private readonly responseRng: Rng;
  private readonly bandEdges: [number, number][];
  private truth: TruthState | null;
  private lastEvaluation: PolicyResult | null = null;
  private currentItem: ChosenCandidate | null = null;
  private trialIndex = 0;
  private stopped = false;
  private stopReason: string | null = null;
  private readonly history: StepRecord[] = [];
  private totalMs = 0;
  private lastStepMs: number | null = null;
  busy = false;

  constructor(
    bank: ServedBank,
    artifact: ConditionalF1ArtifactEnsemble,
    profile: EngineProfile,
    options: SessionOptions,
    private readonly hooks: PersistenceHooks | null = null,
  ) {
    sessionCounter += 1;
    this.id = `local-${Date.now().toString(36)}-${sessionCounter}`;
    if (options.mode !== "manual" && options.mode !== "simulated") {
      throw new HttpError(400, "mode must be \"manual\" or \"simulated\"");
    }
    this.config = {
      mode: options.mode,
      seed: options.seed ?? (Date.now() % 2_147_483_647),
      bankSegments: options.bankSegments ?? 420,
      particles: options.particles ?? 1200,
      mhSteps: options.mhSteps ?? 30,
      truthSd: options.truthSd ?? 1.0,
      selector: { ...DEFAULT_SELECTOR_QUOTAS, ...options.selector },
      perDomainCap: PER_DOMAIN_CAP,
      nMin: N_MIN,
    };
    if (!Number.isInteger(this.config.bankSegments) || this.config.bankSegments < 60
      || this.config.bankSegments > bank.rows.length) {
      throw new HttpError(
        400, `bankSegments must be an integer in [60, ${bank.rows.length}]`,
      );
    }
    if (!Number.isInteger(this.config.particles) || this.config.particles < 64
      || this.config.particles > 4800) {
      throw new HttpError(400, "particles must be an integer in [64, 4800]");
    }
    if (!Number.isInteger(this.config.mhSteps) || this.config.mhSteps < 1
      || this.config.mhSteps > 60) {
      throw new HttpError(400, "mhSteps must be an integer in [1, 60]");
    }
    if (!Number.isFinite(this.config.truthSd) || this.config.truthSd <= 0) {
      throw new HttpError(400, "truthSd must be finite and positive");
    }
    this.artifact = artifact;
    this.profile = profile;
    this.bandEdges = bank.bandEdges.map((pair) => [pair[0], pair[1]]);

    // Seeded per-session draw from the served bank, mirroring qualification.py
    // real-bank sub-sampling (the content-floor edges above stay full-bank).
    const seed = this.config.seed;
    const bankRng = new Rng(seed + 80_000);
    const indices = Array.from({ length: bank.rows.length }, (_, i) => i);
    for (let i = 0; i < this.config.bankSegments; i += 1) {
      const j = i + Math.floor(bankRng.random() * (indices.length - i));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const chosen = indices.slice(0, this.config.bankSegments).sort((a, b) => a - b);
    this.segments = chosen.map((rowIndex, segmentIndex) => ({
      segmentIndex,
      segId: bank.rows[rowIndex].segId,
      sMean: bank.rows[rowIndex].sMean,
      sSd: bank.rows[rowIndex].sSd,
      applicableTaskIdx: [...IIIC_GROUP],
    }));
    this.candidates = this.segments.flatMap((segment) => IIIC_GROUP.map((askedK) => ({
      askedK,
      segmentIndex: segment.segmentIndex,
      segId: segment.segId,
      focalSignal: segment.sMean[askedK],
      focalSignalSd: segment.sSd[askedK],
    })));

    const identity = Array.from({ length: K }, (_, i) => (
      Array.from({ length: K }, (_, j) => (i === j ? 1 : 0))
    ));
    const prior = precomputePriorPair(identity);
    this.state = makeProtocolState(
      this.config.particles, K, prior, new Rng(seed + 20_000),
      { responseAggregation: "draw_latent", artifact },
    );
    this.mhRng = new Rng(seed + 60_000);
    this.responseRng = new Rng(seed + 40_000);
    this.ledger = makeLedger(K, this.segments.map((segment) => segment.segId));

    // The UNCHANGED production stopping policy, constructed exactly as the
    // qualification sidecar does (src/precision_cli.ts frozenInputs).
    const inputs: ComputeEngineInputs = {
      taskCodes: [...PRECISION_TASK_CODES],
      taskLabels: ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
      taskPatternWords: ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"],
      corrL: identity,
      corrT: identity,
      precisionBandEdges: this.bandEdges.map((pair) => [...pair]),
      ellStar: new Array(K).fill(0),
      segments: [],
    };
    this.policy = PrecisionPolicy.fromInputs(inputs, PER_DOMAIN_CAP, N_MIN);
    this.policy.reset(K);

    if (options.mode === "simulated") {
      this.truth = this.makeTruth(resolveReader(options.reader));
    } else {
      this.truth = null;
    }
    this.currentItem = this.selectNext();
    this.hooks?.onCreate(this);
    // Degenerate zero-question stop (a draw with no servable candidates).
    if (this.stopped) this.hooks?.onStop(this);
  }

  /** Production `sessions.participant` value for this local sitting. */
  get participantLabel(): string {
    if (this.config.mode === "manual") return "owner-local";
    const truth = this.truth;
    return `sim:${truth?.reader.preset ?? "custom"}`;
  }

  /** REAL bank seg ids of this session's seeded draw (drawn_seg_ids). */
  get drawnSegIds(): number[] {
    return this.segments.map((segment) => segment.segId);
  }

  get stoppedReason(): string | null {
    return this.stopReason;
  }

  get questionsAsked(): number {
    return this.trialIndex;
  }

  private makeTruth(reader: ReaderConfig): TruthState {
    const truthRng = new Rng(this.config.seed + 10_000);
    const t = new Float64Array(K);
    const l = new Float64Array(K);
    for (let k = 0; k < K; k += 1) {
      t[k] = this.config.truthSd * truthRng.gaussian();
      l[k] = reader.skillShift + this.config.truthSd * truthRng.gaussian();
    }
    t[0] = 0;
    l[0] = 0;
    return { reader, t, l, artifact: truthArtifact(reader) };
  }

  /** Attach a simulated reader to a manual session for autostep hand-off. */
  attachReader(options: SessionOptions["reader"]): void {
    if (this.truth !== null) {
      throw new HttpError(409, "session already has a simulated reader");
    }
    this.truth = this.makeTruth(resolveReader(options));
  }

  get hasReader(): boolean {
    return this.truth !== null;
  }

  get isStopped(): boolean {
    return this.stopped;
  }

  private activeDomains(): Set<number> {
    const states = this.lastEvaluation?.selectionStates;
    const active = new Set<number>();
    for (const k of IIIC_GROUP) {
      if (!states || states[k] === PRECISION_STATUS.ACTIVE) active.add(k);
    }
    return active;
  }

  private selectNext(): ChosenCandidate | null {
    const active = this.activeDomains();
    const available = this.candidates.filter((candidate) => (
      this.ledger.remainingSegmentIds.has(candidate.segId) && active.has(candidate.askedK)
    ));
    if (available.length === 0) {
      this.stopped = true;
      this.stopReason = "bank_exhausted_for_active_domains";
      return null;
    }
    return chooseProtocolCandidate(
      this.state, this.profile, this.artifact, available, this.segments,
      this.config.selector,
    );
  }

  answer(rawPick: number, clientTiming: ClientTiming | null = null): StepRecord {
    if (this.stopped || this.currentItem === null) {
      throw new HttpError(409, "session has already stopped");
    }
    if (!Number.isInteger(rawPick) || !(IIIC_GROUP as readonly number[]).includes(rawPick)) {
      throw new HttpError(400, "pick must be an integer class index in 1..6");
    }
    const startedAt = performance.now();
    const chosen = this.currentItem;
    const diagnostic = advanceProtocol({
      state: this.state,
      ledger: this.ledger,
      profile: this.profile,
      artifact: this.artifact,
      segments: this.segments,
      chosen,
      rawPick,
      trialIndex: this.trialIndex,
      rng: this.mhRng,
      params: {
        essThresholdFraction: ESS_THRESHOLD_FRACTION,
        nMhSteps: this.config.mhSteps,
        proposalScale: PROPOSAL_SCALE,
        bandEdges: this.bandEdges,
      },
      precisionPolicy: this.policy,
    });
    this.trialIndex += 1;

    // advance.ts:498-514 ordering: the administered item was recorded inside
    // advanceProtocol; the policy now sees the post-update cloud, incremented
    // counts, and the post-removal remaining bank (all IIIC domains,
    // regardless of selection state — matching _precision_bank_view).
    const evaluation = evaluateFrozenPrecision({
      policy: this.policy,
      state: this.state,
      nPerTask: this.ledger.nPerTask,
      remainingCandidates: this.candidates.filter((candidate) => (
        this.ledger.remainingSegmentIds.has(candidate.segId)
      )),
      remainingSegmentIds: this.ledger.remainingSegmentIds,
    });
    this.lastEvaluation = evaluation;
    if (evaluation.stop) {
      this.stopped = true;
      this.stopReason = evaluation.stopReason;
      this.currentItem = null;
    } else {
      this.currentItem = this.selectNext();
    }
    const stepMs = performance.now() - startedAt;
    this.totalMs += stepMs;
    this.lastStepMs = stepMs;
    const record: StepRecord = {
      questionNumber: this.trialIndex,
      askedK: chosen.askedK,
      segId: chosen.segId,
      rawPick,
      matchedAskedTask: diagnostic.matchedAskedTask,
      ess: diagnostic.ess,
      rejuvenated: diagnostic.rejuvenated,
      stepMs,
    };
    this.history.push(record);
    if (this.hooks) {
      const mass = atomPosterior(this.state, this.artifact);
      const betas = this.artifact.draws.map((draw) => draw.beta);
      const mapIndex = mass.indexOf(Math.max(...mass));
      const summaries = this.posteriorSummaries();
      this.hooks.onTrial(this, {
        trialIndex: this.trialIndex - 1,
        segId: chosen.segId,
        taskK: chosen.askedK,
        pick: rawPick,
        isCorrect: rawPick === chosen.askedK ? 1 : 0,
        reactionMs: clientTiming?.reactionMs ?? null,
        shownClientUtc: clientTiming?.shownClientUtc ?? null,
        answeredClientUtc: clientTiming?.answeredClientUtc ?? null,
        diag: {
          atom_posterior: mass,
          atom_map_beta: betas[mapIndex],
          atom_mean_beta: betas.reduce(
            (sum, beta, index) => sum + beta * mass[index], 0,
          ),
          skill: summaries.skill,
          bias: summaries.bias,
          ess: diagnostic.ess,
          resampled: diagnostic.rejuvenated,
          precision_statuses: evaluation.selectionStates.slice(),
          policy_stop: evaluation.stop,
        },
      });
      if (this.stopped) this.hooks.onStop(this);
    }
    return record;
  }

  /** One simulated answer, mirroring harness.py truth response generation. */
  simulatedPick(): number {
    if (this.truth === null) {
      throw new HttpError(409, "session has no simulated reader (attach one first)");
    }
    if (this.currentItem === null) throw new HttpError(409, "session has already stopped");
    const segment = this.segments[this.currentItem.segmentIndex];
    const distribution = responseProbabilities(
      this.profile, this.truth.artifact, this.currentItem.askedK, segment,
      this.truth.t, this.truth.l, 0, K,
    );
    const u = this.responseRng.random();
    let cumulative = 0;
    for (const entry of distribution) {
      cumulative += entry.probability;
      if (u < cumulative) return entry.outcome;
    }
    return distribution[distribution.length - 1].outcome;
  }

  /** Per-domain posterior mean + equal-tailed 95% CI for skill l and bias t. */
  private posteriorSummaries(): {
    skill: { mean: number; ci95: [number, number] }[];
    bias: { mean: number; ci95: [number, number] }[];
  } {
    const moments = posteriorMoments(this.state);
    const weights = this.state.w;
    const column = new Float64Array(this.state.N);
    const skill: { mean: number; ci95: [number, number] }[] = [];
    const bias: { mean: number; ci95: [number, number] }[] = [];
    for (let k = 0; k < K; k += 1) {
      for (let n = 0; n < this.state.N; n += 1) column[n] = this.state.l[n * K + k];
      skill.push({
        mean: moments.lMean[k],
        ci95: [
          weightedQuantile(column, weights, 0.025),
          weightedQuantile(column, weights, 0.975),
        ],
      });
      for (let n = 0; n < this.state.N; n += 1) column[n] = this.state.t[n * K + k];
      bias.push({
        mean: moments.tMean[k],
        ci95: [
          weightedQuantile(column, weights, 0.025),
          weightedQuantile(column, weights, 0.975),
        ],
      });
    }
    return { skill, bias };
  }

  status(): SessionStatusView {
    const diag = this.lastEvaluation?.diagnostics as
      | {
        skillTolerance: number[];
        skillPointCenteredRadius: number[];
        guardedPrecisionStatistic: number[];
      }
      | undefined;
    const statuses = this.lastEvaluation?.selectionStates
      ?? new Array<string>(K).fill(PRECISION_STATUS.ACTIVE);
    const terminalReasons = this.lastEvaluation?.terminalReasons
      ?? new Array<string | null>(K).fill(null);
    const streaks = this.lastEvaluation?.streakCounts ?? new Array<number>(K).fill(0);
    const summaries = this.posteriorSummaries();
    const domains: DomainStatusView[] = [];
    for (let k = 0; k < K; k += 1) {
      domains.push({
        k,
        code: PRECISION_TASK_CODES[k],
        label: TASK_LABELS[k],
        status: statuses[k],
        terminalReason: terminalReasons[k],
        n: this.ledger.nPerTask[k],
        streak: streaks[k],
        skill: summaries.skill[k],
        bias: summaries.bias[k],
        radius: diag?.skillPointCenteredRadius[k] ?? null,
        guardedRadius: diag?.guardedPrecisionStatistic[k] ?? null,
        tolerance: diag?.skillTolerance[k] ?? null,
      });
    }
    const mass = atomPosterior(this.state, this.artifact);
    const betas = this.artifact.draws.map((draw) => draw.beta);
    const mapIndex = mass.indexOf(Math.max(...mass));
    const item = this.currentItem;
    const segment = item ? this.segments[item.segmentIndex] : null;
    return {
      sessionId: this.id,
      mode: this.config.mode,
      config: this.config,
      reader: this.truth?.reader ?? null,
      stopped: this.stopped,
      stopReason: this.stopReason,
      questionsAsked: this.trialIndex,
      remainingSegments: this.ledger.remainingSegmentIds.size,
      currentItem: item && segment
        ? {
          questionNumber: this.trialIndex + 1,
          askedK: item.askedK,
          askedClass: TASK_LABELS[item.askedK],
          segId: item.segId,
          focalSignal: item.focalSignal,
          segmentSignals: [...IIIC_GROUP].map((k) => ({
            k,
            label: TASK_LABELS[k],
            sMean: segment.sMean[k],
            sSd: segment.sSd[k],
          })),
        }
        : null,
      nPerTask: this.ledger.nPerTask.slice(),
      domains,
      atomPosterior: {
        betas,
        lapses: this.artifact.draws.map((draw) => draw.distractorLapse),
        mass,
        mapBeta: betas[mapIndex],
        meanBeta: betas.reduce((sum, beta, index) => sum + beta * mass[index], 0),
      },
      policy: this.lastEvaluation
        ? {
          stop: this.lastEvaluation.stop,
          stopReason: this.lastEvaluation.stopReason,
          selectionStates: this.lastEvaluation.selectionStates.slice(),
          ess: this.lastEvaluation.ess,
        }
        : null,
      history: this.history.slice(-25),
      timing: { totalMs: this.totalMs, lastStepMs: this.lastStepMs },
    };
  }
}
