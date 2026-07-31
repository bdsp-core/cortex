// Adaptive session loop — port of CortexSession.run (scripts/session_controller.py),
// reshaped for the browser's async answer flow.
//
// The per-trial step (reweight → maybe rejuvenate → AD6 + per-domain cap →
// choose next) lives in engine/advance.ts as advanceCore, so it can run either
// INLINE (committed straight onto the live core) or SPECULATIVELY: during the
// participant's think-time the worker starts the posterior-predicted answer
// branch on a clone, yields to accept an answer, and computes the other branch
// only when it is still needed. Because both paths call the SAME advanceCore on
// an exact clone, a speculative session is BIT-IDENTICAL to the inline one
// (proven in speculative.test.ts) — speculation hides the heavy N=1200 selection
// in otherwise-idle think-time without changing a single result.

import {
  BranchLifecycleV2, ComputeEngineInputs, EngineStepTiming, TerminationPolicyName,
  TrialDiag,
} from "./types";
import { precomputePriorPair } from "./prior";
import { makeState } from "./particles";
import { aurocSummary } from "./auroc";
import { BankArrays, Chosen, predictedYesProbability } from "./choose_item";
import { nwayResponseRuntimeFor } from "./nway_likelihood";
import { isNWaySession, validateNWayInputs } from "./nway_profile";
import { predictedOutcomeDistribution, rankOutcomes } from "./nway_selector";
import { AD6Policy, EngineTerminationPolicy } from "./policy";
import {
  PRECISION_PER_DOMAIN_CAP, PRECISION_STATUS, PrecisionPolicy,
} from "./precision_policy";
import { Rng } from "./rng";
import type { BranchExecutor } from "./branch_executor";
import type { NWaySelectionExecutor } from "./nway_selector_executor";
import {
  SessionCore, AdvanceParams, AdvanceResult, advanceCore,
  advanceCoreWithSelectionExecutor, cloneCore, chooseNext, precisionRemainingBank,
} from "./advance";
import {
  isSpeculationCancelled, type SpeculationCancellationPhase,
} from "./speculation_cancellation";
import {
  isRankedSpeculationRejuvenationDeferred, RANKED_SPECULATION_PROFILE,
  shouldExpandRankedSpeculation,
} from "./ranked_speculation";

// Default particle count (frozen-pilot instrument). v15 staging (OPT-IN) lets a
// manifest override this via EngineInputs.nParticles (1200); when absent the
// session resolves to this value, keeping the default path bit-identical.
export const N_PARTICLES = 600;
// Legacy fixed-sweep cap (v1.3.6), superseded by adaptive termination: the
// engine loop ends on AD6 resolution, bounded by the per-domain budget below
// (K × PER_DOMAIN_CAP). Kept exported for back-compat; no longer used.
export const MAX_QUESTIONS = 500;
// Per-domain question budget (v1.6 adaptive termination). A task still PENDING
// after this many of its OWN questions is capped out of selection and REFERred
// at finalize — bounding the worst-case single-domain tail so the test ends on
// resolution, not a fixed sweep. Manifest-overridable via
// EngineInputs.perDomainCap (the served v1.6-k7-35k bundle carries 60); this is
// the fallback for bundles that don't declare one. Lowered 120→60 (2026-06-26)
// so a single borderline domain can't consume half the test before REFER.
export const PER_DOMAIN_CAP = 60;
export const N_MH_STEPS = 15;
// PrecisionPolicy uses stronger rejuvenation (mc-guard v2, default 2026-07-18);
// AD6 keeps 15. Mirrors the precision_policy profile's n_mh_steps=30 and its
// requalified guard inflation (PRECISION_RADIUS_MCSE_INFLATION).
export const PRECISION_N_MH_STEPS = 30;
export const ESS_THRESHOLD_FRAC = 0.5;
export const FIRST_ITEM_TOPN = 10;
// Variety cap (v1.3.7 alignment): after this many consecutive picks from the
// same task, that task is excluded from the next pick so the session doesn't
// streak one domain for ten in a row. Fallback inside the loop restores the
// excluded task if no others have remaining items.
export const MAX_CONSECUTIVE_SAME_DOMAIN = 5;
export const PRECISION_N_PARTICLES = 1200;
export const PRECISION_N_SUBSAMPLE = 128;

// SHA-256(sessionId) → 32-bit int seed, matching the desktop's
// int(hashlib.sha256(session_id)[:8], 16).
export async function seedFromSessionId(sessionId: string): Promise<number> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(sessionId));
  const b = new Uint8Array(buf);
  // first 4 bytes (8 hex chars) as the desktop slices [:8]
  return ((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]) >>> 0;
}

export interface SessionCallbacks {
  onItem?: (item: { trialIndex: number; taskK: number; segId: number }) => void;
  onPrefetch?: (hint: { trialIndex: number; segId: number }) => void;
  onTrial?: (diag: TrialDiag) => void;
  onPerformance?: (event: EngineStepTiming) => void;
  onDone?: (result: SessionResult) => void;
}

export interface SessionOptions {
  // Speculative precompute (v1.6): compute the two highest-probability raw
  // outcomes during participant think-time so the common answer→next-item
  // path is a branch adopt. Bit-identical to the inline path; defaults OFF
  // (tests/back-compat opt in explicitly; the worker enables it in production).
  speculative?: boolean;
  branchExecutor?: BranchExecutor;
  selectionExecutor?: NWaySelectionExecutor;
  /** Qualification control; production defaults to bounded rank-two expansion. */
  rankedSpeculation?: boolean;
}

export interface SessionResult {
  sessionId: string;
  nQuestions: number;
  stopReason: string;
  verdicts: string[];
  terminationPolicy: TerminationPolicyName;
  domainStatuses?: string[];
  determinations?: string[];
  terminalReasons?: (string | null)[];
  skillIntervals?: [number, number][];
  biasIntervals?: [number, number][];
  biasFlags?: (string | null)[];
  nwayProfile?: ComputeEngineInputs["nwayProfile"];
  servedSegIds: number[];
  trials: TrialDiag[];
  finalAuroc: number[]; // per-task posterior-mean AUROC (final cloud)
  finalAurocHw: number[]; // per-task (1-α) credible halfwidth
  // Particle-cloud trajectory for the visualization videos (#8). Flattened
  // Float32: t/l are [T,N,K] row-major; w is [T,N]. Fed to the desktop
  // renderers server-side as trajectory.npz (cortex_storage.py schema).
  traj: { t: Float32Array; l: Float32Array; w: Float32Array; shape: [number, number, number] };
}

const NO_ITEM: Chosen = { k: 0, s: 0, sSd: 0, segId: -1, loss: Infinity };

interface SubmittedAnswer {
  pick: number;
  submittedAtEpochMs: number;
  processedAtEpochMs: number;
}

function highResolutionEpochMs(): number {
  return performance.timeOrigin + performance.now();
}

export class WebCortexSession {
  private inputs: ComputeEngineInputs;
  private sessionId: string;
  private seed: number;
  // All mutable per-trial state lives in one cloneable bag so a speculative
  // branch can run on an exact copy and be adopted atomically.
  private core!: SessionCore;
  private trials: TrialDiag[] = [];
  private served: number[] = [];
  // Per-question particle-cloud snapshots for the visualization videos (#8).
  private tTraj: Float64Array[] = [];
  private lTraj: Float64Array[] = [];
  private wTraj: Float64Array[] = [];
  private proposalScale: number;
  private speculative: boolean;
  private branchExecutor?: BranchExecutor;
  private selectionExecutor?: NWaySelectionExecutor;
  private rankedSpeculation: boolean;
  private cb: SessionCallbacks;
  private answerResolver: ((answer: SubmittedAnswer) => void) | null = null;
  private selectionRecovery: Promise<void> | null = null;
  private aborted = false;

  constructor(
    inputs: ComputeEngineInputs, sessionId: string, seed: number,
    cb: SessionCallbacks = {}, opts: SessionOptions = {},
  ) {
    this.inputs = inputs;
    this.sessionId = sessionId;
    this.seed = seed;
    this.cb = cb;
    this.proposalScale = 2.38 / Math.sqrt(2 * inputs.taskCodes.length);
    this.speculative = opts.speculative ?? false;
    this.branchExecutor = opts.branchExecutor;
    this.selectionExecutor = opts.selectionExecutor;
    this.rankedSpeculation = opts.rankedSpeculation ?? true;
  }

  // The GUI calls this with the raw 0-based 6-way pick after each item.
  submitAnswer(pick: number, submittedAtEpochMs = highResolutionEpochMs()): void {
    if (this.answerResolver) {
      const r = this.answerResolver;
      this.answerResolver = null;
      const processedAtEpochMs = highResolutionEpochMs();
      r({
        pick,
        submittedAtEpochMs: Number.isFinite(submittedAtEpochMs)
          ? submittedAtEpochMs : processedAtEpochMs,
        processedAtEpochMs,
      });
    }
  }

  abort(): void {
    this.aborted = true;
    if (this.answerResolver) this.submitAnswer(-1);
  }

  private deferSelectionRecovery(
    executor: NWaySelectionExecutor, recovery: Promise<void>,
  ): void {
    const tracked = recovery.catch(() => {
      executor.dispose();
      if (this.selectionExecutor === executor) this.selectionExecutor = undefined;
    }).finally(() => {
      if (this.selectionRecovery === tracked) this.selectionRecovery = null;
    });
    this.selectionRecovery = tracked;
  }

  private async waitForSelectionRecovery(): Promise<void> {
    await this.selectionRecovery;
  }

  private awaitAnswer(): Promise<SubmittedAnswer> {
    return new Promise((resolve) => (this.answerResolver = resolve));
  }

  // Let the worker service an answer/abort message that arrived while a
  // speculative branch was running. A timer task is used instead of a
  // microtask because Worker message delivery itself is a task.
  private yieldToWorker(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  private advance(
    core: SessionCore, chosen: Chosen, rawPick: number,
    params: AdvanceParams, trialIndex: number,
    preparedPrecisionBank?: BankArrays,
    cancellationSignal?: AbortSignal,
    deferMhRejuvenation = false,
  ): Promise<AdvanceResult> {
    if (this.selectionExecutor) {
      return advanceCoreWithSelectionExecutor(
        core, this.inputs, chosen, rawPick, params, trialIndex,
        this.selectionExecutor, preparedPrecisionBank, cancellationSignal,
        deferMhRejuvenation,
      );
    }
    return Promise.resolve(advanceCore(
      core, this.inputs, chosen, rawPick, params, trialIndex, preparedPrecisionBank,
    ));
  }

  async run(): Promise<SessionResult> {
    const K = this.inputs.taskCodes.length;
    // N is opt-in (v15 staging): default 600 (frozen pilot) unless the manifest
    // carries nParticles. The prior pair uses corrT for the t-block when the
    // manifest carries it (v15), else corrL for both blocks (pilot — bit-
    // identical to the single-PriorPieces era).
    const policyName = this.inputs.terminationPolicy ?? "ad6";
    const nway = isNWaySession(this.inputs);
    if (nway) validateNWayInputs(this.inputs);
    let policy: EngineTerminationPolicy;
    let nParticles: number;
    let perDomainCap: number;
    if (policyName === "precision_v1") {
      if (this.inputs.nParticles !== PRECISION_N_PARTICLES) {
        throw new Error(`precision_v1 freezes nParticles=${PRECISION_N_PARTICLES}`);
      }
      if ((this.inputs.perDomainCap ?? PER_DOMAIN_CAP) !== PRECISION_PER_DOMAIN_CAP) {
        throw new Error(`precision_v1 freezes perDomainCap=${PRECISION_PER_DOMAIN_CAP}`);
      }
      policy = PrecisionPolicy.fromInputs(this.inputs);
      nParticles = PRECISION_N_PARTICLES;
      perDomainCap = PRECISION_PER_DOMAIN_CAP;
    } else if (policyName === "ad6") {
      policy = AD6Policy.fromInputs(this.inputs.ellStar, this.inputs.corrL);
      nParticles = this.inputs.nParticles ?? N_PARTICLES;
      perDomainCap = this.inputs.perDomainCap ?? PER_DOMAIN_CAP;
    } else {
      throw new Error(`unknown termination policy ${String(policyName)}`);
    }
    policy.reset(K);
    const rng = new Rng(this.seed);
    const prior = precomputePriorPair(this.inputs.corrL, this.inputs.corrT);
    // Profile-resolved response runtime: undefined on every mixture session
    // (no extra RNG is consumed, the frozen path is byte-identical); on a
    // draw-latent stamp the state carries the atoms plus per-particle
    // lineage sampled deterministically from the session seed — so resume
    // replay reconstructs the identical lineage.
    const responseRuntime = nway ? nwayResponseRuntimeFor(this.inputs) : undefined;
    const state = makeState(nParticles, K, prior, rng, responseRuntime);
    // Phase-aware selection (desktop session_controller.py l.241+): for K=7
    // bundles with spike at index 0, run the spike block first (Phase A) until
    // spike locks or its bank exhausts, then Phase B (only IIIC).
    const spikeIdx = this.inputs.taskClasses
      ? this.inputs.taskClasses.findIndex((c) => c === "spike")
      : -1;

    this.core = {
      state,
      rng,
      policy,
      remaining: new Set(this.inputs.segments.map((s) => s.segId)),
      nPerTask: new Array(K).fill(0),
      cappedTasks: new Set<number>(),
      lastOutcomes: new Array(K).fill(policy.activeLabel),
      lastTaskK: -1,
      streakCount: 0,
    };
    const params: AdvanceParams = {
      K,
      nParticles,
      perDomainCap,
      nMhSteps: nway || policyName === "precision_v1" ? PRECISION_N_MH_STEPS : N_MH_STEPS,
      essThresholdFrac: ESS_THRESHOLD_FRAC,
      proposalScale: this.proposalScale,
      firstItemTopN: FIRST_ITEM_TOPN,
      maxConsecutiveSameDomain: MAX_CONSECUTIVE_SAME_DOMAIN,
      k7Spike: spikeIdx >= 0,
      spikeIdx,
      ...(policyName === "precision_v1" ? {
        nSubsample: PRECISION_N_SUBSAMPLE,
        uncertaintyAwareSubsample: true,
      } : {}),
    };
    // Safety backstop: every task either resolves or hits its per-domain cap,
    // so K × cap bounds the session even if AD6 never fires.
    const maxQ = Math.min(K * perDomainCap, this.inputs.segments.length);

    let stopReason = "bank_exhausted";
    // Cold start: the opener (chooseFirstItem consumes rng). Guard the empty bank.
    let chosen: Chosen = this.core.remaining.size > 0
      ? chooseNext(this.core, this.inputs, params, 0)
      : NO_ITEM;
    if (chosen.segId === -1 && policyName === "precision_v1"
      && (policy as PrecisionPolicy).domainStatuses.every(
        (s) => s !== PRECISION_STATUS.ACTIVE,
      )) stopReason = "all_estimated_or_undeterminable";

    let trialIndex = 0;
    while (chosen.segId !== -1 && trialIndex < maxQ && !this.aborted) {
      // Arm the resolver before exposing the item. This also makes synchronous
      // test/demo callbacks safe; production answers arrive as Worker messages.
      let observedAnswer: SubmittedAnswer | undefined;
      const answerPromise = this.awaitAnswer().then((answer) => {
        observedAnswer = answer;
        return answer;
      });
      this.cb.onItem?.({ trialIndex, taskK: chosen.k, segId: chosen.segId });
      // A cached prior branch may already have exposed this item while a
      // cancelled expansion rebuilds the pool in the background.
      await this.waitForSelectionRecovery();

      const bankStartedAt = performance.now();
      let preparedPrecisionBank = policyName === "precision_v1"
        ? precisionRemainingBank(this.inputs, this.core.remaining, chosen.segId)
        : undefined;
      let bankPreparationMs = performance.now() - bankStartedAt;

      // Bounded ranked speculation: compute the most probable raw outcome on
      // the coordinator and the next outcomes on immutable helper snapshots up
      // to the measured pool capacity. An uncached observed outcome is always
      // calculated from untouched authoritative state.
      let branches: Map<number, AdvanceResult> | null = null;
      let firstPick: number | null = null;
      let firstReadyAtEpochMs: number | null = null;
      const expandedReadyAtEpochMs = new Map<number, number>();
      const helperJobs = new Map<number, {
        ready: boolean;
        readyAtEpochMs: number | null;
        readyAtAnswer: boolean;
        job: Promise<AdvanceResult>;
        lifecycle: BranchLifecycleV2;
      }>();
      let rankedOutcomes: { outcome: number; probability: number; rank: number }[] = [];
      const branchLifecycle: BranchLifecycleV2[] = [];
      if (this.speculative) {
        branches = new Map<number, AdvanceResult>();
        const distribution = nway
          ? predictedOutcomeDistribution(this.core.state, this.inputs, chosen)
          : (() => {
              const yes = predictedYesProbability(
                this.core.state, chosen.k, chosen.s, chosen.sSd,
              );
              return [
                { outcome: chosen.k, probability: yes },
                { outcome: K, probability: 1 - yes },
              ];
            })();
        rankedOutcomes = rankOutcomes(distribution);
        firstPick = rankedOutcomes[0].outcome;
        const executor = this.branchExecutor;
        const helperRankedOutcomes = executor
          ? rankedOutcomes.slice(1, 1 + executor.capacity) : [];
        for (const helperRanked of helperRankedOutcomes) {
          const helperLifecycle: BranchLifecycleV2 = {
            outcome: helperRanked.outcome,
            rank: helperRanked.rank,
            probability: helperRanked.probability,
            role: "helper",
            queued: true,
            started: true,
            readyAtAnswer: false,
            adopted: false,
            cancelled: false,
            discarded: false,
            durationMs: null,
          };
          branchLifecycle.push(helperLifecycle);
          try {
            const helper = {
              ready: false,
              readyAtEpochMs: null as number | null,
              readyAtAnswer: false,
              lifecycle: helperLifecycle,
              job: Promise.resolve(null as unknown as AdvanceResult),
            };
            helper.job = executor!.advance(
              this.core, chosen, params, trialIndex, helperRanked.outcome,
            ).then((result) => {
              helper.ready = true;
              helper.readyAtEpochMs = highResolutionEpochMs();
              helperLifecycle.durationMs = result.timing.totalMs;
              return result;
            });
            // A predicted response leaves the alternate result unused. Its
            // failure must never surface as an unhandled rejection.
            void helper.job.catch(() => undefined);
            helperJobs.set(helperRanked.outcome, helper);
          } catch {
            executor!.dispose();
            this.branchExecutor = undefined;
            break;
          }
        }
        const firstRanked = rankedOutcomes[0];
        const firstLifecycle: BranchLifecycleV2 = {
          outcome: firstRanked.outcome,
          rank: firstRanked.rank,
          probability: firstRanked.probability,
          role: "coordinator",
          queued: true,
          started: true,
          readyAtAnswer: false,
          adopted: false,
          cancelled: false,
          discarded: false,
          durationMs: null,
        };
        branchLifecycle.push(firstLifecycle);
        const cancellation = new AbortController();
        const firstStartedAt = performance.now();
        const firstJob = this.advance(
          cloneCore(this.core), chosen, firstPick, params, trialIndex,
          preparedPrecisionBank, cancellation.signal,
        );
        const firstCompletion = firstJob.then(
          (result) => ({ kind: "result" as const, result }),
          (error: unknown) => ({ kind: "error" as const, error }),
        );
        const canPreempt = Boolean(this.selectionExecutor?.restartAfterCancellation);
        const firstRace = canPreempt
          ? await Promise.race([
              firstCompletion,
              answerPromise.then((answer) => ({ kind: "answer" as const, answer })),
            ])
          : await firstCompletion;
        if (firstRace.kind === "answer" && firstRace.answer.pick !== firstPick) {
          cancellation.abort();
          let cancellationPhase: SpeculationCancellationPhase = "between_phases";
          const executorAtCancellation = this.selectionExecutor;
          try {
            const restart = executorAtCancellation?.restartAfterCancellation?.();
            cancellationPhase = restart?.report.phases[0] ?? cancellationPhase;
            await restart?.ready;
          } catch {
            executorAtCancellation?.dispose();
            if (this.selectionExecutor === executorAtCancellation) {
              this.selectionExecutor = undefined;
            }
          }
          const completion = await firstCompletion;
          if (completion.kind === "error"
              && !isSpeculationCancelled(completion.error)) {
            throw completion.error;
          }
          firstLifecycle.cancelled = true;
          firstLifecycle.cancellationPhase = cancellationPhase;
          firstLifecycle.durationMs = performance.now() - firstStartedAt;
        } else {
          const completion = firstRace.kind === "answer"
            ? await firstCompletion : firstRace;
          if (completion.kind === "error") throw completion.error;
          const firstResult = completion.result;
          firstReadyAtEpochMs = highResolutionEpochMs();
          firstLifecycle.durationMs = firstResult.timing.totalMs;
          branches.set(firstPick, firstResult);
          if (firstResult.nextChosen.segId !== -1) {
            this.cb.onPrefetch?.({
              trialIndex: trialIndex + 1,
              segId: firstResult.nextChosen.segId,
            });
          }

          const secondRanked = rankedOutcomes[1];
          const expansionEligible = secondRanked !== undefined
            && this.rankedSpeculation
            && shouldExpandRankedSpeculation({
              rank: secondRanked.rank,
              probability: secondRanked.probability,
              workerCount: this.selectionExecutor?.workerCount ?? 1,
              separateBranchExecutorActive: Boolean(this.branchExecutor),
            });
          if (expansionEligible && !observedAnswer) {
            const grace = await Promise.race([
              answerPromise.then((answer) => ({ kind: "answer" as const, answer })),
              new Promise<{ kind: "grace" }>((resolve) => globalThis.setTimeout(
                () => resolve({ kind: "grace" }),
                RANKED_SPECULATION_PROFILE.expansionGraceMs,
              )),
            ]);
            if (grace.kind === "grace") {
              const secondLifecycle: BranchLifecycleV2 = {
                outcome: secondRanked.outcome,
                rank: secondRanked.rank,
                probability: secondRanked.probability,
                role: "coordinator_expansion",
                queued: true,
                started: true,
                readyAtAnswer: false,
                adopted: false,
                cancelled: false,
                discarded: false,
                durationMs: null,
              };
              branchLifecycle.push(secondLifecycle);
              const secondCancellation = new AbortController();
              const secondStartedAt = performance.now();
              const secondJob = this.advance(
                cloneCore(this.core), chosen, secondRanked.outcome, params, trialIndex,
                preparedPrecisionBank, secondCancellation.signal,
                !RANKED_SPECULATION_PROFILE.allowMhRejuvenation,
              );
              const secondCompletion = secondJob.then(
                (result) => ({ kind: "result" as const, result }),
                (error: unknown) => ({ kind: "error" as const, error }),
              );
              const secondRace = await Promise.race([
                secondCompletion,
                answerPromise.then((answer) => ({ kind: "answer" as const, answer })),
              ]);
              if (secondRace.kind === "answer"
                  && secondRace.answer.pick !== secondRanked.outcome) {
                secondCancellation.abort();
                let cancellationPhase: SpeculationCancellationPhase = "between_phases";
                const executorAtCancellation = this.selectionExecutor;
                try {
                  const restart = executorAtCancellation?.restartAfterCancellation?.();
                  cancellationPhase = restart?.report.phases[0] ?? cancellationPhase;
                  if (restart && branches?.has(secondRace.answer.pick)) {
                    this.deferSelectionRecovery(executorAtCancellation!, restart.ready);
                  } else {
                    await restart?.ready;
                  }
                } catch {
                  executorAtCancellation?.dispose();
                  if (this.selectionExecutor === executorAtCancellation) {
                    this.selectionExecutor = undefined;
                  }
                }
                const cancelledCompletion = await secondCompletion;
                if (cancelledCompletion.kind === "error"
                    && !isSpeculationCancelled(cancelledCompletion.error)
                    && !isRankedSpeculationRejuvenationDeferred(
                      cancelledCompletion.error,
                    )) {
                  throw cancelledCompletion.error;
                }
                secondLifecycle.cancelled = true;
                secondLifecycle.cancellationPhase = cancellationPhase;
                secondLifecycle.durationMs = performance.now() - secondStartedAt;
              } else {
                const expandedCompletion = secondRace.kind === "answer"
                  ? await secondCompletion : secondRace;
                if (expandedCompletion.kind === "error") {
                  if (!isRankedSpeculationRejuvenationDeferred(
                    expandedCompletion.error,
                  )) {
                    throw expandedCompletion.error;
                  }
                  secondLifecycle.deferredForRejuvenation = true;
                  secondLifecycle.durationMs = performance.now() - secondStartedAt;
                } else {
                  const secondResult = expandedCompletion.result;
                  const readyAtEpochMs = highResolutionEpochMs();
                  expandedReadyAtEpochMs.set(secondRanked.outcome, readyAtEpochMs);
                  secondLifecycle.durationMs = secondResult.timing.totalMs;
                  branches?.set(secondRanked.outcome, secondResult);
                }
              }
            }
          }

          // Process an answer/abort message that arrived during a final
          // synchronous phase or completed expansion. Helpers continue independently.
          await this.yieldToWorker();
        }
      }
      // An abort that arrived during the (synchronous) speculation above.
      if (this.aborted) { stopReason = "aborted"; break; }

      const answer = observedAnswer ?? await answerPromise;
      const pick = answer.pick;
      if (this.aborted || pick < 0) { stopReason = "aborted"; break; }
      const firstLifecycle = branchLifecycle.find((branch) => branch.role === "coordinator");
      if (firstLifecycle && firstReadyAtEpochMs !== null) {
        // The result exists by this point, but it counts as a cache hit only if
        // it finished before the participant's main-realm submission time.
        firstLifecycle.readyAtAnswer = firstReadyAtEpochMs <= answer.submittedAtEpochMs;
      }
      for (const branch of branchLifecycle) {
        if (branch.role !== "coordinator_expansion") continue;
        const readyAtEpochMs = expandedReadyAtEpochMs.get(branch.outcome);
        branch.readyAtAnswer = readyAtEpochMs !== undefined
          && readyAtEpochMs <= answer.submittedAtEpochMs;
      }
      for (const helper of helperJobs.values()) {
        helper.readyAtAnswer = helper.readyAtEpochMs !== null
          && helper.readyAtEpochMs <= answer.submittedAtEpochMs;
      }

      let res: AdvanceResult;
      const selectedHelper = helperJobs.get(pick);
      if (selectedHelper) {
        const readyAtAnswer = selectedHelper.readyAtAnswer;
        try {
          res = await selectedHelper.job;
          res.timing.requiredBranchReadyAtAnswer = readyAtAnswer;
        } catch {
          // Fail closed to the untouched authoritative core. No helper ever
          // mutates it, so the existing serial calculation remains exact.
          this.branchExecutor?.dispose();
          this.branchExecutor = undefined;
          const fallbackBankStartedAt = performance.now();
          preparedPrecisionBank = policyName === "precision_v1"
            ? precisionRemainingBank(this.inputs, this.core.remaining, chosen.segId)
            : undefined;
          bankPreparationMs = performance.now() - fallbackBankStartedAt;
          res = await this.advance(
            this.core, chosen, pick, params, trialIndex,
            preparedPrecisionBank,
          );
          const requiredRanked = rankedOutcomes.find((entry) => entry.outcome === pick);
          branchLifecycle.push({
            outcome: pick,
            rank: requiredRanked?.rank ?? -1,
            probability: requiredRanked?.probability ?? 0,
            role: "serial_required",
            queued: true,
            started: true,
            readyAtAnswer: false,
            adopted: false,
            cancelled: false,
            discarded: false,
            durationMs: res.timing.totalMs,
          });
          res.timing.executionMode = "serial_fallback";
          res.timing.speculative = false;
        }
      } else {
        const cachedBranch = branches?.get(pick);
        res = cachedBranch ?? await this.advance(
          this.core, chosen, pick, params, trialIndex, preparedPrecisionBank,
        );
        res.timing.bankPreparationMs = bankPreparationMs;
        res.timing.speculative = this.speculative;
        if (helperJobs.size > 0 && pick === firstPick) {
          res.timing.executionMode = "dual_branch";
        }
        const cachedLifecycle = cachedBranch
          ? branchLifecycle.find((branch) => branch.outcome === pick && !branch.cancelled)
          : undefined;
        res.timing.requiredBranchReadyAtAnswer = cachedLifecycle?.readyAtAnswer ?? false;
      }
      if (rankedOutcomes.length > 0
          && !branchLifecycle.some((branch) => branch.outcome === pick
            && !branch.cancelled && !branch.deferredForRejuvenation)) {
        const requiredRanked = rankedOutcomes.find((entry) => entry.outcome === pick);
        branchLifecycle.push({
          outcome: pick,
          rank: requiredRanked?.rank ?? -1,
          probability: requiredRanked?.probability ?? 0,
          role: "serial_required",
          queued: true,
          started: true,
          readyAtAnswer: false,
          adopted: false,
          cancelled: false,
          discarded: false,
          durationMs: res.timing.totalMs,
        });
      }
      for (const branch of branchLifecycle) {
        if (branch.role === "helper") {
          branch.readyAtAnswer = helperJobs.get(branch.outcome)?.readyAtAnswer ?? false;
        }
        branch.adopted = branch.outcome === pick
          && !branch.cancelled
          && !branch.deferredForRejuvenation
          && (branch.role !== "helper" || res.timing.executionMode !== "serial_fallback");
        branch.discarded = !branch.adopted;
      }
      const observedOutcome = rankedOutcomes.find((entry) => entry.outcome === pick);
      if (res.timing.phaseV2) {
        res.timing.phaseV2.observedOutcomeRank = observedOutcome?.rank ?? null;
        res.timing.phaseV2.observedOutcomeProbability = observedOutcome?.probability ?? null;
        res.timing.phaseV2.answerDispatchDelayMs = Math.max(
          0, answer.processedAtEpochMs - answer.submittedAtEpochMs,
        );
        res.timing.phaseV2.cachedProbabilityMass = branchLifecycle.reduce(
          (sum, branch) => sum + (branch.readyAtAnswer ? branch.probability : 0), 0,
        );
        res.timing.phaseV2.branches = branchLifecycle;
      }
      this.core = res.core; // spec: adopt the matching clone; inline: same ref

      this.served.push(res.servedSegId);
      this.trials.push(res.diag);
      this.tTraj.push(res.trajSnapshot.t);
      this.lTraj.push(res.trajSnapshot.l);
      this.wTraj.push(res.trajSnapshot.w);
      this.cb.onTrial?.(res.diag);
      this.cb.onPerformance?.(res.timing);

      if (res.done) { stopReason = res.stopReason; break; }
      chosen = res.nextChosen;
      // No candidate among the still-active domains → their banks are exhausted.
      if (chosen.segId === -1) { stopReason = "bank_exhausted"; break; }
      trialIndex++;
    }

    const finalized = this.core.policy.finalizeResult(this.inputs.ellStar);
    const { mean: finalAuroc, hw: finalAurocHw } = aurocSummary(
      this.core.state.l, this.core.state.w, this.core.state.N, K,
    );
    // Flatten the snapshots into [T,N,K] (t,l) and [T,N] (w) Float32 buffers.
    const T = this.tTraj.length, N = this.core.state.N;
    const t = new Float32Array(T * N * K), l = new Float32Array(T * N * K), w = new Float32Array(T * N);
    for (let i = 0; i < T; i++) {
      t.set(this.tTraj[i], i * N * K);
      l.set(this.lTraj[i], i * N * K);
      w.set(this.wTraj[i], i * N);
    }
    const result: SessionResult = {
      sessionId: this.sessionId,
      nQuestions: this.trials.length,
      stopReason,
      verdicts: finalized.verdicts,
      terminationPolicy: policyName,
      ...(finalized.domainStatuses ? { domainStatuses: finalized.domainStatuses } : {}),
      ...(finalized.determinations ? { determinations: finalized.determinations } : {}),
      ...(finalized.terminalReasons ? { terminalReasons: finalized.terminalReasons } : {}),
      ...(finalized.skillIntervals ? { skillIntervals: finalized.skillIntervals } : {}),
      ...(finalized.biasIntervals ? { biasIntervals: finalized.biasIntervals } : {}),
      ...(finalized.biasFlags ? { biasFlags: finalized.biasFlags } : {}),
      ...(this.inputs.nwayProfile ? { nwayProfile: { ...this.inputs.nwayProfile } } : {}),
      servedSegIds: this.served,
      trials: this.trials,
      finalAuroc,
      finalAurocHw,
      traj: { t, l, w, shape: [T, N, K] },
    };
    this.cb.onDone?.(result);
    return result;
  }
}
