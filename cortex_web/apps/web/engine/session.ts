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

import { EngineInputs, TerminationPolicyName, TrialDiag } from "./types";
import { precomputePriorPair } from "./prior";
import { makeState } from "./particles";
import { aurocSummary } from "./auroc";
import { Chosen, predictedYesProbability } from "./choose_item";
import { AD6Policy, EngineTerminationPolicy } from "./policy";
import {
  PRECISION_PER_DOMAIN_CAP, PRECISION_STATUS, PrecisionPolicy,
} from "./precision_policy";
import { Rng } from "./rng";
import {
  SessionCore, AdvanceParams, AdvanceResult, cloneCore, advanceCore, chooseNext,
  precisionRemainingBank,
} from "./advance";

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
  onTrial?: (diag: TrialDiag) => void;
  onDone?: (result: SessionResult) => void;
}

export interface SessionOptions {
  // Speculative precompute (v1.6): compute both answer branches during the
  // participant's think-time so the answer→next-item path is just a branch
  // adopt. Bit-identical to the inline path; defaults OFF (tests/back-compat
  // opt in explicitly; the worker turns it ON in production).
  speculative?: boolean;
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

export class WebCortexSession {
  private inputs: EngineInputs;
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
  private cb: SessionCallbacks;
  private answerResolver: ((pick: number) => void) | null = null;
  private aborted = false;

  constructor(
    inputs: EngineInputs, sessionId: string, seed: number,
    cb: SessionCallbacks = {}, opts: SessionOptions = {},
  ) {
    this.inputs = inputs;
    this.sessionId = sessionId;
    this.seed = seed;
    this.cb = cb;
    this.proposalScale = 2.38 / Math.sqrt(2 * inputs.taskCodes.length);
    this.speculative = opts.speculative ?? false;
  }

  // The GUI calls this with the raw 0-based 6-way pick after each item.
  submitAnswer(pick: number): void {
    if (this.answerResolver) {
      const r = this.answerResolver;
      this.answerResolver = null;
      r(pick);
    }
  }

  abort(): void {
    this.aborted = true;
    if (this.answerResolver) this.submitAnswer(-1);
  }

  private awaitAnswer(): Promise<number> {
    return new Promise((resolve) => (this.answerResolver = resolve));
  }

  // Let the worker service an answer/abort message that arrived while a
  // speculative branch was running. A timer task is used instead of a
  // microtask because Worker message delivery itself is a task.
  private yieldToWorker(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  async run(): Promise<SessionResult> {
    const K = this.inputs.taskCodes.length;
    // N is opt-in (v15 staging): default 600 (frozen pilot) unless the manifest
    // carries nParticles. The prior pair uses corrT for the t-block when the
    // manifest carries it (v15), else corrL for both blocks (pilot — bit-
    // identical to the single-PriorPieces era).
    const policyName = this.inputs.terminationPolicy ?? "ad6";
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
    const state = makeState(nParticles, K, prior, rng);
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
      nMhSteps: policyName === "precision_v1" ? PRECISION_N_MH_STEPS : N_MH_STEPS,
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
      let observedPick: number | undefined;
      const answerPromise = this.awaitAnswer().then((pick) => {
        observedPick = pick;
        return pick;
      });
      this.cb.onItem?.({ trialIndex, taskK: chosen.k, segId: chosen.segId });

      // The post-answer remaining bank is independent of the answer. Build it
      // once and share it across both speculative branches instead of
      // expanding/filtering/sorting the same 35k manifest twice.
      const preparedPrecisionBank = policyName === "precision_v1"
        ? precisionRemainingBank(this.inputs, this.core.remaining, chosen.segId)
        : undefined;

      // Speculative precompute: start the predicted answer branch during
      // think-time. advanceCore is identical to the inline path, so whichever
      // branch is adopted remains bit-identical to post-answer computation.
      let branches: [AdvanceResult | null, AdvanceResult | null] | null = null;
      if (this.speculative) {
        branches = [null, null];
        const firstY: 0 | 1 = predictedYesProbability(
          this.core.state, chosen.k, chosen.s, chosen.sSd,
        ) >= 0.5 ? 1 : 0;
        branches[firstY] = advanceCore(
          cloneCore(this.core), this.inputs, chosen, firstY, params, trialIndex,
          preparedPrecisionBank,
        );

        // If the participant answered during the first branch, process that
        // message now and compute only the requested branch. Otherwise use the
        // remaining think-time to finish the second branch as before.
        await this.yieldToWorker();
        if (!this.aborted && observedPick !== undefined && observedPick >= 0) {
          const observedY: 0 | 1 = observedPick === chosen.k ? 1 : 0;
          if (!branches[observedY]) {
            branches[observedY] = advanceCore(
              cloneCore(this.core), this.inputs, chosen, observedY, params, trialIndex,
              preparedPrecisionBank,
            );
          }
        } else if (!this.aborted) {
          const secondY: 0 | 1 = firstY === 0 ? 1 : 0;
          branches[secondY] = advanceCore(
            cloneCore(this.core), this.inputs, chosen, secondY, params, trialIndex,
            preparedPrecisionBank,
          );
        }
      }
      // An abort that arrived during the (synchronous) speculation above.
      if (this.aborted) { stopReason = "aborted"; break; }

      const pick = observedPick ?? await answerPromise;
      if (this.aborted || pick < 0) { stopReason = "aborted"; break; }
      const y: 0 | 1 = pick === chosen.k ? 1 : 0;

      const res = branches
        ? branches[y]!
        : advanceCore(
          this.core, this.inputs, chosen, y, params, trialIndex,
          preparedPrecisionBank,
        );
      this.core = res.core; // spec: adopt the matching clone; inline: same ref

      this.served.push(res.servedSegId);
      this.trials.push(res.diag);
      this.tTraj.push(res.trajSnapshot.t);
      this.lTraj.push(res.trajSnapshot.l);
      this.wTraj.push(res.trajSnapshot.w);
      this.cb.onTrial?.(res.diag);

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
