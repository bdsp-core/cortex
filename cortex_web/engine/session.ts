// Adaptive session loop — port of CortexSession.run (scripts/session_controller.py),
// reshaped for the browser's async answer flow.
//
// The desktop blocks a worker thread on a queue.get() for each answer; here
// the loop awaits a Promise that submitAnswer() resolves. The binary-Y
// reduction (Y = 1 iff the rater's 0-based 6-way pick == the task k the
// engine asked) happens in submitAnswer, exactly as EngineWorker._y_source.

import { EngineInputs, ParticleState, TrialDiag } from "./types";
import { precomputePrior } from "./prior";
import { makeState, update, ess, resampleAndRejuvenate, posteriorMeans } from "./particles";
import { BankArrays, chooseItem, chooseFirstItem, Chosen } from "./choose_item";
import { AD6Policy, VERDICT } from "./policy";
import { aurocSummary } from "./auroc";
import { Rng } from "./rng";

export const N_PARTICLES = 600;
export const MAX_QUESTIONS = 300;
export const N_MH_STEPS = 15;
export const ESS_THRESHOLD_FRAC = 0.5;
export const FIRST_ITEM_TOPN = 10;

// SHA-256(sessionId) → 32-bit int seed, matching the desktop's
// int(hashlib.sha256(session_id)[:8], 16).
//
// NB: crypto.subtle (Web Crypto) only exists in a SECURE CONTEXT — https, or
// localhost / 127.0.0.1. It is UNDEFINED on http://0.0.0.0, so `crypto.subtle.
// digest` throws "Cannot read properties of undefined (reading 'digest')". When
// it's absent we fall back to a deterministic FNV-1a 32-bit hash so the engine
// still seeds (the web app needs determinism, not byte-parity with the desktop).
export async function seedFromSessionId(sessionId: string): Promise<number> {
  const subtle = (globalThis.crypto as Crypto | undefined)?.subtle;
  if (subtle?.digest) {
    const buf = await subtle.digest("SHA-256", new TextEncoder().encode(sessionId));
    const b = new Uint8Array(buf);
    // first 4 bytes (8 hex chars) as the desktop slices [:8]
    return ((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]) >>> 0;
  }
  let h = 0x811c9dc5; // FNV-1a offset basis
  for (let i = 0; i < sessionId.length; i++) {
    h ^= sessionId.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export interface SessionCallbacks {
  onItem?: (item: { trialIndex: number; taskK: number; segId: number }) => void;
  onTrial?: (diag: TrialDiag) => void;
  onDone?: (result: SessionResult) => void;
}

export interface SessionResult {
  sessionId: string;
  nQuestions: number;
  stopReason: string;
  verdicts: string[];
  servedSegIds: number[];
  trials: TrialDiag[];
  finalAuroc: number[]; // per-task posterior-mean AUROC (final cloud)
  finalAurocHw: number[]; // per-task (1-α) credible halfwidth
  // Particle-cloud trajectory for the visualization videos (#8). Flattened
  // Float32: t/l are [T,N,K] row-major; w is [T,N]. Fed to the desktop renderers
  // server-side as trajectory.npz (cortex_storage.py:583-590 schema).
  traj: { t: Float32Array; l: Float32Array; w: Float32Array; shape: [number, number, number] };
}

export class WebCortexSession {
  private inputs: EngineInputs;
  private sessionId: string;
  private seed: number;
  private state!: ParticleState;
  private policy!: AD6Policy;
  private rng!: Rng;
  private remaining!: Set<number>;
  private nPerTask!: number[];
  private verdicts!: string[]; // running per-task verdicts (for spike-first sectioning)
  private trials: TrialDiag[] = [];
  private served: number[] = [];
  // Per-question particle-cloud snapshots for the visualization videos (#8).
  private tTraj: Float64Array[] = [];
  private lTraj: Float64Array[] = [];
  private wTraj: Float64Array[] = [];
  private proposalScale: number;
  private cb: SessionCallbacks;
  private answerResolver: ((pick: number) => void) | null = null;
  private aborted = false;

  constructor(inputs: EngineInputs, sessionId: string, seed: number, cb: SessionCallbacks = {}) {
    this.inputs = inputs;
    this.sessionId = sessionId;
    this.seed = seed;
    this.cb = cb;
    this.proposalScale = 2.38 / Math.sqrt(2 * inputs.taskCodes.length);
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

  // Per-task candidate arrays over the remaining bank, FAMILY-GATED: a segment
  // is a candidate for task k only if its sMean[k] is a real number (spike segs
  // have NaN for the 6 IIIC tasks and vice-versa). Mirrors the desktop
  // cortex_engine_inputs_k7.as_engine_arrays NaN masking.
  private bankArrays(): BankArrays {
    const K = this.inputs.taskCodes.length;
    const sMean: number[][] = Array.from({ length: K }, () => []);
    const sSd: number[][] = Array.from({ length: K }, () => []);
    const segId: number[][] = Array.from({ length: K }, () => []);
    for (const seg of this.inputs.segments) {
      if (!this.remaining.has(seg.segId)) continue;
      for (let k = 0; k < K; k++) {
        const sm = seg.sMean[k];
        if (Number.isNaN(sm)) continue; // not a candidate for this task's family
        sMean[k].push(sm);
        sSd[k].push(seg.sSd[k]);
        segId[k].push(seg.segId);
      }
    }
    return { sMean, sSd, segId };
  }

  // Phase-aware active task set (spike-first sectioning, session_controller.py
  // :310-313): for K=7, while spike (task 0) is PENDING and its bank is
  // non-empty, serve ONLY spike; once it resolves, serve the unresolved IIIC
  // tasks. K=6 (legacy): all unresolved tasks with a non-empty bank.
  private computeActiveDomains(bank: BankArrays): number[] {
    const K = this.inputs.taskCodes.length;
    if (K === 7 && this.verdicts[0] === VERDICT.PENDING && bank.sMean[0].length > 0) {
      return [0];
    }
    const base: number[] = [];
    for (let k = 0; k < K; k++) {
      if (bank.sMean[k].length > 0 && this.verdicts[k] === VERDICT.PENDING) base.push(k);
    }
    return base;
  }

  async run(): Promise<SessionResult> {
    const K = this.inputs.taskCodes.length;
    this.rng = new Rng(this.seed);
    const prior = precomputePrior(this.inputs.corrL);
    this.state = makeState(N_PARTICLES, K, prior, this.rng);
    this.policy = AD6Policy.fromInputs(this.inputs.ellStar, this.inputs.corrL);
    this.remaining = new Set(this.inputs.segments.map((s) => s.segId));
    this.nPerTask = new Array(K).fill(0);
    this.verdicts = new Array(K).fill(VERDICT.PENDING);

    let stopReason = "bank_exhausted";
    const maxQ = Math.min(MAX_QUESTIONS, this.inputs.segments.length);

    for (let trialIndex = 0; trialIndex < maxQ; trialIndex++) {
      if (this.remaining.size === 0 || this.aborted) break;
      const bank = this.bankArrays();
      const active = this.computeActiveDomains(bank);
      if (active.length === 0) break; // nothing servable → bank exhausted
      const chosen: Chosen =
        trialIndex === 0
          ? chooseFirstItem(this.state, bank, FIRST_ITEM_TOPN, this.rng, active)
          : chooseItem(this.state, bank, active);
      if (chosen.segId < 0) break;

      this.cb.onItem?.({ trialIndex, taskK: chosen.k, segId: chosen.segId });

      const pick = await this.awaitAnswer();
      if (this.aborted || pick < 0) {
        stopReason = "aborted";
        break;
      }
      const y: 0 | 1 = pick === chosen.k ? 1 : 0;

      update(this.state, chosen.k, chosen.s, y, chosen.sSd);
      let rejuv = false;
      if (ess(this.state.w) < ESS_THRESHOLD_FRAC * N_PARTICLES) {
        resampleAndRejuvenate(this.state, this.rng, N_MH_STEPS, this.proposalScale);
        rejuv = true;
      }

      this.served.push(chosen.segId);
      this.remaining.delete(chosen.segId);
      this.nPerTask[chosen.k] += 1;

      // Snapshot the (post-update) particle cloud for the visualization videos
      // (mirrors session_controller.py:569-572 t_traj/l_traj/w_traj capture).
      this.tTraj.push(this.state.t.slice());
      this.lTraj.push(this.state.l.slice());
      this.wTraj.push(this.state.w.slice());

      const res = this.policy.evaluate(this.state, this.nPerTask);
      this.verdicts = res.verdicts; // drives the next item's spike-first sectioning
      const { tMean, lMean } = posteriorMeans(this.state);
      const diag: TrialDiag = {
        trialIndex,
        taskK: chosen.k,
        segId: chosen.segId,
        s: chosen.s,
        sSd: chosen.sSd,
        y,
        ess: res.ess,
        rejuv,
        pi: res.pi,
        mcse: res.mcse,
        R: res.R,
        verdicts: res.verdicts,
        nPerTask: [...this.nPerTask],
        tMean,
        lMean,
        aurocHw: aurocSummary(this.state.l, this.state.w, this.state.N, K).hw,
      };
      this.trials.push(diag);
      this.cb.onTrial?.(diag);

      if (res.stop) {
        stopReason = res.stopReason;
        break;
      }
    }

    const verdicts = this.policy.finalize();
    const { mean: finalAuroc, hw: finalAurocHw } = aurocSummary(
      this.state.l, this.state.w, this.state.N, K,
    );
    // Flatten the snapshots into [T,N,K] (t,l) and [T,N] (w) Float32 buffers.
    const T = this.tTraj.length, N = this.state.N;
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
      verdicts,
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
