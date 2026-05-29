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
import { AD6Policy } from "./policy";
import { Rng } from "./rng";

export const N_PARTICLES = 600;
export const MAX_QUESTIONS = 300;
export const N_MH_STEPS = 15;
export const ESS_THRESHOLD_FRAC = 0.5;
export const FIRST_ITEM_TOPN = 10;

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

export interface SessionResult {
  sessionId: string;
  nQuestions: number;
  stopReason: string;
  verdicts: string[];
  servedSegIds: number[];
  trials: TrialDiag[];
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
  private trials: TrialDiag[] = [];
  private served: number[] = [];
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

  // build per-task candidate arrays over the remaining (unserved) bank
  private bankArrays(): BankArrays {
    const K = this.inputs.taskCodes.length;
    const sMean: number[][] = Array.from({ length: K }, () => []);
    const sSd: number[][] = Array.from({ length: K }, () => []);
    const segId: number[][] = Array.from({ length: K }, () => []);
    for (const seg of this.inputs.segments) {
      if (!this.remaining.has(seg.segId)) continue;
      for (let k = 0; k < K; k++) {
        sMean[k].push(seg.sMean[k]);
        sSd[k].push(seg.sSd[k]);
        segId[k].push(seg.segId);
      }
    }
    return { sMean, sSd, segId };
  }

  async run(): Promise<SessionResult> {
    const K = this.inputs.taskCodes.length;
    this.rng = new Rng(this.seed);
    const prior = precomputePrior(this.inputs.corrL);
    this.state = makeState(N_PARTICLES, K, prior, this.rng);
    this.policy = AD6Policy.fromInputs(this.inputs.ellStar, this.inputs.corrL);
    this.remaining = new Set(this.inputs.segments.map((s) => s.segId));
    this.nPerTask = new Array(K).fill(0);

    let stopReason = "bank_exhausted";
    const maxQ = Math.min(MAX_QUESTIONS, this.inputs.segments.length);

    for (let trialIndex = 0; trialIndex < maxQ; trialIndex++) {
      if (this.remaining.size === 0 || this.aborted) break;
      const bank = this.bankArrays();
      const chosen: Chosen =
        trialIndex === 0
          ? chooseFirstItem(this.state, bank, FIRST_ITEM_TOPN, this.rng)
          : chooseItem(this.state, bank);

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

      const res = this.policy.evaluate(this.state, this.nPerTask);
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
      };
      this.trials.push(diag);
      this.cb.onTrial?.(diag);

      if (res.stop) {
        stopReason = res.stopReason;
        break;
      }
    }

    const verdicts = this.policy.finalize();
    const result: SessionResult = {
      sessionId: this.sessionId,
      nQuestions: this.trials.length,
      stopReason,
      verdicts,
      servedSegIds: this.served,
      trials: this.trials,
    };
    this.cb.onDone?.(result);
    return result;
  }
}
