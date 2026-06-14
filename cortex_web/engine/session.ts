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
import { aurocSummary } from "./auroc";
import { BankArrays, chooseItem, chooseFirstItem, Chosen } from "./choose_item";
import { AD6Policy } from "./policy";
import { Rng } from "./rng";

export const N_PARTICLES = 600;
// Paper-grade max (v1.3.6 instrument freeze): 500 questions cap, matching
// the desktop. Most sessions finish well before this.
export const MAX_QUESTIONS = 500;
export const N_MH_STEPS = 15;
export const ESS_THRESHOLD_FRAC = 0.5;
export const FIRST_ITEM_TOPN = 10;
// Variety cap (v1.3.7 alignment): after this many consecutive picks from the
// same task, that task is excluded from the next pick so the session doesn't
// streak one domain for ten in a row. Fallback inside the loop restores the
// excluded task if no others have remaining items.
export const MAX_CONSECUTIVE_SAME_DOMAIN = 5;

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
  finalAuroc: number[]; // per-task posterior-mean AUROC (final cloud)
  finalAurocHw: number[]; // per-task (1-α) credible halfwidth
  // Particle-cloud trajectory for the visualization videos (#8). Flattened
  // Float32: t/l are [T,N,K] row-major; w is [T,N]. Fed to the desktop
  // renderers server-side as trajectory.npz (cortex_storage.py schema).
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

  // Build per-task candidate arrays over the remaining (unserved) bank. A
  // segment only contributes to the task indices listed in its
  // applicableTaskIdx — IIIC items to the 6 IIIC tasks, spike items to spike
  // — so the engine can't accidentally select an IIIC clip for the spike task
  // (sMean/sSd at the inapplicable indices are sentinel 0.0, not real signals).
  // Pre-K=7 bundles omit applicableTaskIdx; we then fall back to "all K tasks".
  private bankArrays(): BankArrays {
    const K = this.inputs.taskCodes.length;
    const sMean: number[][] = Array.from({ length: K }, () => []);
    const sSd: number[][] = Array.from({ length: K }, () => []);
    const segId: number[][] = Array.from({ length: K }, () => []);
    for (const seg of this.inputs.segments) {
      if (!this.remaining.has(seg.segId)) continue;
      const applicable = seg.applicableTaskIdx;
      if (applicable) {
        for (const k of applicable) {
          sMean[k].push(seg.sMean[k]);
          sSd[k].push(seg.sSd[k]);
          segId[k].push(seg.segId);
        }
      } else {
        for (let k = 0; k < K; k++) {
          sMean[k].push(seg.sMean[k]);
          sSd[k].push(seg.sSd[k]);
          segId[k].push(seg.segId);
        }
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

    // Phase-aware selection (desktop session_controller.py l.241+). For K=7
    // bundles with spike at index 0 we run the spike block first (Phase A:
    // only spike is selectable) until either the spike verdict locks or the
    // spike bank exhausts, then switch to Phase B (only IIIC). Pre-K=7
    // bundles have no phase concept and select across all tasks.
    const tc = this.inputs.taskClasses;
    const spikeIdx = tc ? tc.findIndex((c) => c === "spike") : -1;
    const k7Spike = spikeIdx >= 0;
    let lastVerdicts: string[] = new Array(K).fill("PENDING");

    // Variety-cap streak tracking. lastTaskK = the task served on the previous
    // trial (or -1 if none); streakCount = how many in a row that task has run.
    let lastTaskK = -1;
    let streakCount = 0;

    for (let trialIndex = 0; trialIndex < maxQ; trialIndex++) {
      if (this.remaining.size === 0 || this.aborted) break;
      const bank = this.bankArrays();

      // Phase exclusion: in K=7, Phase A only allows spike; Phase B excludes
      // spike. The spike bank being empty (no spike items in the bundle yet,
      // or already exhausted) transitions us into Phase B even if spike is
      // still PENDING.
      const phaseExcluded = new Set<number>();
      if (k7Spike) {
        const spikePending = lastVerdicts[spikeIdx] === "PENDING";
        const spikeBankNonEmpty = bank.sMean[spikeIdx]?.length > 0;
        if (spikePending && spikeBankNonEmpty) {
          // Phase A: exclude every non-spike task.
          for (let k = 0; k < K; k++) if (k !== spikeIdx) phaseExcluded.add(k);
        } else {
          // Phase B: exclude spike.
          phaseExcluded.add(spikeIdx);
        }
      }

      // Variety cap composed with phase: don't exclude a task if doing so
      // would empty the candidate set.
      const excluded = new Set(phaseExcluded);
      if (trialIndex > 0 && streakCount >= MAX_CONSECUTIVE_SAME_DOMAIN
            && !excluded.has(lastTaskK)) {
        const otherHasBank = bank.sMean.some(
          (arr, k) => k !== lastTaskK && !excluded.has(k) && arr.length > 0,
        );
        if (otherHasBank) excluded.add(lastTaskK);
      }

      let chosen: Chosen =
        trialIndex === 0
          ? chooseFirstItem(this.state, bank, FIRST_ITEM_TOPN, this.rng, excluded)
          : chooseItem(this.state, bank, excluded);
      // Defensive fallbacks: if variety+phase leaves nothing, drop the
      // variety cap; if still nothing (degenerate), drop the phase too.
      if (chosen.segId === -1)
        chosen = trialIndex === 0
          ? chooseFirstItem(this.state, bank, FIRST_ITEM_TOPN, this.rng, phaseExcluded)
          : chooseItem(this.state, bank, phaseExcluded);
      if (chosen.segId === -1) chosen = chooseItem(this.state, bank);

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
      // (mirrors session_controller.py t_traj/l_traj/w_traj capture).
      this.tTraj.push(this.state.t.slice());
      this.lTraj.push(this.state.l.slice());
      this.wTraj.push(this.state.w.slice());
      // update variety-cap streak
      if (chosen.k === lastTaskK) streakCount += 1;
      else { lastTaskK = chosen.k; streakCount = 1; }

      const res = this.policy.evaluate(this.state, this.nPerTask);
      lastVerdicts = res.verdicts;   // feeds the next trial's phase check
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
