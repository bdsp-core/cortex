// Main-thread client for the engine Web Worker (engine/worker.ts). Keeps the
// particle filter off the UI thread; the client just relays init/answer/abort
// and surfaces item/trial/done/error events.

import type {
  ComputeEngineInputs, EnginePerformanceEvent, EngineWorkerRequest,
  EngineWorkerResponse, RequestedComputeMode, SessionResult, TrialDiag,
} from "../engine";
import { computePayloadTransferables, packComputeInputs } from "../engine";

export interface EngineClientHandlers {
  onItem: (item: { trialIndex: number; taskK: number; segId: number }) => void;
  onPrefetch?: (hint: { trialIndex: number; segId: number }) => void;
  onTrial?: (diag: TrialDiag) => void;
  onPerformance?: (event: EnginePerformanceEvent) => void;
  onDone: (result: SessionResult) => void;
  onError?: (message: string) => void;
}

export interface EngineStartOptions {
  seed?: number;
  requestedComputeMode?: RequestedComputeMode;
  qualificationHardwareConcurrency?: number;
  qualificationRankedSpeculation?: boolean;
}

export class EngineClient {
  private worker: Worker;
  private answerStartedAt: number | null = null;
  private mediaStartedAt: number | null = null;
  private mediaTrialIndex: number | null = null;
  private heartbeatTimer: number | null = null;
  private heartbeatExpectedAt = 0;
  private heartbeatSamples = 0;
  private heartbeatDelayTotal = 0;
  private heartbeatDelayMax = 0;

  constructor(private handlers: EngineClientHandlers) {
    this.worker = new Worker(new URL("../engine/worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (ev: MessageEvent<EngineWorkerResponse>) => {
      const m = ev.data;
      switch (m.type) {
        case "item":
          if (this.answerStartedAt !== null) {
            this.handlers.onPerformance?.({
              kind: "answer_to_item",
              trialIndex: m.trialIndex,
              durationMs: performance.now() - this.answerStartedAt,
            });
            this.mediaTrialIndex = m.trialIndex;
            this.answerStartedAt = null;
          }
          this.handlers.onItem({ trialIndex: m.trialIndex, taskK: m.taskK, segId: m.segId });
          break;
        case "prefetch":
          this.handlers.onPrefetch?.({ trialIndex: m.trialIndex, segId: m.segId });
          break;
        case "trial":
          this.handlers.onTrial?.(m.diag);
          break;
        case "performance":
          this.handlers.onPerformance?.(m.event);
          break;
        case "done":
          this.clearPendingAnswerTimings();
          this.stopHeartbeat(true);
          this.handlers.onDone(m.result);
          break;
        case "error":
          this.clearPendingAnswerTimings();
          this.stopHeartbeat(true);
          this.handlers.onError?.(m.message);
          break;
      }
    };
    // An uncaught throw inside the worker that never posts an {type:"error"}
    // message would otherwise hang the test on the current item with no
    // signal. Route both to onError so the UI can surface it.
    this.worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      this.clearPendingAnswerTimings();
      this.stopHeartbeat(true);
      this.handlers.onError?.(e.message || "engine worker crashed");
    };
    this.worker.onmessageerror = () => {
      this.clearPendingAnswerTimings();
      this.stopHeartbeat(true);
      this.handlers.onError?.("engine worker message error");
    };
  }

  start(
    inputs: ComputeEngineInputs,
    sessionId: string,
    options: EngineStartOptions = {},
  ): void {
    const payload = packComputeInputs(inputs);
    this.post({
      type: "init",
      payload,
      sessionId,
      seed: options.seed,
      requestedComputeMode: options.requestedComputeMode ?? "serial",
      qualificationHardwareConcurrency: options.qualificationHardwareConcurrency,
      qualificationRankedSpeculation: options.qualificationRankedSpeculation,
    }, computePayloadTransferables(payload));
    this.startHeartbeat();
  }

  answer(pick: number): void {
    this.answerStartedAt = performance.now();
    this.mediaStartedAt = this.answerStartedAt;
    this.mediaTrialIndex = null;
    this.post({
      type: "answer", pick,
      submittedAtEpochMs: performance.timeOrigin + this.answerStartedAt,
    });
  }

  abort(): void {
    this.clearPendingAnswerTimings();
    this.post({ type: "abort" });
  }

  mediaReady(trialIndex: number): void {
    if (this.mediaStartedAt === null || this.mediaTrialIndex !== trialIndex) return;
    this.handlers.onPerformance?.({
      kind: "answer_to_media_ready",
      trialIndex,
      durationMs: performance.now() - this.mediaStartedAt,
    });
    this.mediaStartedAt = null;
    this.mediaTrialIndex = null;
  }

  dispose(): void {
    this.clearPendingAnswerTimings();
    this.stopHeartbeat(false);
    this.worker.terminate();
  }

  private startHeartbeat(): void {
    this.stopHeartbeat(false);
    // Observational telemetry only: heartbeat delay never changes pool topology.
    const intervalMs = 250;
    this.heartbeatSamples = 0;
    this.heartbeatDelayTotal = 0;
    this.heartbeatDelayMax = 0;
    this.heartbeatExpectedAt = performance.now() + intervalMs;
    this.heartbeatTimer = window.setInterval(() => {
      const now = performance.now();
      const delay = Math.max(0, now - this.heartbeatExpectedAt);
      this.heartbeatSamples += 1;
      this.heartbeatDelayTotal += delay;
      this.heartbeatDelayMax = Math.max(this.heartbeatDelayMax, delay);
      this.heartbeatExpectedAt = now + intervalMs;
    }, intervalMs);
  }

  private clearPendingAnswerTimings(): void {
    this.answerStartedAt = null;
    this.mediaStartedAt = null;
    this.mediaTrialIndex = null;
  }

  private stopHeartbeat(emit: boolean): void {
    if (this.heartbeatTimer !== null) {
      window.clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (emit && this.heartbeatSamples > 0) {
      this.handlers.onPerformance?.({
        kind: "event_loop_heartbeat",
        sampleCount: this.heartbeatSamples,
        meanDelayMs: this.heartbeatDelayTotal / this.heartbeatSamples,
        maxDelayMs: this.heartbeatDelayMax,
      });
    }
    this.heartbeatSamples = 0;
    this.heartbeatDelayTotal = 0;
    this.heartbeatDelayMax = 0;
  }

  private post(message: EngineWorkerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, { transfer });
  }
}
