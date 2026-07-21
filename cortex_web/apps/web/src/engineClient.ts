// Main-thread client for the engine Web Worker (engine/worker.ts). Keeps the
// particle filter off the UI thread; the client just relays init/answer/abort
// and surfaces item/trial/done/error events.

import type { SessionResult } from "../engine/session";
import type {
  ComputeEngineInputs, EnginePerformanceEvent, RequestedComputeMode, TrialDiag,
} from "../engine/types";
import type {
  EngineWorkerRequest, EngineWorkerResponse,
} from "../engine/worker_protocol";
import {
  computePayloadTransferables, packComputeInputs,
} from "../engine/compute_payload";

export interface EngineClientHandlers {
  onItem: (item: { trialIndex: number; taskK: number; segId: number }) => void;
  onTrial?: (diag: TrialDiag) => void;
  onPerformance?: (event: EnginePerformanceEvent) => void;
  onDone: (result: SessionResult) => void;
  onError?: (message: string) => void;
}

export interface EngineStartOptions {
  seed?: number;
  requestedComputeMode?: RequestedComputeMode;
  qualificationHardwareConcurrency?: number;
}

export class EngineClient {
  private worker: Worker;
  private answerStartedAt: number | null = null;
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
            this.answerStartedAt = null;
          }
          this.handlers.onItem({ trialIndex: m.trialIndex, taskK: m.taskK, segId: m.segId });
          break;
        case "trial":
          this.handlers.onTrial?.(m.diag);
          break;
        case "performance":
          this.handlers.onPerformance?.(m.event);
          break;
        case "done":
          this.stopHeartbeat(true);
          this.handlers.onDone(m.result);
          break;
        case "error":
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
      this.stopHeartbeat(true);
      this.handlers.onError?.(e.message || "engine worker crashed");
    };
    this.worker.onmessageerror = () => {
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
    }, computePayloadTransferables(payload));
    this.startHeartbeat();
  }

  answer(pick: number): void {
    this.answerStartedAt = performance.now();
    this.post({ type: "answer", pick });
  }

  abort(): void {
    this.post({ type: "abort" });
  }

  dispose(): void {
    this.stopHeartbeat(false);
    this.worker.terminate();
  }

  private startHeartbeat(): void {
    this.stopHeartbeat(false);
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
