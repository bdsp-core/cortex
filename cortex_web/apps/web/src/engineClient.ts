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
import {
  RUNTIME_LOAD_PROFILE, runtimeLoadExceedsLimit, type RuntimeLoadSample,
} from "../engine/execution_profile";

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
  qualificationRankedSpeculation?: boolean;
}

export class EngineClient {
  private worker: Worker;
  private answerStartedAt: number | null = null;
  private heartbeatTimer: number | null = null;
  private heartbeatExpectedAt = 0;
  private heartbeatSamples = 0;
  private heartbeatDelayTotal = 0;
  private heartbeatDelayMax = 0;
  private heartbeatWindowSamples = 0;
  private heartbeatWindowDelayTotal = 0;
  private heartbeatWindowDelayMax = 0;

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
      qualificationRankedSpeculation: options.qualificationRankedSpeculation,
    }, computePayloadTransferables(payload));
    this.startHeartbeat();
  }

  answer(pick: number): void {
    this.answerStartedAt = performance.now();
    this.post({
      type: "answer", pick,
      submittedAtEpochMs: performance.timeOrigin + this.answerStartedAt,
    });
  }

  abort(): void {
    this.post({ type: "abort" });
  }

  /** Qualification hook for deterministic protocol/lifecycle tests. Runtime
   * application feedback is generated automatically by the heartbeat window. */
  reportRuntimeLoadForQualification(sample: RuntimeLoadSample): void {
    this.post({ type: "runtime_load", ...sample });
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
    this.resetHeartbeatWindow();
    this.heartbeatExpectedAt = performance.now() + intervalMs;
    this.heartbeatTimer = window.setInterval(() => {
      const now = performance.now();
      const delay = Math.max(0, now - this.heartbeatExpectedAt);
      this.heartbeatSamples += 1;
      this.heartbeatDelayTotal += delay;
      this.heartbeatDelayMax = Math.max(this.heartbeatDelayMax, delay);
      this.heartbeatWindowSamples += 1;
      this.heartbeatWindowDelayTotal += delay;
      this.heartbeatWindowDelayMax = Math.max(this.heartbeatWindowDelayMax, delay);
      this.heartbeatExpectedAt = now + intervalMs;
      if (this.heartbeatWindowSamples >= RUNTIME_LOAD_PROFILE.windowSamples) {
        const sample = {
          sampleCount: this.heartbeatWindowSamples,
          meanDelayMs: this.heartbeatWindowDelayTotal / this.heartbeatWindowSamples,
          maxDelayMs: this.heartbeatWindowDelayMax,
        };
        if (runtimeLoadExceedsLimit(sample)) {
          this.post({ type: "runtime_load", ...sample });
        }
        this.resetHeartbeatWindow();
      }
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
    this.resetHeartbeatWindow();
  }

  private resetHeartbeatWindow(): void {
    this.heartbeatWindowSamples = 0;
    this.heartbeatWindowDelayTotal = 0;
    this.heartbeatWindowDelayMax = 0;
  }

  private post(message: EngineWorkerRequest, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, { transfer });
  }
}
