// Main-thread client for the engine Web Worker (engine/worker.ts). Keeps the
// particle filter off the UI thread; the client just relays init/answer/abort
// and surfaces item/trial/done/error events.

import { EngineInputs, TrialDiag } from "../engine/types";

export interface EngineClientHandlers {
  onItem: (item: { trialIndex: number; taskK: number; segId: number }) => void;
  onTrial?: (diag: TrialDiag) => void;
  onDone: (result: {
    sessionId: string;
    nQuestions: number;
    stopReason: string;
    verdicts: string[];
    servedSegIds: number[];
    trials: TrialDiag[];
    finalAuroc: number[];
    finalAurocHw: number[];
    traj: { t: Float32Array; l: Float32Array; w: Float32Array; shape: [number, number, number] };
  }) => void;
  onError?: (message: string) => void;
}

export class EngineClient {
  private worker: Worker;

  constructor(private handlers: EngineClientHandlers) {
    this.worker = new Worker(new URL("../engine/worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (ev: MessageEvent) => {
      const m = ev.data;
      switch (m.type) {
        case "item":
          this.handlers.onItem({ trialIndex: m.trialIndex, taskK: m.taskK, segId: m.segId });
          break;
        case "trial":
          this.handlers.onTrial?.(m.diag);
          break;
        case "done":
          this.handlers.onDone(m.result);
          break;
        case "error":
          this.handlers.onError?.(m.message);
          break;
      }
    };
    // An uncaught throw inside the worker that never posts an {type:"error"}
    // message would otherwise hang the test on the current item with no
    // signal. Route both to onError so the UI can surface it.
    this.worker.onerror = (e: ErrorEvent) => {
      e.preventDefault();
      this.handlers.onError?.(e.message || "engine worker crashed");
    };
    this.worker.onmessageerror = () => {
      this.handlers.onError?.("engine worker message error");
    };
  }

  start(inputs: EngineInputs, sessionId: string, seed?: number): void {
    this.worker.postMessage({ type: "init", inputs, sessionId, seed });
  }

  answer(pick: number): void {
    this.worker.postMessage({ type: "answer", pick });
  }

  abort(): void {
    this.worker.postMessage({ type: "abort" });
  }

  dispose(): void {
    this.worker.terminate();
  }
}
