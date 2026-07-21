import { describe, expect, it } from "vitest";

import { NWaySelectorWorkerExecutor } from "./nway_selector_executor";
import type {
  NWaySelectorWorkerRequest, NWaySelectorWorkerResponse,
} from "./nway_selector_protocol";
import { precisionGoldenInputs } from "./__testdata__/precision_fixture";
import type { PackedParticleHistory } from "./types";

type FailureMode = "error" | "malformed" | "timeout";

class InjectedWorker {
  private listeners = new Map<string, Set<(event: any) => void>>();
  terminated = false;

  constructor(private readonly mode: FailureMode) {}

  addEventListener(type: string, listener: (event: any) => void): void {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type: string, listener: (event: any) => void): void {
    this.listeners.get(type)?.delete(listener);
  }

  postMessage(message: NWaySelectorWorkerRequest): void {
    if (message.type === "init") {
      queueMicrotask(() => this.emitMessage({ type: "ready" }));
      return;
    }
    if (this.mode === "timeout") return;
    if (this.mode === "error") {
      queueMicrotask(() => this.emitMessage({
        type: "error", jobId: message.jobId, message: "injected helper error",
      }));
      return;
    }
    queueMicrotask(() => this.emitMessage({
      type: "result", jobId: message.jobId, startIndex: 0,
      losses: new Float64Array(1),
    }));
  }

  terminate(): void {
    this.terminated = true;
  }

  private emitMessage(data: NWaySelectorWorkerResponse): void {
    for (const listener of this.listeners.get("message") ?? []) listener({ data });
  }
}

function emptyHistory(K: number): PackedParticleHistory {
  return {
    K, length: 0, capacity: 1,
    kind: new Uint8Array(1),
    taskK: new Int8Array(1),
    pick: new Int8Array(1),
    binaryS: new Float64Array(1),
    binarySd: new Float64Array(1),
    signalMean: new Float64Array(K),
    signalSd: new Float64Array(K),
  };
}

describe("n-way selector pool failure containment", () => {
  for (const mode of ["error", "malformed", "timeout"] as const) {
    it(`rejects and disposes safely after an injected ${mode} response`, async () => {
      const fake = new InjectedWorker(mode);
      const inputs = precisionGoldenInputs();
      const executor = new NWaySelectorWorkerExecutor(inputs, 1, {
        workerFactory: () => fake as unknown as Worker,
        jobTimeoutMs: 5,
      });
      await executor.ready();
      const K = inputs.taskCodes.length;
      const job = executor.historyLikelihood(
        emptyHistory(K), 1, K,
        new Float64Array(K), new Float64Array(K),
      );
      await expect(job).rejects.toThrow(
        mode === "error" ? /injected helper error/
          : mode === "malformed" ? /wrong job kind/ : /timed out/,
      );
      executor.dispose();
      expect(fake.terminated).toBe(true);
    });
  }
});
