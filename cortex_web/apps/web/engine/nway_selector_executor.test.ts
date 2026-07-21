import { describe, expect, it } from "vitest";

import { NWaySelectorWorkerExecutor } from "./nway_selector_executor";
import type {
  NWaySelectorWorkerRequest, NWaySelectorWorkerResponse,
} from "./nway_selector_protocol";
import { precisionGoldenInputs } from "./__testdata__/precision_fixture";
import type { PackedParticleHistory } from "./types";
import { SpeculationCancelledError } from "./speculation_cancellation";

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

class HistoryCachingWorker {
  private listeners = new Map<string, Set<(event: any) => void>>();
  readonly receivedFullHistory: boolean[] = [];
  readonly receivedHistoryVersions: number[] = [];
  terminated = false;

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
    if (message.type !== "history_likelihood") {
      throw new Error("unexpected history caching worker request");
    }
    this.receivedFullHistory.push(message.history !== undefined);
    this.receivedHistoryVersions.push(message.historyVersion);
    queueMicrotask(() => this.emitMessage({
      type: "history_result",
      jobId: message.jobId,
      startIndex: message.startIndex,
      historyVersion: message.historyVersion,
      logLikelihood: new Float64Array(message.N),
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

  it("reduces a pool downward without recreating or increasing workers", async () => {
    const workers = Array.from({ length: 4 }, () => new InjectedWorker("timeout"));
    let nextWorker = 0;
    const executor = new NWaySelectorWorkerExecutor(precisionGoldenInputs(), 4, {
      workerFactory: () => workers[nextWorker++] as unknown as Worker,
    });
    await executor.ready();
    expect(executor.workerCount).toBe(4);
    expect(executor.reduceWorkerCount(2)).toBe(2);
    expect(workers.map((worker) => worker.terminated)).toEqual([false, false, true, true]);
    expect(executor.reduceWorkerCount(4)).toBe(2);
    expect(() => executor.reduceWorkerCount(0)).toThrow(/invalid/);
    executor.dispose();
    expect(workers.every((worker) => worker.terminated)).toBe(true);
  });

  it("cancels active shards and recreates the same pool before reuse", async () => {
    const workers = Array.from({ length: 4 }, () => new InjectedWorker("timeout"));
    let nextWorker = 0;
    const inputs = precisionGoldenInputs();
    const executor = new NWaySelectorWorkerExecutor(inputs, 2, {
      workerFactory: () => workers[nextWorker++] as unknown as Worker,
    });
    await executor.ready();
    const K = inputs.taskCodes.length;
    const active = executor.historyLikelihood(
      emptyHistory(K), 2, K,
      new Float64Array(2 * K), new Float64Array(2 * K),
    );
    const restart = executor.restartAfterCancellation();
    await restart.ready;
    await expect(active).rejects.toBeInstanceOf(SpeculationCancelledError);
    expect(restart.report).toEqual({ cancelledJobs: 2, phases: ["mh_history"] });
    expect(workers.map((worker) => worker.terminated))
      .toEqual([true, true, false, false]);
    expect(executor.workerCount).toBe(2);
    executor.dispose();
    expect(workers.every((worker) => worker.terminated)).toBe(true);
  });

  it("sends each history version once and restores the cache after restart", async () => {
    const workers = [new HistoryCachingWorker(), new HistoryCachingWorker()];
    let nextWorker = 0;
    const inputs = precisionGoldenInputs();
    const executor = new NWaySelectorWorkerExecutor(inputs, 1, {
      workerFactory: () => workers[nextWorker++] as unknown as Worker,
    });
    await executor.ready();
    const K = inputs.taskCodes.length;
    const history = emptyHistory(K);
    const likelihood = () => executor.historyLikelihood(
      history, 1, K, new Float64Array(K), new Float64Array(K),
    );

    await likelihood();
    await likelihood();
    history.length = 1;
    await likelihood();

    expect(workers[0].receivedFullHistory).toEqual([true, false, true]);
    expect(workers[0].receivedHistoryVersions).toEqual([1, 1, 2]);

    const restart = executor.restartAfterCancellation();
    await restart.ready;
    await likelihood();
    expect(workers[1].receivedFullHistory).toEqual([true]);
    expect(workers[1].receivedHistoryVersions).toEqual([2]);

    executor.dispose();
    expect(workers.every((worker) => worker.terminated)).toBe(true);
  });
});
