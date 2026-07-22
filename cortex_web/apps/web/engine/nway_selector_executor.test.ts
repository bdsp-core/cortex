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

class CompactPayloadWorker {
  private listeners = new Map<string, Set<(event: any) => void>>();
  readonly scoreRequests: Extract<
    NWaySelectorWorkerRequest, { type: "score" }
  >[] = [];
  readonly screenRequests: Extract<
    NWaySelectorWorkerRequest, { type: "screen" }
  >[] = [];

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
    if (message.type === "score") {
      const workerMessage = structuredClone(message, { transfer: [
        message.taskKs.buffer, message.segIds.buffer,
      ] });
      this.scoreRequests.push(workerMessage);
      queueMicrotask(() => this.emitMessage({
        type: "result", jobId: workerMessage.jobId,
        startIndex: workerMessage.startIndex,
        losses: Float64Array.from(workerMessage.segIds),
      }));
      return;
    }
    if (message.type === "screen") {
      const workerMessage = structuredClone(message, {
        transfer: [message.segIds.buffer],
      });
      this.screenRequests.push(workerMessage);
      queueMicrotask(() => this.emitMessage({
        type: "screen_result", jobId: workerMessage.jobId,
        taskK: workerMessage.taskK,
        entropySegIds: [workerMessage.segIds[0]],
        fisherSegIds: [workerMessage.segIds[workerMessage.segIds.length - 1]],
        entropyMs: 1, fisherMs: 2,
      }));
      return;
    }
    throw new Error("unexpected compact payload worker request");
  }

  terminate(): void {}

  private emitMessage(data: NWaySelectorWorkerResponse): void {
    for (const listener of this.listeners.get("message") ?? []) listener({ data });
  }
}

class HistoryCachingWorker {
  private listeners = new Map<string, Set<(event: any) => void>>();
  readonly receivedFullHistory: boolean[] = [];
  readonly receivedHistoryVersions: number[] = [];
  readonly receivedOutputValuesBeforeWork: number[] = [];
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
    const workerMessage = structuredClone(message, { transfer: [
      message.t.buffer, message.l.buffer, message.logLikelihood.buffer,
    ] });
    this.receivedFullHistory.push(workerMessage.history !== undefined);
    this.receivedHistoryVersions.push(workerMessage.historyVersion);
    this.receivedOutputValuesBeforeWork.push(workerMessage.logLikelihood[0]);
    workerMessage.logLikelihood.fill(this.receivedHistoryVersions.length);
    const response = structuredClone({
      type: "history_result" as const,
      jobId: workerMessage.jobId,
      startIndex: workerMessage.startIndex,
      historyVersion: workerMessage.historyVersion,
      t: workerMessage.t,
      l: workerMessage.l,
      logLikelihood: workerMessage.logLikelihood,
    }, { transfer: [
      workerMessage.t.buffer, workerMessage.l.buffer,
      workerMessage.logLikelihood.buffer,
    ] });
    queueMicrotask(() => this.emitMessage(response));
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
  it("transfers compact score and screen vectors without cloned candidates", async () => {
    const worker = new CompactPayloadWorker();
    const inputs = precisionGoldenInputs();
    const executor = new NWaySelectorWorkerExecutor(inputs, 1, {
      workerFactory: () => worker as unknown as Worker,
    });
    await executor.ready();
    const K = inputs.taskCodes.length;
    const candidates = inputs.segments.slice(0, 3).map((segment, k) => ({ k, segment }));
    const state = {
      N: 1, K,
      t: new Float64Array(K), l: new Float64Array(K),
      w: new Float64Array([1]),
    };

    await expect(executor.score(state, candidates)).resolves.toEqual(
      Float64Array.from(candidates, (candidate) => candidate.segment.segId),
    );
    expect(worker.scoreRequests).toHaveLength(1);
    expect(worker.scoreRequests[0].taskKs).toBeInstanceOf(Uint8Array);
    expect(Array.from(worker.scoreRequests[0].taskKs)).toEqual([0, 1, 2]);
    expect(worker.scoreRequests[0].segIds).toBeInstanceOf(Uint32Array);
    expect("candidates" in worker.scoreRequests[0]).toBe(false);

    const segIds = candidates.map((candidate) => candidate.segment.segId);
    const moments = {
      K,
      tMean: new Array(K).fill(0), lMean: new Array(K).fill(0),
      tSd: new Array(K).fill(1), lSd: new Array(K).fill(1),
    };
    await expect(executor.screen(moments, [{ taskK: 0, segIds }])).resolves.toEqual([{
      taskK: 0,
      entropySegIds: [segIds[0]], fisherSegIds: [segIds[segIds.length - 1]],
      entropyMs: 1, fisherMs: 2,
    }]);
    expect(worker.screenRequests[0].segIds).toBeInstanceOf(Uint32Array);
    executor.dispose();
  });

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

  it("does not expose post-calibration worker resizing", async () => {
    const workers = Array.from({ length: 4 }, () => new InjectedWorker("timeout"));
    let nextWorker = 0;
    const executor = new NWaySelectorWorkerExecutor(precisionGoldenInputs(), 4, {
      workerFactory: () => workers[nextWorker++] as unknown as Worker,
    });
    await executor.ready();
    expect(executor.workerCount).toBe(4);
    expect((executor as unknown as { reduceWorkerCount?: unknown }).reduceWorkerCount)
      .toBeUndefined();
    expect(workers.every((worker) => !worker.terminated)).toBe(true);
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
    expect(workers[0].receivedOutputValuesBeforeWork).toEqual([0, 1, 2]);

    const restart = executor.restartAfterCancellation();
    await restart.ready;
    await likelihood();
    expect(workers[1].receivedFullHistory).toEqual([true]);
    expect(workers[1].receivedHistoryVersions).toEqual([2]);
    expect(workers[1].receivedOutputValuesBeforeWork).toEqual([0]);

    executor.dispose();
    expect(workers.every((worker) => worker.terminated)).toBe(true);
  });
});
